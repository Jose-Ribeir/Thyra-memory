"""Stage-2 advisory refiner: category classification + semantic/episodic label.

Uses all-MiniLM-L6-v2 for category classification via cosine similarity against
category exemplar embeddings. Falls back to rule-based classification if the
model is unavailable.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

_model = None
_embeddings = None
_categories = None
_model_lock = threading.Lock()

# Thread-local flag: when set to True, _classify_category skips the ML model
# entirely and uses rule-based classification.  Set this in worker threads to
# avoid blocking the consolidation loop on a slow/hanging model load.
_thread_local = threading.local()


def set_rules_only(value: bool = True) -> None:
    """Tell the current thread to use rule-based classification only."""
    _thread_local.rules_only = value


CATEGORY_EXEMPLARS: dict[str, list[str]] = {
    "constraints": [
        "never do this",
        "always avoid",
        "don't use",
        "I'm allergic",
        "prohibited",
        "required by policy",
        "must not",
    ],
    "identity": [
        "my name is",
        "I am a",
        "I work as",
        "I live in",
        "my role",
        "I'm a professional",
        "my background",
    ],
    "preferences": [
        "I prefer",
        "I like",
        "I love",
        "I hate",
        "my favorite",
        "I enjoy",
        "I use",
        "I always choose",
    ],
    "relationships": [
        "my manager is",
        "my colleague",
        "my friend",
        "my team",
        "my boss",
        "I work with",
        "my partner",
    ],
    "tasks": [
        "I need to",
        "I have to",
        "I must finish",
        "pending task",
        "to do",
        "working on",
        "currently implementing",
    ],
    "goals": [
        "my goal is",
        "I want to achieve",
        "I plan to",
        "I aim to",
        "my objective",
        "I'm working toward",
        "I hope to",
    ],
    "context": [
        "currently",
        "right now",
        "at the moment",
        "this week",
        "today",
        "in this project",
        "in this context",
    ],
    "skills": [
        "I know how to",
        "I'm experienced in",
        "I've been using",
        "I'm proficient",
        "I can",
        "my expertise",
        "years of experience",
    ],
    "habits": [
        "I usually",
        "I typically",
        "every morning I",
        "my routine",
        "I tend to",
        "I regularly",
        "my habit",
    ],
    "knowledge": [
        "I know that",
        "I learned",
        "I understand",
        "I'm aware that",
        "I've read",
        "I studied",
        "it is known",
    ],
    "events": [
        "yesterday",
        "last week",
        "we tried",
        "the meeting was",
        "it happened",
        "we discussed",
        "the deploy failed",
    ],
    "communication": [
        "email",
        "Slack",
        "message",
        "call",
        "meeting",
        "responded",
        "I sent",
        "they replied",
    ],
    "health": [
        "I'm allergic",
        "my doctor",
        "I take medication",
        "my diet",
        "I exercise",
        "my health",
        "medical condition",
    ],
    "finance": [
        "my budget",
        "I spent",
        "invoice",
        "payment",
        "salary",
        "cost",
        "financial",
        "expense",
    ],
    "routines": [
        "every day I",
        "my morning routine",
        "I wake up",
        "weekly",
        "daily habit",
        "schedule",
        "recurring",
    ],
}

_EPISODIC_MARKERS = re.compile(
    r"\b(?:yesterday|last (?:week|month|time)|just now|earlier today|"
    r"we tried|it (?:happened|failed|worked)|the deploy|the meeting)\b",
    re.IGNORECASE,
)


def _ensure_hf_cache() -> None:
    """Guarantee HF_HOME points to a writable directory.

    Falls back to the standard HuggingFace default (~/.cache/huggingface)
    when the configured path is unreachable (e.g. HF_HOME=G:\\ on a missing drive).
    """
    import os
    import pathlib

    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        try:
            pathlib.Path(hf_home).mkdir(parents=True, exist_ok=True)
            return  # configured path is usable
        except OSError:
            pass
    # Fall back to the designated HuggingFace cache on H:\
    default_cache = pathlib.Path(r"H:\HuggingFace")
    default_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(default_cache)


def _load_model_worker(result: list) -> None:
    """Load the SentenceTransformer model in a thread; store result in result[0]."""
    try:
        import numpy as np

        _ensure_hf_cache()
        from sentence_transformers import SentenceTransformer

        # local_files_only=True prevents any network requests during load.
        # The model must already be in HF_HOME (H:\HuggingFace).
        m = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        cats = list(CATEGORY_EXEMPLARS.keys())
        centroids = []
        for cat in cats:
            embs = m.encode(CATEGORY_EXEMPLARS[cat])
            centroids.append(embs.mean(axis=0))
        result.append((m, np.array(centroids), cats))
    except Exception:
        result.append(None)


def _model_is_resident() -> bool:
    """True when the ML model is already loaded in memory (not None and not False)."""
    return _model not in (None, False)


def _get_model_and_embeddings():
    global _model, _embeddings, _categories
    rules_only = getattr(_thread_local, "rules_only", False)
    # In rules-only mode: return the model only if it is ALREADY loaded (no cold load).
    # This lets the worker thread use the prewarm result without ever blocking on a load.
    if rules_only:
        if _model_is_resident():
            return _model, _embeddings, _categories
        return None, None, None
    if _model is not None:
        return _model, _embeddings, _categories
    with _model_lock:
        if _model is None:
            try:
                import threading as _threading

                result: list = []
                t = _threading.Thread(
                    target=_load_model_worker, args=(result,), daemon=True
                )
                t.start()
                t.join(timeout=20)  # 20s timeout — fall back to rules if model hangs
                if result and result[0] is not None:
                    m, centroids, cats = result[0]
                    _model = m
                    _embeddings = centroids
                    _categories = cats
                else:
                    _model = False  # timed out or failed — use rule-based fallback
            except Exception:
                _model = False
    return (
        _model if _model is not False else None,
        _embeddings,
        _categories,
    )


def _classify_category(clause: str) -> str:
    """Classify clause into a category using sentence-transformers or rules.

    Always calls _get_model_and_embeddings, which handles the rules-only flag:
    in rules-only threads it returns the already-loaded model (if resident) or
    None (if not yet loaded — never triggers a cold load from a worker thread).
    """
    model, embeddings, categories = _get_model_and_embeddings()
    if model is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            emb = model.encode([clause])
            sims = cosine_similarity(emb, embeddings)[0]
            return categories[int(sims.argmax())]
        except Exception:
            pass
    # Rule-based fallback — all patterns match against clause.lower(), so every
    # literal must be lowercase (no capital I — use i instead).
    lower = clause.lower()
    if re.search(
        r"\b(?:never|always avoid|don't use|must not|prohibited|required by policy|"
        r"from now on|don't forget|be aware|keep in mind|make sure|"
        r"always remember|important note)\b",
        lower,
    ):
        return "constraints"
    if re.search(
        r"\b(?:i (?:am|'m) a\b|i work (?:as|at)\b|i live in\b|my (?:name|role|background|job)\b|"
        r"my (?:title|position) is\b)\b",
        lower,
    ):
        return "identity"
    if re.search(
        r"\b(?:i (?:prefer|like|love|hate|enjoy|dislike|use|always use|always choose)|"
        r"my (?:favorite|preferred|go-to)\b|i (?:always|never) (?:use|write|do)\b)\b",
        lower,
    ):
        return "preferences"
    if re.search(
        r"\b(?:i need to\b|i have to\b|i must finish\b|pending task\b|to do\b|"
        r"working on\b|currently implementing\b|in progress\b)\b",
        lower,
    ):
        return "tasks"
    if re.search(
        r"\b(?:my goal\b|i want to achieve\b|i plan to\b|i aim to\b|"
        r"my objective\b|i hope to\b|i'm working toward\b)\b",
        lower,
    ):
        return "goals"
    if re.search(
        r"\b(?:i usually\b|i typically\b|every (?:morning|day|week)\b|my routine\b|"
        r"i tend to\b|i regularly\b|my habit\b|on a daily basis\b)\b",
        lower,
    ):
        return "habits"
    if re.search(
        r"\b(?:i know (?:that|how)\b|i learned\b|i understand\b|i'm aware\b|"
        r"it is (?:known|confirmed)\b|we (?:found|confirmed|discovered)\b|"
        r"turns out\b|root cause\b|the (?:fix|solution|cause) is\b)\b",
        lower,
    ):
        return "knowledge"
    if re.search(
        r"\b(?:email\b|slack\b|message\b|meeting\b|i sent\b|they replied\b|"
        r"responded\b|called\b)\b",
        lower,
    ):
        return "communication"
    if re.search(
        r"\b(?:yesterday\b|last (?:week|month|time)\b|we tried\b|it failed\b|"
        r"the deploy\b|the meeting was\b|it happened\b)\b",
        lower,
    ):
        return "events"
    if re.search(
        r"\b(?:every day\b|daily\b|weekly\b|schedule\b|recurring\b|"
        r"i wake up\b|morning routine\b)\b",
        lower,
    ):
        return "routines"
    if re.search(
        r"\b(?:i (?:know|am skilled|am proficient|am experienced)\b|"
        r"years of experience\b|my expertise\b)\b",
        lower,
    ):
        return "skills"
    if re.search(
        r"\b(?:my (?:manager|colleague|friend|team|boss|partner)\b|i work with\b)\b",
        lower,
    ):
        return "relationships"
    return "context"


def _classify_type(clause: str) -> str:
    """Classify as 'semantic' (durable) or 'episodic' (one-off)."""
    if _EPISODIC_MARKERS.search(clause):
        return "episodic"
    return "semantic"


def _normalize_content(clause: str) -> str:
    """Clean and canonicalize a clause."""
    # Remove leading/trailing noise
    content = clause.strip().rstrip(".,;:")
    # Capitalize
    if content and not content[0].isupper():
        content = content[0].upper() + content[1:]
    return content


class Stage2Refiner:
    """Advisory refiner: category + semantic/episodic + cue suggestions."""

    def refine(self, clause: str) -> Optional[dict]:
        """Refine a salient clause. Returns None to skip this clause."""
        content = _normalize_content(clause)
        if len(content) < 10:
            return None
        category = _classify_category(content)
        memory_type = _classify_type(content)
        cue_suggestions = self._suggest_cues(content)
        return {
            "content": content,
            "category": category,
            "memory_type": memory_type,
            "cue_suggestions": cue_suggestions,
        }

    def _suggest_cues(self, content: str) -> list[str]:
        """Suggest additional cues beyond deterministic extraction."""
        from thyra.recall.cue_extractor import extract_raw_cues

        return extract_raw_cues(content, max_cues=5)


# Module-level singleton
REFINER = Stage2Refiner()
