"""Automatic memory formation pipeline — orchestrator."""

from __future__ import annotations

import logging
import re
import sqlite3
import time

from thyra.config import (
    BASE_STRENGTH_AUTOMATIC,
    DECAY_EXPLICIT,
    SURFACED_BOOST,
    STRENGTH_CAP,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)
from thyra.models.delta import DeltaEvent
from thyra.models.memory import get_flag, update_memory_strength

log = logging.getLogger("thyra.formation")


def run_formation_pipeline(
    conn: sqlite3.Connection,
    delta: DeltaEvent,
) -> list[tuple[str, str]]:
    """Scan the turn for salient clauses and form memories automatically.

    Returns list of (action, memory_id) pairs: action is "created" or "reinforced".
    """
    user_id = delta.user_id
    agent_id = delta.agent_id

    # Formation master switch
    if get_flag(conn, "formation_enabled", user_id, agent_id).lower() != "true":
        return []

    # Text to analyze: user prompt + assistant response (kept separate for scoring)
    user_text, agent_text = _build_text_parts(delta)
    if not user_text.strip() and not agent_text.strip():
        return []

    from thyra.formation.salience import extract_salient_clauses
    from thyra.formation.refiner import REFINER
    from thyra.formation.dedup import find_near_match, insert_as_probationary

    # Pass 1: regex salience gate — fast, catches directives and disclosures.
    candidates = extract_salient_clauses(
        user_text, conn, user_id, agent_id, source="user"
    )
    if agent_text:
        candidates += extract_salient_clauses(
            agent_text, conn, user_id, agent_id, source="agent"
        )

    # Pass 2: semantic keyphrase extraction — catches research findings and
    # technical facts that don't contain directive words.  Runs on the full
    # combined text so KeyBERT can score phrases against the whole document.
    # L1-vetoed clauses are excluded from the keyphrase input so the semantic
    # pass cannot re-introduce transient/dangling content that the regex gate
    # (which bypasses compute_salience via the fixed 0.60 score) just rejected.
    try:
        from thyra.formation.keyphrase import extract_keyphrases
        from thyra.formation.salience import _split_to_clauses, _is_vetoed

        kp_source = "\n".join(
            c
            for c in _split_to_clauses(
                "\n".join(p for p in [user_text, agent_text] if p)
            )
            if not _is_vetoed(c)
        )
        kp_phrases = extract_keyphrases(kp_source) if kp_source.strip() else []
        # Only add keyphrases that aren't already covered by the regex pass
        # and that pass the noise gate (code, paths, narration are excluded).
        existing_clauses = {c["clause"].lower() for c in candidates}
        for phrase in kp_phrases:
            if phrase.lower() not in existing_clauses and len(phrase) > 10:
                if not _is_noise_sentence(phrase) and not _is_vetoed(phrase):
                    candidates.append(
                        {
                            "clause": phrase,
                            "salience": 0.60,  # semantic extraction implies relevance
                            "novelty": 1.0,  # novelty will be checked in dedup
                            "is_correction": False,
                        }
                    )
    except Exception as exc:
        log.debug("Keyphrase extraction skipped: %s", exc)
    if not candidates:
        return []

    actions = []
    now = int(time.time() * 1000)

    distiller_context = "\n".join(p for p in [user_text, agent_text] if p)
    for candidate in candidates:
        clause = candidate["clause"]
        try:
            # L2 distill-or-drop (advisory). Runs only on candidates that already
            # passed L1, behind DISTILLER_ENABLED, with prewarm-or-skip discipline.
            # verdict None → skipped (disabled or model not resident) → fall back to
            # the L1-gated refiner path. keep=False → drop. keep=True → use the
            # clean atomic fact the judge produced.
            distilled_category = None
            distilled_kind = None
            try:
                from thyra.formation.distiller import distill

                verdict = distill(clause, context=distiller_context)
            except Exception as exc:
                log.debug("Distiller skipped for clause %r: %s", clause[:50], exc)
                verdict = None
            if verdict is not None:
                if not verdict["keep"]:
                    log.debug("Distiller dropped clause %r", clause[:50])
                    continue
                clause = verdict["fact"]
                distilled_category = verdict.get("category")
                distilled_kind = verdict.get("kind")

            refined = REFINER.refine(clause)
            if refined is None:
                continue

            content = refined["content"]
            category = distilled_category or refined["category"]
            memory_type = distilled_kind or refined["memory_type"]
            cue_suggestions = refined["cue_suggestions"]

            # Dedup: near-match → reinforce, no-match → insert.
            # fast=True uses FTS5 + word-overlap only (no ML model load).
            # The formation pipeline runs in a background thread where a blocking
            # model load would stall the worker; the fast path is sufficient here.
            existing = find_near_match(conn, content, user_id, agent_id, fast=False)
            if existing:
                new_strength = min(
                    STRENGTH_CAP, existing.base_strength + SURFACED_BOOST
                )
                update_memory_strength(
                    conn, existing.id, new_strength, now, user_id, agent_id
                )
                actions.append(("reinforced", existing.id))
                log.debug("Formation: reinforced existing %s", existing.id)
            else:
                from thyra.config import DECAY_EPISODIC

                weak_signal = _is_weak_admit(candidate, content)
                mem_id = insert_as_probationary(
                    conn,
                    content,
                    category,
                    memory_type,
                    DECAY_EPISODIC,
                    user_id,
                    agent_id,
                    cue_suggestions,
                    weak_signal=weak_signal,
                )
                actions.append(("created", mem_id))
                log.debug(
                    "Formation: created %s (cat=%s, type=%s, weak=%s)",
                    mem_id,
                    category,
                    memory_type,
                    weak_signal,
                )

        except Exception as exc:
            log.warning("Formation error for clause %r: %s", clause[:50], exc)
            continue

    # Invalidate hot cache if anything changed
    if actions:
        from thyra.recall.cache import HOT_CACHE

        HOT_CACHE.invalidate(f"snapshot:{user_id}:{agent_id}")

    return actions


# ── Weak-admit detection (L3) ────────────────────────────────────────────────────

# Margin above SALIENCE_THRESHOLD within which an admit counts as "just over" the
# bar.  A single weak positive signal (e.g. a bare self-disclosure at 0.42) lands
# inside this band; any compounded or strong signal lands above it.
_WEAK_ADMIT_MARGIN = 0.12


def _is_weak_admit(candidate: dict, content: str) -> bool:
    """True when an admit passed on weak signal alone (L3 master §9.5).

    Weak = no directive, no confirmed/technical reference, not a correction, and
    salience only just over the threshold.  Such admits get the steepest episodic
    decay so an unused borderline admit fades in days, not weeks.  Strong-signal
    admits (directives, technical refs, findings, corrections, keyphrase hits)
    keep their normal probationary tier.
    """
    from thyra.config import SALIENCE_THRESHOLD
    from thyra.formation.salience import (
        _DIRECTIVE_RE,
        _TECHNICAL_REF_RE,
        _AGENT_FINDING_RE,
    )

    if candidate.get("is_correction"):
        return False
    if (
        _DIRECTIVE_RE.search(content)
        or _TECHNICAL_REF_RE.search(content)
        or _AGENT_FINDING_RE.search(content)
    ):
        return False
    return candidate.get("salience", 0.0) <= SALIENCE_THRESHOLD + _WEAK_ADMIT_MARGIN


# ── Noise-gate patterns ────────────────────────────────────────────────────────

# Lines that indicate tool/machine output rather than human speech or findings.
_TOOL_OUTPUT_LINE_RE = re.compile(
    r"(?:"
    r"Output is being written to:"  # task output header
    r"|Exit code \d+"  # process exit code
    r"|Traceback \(most recent call last\)"  # Python traceback
    r"|^\s*\d+[\t ]"  # Read-tool line numbers: "12\t..."
    r"|^\s*\d+[-:]\s"  # grep output:           "343- ..."
    r"|^>>>\s"  # Python REPL prompt
    r"|^\.\.\.\s"  # Python REPL continuation
    r"|^\s*m_[0-9a-f]{16}\b"  # Thyra memory ID (listing output)
    r")",
    re.MULTILINE | re.IGNORECASE,
)

# Matches a Thyra memory ID anywhere in a short string (e.g. standalone reference).
_MEMORY_ID_RE = re.compile(r"\bm_[0-9a-f]{16}\b")

# Matches Thyra XML injection blocks that must be stripped before formation.
_THYRA_INJECTION_RE = re.compile(
    r"<thyra_memories\b[^>]*>[\s\S]*?</thyra_memories>"
    r"|\[MEMORY\s+id=[\"']m_[0-9a-f]{16}[\"'][^\]]*\][\s\S]*?\[/MEMORY\]"
    r"|<memories_used>[^<]*</memories_used>",
    re.IGNORECASE,
)

# Matches a line that is essentially a bare filesystem path (no prose around it).
_BARE_PATH_RE = re.compile(
    r"^[A-Za-z]:\\[^\s,)>\"']{4,}$"  # Windows path
    r"|^/[a-z][a-zA-Z0-9_/.-]{4,}$",  # Unix path
)

# Sentence-level procedural narration: the agent narrating its own thought process
# rather than stating a durable fact.  Matched only at sentence start (prefix).
_NARRATION_PREFIX_RE = re.compile(
    r"^(?:"
    r"let me\b|let's\b|now i\b|i'll\b|i will\b|i need to\b|i should\b|"
    r"first[,\s]|next[,\s]|looking at\b|let me check\b|now let me\b|"
    r"i'm (?:going to|now|about to)\b|to do this\b|starting with\b|"
    r"i have (?:the|a|everything|enough|all)\b|"
    r"i(?:'ve| have) (?:got|confirmed|verified|checked|now)\b|"
    r"good[,.]?\s+(?:i|now)\b"
    r")",
    re.IGNORECASE,
)

# Markdown table separator / pure code lines.
_JUNK_SHAPE_RE = re.compile(r"\|[-:|]+\|")  # table rule like |---|---|


def _clean_for_formation(text: str) -> str:
    """Strip machine/tool noise from text before salience scoring.

    Removes fenced code blocks, line-numbered output lines, bare tool-output
    lines, and Thyra self-referential injection blocks.
    Returns cleaned text (may be empty string).
    """
    if not text:
        return text
    # 0. Remove Thyra XML injection blocks (<thyra_memories>, [MEMORY...],
    #    <memories_used>) — these are meta-system content, not facts about the world.
    text = _THYRA_INJECTION_RE.sub("", text)
    # 1. Remove fenced code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # 2. Remove inline code spans (single-backtick)
    text = re.sub(r"`[^`\n]{1,200}`", "", text)
    # 3. Remove lines that look like tool output / line-numbered code
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        # Skip tool-output markers
        if _TOOL_OUTPUT_LINE_RE.search(stripped):
            continue
        # Skip bare filesystem paths
        if _BARE_PATH_RE.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def _is_noise_sentence(s: str) -> bool:
    """Return True if sentence s should be excluded from formation candidates.

    Catches: tool/machine output markers, procedural narration, code-like
    content, markdown table rules, dominant identifier/path content, and
    self-referential Thyra memory listings.
    """
    if not s or len(s) < 8:
        return True
    stripped = s.strip()
    # Tool / machine output markers (includes lines starting with m_<hex16>)
    if _TOOL_OUTPUT_LINE_RE.search(stripped):
        return True
    # Self-referential: text dominated by Thyra memory IDs
    # (e.g. a listing of other memories the agent produced in its response)
    ids_found = _MEMORY_ID_RE.findall(stripped)
    if ids_found:
        # If the clause itself IS a memory ID reference with no surrounding prose
        # (short clause, or more than a third of words are IDs), treat as noise.
        non_id_text = _MEMORY_ID_RE.sub("", stripped).strip()
        if len(non_id_text) < 30 or len(ids_found) >= 2:
            return True
    # Procedural narration prefix
    if _NARRATION_PREFIX_RE.match(stripped):
        return True
    # Markdown table rule
    if _JUNK_SHAPE_RE.search(s):
        return True
    # Code-like ratio: if more than 60% of characters are non-alphabetic, skip.
    alpha = sum(1 for c in s if c.isalpha())
    if alpha / len(s) < 0.40:
        return True
    # Bare path dominates: strip the path and see if anything is left
    remainder = re.sub(
        r'[A-Za-z]:\\[^\s,)>"\']{4,}|/[a-z][a-zA-Z0-9_/.-]{4,}', "", s
    ).strip()
    if remainder and alpha / len(s) < 0.30:
        return True
    return False


_AGENT_BOILERPLATE_RE = re.compile(
    r"^(?:let me|i'll|i will|sure|okay|of course|here(?:'s| is)|"
    r"sounds good|got it|thank|you're welcome|no problem|happy to|"
    r"certainly|absolutely|great|perfect)\b",
    re.IGNORECASE,
)


def _build_text_parts(delta: DeltaEvent) -> tuple[str, str]:
    """Return (user_text, agent_text) for separate salience scoring.

    Keeping them separate lets the salience scorer apply source-aware signal
    weights: user text is scored for preferences/directives; agent text is
    scored for research findings, technical references, and confirmed facts.

    All reasonably-sized, non-boilerplate, non-noise agent sentences are passed
    to the salience scorer — the regex signals (_DIRECTIVE_RE etc.) act as a score
    boost there, not as a hard entry gate.
    """
    user_text = _clean_for_formation(delta.raw_user_text or "")
    # Apply the same per-sentence noise gate that agent text goes through — catches
    # code-heavy fragments, tool output that slipped past _clean_for_formation, and
    # narration-prefix patterns.
    if user_text:
        _user_sentences = re.split(r"(?<=[.!?])\s+", user_text)
        user_text = "\n".join(
            s.strip()
            for s in _user_sentences
            if s.strip() and not _is_noise_sentence(s.strip())
        )

    agent_parts: list[str] = []
    if delta.raw_assistant_text:
        cleaned = _clean_for_formation(delta.raw_assistant_text)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        for s in sentences:
            s = s.strip()
            if len(s) < 20:
                continue
            if _AGENT_BOILERPLATE_RE.match(s):
                continue
            if _is_noise_sentence(s):
                continue
            agent_parts.append(s)
    agent_text = "\n".join(agent_parts)

    return user_text, agent_text
