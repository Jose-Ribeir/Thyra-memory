"""L2 distill-or-drop judge — advisory meaning-level formation gate.

Pattern: Mem0 / LangMem extraction (master schematic §9.3). The judge takes a
salient clause plus minimal turn context and returns either a clean, atomic,
self-contained fact or a verdict to drop it. This catches non-facts that the L1
regex vetoes cannot — chatter, restated questions, and transient instructions
whose surface form looks durable.

Architecture constraints (master §9.3, plan §4.2):
  * Advisory, never load-bearing. The pipeline runs in a background worker thread
    that must never block on a cold model load. The distiller follows the SAME
    prewarm-or-skip discipline as the MiniLM refiner: if the model is not already
    resident (or we are in a rules-only worker thread), distillation is SKIPPED and
    the caller falls back to an L1-gated insert. We never trigger a cold load here.
  * Gated behind DISTILLER_ENABLED (default off until validated).
  * Only runs on candidates that already passed L1 → bounded call volume.
  * Degradation: model unavailable/invalid → return None (skip). Worst case is
    "store a slightly rough genuine fact", never "store a transient fragment",
    because the L1 vetoes still ran upstream.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import threading
import time
from typing import Optional

from thyra.config import THYRA_DB_PATH

log = logging.getLogger("thyra.formation.distiller")

# Resident-model singleton. None = never attempted; False = attempted and
# unavailable (skip); otherwise a callable text-generation backend.
_judge = None
_judge_lock = threading.Lock()

# Verdict log: every (clause → verdict/fact) pair, for future distillation of the
# judge itself (master §9.6). JSONL alongside the DB.
_VERDICT_LOG = pathlib.Path(THYRA_DB_PATH).parent / "distiller_verdicts.jsonl"

# Live taxonomy is reused for the category field — never a fixed classifier
# (master §7). We accept whatever the judge returns and let the refiner/category
# manager reconcile it; this is only a hint.
_VALID_KINDS = {"semantic", "episodic"}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You distill conversation fragments into durable memories. Given a CLAUSE and "
    "minimal CONTEXT, decide whether the clause states a fact worth remembering "
    "across sessions (a standing preference, identity fact, constraint, decision, "
    "or confirmed finding). Drop chatter, restated questions, and transient "
    "single-turn instructions. If you keep it, rewrite it as ONE short, atomic, "
    "self-contained declarative statement with any dangling referents resolved "
    "from context; if it cannot be made self-contained, drop it. "
    'Respond with ONLY JSON: {"keep": bool, "fact": str|null, '
    '"category": str, "kind": "semantic"|"episodic"}.'
)


def _model_is_resident() -> bool:
    """True only when the judge backend is already loaded (not None, not False)."""
    return _judge not in (None, False)


def _load_judge_backend():
    """Best-effort, non-blocking attempt to obtain a text-generation backend.

    Returns a callable ``generate(prompt: str) -> str`` or None. Any missing
    dependency or load failure yields None — the caller treats that as "skip".
    The chosen model is a local small instruct model (master §9.3 ~8B, local, for
    data sovereignty), selected via the THYRA_DISTILLER_MODEL env var and loaded
    from the same HF cache the refiner uses.
    """
    import os

    model_name = os.environ.get("THYRA_DISTILLER_MODEL", "").strip()
    if not model_name:
        log.debug("Distiller: THYRA_DISTILLER_MODEL unset — skipping")
        return None
    try:
        from thyra.formation.refiner import _ensure_hf_cache

        _ensure_hf_cache()
        from transformers import pipeline as hf_pipeline

        gen = hf_pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"local_files_only": True},
        )

        def _generate(prompt: str) -> str:
            out = gen(
                prompt, max_new_tokens=160, do_sample=False, return_full_text=False
            )
            return out[0]["generated_text"] if out else ""

        return _generate
    except Exception as exc:  # noqa: BLE001 — any failure is a clean skip
        log.warning("Distiller backend unavailable: %s", exc)
        return None


def _get_judge():
    """Return the resident judge backend, honoring prewarm-or-skip discipline.

    In a rules-only worker thread (or before any prewarm) we never trigger a cold
    load: return the backend only if it is already resident, else None.
    """
    global _judge
    from thyra.formation.refiner import _thread_local

    rules_only = getattr(_thread_local, "rules_only", False)
    if rules_only:
        return _judge if _model_is_resident() else None
    if _judge is not None:
        return _judge if _judge is not False else None
    with _judge_lock:
        if _judge is None:
            backend = _load_judge_backend()
            _judge = backend if backend is not None else False
    return _judge if _judge is not False else None


def prewarm() -> bool:
    """Eagerly load the judge backend (call OUTSIDE worker threads).

    Returns True if the backend is resident afterwards. The worker can then use it
    via the rules-only fast path without ever blocking on a cold load.
    """
    from thyra.config import DISTILLER_ENABLED

    if not DISTILLER_ENABLED:
        return False
    global _judge
    with _judge_lock:
        if _judge is None:
            backend = _load_judge_backend()
            _judge = backend if backend is not None else False
    return _model_is_resident()


def _log_verdict(clause: str, verdict: Optional[dict]) -> None:
    try:
        with open(_VERDICT_LOG, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": int(time.time() * 1000),
                        "clause": clause,
                        "verdict": verdict,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:  # noqa: BLE001 — logging must never break formation
        pass


def _parse_verdict(raw: str) -> Optional[dict]:
    """Parse the judge's JSON response into a normalized verdict, or None."""
    m = _JSON_OBJ_RE.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    keep = bool(obj.get("keep"))
    fact = obj.get("fact")
    if keep and (not isinstance(fact, str) or len(fact.strip()) < 10):
        # "keep" with no usable fact is incoherent — treat as drop.
        return {"keep": False, "fact": None, "category": None, "kind": "semantic"}
    kind = obj.get("kind") if obj.get("kind") in _VALID_KINDS else "semantic"
    category = obj.get("category") if isinstance(obj.get("category"), str) else None
    return {
        "keep": keep,
        "fact": fact.strip() if keep and isinstance(fact, str) else None,
        "category": category,
        "kind": kind,
    }


def distill(clause: str, context: str = "") -> Optional[dict]:
    """Judge a clause that already passed L1.

    Returns:
      * ``None`` — distillation was skipped (disabled or model not resident). The
        caller MUST fall back to the L1-gated insert path.
      * ``{"keep": False, ...}`` — drop this candidate (meaning-level non-fact).
      * ``{"keep": True, "fact": <clean atomic fact>, "category": <hint|None>,
        "kind": "semantic"|"episodic"}`` — store the distilled fact.
    """
    from thyra.config import DISTILLER_ENABLED

    if not DISTILLER_ENABLED:
        return None
    judge = _get_judge()
    if judge is None:
        return None  # prewarm-or-skip: never block the worker
    prompt = (
        f"{_SYSTEM_PROMPT}\n\nCONTEXT:\n{context[:600]}\n\nCLAUSE:\n{clause}\n\nJSON:"
    )
    try:
        raw = judge(prompt)
    except Exception as exc:  # noqa: BLE001 — any runtime failure is a clean skip
        log.warning("Distiller generation failed: %s", exc)
        return None
    verdict = _parse_verdict(raw)
    _log_verdict(clause, verdict)
    return verdict
