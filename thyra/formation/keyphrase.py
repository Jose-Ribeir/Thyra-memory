"""Semantic keyphrase extraction using KeyBERT + sentence-transformers.

Supplements the regex salience gate with embedding-based extraction that
catches research findings, technical facts, and architectural decisions that
don't contain directive words.  The model is shared with the refiner to avoid
double-loading.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

log = logging.getLogger("thyra.formation.keyphrase")

_kw_model = None
_kw_lock = threading.Lock()


def _get_kw_model():
    """Lazy singleton — reuses the same sentence-transformer the refiner uses."""
    global _kw_model
    if _kw_model is not None:
        return _kw_model if _kw_model is not False else None
    with _kw_lock:
        if _kw_model is None:
            try:
                from keybert import KeyBERT

                # Import the refiner to reuse its already-loaded SentenceTransformer.
                # If the refiner hasn't loaded yet this will trigger one load total.
                from thyra.formation.refiner import _get_model_and_embeddings

                st_model, _, _ = _get_model_and_embeddings()
                if st_model is not None:
                    _kw_model = KeyBERT(model=st_model)
                else:
                    # sentence-transformer unavailable (model not loaded yet or
                    # rules-only thread).  Mark as unavailable rather than letting
                    # KeyBERT() trigger its own model download which can hang.
                    _kw_model = False
                log.debug("KeyBERT extractor ready")
            except Exception as exc:
                log.warning("KeyBERT unavailable: %s", exc)
                _kw_model = False  # sentinel
    return _kw_model if _kw_model is not False else None


def extract_keyphrases(
    text: str,
    top_n: int = 15,
    ngram_range: tuple[int, int] = (1, 4),
    diversity: float = 0.55,
    min_score: float = 0.25,
) -> list[str]:
    """Return the top keyphrases from *text* using semantic similarity.

    Uses Max Marginal Relevance (MMR) so results are diverse rather than
    repetitive near-duplicates of the highest-scoring phrase.

    Returns a list of phrase strings (not full sentences) that scored above
    *min_score*.  Callers should feed these through the refiner + dedup
    pipeline exactly like regex-extracted clauses.
    """
    if not text or not text.strip():
        return []

    # In rules-only mode (worker thread) skip ML entirely — avoids blocking
    # on a SentenceTransformer load inside the MCP server process.
    from thyra.formation.refiner import _thread_local

    if getattr(_thread_local, "rules_only", False):
        return []

    model = _get_kw_model()
    if model is None:
        return []

    try:
        results = model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=diversity,
        )
        return [phrase for phrase, score in results if score >= min_score]
    except Exception as exc:
        log.warning("KeyBERT extraction failed: %s", exc)
        return []
