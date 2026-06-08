"""Synonym / semantic cue expansion via sentence-transformers (Stage 6).

This is a lazy-loaded, async-safe module. In Stage 2 it is imported but
expand_cues_with_synonyms() returns an empty list until Stage 6 wires it up.
The model loads on first call and is cached for the process lifetime.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Sequence

from thyra.config import (
    SYNONYM_CUE_WEIGHT,
    SYNONYM_MAX_EXPANSIONS,
    SYNONYM_SIMILARITY_THRESHOLD,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    _model = SentenceTransformer("all-MiniLM-L6-v2")
                except Exception:
                    _model = None
    return _model


def expand_cues_with_synonyms(
    conn: sqlite3.Connection,
    cues: Sequence[str],
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> list[str]:
    """Return additional cues semantically similar to the fired cues.

    Returns [] if sentence-transformers is unavailable or no cues match.
    Runs off the recall hot path (called from consolidation worker).
    """
    model = _get_model()
    if model is None:
        return []

    # Fetch all existing cue_ids for this agent
    rows = conn.execute(
        "SELECT cue_id FROM cue_nodes WHERE user_id=? AND agent_id=?",
        (user_id, agent_id),
    ).fetchall()
    existing_cues = [r["cue_id"] for r in rows if r["cue_id"] not in cues]
    if not existing_cues:
        return []

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        query_embs = model.encode(list(cues))
        cand_embs = model.encode(existing_cues)
        sims = cosine_similarity(
            query_embs, cand_embs
        )  # shape (len(cues), len(existing))
        max_sims = sims.max(axis=0)  # best similarity per candidate
        above = [
            existing_cues[i]
            for i, s in enumerate(max_sims)
            if s >= SYNONYM_SIMILARITY_THRESHOLD
        ]
        return above[:SYNONYM_MAX_EXPANSIONS]
    except Exception:
        return []


def seed_synonym_edges_for_memory(
    conn: sqlite3.Connection,
    memory_id: str,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> int:
    """Add candidate cue edges for synonyms of a memory's existing cues.

    Returns count of new edges added.
    """
    rows = conn.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (memory_id, user_id, agent_id),
    ).fetchall()
    memory_cues = [r["cue_id"] for r in rows]
    if not memory_cues:
        return 0

    synonyms = expand_cues_with_synonyms(conn, memory_cues, user_id, agent_id)
    if not synonyms:
        return 0

    import time as _time

    now = int(_time.time() * 1000)
    count = 0
    with conn:
        for cue_id in synonyms:
            existing = conn.execute(
                "SELECT 1 FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
                (cue_id, memory_id, user_id, agent_id),
            ).fetchone()
            if existing:
                continue
            conn.execute(
                """INSERT INTO cue_edges (cue_id, memory_id, user_id, agent_id, weight, fire_count, use_count, candidate, created_at)
                   VALUES (?,?,?,?,?,0,0,1,?)""",
                (cue_id, memory_id, user_id, agent_id, SYNONYM_CUE_WEIGHT, now),
            )
            count += 1
    return count
