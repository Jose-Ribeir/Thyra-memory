"""Near-duplicate detection and provisional memory insertion."""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Optional

from thyra.config import (
    CONTENT_SEED_CUES,
    CONTENT_SEED_WEIGHT,
    DEDUP_CANDIDATE_LIMIT,
    DEDUP_SIMILARITY_THRESHOLD,
    DECAY_EPISODIC,
    DECAY_SEMANTIC,
    WEAK_ADMIT_DECAY,
    BASE_STRENGTH_AUTOMATIC,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)
from thyra.models.memory import MemoryRecord, compute_content_hash, upsert_cue_edge

# Categories whose probationary memories start at semantic (14-day) decay instead of
# episodic (5-day) decay — giving durable facts time to be re-cited and graduate.
_DURABLE_CATEGORIES = frozenset({"constraints", "identity", "preferences"})

# Directive keywords: when present in content, the memory gets a higher base strength
# so one-shot user instructions survive even if never explicitly re-cited.
# "always"/"never" are intentionally excluded: they also appear in observations
# ("the loop never ran") and would inflate base_strength for junk memories.
_DIRECTIVE_WORDS = frozenset(
    {
        "remember",
        "from now on",
        "make sure",
        "please note",
        "important",
        "note that",
        "keep in mind",
        "don't forget",
        "be aware",
        "going forward",
    }
)

_BASE_STRENGTH_DIRECTIVE = 0.6  # vs BASE_STRENGTH_AUTOMATIC (0.4)


def find_near_match(
    conn: sqlite3.Connection,
    content: str,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    fast: bool = False,
) -> Optional[MemoryRecord]:
    """Find an existing active memory that is semantically near-duplicate to content.

    Returns the best match if similarity ≥ threshold, else None.
    First does an O(1) content_hash equality check, then falls back to FTS5 +
    sentence-transformers cosine similarity (or word-overlap when fast=True).

    fast=True skips the ML model entirely (FTS5 + word-overlap only).
    Use fast=True in latency-sensitive paths (e.g. synchronous MCP tool calls)
    to avoid blocking on a cold model load.
    """
    if not content.strip():
        return None

    # O(1) exact-content short-circuit: if normalized content matches any existing
    # memory verbatim, return it immediately without FTS or ML.
    chash = compute_content_hash(content)
    hash_row = conn.execute(
        "SELECT * FROM memories WHERE content_hash=? AND user_id=? AND agent_id=? AND archived=0 LIMIT 1",
        (chash, user_id, agent_id),
    ).fetchone()
    if hash_row:
        return MemoryRecord.from_row(hash_row, int(time.time() * 1000))

    # FTS5 candidate retrieval — wrap each token as a quoted phrase to prevent
    # special-character injection (hyphens, stars, parens crash FTS5 MATCH).
    fts_tokens = content[:800].split()
    if not fts_tokens:
        return None
    fts_query = " ".join('"' + t.replace('"', '""') + '"' for t in fts_tokens[:20])
    try:
        rows = conn.execute(
            """SELECT m.* FROM memories m
               JOIN memory_fts f ON m.memory_int_id = f.rowid
               WHERE memory_fts MATCH ? AND m.user_id=? AND m.agent_id=? AND m.archived=0
               ORDER BY rank LIMIT ?""",
            (fts_query, user_id, agent_id, DEDUP_CANDIDATE_LIMIT),
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    if fast:
        return _word_overlap_match(content, rows, threshold)

    # Cosine similarity ranking
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        from thyra.formation.refiner import _get_model_and_embeddings

        model, _, _ = _get_model_and_embeddings()
        if model is None:
            raise ImportError("model unavailable")

        query_emb = model.encode([content])
        candidate_texts = [r["content"] for r in rows]
        cand_embs = model.encode(candidate_texts)
        sims = cosine_similarity(query_emb, cand_embs)[0]
        best_idx = int(sims.argmax())
        if sims[best_idx] >= threshold:
            return MemoryRecord.from_row(rows[best_idx], int(time.time() * 1000))
        return None
    except Exception:
        # Fallback: treat top FTS5 result as near-match if content overlaps significantly
        return _word_overlap_match(content, rows, threshold)


def _word_overlap_match(content: str, rows, threshold: float) -> Optional[MemoryRecord]:
    """Simple word-overlap fallback dedup."""
    import re

    query_words = set(re.findall(r"\b[a-z]{4,}\b", content.lower()))
    if not query_words:
        return None
    for row in rows:
        cand_words = set(re.findall(r"\b[a-z]{4,}\b", row["content"].lower()))
        if not cand_words:
            continue
        overlap = len(query_words & cand_words) / len(query_words | cand_words)
        if overlap >= threshold * 0.7:  # relaxed threshold for word overlap
            return MemoryRecord.from_row(row, int(time.time() * 1000))
    return None


def _probationary_decay_for(category: str) -> float:
    """Return the appropriate probationary decay rate for a category.

    Durable categories (constraints, identity, preferences) start at semantic
    decay (~14-day half-life) so they survive long enough to be re-cited and
    graduate.  Everything else uses episodic decay (~5-day) as before.
    """
    return DECAY_SEMANTIC if category in _DURABLE_CATEGORIES else DECAY_EPISODIC


def _is_directive_content(content: str) -> bool:
    """True if the content contains an explicit user directive word."""
    lower = content.lower()
    return any(word in lower for word in _DIRECTIVE_WORDS)


def insert_as_probationary(
    conn: sqlite3.Connection,
    content: str,
    category: str,
    memory_type: str,
    decay_rate: float,
    user_id: str,
    agent_id: str,
    cue_suggestions: list[str],
    weak_signal: bool = False,
) -> str:
    """Insert a new auto-formed memory at probationary strength.

    Anti-immortality rule: probationary memories start at a decay rate based on
    their category — durable categories (constraints/identity/preferences) get
    semantic decay (~14-day) so they survive to graduation; everything else gets
    episodic decay (~5-day).  Explicit user directives get a higher base strength.

    L3 weak-admit tagging: when ``weak_signal`` is True (the admit passed on weak
    signal alone — no directive, no confirmed reference, salience just over the
    threshold) the memory starts on the steepest episodic horizon regardless of
    category, so an unused borderline admit fades in days rather than weeks. A
    weak admit that *is* re-cited still graduates normally.
    """
    mem_id = "m_" + uuid.uuid4().hex[:16]
    now = int(time.time() * 1000)
    actual_decay = (
        WEAK_ADMIT_DECAY if weak_signal else _probationary_decay_for(category)
    )
    base_strength = (
        _BASE_STRENGTH_DIRECTIVE
        if _is_directive_content(content)
        else BASE_STRENGTH_AUTOMATIC
    )
    chash = compute_content_hash(content)

    with conn:
        conn.execute(
            """INSERT INTO memories
               (id, user_id, agent_id, content, locked, category, memory_type,
                base_strength, decay_rate, last_access, created_at, probationary,
                archived, archived_at, use_count, content_hash)
               VALUES (?,?,?,?,0,?,?,?,?,?,?,1,0,NULL,0,?)""",
            (
                mem_id,
                user_id,
                agent_id,
                content,
                category,
                memory_type,
                base_strength,
                actual_decay,
                now,
                now,
                chash,
            ),
        )

    # Seed cues from content
    from thyra.recall.cue_extractor import extract_raw_cues

    cues = extract_raw_cues(content, max_cues=CONTENT_SEED_CUES)
    for cue in cues:
        upsert_cue_edge(
            conn,
            cue,
            mem_id,
            user_id,
            agent_id,
            weight=CONTENT_SEED_WEIGHT,
            candidate=False,
        )

    # Add suggested cues as candidate edges
    from thyra.config import SYNONYM_CUE_WEIGHT

    for cue in cue_suggestions:
        if cue not in cues:
            upsert_cue_edge(
                conn,
                cue,
                mem_id,
                user_id,
                agent_id,
                weight=SYNONYM_CUE_WEIGHT,
                candidate=True,
            )

    return mem_id
