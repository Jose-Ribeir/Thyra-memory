"""Cue extraction from text: stopwords, length filter, IDF.

This is the full Stage-2 implementation; a minimal stub used by Stage-1
models is also here (extract_raw_cues) so that cue seeding works from day 1.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from typing import Sequence

from thyra.config import (
    DISCRIMINABILITY_FLOOR,
    MAX_CUES_PER_TURN,
    MIN_CUE_LENGTH,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)

# ── Stopwords ──────────────────────────────────────────────────────────────────

STOPWORDS: frozenset[str] = frozenset(
    {
        # 3-letter function words (newly needed after MIN_CUE_LENGTH dropped to 3)
        "ago",
        "all",
        "and",
        "any",
        "are",
        "but",
        "can",
        "did",
        "for",
        "got",
        "had",
        "has",
        "her",
        "him",
        "his",
        "how",
        "its",
        "may",
        "nor",
        "off",
        "one",
        "our",
        "out",
        "the",
        "via",
        "who",
        "why",
        # 4+ letter stopwords
        "about",
        "above",
        "after",
        "again",
        "against",
        "also",
        "although",
        "always",
        "another",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "came",
        "come",
        "could",
        "didn",
        "does",
        "doing",
        "done",
        "down",
        "during",
        "each",
        "even",
        "ever",
        "every",
        "from",
        "further",
        "give",
        "given",
        "have",
        "having",
        "here",
        "himself",
        "however",
        "into",
        "itself",
        "just",
        "know",
        "like",
        "made",
        "make",
        "many",
        "more",
        "most",
        "much",
        "myself",
        "need",
        "never",
        "next",
        "note",
        "nothing",
        "once",
        "only",
        "other",
        "otherwise",
        "over",
        "same",
        "should",
        "since",
        "some",
        "still",
        "such",
        "tell",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "therefore",
        "these",
        "they",
        "think",
        "this",
        "those",
        "through",
        "thus",
        "told",
        "too",
        "under",
        "until",
        "upon",
        "very",
        "want",
        "was",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
        "your",
        "yourself",
    }
)

_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b")


def extract_raw_cues(text: str, max_cues: int = 8) -> list[str]:
    """Extract keyword cues from text (used by models.memory for seeding).

    Applies morphology normalization so stored cue_ids match what extract_cues
    produces at recall time (otherwise "data" stored != "datum" looked up).
    """
    from thyra.recall.morphology import normalize_cue

    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [
        normalize_cue(t)
        for t in tokens
        if len(t) >= MIN_CUE_LENGTH and t not in STOPWORDS
    ]
    counts = Counter(tokens)
    return [cue for cue, _ in counts.most_common(max_cues)]


def extract_cues(text: str, max_cues: int = MAX_CUES_PER_TURN) -> list[str]:
    """Extract and normalize cues for a turn (used by recall pipeline)."""
    from thyra.recall.morphology import normalize_cue

    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [
        normalize_cue(t)
        for t in tokens
        if len(t) >= MIN_CUE_LENGTH and t not in STOPWORDS
    ]
    counts = Counter(tokens)
    return [cue for cue, _ in counts.most_common(max_cues)]


def compute_idf(
    conn: sqlite3.Connection,
    cues: Sequence[str],
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> dict[str, float]:
    """Return per-agent IDF discrimination score for each cue."""
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchone()
    M = row[0] if row else 0

    result: dict[str, float] = {}
    for cue in cues:
        df_row = conn.execute(
            "SELECT df FROM cue_nodes WHERE cue_id=? AND user_id=? AND agent_id=?",
            (cue, user_id, agent_id),
        ).fetchone()
        df = df_row["df"] if df_row else 0
        if M == 0:
            result[cue] = DISCRIMINABILITY_FLOOR
        else:
            score = math.log(1 + M / max(1, df)) / math.log(1 + M)
            result[cue] = max(DISCRIMINABILITY_FLOOR, score)
    return result
