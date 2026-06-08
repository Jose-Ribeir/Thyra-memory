"""Situation edge crystallization: cue conjunctions → real situation edges.

The key insight: SITUATION_MIN_FIRES=5 can never be reached in a single
3-turn window. Stats must be accumulated cumulatively in the DB across many
calls, then thresholds checked once enough data exists.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

from thyra.config import (
    SITUATION_MIN_FIRES,
    SITUATION_MIN_RATE,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)
from thyra.models.delta import DeltaEvent


def crystallize_situations(
    conn: sqlite3.Connection,
    window: list[DeltaEvent],
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> int:
    """Accumulate (cue-conjunction, memory) co-occurrence stats in the DB,
    then promote candidates that cross SITUATION_MIN_FIRES and SITUATION_MIN_RATE.

    Returns number of edges newly promoted from candidate → real this call.
    """
    now = int(time.time() * 1000)
    promoted = 0

    for delta in window:
        if not delta.cues_fired:
            continue
        # Top-3 sorted cues form the conjunction key
        top_cues = sorted(delta.cues_fired)[:3]
        if not top_cues:
            continue
        cue_set_json = json.dumps(top_cues)

        used_set = set(delta.memories_declared) & set(delta.memories_served)
        served_set = set(delta.memories_served)

        for mem_id in served_set:
            used = mem_id in used_set
            _upsert_situation_candidate(
                conn,
                cue_set_json,
                mem_id,
                user_id,
                agent_id,
                fired=1,
                used=1 if used else 0,
                now=now,
            )

    # Promote all candidates that now cross both thresholds
    promoted = _promote_ready_candidates(conn, user_id, agent_id)
    return promoted


def _upsert_situation_candidate(
    conn: sqlite3.Connection,
    cue_set_json: str,
    mem_id: str,
    user_id: str,
    agent_id: str,
    fired: int,
    used: int,
    now: int,
) -> None:
    """Increment fire/use counts for this (cue_set, memory) pair."""
    existing = conn.execute(
        """SELECT sit_id, fire_count, use_count, candidate
           FROM situation_edges
           WHERE memory_id=? AND cue_set=? AND user_id=? AND agent_id=?""",
        (mem_id, cue_set_json, user_id, agent_id),
    ).fetchone()

    if existing:
        new_fires = existing["fire_count"] + fired
        new_uses = existing["use_count"] + used
        use_rate = new_uses / new_fires if new_fires > 0 else 0.0
        conn.execute(
            """UPDATE situation_edges
               SET fire_count=?, use_count=?, weight=?
               WHERE sit_id=?""",
            (new_fires, new_uses, use_rate, existing["sit_id"]),
        )
    else:
        sit_id = "sit_" + uuid.uuid4().hex[:12]
        use_rate = used / fired if fired > 0 else 0.0
        conn.execute(
            """INSERT INTO situation_edges
               (sit_id, memory_id, user_id, agent_id, cue_set, weight,
                fire_count, use_count, candidate, created_at)
               VALUES (?,?,?,?,?,?,?,?,1,?)""",
            (
                sit_id,
                mem_id,
                user_id,
                agent_id,
                cue_set_json,
                use_rate,
                fired,
                used,
                now,
            ),
        )


def _promote_ready_candidates(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> int:
    """Promote situation edges that cross both fire and use-rate thresholds."""
    rows = conn.execute(
        """SELECT sit_id, fire_count, use_count FROM situation_edges
           WHERE user_id=? AND agent_id=? AND candidate=1
             AND fire_count >= ?""",
        (user_id, agent_id, SITUATION_MIN_FIRES),
    ).fetchall()
    promoted = 0
    for row in rows:
        use_rate = row["use_count"] / row["fire_count"]
        if use_rate >= SITUATION_MIN_RATE:
            conn.execute(
                "UPDATE situation_edges SET candidate=0, weight=? WHERE sit_id=?",
                (use_rate, row["sit_id"]),
            )
            promoted += 1
    return promoted
