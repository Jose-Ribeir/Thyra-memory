"""Cue edge dynamics and Hebbian association formation."""

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from itertools import combinations

from thyra.config import (
    CUE_PROMOTE_THRESHOLD,
    CUE_PRUNE_MAX_RATE,
    CUE_PRUNE_MIN_FIRES,
    HEBBIAN_MIN_CO_USE,
    HEBBIAN_WEIGHT_DELTA,
    ASSOC_WEIGHT_CAP,
    HUB_CUE_FRACTION,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)
from thyra.models.delta import DeltaEvent
from thyra.models.memory import upsert_assoc_edge


def _hub_cues(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> set[str]:
    """Return cue_ids whose df exceeds HUB_CUE_FRACTION * M (non-discriminative)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchone()
    M = row[0] if row else 0
    if M < 2:
        return set()
    threshold = int(M * HUB_CUE_FRACTION)
    rows = conn.execute(
        "SELECT cue_id FROM cue_nodes WHERE user_id=? AND agent_id=? AND df > ?",
        (user_id, agent_id, threshold),
    ).fetchall()
    return {r["cue_id"] for r in rows}


def update_cue_edges(
    conn: sqlite3.Connection,
    delta: DeltaEvent,
) -> None:
    """Strengthen (cue, memory) pairs that were actually used this turn.

    Also fires the cue's fire_count for ALL cues that were present (whether
    the memory was used or not), and increments use_count for used pairs.
    Hub cues (appearing in >HUB_CUE_FRACTION of memories) are excluded from
    weight updates — their fire_count still ticks so weak-rate pruning can
    eventually remove them.
    """
    user_id = delta.user_id
    agent_id = delta.agent_id
    used_set = set(delta.memories_declared) & set(delta.memories_served)
    cues = delta.cues_fired

    if not cues:
        return

    now = int(time.time() * 1000)
    hub = _hub_cues(conn, user_id, agent_id)

    # Increment fire_count for all (cue, memory) pairs where cue was fired
    if cues:
        placeholders = ",".join("?" * len(cues))
        conn.execute(
            f"""UPDATE cue_edges SET fire_count = fire_count + 1
                WHERE cue_id IN ({placeholders}) AND user_id=? AND agent_id=?""",
            (*cues, user_id, agent_id),
        )

    # Strengthen edges for (cue, memory) pairs that were used (skip hub cues)
    for cue in cues:
        if cue in hub:
            continue
        for mem_id in used_set:
            # Check if this cue edge exists
            row = conn.execute(
                "SELECT weight, candidate FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
                (cue, mem_id, user_id, agent_id),
            ).fetchone()
            if row:
                new_w = min(1.0, row["weight"] + 0.05)
                conn.execute(
                    """UPDATE cue_edges SET weight=?, use_count=use_count+1
                       WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?""",
                    (new_w, cue, mem_id, user_id, agent_id),
                )
                # Promote candidate edge if weight crosses threshold
                if row["candidate"] and new_w >= CUE_PROMOTE_THRESHOLD:
                    conn.execute(
                        """UPDATE cue_edges SET candidate=0
                           WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?""",
                        (cue, mem_id, user_id, agent_id),
                    )


def prune_weak_cue_edges(
    conn: sqlite3.Connection,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> int:
    """Remove cue edges with high fire_count but very low use rate.

    Also decrements cue_nodes.df for each pruned edge so that IDF scores
    remain accurate rather than inflating toward over-common over time.
    Returns number of pruned edges.
    """
    # Gather df decrements before deleting
    df_decrements: dict[str, int] = {}
    rows = conn.execute(
        """SELECT cue_id FROM cue_edges
           WHERE user_id=? AND agent_id=?
             AND fire_count >= ?
             AND CAST(use_count AS REAL) / fire_count < ?""",
        (user_id, agent_id, CUE_PRUNE_MIN_FIRES, CUE_PRUNE_MAX_RATE),
    ).fetchall()
    for row in rows:
        df_decrements[row["cue_id"]] = df_decrements.get(row["cue_id"], 0) + 1

    result = conn.execute(
        """DELETE FROM cue_edges
           WHERE user_id=? AND agent_id=?
             AND fire_count >= ?
             AND CAST(use_count AS REAL) / fire_count < ?""",
        (user_id, agent_id, CUE_PRUNE_MIN_FIRES, CUE_PRUNE_MAX_RATE),
    )
    pruned = result.rowcount

    for cue_id, decr in df_decrements.items():
        conn.execute(
            "UPDATE cue_nodes SET df = CASE WHEN df < ? THEN 0 ELSE df - ? END"
            " WHERE cue_id=? AND user_id=? AND agent_id=?",
            (decr, decr, cue_id, user_id, agent_id),
        )

    return pruned


def hebbian_association(
    conn: sqlite3.Connection,
    window: list[DeltaEvent],
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> int:
    """Form/strengthen memory↔memory association edges from co-use in the window.

    Memories that appear together in ≥ HEBBIAN_MIN_CO_USE turns get linked.
    Returns number of edge upserts.
    """
    pair_counts: Counter = Counter()
    for delta in window:
        # Count pairs from the used set (not just served)
        used = set(delta.memories_declared) & set(delta.memories_served)
        for pair in combinations(sorted(used), 2):
            pair_counts[pair] += 1

    upserted = 0
    for (mem_a, mem_b), count in pair_counts.items():
        if count >= HEBBIAN_MIN_CO_USE:
            upsert_assoc_edge(
                conn,
                mem_a,
                mem_b,
                user_id,
                agent_id,
                delta_weight=HEBBIAN_WEIGHT_DELTA * count,
            )
            upserted += 1
    return upserted
