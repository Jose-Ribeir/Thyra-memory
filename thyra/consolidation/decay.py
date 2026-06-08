"""Decay computation, archive checks, and resurrection."""

from __future__ import annotations

import math
import sqlite3
import time

from thyra.config import (
    ARCHIVE_THRESHOLD,
    RESURRECTION_THRESHOLD,
    RESURRECTION_STRENGTH,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)
from thyra.models.memory import MemoryRecord, compute_base_level


def recompute_and_update(
    conn: sqlite3.Connection,
    memory_ids: list[str],
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> None:
    """Lazy decay: update base_strength for the given (accessed-this-turn) memories.

    We don't rewrite every memory on every turn — only those that were touched.
    The nightly sweep handles the full pass.
    """
    now = int(time.time() * 1000)
    for mem_id in memory_ids:
        row = conn.execute(
            "SELECT base_strength, decay_rate, last_access FROM memories "
            "WHERE id=? AND user_id=? AND agent_id=? AND archived=0",
            (mem_id, user_id, agent_id),
        ).fetchone()
        if row is None:
            continue
        level = compute_base_level(
            row["base_strength"], row["decay_rate"], row["last_access"], now
        )
        conn.execute(
            "UPDATE memories SET base_strength=?, last_access=? WHERE id=? AND user_id=? AND agent_id=?",
            (level, now, mem_id, user_id, agent_id),
        )


def archive_check(
    conn: sqlite3.Connection,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> list[str]:
    """Archive any active memories whose computed base_level < ARCHIVE_THRESHOLD.

    Returns list of memory IDs that were archived.
    """
    now = int(time.time() * 1000)
    rows = conn.execute(
        "SELECT id, base_strength, decay_rate, last_access FROM memories "
        "WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchall()
    archived_ids = []
    for row in rows:
        level = compute_base_level(
            row["base_strength"], row["decay_rate"], row["last_access"], now
        )
        if level < ARCHIVE_THRESHOLD:
            conn.execute(
                "UPDATE memories SET archived=1, archived_at=?, base_strength=? "
                "WHERE id=? AND user_id=? AND agent_id=?",
                (now, level, row["id"], user_id, agent_id),
            )
            archived_ids.append(row["id"])
    return archived_ids


def check_resurrection(
    conn: sqlite3.Connection,
    mem_id: str,
    cue_activation: float,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> bool:
    """Resurrect an archived memory if cue_activation is strong enough.

    Returns True if the memory was resurrected.
    """
    if cue_activation <= RESURRECTION_THRESHOLD:
        return False
    now = int(time.time() * 1000)
    row = conn.execute(
        "SELECT id FROM memories WHERE id=? AND user_id=? AND agent_id=? AND archived=1",
        (mem_id, user_id, agent_id),
    ).fetchone()
    if row is None:
        return False
    conn.execute(
        "UPDATE memories SET archived=0, archived_at=NULL, base_strength=?, last_access=? "
        "WHERE id=? AND user_id=? AND agent_id=?",
        (RESURRECTION_STRENGTH, now, mem_id, user_id, agent_id),
    )
    return True
