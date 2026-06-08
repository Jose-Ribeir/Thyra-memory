"""Memory dataclasses and all CRUD operations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from thyra.config import (
    ARCHIVE_THRESHOLD,
    BASE_STRENGTH_EXPLICIT,
    CONTENT_SEED_WEIGHT,
    DECAY_EXPLICIT,
    STRENGTH_CAP,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)


# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class MemoryRecord:
    id: str
    user_id: str
    agent_id: str
    content: str
    locked: bool
    category: str
    memory_type: str  # explicit | semantic | episodic
    base_strength: float
    decay_rate: float
    last_access: int  # ms since epoch
    created_at: int
    probationary: bool
    archived: bool
    archived_at: Optional[int]
    use_count: int
    memory_int_id: int = 0
    computed_base_level: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row, now_ms: int | None = None) -> "MemoryRecord":
        rec = cls(
            id=row["id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            content=row["content"],
            locked=bool(row["locked"]),
            category=row["category"],
            memory_type=row["memory_type"],
            base_strength=row["base_strength"],
            decay_rate=row["decay_rate"],
            last_access=row["last_access"],
            created_at=row["created_at"],
            probationary=bool(row["probationary"]),
            archived=bool(row["archived"]),
            archived_at=row["archived_at"],
            use_count=row["use_count"],
            memory_int_id=row["memory_int_id"],
        )
        if now_ms is not None:
            rec.computed_base_level = compute_base_level(
                rec.base_strength, rec.decay_rate, rec.last_access, now_ms
            )
        return rec


@dataclass
class CueEdge:
    cue_id: str
    memory_id: str
    user_id: str
    agent_id: str
    weight: float = CONTENT_SEED_WEIGHT
    fire_count: int = 0
    use_count: int = 0
    candidate: bool = False
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class AssocEdge:
    memory_a: str
    memory_b: str
    user_id: str
    agent_id: str
    weight: float = 0.0
    co_use: int = 0
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class SituationEdge:
    sit_id: str
    memory_id: str
    user_id: str
    agent_id: str
    cue_set: list[str]
    weight: float = 0.0
    fire_count: int = 0
    use_count: int = 0
    candidate: bool = True
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))


# ── Content hash ──────────────────────────────────────────────────────────────


def compute_content_hash(content: str) -> str:
    """sha1 of normalized content: lowercase, strip punctuation, collapse whitespace."""
    s = content.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode()).hexdigest()


# ── Decay formula ─────────────────────────────────────────────────────────────


def compute_base_level(
    base_strength: float,
    decay_rate: float,
    last_access_ms: int,
    now_ms: int,
) -> float:
    days = max(0.0, (now_ms - last_access_ms) / 86_400_000)
    return base_strength * math.exp(-decay_rate * days)


# ── Memory CRUD ───────────────────────────────────────────────────────────────


def create_memory(
    conn: sqlite3.Connection,
    content: str,
    category: str = "context",
    memory_type: str = "explicit",
    base_strength: float = BASE_STRENGTH_EXPLICIT,
    decay_rate: float = DECAY_EXPLICIT,
    probationary: bool = False,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
    locked: bool = False,
    seed_cues: bool = True,
) -> str:
    mem_id = "m_" + uuid.uuid4().hex[:16]
    now = int(time.time() * 1000)
    chash = compute_content_hash(content) if not locked else None
    with conn:
        conn.execute(
            """INSERT INTO memories
               (id, user_id, agent_id, content, locked, category, memory_type,
                base_strength, decay_rate, last_access, created_at, probationary,
                archived, archived_at, use_count, content_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,NULL,0,?)""",
            (
                mem_id,
                user_id,
                agent_id,
                content,
                int(locked),
                category,
                memory_type,
                base_strength,
                decay_rate,
                now,
                now,
                int(probationary),
                chash,
            ),
        )
    if seed_cues and not locked:
        _seed_cues_from_content(conn, mem_id, content, user_id, agent_id, now)
    return mem_id


def get_memory(
    conn: sqlite3.Connection,
    mem_id: str,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> MemoryRecord | None:
    row = conn.execute(
        "SELECT * FROM memories WHERE id=? AND user_id=? AND agent_id=?",
        (mem_id, user_id, agent_id),
    ).fetchone()
    if row is None:
        return None
    return MemoryRecord.from_row(row, int(time.time() * 1000))


def list_active_memories(
    conn: sqlite3.Connection,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> list[MemoryRecord]:
    now = int(time.time() * 1000)
    rows = conn.execute(
        "SELECT * FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchall()
    return [MemoryRecord.from_row(r, now) for r in rows]


def update_memory_strength(
    conn: sqlite3.Connection,
    mem_id: str,
    new_strength: float,
    now_ms: int,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> None:
    capped = min(STRENGTH_CAP, new_strength)
    with conn:
        conn.execute(
            """UPDATE memories SET base_strength=?, last_access=?
               WHERE id=? AND user_id=? AND agent_id=?""",
            (capped, now_ms, mem_id, user_id, agent_id),
        )


def archive_memory(
    conn: sqlite3.Connection,
    mem_id: str,
    now_ms: int,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> None:
    with conn:
        conn.execute(
            """UPDATE memories SET archived=1, archived_at=?
               WHERE id=? AND user_id=? AND agent_id=?""",
            (now_ms, mem_id, user_id, agent_id),
        )


def delete_memory(
    conn: sqlite3.Connection,
    mem_id: str,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> None:
    with conn:
        conn.execute(
            "DELETE FROM memories WHERE id=? AND user_id=? AND agent_id=?",
            (mem_id, user_id, agent_id),
        )


def graduate_memory(
    conn: sqlite3.Connection,
    mem_id: str,
    new_strength: float,
    category_decay_rate: float,
    now_ms: int,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> None:
    capped = min(STRENGTH_CAP, new_strength)
    with conn:
        conn.execute(
            """UPDATE memories
               SET base_strength=?, decay_rate=?, probationary=0, last_access=?,
                   use_count = use_count + 1
               WHERE id=? AND user_id=? AND agent_id=?""",
            (capped, category_decay_rate, now_ms, mem_id, user_id, agent_id),
        )


def count_active(
    conn: sqlite3.Connection,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchone()
    return row[0] if row else 0


# ── Cue edge CRUD ──────────────────────────────────────────────────────────────


def upsert_cue_edge(
    conn: sqlite3.Connection,
    cue_id: str,
    mem_id: str,
    user_id: str,
    agent_id: str,
    weight: float = CONTENT_SEED_WEIGHT,
    candidate: bool = False,
) -> None:
    now = int(time.time() * 1000)
    with conn:
        existing = conn.execute(
            "SELECT weight FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
            (cue_id, mem_id, user_id, agent_id),
        ).fetchone()
        if existing:
            new_w = min(1.0, existing["weight"] + weight * 0.5)
            conn.execute(
                "UPDATE cue_edges SET weight=? WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
                (new_w, cue_id, mem_id, user_id, agent_id),
            )
        else:
            conn.execute(
                """INSERT INTO cue_edges
                   (cue_id, memory_id, user_id, agent_id, weight, fire_count, use_count, candidate, created_at)
                   VALUES (?,?,?,?,?,0,0,?,?)""",
                (cue_id, mem_id, user_id, agent_id, weight, int(candidate), now),
            )
            _increment_df(conn, cue_id, user_id, agent_id)


def get_cue_edges_for_memory(
    conn: sqlite3.Connection,
    mem_id: str,
    user_id: str,
    agent_id: str,
) -> list[CueEdge]:
    rows = conn.execute(
        "SELECT * FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (mem_id, user_id, agent_id),
    ).fetchall()
    return [CueEdge(**{k: row[k] for k in row.keys()}) for row in rows]


def load_cue_edge_map(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> dict[str, list[tuple[str, float]]]:
    """Returns {cue_id: [(memory_id, weight), ...]} for all non-candidate edges."""
    rows = conn.execute(
        """SELECT cue_id, memory_id, weight FROM cue_edges
           WHERE user_id=? AND agent_id=? AND candidate=0""",
        (user_id, agent_id),
    ).fetchall()
    result: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        result.setdefault(row["cue_id"], []).append((row["memory_id"], row["weight"]))
    return result


def load_assoc_edge_map(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> dict[str, list[tuple[str, float]]]:
    """Returns {memory_id: [(neighbor_id, weight), ...]}."""
    rows = conn.execute(
        "SELECT memory_a, memory_b, weight FROM association_edges WHERE user_id=? AND agent_id=?",
        (user_id, agent_id),
    ).fetchall()
    result: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        result.setdefault(row["memory_a"], []).append((row["memory_b"], row["weight"]))
        result.setdefault(row["memory_b"], []).append((row["memory_a"], row["weight"]))
    return result


def load_situation_edges(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> list[SituationEdge]:
    rows = conn.execute(
        "SELECT * FROM situation_edges WHERE user_id=? AND agent_id=? AND candidate=0",
        (user_id, agent_id),
    ).fetchall()
    return [
        SituationEdge(
            sit_id=r["sit_id"],
            memory_id=r["memory_id"],
            user_id=r["user_id"],
            agent_id=r["agent_id"],
            cue_set=json.loads(r["cue_set"]),
            weight=r["weight"],
            fire_count=r["fire_count"],
            use_count=r["use_count"],
            candidate=bool(r["candidate"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


def upsert_assoc_edge(
    conn: sqlite3.Connection,
    mem_a: str,
    mem_b: str,
    user_id: str,
    agent_id: str,
    delta_weight: float = 0.1,
) -> None:
    a, b = (mem_a, mem_b) if mem_a < mem_b else (mem_b, mem_a)
    now = int(time.time() * 1000)
    with conn:
        existing = conn.execute(
            "SELECT weight, co_use FROM association_edges WHERE memory_a=? AND memory_b=? AND user_id=? AND agent_id=?",
            (a, b, user_id, agent_id),
        ).fetchone()
        if existing:
            new_w = min(1.0, existing["weight"] + delta_weight)
            conn.execute(
                """UPDATE association_edges SET weight=?, co_use=co_use+1
                   WHERE memory_a=? AND memory_b=? AND user_id=? AND agent_id=?""",
                (new_w, a, b, user_id, agent_id),
            )
        else:
            conn.execute(
                """INSERT INTO association_edges
                   (memory_a, memory_b, user_id, agent_id, weight, co_use, created_at)
                   VALUES (?,?,?,?,?,1,?)""",
                (a, b, user_id, agent_id, delta_weight, now),
            )


# ── System flags ──────────────────────────────────────────────────────────────


def get_flag(
    conn: sqlite3.Connection,
    flag_key: str,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
    default: str = "true",
) -> str:
    row = conn.execute(
        "SELECT flag_value FROM system_flags WHERE flag_key=? AND user_id=? AND agent_id=?",
        (flag_key, user_id, agent_id),
    ).fetchone()
    return row["flag_value"] if row else default


def set_flag(
    conn: sqlite3.Connection,
    flag_key: str,
    value: str,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> None:
    now = int(time.time() * 1000)
    with conn:
        conn.execute(
            """INSERT INTO system_flags (flag_key, user_id, agent_id, flag_value, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(flag_key, user_id, agent_id) DO UPDATE SET flag_value=excluded.flag_value, updated_at=excluded.updated_at""",
            (flag_key, user_id, agent_id, value, now),
        )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _seed_cues_from_content(
    conn: sqlite3.Connection,
    mem_id: str,
    content: str,
    user_id: str,
    agent_id: str,
    now: int,
) -> None:
    from thyra.recall.cue_extractor import extract_raw_cues

    cues = extract_raw_cues(content, max_cues=8)
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


def _increment_df(
    conn: sqlite3.Connection,
    cue_id: str,
    user_id: str,
    agent_id: str,
) -> None:
    conn.execute(
        """INSERT INTO cue_nodes (cue_id, user_id, agent_id, df)
           VALUES (?,?,?,1)
           ON CONFLICT(cue_id, user_id, agent_id) DO UPDATE SET df=df+1""",
        (cue_id, user_id, agent_id),
    )
