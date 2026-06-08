"""Version-checked incremental migration runner."""

import hashlib
import re
import sqlite3
import time
from thyra.db.schema import SCHEMA_SQL, SEED_CATEGORIES_SQL, SEED_FLAGS_SQL
from thyra.config import THYRA_USER_ID, THYRA_AGENT_ID

_CURRENT_VERSION = 3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    version = row[0] if row else 0

    if version < 1:
        _v1(conn)
        if row:
            conn.execute("UPDATE schema_version SET version = 1")
        else:
            conn.execute("INSERT INTO schema_version VALUES (1)")
        conn.commit()
        version = 1

    if version < 2:
        _v2(conn)
        conn.execute("UPDATE schema_version SET version = 2")
        conn.commit()
        version = 2

    if version < 3:
        _v3(conn)
        conn.execute("UPDATE schema_version SET version = 3")
        conn.commit()


def _v1(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    ts = int(time.time() * 1000)
    params = {"u": THYRA_USER_ID, "a": THYRA_AGENT_ID, "ts": ts}
    conn.execute(SEED_CATEGORIES_SQL, params)
    conn.execute(SEED_FLAGS_SQL, params)


def _v2(conn: sqlite3.Connection) -> None:
    """Seed max_memories flag for databases already at v1 that lack it."""
    ts = int(time.time() * 1000)
    conn.execute(
        """INSERT OR IGNORE INTO system_flags (flag_key, user_id, agent_id, flag_value, updated_at)
           VALUES ('max_memories', ?, ?, '0', ?)""",
        (THYRA_USER_ID, THYRA_AGENT_ID, ts),
    )


def _v3(conn: sqlite3.Connection) -> None:
    """Add content_hash column + index; backfill existing rows."""
    # Add column if not already present (safe to run on fresh DBs too)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "content_hash" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mem_content_hash "
        "ON memories(content_hash, user_id, agent_id) WHERE content_hash IS NOT NULL"
    )
    # Backfill existing rows that have no hash yet
    rows = conn.execute(
        "SELECT memory_int_id, content FROM memories WHERE content_hash IS NULL"
    ).fetchall()
    for row in rows:
        h = _compute_content_hash(row[1])
        conn.execute(
            "UPDATE memories SET content_hash=? WHERE memory_int_id=?",
            (h, row[0]),
        )
    conn.commit()


def _compute_content_hash(content: str) -> str:
    """sha1 of normalized content: lowercase, strip punctuation, collapse whitespace."""
    s = content.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode()).hexdigest()
