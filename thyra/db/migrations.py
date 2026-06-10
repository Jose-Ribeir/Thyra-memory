"""Version-checked incremental migration runner."""

import hashlib
import re
import sqlite3
import time
from thyra.db.schema import SCHEMA_SQL, SEED_CATEGORIES_SQL, SEED_FLAGS_SQL
from thyra.config import THYRA_USER_ID, THYRA_AGENT_ID

_CURRENT_VERSION = 5


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
        version = 3

    if version < 4:
        _v4(conn)
        conn.execute("UPDATE schema_version SET version = 4")
        conn.commit()
        version = 4

    if version < 5:
        _v5(conn)
        conn.execute("UPDATE schema_version SET version = 5")
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


def _v4(conn: sqlite3.Connection) -> None:
    """Scrub corrupted category names with repeated _cluster suffixes.

    _derive_label() appended _cluster to whatever category was already on the
    memory, so repeated clustering runs produced names like
    'constraints_cluster_cluster_cluster_...'.  Strip all but one _cluster suffix.
    Also cleans the categories table cat_id column the same way.
    """

    def _normalise(cat: str) -> str:
        base = cat
        while base.endswith("_cluster"):
            base = base[: -len("_cluster")]
        # Keep one _cluster suffix only if the original had at least one
        if "_cluster" in cat:
            return base + "_cluster"
        return base

    # Fix memories.category
    rows = conn.execute(
        "SELECT DISTINCT category FROM memories WHERE category LIKE '%_cluster_%'"
    ).fetchall()
    for row in rows:
        old_cat = row[0]
        new_cat = _normalise(old_cat)
        if new_cat != old_cat:
            conn.execute(
                "UPDATE memories SET category=? WHERE category=?", (new_cat, old_cat)
            )

    # Fix categories.cat_id
    rows = conn.execute(
        "SELECT cat_id FROM categories WHERE cat_id LIKE '%_cluster_%'"
    ).fetchall()
    for row in rows:
        old_cat = row[0]
        new_cat = _normalise(old_cat)
        if new_cat != old_cat:
            # cat_id is likely the PK — update or just delete the duplicate
            existing = conn.execute(
                "SELECT 1 FROM categories WHERE cat_id=?", (new_cat,)
            ).fetchone()
            if existing:
                conn.execute("DELETE FROM categories WHERE cat_id=?", (old_cat,))
            else:
                conn.execute(
                    "UPDATE categories SET cat_id=? WHERE cat_id=?",
                    (new_cat, old_cat),
                )

    conn.commit()


def _v5(conn: sqlite3.Connection) -> None:
    """Pre-insert last_monitor_ping=0 so thyra_status() never gets a missing-key error."""
    ts = int(time.time() * 1000)
    conn.execute(
        """INSERT OR IGNORE INTO system_flags (flag_key, user_id, agent_id, flag_value, updated_at)
           VALUES ('last_monitor_ping', ?, ?, '0', ?)""",
        (THYRA_USER_ID, THYRA_AGENT_ID, ts),
    )


def _compute_content_hash(content: str) -> str:
    """sha1 of normalized content: lowercase, strip punctuation, collapse whitespace."""
    s = content.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode()).hexdigest()
