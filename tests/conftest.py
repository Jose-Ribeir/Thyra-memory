"""Shared pytest fixtures and helpers."""

import json
import os
import sqlite3
import tempfile
import time
import pytest

from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A
from thyra.db.connection import DBConnection
from thyra.models.delta import DeltaEvent


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Provide a fresh temporary database for each test."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("THYRA_DB_PATH", db_file)

    # Patch the module-level default so DBConnection picks up the env var
    import thyra.config as cfg

    monkeypatch.setattr(cfg, "THYRA_DB_PATH", db_file)

    # Reset thread-local connection so it rebuilds with the new path
    DBConnection._local = type(DBConnection._local)()

    # Clear module-level hot cache so tests don't see each other's snapshots
    from thyra.recall.cache import HOT_CACHE

    HOT_CACHE.clear()

    # Clear drain.py's per-process Hebbian turn windows so stale memory IDs
    # from a previous test don't leak into crystallize_situations calls for the
    # fresh test DB (stale IDs trigger FK constraint failures in situation_edges).
    try:
        import thyra.consolidation.drain as _drain

        _drain._turn_windows.clear()
    except ImportError:
        pass

    conn = DBConnection.get(db_file)
    yield conn

    DBConnection.close()
    DBConnection._local = type(DBConnection._local)()
    HOT_CACHE.clear()


# ── Shared helpers (import these in any test file that needs them) ─────────────


def make_delta(
    user_text="", asst_text="", served=None, declared=None, cues=None
) -> DeltaEvent:
    return DeltaEvent(
        session_id="test",
        turn_id=f"t{int(time.time() * 1000)}_{id(user_text)}",
        user_id=U,
        agent_id=A,
        timestamp=int(time.time() * 1000),
        raw_user_text=user_text,
        raw_assistant_text=asst_text,
        memories_served=served or [],
        memories_declared=declared or [],
        cues_fired=cues or [],
    )


def apply_delta_sync(conn, delta: DeltaEvent, window: list | None = None) -> dict:
    """Run the full consolidation pipeline synchronously (bypasses the file queue)."""
    from thyra.consolidation.decay import recompute_and_update, archive_check
    from thyra.consolidation.reinforcement import apply_reinforcement
    from thyra.consolidation.edges import update_cue_edges, hebbian_association
    from thyra.consolidation.situation import crystallize_situations
    from thyra.formation.pipeline import run_formation_pipeline
    from thyra.recall.cache import HOT_CACHE

    actions = run_formation_pipeline(conn, delta)

    recompute_and_update(conn, delta.memories_served, U, A)
    apply_reinforcement(conn, delta)
    update_cue_edges(conn, delta)

    if window is None:
        window = [delta]
    else:
        window.append(delta)
    hebbian_association(conn, list(window), U, A)
    crystallize_situations(conn, list(window), U, A)
    archive_check(conn, U, A)

    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_turns (turn_id, processed_at) VALUES (?,?)",
            (delta.turn_id, int(time.time() * 1000)),
        )
        conn.execute(
            """INSERT OR IGNORE INTO turn_log
               (turn_id, session_id, user_id, agent_id, memories_served, memories_used, cues_fired, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                delta.turn_id,
                delta.session_id,
                U,
                A,
                json.dumps(delta.memories_served),
                json.dumps(delta.memories_declared),
                json.dumps(delta.cues_fired),
                delta.timestamp,
            ),
        )

    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    return {a: mid for a, mid in actions}
