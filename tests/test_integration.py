"""Stage-8 integration tests — full lifecycle from formation through decay."""

from __future__ import annotations

import json
import math
import time
import pathlib

import pytest

from thyra.config import (
    THYRA_USER_ID as U,
    THYRA_AGENT_ID as A,
    ARCHIVE_THRESHOLD,
    HARD_DELETE_DAYS,
    DECAY_EPISODIC,
    BASE_STRENGTH_AUTOMATIC,
)
from thyra.models.delta import DeltaEvent
from thyra.models.memory import (
    create_memory,
    get_memory,
    list_active_memories,
    get_flag,
    set_flag,
)
from thyra.recall.intent import recall_pipeline
from thyra.db.connection import DBConnection
from thyra.recall.cache import HOT_CACHE


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_delta(
    user_text="", asst_text="", served=None, declared=None, cues=None
) -> DeltaEvent:
    return DeltaEvent(
        session_id="int-test",
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
    """Run the full consolidation pipeline synchronously (bypasses file queue).

    Pass a shared `window` list across calls to accumulate a Hebbian window.
    """
    from thyra.consolidation.decay import recompute_and_update, archive_check
    from thyra.consolidation.reinforcement import apply_reinforcement
    from thyra.consolidation.edges import update_cue_edges, hebbian_association
    from thyra.consolidation.situation import crystallize_situations
    from thyra.formation.pipeline import run_formation_pipeline

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


# ── Full lifecycle ─────────────────────────────────────────────────────────────


class TestFullLifecycle:
    def test_explicit_save_recalled(self, tmp_db):
        """Explicitly saved memory should be recalled when prompt matches cues."""
        mem_id = create_memory(
            tmp_db,
            "I prefer Python for data science projects",
            "preferences",
        )
        from thyra.models.memory import upsert_cue_edge

        upsert_cue_edge(tmp_db, "python", mem_id, U, A, weight=0.8)
        upsert_cue_edge(tmp_db, "data", mem_id, U, A, weight=0.5)
        HOT_CACHE.clear()

        xml, served = recall_pipeline(
            tmp_db,
            U,
            A,
            "What Python libraries do you recommend for data science?",
            "s1",
            "t1",
        )
        assert mem_id in served
        assert "python" in xml.lower() or mem_id in xml

    def test_auto_formation_creates_memory(self, tmp_db):
        """A self-disclosure in user text should create a probationary memory."""
        delta = make_delta(user_text="I always prefer tabs over spaces in Python code")
        result = apply_delta_sync(tmp_db, delta)
        mems = list_active_memories(tmp_db, U, A)
        assert len(mems) >= 1
        created = [m for m in mems if m.probationary]
        assert len(created) >= 1

    def test_reinforcement_grows_strength(self, tmp_db):
        """Explicitly declared used memories should grow in strength."""
        mem_id = create_memory(
            tmp_db, "I prefer dark mode", "preferences", base_strength=0.5
        )
        initial = get_memory(tmp_db, mem_id).base_strength

        delta = make_delta(
            user_text="Please use dark mode",
            served=[mem_id],
            declared=[mem_id],
        )
        apply_delta_sync(tmp_db, delta)

        final = get_memory(tmp_db, mem_id).base_strength
        assert final > initial, f"Expected strength to grow from {initial}, got {final}"

    def test_graduation_changes_decay_rate(self, tmp_db):
        """First real use of a probationary memory should graduate it."""
        mem_id = create_memory(
            tmp_db,
            "I prefer dark mode",
            "preferences",
            base_strength=BASE_STRENGTH_AUTOMATIC,
            decay_rate=DECAY_EPISODIC,
            probationary=True,
        )
        assert get_memory(tmp_db, mem_id).probationary is True

        delta = make_delta(served=[mem_id], declared=[mem_id])
        apply_delta_sync(tmp_db, delta)

        graduated = get_memory(tmp_db, mem_id)
        assert graduated.probationary is False
        assert graduated.decay_rate < DECAY_EPISODIC  # graduated to slower decay

    def test_decay_archives_weak_memory(self, tmp_db):
        """A memory at ARCHIVE_THRESHOLD should be archived by archive_check."""
        mem_id = create_memory(tmp_db, "Old irrelevant note", "context")
        with tmp_db:
            tmp_db.execute(
                "UPDATE memories SET base_strength=? WHERE id=?",
                (ARCHIVE_THRESHOLD - 0.001, mem_id),
            )
        HOT_CACHE.clear()

        from thyra.consolidation.decay import archive_check

        archive_check(tmp_db, U, A)
        mem = get_memory(tmp_db, mem_id)
        assert mem.archived is True

    def test_archived_memory_not_in_normal_recall(self, tmp_db):
        """Archived memories should not appear in normal (non-intent) recall."""
        mem_id = create_memory(tmp_db, "Old Python project note", "context")
        from thyra.models.memory import upsert_cue_edge

        upsert_cue_edge(tmp_db, "python", mem_id, U, A, weight=0.8)
        with tmp_db:
            tmp_db.execute(
                "UPDATE memories SET archived=1, archived_at=? WHERE id=?",
                (int(time.time() * 1000), mem_id),
            )
        HOT_CACHE.clear()

        _, served = recall_pipeline(tmp_db, U, A, "python", "s1", "t1")
        assert mem_id not in served

    def test_archived_memory_returned_on_recall_intent(self, tmp_db):
        """Archived memories with cue matches should appear on recall-intent prompts."""
        mem_id = create_memory(tmp_db, "Python data science work", "context")
        from thyra.models.memory import upsert_cue_edge

        # Use morphology-normalized cue IDs to match what extract_cues produces
        # "python" → "python", "data" → "datum" (Porter stemmer), "science" → "scienc"
        upsert_cue_edge(tmp_db, "python", mem_id, U, A, weight=0.9)
        upsert_cue_edge(tmp_db, "datum", mem_id, U, A, weight=0.9)
        upsert_cue_edge(tmp_db, "scienc", mem_id, U, A, weight=0.9)
        with tmp_db:
            tmp_db.execute(
                "UPDATE memories SET archived=1, archived_at=?, base_strength=0.5 WHERE id=?",
                (int(time.time() * 1000), mem_id),
            )
        HOT_CACHE.clear()

        _, served = recall_pipeline(
            tmp_db,
            U,
            A,
            "Do you remember when we discussed Python data science?",
            "s1",
            "t1",
        )
        assert mem_id in served

    def test_system_disabled_returns_empty(self, tmp_db):
        """Disabling the system master switch should suppress all recall."""
        mem_id = create_memory(tmp_db, "I prefer Python")
        set_flag(tmp_db, "system_enabled", "false", U, A)
        tmp_db.commit()
        HOT_CACHE.clear()

        xml, served = recall_pipeline(tmp_db, U, A, "python", "s1", "t1")
        assert xml == ""
        assert served == []


class TestNightlySweep:
    def test_nightly_archives_weak(self, tmp_db):
        """Nightly sweep should archive memories below the threshold."""
        mem_id = create_memory(tmp_db, "Low strength memory", "context")
        with tmp_db:
            tmp_db.execute(
                "UPDATE memories SET base_strength=0.001 WHERE id=?", (mem_id,)
            )
        from thyra.consolidation.nightly import run_nightly_sweep

        result = run_nightly_sweep(tmp_db, U, A)
        mem = get_memory(tmp_db, mem_id)
        assert mem.archived is True
        assert result["archived"] >= 1

    def test_nightly_hard_deletes_old_archived(self, tmp_db):
        """Nightly sweep should hard-delete archived memories older than HARD_DELETE_DAYS."""
        mem_id = create_memory(tmp_db, "Very old archived memory", "context")
        old_ts = int(time.time() * 1000) - (HARD_DELETE_DAYS + 1) * 86_400_000
        with tmp_db:
            tmp_db.execute(
                "UPDATE memories SET archived=1, archived_at=? WHERE id=?",
                (old_ts, mem_id),
            )
        from thyra.consolidation.nightly import run_nightly_sweep

        run_nightly_sweep(tmp_db, U, A)
        mem = get_memory(tmp_db, mem_id)
        assert mem is None  # hard-deleted

    def test_nightly_prunes_turn_log(self, tmp_db):
        """Nightly sweep should remove turn log entries older than retention period."""
        from thyra.config import TURN_LOG_RETENTION_DAYS

        old_ts = int(time.time() * 1000) - (TURN_LOG_RETENTION_DAYS + 1) * 86_400_000
        with tmp_db:
            tmp_db.execute(
                """INSERT INTO turn_log (turn_id, session_id, user_id, agent_id,
                   memories_served, memories_used, cues_fired, created_at)
                   VALUES ('old_turn', 's1', ?, ?, '[]', '[]', '[]', ?)""",
                (U, A, old_ts),
            )
        from thyra.consolidation.nightly import run_nightly_sweep

        run_nightly_sweep(tmp_db, U, A)
        row = tmp_db.execute(
            "SELECT 1 FROM turn_log WHERE turn_id='old_turn'"
        ).fetchone()
        assert row is None

    def test_nightly_returns_summary_dict(self, tmp_db):
        """run_nightly_sweep should return a summary dict with expected keys."""
        from thyra.consolidation.nightly import run_nightly_sweep

        summary = run_nightly_sweep(tmp_db, U, A)
        for key in [
            "decayed",
            "archived",
            "hard_deleted",
            "cue_edges_pruned",
            "assoc_edges_pruned",
            "turn_log_pruned",
        ]:
            assert key in summary, f"Missing key: {key}"


class TestEdgeToEdge:
    def test_multiple_turns_hebbian_edge(self, tmp_db):
        """Co-used memories across 3 turns should gain an association edge."""
        mem_a = create_memory(
            tmp_db, "Python preference", "preferences", base_strength=0.8
        )
        mem_b = create_memory(
            tmp_db, "Dark mode preference", "preferences", base_strength=0.8
        )

        window: list = []
        for i in range(3):
            delta = make_delta(
                user_text=f"Turn {i}",
                served=[mem_a, mem_b],
                declared=[mem_a, mem_b],
            )
            apply_delta_sync(tmp_db, delta, window=window)

        a, b = (mem_a, mem_b) if mem_a < mem_b else (mem_b, mem_a)
        edge = tmp_db.execute(
            "SELECT weight, co_use FROM association_edges WHERE memory_a=? AND memory_b=? AND user_id=? AND agent_id=?",
            (a, b, U, A),
        ).fetchone()
        assert edge is not None
        assert edge["co_use"] >= 1  # edge upserted at least once
        assert edge["weight"] > 0

    def test_anti_spoofing_blocks_undeclared(self, tmp_db):
        """A memory declared as used but not served should not be reinforced."""
        mem_id = create_memory(tmp_db, "Unreserved memory", base_strength=0.5)
        initial = get_memory(tmp_db, mem_id).base_strength

        # Declare mem_id used but do NOT include it in served
        delta = make_delta(
            user_text="Hello",
            served=[],  # not served
            declared=[mem_id],  # declared but not served
        )
        apply_delta_sync(tmp_db, delta)

        final = get_memory(tmp_db, mem_id).base_strength
        assert final == pytest.approx(initial)  # strength unchanged

    def test_pipeline_never_raises(self, tmp_db, monkeypatch):
        """recall_pipeline must return ('', []) rather than raising on error."""
        monkeypatch.setattr(
            "thyra.recall.intent.list_active_memories",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("injected error")),
        )
        HOT_CACHE.clear()
        xml, served = recall_pipeline(tmp_db, U, A, "anything", "s1", "t1")
        assert xml == ""
        assert served == []
