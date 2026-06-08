"""Stage-1 CRUD tests: insert, retrieve, archive, hard-delete, cue edges."""

import time
import pytest

from thyra.models.memory import (
    create_memory,
    get_memory,
    list_active_memories,
    update_memory_strength,
    archive_memory,
    delete_memory,
    graduate_memory,
    count_active,
    upsert_cue_edge,
    load_cue_edge_map,
    upsert_assoc_edge,
    load_assoc_edge_map,
    get_flag,
    set_flag,
    compute_base_level,
)
from thyra.config import THYRA_USER_ID, THYRA_AGENT_ID, ARCHIVE_THRESHOLD


U = THYRA_USER_ID
A = THYRA_AGENT_ID


class TestCreateAndRetrieve:
    def test_create_returns_id(self, tmp_db):
        mem_id = create_memory(tmp_db, "I prefer dark mode.", "preferences")
        assert mem_id.startswith("m_")

    def test_get_memory(self, tmp_db):
        mem_id = create_memory(tmp_db, "My name is Josef.", "identity")
        rec = get_memory(tmp_db, mem_id)
        assert rec is not None
        assert rec.content == "My name is Josef."
        assert rec.category == "identity"
        assert rec.archived is False
        assert rec.base_strength == 1.0

    def test_get_nonexistent_returns_none(self, tmp_db):
        assert get_memory(tmp_db, "m_doesnotexist") is None

    def test_list_active_memories(self, tmp_db):
        ids = [create_memory(tmp_db, f"Fact {i}") for i in range(5)]
        active = list_active_memories(tmp_db)
        assert len(active) == 5
        for rec in active:
            assert not rec.archived

    def test_tenant_isolation(self, tmp_db):
        create_memory(tmp_db, "Tenant A secret", user_id="user_a", agent_id="agent_x")
        create_memory(tmp_db, "Tenant B secret", user_id="user_b", agent_id="agent_x")
        a_mems = list_active_memories(tmp_db, user_id="user_a", agent_id="agent_x")
        b_mems = list_active_memories(tmp_db, user_id="user_b", agent_id="agent_x")
        assert len(a_mems) == 1
        assert len(b_mems) == 1
        assert a_mems[0].content == "Tenant A secret"

    def test_base_level_computed(self, tmp_db):
        mem_id = create_memory(
            tmp_db, "Fresh memory", base_strength=1.0, decay_rate=0.02
        )
        rec = get_memory(tmp_db, mem_id)
        assert rec.computed_base_level > ARCHIVE_THRESHOLD


class TestStrengthAndArchive:
    def test_update_strength(self, tmp_db):
        mem_id = create_memory(tmp_db, "Reinforce me")
        now = int(time.time() * 1000)
        update_memory_strength(tmp_db, mem_id, 2.5, now)
        rec = get_memory(tmp_db, mem_id)
        assert rec.base_strength == pytest.approx(2.5)

    def test_strength_capped_at_10(self, tmp_db):
        mem_id = create_memory(tmp_db, "Very strong memory")
        now = int(time.time() * 1000)
        update_memory_strength(tmp_db, mem_id, 999.0, now)
        rec = get_memory(tmp_db, mem_id)
        assert rec.base_strength == pytest.approx(10.0)

    def test_archive_memory(self, tmp_db):
        mem_id = create_memory(tmp_db, "Old memory")
        now = int(time.time() * 1000)
        archive_memory(tmp_db, mem_id, now)
        rec = get_memory(tmp_db, mem_id)
        assert rec is not None
        assert rec.archived is True
        assert rec.archived_at is not None

    def test_archived_not_in_active_list(self, tmp_db):
        mem_id = create_memory(tmp_db, "Will be archived")
        now = int(time.time() * 1000)
        archive_memory(tmp_db, mem_id, now)
        active = list_active_memories(tmp_db)
        assert all(r.id != mem_id for r in active)

    def test_delete_memory(self, tmp_db):
        mem_id = create_memory(tmp_db, "Delete me")
        delete_memory(tmp_db, mem_id)
        assert get_memory(tmp_db, mem_id) is None

    def test_count_active(self, tmp_db):
        for i in range(3):
            create_memory(tmp_db, f"Memory {i}")
        assert count_active(tmp_db) == 3


class TestGraduation:
    def test_graduate_clears_probationary(self, tmp_db):
        mem_id = create_memory(
            tmp_db,
            "Auto-formed memory",
            memory_type="semantic",
            base_strength=0.4,
            decay_rate=0.05,
            probationary=True,
        )
        now = int(time.time() * 1000)
        graduate_memory(
            tmp_db, mem_id, new_strength=0.7, category_decay_rate=0.02, now_ms=now
        )
        rec = get_memory(tmp_db, mem_id)
        assert rec.probationary is False
        assert rec.decay_rate == pytest.approx(0.02)
        assert rec.base_strength == pytest.approx(0.7)

    def test_graduate_increments_use_count(self, tmp_db):
        mem_id = create_memory(tmp_db, "Probationary", probationary=True)
        now = int(time.time() * 1000)
        graduate_memory(tmp_db, mem_id, 1.3, 0.02, now)
        rec = get_memory(tmp_db, mem_id)
        assert rec.use_count == 1


class TestCueEdges:
    def test_cue_edge_created_on_save(self, tmp_db):
        mem_id = create_memory(tmp_db, "I prefer Python programming language.")
        edge_map = load_cue_edge_map(tmp_db, U, A)
        # At least one of the content cues should appear
        all_memory_ids = {mid for mids in edge_map.values() for mid, _ in mids}
        assert mem_id in all_memory_ids

    def test_upsert_cue_edge_strengthens(self, tmp_db):
        mem_id = create_memory(tmp_db, "test memory", seed_cues=False)
        upsert_cue_edge(tmp_db, "python", mem_id, U, A, weight=0.30)
        upsert_cue_edge(tmp_db, "python", mem_id, U, A, weight=0.30)
        edge_map = load_cue_edge_map(tmp_db, U, A)
        assert "python" in edge_map
        pairs = {mid: w for mid, w in edge_map["python"]}
        assert pairs[mem_id] > 0.30  # strengthened above seed

    def test_df_incremented(self, tmp_db):
        mem1 = create_memory(tmp_db, "test mem 1", seed_cues=False)
        mem2 = create_memory(tmp_db, "test mem 2", seed_cues=False)
        upsert_cue_edge(tmp_db, "shared", mem1, U, A)
        upsert_cue_edge(tmp_db, "shared", mem2, U, A)
        row = tmp_db.execute(
            "SELECT df FROM cue_nodes WHERE cue_id='shared' AND user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()
        assert row["df"] == 2


class TestAssocEdges:
    def test_upsert_assoc_edge(self, tmp_db):
        mem1 = create_memory(tmp_db, "Memory alpha")
        mem2 = create_memory(tmp_db, "Memory beta")
        upsert_assoc_edge(tmp_db, mem1, mem2, U, A, delta_weight=0.1)
        edge_map = load_assoc_edge_map(tmp_db, U, A)
        assert mem2 in {mid for mid, _ in edge_map.get(mem1, [])}
        assert mem1 in {mid for mid, _ in edge_map.get(mem2, [])}

    def test_assoc_edge_accumulates(self, tmp_db):
        mem1 = create_memory(tmp_db, "Alpha")
        mem2 = create_memory(tmp_db, "Beta")
        for _ in range(3):
            upsert_assoc_edge(tmp_db, mem1, mem2, U, A, delta_weight=0.1)
        edge_map = load_assoc_edge_map(tmp_db, U, A)
        weight = dict(edge_map.get(mem1, []))[mem2]
        assert weight == pytest.approx(0.3)


class TestSystemFlags:
    def test_flag_default_true(self, tmp_db):
        val = get_flag(tmp_db, "system_enabled")
        assert val == "true"

    def test_flag_set_and_get(self, tmp_db):
        set_flag(tmp_db, "system_enabled", "false")
        assert get_flag(tmp_db, "system_enabled") == "false"

    def test_flag_missing_returns_default(self, tmp_db):
        val = get_flag(tmp_db, "nonexistent_flag", default="missing")
        assert val == "missing"


class TestDecayFormula:
    def test_fresh_memory_near_full_strength(self):
        level = compute_base_level(
            1.0, 0.02, int(time.time() * 1000), int(time.time() * 1000)
        )
        assert level == pytest.approx(1.0, abs=0.01)

    def test_old_memory_decays(self):
        now = int(time.time() * 1000)
        old = now - 35 * 86_400_000  # 35 days ago
        level = compute_base_level(1.0, 0.02, old, now)
        assert 0.3 < level < 0.6  # roughly half-life at 35 days

    def test_fast_decay_archives_quickly(self):
        now = int(time.time() * 1000)
        two_weeks_ago = now - 14 * 86_400_000
        level = compute_base_level(0.4, 0.15, two_weeks_ago, now)
        assert level < 0.05  # should be below archive threshold
