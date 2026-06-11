"""Stage-4 consolidation tests: decay, reinforcement, edges, worker."""

import json
import pathlib
import time
import pytest

from thyra.models.memory import (
    create_memory,
    get_memory,
    update_memory_strength,
    upsert_cue_edge,
    load_cue_edge_map,
    upsert_assoc_edge,
)
from thyra.models.delta import DeltaEvent
from thyra.consolidation.decay import (
    recompute_and_update,
    archive_check,
    check_resurrection,
)
from thyra.consolidation.reinforcement import apply_reinforcement
from thyra.consolidation.edges import (
    update_cue_edges,
    prune_weak_cue_edges,
    hebbian_association,
)
from thyra.config import (
    THYRA_USER_ID as U,
    THYRA_AGENT_ID as A,
    ARCHIVE_THRESHOLD,
    REINFORCE_BASE,
    SURFACED_BOOST,
    CATEGORY_MULTIPLIERS,
    DECLARED_UNCORROBORATED_MULT,
    DECLARED_UNCORROBORATED_BEHAVIORAL_MULT,
)


def make_delta(
    served=None, declared=None, cues=None, user_text="", asst_text="", turn_id=None
) -> DeltaEvent:
    return DeltaEvent(
        session_id="test",
        turn_id=turn_id or f"t{int(time.time() * 1000)}",
        user_id=U,
        agent_id=A,
        timestamp=int(time.time() * 1000),
        memories_served=served or [],
        memories_declared=declared or [],
        cues_fired=cues or [],
        raw_user_text=user_text,
        raw_assistant_text=asst_text,
    )


class TestDecay:
    def test_lazy_decay_updates_strength(self, tmp_db):
        mem_id = create_memory(tmp_db, "Old memory", base_strength=1.0, decay_rate=0.5)
        # Backdate last_access by 10 days
        past = int(time.time() * 1000) - 10 * 86_400_000
        tmp_db.execute("UPDATE memories SET last_access=? WHERE id=?", (past, mem_id))
        tmp_db.commit()
        recompute_and_update(tmp_db, [mem_id], U, A)
        rec = get_memory(tmp_db, mem_id)
        assert rec.base_strength < 1.0  # decayed

    def test_archive_check_archives_weak_memory(self, tmp_db):
        mem_id = create_memory(
            tmp_db, "Very old memory", base_strength=0.01, decay_rate=0.5
        )
        archived = archive_check(tmp_db, U, A)
        assert mem_id in archived
        rec = get_memory(tmp_db, mem_id)
        assert rec.archived is True

    def test_archive_check_leaves_strong_memory(self, tmp_db):
        mem_id = create_memory(
            tmp_db, "Fresh memory", base_strength=1.0, decay_rate=0.02
        )
        archived = archive_check(tmp_db, U, A)
        assert mem_id not in archived
        rec = get_memory(tmp_db, mem_id)
        assert rec.archived is False

    def test_resurrection_requires_strong_activation(self, tmp_db):
        mem_id = create_memory(tmp_db, "Old memory")
        now = int(time.time() * 1000)
        tmp_db.execute(
            "UPDATE memories SET archived=1, archived_at=? WHERE id=?", (now, mem_id)
        )
        tmp_db.commit()

        # Weak activation — should NOT resurrect
        result = check_resurrection(tmp_db, mem_id, cue_activation=0.05)
        assert result is False
        rec = get_memory(tmp_db, mem_id)
        assert rec.archived is True

        # Strong activation — should resurrect
        result = check_resurrection(tmp_db, mem_id, cue_activation=0.20)
        assert result is True
        rec = get_memory(tmp_db, mem_id)
        assert rec.archived is False


class TestReinforcement:
    def test_declared_corroborated_gets_full_boost(self, tmp_db):
        mem_id = create_memory(
            tmp_db,
            "User prefers concise technical answers without preamble",
            "preferences",
            base_strength=0.5,
        )
        # Response echoes the memory's content => corroborated => full boost.
        delta = make_delta(
            served=[mem_id],
            declared=[mem_id],
            asst_text="Here are concise technical answers without preamble as you prefer.",
        )
        apply_reinforcement(tmp_db, delta)
        rec = get_memory(tmp_db, mem_id)
        expected = 0.5 + REINFORCE_BASE * CATEGORY_MULTIPLIERS.get("preferences", 1.0)
        assert rec.base_strength == pytest.approx(expected, abs=0.01)

    def test_declared_uncorroborated_behavioral_gets_reduced_boost(self, tmp_db):
        # Declared, but nothing in the response corroborates it (behavioral category).
        mem_id = create_memory(
            tmp_db,
            "User prefers concise technical answers without preamble",
            "preferences",
            base_strength=0.5,
        )
        delta = make_delta(
            served=[mem_id],
            declared=[mem_id],
            asst_text="The weather today is unrelated to anything stored.",
        )
        apply_reinforcement(tmp_db, delta)
        rec = get_memory(tmp_db, mem_id)
        full = REINFORCE_BASE * CATEGORY_MULTIPLIERS.get("preferences", 1.0)
        reduced = full * DECLARED_UNCORROBORATED_BEHAVIORAL_MULT
        assert rec.base_strength == pytest.approx(0.5 + reduced, abs=0.01)
        assert rec.base_strength < 0.5 + full

    def test_over_reported_knowledge_memory_discounted(self, tmp_db):
        # Core over-report case: a knowledge memory declared but not actually used.
        mem_id = create_memory(
            tmp_db,
            "The deployment script lives in the scripts deploy folder somewhere",
            "knowledge",
            base_strength=0.5,
        )
        delta = make_delta(
            served=[mem_id],
            declared=[mem_id],
            asst_text="Let me tell you about cats and dogs and the ocean.",
        )
        apply_reinforcement(tmp_db, delta)
        rec = get_memory(tmp_db, mem_id)
        full = REINFORCE_BASE * CATEGORY_MULTIPLIERS.get("knowledge", 1.0)
        reduced = full * DECLARED_UNCORROBORATED_MULT
        assert rec.base_strength == pytest.approx(0.5 + reduced, abs=0.01)

    def test_tool_activity_corroborates_declaration(self, tmp_db):
        # No lexical trace in prose, but the memory is echoed in this turn's tool
        # activity => corroborated => full boost.
        mem_id = create_memory(
            tmp_db,
            "Project database migrations live under the thyra db migrations module",
            "knowledge",
            base_strength=0.5,
        )
        delta = make_delta(served=[mem_id], declared=[mem_id], asst_text="Done.")
        delta.tool_activity = "Read thyra db migrations module schema database project"
        apply_reinforcement(tmp_db, delta)
        rec = get_memory(tmp_db, mem_id)
        expected = 0.5 + REINFORCE_BASE * CATEGORY_MULTIPLIERS.get("knowledge", 1.0)
        assert rec.base_strength == pytest.approx(expected, abs=0.01)

    def test_probationary_uncorroborated_does_not_graduate(self, tmp_db):
        mem_id = create_memory(
            tmp_db,
            "User prefers tabs over spaces for indentation",
            "preferences",
            base_strength=0.4,
            decay_rate=0.05,
            probationary=True,
        )
        delta = make_delta(
            served=[mem_id],
            declared=[mem_id],
            asst_text="Unrelated response about the weather and the news.",
        )
        apply_reinforcement(tmp_db, delta)
        rec = get_memory(tmp_db, mem_id)
        assert rec.probationary is True  # stayed probationary — no corroboration

    def test_spoofed_id_not_reinforced(self, tmp_db):
        mem_id = create_memory(tmp_db, "Memory A", base_strength=0.5)
        fake_id = "m_doesnotexist00000"
        # declared includes a fake ID not in served
        delta = make_delta(served=[mem_id], declared=[mem_id, fake_id])
        apply_reinforcement(tmp_db, delta)
        # fake_id was not in served set → should not cause error
        assert get_memory(tmp_db, mem_id) is not None

    def test_cross_tenant_id_not_reinforced(self, tmp_db):
        mem_a = create_memory(
            tmp_db, "Tenant A memory", user_id="user_a", agent_id="agent_x"
        )
        mem_b = create_memory(
            tmp_db, "Tenant B memory", user_id="user_b", agent_id="agent_x"
        )
        # Tenant B tries to declare user_a's memory
        delta = DeltaEvent(
            session_id="test",
            turn_id="t1",
            user_id="user_b",
            agent_id="agent_x",
            timestamp=int(time.time() * 1000),
            memories_served=[mem_b],
            memories_declared=[mem_b, mem_a],  # mem_a is from another tenant
            cues_fired=[],
        )
        apply_reinforcement(tmp_db, delta)
        # mem_a should NOT be strengthened
        rec_a = get_memory(tmp_db, mem_a, user_id="user_a", agent_id="agent_x")
        assert rec_a.base_strength == pytest.approx(1.0)

    def test_surfaced_only_gets_small_boost(self, tmp_db):
        mem_id = create_memory(tmp_db, "Surfaced memory", base_strength=0.5)
        delta = make_delta(served=[mem_id], declared=[])  # served but not declared
        apply_reinforcement(tmp_db, delta)
        rec = get_memory(tmp_db, mem_id)
        assert rec.base_strength == pytest.approx(0.5 + SURFACED_BOOST, abs=0.01)

    def test_probationary_memory_graduates(self, tmp_db):
        mem_id = create_memory(
            tmp_db,
            "User prefers tabs over spaces for indentation",
            "preferences",
            base_strength=0.4,
            decay_rate=0.05,
            probationary=True,
        )
        # Corroborated declaration => graduates.
        delta = make_delta(
            served=[mem_id],
            declared=[mem_id],
            asst_text="I'll use tabs over spaces for indentation as you prefer.",
        )
        apply_reinforcement(tmp_db, delta)
        rec = get_memory(tmp_db, mem_id)
        assert rec.probationary is False
        assert rec.decay_rate == pytest.approx(0.02)  # graduated to category decay

    def test_constraint_multiplier_gives_lower_boost(self, tmp_db):
        pref_id = create_memory(
            tmp_db, "A preference", "preferences", base_strength=0.5
        )
        const_id = create_memory(
            tmp_db, "A constraint", "constraints", base_strength=0.5
        )
        delta = make_delta(served=[pref_id, const_id], declared=[pref_id, const_id])
        apply_reinforcement(tmp_db, delta)
        pref = get_memory(tmp_db, pref_id)
        const = get_memory(tmp_db, const_id)
        # preferences multiplier (1.0) > constraints multiplier (0.3)
        pref_gain = pref.base_strength - 0.5
        const_gain = const.base_strength - 0.5
        assert pref_gain > const_gain


class TestCueEdges:
    def test_fire_count_incremented_for_all_cues(self, tmp_db):
        mem_id = create_memory(tmp_db, "test memory", seed_cues=False)
        upsert_cue_edge(tmp_db, "python", mem_id, U, A)
        upsert_cue_edge(tmp_db, "code", mem_id, U, A)
        tmp_db.commit()
        delta = make_delta(served=[mem_id], declared=[mem_id], cues=["python", "code"])
        update_cue_edges(tmp_db, delta)
        row = tmp_db.execute(
            "SELECT fire_count, use_count FROM cue_edges WHERE cue_id='python' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["fire_count"] >= 1
        assert row["use_count"] >= 1

    def test_prune_removes_useless_edges(self, tmp_db):
        mem_id = create_memory(tmp_db, "test", seed_cues=False)
        upsert_cue_edge(tmp_db, "useless", mem_id, U, A)
        # Simulate many fires with zero uses
        tmp_db.execute(
            "UPDATE cue_edges SET fire_count=10, use_count=0 WHERE cue_id='useless' AND memory_id=?",
            (mem_id,),
        )
        tmp_db.commit()
        pruned = prune_weak_cue_edges(tmp_db, U, A)
        assert pruned >= 1
        row = tmp_db.execute(
            "SELECT * FROM cue_edges WHERE cue_id='useless' AND memory_id=?", (mem_id,)
        ).fetchone()
        assert row is None

    def test_hebbian_association_forms_edge(self, tmp_db):
        mem1 = create_memory(tmp_db, "Memory A")
        mem2 = create_memory(tmp_db, "Memory B")
        # Simulate 3 deltas where both memories are co-used
        window = [
            make_delta(served=[mem1, mem2], declared=[mem1, mem2]) for _ in range(3)
        ]
        upserted = hebbian_association(tmp_db, window, U, A)
        assert upserted >= 1
        from thyra.models.memory import load_assoc_edge_map

        assoc = load_assoc_edge_map(tmp_db, U, A)
        # mem1 should be linked to mem2
        assert any(mid == mem2 for mid, _ in assoc.get(mem1, []))


class TestWorkerIntegration:
    def test_worker_processes_delta_file(self, tmp_db, tmp_path, monkeypatch):
        """End-to-end: write a delta file, run worker, assert memory reinforced."""
        import os

        db_path = os.environ.get("THYRA_DB_PATH", str(tmp_path / "test.db"))
        queue_dir = pathlib.Path(db_path).parent / "delta_queue"
        queue_dir.mkdir(parents=True, exist_ok=True)

        mem_id = create_memory(
            tmp_db, "I use Python for all projects", "preferences", base_strength=0.5
        )
        initial_strength = 0.5

        delta_data = {
            "session_id": "worker-test",
            "turn_id": f"wt-{int(time.time())}",
            "user_id": U,
            "agent_id": A,
            "timestamp": int(time.time() * 1000),
            "memories_served": [mem_id],
            "memories_declared": [mem_id],
            "cues_fired": ["python"],
            "raw_user_text": "Tell me about Python",
            "raw_assistant_text": f"Sure! <memories_used>{mem_id}</memories_used>",
        }
        delta_file = queue_dir / f"{delta_data['timestamp']}_test.json"
        delta_file.write_text(json.dumps(delta_data), encoding="utf-8")

        from thyra.consolidation.worker import BackgroundWorker
        from thyra.recall.cache import HOT_CACHE

        worker = BackgroundWorker(db_path=db_path)
        worker._process_queue()

        HOT_CACHE.clear()
        rec = get_memory(tmp_db, mem_id)
        assert rec.base_strength > initial_strength
        assert not delta_file.exists()  # file consumed
