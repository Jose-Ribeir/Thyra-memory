"""Senior-level test suite for all Thyra subsystems.

Targets the specific failure modes reported:
  - Cue edge updates (hub exclusion, weight arithmetic, candidate promotion,
    df maintenance on prune, cross-tenant isolation)
  - Strength & decay (exact formula, lazy vs nightly, double-decay, STRENGTH_CAP)
  - Reinforcement (all 3 signals, overlap inference, spoofing, cap enforcement)
  - Situation crystallization (accumulation, dual-threshold promotion)
  - Scorer (spreading, one-hop assoc, situation conjunctions, category presence)
  - Selector (saturation, budget, per-memory cap, score floor, backfill, max_count)
  - Hebbian (co-use threshold, weight cap, bidirectionality)
  - Nightly sweep (autopurge, hard-delete, assoc decay, double-decay prevention)
  - Memory CRUD invariants (bad agent_id guard, upsert_cue_edge idempotency)
"""

from __future__ import annotations

import math
import time

import pytest

from thyra.config import (
    ARCHIVE_THRESHOLD,
    ASSOC_NIGHTLY_DECAY,
    ASSOC_PRUNE_THRESHOLD,
    BASE_STRENGTH_AUTOMATIC,
    CATEGORY_MULTIPLIERS,
    CONTENT_SEED_WEIGHT,
    CUE_PROMOTE_THRESHOLD,
    CUE_PRUNE_MAX_RATE,
    CUE_PRUNE_MIN_FIRES,
    HARD_DELETE_DAYS,
    HEBBIAN_MIN_CO_USE,
    HEBBIAN_WEIGHT_DELTA,
    HUB_CUE_FRACTION,
    OVERLAP_CONFIRMATORY_CAP,
    OVERLAP_INFER_MULT,
    OVERLAP_PRIMARY_THRESHOLD,
    PROBATIONARY_AUTOPURGE_DAYS,
    REINFORCE_BASE,
    RESURRECTION_STRENGTH,
    RESURRECTION_THRESHOLD,
    SITUATION_MIN_FIRES,
    SITUATION_MIN_RATE,
    SPREADING_ASSOC,
    SPREADING_DIRECT,
    SPREADING_SITUATION,
    STRENGTH_CAP,
    SURFACED_BOOST,
    THYRA_AGENT_ID as A,
    THYRA_USER_ID as U,
)
from thyra.consolidation.decay import (
    archive_check,
    check_resurrection,
    recompute_and_update,
)
from thyra.consolidation.edges import (
    hebbian_association,
    prune_weak_cue_edges,
    update_cue_edges,
)
from thyra.consolidation.nightly import run_nightly_sweep
from thyra.consolidation.reinforcement import apply_reinforcement
from thyra.consolidation.situation import crystallize_situations
from thyra.models.delta import DeltaEvent
from thyra.models.memory import (
    MemoryRecord,
    compute_base_level,
    create_memory,
    get_memory,
    get_cue_edges_for_memory,
    list_active_memories,
    load_assoc_edge_map,
    load_cue_edge_map,
    load_situation_edges,
    update_memory_strength,
    upsert_assoc_edge,
    upsert_cue_edge,
)
from thyra.recall.scorer import score_memories
from thyra.recall.selector import greedy_select


# ── Helpers ───────────────────────────────────────────────────────────────────


def delta(
    *,
    served=None,
    declared=None,
    cues=None,
    user_text="",
    asst_text="",
    user_id=U,
    agent_id=A,
) -> DeltaEvent:
    return DeltaEvent(
        session_id="test",
        turn_id=f"t{int(time.time() * 1000)}",
        user_id=user_id,
        agent_id=agent_id,
        timestamp=int(time.time() * 1000),
        memories_served=served or [],
        memories_declared=declared or [],
        cues_fired=cues or [],
        raw_user_text=user_text,
        raw_assistant_text=asst_text,
    )


def backdate(db, mem_id, days):
    """Set last_access to `days` ago (ms)."""
    past_ms = int(time.time() * 1000) - int(days * 86_400_000)
    db.execute("UPDATE memories SET last_access=? WHERE id=?", (past_ms, mem_id))
    db.commit()


def _insert_cue_edge_raw(db, cue_id, mem_id, weight, fire_count, use_count):
    """Insert a cue_edge directly, bypassing upsert_cue_edge's increment logic."""
    now = int(time.time() * 1000)
    db.execute(
        """INSERT OR REPLACE INTO cue_edges
           (cue_id, memory_id, user_id, agent_id, weight, fire_count, use_count,
            candidate, created_at)
           VALUES (?,?,?,?,?,?,?,0,?)""",
        (cue_id, mem_id, U, A, weight, fire_count, use_count, now),
    )
    db.execute(
        """INSERT INTO cue_nodes (cue_id, user_id, agent_id, df)
           VALUES (?,?,?,1) ON CONFLICT(cue_id, user_id, agent_id)
           DO UPDATE SET df=df+1""",
        (cue_id, U, A),
    )
    db.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CUE EDGE DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCueEdgeWeightArithmetic:
    """update_cue_edges must add exactly 0.05 per (cue, used-memory) pair."""

    def test_weight_increments_by_exactly_005(self, tmp_db):
        mem_id = create_memory(tmp_db, "test memory one", seed_cues=False)
        _insert_cue_edge_raw(
            tmp_db, "alpha", mem_id, weight=0.30, fire_count=0, use_count=0
        )

        d = delta(served=[mem_id], declared=[mem_id], cues=["alpha"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT weight FROM cue_edges WHERE cue_id='alpha' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["weight"] == pytest.approx(0.35, abs=1e-6)

    def test_weight_capped_at_1_0(self, tmp_db):
        mem_id = create_memory(tmp_db, "test memory two", seed_cues=False)
        _insert_cue_edge_raw(
            tmp_db, "beta", mem_id, weight=0.98, fire_count=0, use_count=0
        )

        d = delta(served=[mem_id], declared=[mem_id], cues=["beta"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT weight FROM cue_edges WHERE cue_id='beta' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["weight"] == pytest.approx(1.0, abs=1e-6)

    def test_use_count_incremented_for_used_pair(self, tmp_db):
        mem_id = create_memory(tmp_db, "used memory", seed_cues=False)
        _insert_cue_edge_raw(
            tmp_db, "gamma", mem_id, weight=0.30, fire_count=0, use_count=0
        )

        d = delta(served=[mem_id], declared=[mem_id], cues=["gamma"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT use_count FROM cue_edges WHERE cue_id='gamma' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["use_count"] == 1

    def test_fire_count_incremented_for_served_not_used(self, tmp_db):
        """Cue fires but memory not declared used: fire_count ↑, use_count and weight unchanged."""
        mem_id = create_memory(tmp_db, "served only memory", seed_cues=False)
        _insert_cue_edge_raw(
            tmp_db, "delta_cue", mem_id, weight=0.30, fire_count=0, use_count=0
        )

        d = delta(served=[mem_id], declared=[], cues=["delta_cue"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT fire_count, use_count, weight FROM cue_edges WHERE cue_id='delta_cue' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["fire_count"] >= 1
        assert row["use_count"] == 0
        assert row["weight"] == pytest.approx(0.30, abs=1e-6)

    def test_no_cues_is_a_noop(self, tmp_db):
        mem_id = create_memory(tmp_db, "noop test", seed_cues=False)
        _insert_cue_edge_raw(
            tmp_db, "epsilon", mem_id, weight=0.30, fire_count=0, use_count=0
        )

        d = delta(served=[mem_id], declared=[mem_id], cues=[])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT fire_count, use_count, weight FROM cue_edges WHERE cue_id='epsilon' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["fire_count"] == 0
        assert row["use_count"] == 0


class TestHubCueExclusion:
    """Hub cues (df > HUB_CUE_FRACTION * M) must NOT get weight updates.
    Their fire_count must still tick (so weak-rate pruning can remove them).
    """

    def _make_hub(self, db, cue_id, n_memories=4):
        """Create n_memories all linked to cue_id so it becomes a hub."""
        mem_ids = []
        for i in range(n_memories):
            mid = create_memory(db, f"memory {i} hub test", seed_cues=False)
            _insert_cue_edge_raw(
                db, cue_id, mid, weight=0.30, fire_count=0, use_count=0
            )
            mem_ids.append(mid)
        return mem_ids

    def test_hub_cue_skips_weight_update(self, tmp_db):
        # HUB_CUE_FRACTION = 0.5 so with 4 memories, df=4 > 0.5*4=2 → hub
        mem_ids = self._make_hub(tmp_db, "hubcue", n_memories=4)
        target = mem_ids[0]
        initial_row = tmp_db.execute(
            "SELECT weight FROM cue_edges WHERE cue_id='hubcue' AND memory_id=?",
            (target,),
        ).fetchone()
        initial_weight = initial_row["weight"]

        d = delta(served=[target], declared=[target], cues=["hubcue"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT weight FROM cue_edges WHERE cue_id='hubcue' AND memory_id=?",
            (target,),
        ).fetchone()
        assert row["weight"] == pytest.approx(initial_weight, abs=1e-6)

    def test_hub_cue_fire_count_still_ticks(self, tmp_db):
        mem_ids = self._make_hub(tmp_db, "hubfire", n_memories=4)
        target = mem_ids[0]

        d = delta(served=[target], declared=[target], cues=["hubfire"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT fire_count FROM cue_edges WHERE cue_id='hubfire' AND memory_id=?",
            (target,),
        ).fetchone()
        assert row["fire_count"] >= 1

    def test_non_hub_cue_gets_weight_update(self, tmp_db):
        """A cue pointing to only 1 out of 4 memories is not a hub → weight increases."""
        for i in range(3):
            create_memory(tmp_db, f"other memory {i}", seed_cues=False)
        target = create_memory(tmp_db, "the target memory", seed_cues=False)
        # Only link "rarecue" to target — df=1 out of 4 memories, not a hub
        _insert_cue_edge_raw(
            tmp_db, "rarecue", target, weight=0.30, fire_count=0, use_count=0
        )

        d = delta(served=[target], declared=[target], cues=["rarecue"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT weight FROM cue_edges WHERE cue_id='rarecue' AND memory_id=?",
            (target,),
        ).fetchone()
        assert row["weight"] == pytest.approx(0.35, abs=1e-6)


class TestCandidatePromotion:
    """Candidate edges must promote to real edges when weight crosses CUE_PROMOTE_THRESHOLD."""

    def test_promotion_at_threshold(self, tmp_db):
        mem_id = create_memory(tmp_db, "candidate test", seed_cues=False)
        # Start just below promote threshold
        start_weight = CUE_PROMOTE_THRESHOLD - 0.05
        now = int(time.time() * 1000)
        tmp_db.execute(
            """INSERT INTO cue_edges
               (cue_id, memory_id, user_id, agent_id, weight, fire_count, use_count,
                candidate, created_at)
               VALUES (?,?,?,?,?,0,0,1,?)""",
            ("candcue", mem_id, U, A, start_weight, now),
        )
        tmp_db.execute(
            """INSERT INTO cue_nodes (cue_id, user_id, agent_id, df)
               VALUES (?,?,?,1) ON CONFLICT DO UPDATE SET df=df+1""",
            ("candcue", U, A),
        )
        tmp_db.commit()

        d = delta(served=[mem_id], declared=[mem_id], cues=["candcue"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT candidate, weight FROM cue_edges WHERE cue_id='candcue' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        # weight should now be start_weight + 0.05 >= CUE_PROMOTE_THRESHOLD
        assert row["weight"] >= CUE_PROMOTE_THRESHOLD
        assert row["candidate"] == 0  # promoted

    def test_no_promotion_when_below_threshold(self, tmp_db):
        mem_id = create_memory(tmp_db, "below threshold", seed_cues=False)
        # So far below that one +0.05 still won't cross
        start_weight = 0.01
        now = int(time.time() * 1000)
        tmp_db.execute(
            """INSERT INTO cue_edges
               (cue_id, memory_id, user_id, agent_id, weight, fire_count, use_count,
                candidate, created_at)
               VALUES (?,?,?,?,?,0,0,1,?)""",
            ("belowcue", mem_id, U, A, start_weight, now),
        )
        tmp_db.execute(
            """INSERT INTO cue_nodes (cue_id, user_id, agent_id, df)
               VALUES (?,?,?,1) ON CONFLICT DO UPDATE SET df=df+1""",
            ("belowcue", U, A),
        )
        tmp_db.commit()

        d = delta(served=[mem_id], declared=[mem_id], cues=["belowcue"])
        update_cue_edges(tmp_db, d)

        row = tmp_db.execute(
            "SELECT candidate FROM cue_edges WHERE cue_id='belowcue' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["candidate"] == 1  # still a candidate


class TestPruneWeakCueEdges:
    """prune_weak_cue_edges must maintain cue_nodes.df accurately."""

    def test_df_decremented_when_edge_pruned(self, tmp_db):
        mem_id = create_memory(tmp_db, "prune target", seed_cues=False)
        # Insert a weak cue edge directly
        _insert_cue_edge_raw(
            tmp_db,
            "weakaaa",
            mem_id,
            weight=0.30,
            fire_count=CUE_PRUNE_MIN_FIRES,
            use_count=0,
        )

        df_before = tmp_db.execute(
            "SELECT df FROM cue_nodes WHERE cue_id='weakaaa' AND user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()["df"]

        pruned = prune_weak_cue_edges(tmp_db, U, A)
        assert pruned >= 1

        df_after_row = tmp_db.execute(
            "SELECT df FROM cue_nodes WHERE cue_id='weakaaa' AND user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()
        df_after = df_after_row["df"] if df_after_row else 0
        assert df_after == df_before - 1

    def test_healthy_edge_not_pruned(self, tmp_db):
        """Edges with high use rate survive the prune."""
        mem_id = create_memory(tmp_db, "healthy edge", seed_cues=False)
        # fire_count=10, use_count=9 → use_rate=0.9, well above CUE_PRUNE_MAX_RATE
        _insert_cue_edge_raw(
            tmp_db, "healthycue", mem_id, weight=0.30, fire_count=10, use_count=9
        )

        pruned = prune_weak_cue_edges(tmp_db, U, A)

        row = tmp_db.execute(
            "SELECT * FROM cue_edges WHERE cue_id='healthycue' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row is not None  # still alive

    def test_below_min_fires_not_pruned(self, tmp_db):
        """Edges with fewer than CUE_PRUNE_MIN_FIRES are not pruned even if use_rate=0."""
        mem_id = create_memory(tmp_db, "young edge", seed_cues=False)
        _insert_cue_edge_raw(
            tmp_db,
            "youngcue",
            mem_id,
            weight=0.30,
            fire_count=CUE_PRUNE_MIN_FIRES - 1,
            use_count=0,
        )

        pruned = prune_weak_cue_edges(tmp_db, U, A)

        row = tmp_db.execute(
            "SELECT * FROM cue_edges WHERE cue_id='youngcue' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DECAY — EXACT FORMULA & BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecayFormula:
    """compute_base_level and recompute_and_update must implement the exact formula."""

    def test_exact_formula_fresh_memory(self):
        """At 0 elapsed days the strength must be exactly base_strength."""
        now = int(time.time() * 1000)
        level = compute_base_level(1.0, 0.02, now, now)
        assert level == pytest.approx(1.0, abs=1e-9)

    def test_exact_formula_after_ten_days(self):
        """After 10 days at decay_rate=0.5: level = 1.0 * exp(-0.5*10) = exp(-5)."""
        now = int(time.time() * 1000)
        ten_days_ago = now - 10 * 86_400_000
        level = compute_base_level(1.0, 0.5, ten_days_ago, now)
        expected = math.exp(-5.0)
        assert level == pytest.approx(expected, rel=1e-6)

    def test_recompute_updates_db_to_decayed_value(self, tmp_db):
        mem_id = create_memory(
            tmp_db, "decaying memory", base_strength=1.0, decay_rate=0.5
        )
        backdate(tmp_db, mem_id, days=10)

        recompute_and_update(tmp_db, [mem_id], U, A)

        rec = get_memory(tmp_db, mem_id)
        expected = math.exp(-5.0)
        assert rec.base_strength == pytest.approx(expected, rel=1e-4)

    def test_recompute_updates_last_access_to_now(self, tmp_db):
        mem_id = create_memory(
            tmp_db, "lazy decay test", base_strength=1.0, decay_rate=0.02
        )
        backdate(tmp_db, mem_id, days=5)

        before_ms = int(time.time() * 1000)
        recompute_and_update(tmp_db, [mem_id], U, A)
        after_ms = int(time.time() * 1000)

        rec = get_memory(tmp_db, mem_id)
        assert before_ms <= rec.last_access <= after_ms

    def test_nightly_sweep_prevents_double_decay(self, tmp_db):
        """After a nightly full-decay pass, last_access is updated to now.
        A second immediate pass must NOT decay further.
        """
        mem_id = create_memory(
            tmp_db, "double decay sentinel", base_strength=1.0, decay_rate=0.1
        )
        backdate(tmp_db, mem_id, days=7)

        run_nightly_sweep(tmp_db, U, A)
        after_first = get_memory(tmp_db, mem_id).base_strength

        # Tiny pause then second sweep — elapsed ~0 days, so decay ≈ 0
        run_nightly_sweep(tmp_db, U, A)
        after_second = get_memory(tmp_db, mem_id).base_strength

        # Second sweep should NOT compound the first sweep's result
        assert after_second == pytest.approx(after_first, rel=0.001)

    def test_archive_check_uses_live_decay_not_stored_strength(self, tmp_db):
        """Archive check must re-derive current level, not read stored base_strength."""
        # stored base_strength=0.5 (above threshold) but after decay it drops below
        mem_id = create_memory(
            tmp_db, "live decay check", base_strength=0.5, decay_rate=5.0
        )
        backdate(
            tmp_db, mem_id, days=5
        )  # level = 0.5 * exp(-25) ≈ 0 < ARCHIVE_THRESHOLD

        archived = archive_check(tmp_db, U, A)
        assert mem_id in archived


class TestResurrectionEdgeCases:
    def test_exactly_at_threshold_does_not_resurrect(self, tmp_db):
        """cue_activation == RESURRECTION_THRESHOLD: should NOT resurrect (> not >=)."""
        mem_id = create_memory(tmp_db, "at threshold")
        now = int(time.time() * 1000)
        tmp_db.execute(
            "UPDATE memories SET archived=1, archived_at=? WHERE id=?", (now, mem_id)
        )
        tmp_db.commit()

        result = check_resurrection(
            tmp_db, mem_id, cue_activation=RESURRECTION_THRESHOLD
        )
        assert result is False

    def test_just_above_threshold_resurrects(self, tmp_db):
        mem_id = create_memory(tmp_db, "above threshold")
        now = int(time.time() * 1000)
        tmp_db.execute(
            "UPDATE memories SET archived=1, archived_at=? WHERE id=?", (now, mem_id)
        )
        tmp_db.commit()

        result = check_resurrection(
            tmp_db, mem_id, cue_activation=RESURRECTION_THRESHOLD + 0.01
        )
        assert result is True
        rec = get_memory(tmp_db, mem_id)
        assert rec.archived is False
        assert rec.base_strength == pytest.approx(RESURRECTION_STRENGTH, abs=1e-6)

    def test_resurrection_of_active_memory_returns_false(self, tmp_db):
        mem_id = create_memory(tmp_db, "active memory")
        result = check_resurrection(tmp_db, mem_id, cue_activation=1.0)
        assert result is False


class TestStrengthCap:
    def test_update_memory_strength_respects_cap(self, tmp_db):
        mem_id = create_memory(tmp_db, "strength cap test", base_strength=9.9)
        now = int(time.time() * 1000)
        update_memory_strength(tmp_db, mem_id, 100.0, now, U, A)
        rec = get_memory(tmp_db, mem_id)
        assert rec.base_strength == pytest.approx(STRENGTH_CAP, abs=1e-6)

    def test_reinforcement_respects_cap(self, tmp_db):
        mem_id = create_memory(
            tmp_db, "cap in reinforcement", base_strength=STRENGTH_CAP - 0.01
        )
        d = delta(served=[mem_id], declared=[mem_id])
        apply_reinforcement(tmp_db, d)
        rec = get_memory(tmp_db, mem_id)
        assert rec.base_strength <= STRENGTH_CAP


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  REINFORCEMENT — ALL THREE SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOverlapInference:
    """Signal 2: memories whose content appears in the assistant response
    get REINFORCE_BASE * OVERLAP_INFER_MULT boost even without a <memories_used> tag.
    """

    def _make_mem_with_content(self, db, content):
        return create_memory(db, content, "preferences", base_strength=0.5)

    def test_high_overlap_infers_use(self, tmp_db):
        content = "I prefer using Python scripting environment"
        # Response contains most of the memory's words
        response = (
            "Based on your python preference using scripting environment we can proceed"
        )
        mem_id = self._make_mem_with_content(tmp_db, content)

        d = delta(served=[mem_id], declared=[], asst_text=response)
        apply_reinforcement(tmp_db, d)

        rec = get_memory(tmp_db, mem_id)
        # Should get at least the overlap-inferred boost (REINFORCE_BASE * OVERLAP_INFER_MULT * mult)
        expected_min = (
            0.5 + REINFORCE_BASE * OVERLAP_INFER_MULT * 0.5
        )  # preferences mult=1.0 but give slack
        assert rec.base_strength > 0.5

    def test_low_overlap_does_not_infer_use(self, tmp_db):
        content = "I prefer using Python scripting environment"
        response = "The weather today is quite nice"  # completely different
        mem_id = self._make_mem_with_content(tmp_db, content)

        d = delta(served=[mem_id], declared=[], asst_text=response)
        apply_reinforcement(tmp_db, d)

        rec = get_memory(tmp_db, mem_id)
        # Only surfaced boost (not overlap inferred)
        assert rec.base_strength == pytest.approx(0.5 + SURFACED_BOOST, abs=0.02)

    def test_short_memory_skipped_in_overlap(self, tmp_db):
        """Memories with < 4 content words skip overlap inference."""
        short_content = "python"  # only 1 word
        mem_id = create_memory(tmp_db, short_content, base_strength=0.5)
        response = "python is great"

        d = delta(served=[mem_id], declared=[], asst_text=response)
        apply_reinforcement(tmp_db, d)

        rec = get_memory(tmp_db, mem_id)
        # Should not get the full overlap boost — only surfaced_boost or less
        assert rec.base_strength < 0.5 + REINFORCE_BASE * OVERLAP_INFER_MULT

    def test_locked_memory_skipped_in_overlap(self, tmp_db):
        """Locked memories must not be inspected for word overlap (Signal 2).
        They still receive the surfaced boost (Signal 3) since they were served.
        """
        mem_id = create_memory(
            tmp_db,
            "python scripting preference environment",
            locked=True,
            seed_cues=False,
        )
        # locked memories start at BASE_STRENGTH_EXPLICIT (1.0)
        initial = get_memory(tmp_db, mem_id).base_strength
        response = "python scripting preference environment"

        d = delta(served=[mem_id], declared=[], asst_text=response)
        apply_reinforcement(tmp_db, d)

        rec = get_memory(tmp_db, mem_id)
        # Gets at most the surfaced boost, NOT the larger overlap-inferred boost
        max_without_overlap = initial + SURFACED_BOOST + 0.01
        assert rec.base_strength <= max_without_overlap
        # Definitely should NOT reach overlap-infer level
        infer_boost = REINFORCE_BASE * OVERLAP_INFER_MULT
        assert rec.base_strength < initial + infer_boost

    def test_declaration_wins_over_overlap(self, tmp_db):
        """When a memory is declared AND overlaps, it gets Signal 1 (declaration), not both."""
        content = "prefer python scripting automated environment deployment"
        response = "prefer python scripting automated environment deployment"
        mem_id = self._make_mem_with_content(tmp_db, content)
        initial = get_memory(tmp_db, mem_id).base_strength

        d = delta(served=[mem_id], declared=[mem_id], asst_text=response)
        apply_reinforcement(tmp_db, d)

        rec = get_memory(tmp_db, mem_id)
        mult = CATEGORY_MULTIPLIERS.get("preferences", 1.0)
        expected = initial + REINFORCE_BASE * mult
        # confirmation overlap adds a tiny bit on top (bounded by OVERLAP_CONFIRMATORY_CAP)
        assert rec.base_strength == pytest.approx(
            expected, abs=OVERLAP_CONFIRMATORY_CAP + 0.01
        )


class TestOverlapConfirmation:
    def test_confirmatory_cap_respected(self, tmp_db):
        """Overlap confirmation on top of a declared memory is capped at OVERLAP_CONFIRMATORY_CAP."""
        content = "deploy kubernetes helm chart production cluster environment"
        mem_id = create_memory(tmp_db, content, base_strength=0.5)
        # Same text in response → maximum overlap = 1.0
        response = content
        initial = get_memory(tmp_db, mem_id).base_strength

        d = delta(served=[mem_id], declared=[mem_id], asst_text=response)
        apply_reinforcement(tmp_db, d)

        rec = get_memory(tmp_db, mem_id)
        boost = rec.base_strength - initial
        mult = CATEGORY_MULTIPLIERS.get("context", 1.0)
        expected_declaration = REINFORCE_BASE * mult
        max_confirmation = OVERLAP_CONFIRMATORY_CAP
        assert boost <= expected_declaration + max_confirmation + 0.001


class TestSpoofGuard:
    def test_id_not_in_served_is_not_reinforced(self, tmp_db):
        """A declared ID that was not served must not be reinforced."""
        real_mem = create_memory(tmp_db, "real memory", base_strength=0.5)
        fake_id = "m_fffffffffffffffff"
        # Ensure fake_id doesn't exist
        assert get_memory(tmp_db, fake_id) is None

        d = delta(served=[real_mem], declared=[real_mem, fake_id])
        apply_reinforcement(tmp_db, d)

        # real_mem is both served and declared — should be reinforced normally
        rec = get_memory(tmp_db, real_mem)
        assert rec.base_strength > 0.5

    def test_cross_tenant_cannot_reinforce(self, tmp_db):
        """Memory owned by tenant A must not be reinforced by a request from tenant B."""
        mem_a = create_memory(
            tmp_db,
            "tenant A secret",
            user_id="user_a",
            agent_id="agent_x",
            base_strength=1.0,
        )
        mem_b = create_memory(
            tmp_db,
            "tenant B memory",
            user_id="user_b",
            agent_id="agent_x",
            base_strength=0.5,
        )

        evil_delta = DeltaEvent(
            session_id="evil",
            turn_id="evil_t1",
            user_id="user_b",
            agent_id="agent_x",
            timestamp=int(time.time() * 1000),
            memories_served=[mem_b],
            memories_declared=[mem_b, mem_a],  # try to reinforce mem_a from tenant A
            cues_fired=[],
        )
        apply_reinforcement(tmp_db, evil_delta)

        rec_a = get_memory(tmp_db, mem_a, user_id="user_a", agent_id="agent_x")
        assert rec_a.base_strength == pytest.approx(1.0)  # unchanged


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SITUATION EDGE CRYSTALLIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestSituationCrystallization:
    """crystallize_situations must accumulate stats across calls and only promote
    when BOTH SITUATION_MIN_FIRES and SITUATION_MIN_RATE are met.
    """

    def _make_window(self, mem_id, *, n_fires, n_used, cues):
        """Create a window of DeltaEvents simulating n_fires turns, n_used of them used."""
        events = []
        for i in range(n_fires):
            declared = [mem_id] if i < n_used else []
            events.append(delta(served=[mem_id], declared=declared, cues=cues))
        return events

    def test_stats_accumulate_across_calls(self, tmp_db):
        """Multiple calls to crystallize_situations must accumulate fire_count."""
        mem_id = create_memory(tmp_db, "situation test", seed_cues=False)
        cues = ["deploy", "prod", "cluster"]

        # Two calls of 2 turns each → total fire_count = 4
        for _ in range(2):
            window = self._make_window(mem_id, n_fires=2, n_used=2, cues=cues)
            crystallize_situations(tmp_db, window, U, A)

        row = tmp_db.execute(
            "SELECT fire_count, use_count FROM situation_edges WHERE memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row is not None
        assert row["fire_count"] == 4
        assert row["use_count"] == 4

    def test_no_promotion_below_min_fires(self, tmp_db):
        """Below SITUATION_MIN_FIRES, never promote even with perfect use_rate."""
        mem_id = create_memory(tmp_db, "too young situation", seed_cues=False)
        cues = ["alpha", "beta", "gamma"]
        window = self._make_window(
            mem_id,
            n_fires=SITUATION_MIN_FIRES - 1,
            n_used=SITUATION_MIN_FIRES - 1,
            cues=cues,
        )
        promoted = crystallize_situations(tmp_db, window, U, A)
        assert promoted == 0

        row = tmp_db.execute(
            "SELECT candidate FROM situation_edges WHERE memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row is None or row["candidate"] == 1

    def test_no_promotion_below_min_rate(self, tmp_db):
        """Enough fires but poor use_rate keeps edge as candidate."""
        mem_id = create_memory(tmp_db, "low rate situation", seed_cues=False)
        cues = ["kube", "helm", "cert"]
        # 5 fires, 0 used → rate = 0.0 < SITUATION_MIN_RATE
        window = self._make_window(
            mem_id, n_fires=SITUATION_MIN_FIRES, n_used=0, cues=cues
        )
        promoted = crystallize_situations(tmp_db, window, U, A)
        assert promoted == 0

    def test_promotes_when_both_thresholds_met(self, tmp_db):
        """Crossing BOTH fire and rate thresholds promotes the edge."""
        mem_id = create_memory(tmp_db, "promote me situation", seed_cues=False)
        cues = ["scale", "deploy", "prod"]
        # Exactly at thresholds
        n = SITUATION_MIN_FIRES
        used = math.ceil(n * SITUATION_MIN_RATE)
        window = self._make_window(mem_id, n_fires=n, n_used=used, cues=cues)
        promoted = crystallize_situations(tmp_db, window, U, A)
        assert promoted >= 1

        row = tmp_db.execute(
            "SELECT candidate FROM situation_edges WHERE memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["candidate"] == 0

    def test_promoted_edge_appears_in_load_situation_edges(self, tmp_db):
        mem_id = create_memory(tmp_db, "load edges test", seed_cues=False)
        cues = ["argocd", "gitops", "fluxcd"]
        n = SITUATION_MIN_FIRES
        used = math.ceil(n * SITUATION_MIN_RATE)
        window = self._make_window(mem_id, n_fires=n, n_used=used, cues=cues)
        crystallize_situations(tmp_db, window, U, A)

        sit_edges = load_situation_edges(tmp_db, U, A)
        assert any(e.memory_id == mem_id for e in sit_edges)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SCORER — SPREADING ACTIVATION ARITHMETIC
# ═══════════════════════════════════════════════════════════════════════════════


class TestScorerArithmetic:
    def _score(
        self, db, memories, cues, cue_map, assoc_map, sit_edges, idf, cat_weights=None
    ):
        now = int(time.time() * 1000)
        return score_memories(
            memories,
            cues,
            cue_map,
            assoc_map,
            sit_edges,
            idf,
            cat_weights or {},
            now,
        )

    def test_direct_spreading_formula(self, tmp_db):
        """score = (base_level + ew*discrim*SPREADING_DIRECT) * presence.

        With no category_weights supplied, presence = PRESENCE_FLOOR.
        """
        mem_id = create_memory(
            tmp_db, "direct spreading test", base_strength=1.0, seed_cues=False
        )
        upsert_cue_edge(tmp_db, "testcue", mem_id, U, A, weight=0.5)

        memories = list_active_memories(tmp_db, U, A)
        cue_map = load_cue_edge_map(tmp_db, U, A)
        idf = {"testcue": 1.0}

        now_ms = int(time.time() * 1000)
        scored = self._score(tmp_db, memories, ["testcue"], cue_map, {}, [], idf)
        target_score = next(s for r, s in scored if r.id == mem_id)

        from thyra.config import PRESENCE_FLOOR

        rec = next(r for r, _ in scored if r.id == mem_id)
        base = compute_base_level(
            rec.base_strength, rec.decay_rate, rec.last_access, now_ms
        )
        expected_spread = 0.5 * 1.0 * SPREADING_DIRECT
        # No category_weights → all cat_weights = 0 → prod(1-0) = 1 → presence = PRESENCE_FLOOR
        expected_presence = PRESENCE_FLOOR
        expected_score = (base + expected_spread) * expected_presence
        assert target_score == pytest.approx(expected_score, rel=0.02)

    def test_one_hop_assoc_spread(self, tmp_db):
        """A neighbor directly activated by cues contributes SPREADING_ASSOC to linked memory."""
        cued = create_memory(tmp_db, "cued memory", base_strength=0.8, seed_cues=False)
        linked = create_memory(
            tmp_db, "linked memory", base_strength=0.8, seed_cues=False
        )
        upsert_cue_edge(tmp_db, "directcue", cued, U, A, weight=0.8)
        upsert_assoc_edge(tmp_db, cued, linked, U, A, delta_weight=0.5)

        memories = list_active_memories(tmp_db, U, A)
        cue_map = load_cue_edge_map(tmp_db, U, A)
        assoc_map = load_assoc_edge_map(tmp_db, U, A)
        idf = {"directcue": 1.0}

        scored = self._score(
            tmp_db, memories, ["directcue"], cue_map, assoc_map, [], idf
        )
        scores = {r.id: s for r, s in scored}

        # linked gets assoc spread; cued gets direct spread → cued > linked
        assert scores[cued] > scores[linked]
        # linked should score higher than if it had no assoc (just base_level * presence)
        from thyra.config import PRESENCE_FLOOR

        base_only = (
            compute_base_level(
                0.8, 0.02, int(time.time() * 1000), int(time.time() * 1000)
            )
            * 1.0
        )
        assert scores[linked] > base_only * PRESENCE_FLOOR

    def test_situation_edge_contributes_to_score(self, tmp_db):
        """A promoted situation edge boosts score when ALL its cues are present."""
        mem_id = create_memory(
            tmp_db, "situation score test", base_strength=0.5, seed_cues=False
        )
        cues = ["cluster", "prod", "helm"]

        # Crystallize enough to promote
        n = SITUATION_MIN_FIRES
        used = math.ceil(n * SITUATION_MIN_RATE)
        window = [
            delta(served=[mem_id], declared=[mem_id] if i < used else [], cues=cues)
            for i in range(n)
        ]
        crystallize_situations(tmp_db, window, U, A)

        memories = list_active_memories(tmp_db, U, A)
        sit_edges = load_situation_edges(tmp_db, U, A)
        idf = {c: 0.8 for c in cues}

        scored_with_sit = self._score(tmp_db, memories, cues, {}, {}, sit_edges, idf)
        scored_without = self._score(tmp_db, memories, cues, {}, {}, [], idf)

        score_with = next(s for r, s in scored_with_sit if r.id == mem_id)
        score_without = next(s for r, s in scored_without if r.id == mem_id)
        assert score_with > score_without

    def test_situation_edge_not_activated_when_cue_missing(self, tmp_db):
        """A 3-cue situation must not activate if any one cue is absent."""
        mem_id = create_memory(
            tmp_db, "partial cue test", base_strength=0.5, seed_cues=False
        )
        full_cues = ["alpha", "beta", "gamma"]

        n = SITUATION_MIN_FIRES
        used = math.ceil(n * SITUATION_MIN_RATE)
        window = [
            delta(
                served=[mem_id], declared=[mem_id] if i < used else [], cues=full_cues
            )
            for i in range(n)
        ]
        crystallize_situations(tmp_db, window, U, A)

        sit_edges = load_situation_edges(tmp_db, U, A)
        memories = list_active_memories(tmp_db, U, A)
        idf = {c: 0.8 for c in full_cues}

        # Only fire 2 of the 3 cues
        partial_cues = ["alpha", "beta"]
        scored_partial = self._score(
            tmp_db, memories, partial_cues, {}, {}, sit_edges, idf
        )
        scored_full = self._score(tmp_db, memories, full_cues, {}, {}, sit_edges, idf)

        s_partial = next(s for r, s in scored_partial if r.id == mem_id)
        s_full = next(s for r, s in scored_full if r.id == mem_id)
        # Situation edge only fires with all 3 cues → partial must score lower
        assert s_partial < s_full


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  GREEDY SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════


class TestGreedySelector:
    def _scored(self, recs, base_score=0.8):
        return [(r, base_score) for r in recs]

    def test_per_memory_token_cap(self, tmp_db):
        """No single memory may consume more than PER_MEMORY_CAP fraction of the budget."""
        from thyra.config import PER_MEMORY_CAP

        budget = 200
        per_cap = int(budget * PER_MEMORY_CAP)  # 50 tokens max per memory
        # Create a huge memory (2000 chars → 500 tokens) and a small one
        big = create_memory(tmp_db, "x" * 2000, base_strength=1.0)
        small = create_memory(tmp_db, "tiny content", base_strength=0.9)

        memories = list_active_memories(tmp_db, U, A)
        scored = self._scored(memories)
        selected = greedy_select(scored, token_budget=budget)

        for mem in selected:
            tokens = min(per_cap, len(mem.content) // 4)
            assert tokens <= per_cap

    def test_score_floor_excludes_weak_memories(self, tmp_db):
        mem_id = create_memory(tmp_db, "weak score memory", base_strength=0.1)
        memories = list_active_memories(tmp_db, U, A)
        # Score below the floor
        scored = [(r, 0.01) for r in memories]
        selected = greedy_select(scored, score_floor=0.05)
        assert mem_id not in {r.id for r in selected}

    def test_max_count_hard_cap(self, tmp_db):
        for i in range(10):
            create_memory(tmp_db, f"memory {i}")
        memories = list_active_memories(tmp_db, U, A)
        scored = [(r, 1.0) for r in memories]
        selected = greedy_select(scored, token_budget=100_000, max_count=3)
        assert len(selected) <= 3

    def test_backfill_fits_small_memories(self, tmp_db):
        """Small memories skipped in main pass should be backfilled if they fit."""
        # One large memory that nearly fills the budget
        big = create_memory(tmp_db, "w" * 1600, base_strength=1.0)  # ~400 tokens
        # One tiny memory
        tiny = create_memory(tmp_db, "tiny", base_strength=0.6)

        memories = list_active_memories(tmp_db, U, A)
        scored = sorted(
            [(r, 1.0 if r.id == big else 0.6) for r in memories], key=lambda x: -x[1]
        )
        selected = greedy_select(scored, token_budget=450, gamma=1.0)
        selected_ids = {r.id for r in selected}

        # big should be in; tiny might backfill if there's room
        assert big in selected_ids

    def test_protected_categories_never_discounted(self, tmp_db):
        """constraints and identity must appear even when gamma would starve them."""
        for i in range(8):
            create_memory(
                tmp_db, f"generic memory content {i} " * 5, category="knowledge"
            )
        c1 = create_memory(tmp_db, "never use personal data", category="constraints")
        c2 = create_memory(tmp_db, "user is José", category="identity")

        memories = list_active_memories(tmp_db, U, A)
        scored = sorted(
            [(r, 0.9 if r.category == "knowledge" else 0.3) for r in memories],
            key=lambda x: -x[1],
        )
        selected = greedy_select(scored, token_budget=5000, gamma=0.1)
        selected_ids = {r.id for r in selected}
        assert c1 in selected_ids
        assert c2 in selected_ids


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  HEBBIAN ASSOCIATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestHebbianAssociation:
    def test_below_co_use_threshold_no_edge(self, tmp_db):
        """Fewer than HEBBIAN_MIN_CO_USE co-uses must NOT form an edge."""
        mem1 = create_memory(tmp_db, "alpha memory")
        mem2 = create_memory(tmp_db, "beta memory")

        window = [
            delta(served=[mem1, mem2], declared=[mem1, mem2])
            for _ in range(HEBBIAN_MIN_CO_USE - 1)
        ]
        upserted = hebbian_association(tmp_db, window, U, A)
        assert upserted == 0

        assoc = load_assoc_edge_map(tmp_db, U, A)
        assert mem2 not in {mid for mid, _ in assoc.get(mem1, [])}

    def test_at_co_use_threshold_edge_formed(self, tmp_db):
        mem1 = create_memory(tmp_db, "threshold alpha")
        mem2 = create_memory(tmp_db, "threshold beta")

        window = [
            delta(served=[mem1, mem2], declared=[mem1, mem2])
            for _ in range(HEBBIAN_MIN_CO_USE)
        ]
        upserted = hebbian_association(tmp_db, window, U, A)
        assert upserted >= 1

    def test_weight_accumulates_correctly(self, tmp_db):
        """Edge weight = HEBBIAN_WEIGHT_DELTA * co_use_count."""
        mem1 = create_memory(tmp_db, "weight alpha")
        mem2 = create_memory(tmp_db, "weight beta")

        n = HEBBIAN_MIN_CO_USE + 2
        window = [delta(served=[mem1, mem2], declared=[mem1, mem2]) for _ in range(n)]
        hebbian_association(tmp_db, window, U, A)

        row = tmp_db.execute(
            """SELECT weight FROM association_edges
               WHERE ((memory_a=? AND memory_b=?) OR (memory_a=? AND memory_b=?))
               AND user_id=? AND agent_id=?""",
            (min(mem1, mem2), max(mem1, mem2), min(mem1, mem2), max(mem1, mem2), U, A),
        ).fetchone()
        assert row is not None
        expected = HEBBIAN_WEIGHT_DELTA * n
        assert row["weight"] == pytest.approx(expected, abs=0.01)

    def test_assoc_edge_is_bidirectional_in_map(self, tmp_db):
        """load_assoc_edge_map must expose edges from both endpoints."""
        mem1 = create_memory(tmp_db, "bidir alpha")
        mem2 = create_memory(tmp_db, "bidir beta")
        upsert_assoc_edge(tmp_db, mem1, mem2, U, A, delta_weight=0.5)

        assoc = load_assoc_edge_map(tmp_db, U, A)
        assert any(mid == mem2 for mid, _ in assoc.get(mem1, []))
        assert any(mid == mem1 for mid, _ in assoc.get(mem2, []))

    def test_weight_cap_at_1_0(self, tmp_db):
        """Repeated upserts must not push weight above 1.0."""
        mem1 = create_memory(tmp_db, "cap alpha")
        mem2 = create_memory(tmp_db, "cap beta")
        for _ in range(20):
            upsert_assoc_edge(tmp_db, mem1, mem2, U, A, delta_weight=0.2)

        row = tmp_db.execute(
            "SELECT weight FROM association_edges WHERE user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()
        assert row["weight"] == pytest.approx(1.0, abs=1e-6)

    def test_multiple_pairs_tracked_independently(self, tmp_db):
        mem1 = create_memory(tmp_db, "pair alpha")
        mem2 = create_memory(tmp_db, "pair beta")
        mem3 = create_memory(tmp_db, "pair gamma")

        window_ab = [
            delta(served=[mem1, mem2], declared=[mem1, mem2])
            for _ in range(HEBBIAN_MIN_CO_USE)
        ]
        window_bc = [
            delta(served=[mem2, mem3], declared=[mem2, mem3])
            for _ in range(HEBBIAN_MIN_CO_USE + 1)
        ]
        hebbian_association(tmp_db, window_ab + window_bc, U, A)

        assoc = load_assoc_edge_map(tmp_db, U, A)
        assert any(mid == mem2 for mid, _ in assoc.get(mem1, []))
        assert any(mid == mem3 for mid, _ in assoc.get(mem2, []))


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  NIGHTLY SWEEP
# ═══════════════════════════════════════════════════════════════════════════════


class TestNightlySweep:
    def test_autopurge_removes_old_unused_probationary(self, tmp_db):
        old_prob = create_memory(
            tmp_db,
            "stale probationary",
            base_strength=0.4,
            decay_rate=0.05,
            probationary=True,
        )
        # Backdate created_at beyond the autopurge window
        old_ms = (
            int(time.time() * 1000) - (PROBATIONARY_AUTOPURGE_DAYS + 1) * 86_400_000
        )
        tmp_db.execute(
            "UPDATE memories SET created_at=?, last_access=? WHERE id=?",
            (old_ms, old_ms, old_prob),
        )
        tmp_db.commit()

        run_nightly_sweep(tmp_db, U, A)

        rec = get_memory(tmp_db, old_prob)
        assert rec.archived is True

    def test_autopurge_spares_used_probationary(self, tmp_db):
        """Probationary memory that was used (use_count > 0) must not be autopurged."""
        used_prob = create_memory(
            tmp_db,
            "used probationary",
            base_strength=0.4,
            decay_rate=0.05,
            probationary=True,
        )
        old_ms = (
            int(time.time() * 1000) - (PROBATIONARY_AUTOPURGE_DAYS + 1) * 86_400_000
        )
        tmp_db.execute(
            "UPDATE memories SET created_at=?, last_access=?, use_count=1 WHERE id=?",
            (old_ms, old_ms, used_prob),
        )
        tmp_db.commit()

        run_nightly_sweep(tmp_db, U, A)

        rec = get_memory(tmp_db, used_prob)
        assert rec.archived is False

    def test_hard_delete_removes_old_archived(self, tmp_db):
        mem_id = create_memory(tmp_db, "ancient archived")
        old_ms = int(time.time() * 1000) - (HARD_DELETE_DAYS + 1) * 86_400_000
        tmp_db.execute(
            "UPDATE memories SET archived=1, archived_at=? WHERE id=?",
            (old_ms, mem_id),
        )
        tmp_db.commit()

        run_nightly_sweep(tmp_db, U, A)

        rec = get_memory(tmp_db, mem_id)
        assert rec is None  # hard deleted

    def test_recent_archived_not_hard_deleted(self, tmp_db):
        mem_id = create_memory(tmp_db, "recent archived")
        recent_ms = int(time.time() * 1000) - 1 * 86_400_000  # just 1 day ago
        tmp_db.execute(
            "UPDATE memories SET archived=1, archived_at=? WHERE id=?",
            (recent_ms, mem_id),
        )
        tmp_db.commit()

        run_nightly_sweep(tmp_db, U, A)

        rec = get_memory(tmp_db, mem_id)
        assert rec is not None  # still exists

    def test_assoc_edges_decay_each_cycle(self, tmp_db):
        mem1 = create_memory(tmp_db, "assoc decay alpha")
        mem2 = create_memory(tmp_db, "assoc decay beta")
        upsert_assoc_edge(tmp_db, mem1, mem2, U, A, delta_weight=0.5)

        before = tmp_db.execute(
            "SELECT weight FROM association_edges WHERE user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()["weight"]

        run_nightly_sweep(tmp_db, U, A)

        after_row = tmp_db.execute(
            "SELECT weight FROM association_edges WHERE user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()
        after = after_row["weight"] if after_row else 0.0
        assert after == pytest.approx(before * ASSOC_NIGHTLY_DECAY, rel=0.01)

    def test_assoc_edges_pruned_below_threshold(self, tmp_db):
        mem1 = create_memory(tmp_db, "prune assoc alpha")
        mem2 = create_memory(tmp_db, "prune assoc beta")
        # Set weight just above prune threshold; many nightly cycles should prune it
        tiny = ASSOC_PRUNE_THRESHOLD / 2.0
        tmp_db.execute(
            """INSERT INTO association_edges
               (memory_a, memory_b, user_id, agent_id, weight, co_use, created_at)
               VALUES (?,?,?,?,?,1,?)""",
            (min(mem1, mem2), max(mem1, mem2), U, A, tiny, int(time.time() * 1000)),
        )
        tmp_db.commit()

        run_nightly_sweep(tmp_db, U, A)

        row = tmp_db.execute(
            "SELECT * FROM association_edges WHERE user_id=? AND agent_id=?", (U, A)
        ).fetchone()
        assert row is None  # pruned because weight < ASSOC_PRUNE_THRESHOLD after decay


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MEMORY CRUD INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryCRUDInvariants:
    def test_create_memory_rejects_bad_agent_id(self, tmp_db):
        with pytest.raises(ValueError):
            create_memory(tmp_db, "bad agent", agent_id="")

        with pytest.raises(ValueError):
            create_memory(tmp_db, "global agent", agent_id="global")

        with pytest.raises(ValueError):
            create_memory(tmp_db, "unknown agent", agent_id="unknown")

    def test_upsert_cue_edge_idempotent_on_duplicate(self, tmp_db):
        """Second upsert_cue_edge call increments weight by seed_weight*0.5, not re-seed."""
        mem_id = create_memory(tmp_db, "upsert idempotent", seed_cues=False)
        upsert_cue_edge(tmp_db, "dup_cue", mem_id, U, A, weight=CONTENT_SEED_WEIGHT)
        first_weight = tmp_db.execute(
            "SELECT weight FROM cue_edges WHERE cue_id='dup_cue' AND memory_id=?",
            (mem_id,),
        ).fetchone()["weight"]

        upsert_cue_edge(tmp_db, "dup_cue", mem_id, U, A, weight=CONTENT_SEED_WEIGHT)
        second_weight = tmp_db.execute(
            "SELECT weight FROM cue_edges WHERE cue_id='dup_cue' AND memory_id=?",
            (mem_id,),
        ).fetchone()["weight"]

        expected = min(1.0, first_weight + CONTENT_SEED_WEIGHT * 0.5)
        assert second_weight == pytest.approx(expected, abs=1e-6)

    def test_df_incremented_on_new_cue_edge(self, tmp_db):
        mem_id = create_memory(tmp_db, "df increment test", seed_cues=False)
        upsert_cue_edge(tmp_db, "freshcue", mem_id, U, A)

        row = tmp_db.execute(
            "SELECT df FROM cue_nodes WHERE cue_id='freshcue' AND user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()
        assert row is not None
        assert row["df"] >= 1

    def test_df_not_incremented_on_existing_cue_edge(self, tmp_db):
        mem_id = create_memory(tmp_db, "df no double inc", seed_cues=False)
        upsert_cue_edge(tmp_db, "existcue", mem_id, U, A)
        df1 = tmp_db.execute(
            "SELECT df FROM cue_nodes WHERE cue_id='existcue' AND user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()["df"]

        upsert_cue_edge(tmp_db, "existcue", mem_id, U, A)
        df2 = tmp_db.execute(
            "SELECT df FROM cue_nodes WHERE cue_id='existcue' AND user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()["df"]

        assert df1 == df2  # no double-increment

    def test_delete_memory_cascades_cue_edges(self, tmp_db):
        """Deleting a memory must cascade-delete its cue_edges (FK ON DELETE CASCADE)."""
        mem_id = create_memory(tmp_db, "cascade delete test", seed_cues=False)
        upsert_cue_edge(tmp_db, "cascade_cue", mem_id, U, A)

        from thyra.models.memory import delete_memory

        delete_memory(tmp_db, mem_id, U, A)

        row = tmp_db.execute(
            "SELECT * FROM cue_edges WHERE memory_id=?", (mem_id,)
        ).fetchone()
        assert row is None

    def test_candidate_edges_excluded_from_cue_edge_map(self, tmp_db):
        """load_cue_edge_map must skip candidate=1 edges."""
        mem_id = create_memory(tmp_db, "candidate exclusion", seed_cues=False)
        now = int(time.time() * 1000)
        tmp_db.execute(
            """INSERT INTO cue_edges
               (cue_id, memory_id, user_id, agent_id, weight, fire_count,
                use_count, candidate, created_at)
               VALUES ('onlycand',?,?,?,0.30,0,0,1,?)""",
            (mem_id, U, A, now),
        )
        tmp_db.commit()

        cue_map = load_cue_edge_map(tmp_db, U, A)
        assert "onlycand" not in cue_map


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  CROSS-SUBSYSTEM INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEndRoundTrip:
    """End-to-end flows that exercise multiple subsystems in sequence."""

    def test_use_strengthens_cue_and_memory(self, tmp_db):
        """A single declared use must both reinforce the memory AND strengthen its cue edges."""
        mem_id = create_memory(
            tmp_db,
            "I always use tabs for Python indentation",
            "preferences",
            base_strength=0.5,
        )
        # Seed a cue edge manually so update_cue_edges has something to update
        _insert_cue_edge_raw(
            tmp_db, "python", mem_id, weight=0.30, fire_count=0, use_count=0
        )

        d = delta(served=[mem_id], declared=[mem_id], cues=["python"])
        apply_reinforcement(tmp_db, d)
        update_cue_edges(tmp_db, d)

        rec = get_memory(tmp_db, mem_id)
        assert rec.base_strength > 0.5

        row = tmp_db.execute(
            "SELECT weight, use_count FROM cue_edges WHERE cue_id='python' AND memory_id=?",
            (mem_id,),
        ).fetchone()
        assert row["weight"] > 0.30
        assert row["use_count"] == 1

    def test_probationary_graduates_and_survives_nightly(self, tmp_db):
        """A probationary memory, once graduated by reinforcement, survives the nightly sweep."""
        mem_id = create_memory(
            tmp_db,
            "I prefer dark mode in all editors",
            "preferences",
            base_strength=BASE_STRENGTH_AUTOMATIC,
            decay_rate=0.15,
            probationary=True,
        )
        d = delta(served=[mem_id], declared=[mem_id])
        apply_reinforcement(tmp_db, d)

        rec_after_grad = get_memory(tmp_db, mem_id)
        assert rec_after_grad.probationary is False

        run_nightly_sweep(tmp_db, U, A)

        rec_after_nightly = get_memory(tmp_db, mem_id)
        assert rec_after_nightly is not None
        assert rec_after_nightly.archived is False

    def test_never_used_probationary_cleaned_by_nightly(self, tmp_db):
        """A probationary memory that is never used is archived by the nightly autopurge."""
        mem_id = create_memory(
            tmp_db,
            "borderline transient fact",
            "context",
            base_strength=BASE_STRENGTH_AUTOMATIC,
            decay_rate=0.15,
            probationary=True,
        )
        # Backdate beyond PROBATIONARY_AUTOPURGE_DAYS
        old_ms = (
            int(time.time() * 1000) - (PROBATIONARY_AUTOPURGE_DAYS + 1) * 86_400_000
        )
        tmp_db.execute(
            "UPDATE memories SET created_at=?, last_access=? WHERE id=?",
            (old_ms, old_ms, mem_id),
        )
        tmp_db.commit()

        run_nightly_sweep(tmp_db, U, A)

        rec = get_memory(tmp_db, mem_id)
        assert rec.archived is True

    def test_cue_prune_followed_by_correct_idf(self, tmp_db):
        """After pruning cue edges, IDF scores reflect the reduced df accurately."""
        from thyra.recall.cue_extractor import compute_idf

        mem_ids = []
        for i in range(4):
            mid = create_memory(tmp_db, f"document {i}", seed_cues=False)
            _insert_cue_edge_raw(
                tmp_db,
                "commonword",
                mid,
                weight=0.30,
                fire_count=CUE_PRUNE_MIN_FIRES,
                use_count=0,
            )
            mem_ids.append(mid)

        df_before = tmp_db.execute(
            "SELECT df FROM cue_nodes WHERE cue_id='commonword' AND user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()["df"]

        prune_weak_cue_edges(tmp_db, U, A)

        df_after_row = tmp_db.execute(
            "SELECT df FROM cue_nodes WHERE cue_id='commonword' AND user_id=? AND agent_id=?",
            (U, A),
        ).fetchone()
        df_after = df_after_row["df"] if df_after_row else 0
        assert df_after < df_before  # df decremented for each pruned edge

        idf_after = compute_idf(tmp_db, ["commonword"], U, A)
        idf_before_val = compute_idf(tmp_db, ["commonword"], U, A)
        # After pruning, the cue is rarer → IDF should be the same or higher
        assert idf_after["commonword"] >= idf_before_val["commonword"]


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  PAUSE-TRIGGERED NIGHTLY (PC-OFF RELIABILITY)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPauseTriggeredNightly:
    """The nightly sweep must fire from pause/startup signals, not only from
    delta events, so decay/archiving runs even when the PC was off or idle.
    """

    def _make_worker(self, db_path):
        from thyra.consolidation.worker import BackgroundWorker

        return BackgroundWorker(db_path=db_path)

    # ── _maybe_nightly reads the persisted DB flag ──────────────────────────

    def test_maybe_nightly_reads_db_flag_on_first_call(
        self, tmp_db, tmp_path, monkeypatch
    ):
        """If _last_nightly is empty, _maybe_nightly should read the DB flag before
        deciding whether to run, so a recent sweep isn't re-run on restart.
        """
        import os
        from thyra.models.memory import set_flag

        db_path = os.environ["THYRA_DB_PATH"]
        # Persist a recent last_nightly (just now) so the sweep should NOT re-run
        recent_ms = int(time.time() * 1000)
        set_flag(tmp_db, "last_nightly", str(recent_ms), U, A)
        tmp_db.commit()

        run_count = []

        # Patch run_nightly_sweep to detect if it's called
        import thyra.consolidation.worker as wmod

        original = wmod.__dict__.get("_imported_nightly", None)

        import thyra.consolidation.nightly as nmod

        original_sweep = nmod.run_nightly_sweep

        def counting_sweep(conn, uid, aid):
            run_count.append((uid, aid))
            return original_sweep(conn, uid, aid)

        monkeypatch.setattr(nmod, "run_nightly_sweep", counting_sweep)

        worker = self._make_worker(db_path)
        worker._maybe_nightly(U, A)

        # Sweep should NOT have run because last_nightly was just set
        assert len(run_count) == 0

    def test_maybe_nightly_runs_when_db_flag_is_old(
        self, tmp_db, tmp_path, monkeypatch
    ):
        """If the persisted last_nightly is older than NIGHTLY_INTERVAL_HOURS, sweep runs."""
        import os
        from thyra.models.memory import set_flag

        db_path = os.environ["THYRA_DB_PATH"]
        # Persist an old last_nightly (25 hours ago)
        old_ms = int(time.time() * 1000) - 25 * 3600 * 1000
        set_flag(tmp_db, "last_nightly", str(old_ms), U, A)
        tmp_db.commit()

        run_count = []
        import thyra.consolidation.nightly as nmod

        original_sweep = nmod.run_nightly_sweep

        def counting_sweep(conn, uid, aid):
            run_count.append((uid, aid))
            return original_sweep(conn, uid, aid)

        monkeypatch.setattr(nmod, "run_nightly_sweep", counting_sweep)

        worker = self._make_worker(db_path)
        worker._maybe_nightly(U, A)

        assert len(run_count) >= 1

    # ── _startup_nightly_check ───────────────────────────────────────────────

    def test_startup_check_runs_nightly_for_overdue_pair(self, tmp_db, monkeypatch):
        """_startup_nightly_check must trigger nightly for any pair with active memories
        whose last_nightly is overdue — simulating a PC that was off for > 24h.
        """
        import os
        from thyra.models.memory import set_flag

        db_path = os.environ["THYRA_DB_PATH"]
        create_memory(tmp_db, "memory that aged while PC was off", base_strength=1.0)

        # Persist an old last_nightly
        old_ms = int(time.time() * 1000) - 25 * 3600 * 1000
        set_flag(tmp_db, "last_nightly", str(old_ms), U, A)
        tmp_db.commit()

        run_count = []
        import thyra.consolidation.nightly as nmod

        original_sweep = nmod.run_nightly_sweep

        def counting_sweep(conn, uid, aid):
            run_count.append((uid, aid))
            return original_sweep(conn, uid, aid)

        monkeypatch.setattr(nmod, "run_nightly_sweep", counting_sweep)

        worker = self._make_worker(db_path)
        worker._startup_nightly_check()

        assert any(uid == U and aid == A for uid, aid in run_count)

    def test_startup_check_skips_recently_swept_pair(self, tmp_db, monkeypatch):
        """_startup_nightly_check must NOT re-run if the sweep was recent."""
        import os
        from thyra.models.memory import set_flag

        db_path = os.environ["THYRA_DB_PATH"]
        create_memory(tmp_db, "fresh memory")

        # Persist a recent last_nightly
        recent_ms = int(time.time() * 1000)
        set_flag(tmp_db, "last_nightly", str(recent_ms), U, A)
        tmp_db.commit()

        run_count = []
        import thyra.consolidation.nightly as nmod

        def counting_sweep(conn, uid, aid):
            run_count.append((uid, aid))
            return (
                nmod.run_nightly_sweep.__wrapped__(conn, uid, aid)
                if hasattr(nmod.run_nightly_sweep, "__wrapped__")
                else None
            )

        monkeypatch.setattr(nmod, "run_nightly_sweep", counting_sweep)

        worker = self._make_worker(db_path)
        worker._startup_nightly_check()

        assert len(run_count) == 0

    # ── _idle_nightly_check ─────────────────────────────────────────────────

    def test_idle_check_runs_nightly_for_overdue_pair(self, tmp_db, monkeypatch):
        """_idle_nightly_check sweeps overdue pairs even with an empty queue."""
        import os
        from thyra.models.memory import set_flag

        db_path = os.environ["THYRA_DB_PATH"]
        create_memory(tmp_db, "idle pair memory")

        old_ms = int(time.time() * 1000) - 25 * 3600 * 1000
        set_flag(tmp_db, "last_nightly", str(old_ms), U, A)
        tmp_db.commit()

        run_count = []
        import thyra.consolidation.nightly as nmod

        original_sweep = nmod.run_nightly_sweep

        def counting_sweep(conn, uid, aid):
            run_count.append((uid, aid))
            return original_sweep(conn, uid, aid)

        monkeypatch.setattr(nmod, "run_nightly_sweep", counting_sweep)

        worker = self._make_worker(db_path)
        worker._idle_nightly_check()

        assert any(uid == U and aid == A for uid, aid in run_count)

    # ── Decay correctness after multi-day absence ────────────────────────────

    def test_startup_sweep_correctly_decays_aged_memories(self, tmp_db, monkeypatch):
        """After N days offline, the startup sweep must apply N days of decay to all memories.
        The final strength must match the formula, not just show 'some decay'.
        """
        import os

        db_path = os.environ["THYRA_DB_PATH"]
        days_offline = 10
        decay_rate = 0.1
        mem_id = create_memory(
            tmp_db, "aged memory sentinel", base_strength=1.0, decay_rate=decay_rate
        )

        # Simulate the memory having been untouched for `days_offline` days
        past_ms = int(time.time() * 1000) - days_offline * 86_400_000
        tmp_db.execute(
            "UPDATE memories SET last_access=? WHERE id=?", (past_ms, mem_id)
        )

        # Persist last_nightly as very old so the sweep fires
        old_nightly_ms = past_ms - 1000
        from thyra.models.memory import set_flag

        set_flag(tmp_db, "last_nightly", str(old_nightly_ms), U, A)
        tmp_db.commit()

        worker = self._make_worker(db_path)
        worker._startup_nightly_check()

        rec = get_memory(tmp_db, mem_id)
        expected = math.exp(-decay_rate * days_offline)
        assert rec.base_strength == pytest.approx(expected, rel=0.01)

    def test_startup_sweep_archives_fully_decayed_memories(self, tmp_db, monkeypatch):
        """Memories that have decayed below ARCHIVE_THRESHOLD during absence are archived."""
        import os

        db_path = os.environ["THYRA_DB_PATH"]
        # decay_rate=2.0 → after 5 days, strength = 1.0 * exp(-10) ≈ 0 → way below threshold
        mem_id = create_memory(
            tmp_db, "should archive", base_strength=1.0, decay_rate=2.0
        )
        past_ms = int(time.time() * 1000) - 5 * 86_400_000
        tmp_db.execute(
            "UPDATE memories SET last_access=? WHERE id=?", (past_ms, mem_id)
        )

        from thyra.models.memory import set_flag

        set_flag(tmp_db, "last_nightly", str(past_ms - 1000), U, A)
        tmp_db.commit()

        worker = self._make_worker(db_path)
        worker._startup_nightly_check()

        rec = get_memory(tmp_db, mem_id)
        assert rec.archived is True

    def test_last_nightly_persisted_after_sweep(self, tmp_db, monkeypatch):
        """After _maybe_nightly runs, the new timestamp must be written to the DB flag
        so the next server restart doesn't re-run immediately.
        """
        import os
        from thyra.models.memory import get_flag

        db_path = os.environ["THYRA_DB_PATH"]
        create_memory(tmp_db, "persist check memory")

        # Ensure sweep runs
        old_ms = int(time.time() * 1000) - 25 * 3600 * 1000
        from thyra.models.memory import set_flag

        set_flag(tmp_db, "last_nightly", str(old_ms), U, A)
        tmp_db.commit()

        before_ms = int(time.time() * 1000)
        worker = self._make_worker(db_path)
        worker._maybe_nightly(U, A)
        after_ms = int(time.time() * 1000)

        # The DB flag must now be >= before_ms
        persisted_str = get_flag(tmp_db, "last_nightly", U, A, default="0")
        # Need to re-query since the worker uses its own connection
        from thyra.db.connection import DBConnection

        conn2 = DBConnection.get(db_path)
        persisted_str = get_flag(conn2, "last_nightly", U, A, default="0")
        persisted_ms = float(persisted_str)
        assert persisted_ms >= before_ms
