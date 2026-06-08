"""Stage-7 category system tests."""

import time
import pytest

from thyra.config import (
    THYRA_USER_ID as U,
    THYRA_AGENT_ID as A,
    CATEGORY_SOFT_CAP,
)
from thyra.categories.taxonomy import (
    PROTECTED_CATEGORIES,
    SEED_CATEGORIES,
    CATEGORY_DECAY_RATES,
    CATEGORY_RELEVANCE_FLOORS,
)
from thyra.categories.manager import CategoryManager
from thyra.categories.crystallizer import (
    _find_hub_cues,
    _connected_components,
    detect_emergent_categories,
)
from thyra.models.memory import create_memory, upsert_cue_edge, upsert_assoc_edge


class TestTaxonomy:
    def test_seed_categories_count(self):
        assert len(SEED_CATEGORIES) == 15

    def test_protected_set(self):
        assert "constraints" in PROTECTED_CATEGORIES
        assert "identity" in PROTECTED_CATEGORIES
        assert "tasks" not in PROTECTED_CATEGORIES

    def test_slow_decay_for_constraints(self):
        assert CATEGORY_DECAY_RATES["constraints"] <= 0.001

    def test_relevance_floor_ordering(self):
        # constraints > identity > preferences > routines floor
        assert (
            CATEGORY_RELEVANCE_FLOORS["constraints"]
            > CATEGORY_RELEVANCE_FLOORS["preferences"]
        )
        assert (
            CATEGORY_RELEVANCE_FLOORS["identity"] > CATEGORY_RELEVANCE_FLOORS["tasks"]
        )

    def test_all_seeds_have_decay_rate(self):
        for cat in SEED_CATEGORIES:
            assert cat in CATEGORY_DECAY_RATES, (
                f"{cat} missing from CATEGORY_DECAY_RATES"
            )


class TestCategoryManager:
    def test_get_weight_protected_category(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        w = mgr.get_weight("constraints")
        assert w >= 0.40  # protected categories have higher floors

    def test_get_weight_unknown_category(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        w = mgr.get_weight("nonexistent_cat_xyz")
        assert w >= 0.0

    def test_get_all_weights_returns_all_seeded(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        weights = mgr.get_all_weights()
        # All 15 seed categories should be present
        for cat in SEED_CATEGORIES:
            assert cat in weights, f"{cat} missing from get_all_weights()"

    def test_get_decay_rate(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        assert mgr.get_decay_rate("constraints") == pytest.approx(0.001)
        assert mgr.get_decay_rate("preferences") == pytest.approx(0.02)

    def test_soft_cap_no_action_when_under_cap(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        # 15 seed categories, well under cap of 35
        dissolved = mgr.soft_cap_enforce()
        assert dissolved == []

    def test_soft_cap_dissolves_emergent_when_over_cap(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        # Add enough emergent categories to exceed cap
        for i in range(CATEGORY_SOFT_CAP - 15 + 3):
            mgr.add_emergent_category(f"emergent_{i}", decay_rate=0.02)
        dissolved = mgr.soft_cap_enforce()
        assert len(dissolved) == 3

    def test_soft_cap_never_dissolves_protected(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        # Add emergent cats beyond cap
        for i in range(CATEGORY_SOFT_CAP - 15 + 5):
            mgr.add_emergent_category(f"emergent_protect_{i}")
        dissolved = mgr.soft_cap_enforce()
        # None of the protected categories should be dissolved
        for cat in PROTECTED_CATEGORIES:
            assert cat not in dissolved

    def test_add_emergent_category(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        mgr.add_emergent_category("my_custom_topic")
        weights = mgr.get_all_weights()
        assert "my_custom_topic" in weights

    def test_add_emergent_idempotent(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        mgr.add_emergent_category("duplicate_topic")
        mgr.add_emergent_category("duplicate_topic")  # should not raise
        weights = mgr.get_all_weights()
        assert "duplicate_topic" in weights

    def test_dissolved_category_renames_memories(self, tmp_db):
        mgr = CategoryManager(tmp_db, U, A)
        mgr.add_emergent_category("doomed_cat")
        mem_id = create_memory(tmp_db, "This memory will be rehomed", "doomed_cat")
        # Force dissolution
        mgr._dissolve_category("doomed_cat")
        row = tmp_db.execute(
            "SELECT category FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert row["category"] == "context"


class TestCrystallizer:
    def test_find_hub_cues_empty(self, tmp_db):
        hubs = _find_hub_cues(tmp_db, U, A)
        assert hubs == set()

    def test_find_hub_cues_filters_common(self, tmp_db):
        # Create 4 memories and link a cue to all of them (>50% = hub)
        ids = []
        for i in range(4):
            mid = create_memory(tmp_db, f"memory content {i}")
            ids.append(mid)
        for mid in ids:
            upsert_cue_edge(tmp_db, "common_cue", mid, U, A, weight=0.3)
        hubs = _find_hub_cues(tmp_db, U, A)
        # common_cue links to 4/4 = 100% > 50% threshold
        assert "common_cue" in hubs

    def test_connected_components_empty(self):
        result = _connected_components({}, set())
        assert result == []

    def test_connected_components_simple(self):
        adj = {"a": {"b"}, "b": {"a"}, "c": {"d"}, "d": {"c"}, "e": set()}
        components = _connected_components(adj, {"a", "b", "c", "d", "e"})
        # "e" is isolated (size 1, filtered out)
        sizes = sorted(len(c) for c in components)
        assert sizes == [2, 2]

    def test_detect_emergent_returns_list(self, tmp_db):
        # With <4 memories, no community can qualify
        create_memory(tmp_db, "memory one")
        create_memory(tmp_db, "memory two")
        result = detect_emergent_categories(tmp_db, U, A)
        assert isinstance(result, list)

    def test_detect_emergent_creates_category(self, tmp_db):
        # Create memories connected by strong association edges
        ids = []
        for i in range(5):
            mid = create_memory(tmp_db, f"related memory cluster {i}", "skills")
            ids.append(mid)

        # Link them with assoc_edges above threshold (0.30)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                upsert_assoc_edge(tmp_db, ids[i], ids[j], U, A, delta_weight=0.35)

        # Fabricate turn_log entries to meet EMERGENT_MIN_TURNS (10)
        import json

        ts = int(time.time() * 1000)
        for t in range(12):
            tmp_db.execute(
                """INSERT OR IGNORE INTO turn_log
                   (turn_id, session_id, user_id, agent_id, memories_served, memories_used, cues_fired, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    f"t_{t}",
                    "s1",
                    U,
                    A,
                    json.dumps(ids[:2]),
                    json.dumps(ids[:2]),
                    "[]",
                    ts + t,
                ),
            )
        tmp_db.commit()

        new_cats = detect_emergent_categories(tmp_db, U, A)
        # A community of 5 with 12 co-activation turns should produce ≥1 category
        assert len(new_cats) >= 1
