"""Stage-2 recall engine tests."""

import time
import pytest

from thyra.models.memory import create_memory, upsert_cue_edge, upsert_assoc_edge
from thyra.recall.cue_extractor import extract_cues, compute_idf
from thyra.recall.morphology import normalize_cue
from thyra.recall.cache import HotCache
from thyra.recall.scorer import score_memories
from thyra.recall.selector import greedy_select
from thyra.recall.injector import format_injection
from thyra.recall.intent import detect_recall_intent, recall_pipeline
from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A


# ── Cue extraction tests ───────────────────────────────────────────────────────


class TestCueExtraction:
    def test_basic_extraction(self):
        cues = extract_cues("I prefer Python programming language")
        assert "python" in cues
        assert "program" in cues  # NLTK Porter: "programming" → "program"

    def test_stopwords_removed(self):
        cues = extract_cues("that which should have been done")
        # All words are stopwords or short
        assert len(cues) == 0

    def test_short_words_excluded(self):
        # MIN_CUE_LENGTH=3: 3-char content words are now valid cues
        cues = extract_cues("The sky was red")
        assert "sky" in cues
        assert "red" in cues
        # 2-char words are still excluded (regex requires ≥3 chars)
        cues2 = extract_cues("go do it")
        assert len(cues2) == 0

    def test_max_cues_respected(self):
        text = " ".join([f"keyword{i}word" for i in range(50)])
        cues = extract_cues(text, max_cues=12)
        assert len(cues) <= 12

    def test_frequency_ranking(self):
        cues = extract_cues("python python python java java ruby", max_cues=3)
        assert cues[0] == "python"  # highest frequency first


class TestMorphology:
    def test_plural_normalized(self):
        # NLTK Porter stems "preferences" → "prefer"
        result = normalize_cue("preferences")
        assert result in ("prefer", "preference")  # accept either stemmer output

    def test_ing_stripped(self):
        assert normalize_cue("programming") == "program"

    def test_irregular_plural(self):
        assert normalize_cue("criteria") == "criterion"

    def test_no_change_for_normal(self):
        assert normalize_cue("python") == "python"

    def test_ies_ending(self):
        # Porter may produce "categori" or similar; key is it's shorter than "categories"
        result = normalize_cue("categories")
        assert len(result) < len("categories")


class TestIDF:
    def test_idf_rare_cue_scores_high(self, tmp_db):
        create_memory(tmp_db, "python programming unique fact", seed_cues=True)
        create_memory(tmp_db, "java spring boot framework", seed_cues=True)
        idf = compute_idf(tmp_db, ["python"], U, A)
        assert idf["python"] > 0.5

    def test_idf_floor_for_hub_cue(self, tmp_db):
        # Create memories all linked to same cue
        for i in range(5):
            mem_id = create_memory(tmp_db, f"fact {i}", seed_cues=False)
            upsert_cue_edge(tmp_db, "common", mem_id, U, A)
        idf = compute_idf(tmp_db, ["common"], U, A)
        assert idf["common"] >= 0.15  # at floor
        assert idf["common"] < 0.5  # not high

    def test_idf_unknown_cue_returns_floor(self, tmp_db):
        create_memory(tmp_db, "some memory here today")
        idf = compute_idf(tmp_db, ["unknowncue"], U, A)
        assert idf["unknowncue"] >= 0.15


class TestHotCache:
    def test_set_and_get(self):
        cache = HotCache(ttl=60)
        cache.set("key1", {"data": 42})
        assert cache.get("key1") == {"data": 42}

    def test_miss_returns_none(self):
        cache = HotCache()
        assert cache.get("nonexistent") is None

    def test_invalidate(self):
        cache = HotCache()
        cache.set("k", 1)
        cache.invalidate("k")
        assert cache.get("k") is None

    def test_ttl_expiry(self):
        import time as t

        cache = HotCache(ttl=1)
        cache.set("x", 99)
        # Manually set the expiry to the past by manipulating the entry
        cache._store["x"].expires_at = t.monotonic() - 1
        assert cache.get("x") is None


class TestScoring:
    def test_cue_hit_raises_score(self, tmp_db):
        mem_id = create_memory(
            tmp_db, "Python is great for data science", seed_cues=False
        )
        upsert_cue_edge(tmp_db, "python", mem_id, U, A, weight=0.5)

        from thyra.models.memory import (
            list_active_memories,
            load_cue_edge_map,
            load_assoc_edge_map,
            load_situation_edges,
        )

        memories = list_active_memories(tmp_db, U, A)
        cue_map = load_cue_edge_map(tmp_db, U, A)
        assoc_map = load_assoc_edge_map(tmp_db, U, A)
        sit_edges = load_situation_edges(tmp_db, U, A)
        idf = {"python": 0.9}
        now = int(time.time() * 1000)

        scored = score_memories(
            memories, ["python"], cue_map, assoc_map, sit_edges, idf, {}, now
        )
        assert len(scored) > 0
        assert scored[0][0].id == mem_id
        assert scored[0][1] > 0.1

    def test_no_cue_hit_low_score(self, tmp_db):
        mem_id = create_memory(tmp_db, "Completely unrelated memory", seed_cues=False)
        from thyra.models.memory import (
            list_active_memories,
            load_cue_edge_map,
            load_assoc_edge_map,
            load_situation_edges,
        )

        memories = list_active_memories(tmp_db, U, A)
        cue_map = load_cue_edge_map(tmp_db, U, A)
        assoc_map = {}
        sit_edges = []
        idf = {"python": 0.9}
        now = int(time.time() * 1000)

        scored = score_memories(
            memories, ["python"], cue_map, assoc_map, sit_edges, idf, {}, now
        )
        # Memory has no "python" cue edge so spreading=0.
        # Score = base_level * PRESENCE_FLOOR * recency_mult.
        # With RECENCY_BOOST_MAX=0.80 a brand-new memory gets recency_mult=1.80,
        # giving max score = 1.0 * 0.20 * 1.80 = 0.36. Threshold updated accordingly.
        python_score = next((s for r, s in scored if r.id == mem_id), 0)
        assert python_score < 0.40

    def test_strong_memory_scores_higher_than_weak(self, tmp_db):
        strong = create_memory(
            tmp_db, "strong memory", base_strength=1.0, seed_cues=False
        )
        weak = create_memory(tmp_db, "weak memory", base_strength=0.1, seed_cues=False)
        for mid in (strong, weak):
            upsert_cue_edge(tmp_db, "testcue", mid, U, A, weight=0.5)

        from thyra.models.memory import (
            list_active_memories,
            load_cue_edge_map,
            load_assoc_edge_map,
            load_situation_edges,
        )

        memories = list_active_memories(tmp_db, U, A)
        cue_map = load_cue_edge_map(tmp_db, U, A)
        idf = {"testcue": 0.8}
        now = int(time.time() * 1000)
        scored = score_memories(memories, ["testcue"], cue_map, {}, [], idf, {}, now)

        scores = {r.id: s for r, s in scored}
        assert scores[strong] > scores[weak]


class TestSelection:
    def test_respects_token_budget(self, tmp_db):
        for i in range(20):
            create_memory(tmp_db, f"Memory number {i} with some content here " * 5)
        from thyra.models.memory import (
            list_active_memories,
            load_cue_edge_map,
            load_assoc_edge_map,
            load_situation_edges,
        )

        memories = list_active_memories(tmp_db, U, A)
        now = int(time.time() * 1000)
        scored = [(r, r.base_strength) for r in memories]
        selected = greedy_select(scored, token_budget=200)
        total_tokens = sum(len(r.content) // 4 for r in selected)
        assert total_tokens <= 200

    def test_protected_categories_not_starved(self, tmp_db):
        for i in range(10):
            create_memory(tmp_db, f"Regular memory {i}", category="knowledge")
        constraint_id = create_memory(
            tmp_db, "Never discuss competitors", category="constraints"
        )
        from thyra.models.memory import list_active_memories

        memories = list_active_memories(tmp_db, U, A)
        scored = [(r, 0.9 if r.category == "knowledge" else 0.5) for r in memories]
        selected = greedy_select(scored, token_budget=2000, gamma=0.3)
        selected_ids = {r.id for r in selected}
        assert constraint_id in selected_ids


class TestInjector:
    def test_injection_contains_memory_ids(self, tmp_db):
        mem_id = create_memory(tmp_db, "My preferred editor is Neovim", "preferences")
        from thyra.models.memory import get_memory

        rec = get_memory(tmp_db, mem_id)
        xml = format_injection([rec], A)
        assert mem_id in xml
        assert "Neovim" in xml
        assert "<thyra_memories" in xml
        assert "</thyra_memories>" in xml

    def test_locked_memory_shows_placeholder(self, tmp_db):
        mem_id = create_memory(tmp_db, "Secret content", locked=True, seed_cues=False)
        from thyra.models.memory import get_memory

        rec = get_memory(tmp_db, mem_id)
        xml = format_injection([rec], A)
        assert "LOCKED" in xml
        assert "Secret content" not in xml

    def test_empty_list_returns_empty_string(self):
        assert format_injection([], A) == ""


class TestRecallIntent:
    def test_detects_remember_when(self):
        assert detect_recall_intent("remember when we discussed that feature?")

    def test_detects_did_we_discuss(self):
        assert detect_recall_intent("Did we discuss this approach before?")

    def test_detects_you_told_me(self):
        assert detect_recall_intent("You told me to use tabs, right?")

    def test_does_not_trigger_on_remember_directive(self):
        # "remember that" as directive should NOT trigger recall-intent
        # (disambiguation: imperative + fact is store, not retrieve)
        assert not detect_recall_intent("remember that I prefer Python")

    def test_no_intent_on_normal_prompt(self):
        assert not detect_recall_intent("How do I sort a list in Python?")


class TestRecallPipeline:
    def test_returns_empty_when_no_memories(self, tmp_db):
        xml, served = recall_pipeline(tmp_db, U, A, "Python programming", "s1", "t1")
        assert xml == ""
        assert served == []

    def test_returns_relevant_memory(self, tmp_db):
        mem_id = create_memory(tmp_db, "I work with Python every day", "knowledge")
        upsert_cue_edge(tmp_db, "python", mem_id, U, A, weight=0.8)
        xml, served = recall_pipeline(
            tmp_db, U, A, "Tell me about python projects", "s1", "t1"
        )
        assert mem_id in served
        assert "python" in xml.lower() or mem_id in xml

    def test_disabled_master_switch_returns_empty(self, tmp_db):
        from thyra.models.memory import set_flag

        set_flag(tmp_db, "system_enabled", "false")
        create_memory(tmp_db, "Some memory")
        xml, served = recall_pipeline(
            tmp_db, U, A, "anything python related", "s1", "t1"
        )
        assert xml == ""
        assert served == []

    def test_pipeline_never_raises(self, tmp_db):
        # Corrupt call that would normally crash
        try:
            result = recall_pipeline(None, U, A, "test", "s", "t")  # type: ignore
            assert result == ("", [])
        except Exception:
            pytest.fail("recall_pipeline raised an exception — it must never raise")
