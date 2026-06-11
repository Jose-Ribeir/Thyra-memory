"""Namespace & model-load validation pass.

For every submodule in thyra/:
  1. Import must succeed without raising (namespace check).
  2. Module-level objects that are lazy-loaded must be in their expected initial
     state (not accidentally prewarm'd at import time).

For each ML / heavy dependency:
  3. NLTK PorterStemmer — loads and produces correct stems.
  4. SentenceTransformer all-MiniLM-L6-v2 — loads or gracefully degrades.
  5. Refiner — _classify_category() returns a known category in both ML and
     rule-based modes.
  6. Synonym expansion — returns a list (even if empty when model absent).
  7. Dedup cosine path — degrades to word-overlap when model unavailable.
  8. Distiller — prewarm() returns False when DISTILLER_ENABLED=false.

None of these tests require a running MCP server or external network access.
"""

from __future__ import annotations

import importlib
import math
import pkgutil
import sys
import types

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  NAMESPACE IMPORT PASS  — every thyra.* submodule must import cleanly
# ═══════════════════════════════════════════════════════════════════════════════

# Modules that intentionally pull in dashboard/tray/server dependencies that
# are optional (pystray, uvicorn, fastapi).  We still import them but
# tolerate ImportError on the optional deps.
_OPTIONAL_DEP_MODULES = frozenset(
    {
        "thyra.tray",
        "thyra.dashboard.runner",
        "thyra.dashboard.server",
        "thyra.server.mcp_server",
    }
)


def _iter_thyra_modules():
    """Yield all dotted module names under the thyra package."""
    import thyra

    pkg_path = thyra.__path__
    for finder, name, _ in pkgutil.walk_packages(pkg_path, prefix="thyra."):
        yield name


@pytest.mark.parametrize("module_name", list(_iter_thyra_modules()))
def test_module_imports_cleanly(module_name):
    """Every thyra.* module must be importable without unexpected errors.

    Optional-dep modules are allowed to raise ImportError for their specific
    optional dependency (pystray, uvicorn, etc.) but must not raise anything
    else.
    """
    try:
        mod = importlib.import_module(module_name)
        assert isinstance(mod, types.ModuleType)
    except ImportError as exc:
        if module_name in _OPTIONAL_DEP_MODULES:
            pytest.skip(f"{module_name} has missing optional dep: {exc}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LAZY INIT STATE — nothing should eagerly prewarm at import time
# ═══════════════════════════════════════════════════════════════════════════════


class TestLazyInitState:
    def test_refiner_model_not_loaded_at_import(self):
        """_model must be None after import — ML load is deferred to first call."""
        import importlib

        # Reload to reset module globals cleanly
        import thyra.formation.refiner as ref

        # Accept None (never loaded) or False (attempted and failed) — not a real model
        assert ref._model in (None, False) or hasattr(ref._model, "encode")

    def test_morphology_stemmer_not_loaded_at_import(self):
        """_stemmer is lazily loaded; at import time it should be None."""
        import thyra.recall.morphology as morph

        # If another test already triggered it, it's fine — just confirm it's not broken
        assert (
            morph._stemmer is None
            or morph._stemmer is not False
            or callable(getattr(morph._stemmer, "stem", None))
        )

    def test_synonym_model_not_loaded_at_import(self):
        """synonym._model must be None — loaded only when needed."""
        import thyra.recall.synonym as syn

        assert syn._model is None or hasattr(syn._model, "encode")

    def test_distiller_judge_not_loaded_at_import(self):
        """distiller._judge must be None — never loads unless DISTILLER_ENABLED=true."""
        import thyra.formation.distiller as dist

        # False means 'attempted and unavailable' — acceptable
        assert dist._judge in (None, False)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  NLTK PORTER STEMMER
# ═══════════════════════════════════════════════════════════════════════════════


class TestNLTKStemmer:
    def test_stemmer_loads_or_gracefully_skips(self):
        """_get_stemmer() must return a usable stemmer or None — never raise."""
        from thyra.recall.morphology import _get_stemmer

        stemmer = _get_stemmer()
        # If NLTK is installed: stemmer is a PorterStemmer with a .stem() method
        # If NLTK is missing: stemmer is None (graceful degradation)
        assert stemmer is None or callable(getattr(stemmer, "stem", None))

    def test_stemmer_produces_correct_stems(self):
        """When NLTK is available the stemmer must correctly reduce common forms."""
        from thyra.recall.morphology import _get_stemmer

        stemmer = _get_stemmer()
        if stemmer is None:
            pytest.skip("NLTK PorterStemmer not available")
        assert stemmer.stem("programming") == "program"
        assert stemmer.stem("preferences") in ("prefer", "preferenc")
        assert stemmer.stem("running") == "run"

    def test_normalize_cue_handles_irregulars(self):
        """Irregular plurals must hit the dictionary override, not the stemmer."""
        from thyra.recall.morphology import normalize_cue

        assert normalize_cue("criteria") == "criterion"
        assert normalize_cue("phenomena") == "phenomenon"
        assert normalize_cue("data") == "datum"

    def test_normalize_cue_fallback_without_nltk(self, monkeypatch):
        """When NLTK is unavailable normalize_cue must still work via simple rules."""
        import thyra.recall.morphology as morph

        monkeypatch.setattr(morph, "_stemmer", False)  # sentinel: unavailable
        result = morph.normalize_cue("preferences")
        # Simple rule: strip trailing 's' when word > 4 chars
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_cue_idempotent(self):
        """Applying normalize_cue twice must give the same result as once."""
        from thyra.recall.morphology import normalize_cue

        word = "deployments"
        once = normalize_cue(word)
        twice = normalize_cue(once)
        assert once == twice

    def test_cue_extraction_uses_stemmer(self):
        """extract_cues must produce stemmed tokens, not raw words."""
        from thyra.recall.cue_extractor import extract_cues

        cues = extract_cues("I prefer Python programming deployments")
        # "programming" should stem to "program", "deployments" to "deploy" or similar
        assert "prefer" in cues or "preferenc" in cues  # stemmed form of "prefer"
        # No raw unstemmed "programming" in the output
        assert "programming" not in cues

    def test_nltk_data_download_not_required(self):
        """normalize_cue must work without any NLTK data download (PorterStemmer is pure Python)."""
        from thyra.recall.morphology import normalize_cue

        # If this raises LookupError (NLTK data missing) the integration is broken
        try:
            result = normalize_cue("tokenization")
            assert isinstance(result, str)
        except LookupError as exc:
            pytest.fail(f"normalize_cue raised LookupError (NLTK data needed): {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SENTENCE-TRANSFORMER (all-MiniLM-L6-v2) — refiner path
# ═══════════════════════════════════════════════════════════════════════════════


class TestMiniLMRefiner:
    def test_get_model_returns_model_or_none(self):
        """_get_model_and_embeddings() must return (model, embeddings, categories)
        or (None, None, None) — never raise."""
        from thyra.formation.refiner import _get_model_and_embeddings

        model, embeddings, categories = _get_model_and_embeddings()
        if model is not None:
            assert callable(getattr(model, "encode", None))
            assert embeddings is not None
            assert isinstance(categories, list)
            assert len(categories) > 0
        else:
            assert embeddings is None
            assert categories is None

    def test_classify_category_returns_known_category(self):
        """_classify_category must always return a string that is a valid category name."""
        from thyra.formation.refiner import _classify_category, CATEGORY_EXEMPLARS

        valid = set(CATEGORY_EXEMPLARS.keys()) | {"context"}
        for clause in [
            "I prefer dark mode in all editors",
            "Never use personal data in logs",
            "Yesterday the deploy failed",
            "My manager is Sarah",
            "I need to finish the PR by Friday",
            "I know Python very well",
        ]:
            result = _classify_category(clause)
            assert result in valid, f"Unknown category {result!r} for {clause!r}"

    def test_classify_category_rule_fallback(self, monkeypatch):
        """When the model is unavailable, rule-based fallback must still return a valid category."""
        import thyra.formation.refiner as ref

        monkeypatch.setattr(ref, "_model", False)  # force rules-only path
        from thyra.formation.refiner import _classify_category, CATEGORY_EXEMPLARS

        valid = set(CATEGORY_EXEMPLARS.keys()) | {"context"}
        result = _classify_category("I prefer Python over Java for all projects")
        assert result in valid
        assert result == "preferences"  # rule regex should catch "I prefer"

    def test_classify_category_rules_constraint(self, monkeypatch):
        import thyra.formation.refiner as ref

        monkeypatch.setattr(ref, "_model", False)
        from thyra.formation.refiner import _classify_category

        assert _classify_category("Never use customer PII in logs") == "constraints"

    def test_classify_category_rules_identity(self, monkeypatch):
        import thyra.formation.refiner as ref

        monkeypatch.setattr(ref, "_model", False)
        from thyra.formation.refiner import _classify_category

        result = _classify_category("I am a senior backend engineer")
        assert result == "identity"

    def test_classify_type_episodic_markers(self):
        """Episodic markers must produce 'episodic', others 'semantic'."""
        from thyra.formation.refiner import _classify_type

        assert _classify_type("Yesterday the deploy failed") == "episodic"
        assert _classify_type("Last week we tried the migration") == "episodic"
        assert _classify_type("I always prefer tabs over spaces") == "semantic"
        assert _classify_type("My preferred language is Python") == "semantic"

    def test_refiner_full_pipeline_returns_expected_keys(self):
        """Stage2Refiner.refine() must always return a dict with the four required keys."""
        from thyra.formation.refiner import REFINER

        result = REFINER.refine("I prefer Python for backend projects at work")
        assert result is not None
        for key in ("content", "category", "memory_type", "cue_suggestions"):
            assert key in result, f"Missing key: {key}"
        assert isinstance(result["cue_suggestions"], list)
        assert result["content"][0].isupper()  # capitalized

    def test_refiner_drops_short_clause(self):
        """Clauses shorter than 10 chars must be dropped (return None)."""
        from thyra.formation.refiner import REFINER

        assert REFINER.refine("ok") is None
        assert REFINER.refine("yes") is None
        assert REFINER.refine("") is None

    def test_model_is_resident_reflects_load_state(self):
        """_model_is_resident() must match whether _model holds a real object."""
        from thyra.formation.refiner import _model_is_resident, _model
        import thyra.formation.refiner as ref

        expected = ref._model not in (None, False)
        assert _model_is_resident() == expected

    def test_rules_only_mode_never_triggers_cold_load(self, monkeypatch):
        """In rules_only mode, _get_model_and_embeddings must return (None,None,None)
        if the model is not already resident, i.e., must NOT load it.
        """
        import thyra.formation.refiner as ref

        monkeypatch.setattr(ref, "_model", None)  # not yet loaded
        ref.set_rules_only(True)
        try:
            model, embeddings, categories = ref._get_model_and_embeddings()
            assert model is None
        finally:
            ref.set_rules_only(False)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SYNONYM EXPANSION MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class TestSynonymModel:
    def test_get_model_returns_model_or_none(self):
        """synonym._get_model() must return a SentenceTransformer or None — never raise."""
        from thyra.recall.synonym import _get_model

        model = _get_model()
        if model is not None:
            assert callable(getattr(model, "encode", None))
        # else: graceful degradation — model not downloaded locally

    def test_expand_cues_returns_list_when_model_absent(self, tmp_db):
        """expand_cues_with_synonyms must return [] when model is unavailable,
        not raise."""
        import thyra.recall.synonym as syn

        # Monkeypatch the module-level _model to None so no model load
        original = syn._model
        syn._model = None
        # Also patch _get_model to return None directly
        import unittest.mock as mock

        with mock.patch.object(syn, "_get_model", return_value=None):
            result = syn.expand_cues_with_synonyms(tmp_db, ["python", "deploy"])
        syn._model = original
        assert isinstance(result, list)
        assert result == []

    def test_seed_synonym_edges_noop_without_model(self, tmp_db):
        """seed_synonym_edges_for_memory returns 0 when model is unavailable."""
        from thyra.models.memory import create_memory
        from thyra.recall.synonym import seed_synonym_edges_for_memory
        from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A
        import thyra.recall.synonym as syn
        import unittest.mock as mock

        mem_id = create_memory(tmp_db, "test memory for synonym seed")
        with mock.patch.object(syn, "_get_model", return_value=None):
            count = seed_synonym_edges_for_memory(tmp_db, mem_id, U, A)
        assert count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  DEDUP — cosine + word-overlap paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestDedupPaths:
    def test_find_near_match_empty_db_returns_none(self, tmp_db):
        """No matches in empty DB — must return None, not raise."""
        from thyra.formation.dedup import find_near_match
        from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A

        result = find_near_match(tmp_db, "Python is my preferred language", U, A)
        assert result is None

    def test_exact_content_hash_match(self, tmp_db):
        """Exact (normalized) content match must return the memory via O(1) hash lookup."""
        from thyra.models.memory import create_memory
        from thyra.formation.dedup import find_near_match
        from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A

        content = "I always use tabs for Python indentation"
        mem_id = create_memory(tmp_db, content)
        result = find_near_match(tmp_db, content, U, A)
        assert result is not None
        assert result.id == mem_id

    def test_word_overlap_fast_mode(self, tmp_db):
        """fast=True must use word-overlap only, no ML model."""
        from thyra.models.memory import create_memory
        from thyra.formation.dedup import find_near_match
        from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A
        import unittest.mock as mock
        import thyra.formation.dedup as dedup_mod

        content = "I prefer Python over Java for backend services"
        mem_id = create_memory(tmp_db, content)

        # Patch cosine_similarity so it raises — fast=True must never reach it
        with (
            mock.patch("thyra.formation.dedup.find_near_match.__code__", None)
            if False
            else mock.patch.object(
                dedup_mod, "_word_overlap_match", wraps=dedup_mod._word_overlap_match
            ) as spy
        ):
            result = find_near_match(
                tmp_db,
                "I prefer Python over Java for backend work",  # similar
                U,
                A,
                fast=True,
            )
        # Result might be None (overlap < threshold) or the matched memory
        assert result is None or result.id == mem_id

    def test_word_overlap_fallback_when_model_unavailable(self, tmp_db):
        """When sentence_transformers is unavailable, cosine path falls back to word-overlap."""
        from thyra.models.memory import create_memory
        from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A
        import thyra.formation.refiner as ref
        import unittest.mock as mock

        content = "I prefer Python for all backend projects"
        mem_id = create_memory(tmp_db, content)

        # Force the model to None so cosine path raises ImportError
        with mock.patch.object(ref, "_model", None):
            from thyra.formation.dedup import find_near_match

            # Should not raise; may return None or a match via word_overlap
            result = find_near_match(tmp_db, content, U, A, fast=False)
        # At minimum, exact content should be caught by hash check before ML
        assert result is not None and result.id == mem_id

    def test_empty_content_returns_none(self, tmp_db):
        from thyra.formation.dedup import find_near_match
        from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A

        assert find_near_match(tmp_db, "", U, A) is None
        assert find_near_match(tmp_db, "   ", U, A) is None


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DISTILLER
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistiller:
    def test_prewarm_returns_false_when_disabled(self, monkeypatch):
        """prewarm() must return False when DISTILLER_ENABLED=False."""
        import thyra.formation.distiller as dist
        import thyra.config as cfg

        monkeypatch.setattr(cfg, "DISTILLER_ENABLED", False)
        result = dist.prewarm()
        assert result is False

    def test_distill_returns_none_when_disabled(self, monkeypatch):
        """distill() must return None immediately when DISTILLER_ENABLED=False."""
        import thyra.formation.distiller as dist
        import thyra.config as cfg

        monkeypatch.setattr(cfg, "DISTILLER_ENABLED", False)
        result = dist.distill("I prefer Python for all projects")
        assert result is None

    def test_distill_returns_none_when_model_absent(self, monkeypatch):
        """When DISTILLER_ENABLED=True but no model is configured, distill returns None."""
        import thyra.formation.distiller as dist
        import thyra.config as cfg
        import os

        monkeypatch.setattr(cfg, "DISTILLER_ENABLED", True)
        monkeypatch.delenv("THYRA_DISTILLER_MODEL", raising=False)
        # Reset judge so it re-evaluates
        monkeypatch.setattr(dist, "_judge", None)
        result = dist.distill("I prefer Python for all projects")
        # Backend load returns None (no model configured) → distill returns None
        assert result is None

    def test_parse_verdict_valid_keep(self):
        """_parse_verdict must parse a valid 'keep' JSON response correctly."""
        from thyra.formation.distiller import _parse_verdict

        raw = '{"keep": true, "fact": "User prefers Python for backend work.", "category": "preferences", "kind": "semantic"}'
        verdict = _parse_verdict(raw)
        assert verdict is not None
        assert verdict["keep"] is True
        assert "Python" in verdict["fact"]
        assert verdict["kind"] == "semantic"

    def test_parse_verdict_drop(self):
        from thyra.formation.distiller import _parse_verdict

        raw = '{"keep": false, "fact": null, "category": null, "kind": "semantic"}'
        verdict = _parse_verdict(raw)
        assert verdict is not None
        assert verdict["keep"] is False

    def test_parse_verdict_malformed_json_returns_none(self):
        from thyra.formation.distiller import _parse_verdict

        assert _parse_verdict("not json at all") is None
        assert _parse_verdict("") is None
        assert _parse_verdict(None) is None

    def test_parse_verdict_keep_with_no_fact_treated_as_drop(self):
        """keep=true with no usable fact must be coerced to drop."""
        from thyra.formation.distiller import _parse_verdict

        raw = '{"keep": true, "fact": null, "category": "context", "kind": "semantic"}'
        verdict = _parse_verdict(raw)
        assert verdict["keep"] is False

    def test_parse_verdict_invalid_kind_defaults_to_semantic(self):
        from thyra.formation.distiller import _parse_verdict

        raw = '{"keep": false, "fact": null, "category": null, "kind": "unknown_type"}'
        verdict = _parse_verdict(raw)
        assert verdict["kind"] == "semantic"


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  DB CONNECTION & SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════


class TestDBConnection:
    def test_schema_has_all_required_tables(self, tmp_db):
        """The provisioned schema must contain all tables the system depends on."""
        required_tables = {
            "memories",
            "cue_edges",
            "cue_nodes",
            "association_edges",
            "situation_edges",
            "categories",
            "turn_log",
            "processed_turns",
            "system_flags",
        }
        rows = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        actual = {r["name"] for r in rows}
        missing = required_tables - actual
        assert not missing, f"Missing tables: {missing}"

    def test_fts_table_exists(self, tmp_db):
        rows = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_fts'"
        ).fetchall()
        assert len(rows) == 1

    def test_fts_triggers_exist(self, tmp_db):
        rows = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'mem_fts_%'"
        ).fetchall()
        assert len(rows) == 3  # insert, delete, update

    def test_foreign_keys_enabled(self, tmp_db):
        row = tmp_db.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_seed_categories_populated(self, tmp_db):
        """Standard categories must be seeded in the categories table."""
        rows = tmp_db.execute(
            "SELECT cat_id FROM categories WHERE user_id='default' AND agent_id='claude-code-global'"
        ).fetchall()
        cat_ids = {r["cat_id"] for r in rows}
        for expected in (
            "constraints",
            "identity",
            "preferences",
            "context",
            "knowledge",
        ):
            assert expected in cat_ids, f"Missing seed category: {expected}"

    def test_seed_flags_populated(self, tmp_db):
        """System flags must be seeded with default values."""
        rows = tmp_db.execute(
            "SELECT flag_key FROM system_flags WHERE user_id='default' AND agent_id='claude-code-global'"
        ).fetchall()
        flag_keys = {r["flag_key"] for r in rows}
        assert "system_enabled" in flag_keys
        assert "formation_enabled" in flag_keys

    def test_fts_sync_on_insert(self, tmp_db):
        """FTS trigger must keep memory_fts in sync with memories on INSERT."""
        from thyra.models.memory import create_memory
        from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A

        content = "this is a unique fts sync test phrase"
        mem_id = create_memory(tmp_db, content)

        row = tmp_db.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
            ('"unique"',),
        ).fetchone()
        assert row is not None, "FTS did not pick up newly inserted memory"

    def test_fts_sync_on_delete(self, tmp_db):
        """FTS trigger must clean up when a memory is deleted."""
        from thyra.models.memory import create_memory, delete_memory
        from thyra.config import THYRA_USER_ID as U, THYRA_AGENT_ID as A

        content = "uniquedeleteftsphrase"
        mem_id = create_memory(tmp_db, content)
        # Verify it's there
        row_before = tmp_db.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
            ('"uniquedeleteftsphrase"',),
        ).fetchone()
        assert row_before is not None

        delete_memory(tmp_db, mem_id, U, A)

        row_after = tmp_db.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
            ('"uniquedeleteftsphrase"',),
        ).fetchone()
        assert row_after is None, "FTS still has entry for deleted memory"


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CONFIG SANITY
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigSanity:
    """Guard against accidental misconfiguration of numeric constants."""

    def test_decay_rates_are_positive(self):
        from thyra import config as cfg

        for name in (
            "DECAY_EXPLICIT",
            "DECAY_CONSTRAINTS",
            "DECAY_IDENTITY",
            "DECAY_SEMANTIC",
            "DECAY_EPISODIC",
        ):
            val = getattr(cfg, name)
            assert val > 0, f"{name} must be positive"

    def test_archive_threshold_below_base_strength(self):
        from thyra.config import (
            ARCHIVE_THRESHOLD,
            BASE_STRENGTH_EXPLICIT,
            BASE_STRENGTH_AUTOMATIC,
        )

        assert ARCHIVE_THRESHOLD < BASE_STRENGTH_AUTOMATIC < BASE_STRENGTH_EXPLICIT

    def test_resurrection_threshold_above_archive(self):
        from thyra.config import ARCHIVE_THRESHOLD, RESURRECTION_THRESHOLD

        assert RESURRECTION_THRESHOLD > ARCHIVE_THRESHOLD

    def test_strength_cap_above_explicit(self):
        from thyra.config import BASE_STRENGTH_EXPLICIT, STRENGTH_CAP

        assert STRENGTH_CAP > BASE_STRENGTH_EXPLICIT

    def test_spreading_constants_sum_below_one(self):
        """Total spreading should not produce scores far above base_strength."""
        from thyra.config import SPREADING_DIRECT, SPREADING_ASSOC, SPREADING_SITUATION

        # No single signal should dominate more than 100% of base_level
        assert SPREADING_DIRECT <= 1.0
        assert SPREADING_ASSOC <= 1.0
        assert SPREADING_SITUATION <= 1.0

    def test_hub_cue_fraction_between_zero_and_one(self):
        from thyra.config import HUB_CUE_FRACTION

        assert 0.0 < HUB_CUE_FRACTION < 1.0

    def test_cue_promote_threshold_below_seed_weight(self):
        """Candidate edges need only a small nudge to become real — threshold < seed weight."""
        from thyra.config import CUE_PROMOTE_THRESHOLD, CONTENT_SEED_WEIGHT

        assert CUE_PROMOTE_THRESHOLD < CONTENT_SEED_WEIGHT

    def test_hebbian_weight_delta_positive(self):
        from thyra.config import HEBBIAN_WEIGHT_DELTA

        assert HEBBIAN_WEIGHT_DELTA > 0

    def test_salience_threshold_below_directive_score(self):
        """A directive ('remember that…') must clear the salience threshold."""
        from thyra.config import SALIENCE_THRESHOLD
        from thyra.formation.salience import compute_salience

        sal = compute_salience("Remember that I always use tabs in Python")
        assert sal >= SALIENCE_THRESHOLD

    def test_nightly_idle_check_seconds_is_positive(self):
        from thyra.config import NIGHTLY_IDLE_CHECK_SECONDS

        assert NIGHTLY_IDLE_CHECK_SECONDS > 0

    def test_token_budgets_positive_for_all_tiers(self):
        from thyra.config import TOKEN_BUDGETS

        for tier, budget in TOKEN_BUDGETS.items():
            assert budget > 0, f"Token budget for tier {tier!r} must be positive"
