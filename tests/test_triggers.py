"""Senior-level tests for trigger mechanisms.

Tests six previously-uncovered paths:
  1. TestCueOverlapPairs      — count_cue_overlap_pairs SQL algorithm
  2. TestCueOverlapNightlyFire — nightly fires when and only when cue_due fires
  3. TestJunkCleanupPass1     — exact duplicate dedup (pass 1)
  4. TestJunkCleanupPass2     — content heuristic cleanup (pass 2)
  5. TestFormationRoundTrip   — formation → cue edges → recall end-to-end
  6. TestHotCacheAutoInvalidation — HOT_CACHE cleared after _apply_delta
"""

from __future__ import annotations

import time

import pytest

from thyra.config import (
    THYRA_USER_ID as U,
    THYRA_AGENT_ID as A,
    NIGHTLY_CUE_OVERLAP_MIN_SHARED,
    NIGHTLY_CUE_OVERLAP_PAIR_LIMIT,
    NIGHTLY_CUE_OVERLAP_THRESHOLD,
    CLEANUP_INTERVAL_HOURS,
)
from tests.conftest import make_delta, apply_delta_sync


# ── Schema helpers ─────────────────────────────────────────────────────────────


def _insert_memory(conn, mem_id, user_id=U, agent_id=A, strength=0.5, content=None):
    now = int(time.time() * 1000)
    conn.execute(
        "INSERT OR IGNORE INTO memories "
        "(id, content, user_id, agent_id, base_strength, created_at, last_access, category, memory_type) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            mem_id,
            content or f"Memory {mem_id}",
            user_id,
            agent_id,
            strength,
            now,
            now,
            "context",
            "semantic",
        ),
    )


def _insert_cue(conn, cue_id, memory_id, user_id=U, agent_id=A, candidate=0):
    now = int(time.time() * 1000)
    conn.execute(
        "INSERT OR IGNORE INTO cue_edges "
        "(cue_id, memory_id, user_id, agent_id, weight, candidate, created_at) "
        "VALUES (?,?,?,?,0.8,?,?)",
        (cue_id, memory_id, user_id, agent_id, candidate, now),
    )


# ── Priority 1: count_cue_overlap_pairs algorithm ─────────────────────────────


class TestCueOverlapPairs:
    """count_cue_overlap_pairs uses overlap coefficient (shared / min(a, b)).

    A small memory fully contained in a larger one scores 1.0, which is the
    case batch-dedup resolves. min_shared=4 filters single-cue false positives.
    """

    def test_qualifying_pair_counted(self, tmp_db):
        """A pair sharing 4 out of 5 cues each (overlap 0.80 >= 0.70) counts as 1."""
        from thyra.consolidation.nightly import count_cue_overlap_pairs

        _insert_memory(tmp_db, "m_aaa")
        _insert_memory(tmp_db, "m_bbb")
        shared = ["cue_a", "cue_b", "cue_c", "cue_d"]
        for c in shared:
            _insert_cue(tmp_db, c, "m_aaa")
            _insert_cue(tmp_db, c, "m_bbb")
        _insert_cue(tmp_db, "cue_e", "m_aaa")  # unique to m_aaa
        _insert_cue(tmp_db, "cue_f", "m_bbb")  # unique to m_bbb
        tmp_db.commit()

        result = count_cue_overlap_pairs(
            tmp_db,
            U,
            A,
            min_shared=4,
            threshold=0.70,
            pair_limit=10,
        )
        assert result == 1

    def test_below_min_shared_not_counted(self, tmp_db):
        """A pair sharing only 3 cues is rejected by the min_shared=4 floor."""
        from thyra.consolidation.nightly import count_cue_overlap_pairs

        _insert_memory(tmp_db, "m_ccc")
        _insert_memory(tmp_db, "m_ddd")
        for c in ["cue_x", "cue_y", "cue_z"]:  # only 3 shared
            _insert_cue(tmp_db, c, "m_ccc")
            _insert_cue(tmp_db, c, "m_ddd")
        _insert_cue(
            tmp_db, "cue_w", "m_ddd"
        )  # overlap would be 3/3=1.0 but < min_shared
        tmp_db.commit()

        result = count_cue_overlap_pairs(
            tmp_db,
            U,
            A,
            min_shared=4,
            threshold=0.70,
            pair_limit=10,
        )
        assert result == 0

    def test_pair_limit_reached_returns_limit(self, tmp_db):
        """With pair_limit=3 qualifying pairs present, function returns 3."""
        from thyra.consolidation.nightly import count_cue_overlap_pairs

        shared_cues = ["cue_1", "cue_2", "cue_3", "cue_4"]
        pairs = [("m_p1a", "m_p1b"), ("m_p2a", "m_p2b"), ("m_p3a", "m_p3b")]
        for ma, mb in pairs:
            _insert_memory(tmp_db, ma)
            _insert_memory(tmp_db, mb)
            for c in shared_cues:
                _insert_cue(tmp_db, c, ma)
                _insert_cue(tmp_db, c, mb)
            _insert_cue(tmp_db, f"uniq_{ma}", ma)  # unique to ma
            _insert_cue(tmp_db, f"uniq_{mb}", mb)  # unique to mb
        tmp_db.commit()

        result = count_cue_overlap_pairs(
            tmp_db,
            U,
            A,
            min_shared=4,
            threshold=0.70,
            pair_limit=3,
        )
        assert result == 3

    def test_two_pairs_below_limit(self, tmp_db):
        """With only 2 qualifying pairs and pair_limit=3, returns 2 (does not fire).

        Each pair uses its own exclusive set of shared cues so the memories in
        different pairs do NOT overlap with each other, giving exactly 2 pairs.
        """
        from thyra.consolidation.nightly import count_cue_overlap_pairs

        pairs = [("m_q1a", "m_q1b"), ("m_q2a", "m_q2b")]
        for i, (ma, mb) in enumerate(pairs):
            # Use pair-specific cue names to avoid cross-pair contamination
            shared = [f"p{i}_cue_{j}" for j in range(4)]
            _insert_memory(tmp_db, ma)
            _insert_memory(tmp_db, mb)
            for c in shared:
                _insert_cue(tmp_db, c, ma)
                _insert_cue(tmp_db, c, mb)
            _insert_cue(tmp_db, f"uniq_{ma}", ma)
            _insert_cue(tmp_db, f"uniq_{mb}", mb)
        tmp_db.commit()

        result = count_cue_overlap_pairs(
            tmp_db,
            U,
            A,
            min_shared=4,
            threshold=0.70,
            pair_limit=3,
        )
        assert result == 2
        assert result < NIGHTLY_CUE_OVERLAP_PAIR_LIMIT  # does not fire

    def test_min_shared_boundary_4_counts_3_does_not(self, tmp_db):
        """min_shared=4 is inclusive: exactly 4 qualifies, exactly 3 does not."""
        from thyra.consolidation.nightly import count_cue_overlap_pairs

        # Pair with exactly 4 shared cues (min_shared=4 → qualifies)
        _insert_memory(tmp_db, "m_r1")
        _insert_memory(tmp_db, "m_r2")
        for c in ["ca", "cb", "cc", "cd"]:
            _insert_cue(tmp_db, c, "m_r1")
            _insert_cue(tmp_db, c, "m_r2")
        _insert_cue(tmp_db, "ce", "m_r1")
        tmp_db.commit()

        assert (
            count_cue_overlap_pairs(
                tmp_db, U, A, min_shared=4, threshold=0.70, pair_limit=5
            )
            == 1
        )

        # Pair with exactly 3 shared cues (< min_shared → does NOT qualify)
        _insert_memory(tmp_db, "m_s1")
        _insert_memory(tmp_db, "m_s2")
        for c in ["cx", "cy", "cz"]:
            _insert_cue(tmp_db, c, "m_s1")
            _insert_cue(tmp_db, c, "m_s2")
        _insert_cue(tmp_db, "cw", "m_s1")
        tmp_db.commit()

        # Only the 4-shared pair qualifies; s1/s2 pair does not
        assert (
            count_cue_overlap_pairs(
                tmp_db, U, A, min_shared=4, threshold=0.70, pair_limit=5
            )
            == 1
        )


# ── Priority 1: cue-overlap trigger wiring ────────────────────────────────────


class TestCueOverlapNightlyFire:
    """Verify the nightly fires on cue_due=True and not before."""

    def _worker(self, db_path):
        from thyra.consolidation.worker import BackgroundWorker

        return BackgroundWorker(db_path=db_path)

    def _set_recent_nightly(self, conn):
        from thyra.models.memory import set_flag

        now_ms = int(time.time() * 1000)
        set_flag(conn, "last_nightly", str(now_ms), U, A)
        conn.commit()

    def _seed_n_qualifying_pairs(self, conn, n):
        # Use pair-specific cue names to avoid cross-pair contamination;
        # otherwise all n pairs would share cues and produce C(2n,2) pairs.
        for i in range(n):
            shared = [f"fire_{i}_c{j}" for j in range(4)]
            ma, mb = f"m_fire_{i}a", f"m_fire_{i}b"
            _insert_memory(conn, ma)
            _insert_memory(conn, mb)
            for c in shared:
                _insert_cue(conn, c, ma)
                _insert_cue(conn, c, mb)
            _insert_cue(conn, f"u_{ma}", ma)
            _insert_cue(conn, f"u_{mb}", mb)
        conn.commit()

    def test_cue_overlap_fires_nightly(self, tmp_db, monkeypatch):
        """With time_due=False, usage_due=False, 3 qualifying pairs → nightly fires."""
        import os
        import thyra.consolidation.nightly as nmod

        db_path = os.environ["THYRA_DB_PATH"]
        self._set_recent_nightly(tmp_db)
        self._seed_n_qualifying_pairs(tmp_db, NIGHTLY_CUE_OVERLAP_PAIR_LIMIT)

        calls = []
        original_sweep = nmod.run_nightly_sweep

        def tracking_sweep(conn, uid, aid):
            calls.append((uid, aid))
            return original_sweep(conn, uid, aid)

        monkeypatch.setattr(nmod, "run_nightly_sweep", tracking_sweep)

        worker = self._worker(db_path)
        worker._maybe_nightly(U, A)

        assert len(calls) == 1, "Nightly should have fired exactly once"
        assert calls[0] == (U, A)

    def test_cue_overlap_below_limit_does_not_fire(self, tmp_db, monkeypatch):
        """With only 2 qualifying pairs (< pair_limit=3), nightly does NOT fire."""
        import os
        import thyra.consolidation.nightly as nmod

        db_path = os.environ["THYRA_DB_PATH"]
        self._set_recent_nightly(tmp_db)
        self._seed_n_qualifying_pairs(tmp_db, NIGHTLY_CUE_OVERLAP_PAIR_LIMIT - 1)

        calls = []
        monkeypatch.setattr(
            nmod, "run_nightly_sweep", lambda conn, u, a: calls.append((u, a)) or {}
        )

        worker = self._worker(db_path)
        worker._maybe_nightly(U, A)

        assert len(calls) == 0, "Nightly should NOT fire with too few overlapping pairs"

    def test_reason_is_cue_overlap_not_time_or_usage(self, tmp_db, monkeypatch, caplog):
        """When only cue_due fires, the logged reason is 'cue_overlap'."""
        import logging
        import os
        import thyra.consolidation.nightly as nmod

        db_path = os.environ["THYRA_DB_PATH"]
        self._set_recent_nightly(tmp_db)
        self._seed_n_qualifying_pairs(tmp_db, NIGHTLY_CUE_OVERLAP_PAIR_LIMIT)

        monkeypatch.setattr(nmod, "run_nightly_sweep", lambda conn, u, a: {})

        worker = self._worker(db_path)
        with caplog.at_level(logging.INFO, logger="thyra.worker"):
            worker._maybe_nightly(U, A)

        assert any("cue_overlap" in r.message for r in caplog.records), (
            "Expected 'cue_overlap' in log; got: "
            + str([r.message for r in caplog.records])
        )


# ── Priority 2: junk cleanup pass 1 (exact duplicate dedup) ───────────────────


class TestJunkCleanupPass1:
    """run_junk_cleanup pass 1: exact duplicates — keep highest-strength copy."""

    def test_weaker_duplicate_deleted_stronger_survives(self, tmp_db):
        from thyra.consolidation.cleanup import run_junk_cleanup
        from thyra.models.memory import compute_content_hash

        content = "I prefer Python for all data science work"
        chash = compute_content_hash(content)

        now = int(time.time() * 1000)
        # stronger memory (base_strength=0.8)
        tmp_db.execute(
            "INSERT INTO memories "
            "(id, content, content_hash, user_id, agent_id, base_strength, created_at, last_access, category, memory_type) "
            "VALUES (?,?,?,?,?,0.8,?,?,?,?)",
            ("m_strong", content, chash, U, A, now, now, "context", "semantic"),
        )
        # weaker duplicate
        tmp_db.execute(
            "INSERT INTO memories "
            "(id, content, content_hash, user_id, agent_id, base_strength, created_at, last_access, category, memory_type) "
            "VALUES (?,?,?,?,?,0.4,?,?,?,?)",
            ("m_weak", content, chash, U, A, now, now, "context", "semantic"),
        )
        # cue_edge pointing to the weaker memory
        tmp_db.execute(
            "INSERT INTO cue_edges (cue_id, memory_id, user_id, agent_id, weight, candidate, created_at) "
            "VALUES ('python', 'm_weak', ?, ?, 0.8, 0, ?)",
            (U, A, now),
        )
        tmp_db.commit()

        deleted = run_junk_cleanup(tmp_db, U, A)

        assert deleted == 1
        rows = tmp_db.execute(
            "SELECT id, base_strength FROM memories WHERE archived=0"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["id"] == "m_strong"
        assert abs(rows[0]["base_strength"] - 0.8) < 1e-6

        # ON DELETE CASCADE should have removed the cue_edge
        cue_rows = tmp_db.execute(
            "SELECT * FROM cue_edges WHERE memory_id='m_weak'"
        ).fetchall()
        assert len(cue_rows) == 0

    def test_rate_limit_via_worker_maybe_cleanup(self, tmp_db, monkeypatch):
        """_maybe_cleanup is a no-op when called twice quickly; fires after cooldown."""
        import os
        from thyra.consolidation.worker import BackgroundWorker

        db_path = os.environ["THYRA_DB_PATH"]
        worker = BackgroundWorker(db_path=db_path)

        call_count = [0]
        original = __import__(
            "thyra.consolidation.cleanup", fromlist=["run_junk_cleanup"]
        ).run_junk_cleanup

        def counting_cleanup(conn, uid, aid):
            call_count[0] += 1
            return original(conn, uid, aid)

        import thyra.consolidation.cleanup as cmod

        monkeypatch.setattr(cmod, "run_junk_cleanup", counting_cleanup)

        # First call — should run
        worker._maybe_cleanup(U, A)
        assert call_count[0] == 1

        # Second call immediately — rate-limited, should be a no-op
        worker._maybe_cleanup(U, A)
        assert call_count[0] == 1

        # Back-date the timestamp past the cooldown window
        pair_key = f"{U}:{A}"
        worker._last_cleanup[pair_key] = time.time() - CLEANUP_INTERVAL_HOURS * 3600 - 1
        worker._maybe_cleanup(U, A)
        assert call_count[0] == 2


# ── Priority 2: junk cleanup pass 2 (content heuristics) ──────────────────────


class TestJunkCleanupPass2:
    """run_junk_cleanup pass 2: classify and delete noise by content heuristics."""

    def test_id_listing_memory_deleted(self, tmp_db):
        from thyra.consolidation.cleanup import run_junk_cleanup, _classify_junk_reasons

        # 16 lowercase hex chars after "m_" to satisfy _MEM_ID_RE
        junk_content = "m_1234567890abcdef, m_fedcba0987654321"
        legit_content = "My preferred editor is VS Code with dark theme"

        now = int(time.time() * 1000)
        tmp_db.execute(
            "INSERT INTO memories "
            "(id, content, user_id, agent_id, base_strength, created_at, last_access, category, memory_type) "
            "VALUES (?,?,?,?,0.4,?,?,?,?)",
            ("m_junk", junk_content, U, A, now, now, "context", "semantic"),
        )
        tmp_db.execute(
            "INSERT INTO memories "
            "(id, content, user_id, agent_id, base_strength, created_at, last_access, category, memory_type) "
            "VALUES (?,?,?,?,0.5,?,?,?,?)",
            ("m_legit", legit_content, U, A, now, now, "preferences", "semantic"),
        )
        tmp_db.commit()

        deleted = run_junk_cleanup(tmp_db, U, A)

        assert deleted == 1
        survivors = tmp_db.execute(
            "SELECT id FROM memories WHERE archived=0"
        ).fetchall()
        assert len(survivors) == 1
        assert survivors[0]["id"] == "m_legit"

    def test_classify_junk_reasons_returns_memory_id_listing(self, tmp_db):
        from thyra.consolidation.cleanup import _classify_junk_reasons

        # IDs must be exactly 16 lowercase hex chars after "m_" to match _MEM_ID_RE
        junk = "m_1234567890abcdef, m_fedcba0987654321"
        reasons = _classify_junk_reasons(junk)
        assert "memory-id-listing" in reasons, (
            f"Expected 'memory-id-listing' in reasons, got: {reasons}"
        )

    def test_legitimate_content_not_classified_as_junk(self, tmp_db):
        from thyra.consolidation.cleanup import _classify_junk_reasons

        legit = "My preferred editor is VS Code with dark theme"
        reasons = _classify_junk_reasons(legit)
        assert reasons == [], (
            f"Legitimate memory should not be classified as junk: {reasons}"
        )


# ── Priority 3: formation round-trip (formation → cues → recall) ──────────────


class TestFormationRoundTrip:
    """Formation creates a memory, seeds cue edges, and recall finds it."""

    def test_formed_memory_is_recalled(self, tmp_db):
        from thyra.recall.intent import recall_pipeline

        delta = make_delta(
            user_text="My manager is Sarah and she leads the platform team"
        )
        result = apply_delta_sync(tmp_db, delta)

        # Step 1: a memory was created
        assert "created" in result, (
            f"Formation vetoed the sentence — adjust TRANSIENT_SAMPLES or the test text. "
            f"Actions: {result}"
        )
        mem_id = result["created"]

        # Cue edges were seeded for the new memory
        cue_rows = tmp_db.execute(
            "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
            (mem_id, U, A),
        ).fetchall()
        assert len(cue_rows) > 0, "No cue edges found for the newly formed memory"

        # Step 2: recall with a query touching a seeded cue
        xml, served = recall_pipeline(tmp_db, U, A, "Who is Sarah?", "s1", "t2")
        assert mem_id in served, (
            f"Newly formed memory {mem_id} not found in recall results. "
            f"Served: {served}\nCues: {[r['cue_id'] for r in cue_rows]}"
        )


# ── Priority 4: HOT_CACHE invalidation after delta processing ─────────────────


class TestHotCacheAutoInvalidation:
    """drain._apply_delta invalidates the hot cache for the processed pair only."""

    def test_cache_cleared_for_touched_pair(self, tmp_db):
        import os
        from thyra.consolidation.drain import _apply_delta
        from thyra.recall.cache import HOT_CACHE

        db_path = os.environ["THYRA_DB_PATH"]

        # Pre-seed the cache with a dummy snapshot for (U, A)
        HOT_CACHE.set(f"snapshot:{U}:{A}", {"dummy": True})
        assert HOT_CACHE.get(f"snapshot:{U}:{A}") is not None

        delta = make_delta(user_text="My preferred shell is PowerShell")
        _apply_delta(delta, db_path)

        assert HOT_CACHE.get(f"snapshot:{U}:{A}") is None, (
            "HOT_CACHE should be cleared for the processed (user, agent) pair"
        )

    def test_other_pair_cache_not_cleared(self, tmp_db):
        import os
        from thyra.consolidation.drain import _apply_delta
        from thyra.recall.cache import HOT_CACHE

        db_path = os.environ["THYRA_DB_PATH"]

        other_key = "snapshot:other_user:other_agent"
        HOT_CACHE.set(other_key, {"unrelated": True})
        HOT_CACHE.set(f"snapshot:{U}:{A}", {"dummy": True})

        delta = make_delta(user_text="I use tabs not spaces")
        _apply_delta(delta, db_path)

        # The OTHER pair's cache entry must NOT be evicted
        assert HOT_CACHE.get(other_key) is not None, (
            "HOT_CACHE for an unrelated pair should not be cleared"
        )
