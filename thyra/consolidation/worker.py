"""Background consolidation worker — polls delta_queue/ and applies learning."""

from __future__ import annotations

import logging
import time

from thyra.config import (
    CLEANUP_INTERVAL_HOURS,
    NIGHTLY_CUE_OVERLAP_MIN_SHARED,
    NIGHTLY_CUE_OVERLAP_PAIR_LIMIT,
    NIGHTLY_CUE_OVERLAP_THRESHOLD,
    NIGHTLY_IDLE_CHECK_SECONDS,
    NIGHTLY_INTERVAL_HOURS,
    SWEEP_FORMATION_THRESHOLD,
    THYRA_DB_PATH,
    WORKER_POLL_SECONDS,
)

log = logging.getLogger("thyra.worker")


class BackgroundWorker:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or THYRA_DB_PATH
        self._running = True
        self._last_nightly: dict[str, float] = {}
        self._last_cleanup: dict[str, float] = {}

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        log.info("Thyra consolidation worker started (db=%s)", self._db_path)
        self._startup_nightly_check()
        _last_idle_check = time.time()
        while self._running:
            try:
                self._process_queue()
            except Exception as exc:
                log.exception("Worker loop error: %s", exc)
            if time.time() - _last_idle_check >= NIGHTLY_IDLE_CHECK_SECONDS:
                self._idle_nightly_check()
                _last_idle_check = time.time()
            time.sleep(WORKER_POLL_SECONDS)

    # ── Queue processing ───────────────────────────────────────────────────────

    def _process_queue(self) -> None:
        from thyra.consolidation.drain import drain_queue

        drain_queue(self._db_path, budget_ms=None)

    # ── Cleanup & nightly ──────────────────────────────────────────────────────

    def _maybe_cleanup(self, user_id: str, agent_id: str) -> None:
        """Run junk cleanup for (user_id, agent_id) if enough time has passed.

        Rate-limited to once per CLEANUP_INTERVAL_HOURS per pair.
        """
        pair_key = f"{user_id}:{agent_id}"
        if (
            time.time() - self._last_cleanup.get(pair_key, 0.0)
            < CLEANUP_INTERVAL_HOURS * 3600
        ):
            return
        self._last_cleanup[pair_key] = time.time()
        try:
            from thyra.consolidation.cleanup import run_junk_cleanup
            from thyra.db.connection import DBConnection

            conn = DBConnection.get(self._db_path)
            deleted = run_junk_cleanup(conn, user_id, agent_id)
            if deleted:
                log.info(
                    "Post-usage cleanup: %d deleted for %s:%s",
                    deleted,
                    user_id,
                    agent_id,
                )
        except ImportError:
            pass
        except Exception as exc:
            log.warning("Cleanup error: %s", exc)

    def _maybe_nightly(self, user_id: str, agent_id: str) -> None:
        """Run the nightly sweep for (user_id, agent_id) if any trigger fires.

        Uses the DB-persisted last_nightly timestamp so the check survives
        worker restarts.
        """
        pair_key = f"{user_id}:{agent_id}"
        last = self._last_nightly.get(pair_key, 0.0)

        if last == 0.0:
            try:
                from thyra.db.connection import DBConnection
                from thyra.models.memory import get_flag

                conn = DBConnection.get(self._db_path)
                db_last_ms = float(
                    get_flag(conn, "last_nightly", user_id, agent_id, default="0")
                )
                last = db_last_ms / 1000.0
                self._last_nightly[pair_key] = last
            except Exception:
                pass

        time_due = time.time() - last >= NIGHTLY_INTERVAL_HOURS * 3600
        usage_due = False
        cue_due = False
        if not time_due:
            try:
                from thyra.db.connection import DBConnection

                conn = DBConnection.get(self._db_path)
                usage_due = (
                    _formations_since(conn, user_id, agent_id, int(last * 1000))
                    >= SWEEP_FORMATION_THRESHOLD
                )
            except Exception:
                pass

        if not time_due and not usage_due:
            try:
                from thyra.consolidation.nightly import count_cue_overlap_pairs
                from thyra.db.connection import DBConnection

                conn = DBConnection.get(self._db_path)
                cue_due = (
                    count_cue_overlap_pairs(
                        conn,
                        user_id,
                        agent_id,
                        min_shared=NIGHTLY_CUE_OVERLAP_MIN_SHARED,
                        threshold=NIGHTLY_CUE_OVERLAP_THRESHOLD,
                        pair_limit=NIGHTLY_CUE_OVERLAP_PAIR_LIMIT,
                    )
                    >= NIGHTLY_CUE_OVERLAP_PAIR_LIMIT
                )
            except Exception:
                pass

        if time_due or usage_due or cue_due:
            self._last_nightly[pair_key] = time.time()
            try:
                from thyra.consolidation.nightly import run_nightly_sweep
                from thyra.db.connection import DBConnection

                conn = DBConnection.get(self._db_path)
                run_nightly_sweep(conn, user_id, agent_id)
                reason = "usage" if usage_due else "cue_overlap" if cue_due else "time"
                log.info(
                    "Nightly sweep complete for %s:%s (reason=%s)",
                    user_id,
                    agent_id,
                    reason,
                )
            except ImportError:
                pass
            except Exception as exc:
                log.warning("Nightly sweep error: %s", exc)

    def _startup_nightly_check(self) -> None:
        """On worker startup, run nightly for any pair that is overdue."""
        try:
            from thyra.db.connection import DBConnection

            conn = DBConnection.get(self._db_path)
            rows = conn.execute(
                "SELECT DISTINCT user_id, agent_id FROM memories WHERE archived=0"
            ).fetchall()
            for row in rows:
                self._maybe_cleanup(row["user_id"], row["agent_id"])
                self._maybe_nightly(row["user_id"], row["agent_id"])
        except Exception as exc:
            log.warning("Startup nightly check error: %s", exc)

    def _idle_nightly_check(self) -> None:
        """Periodic scan: run cleanup and nightly for any overdue pair."""
        try:
            from thyra.db.connection import DBConnection

            conn = DBConnection.get(self._db_path)
            rows = conn.execute(
                "SELECT DISTINCT user_id, agent_id FROM memories WHERE archived=0"
            ).fetchall()
            for row in rows:
                self._maybe_cleanup(row["user_id"], row["agent_id"])
                self._maybe_nightly(row["user_id"], row["agent_id"])
        except Exception as exc:
            log.warning("Idle nightly check error: %s", exc)


def _formations_since(conn, user_id: str, agent_id: str, since_ms: int) -> int:
    """Count active memories created after since_ms (epoch milliseconds)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? "
        "AND archived=0 AND created_at > ?",
        (user_id, agent_id, since_ms),
    ).fetchone()
    return row[0] if row else 0
