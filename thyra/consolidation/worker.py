"""Background consolidation worker — polls delta_queue/ and applies learning."""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from collections import deque

from thyra.config import (
    NIGHTLY_IDLE_CHECK_SECONDS,
    NIGHTLY_INTERVAL_HOURS,
    THYRA_DB_PATH,
    WORKER_POLL_SECONDS,
)
from thyra.models.delta import DeltaEvent

log = logging.getLogger("thyra.worker")


class BackgroundWorker:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or THYRA_DB_PATH
        self._queue_dir = pathlib.Path(self._db_path).parent / "delta_queue"
        self._queue_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._last_nightly: dict[str, float] = {}
        # Per-pair turn window (deque maxlen=3) for Hebbian association
        self._turn_windows: dict[str, deque] = {}
        # Per-pair locks to serialize consolidation
        self._pair_locks: dict[str, threading.Lock] = {}
        self._pair_locks_lock = threading.Lock()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        log.info("Thyra consolidation worker started (db=%s)", self._db_path)
        # On startup, catch up any pairs whose nightly is overdue — covers the
        # common case where the PC was off and the scheduled window was missed.
        self._startup_nightly_check()
        _last_idle_check = time.time()
        while self._running:
            try:
                self._process_queue()
            except Exception as exc:
                log.exception("Worker loop error: %s", exc)
            # Periodic idle scan: run nightly for overdue pairs even when no
            # turns are happening (queue stays empty indefinitely).
            if time.time() - _last_idle_check >= NIGHTLY_IDLE_CHECK_SECONDS:
                self._idle_nightly_check()
                _last_idle_check = time.time()
            time.sleep(WORKER_POLL_SECONDS)

    # ── Queue processing ───────────────────────────────────────────────────────

    def _process_queue(self) -> None:
        import traceback

        files = sorted(self._queue_dir.glob("*.json"))
        for fpath in files:
            if not self._running:
                break
            try:
                self._process_file(fpath)
            except Exception as exc:
                log.warning(
                    "Failed processing %s: %s\n%s",
                    fpath.name,
                    exc,
                    traceback.format_exc(),
                )
                self._move_to_errors(fpath)

    def _process_file(self, fpath: pathlib.Path) -> None:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        delta = DeltaEvent.from_dict(data)
        pair_key = f"{delta.user_id}:{delta.agent_id}"
        lock = self._get_pair_lock(pair_key)
        with lock:
            self._apply_delta(delta)
            self._maybe_nightly(delta.user_id, delta.agent_id)
        fpath.unlink(missing_ok=True)

    # ── Core delta processing (ordered — do not reorder) ─────────────────────

    def _apply_delta(self, delta: DeltaEvent) -> None:
        from thyra.db.connection import DBConnection
        from thyra.consolidation.decay import recompute_and_update, archive_check
        from thyra.consolidation.reinforcement import apply_reinforcement
        from thyra.consolidation.edges import update_cue_edges, hebbian_association
        from thyra.consolidation.situation import crystallize_situations
        from thyra.recall.cache import HOT_CACHE

        conn = DBConnection.get(self._db_path)

        # Step 1: Idempotency
        if delta.turn_id:
            existing = conn.execute(
                "SELECT 1 FROM processed_turns WHERE turn_id=?", (delta.turn_id,)
            ).fetchone()
            if existing:
                return

        # Step 2: Auto-formation
        # Run in rules-only mode so the worker thread never blocks on a model load.
        # find_near_match already uses fast=True; category classification uses rules.
        new_memory_ids: list[str] = []
        try:
            from thyra.formation.pipeline import run_formation_pipeline
            from thyra.formation.refiner import set_rules_only

            set_rules_only(True)
            try:
                actions = run_formation_pipeline(conn, delta)
            finally:
                set_rules_only(False)
            new_memory_ids = [mid for action, mid in actions if action == "created"]
        except ImportError:
            pass
        except Exception as exc:
            log.warning("Formation pipeline error: %s", exc)

        # Step 2.5: Synonym expansion for newly formed memories (Stage 6)
        if new_memory_ids:
            try:
                from thyra.config import SYNONYM_EXPANSION_ENABLED

                if SYNONYM_EXPANSION_ENABLED:
                    from thyra.recall.synonym import seed_synonym_edges_for_memory

                    for mid in new_memory_ids:
                        seed_synonym_edges_for_memory(
                            conn, mid, delta.user_id, delta.agent_id
                        )
            except Exception as exc:
                log.debug("Synonym expansion error: %s", exc)

        # Step 3: Lazy decay on touched memories
        recompute_and_update(conn, delta.memories_served, delta.user_id, delta.agent_id)

        # Step 4: Reinforcement
        apply_reinforcement(conn, delta)

        # Step 5: Cue edge updates
        update_cue_edges(conn, delta)

        # Step 6: Hebbian association (needs the window)
        pair_key = f"{delta.user_id}:{delta.agent_id}"
        if pair_key not in self._turn_windows:
            self._turn_windows[pair_key] = deque(maxlen=3)
        window = self._turn_windows[pair_key]
        window.append(delta)
        hebbian_association(conn, list(window), delta.user_id, delta.agent_id)

        # Step 7: Situation crystallization
        crystallize_situations(conn, list(window), delta.user_id, delta.agent_id)

        # Step 8: Archive check LAST (so boosted memories can't be archived same batch)
        archive_check(conn, delta.user_id, delta.agent_id)

        # Commit + mark processed
        with conn:
            if delta.turn_id:
                conn.execute(
                    "INSERT OR IGNORE INTO processed_turns (turn_id, processed_at) VALUES (?,?)",
                    (delta.turn_id, int(time.time() * 1000)),
                )
            # Log the turn for association tracking
            _log_turn(conn, delta)

        # Invalidate hot cache
        HOT_CACHE.invalidate(f"snapshot:{delta.user_id}:{delta.agent_id}")

    # ── Nightly sweep ──────────────────────────────────────────────────────────

    def _maybe_nightly(self, user_id: str, agent_id: str) -> None:
        """Run the nightly sweep for (user_id, agent_id) if enough time has passed.

        Uses the DB-persisted last_nightly timestamp as the authoritative source so
        the check survives server restarts (in-memory dict resets to 0 on each start,
        which would re-run the sweep on the very first delta regardless of when it
        last ran).
        """
        pair_key = f"{user_id}:{agent_id}"
        last = self._last_nightly.get(pair_key, 0.0)

        # On first encounter for this pair, read the persisted timestamp from the DB.
        # This prevents re-running the sweep immediately after a server restart when
        # it was already run recently.
        if last == 0.0:
            try:
                from thyra.db.connection import DBConnection
                from thyra.models.memory import get_flag

                conn = DBConnection.get(self._db_path)
                db_last_ms = float(
                    get_flag(conn, "last_nightly", user_id, agent_id, default="0")
                )
                last = db_last_ms / 1000.0  # ms → seconds
                self._last_nightly[pair_key] = last
            except Exception:
                pass  # fall through — 0.0 will trigger the sweep, which is safe

        if time.time() - last >= NIGHTLY_INTERVAL_HOURS * 3600:
            self._last_nightly[pair_key] = time.time()
            try:
                from thyra.consolidation.nightly import run_nightly_sweep
                from thyra.db.connection import DBConnection

                conn = DBConnection.get(self._db_path)
                run_nightly_sweep(conn, user_id, agent_id)
                log.info("Nightly sweep complete for %s:%s", user_id, agent_id)
            except ImportError:
                pass
            except Exception as exc:
                log.warning("Nightly sweep error: %s", exc)

    def _startup_nightly_check(self) -> None:
        """On worker startup, run nightly for any (user, agent) pair that is overdue.

        The PC may have been off for days — this catches up decay, archiving, and
        pruning that would otherwise only run after the user's next conversation turn.
        """
        try:
            from thyra.db.connection import DBConnection

            conn = DBConnection.get(self._db_path)
            rows = conn.execute(
                "SELECT DISTINCT user_id, agent_id FROM memories WHERE archived=0"
            ).fetchall()
            for row in rows:
                self._maybe_nightly(row["user_id"], row["agent_id"])
        except Exception as exc:
            log.warning("Startup nightly check error: %s", exc)

    def _idle_nightly_check(self) -> None:
        """Periodic idle scan: run nightly for any pair whose interval is overdue.

        Called by the run loop every NIGHTLY_IDLE_CHECK_SECONDS regardless of queue
        activity — covers the case where the PC stays on but the user never opens
        Claude Code for days.
        """
        try:
            from thyra.db.connection import DBConnection

            conn = DBConnection.get(self._db_path)
            rows = conn.execute(
                "SELECT DISTINCT user_id, agent_id FROM memories WHERE archived=0"
            ).fetchall()
            for row in rows:
                self._maybe_nightly(row["user_id"], row["agent_id"])
        except Exception as exc:
            log.warning("Idle nightly check error: %s", exc)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_pair_lock(self, pair_key: str) -> threading.Lock:
        with self._pair_locks_lock:
            if pair_key not in self._pair_locks:
                self._pair_locks[pair_key] = threading.Lock()
            return self._pair_locks[pair_key]

    def _move_to_errors(self, fpath: pathlib.Path) -> None:
        errors_dir = self._queue_dir / "errors"
        errors_dir.mkdir(exist_ok=True)
        dest = errors_dir / fpath.name
        try:
            fpath.rename(dest)
        except Exception:
            pass


def _log_turn(conn, delta: DeltaEvent) -> None:
    import json

    conn.execute(
        """INSERT OR IGNORE INTO turn_log
           (turn_id, session_id, user_id, agent_id, memories_served, memories_used, cues_fired, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            delta.turn_id,
            delta.session_id,
            delta.user_id,
            delta.agent_id,
            json.dumps(delta.memories_served),
            json.dumps(delta.memories_declared),
            json.dumps(delta.cues_fired),
            delta.timestamp,
        ),
    )
