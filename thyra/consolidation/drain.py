"""Shared delta-queue drainer — callable from stop hook, worker, and scripts.

Provides two public functions:
  drain_queue(db_path, budget_ms, rules_only) — process pending delta files
  run_nightly_for_all(db_path)               — nightly sweep for every pair

The drain uses a rename-to-processing/ lock so two concurrent processes (e.g.
the stop hook and the MCP-server worker) never apply the same delta. Idempotency
is also guaranteed at the DB level via processed_turns.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from collections import deque

log = logging.getLogger("thyra.drain")

# Per-process Hebbian turn windows. Resets when the process exits, which is
# acceptable — the stop hook is a new process each turn so the window is
# always length-1, but Hebbian still runs without crashing.
_turn_windows: dict[str, deque] = {}

_STALE_PROCESSING_SECONDS = 30


def drain_queue(
    db_path: str,
    budget_ms: float | None = None,
    rules_only: bool = True,
) -> int:
    """Process all pending *.json files in delta_queue/, in timestamp order.

    Before touching each file, renames it into delta_queue/processing/ as an
    atomic lock so concurrent callers never apply the same delta twice.
    Deletes on success; renames back on failure.

    budget_ms: wall-clock spending cap in milliseconds; None = no limit.
    rules_only: keep formation in rules-only mode (avoids ML model loads).

    Returns count of deltas processed.
    """
    queue_dir = pathlib.Path(db_path).parent / "delta_queue"
    if not queue_dir.exists():
        return 0

    processing_dir = queue_dir / "processing"
    processing_dir.mkdir(exist_ok=True)

    _reclaim_stale(processing_dir, queue_dir)

    files = sorted(queue_dir.glob("*.json"))
    start = time.monotonic()
    processed = 0

    for fpath in files:
        if budget_ms is not None and (time.monotonic() - start) * 1000 >= budget_ms:
            break
        dest = processing_dir / fpath.name
        try:
            fpath.rename(dest)
        except (FileNotFoundError, OSError):
            continue  # another process claimed it first
        try:
            with open(dest, encoding="utf-8") as f:
                data = json.load(f)
            from thyra.models.delta import DeltaEvent

            delta = DeltaEvent.from_dict(data)
            _apply_delta(delta, db_path, rules_only=rules_only)
            dest.unlink(missing_ok=True)
            processed += 1
        except Exception as exc:
            log.warning("drain_queue: failed on %s: %s", dest.name, exc)
            try:
                dest.rename(fpath)
            except Exception:
                pass

    return processed


def run_nightly_for_all(db_path: str) -> dict[str, dict]:
    """Run the nightly sweep for every active (user_id, agent_id) pair that is overdue.

    A pair is overdue when its last_nightly flag is older than NIGHTLY_INTERVAL_HOURS.
    Opens its own connection so it is safe to call from scripts.
    Returns {"user:agent": sweep_summary, ...}.
    """
    import sqlite3

    from thyra.config import NIGHTLY_INTERVAL_HOURS
    from thyra.consolidation.nightly import run_nightly_sweep
    from thyra.models.memory import get_flag

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        pairs = conn.execute(
            "SELECT DISTINCT user_id, agent_id FROM memories WHERE archived=0"
        ).fetchall()
        now_s = time.time()
        results: dict[str, dict] = {}
        for row in pairs:
            uid, aid = row["user_id"], row["agent_id"]
            key = f"{uid}:{aid}"
            try:
                last_ms = float(get_flag(conn, "last_nightly", uid, aid, default="0"))
                age_s = now_s - last_ms / 1000.0
                if age_s < NIGHTLY_INTERVAL_HOURS * 3600:
                    results[key] = {"skipped": 1}
                    continue
                results[key] = run_nightly_sweep(conn, uid, aid)
            except Exception as exc:
                log.warning("run_nightly_for_all: %s: %s", key, exc)
                results[key] = {"error": str(exc)}
    finally:
        conn.close()
    return results


# ── Internals ─────────────────────────────────────────────────────────────────


def _reclaim_stale(processing_dir: pathlib.Path, queue_dir: pathlib.Path) -> None:
    now = time.time()
    for stale in processing_dir.glob("*.json"):
        try:
            if now - stale.stat().st_mtime > _STALE_PROCESSING_SECONDS:
                stale.rename(queue_dir / stale.name)
        except Exception:
            pass


def _apply_delta(delta, db_path: str, *, rules_only: bool = True) -> None:
    """Full consolidation pipeline for one DeltaEvent (formation → archive)."""
    from thyra.db.connection import DBConnection
    from thyra.consolidation.decay import recompute_and_update, archive_check
    from thyra.consolidation.reinforcement import apply_reinforcement
    from thyra.consolidation.edges import update_cue_edges, hebbian_association
    from thyra.consolidation.situation import crystallize_situations
    from thyra.recall.cache import HOT_CACHE

    conn = DBConnection.get(db_path)

    # Idempotency — skip if this turn was already processed
    if delta.turn_id:
        if conn.execute(
            "SELECT 1 FROM processed_turns WHERE turn_id=?", (delta.turn_id,)
        ).fetchone():
            return

    # Formation (rules-only so no ML model loads block the caller)
    new_memory_ids: list[str] = []
    try:
        from thyra.formation.pipeline import run_formation_pipeline
        from thyra.formation.refiner import set_rules_only

        if rules_only:
            set_rules_only(True)
        try:
            actions = run_formation_pipeline(conn, delta)
        finally:
            if rules_only:
                set_rules_only(False)
        new_memory_ids = [mid for action, mid in actions if action == "created"]
    except ImportError:
        pass
    except Exception as exc:
        log.warning("Formation pipeline error: %s", exc)

    # Synonym expansion for newly formed memories
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

    # Lazy decay on served memories
    recompute_and_update(conn, delta.memories_served, delta.user_id, delta.agent_id)

    # Reinforcement
    apply_reinforcement(conn, delta)

    # Cue edge updates
    update_cue_edges(conn, delta)

    # Hebbian association — 3-turn sliding window (per-process)
    pair_key = f"{delta.user_id}:{delta.agent_id}"
    if pair_key not in _turn_windows:
        _turn_windows[pair_key] = deque(maxlen=3)
    window = _turn_windows[pair_key]
    window.append(delta)
    hebbian_association(conn, list(window), delta.user_id, delta.agent_id)

    # Situation crystallization
    crystallize_situations(conn, list(window), delta.user_id, delta.agent_id)

    # Archive check LAST (boosted memories must not be archived in the same batch)
    archive_check(conn, delta.user_id, delta.agent_id)

    # Commit + mark processed
    with conn:
        if delta.turn_id:
            conn.execute(
                "INSERT OR IGNORE INTO processed_turns (turn_id, processed_at) VALUES (?,?)",
                (delta.turn_id, int(time.time() * 1000)),
            )
        _log_turn(conn, delta)

    HOT_CACHE.invalidate(f"snapshot:{delta.user_id}:{delta.agent_id}")


def _log_turn(conn, delta) -> None:
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
