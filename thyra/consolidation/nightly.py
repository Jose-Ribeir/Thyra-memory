"""Full nightly sweep — runs once per ~24h per (user_id, agent_id) pair."""

from __future__ import annotations

import logging
import sqlite3
import time

from thyra.config import (
    ASSOC_NIGHTLY_DECAY,
    ASSOC_PRUNE_THRESHOLD,
    ARCHIVE_THRESHOLD,
    HARD_DELETE_DAYS,
    NIGHTLY_INTERVAL_HOURS,
    PROBATIONARY_AUTOPURGE_DAYS,
    TURN_LOG_RETENTION_DAYS,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)

log = logging.getLogger("thyra.nightly")


def run_nightly_sweep(
    conn: sqlite3.Connection,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> dict:
    """Execute the full nightly sweep for one (user_id, agent_id) pair.

    Returns a summary dict with counts of each action taken.
    """
    now_ms = int(time.time() * 1000)
    summary: dict[str, int] = {}

    summary["decayed"] = _full_decay_pass(conn, user_id, agent_id, now_ms)
    summary["probationary_purged"] = _autopurge_unused_probationary(
        conn, user_id, agent_id, now_ms
    )
    summary["archived"] = _archive_below_threshold(conn, user_id, agent_id, now_ms)
    summary["hard_deleted"] = _hard_delete_old_archived(conn, user_id, agent_id, now_ms)
    summary["cue_edges_pruned"] = _prune_cue_edges(conn, user_id, agent_id)
    summary["hub_cue_edges_pruned"] = _prune_hub_cue_edges(conn, user_id, agent_id)
    summary["orphan_cues_pruned"] = _prune_orphan_cues(conn, user_id, agent_id)
    summary["assoc_edges_pruned"] = _decay_and_prune_assoc_edges(
        conn, user_id, agent_id
    )
    summary["category_scores_decayed"] = _decay_category_scores(conn, user_id, agent_id)
    summary["emergent_categories"] = _run_category_detection(conn, user_id, agent_id)
    summary["categories_dissolved"] = _enforce_category_cap(conn, user_id, agent_id)
    summary["turn_log_pruned"] = _prune_turn_log(conn, user_id, agent_id, now_ms)

    _update_nightly_flag(conn, user_id, agent_id, now_ms)
    conn.commit()

    log.info("Nightly sweep complete for %s:%s — %s", user_id, agent_id, summary)
    return summary


# ── Individual sweep steps ─────────────────────────────────────────────────────


def _full_decay_pass(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
    now_ms: int,
) -> int:
    """Recompute base_strength for ALL active memories (not just accessed ones)."""
    from math import exp

    rows = conn.execute(
        "SELECT id, base_strength, decay_rate, last_access FROM memories "
        "WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchall()
    count = 0
    with conn:
        for row in rows:
            days = (now_ms - row["last_access"]) / 86_400_000
            new_strength = row["base_strength"] * exp(-row["decay_rate"] * days)
            if abs(new_strength - row["base_strength"]) > 1e-6:
                # Must also update last_access to now_ms; otherwise the next lazy-decay
                # call re-applies all the same elapsed days on top of the already-decayed
                # base_strength, producing exponential double-decay.
                conn.execute(
                    "UPDATE memories SET base_strength=?, last_access=? WHERE id=? AND user_id=? AND agent_id=?",
                    (new_strength, now_ms, row["id"], user_id, agent_id),
                )
                count += 1
    return count


def _autopurge_unused_probationary(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
    now_ms: int,
) -> int:
    """A-Mem-style auto-purge (L3, master §9.5).

    Archive probationary memories that were never used (use_count == 0) and are
    older than PROBATIONARY_AUTOPURGE_DAYS, without waiting for the slow strength
    curve to cross ARCHIVE_THRESHOLD.  This is what makes weak-signal admits (on
    the steep WEAK_ADMIT_DECAY horizon) disappear in days rather than weeks.  A
    probationary memory that *was* re-cited (use_count > 0) is spared — it is on
    the graduation path and handled by the normal lifecycle.
    """
    cutoff_ms = now_ms - PROBATIONARY_AUTOPURGE_DAYS * 86_400_000
    with conn:
        cur = conn.execute(
            "UPDATE memories SET archived=1, archived_at=? "
            "WHERE user_id=? AND agent_id=? AND archived=0 "
            "AND probationary=1 AND use_count=0 AND created_at < ?",
            (now_ms, user_id, agent_id, cutoff_ms),
        )
    return cur.rowcount


def _archive_below_threshold(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
    now_ms: int,
) -> int:
    with conn:
        cur = conn.execute(
            "UPDATE memories SET archived=1, archived_at=? "
            "WHERE user_id=? AND agent_id=? AND archived=0 AND base_strength < ?",
            (now_ms, user_id, agent_id, ARCHIVE_THRESHOLD),
        )
    return cur.rowcount


def _hard_delete_old_archived(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
    now_ms: int,
) -> int:
    cutoff_ms = now_ms - HARD_DELETE_DAYS * 86_400_000
    with conn:
        cur = conn.execute(
            "DELETE FROM memories WHERE user_id=? AND agent_id=? AND archived=1 AND archived_at < ?",
            (user_id, agent_id, cutoff_ms),
        )
    return cur.rowcount


def _prune_cue_edges(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> int:
    from thyra.consolidation.edges import prune_weak_cue_edges

    return prune_weak_cue_edges(conn, user_id, agent_id)


def _prune_hub_cue_edges(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> int:
    """Delete cue edges for hub cues (df > HUB_CUE_FRACTION * M).

    Hub cues appear in too many memories to be discriminative — IDF pushes
    their recall contribution near zero. Removing their edges keeps
    load_cue_edge_map lean and avoids wasted scoring iterations.
    """
    try:
        from thyra.categories.crystallizer import _find_hub_cues

        hub_cues = _find_hub_cues(conn, user_id, agent_id)
        if not hub_cues:
            return 0
        placeholders = ",".join("?" * len(hub_cues))
        # Decrement df in cue_nodes before deleting the edges
        rows = conn.execute(
            f"""SELECT cue_id, COUNT(*) as cnt FROM cue_edges
                WHERE cue_id IN ({placeholders}) AND user_id=? AND agent_id=?
                GROUP BY cue_id""",
            (*hub_cues, user_id, agent_id),
        ).fetchall()
        with conn:
            for row in rows:
                conn.execute(
                    "UPDATE cue_nodes SET df = CASE WHEN df < ? THEN 0 ELSE df - ? END"
                    " WHERE cue_id=? AND user_id=? AND agent_id=?",
                    (row["cnt"], row["cnt"], row["cue_id"], user_id, agent_id),
                )
            cur = conn.execute(
                f"""DELETE FROM cue_edges WHERE cue_id IN ({placeholders})
                    AND user_id=? AND agent_id=?""",
                (*hub_cues, user_id, agent_id),
            )
        return cur.rowcount
    except Exception as exc:
        log.warning("Hub cue edge pruning error: %s", exc)
        return 0


def _prune_orphan_cues(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> int:
    with conn:
        cur = conn.execute(
            """DELETE FROM cue_nodes WHERE user_id=? AND agent_id=?
               AND cue_id NOT IN (
                   SELECT DISTINCT cue_id FROM cue_edges
                   WHERE user_id=? AND agent_id=?
               )""",
            (user_id, agent_id, user_id, agent_id),
        )
    return cur.rowcount


def _decay_and_prune_assoc_edges(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> int:
    with conn:
        conn.execute(
            "UPDATE association_edges SET weight=weight*? "
            "WHERE user_id=? AND agent_id=?",
            (ASSOC_NIGHTLY_DECAY, user_id, agent_id),
        )
        cur = conn.execute(
            "DELETE FROM association_edges WHERE user_id=? AND agent_id=? AND weight < ?",
            (user_id, agent_id, ASSOC_PRUNE_THRESHOLD),
        )
    return cur.rowcount


def _decay_category_scores(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> int:
    """Decay activation_score for non-protected categories each nightly cycle.

    Protected categories (is_protected=1) keep activation_score=1.0 permanently.
    Emergent categories decay toward zero if unused; at 0.05 they are dissolved.
    """
    with conn:
        cur = conn.execute(
            """UPDATE categories SET activation_score = MAX(0.05, activation_score * 0.95)
               WHERE user_id=? AND agent_id=? AND is_protected=0""",
            (user_id, agent_id),
        )
    return cur.rowcount


def _run_category_detection(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> int:
    try:
        from thyra.categories.crystallizer import detect_emergent_categories

        new_cats = detect_emergent_categories(conn, user_id, agent_id)
        return len(new_cats)
    except Exception as exc:
        log.warning("Category detection error: %s", exc)
        return 0


def _enforce_category_cap(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> int:
    try:
        from thyra.categories.manager import CategoryManager

        mgr = CategoryManager(conn, user_id, agent_id)
        dissolved = mgr.soft_cap_enforce()
        return len(dissolved)
    except Exception as exc:
        log.warning("Category cap error: %s", exc)
        return 0


def _prune_turn_log(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
    now_ms: int,
) -> int:
    cutoff_ms = now_ms - TURN_LOG_RETENTION_DAYS * 86_400_000
    with conn:
        cur = conn.execute(
            "DELETE FROM turn_log WHERE user_id=? AND agent_id=? AND created_at < ?",
            (user_id, agent_id, cutoff_ms),
        )
    return cur.rowcount


def _update_nightly_flag(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
    now_ms: int,
) -> None:
    from thyra.models.memory import set_flag

    set_flag(conn, "last_nightly", str(now_ms), user_id, agent_id)
