"""recall_pipeline() — the hot path called by pre_turn.py before every response."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time

from thyra.config import (
    GAMMA_DEFAULT,
    RECALL_INTENT_ARCHIVE_LIMIT,
    RECALL_INTENT_BUDGET_MULT,
    RECALL_INTENT_GAMMA_RELAX,
    RESURRECTION_THRESHOLD,
    SCORE_FLOOR,
    THYRA_AGENT_ID,
    THYRA_DB_PATH,
    THYRA_USER_ID,
    TOKEN_BUDGET,
)
from thyra.models.memory import (
    MemoryRecord,
    compute_base_level,
    get_flag,
    list_active_memories,
    load_assoc_edge_map,
    load_cue_edge_map,
    load_situation_edges,
)
from thyra.recall.cache import HOT_CACHE
from thyra.recall.cue_extractor import compute_idf, extract_cues
from thyra.recall.injector import format_injection
from thyra.recall.scorer import score_memories
from thyra.recall.selector import greedy_select

# ── Recall intent detection ────────────────────────────────────────────────────

_RECALL_INTENT_RE = re.compile(
    r"\b(?:"
    r"remember when|remember that time|do you remember|did we (?:discuss|talk|mention)|"
    r"you (?:told|said|mentioned)|as (?:I|we) mentioned|"
    r"didn't we|haven't we|have we (?:already|ever)|"
    r"recall (?:when|that)|think back|go back to|earlier (?:you|we)"
    r")\b",
    re.IGNORECASE,
)


def detect_recall_intent(text: str) -> bool:
    return bool(_RECALL_INTENT_RE.search(text))


# ── Snapshot loading ───────────────────────────────────────────────────────────


def _snapshot_key(user_id: str, agent_id: str) -> str:
    return f"snapshot:{user_id}:{agent_id}"


def _load_snapshot(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> tuple[list[MemoryRecord], dict, dict, list]:
    """Load (or rebuild) the in-memory snapshot for a tenant pair."""
    key = _snapshot_key(user_id, agent_id)
    cached = HOT_CACHE.get(key)
    if cached is not None:
        return cached

    now_ms = int(time.time() * 1000)
    memories = list_active_memories(conn, user_id, agent_id)
    for rec in memories:
        rec.computed_base_level = compute_base_level(
            rec.base_strength, rec.decay_rate, rec.last_access, now_ms
        )
    cue_edge_map = load_cue_edge_map(conn, user_id, agent_id)
    assoc_edge_map = load_assoc_edge_map(conn, user_id, agent_id)
    sit_edges = load_situation_edges(conn, user_id, agent_id)

    snapshot = (memories, cue_edge_map, assoc_edge_map, sit_edges)
    HOT_CACHE.set(key, snapshot)
    return snapshot


# ── Archived recall (recall-intent only) ──────────────────────────────────────


def _fetch_archived_matches(
    conn: sqlite3.Connection,
    cues: list[str],
    user_id: str,
    agent_id: str,
    limit: int = RECALL_INTENT_ARCHIVE_LIMIT,
) -> list[MemoryRecord]:
    if not cues:
        return []
    placeholders = ",".join("?" * len(cues))
    rows = conn.execute(
        f"""SELECT DISTINCT m.* FROM memories m
            JOIN cue_edges ce ON m.id = ce.memory_id
            WHERE ce.cue_id IN ({placeholders})
              AND m.user_id=? AND m.agent_id=? AND m.archived=1
            ORDER BY ce.weight DESC
            LIMIT ?""",
        (*cues, user_id, agent_id, limit),
    ).fetchall()
    now_ms = int(time.time() * 1000)
    return [MemoryRecord.from_row(r, now_ms) for r in rows]


# ── Turn state persistence (for stop_hook / thyra_end_turn) ──────────────────
#
# State files live in the thyra DATA directory (next to the DB), NOT in TEMP.
# Reason: the hook subprocess and the MCP server may resolve TEMP differently
# across process boundaries in CCD mode.  THYRA_DB_PATH is always the same
# canonical path for both, so parent(THYRA_DB_PATH) is a reliable shared dir.


def _turn_state_path(session_id: str) -> str:
    """Canonical path for a turn's state file — data dir, not TEMP."""
    import pathlib

    # Sanitize session_id so it is safe as a filename component.
    safe_sid = (session_id or "unknown").replace("/", "_").replace("\\", "_")
    data_dir = pathlib.Path(THYRA_DB_PATH).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / f"turn_state_{safe_sid}.json")


def _store_turn_state(
    session_id: str,
    turn_id: str,
    served_ids: list[str],
    cues_fired: list[str],
) -> None:
    state = {
        "turn_id": turn_id,
        "served_ids": served_ids,
        "cues_fired": cues_fired,
    }
    try:
        with open(_turn_state_path(session_id), "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


# ── Category weights (stub: uniform until Stage 7) ───────────────────────────


def _get_category_weights(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> dict[str, float]:
    """Return per-category weights used by the scorer's category_presence formula.

    Weight = max(relevance_floor, activation_score * 0.5), matching CategoryManager.
    Protected categories always return their full floor; emergent categories fade as
    their activation_score decays nightly.
    """
    rows = conn.execute(
        "SELECT cat_id, relevance_floor, activation_score FROM categories WHERE user_id=? AND agent_id=?",
        (user_id, agent_id),
    ).fetchall()
    return {
        r["cat_id"]: max(r["relevance_floor"], r["activation_score"] * 0.5)
        for r in rows
    }


# ── Main hot path ─────────────────────────────────────────────────────────────


def recall_pipeline(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
    prompt: str,
    session_id: str,
    turn_id: str,
) -> tuple[str, list[str]]:
    """Run full recall; return (injection_xml, served_ids).

    Any exception returns ("", []) — the turn must never fail.
    """
    try:
        # Master switch check (cheap: reads from cache or single DB row)
        enabled = get_flag(conn, "system_enabled", user_id, agent_id)
        if enabled.lower() != "true":
            return ("", [])

        now_ms = int(time.time() * 1000)
        is_intent = detect_recall_intent(prompt)

        # Snapshot
        memories, cue_edge_map, assoc_edge_map, sit_edges = _load_snapshot(
            conn, user_id, agent_id
        )
        if not memories and not is_intent:
            return ("", [])

        # Cue extraction
        cues = extract_cues(prompt)
        if not cues:
            return ("", [])

        # Per-agent IDF
        idf = compute_idf(conn, cues, user_id, agent_id)

        # Category weights
        cat_weights = _get_category_weights(conn, user_id, agent_id)

        # Score
        scored = score_memories(
            memories,
            cues,
            cue_edge_map,
            assoc_edge_map,
            sit_edges,
            idf,
            cat_weights,
            now_ms,
        )

        # Budget + gamma adjustments for recall intent
        budget = TOKEN_BUDGET
        gamma = GAMMA_DEFAULT
        floor = SCORE_FLOOR
        if is_intent:
            budget = int(budget * RECALL_INTENT_BUDGET_MULT)
            gamma = min(1.0, gamma + RECALL_INTENT_GAMMA_RELAX)
            floor = floor * 0.5

        # Max memories setting (0 = unlimited)
        try:
            max_count = int(get_flag(conn, "max_memories", user_id, agent_id) or "0")
        except (ValueError, TypeError):
            max_count = 0

        # Select
        selected = greedy_select(
            scored,
            token_budget=budget,
            gamma=gamma,
            score_floor=floor,
            max_count=max_count,
        )

        # Recall-intent: bounded archived lookup
        if is_intent:
            archived = _fetch_archived_matches(conn, cues, user_id, agent_id)
            # Add archived memories not already selected
            selected_ids = {r.id for r in selected}
            for rec in archived:
                level = compute_base_level(
                    rec.base_strength, rec.decay_rate, rec.last_access, now_ms
                )
                # Compute rough activation for resurrection check
                activation = sum(
                    ew * idf.get(cue, 0.15)
                    for cue in cues
                    for mid, ew in cue_edge_map.get(cue, [])
                    if mid == rec.id
                )
                if rec.id not in selected_ids and activation > RESURRECTION_THRESHOLD:
                    selected.append(rec)
                    selected_ids.add(rec.id)

        if not selected:
            return ("", [])

        served_ids = [r.id for r in selected]
        xml = format_injection(selected, agent_id)
        _store_turn_state(session_id, turn_id, served_ids, cues)
        return (xml, served_ids)

    except Exception:
        return ("", [])
