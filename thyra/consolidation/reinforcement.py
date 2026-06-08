"""Reinforcement: strengthen memories declared used this turn."""

from __future__ import annotations

import re
import sqlite3
import time

from thyra.config import (
    CATEGORY_MULTIPLIERS,
    OVERLAP_CONFIRMATORY_CAP,
    OVERLAP_INFER_MULT,
    OVERLAP_PRIMARY_THRESHOLD,
    REINFORCE_BASE,
    STRENGTH_CAP,
    SURFACED_BOOST,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)
from thyra.models.delta import DeltaEvent
from thyra.models.memory import get_memory, graduate_memory, update_memory_strength


def apply_reinforcement(
    conn: sqlite3.Connection,
    delta: DeltaEvent,
) -> dict[str, str]:
    """Apply reinforcement from the usage manifest and API-level overlap inference.

    Three signals, in priority order:
      1. Declaration (LLM <memories_used> tag): full REINFORCE_BASE boost.
         Anti-spoofed: only declared IDs that were actually served count.
      2. Overlap inference (API-level, no LLM needed): served memories whose
         content words appear significantly in the response get REINFORCE_BASE *
         OVERLAP_INFER_MULT boost. This fires even when the LLM declares nothing,
         making reinforcement work without any tool call or tag.
      3. Surfaced-only: served but neither declared nor overlapping → tiny SURFACED_BOOST.

    Declaration (1) and overlap (2) are not mutually exclusive — a memory that is
    both declared AND overlapping gets (1); overlap is the safety net for (2) only.

    Returns a map {memory_id: action} for logging.
    """
    user_id = delta.user_id
    agent_id = delta.agent_id
    now = int(time.time() * 1000)

    served_set = set(delta.memories_served)
    declared_set = set(delta.memories_declared)

    # Tenant check — restrict all operations to IDs owned by this (user, agent).
    tenant_ids = _get_tenant_ids(conn, served_set, user_id, agent_id)

    # Signal 1: declaration anti-spoofed against served set.
    valid_ids = (declared_set & served_set) & tenant_ids

    # Signal 2: API-level overlap inference — works even with no declaration.
    inferred_ids: set[str] = set()
    if delta.raw_assistant_text and served_set:
        inferred_ids = _infer_used_by_overlap(
            conn,
            delta.raw_assistant_text,
            served_set & tenant_ids,
            user_id,
            agent_id,
        )
        # Declaration already covers these; overlap is the fallback signal only.
        inferred_ids -= valid_ids

    actions: dict[str, str] = {}

    # Apply Signal 1 — full boost + graduation.
    for mem_id in valid_ids:
        rec = get_memory(conn, mem_id, user_id, agent_id)
        if rec is None:
            continue
        mult = CATEGORY_MULTIPLIERS.get(rec.category, 1.0)
        new_strength = min(STRENGTH_CAP, rec.base_strength + REINFORCE_BASE * mult)
        if rec.probationary:
            cat_decay = _category_decay(rec.category)
            graduate_memory(
                conn, mem_id, new_strength, cat_decay, now, user_id, agent_id
            )
            actions[mem_id] = "graduated"
        else:
            update_memory_strength(conn, mem_id, new_strength, now, user_id, agent_id)
            conn.execute(
                "UPDATE memories SET use_count = use_count + 1 WHERE id=? AND user_id=? AND agent_id=?",
                (mem_id, user_id, agent_id),
            )
            actions[mem_id] = "reinforced"

    # Apply Signal 2 — overlap-inferred boost (smaller than full declaration).
    for mem_id in inferred_ids:
        rec = get_memory(conn, mem_id, user_id, agent_id)
        if rec is None:
            continue
        mult = CATEGORY_MULTIPLIERS.get(rec.category, 1.0)
        boost = REINFORCE_BASE * OVERLAP_INFER_MULT * mult
        new_strength = min(STRENGTH_CAP, rec.base_strength + boost)
        if rec.probationary:
            cat_decay = _category_decay(rec.category)
            graduate_memory(
                conn, mem_id, new_strength, cat_decay, now, user_id, agent_id
            )
            actions[mem_id] = "graduated-overlap"
        else:
            update_memory_strength(conn, mem_id, new_strength, now, user_id, agent_id)
            conn.execute(
                "UPDATE memories SET use_count = use_count + 1 WHERE id=? AND user_id=? AND agent_id=?",
                (mem_id, user_id, agent_id),
            )
            actions[mem_id] = "reinforced-overlap"

    # Apply Signal 3 — surfaced only (neither declared nor overlapping).
    surfaced_only = (served_set & tenant_ids) - valid_ids - inferred_ids
    for mem_id in surfaced_only:
        rec = get_memory(conn, mem_id, user_id, agent_id)
        if rec is None:
            continue
        new_strength = min(STRENGTH_CAP, rec.base_strength + SURFACED_BOOST)
        update_memory_strength(conn, mem_id, new_strength, now, user_id, agent_id)
        actions[mem_id] = "surfaced"

    # Confirmatory overlap on top of declared memories (additive cap, as before).
    if delta.raw_assistant_text and valid_ids:
        _apply_overlap_confirmation(conn, delta, valid_ids, now, user_id, agent_id)

    return actions


def _get_tenant_ids(
    conn: sqlite3.Connection,
    ids: set[str],
    user_id: str,
    agent_id: str,
) -> set[str]:
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id FROM memories WHERE id IN ({placeholders}) AND user_id=? AND agent_id=?",
        (*ids, user_id, agent_id),
    ).fetchall()
    return {r["id"] for r in rows}


def _category_decay(category: str) -> float:
    from thyra.config import DECAY_CONSTRAINTS, DECAY_IDENTITY, DECAY_EXPLICIT

    if category in ("constraints",):
        return DECAY_CONSTRAINTS
    if category in ("identity",):
        return DECAY_IDENTITY
    return DECAY_EXPLICIT


def _infer_used_by_overlap(
    conn: sqlite3.Connection,
    response_text: str,
    candidate_ids: set[str],
    user_id: str,
    agent_id: str,
) -> set[str]:
    """Return the subset of candidate_ids whose content is significantly present
    in response_text (API-level signal: we know what went in and what came out).

    Uses OVERLAP_PRIMARY_THRESHOLD (fraction of memory's content words found in
    the response) as the gate.  Short memories (< 4 content words) are skipped
    because their overlap fraction is noisy at small denominators.
    """
    if not response_text or not candidate_ids:
        return set()
    response_words = set(re.findall(r"\b[a-z]{4,}\b", response_text.lower()))
    if not response_words:
        return set()
    used: set[str] = set()
    for mem_id in candidate_ids:
        rec = get_memory(conn, mem_id, user_id, agent_id)
        if rec is None or rec.locked:
            continue
        mem_words = set(re.findall(r"\b[a-z]{4,}\b", rec.content.lower()))
        if len(mem_words) < 4:
            continue
        overlap = len(response_words & mem_words) / len(mem_words)
        if overlap >= OVERLAP_PRIMARY_THRESHOLD:
            used.add(mem_id)
    return used


def _apply_overlap_confirmation(
    conn: sqlite3.Connection,
    delta: DeltaEvent,
    valid_ids: set[str],
    now: int,
    user_id: str,
    agent_id: str,
) -> None:
    response_words = set(re.findall(r"\b[a-z]{4,}\b", delta.raw_assistant_text.lower()))
    for mem_id in valid_ids:
        rec = get_memory(conn, mem_id, user_id, agent_id)
        if rec is None or rec.locked:
            continue
        mem_words = set(re.findall(r"\b[a-z]{4,}\b", rec.content.lower()))
        if not mem_words:
            continue
        overlap = len(response_words & mem_words) / len(mem_words)
        if overlap > 0.3:
            boost = min(OVERLAP_CONFIRMATORY_CAP, overlap * 0.1)
            new_strength = min(STRENGTH_CAP, rec.base_strength + boost)
            update_memory_strength(conn, mem_id, new_strength, now, user_id, agent_id)
