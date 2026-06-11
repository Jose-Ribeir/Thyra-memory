"""Reinforcement: strengthen memories actually used this turn.

A <memories_used> declaration is an unreliable self-report -- models over-report,
naming memories they merely glanced at. So a bare declaration no longer earns the
full boost. Reinforcement is graded by *corroboration*:

  1. Declared + corroborated -- the memory's content overlaps the response text or
     this turn's tool activity. Full REINFORCE_BASE boost; probationary memories
     graduate. Highest confidence it was genuinely used.
  2. Declared but uncorroborated -- claimed, no evidence. Reduced boost, NEVER
     graduates. Behavioral memories (preferences/constraints/...) keep a higher floor
     because they legitimately leave no lexical trace; everything else is discounted
     harder. This is the anti-over-report path.
  3. Inferred (not declared) -- served memory whose content overlaps the evidence.
     Smaller boost (OVERLAP_INFER_MULT). Reinforcement without any tag.
  4. Surfaced only -- served but neither declared nor overlapping. Tiny SURFACED_BOOST.

Corroboration evidence = the assistant response text PLUS this turn's tool calls and
results, so a memory that shaped a search query or file path still counts even when
it never appears in the prose.
"""

from __future__ import annotations

import re
import sqlite3
import time

from thyra.config import (
    BEHAVIORAL_CATEGORIES,
    CATEGORY_MULTIPLIERS,
    DECLARED_UNCORROBORATED_BEHAVIORAL_MULT,
    DECLARED_UNCORROBORATED_MULT,
    OVERLAP_INFER_MULT,
    OVERLAP_PRIMARY_THRESHOLD,
    REINFORCE_BASE,
    STRENGTH_CAP,
    SURFACED_BOOST,
)
from thyra.models.delta import DeltaEvent
from thyra.models.memory import get_memory, graduate_memory, update_memory_strength

_WORD_RE = re.compile(r"\b[a-z]{4,}\b")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) if text else set()


def apply_reinforcement(
    conn: sqlite3.Connection,
    delta: DeltaEvent,
) -> dict[str, str]:
    """Apply graded reinforcement. Returns {memory_id: action} for logging."""
    user_id = delta.user_id
    agent_id = delta.agent_id
    now = int(time.time() * 1000)

    served_set = set(delta.memories_served)
    declared_set = set(delta.memories_declared)
    tenant_ids = _get_tenant_ids(conn, served_set, user_id, agent_id)
    served_tenant = served_set & tenant_ids

    # Fetch each served+owned memory once.
    served_recs: dict[str, object] = {}
    for mem_id in served_tenant:
        rec = get_memory(conn, mem_id, user_id, agent_id)
        if rec is not None:
            served_recs[mem_id] = rec

    # Anti-spoof: a declaration only counts for a memory actually served to us.
    valid_declared = declared_set & served_tenant

    # Evidence for corroboration: the response text PLUS this turn's tool activity.
    evidence_parts = [delta.raw_assistant_text or ""]
    tool_activity = getattr(delta, "tool_activity", "") or ""
    if tool_activity:
        evidence_parts.append(tool_activity)
    evidence_words = _words(" ".join(evidence_parts))

    actions: dict[str, str] = {}

    # -- Signal 1: declarations, graded by corroboration --
    for mem_id in valid_declared:
        rec = served_recs.get(mem_id)
        if rec is None:
            continue
        mult = CATEGORY_MULTIPLIERS.get(rec.category, 1.0)
        if _corroborated(rec, evidence_words):
            new_strength = min(STRENGTH_CAP, rec.base_strength + REINFORCE_BASE * mult)
            if rec.probationary:
                graduate_memory(
                    conn,
                    mem_id,
                    new_strength,
                    _category_decay(rec.category),
                    now,
                    user_id,
                    agent_id,
                )
                actions[mem_id] = "graduated"
            else:
                update_memory_strength(
                    conn, mem_id, new_strength, now, user_id, agent_id
                )
                conn.execute(
                    "UPDATE memories SET use_count = use_count + 1 WHERE id=? AND user_id=? AND agent_id=?",
                    (mem_id, user_id, agent_id),
                )
                actions[mem_id] = "reinforced"
        else:
            # Claimed but unverified -- reduced boost, NEVER graduate.
            if rec.category in BEHAVIORAL_CATEGORIES:
                claim_mult = DECLARED_UNCORROBORATED_BEHAVIORAL_MULT
                tag = "claimed-behavioral"
            else:
                claim_mult = DECLARED_UNCORROBORATED_MULT
                tag = "claimed"
            boost = REINFORCE_BASE * mult * claim_mult
            new_strength = min(STRENGTH_CAP, rec.base_strength + boost)
            update_memory_strength(conn, mem_id, new_strength, now, user_id, agent_id)
            actions[mem_id] = tag

    # -- Signal 2: overlap inference for NON-declared served memories --
    inferred_ids: set[str] = set()
    for mem_id, rec in served_recs.items():
        if mem_id in valid_declared:
            continue
        if not _corroborated(rec, evidence_words):
            continue
        inferred_ids.add(mem_id)
        mult = CATEGORY_MULTIPLIERS.get(rec.category, 1.0)
        boost = REINFORCE_BASE * OVERLAP_INFER_MULT * mult
        new_strength = min(STRENGTH_CAP, rec.base_strength + boost)
        if rec.probationary:
            graduate_memory(
                conn,
                mem_id,
                new_strength,
                _category_decay(rec.category),
                now,
                user_id,
                agent_id,
            )
            actions[mem_id] = "graduated-overlap"
        else:
            update_memory_strength(conn, mem_id, new_strength, now, user_id, agent_id)
            conn.execute(
                "UPDATE memories SET use_count = use_count + 1 WHERE id=? AND user_id=? AND agent_id=?",
                (mem_id, user_id, agent_id),
            )
            actions[mem_id] = "reinforced-overlap"

    # -- Signal 3: surfaced only (served, neither declared nor overlapping) --
    for mem_id, rec in served_recs.items():
        if mem_id in valid_declared or mem_id in inferred_ids:
            continue
        new_strength = min(STRENGTH_CAP, rec.base_strength + SURFACED_BOOST)
        update_memory_strength(conn, mem_id, new_strength, now, user_id, agent_id)
        actions[mem_id] = "surfaced"

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


def _corroborated(rec, evidence_words: set[str]) -> bool:
    """True if the memory's content is significantly present in the evidence
    (response text + tool activity).

    Short memories (< 4 content words) cannot be judged reliably at small
    denominators, so they are treated as NOT corroborated (conservative -- they
    fall to the reduced 'claimed' boost rather than earning the full boost).
    """
    if rec is None or rec.locked or not evidence_words:
        return False
    mem_words = _words(rec.content)
    if len(mem_words) < 4:
        return False
    overlap = len(evidence_words & mem_words) / len(mem_words)
    return overlap >= OVERLAP_PRIMARY_THRESHOLD
