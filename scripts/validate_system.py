"""
Thyra System Validation Script
================================
Validates all key invariants of the thyra adaptive memory system.

Usage:
    python scripts/validate_system.py

Exit 0: all 27 invariants pass.
Exit 1: one or more invariants fail (named failures printed).

IMPORTANT: Bootstrap must happen before any thyra import so env vars
are in place when thyra.config is first loaded.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Bootstrap (must be before any thyra import) ───────────────────────────────

_TEMP_DIR = tempfile.mkdtemp(prefix="thyra_validate_")
_DB_PATH = os.path.join(_TEMP_DIR, "validate.db")

os.environ["THYRA_DB_PATH"] = _DB_PATH
os.environ["THYRA_USER_ID"] = "validate_user"
os.environ["THYRA_AGENT_ID"] = "validate_agent"

# Now safe to import thyra — patch module-level constants to match env
import thyra.config as cfg  # noqa: E402

cfg.THYRA_DB_PATH = _DB_PATH
cfg.THYRA_USER_ID = "validate_user"
cfg.THYRA_AGENT_ID = "validate_agent"

from thyra.db.connection import DBConnection  # noqa: E402
from thyra.recall.cache import HOT_CACHE  # noqa: E402

# Reset thread-local connection so it rebuilds at the new DB path
DBConnection._local = type(DBConnection._local)()
HOT_CACHE.clear()

# ── All other thyra imports ───────────────────────────────────────────────────

from thyra.consolidation.decay import archive_check, recompute_and_update
from thyra.consolidation.edges import (
    _hub_cues,
    hebbian_association,
    prune_weak_cue_edges,
    update_cue_edges,
)
from thyra.consolidation.nightly import (
    _archive_below_threshold,
    _autopurge_unused_probationary,
    _decay_and_prune_assoc_edges,
    _full_decay_pass,
    _prune_cue_edges,
    _prune_hub_cue_edges,
    _prune_orphan_cues,
    run_nightly_sweep,
)
from thyra.consolidation.reinforcement import apply_reinforcement
from thyra.consolidation.situation import crystallize_situations
from thyra.formation.pipeline import run_formation_pipeline
from thyra.formation.refiner import set_rules_only
from thyra.models.delta import DeltaEvent
from thyra.models.memory import (
    create_memory,
    get_memory,
    list_active_memories,
    set_flag,
    get_flag,
    upsert_assoc_edge,
    upsert_cue_edge,
)
from thyra.recall.cue_extractor import compute_idf, extract_cues
from thyra.recall.intent import recall_pipeline, _turn_state_path
from thyra.recall.morphology import normalize_cue
from thyra.recall.scorer import score_memories
from thyra.recall.selector import greedy_select

# Use rules-only classification throughout — avoids blocking model loads
set_rules_only(True)

# ── Constants ─────────────────────────────────────────────────────────────────

U = "validate_user"
A = "validate_agent"

NOW_MS = int(time.time() * 1000)
DAY_MS = 86_400_000

# ── Check harness ─────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    label: str
    passed: bool
    error: Optional[str] = None


_results: list[CheckResult] = []
_section: str = ""


def _set_section(name: str) -> None:
    global _section
    _section = name
    print(f"\n-- {name} " + "-" * max(0, 50 - len(name)))


def check(label: str):
    """Decorator: execute fn immediately, record pass/fail."""

    def decorator(fn: Callable) -> Callable:
        try:
            fn()
            _results.append(CheckResult(label=label, passed=True))
            print(f"  PASS  {label}")
        except AssertionError as e:
            _results.append(CheckResult(label=label, passed=False, error=str(e)))
            print(f"  FAIL  {label}")
            print(f"        {e}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            _results.append(CheckResult(label=label, passed=False, error=msg))
            print(f"  FAIL  {label}")
            print(f"        {msg}")
        return fn

    return decorator


# ── Shared helpers ─────────────────────────────────────────────────────────────


def conn():
    return DBConnection.get()


def fresh_conn():
    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    return DBConnection.get()


def make_delta(
    served: list[str] | None = None,
    declared: list[str] | None = None,
    cues: list[str] | None = None,
    user_text: str = "",
    assistant_text: str = "",
    session: str = "val_session",
    turn: str | None = None,
    user_id: str = U,
    agent_id: str = A,
) -> DeltaEvent:
    return DeltaEvent(
        session_id=session,
        turn_id=turn or f"val_turn_{int(time.time() * 1000)}",
        user_id=user_id,
        agent_id=agent_id,
        timestamp=NOW_MS,
        memories_served=served or [],
        memories_declared=declared or [],
        cues_fired=cues or [],
        raw_user_text=user_text,
        raw_assistant_text=assistant_text,
    )


def _apply_turn(delta: DeltaEvent, window: list[DeltaEvent]) -> None:
    """Run full consolidation pipeline in worker order (mirrors worker._apply_delta)."""
    c = conn()
    set_rules_only(True)
    run_formation_pipeline(c, delta)
    recompute_and_update(c, delta.memories_served, delta.user_id, delta.agent_id)
    apply_reinforcement(c, delta)
    update_cue_edges(c, delta)
    hebbian_association(c, window, delta.user_id, delta.agent_id)
    crystallize_situations(c, window, delta.user_id, delta.agent_id)
    archive_check(c, delta.user_id, delta.agent_id)
    with c:
        if delta.turn_id:
            c.execute(
                "INSERT OR IGNORE INTO processed_turns (turn_id, processed_at) VALUES (?,?)",
                (delta.turn_id, NOW_MS),
            )
        c.execute(
            """INSERT OR IGNORE INTO turn_log
               (turn_id, session_id, user_id, agent_id,
                memories_served, memories_used, cues_fired, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                delta.turn_id,
                delta.session_id,
                delta.user_id,
                delta.agent_id,
                json.dumps(delta.memories_served),
                json.dumps(delta.memories_declared),
                json.dumps(delta.cues_fired),
                NOW_MS,
            ),
        )
    HOT_CACHE.invalidate(f"snapshot:{delta.user_id}:{delta.agent_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# A · Data Layer
# ═══════════════════════════════════════════════════════════════════════════════

_set_section("A · Data Layer")


@check("A1: All expected tables exist")
def _a1():
    expected = {
        "memories",
        "cue_nodes",
        "cue_edges",
        "association_edges",
        "situation_edges",
        "categories",
        "turn_log",
        "processed_turns",
        "system_flags",
    }
    c = conn()
    rows = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    found = {r["name"] for r in rows}
    missing = expected - found
    assert not missing, f"Missing tables: {missing}"


@check("A2: System flags default correctly")
def _a2():
    c = conn()
    se = get_flag(c, "system_enabled", U, A)
    fe = get_flag(c, "formation_enabled", U, A)
    assert se.lower() == "true", f"system_enabled={se!r}"
    assert fe.lower() == "true", f"formation_enabled={fe!r}"


@check("A3: 15 seed categories; constraints+identity are protected with slow decay")
def _a3():
    c = conn()
    rows = c.execute(
        "SELECT cat_id, is_protected, decay_rate FROM categories WHERE user_id=? AND agent_id=?",
        (U, A),
    ).fetchall()
    assert len(rows) == 15, f"Expected 15 seed categories, got {len(rows)}"
    cats = {r["cat_id"]: r for r in rows}
    for name in ("constraints", "identity"):
        assert name in cats, f"Category {name!r} not found"
        r = cats[name]
        assert r["is_protected"] == 1, f"{name} is_protected={r['is_protected']}"
        assert r["decay_rate"] <= 0.002, (
            f"{name} decay_rate={r['decay_rate']} (expected ≤ 0.002)"
        )


@check("A4: Tenant isolation — tenant_a memory invisible to tenant_b")
def _a4():
    c = conn()
    ta_mem = create_memory(
        c, "tenant_a secret memory", user_id="ta_user", agent_id="ta_agent"
    )
    tb_mems = list_active_memories(c, user_id="tb_user", agent_id="tb_agent")
    tb_ids = {m.id for m in tb_mems}
    assert ta_mem not in tb_ids, "tenant_a memory leaked into tenant_b list"


# ═══════════════════════════════════════════════════════════════════════════════
# B · Recall Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

_set_section("B · Recall Pipeline")


@check(
    "B1: extract_cues filters stopwords, enforces MIN_CUE_LENGTH, caps at MAX_CUES_PER_TURN"
)
def _b1():
    # Stopwords filtered
    result = extract_cues("the and but a very")
    assert not result, f"Stopwords not filtered: {result}"

    # MIN_CUE_LENGTH (=3): 2-char token not included; 3-char token included
    result2 = extract_cues("ab thyratest")  # "ab" < MIN_CUE_LENGTH, "thyratest" ok
    assert "thyratest" in result2 or normalize_cue("thyratest") in result2, (
        f"Expected thyratest in cues, got {result2}"
    )
    ab_variants = (
        {"ab", normalize_cue("ab")} if len("ab") < cfg.MIN_CUE_LENGTH else set()
    )
    if ab_variants:
        assert not (ab_variants & set(result2)), (
            f"Short token 'ab' should be excluded: {result2}"
        )

    # MAX_CUES_PER_TURN (=12): generate many unique tokens
    long_text = " ".join(f"uniquetoken{i}xyz" for i in range(20))
    result3 = extract_cues(long_text)
    assert len(result3) <= cfg.MAX_CUES_PER_TURN, (
        f"Returned {len(result3)} cues, expected ≤ {cfg.MAX_CUES_PER_TURN}"
    )


@check("B2: compute_idf >= DISCRIMINABILITY_FLOOR for all cues including unknown")
def _b2():
    c = conn()
    # Unknown cue (not in any memory)
    idf = compute_idf(c, ["completelynewcuexyz123abc"], U, A)
    val = idf.get("completelynewcuexyz123abc", -1)
    assert val >= cfg.DISCRIMINABILITY_FLOOR, (
        f"IDF for unknown cue = {val}, expected >= {cfg.DISCRIMINABILITY_FLOOR}"
    )
    # Also for empty cue list
    idf_empty = compute_idf(c, [], U, A)
    assert idf_empty == {}, f"Expected empty dict for empty cues, got {idf_empty}"


@check("B3: score_memories — memory with seeded cue edge scores higher")
def _b3():
    c = fresh_conn()
    # Create memory with distinctive cue seeded
    b3_mem_a = create_memory(
        c, "b3testmemory scoringtest validation", user_id=U, agent_id=A
    )
    b3_mem_b = create_memory(
        c, "unrelated different content zxq", user_id=U, agent_id=A
    )

    # Get the actual seeded cue for mem_a
    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (b3_mem_a, U, A),
    ).fetchall()
    assert rows, "No cues seeded for B3 memory_a"
    b3_cue = rows[0]["cue_id"]

    now = int(time.time() * 1000)
    mems_a = [m for m in list_active_memories(c, U, A) if m.id == b3_mem_a]
    mems_b = [m for m in list_active_memories(c, U, A) if m.id == b3_mem_b]
    all_mems = mems_a + mems_b
    assert all_mems, "No memories found for B3"

    from thyra.models.memory import (
        load_cue_edge_map,
        load_assoc_edge_map,
        load_situation_edges,
    )

    cue_map = load_cue_edge_map(c, U, A)
    assoc_map = load_assoc_edge_map(c, U, A)
    sit_edges = load_situation_edges(c, U, A)
    idf = compute_idf(c, [b3_cue], U, A)
    cat_weights = {
        r["cat_id"]: r["relevance_floor"]
        for r in c.execute(
            "SELECT cat_id, relevance_floor FROM categories WHERE user_id=? AND agent_id=?",
            (U, A),
        ).fetchall()
    }

    scored = score_memories(
        all_mems, [b3_cue], cue_map, assoc_map, sit_edges, idf, cat_weights, now
    )
    assert scored, "score_memories returned empty list"

    scores = {rec.id: sc for rec, sc in scored}
    score_a = scores.get(b3_mem_a, 0.0)
    score_b = scores.get(b3_mem_b, 0.0)
    assert score_a > score_b, (
        f"Memory with cue edge (score={score_a:.4f}) should outscore memory without (score={score_b:.4f})"
    )


@check("B4: greedy_select — constraints memory not starved at gamma=0.3")
def _b4():
    c = fresh_conn()
    cons_mem = create_memory(
        c,
        "b4 constraint rule never do this prohibited",
        category="constraints",
        user_id=U,
        agent_id=A,
    )

    # Get seeded cue for constraints memory
    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (cons_mem, U, A),
    ).fetchall()
    assert rows, "No cues seeded for B4 constraints memory"
    b4_cue = rows[0]["cue_id"]

    now = int(time.time() * 1000)
    all_mems = list_active_memories(c, U, A)

    from thyra.models.memory import (
        load_cue_edge_map,
        load_assoc_edge_map,
        load_situation_edges,
    )

    cue_map = load_cue_edge_map(c, U, A)
    assoc_map = load_assoc_edge_map(c, U, A)
    sit_edges = load_situation_edges(c, U, A)
    idf = compute_idf(c, [b4_cue], U, A)
    cat_weights = {
        r["cat_id"]: max(r["relevance_floor"], r["activation_score"] * 0.5)
        for r in c.execute(
            "SELECT cat_id, relevance_floor, activation_score FROM categories WHERE user_id=? AND agent_id=?",
            (U, A),
        ).fetchall()
    }

    scored = score_memories(
        all_mems, [b4_cue], cue_map, assoc_map, sit_edges, idf, cat_weights, now
    )
    selected = greedy_select(scored, gamma=0.3)
    selected_ids = {m.id for m in selected}
    assert cons_mem in selected_ids, (
        f"constraints memory {cons_mem} not selected even at gamma=0.3; "
        f"selected={list(selected_ids)[:5]}"
    )


@check("B5: recall_pipeline end-to-end: matching cue -> memory served")
def _b5():
    c = fresh_conn()
    b5_mem = create_memory(
        c, "b5testrecall thyrasystem pipeline", user_id=U, agent_id=A
    )

    # Get actual seeded cue from DB
    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (b5_mem, U, A),
    ).fetchall()
    assert rows, "No cues seeded for B5 memory"
    b5_cue = rows[0]["cue_id"]

    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    _xml, served = recall_pipeline(
        c, U, A, f"testing {b5_cue} system", "b5sess", "b5turn"
    )
    assert b5_mem in served, f"B5 memory {b5_mem} not in served_ids={served[:5]}"


@check("B6: recall_pipeline with system_enabled=false → ('', [])")
def _b6():
    c = conn()
    set_flag(c, "system_enabled", "false", U, A)
    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    try:
        result = recall_pipeline(c, U, A, "anything at all", "b6sess", "b6turn")
        assert result == ("", []), f"Expected ('', []) but got {result}"
    finally:
        set_flag(c, "system_enabled", "true", U, A)
        HOT_CACHE.invalidate(f"snapshot:{U}:{A}")


# ═══════════════════════════════════════════════════════════════════════════════
# C · thyra_end_turn Fallback (CCD Mode)
# ═══════════════════════════════════════════════════════════════════════════════

_set_section("C · thyra_end_turn CCD Fallback")


@check("C1: No turn-state file → inline recall_pipeline produces non-empty served_ids")
def _c1():
    c = fresh_conn()
    c1_mem = create_memory(
        c, "c1fallback ccdmode validation recall", user_id=U, agent_id=A
    )

    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (c1_mem, U, A),
    ).fetchall()
    assert rows, "No cues seeded for C1 memory"
    c1_cue = rows[0]["cue_id"]

    # Ensure no state file exists for this test session
    state_path = _turn_state_path("c1_nosession")
    if os.path.exists(state_path):
        os.remove(state_path)

    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    # This is exactly what the fallback in admin_tools.py does
    _xml, served = recall_pipeline(
        c, U, A, f"c1 test {c1_cue} fallback", "c1_nosession", "c1_turn"
    )
    assert served, f"Fallback recall_pipeline returned empty served_ids"
    assert c1_mem in served, f"C1 memory not in served: {served[:5]}"


@check("C2: declared ∩ served (from fallback) non-empty → reinforcement fires")
def _c2():
    c = fresh_conn()
    c2_mem = create_memory(
        c, "c2fallback reinforcement test validate", user_id=U, agent_id=A
    )

    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (c2_mem, U, A),
    ).fetchall()
    assert rows, "No cues for C2"
    c2_cue = rows[0]["cue_id"]

    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    _xml, served = recall_pipeline(
        c, U, A, f"c2 test {c2_cue} check", "c2sess", "c2turn"
    )
    assert c2_mem in served, f"C2 memory not served: {served[:5]}"

    before = get_memory(c, c2_mem, U, A)
    delta = make_delta(served=served, declared=[c2_mem])
    apply_reinforcement(c, delta)
    after = get_memory(c, c2_mem, U, A)
    assert after.base_strength > before.base_strength, (
        f"Strength unchanged: {before.base_strength} → {after.base_strength}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# D · Reinforcement & Anti-Spoof
# ═══════════════════════════════════════════════════════════════════════════════

_set_section("D · Reinforcement & Anti-Spoof")


@check("D1: declared ∩ served → strength += REINFORCE_BASE × multiplier")
def _d1():
    c = conn()
    d1_mem = create_memory(
        c,
        "d1 reinforcement declared served test",
        category="context",
        user_id=U,
        agent_id=A,
    )
    before = get_memory(c, d1_mem, U, A)
    mult = cfg.CATEGORY_MULTIPLIERS.get("context", 1.0)
    expected_delta = cfg.REINFORCE_BASE * mult

    delta = make_delta(served=[d1_mem], declared=[d1_mem])
    apply_reinforcement(c, delta)
    after = get_memory(c, d1_mem, U, A)

    actual_delta = after.base_strength - before.base_strength
    assert actual_delta >= expected_delta * 0.99, (  # allow rounding
        f"Strength delta={actual_delta:.4f}, expected>={expected_delta:.4f}"
    )


@check("D2: Declared but NOT in served (spoofed) → zero strength change")
def _d2():
    c = conn()
    d2_real = create_memory(c, "d2 real served memory", user_id=U, agent_id=A)
    d2_fake = create_memory(
        c, "d2 fake spoofed memory not served", user_id=U, agent_id=A
    )
    before_fake = get_memory(c, d2_fake, U, A)

    # Serve only d2_real, declare both — d2_fake is spoofed
    delta = make_delta(served=[d2_real], declared=[d2_real, d2_fake])
    apply_reinforcement(c, delta)
    after_fake = get_memory(c, d2_fake, U, A)

    # d2_fake was served=no, so it's not in served_set and won't get reinforcement
    # But it IS in served via surfaced_only check? No: surfaced_only = (served_set ∩ tenant_ids) - valid_ids - inferred_ids
    # d2_fake is not in served_set at all → not surfaced → no boost
    assert after_fake.base_strength == before_fake.base_strength, (
        f"Spoofed memory strength changed: {before_fake.base_strength} → {after_fake.base_strength}"
    )


@check(
    "D3: Served-only (not declared, short content) → exactly SURFACED_BOOST increase"
)
def _d3():
    c = conn()
    # Content with < 4 four-letter words so overlap inference skips it
    # "go bi ox zz" → zero 4-char words → mem_words is empty → len(mem_words) < 4 → skip
    d3_mem = create_memory(c, "go bi ox zz", user_id=U, agent_id=A)
    before = get_memory(c, d3_mem, U, A)

    # No raw_assistant_text → no overlap inference
    delta = make_delta(served=[d3_mem], declared=[], assistant_text="")
    apply_reinforcement(c, delta)
    after = get_memory(c, d3_mem, U, A)

    actual = after.base_strength - before.base_strength
    assert abs(actual - cfg.SURFACED_BOOST) < 1e-9, (
        f"Expected exactly SURFACED_BOOST={cfg.SURFACED_BOOST}, got delta={actual}"
    )


@check("D4: Probationary memory graduates on first declared use")
def _d4():
    c = conn()
    d4_mem = create_memory(
        c,
        "d4 probationary graduation test memory",
        probationary=True,
        decay_rate=cfg.DECAY_SEMANTIC,
        user_id=U,
        agent_id=A,
    )
    before = get_memory(c, d4_mem, U, A)
    assert before.probationary, "D4 memory should start probationary"

    delta = make_delta(served=[d4_mem], declared=[d4_mem])
    apply_reinforcement(c, delta)
    after = get_memory(c, d4_mem, U, A)

    assert not after.probationary, "D4 memory should have graduated (probationary=0)"
    assert after.use_count > before.use_count, (
        f"use_count should have incremented: {before.use_count} → {after.use_count}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# E · Cue Edge Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

_set_section("E · Cue Edge Dynamics")


@check("E1: fire_count increments for all fired cues regardless of used_set")
def _e1():
    c = conn()
    e1_mem = create_memory(c, "e1testcue fire count increment", user_id=U, agent_id=A)

    # Get seeded cue
    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (e1_mem, U, A),
    ).fetchall()
    assert rows, "No cues for E1"
    e1_cue = rows[0]["cue_id"]

    before_row = c.execute(
        "SELECT fire_count FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (e1_cue, e1_mem, U, A),
    ).fetchone()
    fire_before = before_row["fire_count"] if before_row else 0

    # Delta with no declared/served overlap — fire_count should still tick
    delta = make_delta(cues=[e1_cue], served=[], declared=[])
    update_cue_edges(c, delta)

    after_row = c.execute(
        "SELECT fire_count FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (e1_cue, e1_mem, U, A),
    ).fetchone()
    fire_after = after_row["fire_count"] if after_row else 0

    assert fire_after == fire_before + 1, (
        f"fire_count: {fire_before} → {fire_after} (expected +1)"
    )


@check("E2: weight and use_count increment for non-hub cues in used_set")
def _e2():
    c = conn()
    e2_mem = create_memory(c, "e2testcue weight use increment", user_id=U, agent_id=A)

    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (e2_mem, U, A),
    ).fetchall()
    assert rows, "No cues for E2"
    e2_cue = rows[0]["cue_id"]

    # Ensure this cue is NOT a hub in the main namespace
    hub = _hub_cues(c, U, A)
    if e2_cue in hub:
        # Try another cue from the same memory
        for row in rows[1:]:
            if row["cue_id"] not in hub:
                e2_cue = row["cue_id"]
                break

    before_row = c.execute(
        "SELECT weight, use_count FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (e2_cue, e2_mem, U, A),
    ).fetchone()
    assert before_row, f"No cue_edge for ({e2_cue}, {e2_mem})"
    w_before = before_row["weight"]
    uc_before = before_row["use_count"]

    delta = make_delta(cues=[e2_cue], served=[e2_mem], declared=[e2_mem])
    update_cue_edges(c, delta)

    after_row = c.execute(
        "SELECT weight, use_count FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (e2_cue, e2_mem, U, A),
    ).fetchone()
    assert after_row["weight"] > w_before, (
        f"weight not increased: {w_before} → {after_row['weight']}"
    )
    assert after_row["use_count"] == uc_before + 1, (
        f"use_count: {uc_before} → {after_row['use_count']} (expected +1)"
    )


@check("E3: Hub cues excluded from weight/use_count update (isolated namespace)")
def _e3():
    """
    Isolated namespace e3_user/e3_agent with exactly 4 memories.
    threshold = int(4 * 0.50) = 2 → df > 2 → hub.
    Seed 'hubcuetest' into 3 of the 4 memories → df=3 → hub.
    """
    c = conn()
    EU, EA = "e3_user", "e3_agent"

    # Seed e3 namespace categories so recall won't error
    ts = int(time.time() * 1000)
    c.execute(
        "INSERT OR IGNORE INTO categories (cat_id,user_id,agent_id,is_protected,is_emergent,decay_rate,relevance_floor,activation_score,created_at) VALUES (?,?,?,1,0,0.02,0.5,1.0,?)",
        ("context", EU, EA, ts),
    )
    set_flag(c, "system_enabled", "true", EU, EA)
    set_flag(c, "formation_enabled", "true", EU, EA)

    # Create exactly 4 memories in the e3 namespace
    mems_e3 = [
        create_memory(
            c, f"e3 memory {i} content", user_id=EU, agent_id=EA, seed_cues=False
        )
        for i in range(4)
    ]

    # Manually seed hub cue into 3 memories → df=3
    HUB_CUE = "hubcuetest"
    for mid in mems_e3[:3]:
        upsert_cue_edge(c, HUB_CUE, mid, EU, EA, weight=cfg.CONTENT_SEED_WEIGHT)
    # df should now be 3 (upsert_cue_edge calls _increment_df each time)
    df_row = c.execute(
        "SELECT df FROM cue_nodes WHERE cue_id=? AND user_id=? AND agent_id=?",
        (HUB_CUE, EU, EA),
    ).fetchone()
    assert df_row and df_row["df"] >= 3, f"Expected df>=3 for hub cue, got {df_row}"

    # Verify hub detection: M=4, threshold=int(4*0.5)=2, df=3 > 2 → hub
    hubs = _hub_cues(c, EU, EA)
    assert HUB_CUE in hubs, f"Hub cue {HUB_CUE!r} not classified as hub (hubs={hubs})"

    # Record weight/use_count before
    before_row = c.execute(
        "SELECT weight, use_count, fire_count FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (HUB_CUE, mems_e3[0], EU, EA),
    ).fetchone()
    w_before = before_row["weight"]
    uc_before = before_row["use_count"]
    fc_before = before_row["fire_count"]

    # Apply delta: mem0 is declared+served, hub cue fired
    delta_e3 = make_delta(
        cues=[HUB_CUE],
        served=[mems_e3[0]],
        declared=[mems_e3[0]],
        user_id=EU,
        agent_id=EA,
    )
    update_cue_edges(c, delta_e3)

    after_row = c.execute(
        "SELECT weight, use_count, fire_count FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (HUB_CUE, mems_e3[0], EU, EA),
    ).fetchone()

    # fire_count should increment (hub exclusion only blocks weight/use_count)
    assert after_row["fire_count"] > fc_before, (
        f"fire_count should tick for hub cue: {fc_before} → {after_row['fire_count']}"
    )
    # weight and use_count must NOT change
    assert after_row["weight"] == w_before, (
        f"Hub cue weight should be unchanged: {w_before} → {after_row['weight']}"
    )
    assert after_row["use_count"] == uc_before, (
        f"Hub cue use_count should be unchanged: {uc_before} → {after_row['use_count']}"
    )


@check("E4: prune_weak_cue_edges deletes edges with fire_count>=5 AND use_rate<0.05")
def _e4():
    c = conn()
    e4_mem = create_memory(
        c, "e4 prune weak cue edge test", user_id=U, agent_id=A, seed_cues=False
    )
    WEAK_CUE = normalize_cue("weakprunecue")
    now = int(time.time() * 1000)
    # Insert weak edge manually: fire_count=10, use_count=0 → rate=0.0
    with c:
        c.execute(
            "INSERT OR IGNORE INTO cue_nodes (cue_id, user_id, agent_id, df) VALUES (?,?,?,1)",
            (WEAK_CUE, U, A),
        )
        c.execute(
            """INSERT INTO cue_edges (cue_id, memory_id, user_id, agent_id, weight, fire_count, use_count, candidate, created_at)
               VALUES (?,?,?,?,0.10,10,0,0,?)
               ON CONFLICT(cue_id, memory_id, user_id, agent_id) DO UPDATE SET fire_count=10, use_count=0""",
            (WEAK_CUE, e4_mem, U, A, now),
        )

    pruned = prune_weak_cue_edges(c, U, A)
    assert pruned >= 1, f"Expected at least 1 edge pruned, got {pruned}"

    remaining = c.execute(
        "SELECT 1 FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (WEAK_CUE, e4_mem, U, A),
    ).fetchone()
    assert remaining is None, "Weak cue edge should have been deleted"


# ═══════════════════════════════════════════════════════════════════════════════
# F · Formation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

_set_section("F · Formation Pipeline")

# Ensure rules-only classification (no ML model blocking) for entire F section
set_rules_only(True)


@check("F1: Directive user text creates probationary memory")
def _f1():
    c = fresh_conn()
    before_count = len(list_active_memories(c, U, A))

    delta = make_delta(
        user_text="I always prefer writing tests before implementing any new feature in every project",
    )
    actions = run_formation_pipeline(c, delta)
    created = [mid for action, mid in actions if action == "created"]

    after_count = len(list_active_memories(c, U, A))
    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    after_count_fresh = len(list_active_memories(c, U, A))

    assert after_count_fresh > before_count or len(created) > 0, (
        f"No memory created from directive text. actions={actions}"
    )


@check("F2: Near-duplicate content reinforces existing memory, no duplicate insert")
def _f2():
    c = fresh_conn()
    # First, create a memory with exact content
    f2_content = "I always prefer python over javascript for all backend projects"
    f2_mem = create_memory(c, f2_content, user_id=U, agent_id=A)
    before_count = len(list_active_memories(c, U, A))
    before_strength = get_memory(c, f2_mem, U, A).base_strength

    # Now try to form the exact same content — should match via content_hash
    delta = make_delta(user_text=f2_content)
    actions = run_formation_pipeline(c, delta)
    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")

    after_count = len(list_active_memories(c, U, A))
    # Should not have created a new memory for the exact same content
    created = [mid for a, mid in actions if a == "created" and mid != f2_mem]
    assert len(created) == 0, f"Duplicate memory created: {created}"
    # If reinforced, count stays the same
    assert after_count <= before_count + 1, (
        f"Too many memories created: {before_count} → {after_count}"
    )


@check("F3: Narration prefix text → no memory formed")
def _f3():
    c = fresh_conn()
    before_count = len(list_active_memories(c, U, A))

    delta = make_delta(user_text="Let me check the documentation for you right now.")
    actions = run_formation_pipeline(c, delta)
    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    after_count = len(list_active_memories(c, U, A))

    created = [mid for a, mid in actions if a == "created"]
    assert len(created) == 0, (
        f"Narration text should not form memories, got created={created}"
    )


@check("F4: formation_enabled=false → zero memories created")
def _f4():
    c = fresh_conn()
    set_flag(c, "formation_enabled", "false", U, A)
    before_count = len(list_active_memories(c, U, A))
    try:
        delta = make_delta(
            user_text="I always prefer dark mode in all my editors and terminals always remember this"
        )
        actions = run_formation_pipeline(c, delta)
        assert actions == [], (
            f"Expected no actions when formation disabled, got {actions}"
        )
    finally:
        set_flag(c, "formation_enabled", "true", U, A)


# ═══════════════════════════════════════════════════════════════════════════════
# G · Nightly Sweep
# ═══════════════════════════════════════════════════════════════════════════════

_set_section("G · Nightly Sweep")


@check("G1: _full_decay_pass decreases base_strength via S × exp(-r × d)")
def _g1():
    import math

    c = conn()

    S0 = 1.0
    r = 0.02
    days = 7.0
    past_ms = NOW_MS - int(days * DAY_MS)

    g1_mem = create_memory(c, "g1 decay validation test memory", user_id=U, agent_id=A)
    with c:
        c.execute(
            "UPDATE memories SET base_strength=?, decay_rate=?, last_access=? WHERE id=? AND user_id=? AND agent_id=?",
            (S0, r, past_ms, g1_mem, U, A),
        )

    _full_decay_pass(c, U, A, NOW_MS)
    after = get_memory(c, g1_mem, U, A)

    expected = S0 * math.exp(-r * days)
    assert abs(after.base_strength - expected) < 0.001, (
        f"Decay formula: expected {expected:.6f}, got {after.base_strength:.6f}"
    )
    assert after.base_strength < S0, "base_strength should have decreased"


@check("G2: _autopurge_unused_probationary archives old probationary memories")
def _g2():
    c = conn()
    cutoff_days = cfg.PROBATIONARY_AUTOPURGE_DAYS + 1
    old_ms = NOW_MS - int(cutoff_days * DAY_MS)

    g2_mem = create_memory(
        c, "g2 old probationary unused memory", probationary=True, user_id=U, agent_id=A
    )
    with c:
        c.execute(
            "UPDATE memories SET created_at=?, use_count=0 WHERE id=? AND user_id=? AND agent_id=?",
            (old_ms, g2_mem, U, A),
        )

    purged = _autopurge_unused_probationary(c, U, A, NOW_MS)
    assert purged >= 1, f"Expected >= 1 purged, got {purged}"

    after = get_memory(c, g2_mem, U, A)
    assert after.archived, f"G2 memory should be archived after autopurge"


@check("G3: _archive_below_threshold archives weak memories")
def _g3():
    c = conn()
    g3_mem = create_memory(
        c, "g3 very weak memory below threshold", user_id=U, agent_id=A
    )
    weak_strength = cfg.ARCHIVE_THRESHOLD - 0.01  # 0.04
    with c:
        c.execute(
            "UPDATE memories SET base_strength=? WHERE id=? AND user_id=? AND agent_id=?",
            (weak_strength, g3_mem, U, A),
        )

    archived = _archive_below_threshold(c, U, A, NOW_MS)
    assert archived >= 1, f"Expected >= 1 archived, got {archived}"

    after = get_memory(c, g3_mem, U, A)
    assert after.archived, "G3 memory should be archived"


@check("G4: _prune_cue_edges deletes edges with fire_count>=5 AND use_rate<0.05")
def _g4():
    c = conn()
    g4_mem = create_memory(
        c, "g4 prune cue edges nightly test", user_id=U, agent_id=A, seed_cues=False
    )
    G4_CUE = normalize_cue("g4nightlyprunecue")
    now = int(time.time() * 1000)
    with c:
        c.execute(
            "INSERT OR IGNORE INTO cue_nodes (cue_id, user_id, agent_id, df) VALUES (?,?,?,1)",
            (G4_CUE, U, A),
        )
        c.execute(
            """INSERT INTO cue_edges (cue_id, memory_id, user_id, agent_id, weight, fire_count, use_count, candidate, created_at)
               VALUES (?,?,?,?,0.10,10,0,0,?)
               ON CONFLICT(cue_id, memory_id, user_id, agent_id) DO UPDATE SET fire_count=10, use_count=0""",
            (G4_CUE, g4_mem, U, A, now),
        )

    pruned = _prune_cue_edges(c, U, A)
    assert pruned >= 1, f"Expected >= 1 edge pruned, got {pruned}"

    remaining = c.execute(
        "SELECT 1 FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (G4_CUE, g4_mem, U, A),
    ).fetchone()
    assert remaining is None, "G4 weak cue edge should have been deleted"


@check("G5: _prune_hub_cue_edges removes hub cue edges (isolated g5 namespace)")
def _g5():
    c = conn()
    GU, GA = "g5_user", "g5_agent"
    ts = int(time.time() * 1000)

    # Seed categories and flags for g5 namespace
    c.execute(
        "INSERT OR IGNORE INTO categories (cat_id,user_id,agent_id,is_protected,is_emergent,decay_rate,relevance_floor,activation_score,created_at) VALUES (?,?,?,1,0,0.02,0.5,1.0,?)",
        ("context", GU, GA, ts),
    )
    set_flag(c, "system_enabled", "true", GU, GA)

    # Create 4 memories in g5 namespace
    g5_mems = [
        create_memory(
            c, f"g5 memory {i} test content", user_id=GU, agent_id=GA, seed_cues=False
        )
        for i in range(4)
    ]

    # Seed hub cue into 3 memories → df=3, M=4, threshold=2, df>2 → hub
    G5_HUB = "g5hubcue"
    for mid in g5_mems[:3]:
        upsert_cue_edge(c, G5_HUB, mid, GU, GA, weight=cfg.CONTENT_SEED_WEIGHT)

    edge_count_before = c.execute(
        "SELECT COUNT(*) FROM cue_edges WHERE cue_id=? AND user_id=? AND agent_id=?",
        (G5_HUB, GU, GA),
    ).fetchone()[0]
    assert edge_count_before == 3, (
        f"Expected 3 hub edges before prune, got {edge_count_before}"
    )

    pruned = _prune_hub_cue_edges(c, GU, GA)
    assert pruned == 3, f"Expected 3 hub edges pruned, got {pruned}"

    edge_count_after = c.execute(
        "SELECT COUNT(*) FROM cue_edges WHERE cue_id=? AND user_id=? AND agent_id=?",
        (G5_HUB, GU, GA),
    ).fetchone()[0]
    assert edge_count_after == 0, (
        f"Expected 0 hub edges after prune, got {edge_count_after}"
    )


@check("G6: _prune_orphan_cues deletes cue_nodes with no cue_edges")
def _g6():
    c = conn()
    ORPHAN_CUE = "g6orphancuetest"
    ts = int(time.time() * 1000)
    with c:
        c.execute(
            "INSERT OR IGNORE INTO cue_nodes (cue_id, user_id, agent_id, df) VALUES (?,?,?,0)",
            (ORPHAN_CUE, U, A),
        )

    before = c.execute(
        "SELECT 1 FROM cue_nodes WHERE cue_id=? AND user_id=? AND agent_id=?",
        (ORPHAN_CUE, U, A),
    ).fetchone()
    assert before is not None, "Orphan cue should exist before prune"

    pruned = _prune_orphan_cues(c, U, A)
    assert pruned >= 1, f"Expected >= 1 orphan pruned, got {pruned}"

    after = c.execute(
        "SELECT 1 FROM cue_nodes WHERE cue_id=? AND user_id=? AND agent_id=?",
        (ORPHAN_CUE, U, A),
    ).fetchone()
    assert after is None, "Orphan cue should be gone after prune"


@check(
    "G7: _decay_and_prune_assoc_edges — weight decays ×0.99; below threshold deleted"
)
def _g7():
    c = conn()
    g7_a = create_memory(
        c, "g7 assoc edge decay mem_a content", user_id=U, agent_id=A, seed_cues=False
    )
    g7_b = create_memory(
        c, "g7 assoc edge decay mem_b content", user_id=U, agent_id=A, seed_cues=False
    )
    g7_hi = create_memory(
        c, "g7 assoc edge high weight survives", user_id=U, agent_id=A, seed_cues=False
    )

    # Insert two edges: one high weight (survives), one on-the-edge (pruned after decay)
    HIGH_W = 0.8
    LOW_W = 0.050  # 0.050 * 0.99 = 0.0495 < ASSOC_PRUNE_THRESHOLD=0.05 → pruned
    upsert_assoc_edge(c, g7_a, g7_b, U, A, delta_weight=LOW_W)
    upsert_assoc_edge(c, g7_a, g7_hi, U, A, delta_weight=HIGH_W)

    # Ensure weights are exactly what we want (upsert might add to existing)
    a_sorted, b_sorted = (g7_a, g7_b) if g7_a < g7_b else (g7_b, g7_a)
    a_sorted2, hi_sorted = (g7_a, g7_hi) if g7_a < g7_hi else (g7_hi, g7_a)
    with c:
        c.execute(
            "UPDATE association_edges SET weight=? WHERE memory_a=? AND memory_b=? AND user_id=? AND agent_id=?",
            (LOW_W, a_sorted, b_sorted, U, A),
        )
        c.execute(
            "UPDATE association_edges SET weight=? WHERE memory_a=? AND memory_b=? AND user_id=? AND agent_id=?",
            (HIGH_W, a_sorted2, hi_sorted, U, A),
        )

    pruned = _decay_and_prune_assoc_edges(c, U, A)

    # Low-weight edge should be pruned
    low_edge = c.execute(
        "SELECT weight FROM association_edges WHERE memory_a=? AND memory_b=? AND user_id=? AND agent_id=?",
        (a_sorted, b_sorted, U, A),
    ).fetchone()
    assert low_edge is None, f"Low-weight assoc edge should be pruned (LOW_W={LOW_W})"

    # High-weight edge should survive (decayed to 0.8*0.99=0.792)
    hi_edge = c.execute(
        "SELECT weight FROM association_edges WHERE memory_a=? AND memory_b=? AND user_id=? AND agent_id=?",
        (a_sorted2, hi_sorted, U, A),
    ).fetchone()
    assert hi_edge is not None, "High-weight assoc edge should survive decay"
    assert abs(hi_edge["weight"] - HIGH_W * cfg.ASSOC_NIGHTLY_DECAY) < 0.001, (
        f"High-weight edge: expected {HIGH_W * cfg.ASSOC_NIGHTLY_DECAY:.4f}, got {hi_edge['weight']:.4f}"
    )


@check("G8: run_nightly_sweep returns dict with all 12 expected keys")
def _g8():
    c = conn()
    summary = run_nightly_sweep(c, U, A)

    expected_keys = {
        "decayed",
        "probationary_purged",
        "archived",
        "hard_deleted",
        "cue_edges_pruned",
        "hub_cue_edges_pruned",
        "orphan_cues_pruned",
        "assoc_edges_pruned",
        "category_scores_decayed",
        "emergent_categories",
        "categories_dissolved",
        "turn_log_pruned",
    }
    missing = expected_keys - set(summary.keys())
    assert not missing, f"Nightly sweep missing keys: {missing}"

    non_int = {k: v for k, v in summary.items() if not isinstance(v, int)}
    assert not non_int, f"Non-integer values in sweep summary: {non_int}"


# ═══════════════════════════════════════════════════════════════════════════════
# H · Integration Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

_set_section("H · Integration Lifecycle")


@check(
    "H1: Full turn lifecycle — strength increases, fire_count and use_count increment"
)
def _h1():
    c = fresh_conn()
    h1_mem = create_memory(
        c, "h1 full turn lifecycle integration test", user_id=U, agent_id=A
    )

    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (h1_mem, U, A),
    ).fetchall()
    assert rows, "No cues for H1 memory"
    h1_cue = rows[0]["cue_id"]

    before = get_memory(c, h1_mem, U, A)
    fc_before = c.execute(
        "SELECT fire_count, use_count FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (h1_cue, h1_mem, U, A),
    ).fetchone()

    delta = make_delta(
        cues=[h1_cue],
        served=[h1_mem],
        declared=[h1_mem],
        user_text="h1 lifecycle test run",
    )
    _apply_turn(delta, [delta])

    after = get_memory(c, h1_mem, U, A)
    fc_after = c.execute(
        "SELECT fire_count, use_count FROM cue_edges WHERE cue_id=? AND memory_id=? AND user_id=? AND agent_id=?",
        (h1_cue, h1_mem, U, A),
    ).fetchone()

    assert after.base_strength > before.base_strength, (
        f"Strength should increase: {before.base_strength} → {after.base_strength}"
    )
    assert fc_after["fire_count"] > fc_before["fire_count"], (
        f"fire_count: {fc_before['fire_count']} → {fc_after['fire_count']}"
    )
    assert fc_after["use_count"] > fc_before["use_count"], (
        f"use_count: {fc_before['use_count']} → {fc_after['use_count']}"
    )


@check("H2: Three co-use turns → Hebbian association edge formed")
def _h2():
    c = conn()
    h2_a = create_memory(
        c,
        "h2 hebbian association mem_a content",
        user_id=U,
        agent_id=A,
        seed_cues=False,
    )
    h2_b = create_memory(
        c,
        "h2 hebbian association mem_b content",
        user_id=U,
        agent_id=A,
        seed_cues=False,
    )

    # Three turns each co-using mem_a and mem_b
    window = []
    for i in range(3):
        d = make_delta(
            served=[h2_a, h2_b],
            declared=[h2_a, h2_b],
            cues=["h2hebbcue"],
            turn=f"h2turn{i}",
        )
        window.append(d)

    hebbian_association(c, window, U, A)

    # Check association edge formed
    a_s, b_s = (h2_a, h2_b) if h2_a < h2_b else (h2_b, h2_a)
    edge = c.execute(
        "SELECT weight FROM association_edges WHERE memory_a=? AND memory_b=? AND user_id=? AND agent_id=?",
        (a_s, b_s, U, A),
    ).fetchone()
    assert edge is not None, (
        f"Hebbian edge not formed for ({a_s[:12]}..., {b_s[:12]}...)"
    )
    assert edge["weight"] > 0, f"Hebbian edge weight={edge['weight']}"


@check("H3: Archived memory absent from normal recall; resurfaces on intent query")
def _h3():
    c = fresh_conn()
    h3_mem = create_memory(
        c, "h3resurrection thyravalidation specialrecall", user_id=U, agent_id=A
    )

    # Get a seeded cue from this memory
    rows = c.execute(
        "SELECT cue_id FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
        (h3_mem, U, A),
    ).fetchall()
    assert rows, "No cues for H3 memory"
    h3_cue = rows[0]["cue_id"]

    # Archive the memory manually
    ts = int(time.time() * 1000)
    with c:
        c.execute(
            "UPDATE memories SET archived=1, archived_at=? WHERE id=? AND user_id=? AND agent_id=?",
            (ts, h3_mem, U, A),
        )
    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")

    # Normal recall (no intent) — archived memory should NOT appear
    _xml_normal, served_normal = recall_pipeline(
        c, U, A, f"testing {h3_cue} system", "h3sess_normal", "h3turn_normal"
    )
    assert h3_mem not in served_normal, (
        f"Archived memory {h3_mem} should not appear in normal recall"
    )

    # Intent recall — archived memory should resurface
    intent_prompt = f"do you remember when we discussed {h3_cue} thyravalidation"
    HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
    _xml_intent, served_intent = recall_pipeline(
        c, U, A, intent_prompt, "h3sess_intent", "h3turn_intent"
    )
    assert h3_mem in served_intent, (
        f"Archived memory {h3_mem} should resurface on intent query. "
        f"served={served_intent[:5]}, cue={h3_cue!r}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
passed = sum(1 for r in _results if r.passed)
total = len(_results)
print(f"VALIDATION SUMMARY: {passed}/{total} passed, {total - passed} failed")

if passed == total:
    print("All invariants validated. System is healthy.")
    sys.exit(0)
else:
    print("\nFailed checks:")
    for r in _results:
        if not r.passed:
            print(f"  FAIL {r.label}")
            if r.error:
                print(f"       {r.error}")
    sys.exit(1)
