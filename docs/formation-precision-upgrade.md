# Formation-Precision Upgrade — Implementation Schematic

> **Scope.** This is the *work plan* for raising automatic-formation **precision** so transient,
> non-self-contained fragments stop being written as memories. It is the code-specific companion to the
> code-agnostic master schematic (`J:\cloud\AI\adaptive-memory-method-schematic.md`, §9 / §9.2 / §9.3 /
> §9.5 / §12 / §14). The master describes *what the method does*; this describes *what we change in
> `thyra/` to get there*. Cite this doc in PRs; cite the master for behavioral rationale.

---

## 0. The triggering failure

Memory `m_43eee26110844cdc` was formed from the user clause:

> *"I dont see it anywhere and i want to remove all of those instructions"*

Stored as `context / semantic / probationary / base_strength 0.40`. It is not a fact — it is a
**single-turn request** with an **unresolved referent** ("those instructions").

**Exact leak path** (traced in `thyra/formation/salience.py`):
1. No comma-before-conjunction → `_split_to_clauses` keeps it as one clause.
2. `_SELF_DISCLOSURE_RE` matches `I want` → `+_SELF_DISCLOSURE_W (0.42)`.
3. `SALIENCE_THRESHOLD = 0.32` (config.py:108) → **passes**.
4. Novelty high (fresh vocabulary) → passes.
5. `refiner.py` finds no category rule → `context`; no episodic marker → `semantic`.

**Root cause:** the gate scores **surface form** (presence of `I want`), never **durability** (will this
be true/useful next week?). Regex cannot distinguish `"I want tabs not spaces"` (durable) from `"I want to
remove those"` (transient). Precision was deferred *entirely* to decay, so the fragment is still written
and sits at full provisional strength for ~6 weeks.

---

## 1. SOTA basis (why these layers)

The 2025–26 agent-memory literature converged on the same conclusion: **don't decide "what to keep" with a
scorer — distill first, keep what survives.**

- **Mem0** ([arxiv 2504.19413](https://arxiv.org/abs/2504.19413)): an LLM **extraction phase** rewrites raw
  messages into short *standalone declarative facts* ("I love pizza" → "Loves pizza") and **drops
  non-facts**. Filtering is a side effect of distillation, not a threshold.
- **LangMem / proactive extraction** ([langmem](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/),
  [arxiv 2601.04463](https://arxiv.org/pdf/2601.04463)): a reflection pass compresses a turn into **atomic,
  categorized factoids**, separating *factual* from *reflective (preference)* memory and pruning
  redundancy/noise.
- **A-Mem** ([arxiv 2502.12110](https://arxiv.org/abs/2502.12110)): every memory is a structured note;
  weak/contradicted notes are retroactively revised or pruned — a self-evolving store, not a write-once log.

Mapped to our design: regex salience stays a **recall-biased prefilter**; the **final** keep/drop decision
moves to (a) deterministic vetoes that catch the obvious non-facts cheaply and (b) an advisory model that
distills-or-drops on *meaning*. Decay becomes the *second* net, not the only one.

---

## 2. The four layers

| Layer | What it adds | Master ref | Cost | Ships |
|---|---|---|---|---|
| **L1** Deterministic transience + self-containedness vetoes | Hard reject non-facts in `compute_salience` | §9.2 | ~0 | now |
| **L2** Distill-or-drop judge | Advisory model returns clean atomic fact **or** drop | §9.3 | 1 small-model call / candidate (gated, flagged) | next |
| **L3** Tightened probationary lifecycle | Weak-signal unused admits expire in days | §9.5 | ~0 | with L1 |
| **L4** Precision eval harness | Labeled transient-vs-durable fixtures; measure precision | — | ~0 | with L1 |

Rollout order: **L1 + L3 + L4 together** (deterministic, regression-netted), then **L2** as the durable
fix. L1 plugs this exact class today; L2 generalizes it to cases regex misses.

---

## 3. Layer 1 — deterministic vetoes (load-bearing)

**File:** `thyra/formation/salience.py`. **Principle:** make salience *subtractive*, not only additive — a
veto overrides all positive signal (master §9.2).

### 3.1 New patterns
```python
# Transient single-turn request/action about the live session (NOT a standing fact).
_TRANSIENT_REQUEST_RE = re.compile(
    r"\b(?:remove|delete|undo|revert|get rid of|change (?:that|this|it|those)|"
    r"fix (?:this|that|it)|can you|could you|please (?:remove|delete|change|fix)|"
    r"i want (?:you )?to (?:remove|delete|change|undo|stop|get rid))\b",
    re.IGNORECASE,
)
# "Durable framing" that rescues an otherwise-transient clause.
_DURABLE_FRAMING_RE = re.compile(
    r"\b(?:from now on|always|never|i prefer|i (?:usually|tend to)|going forward|by default)\b",
    re.IGNORECASE,
)
# Unresolved deixis: pronoun/demonstrative referent...
_DEIXIS_RE = re.compile(r"\b(?:those|that|this|these|it|them|here|there|this one)\b", re.IGNORECASE)
# ...with NO concrete anchor to bind it (proper noun / path / filename / quoted / number / ALL_CAPS const).
_CONCRETE_ANCHOR_RE = re.compile(
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"          # multi-word proper noun
    r"|[A-Za-z]:\\[^\s]{3,}|/[a-z][\w/.-]{3,}"  # path
    r"|\b\w+\.(?:py|ts|js|json|yaml|toml|md)\b" # filename
    r"|\"[^\"]{3,}\"|'[^']{3,}'"               # quoted term
    r"|\b\d+\b|\b[A-Z][A-Z0-9_]{3,}\b",        # number / ALL_CAPS const
)
```

### 3.2 Wire into `compute_salience`
Add **before** the additive scoring (after the existing question / anti-signal short-circuits):
```python
# Transience veto — single-turn request with no durable framing → not a memory.
if _TRANSIENT_REQUEST_RE.search(clause) and not _DURABLE_FRAMING_RE.search(clause):
    return 0.0
# Self-containedness veto — dangling referent with no concrete anchor → not self-contained.
if _DEIXIS_RE.search(clause) and not _CONCRETE_ANCHOR_RE.search(clause) \
        and not _DURABLE_FRAMING_RE.search(clause):
    return 0.0
```

### 3.3 Notes / edge cases
- The triggering fragment trips **both** vetoes (`i want to remove` + `those` with no anchor). Either alone
  is sufficient; keep both — they catch different classes.
- `_DURABLE_FRAMING_RE` is the escape hatch so `"from now on, remove trailing whitespace"` and `"I prefer
  that over tabs"` survive (anchored or framed).
- Keep vetoes **conservative** (precision over recall on the veto itself): a missed transient is cleaned by
  L2/L3; a wrongly-vetoed genuine fact is lost. When unsure, do **not** veto — let L2 judge.
- These run for **both** `source="user"` and `source="agent"`.

---

## 4. Layer 2 — distill-or-drop judge (advisory)

**File:** new `thyra/formation/distiller.py`; called from `pipeline.py` between salience and `REFINER`.
**Pattern:** Mem0/LangMem extraction — input clause(s) + minimal turn context → JSON
`{keep: bool, fact: str|null, category: str, kind: "semantic"|"episodic"}`.

### 4.1 Contract
- **drop** when the clause is chatter, a restated question, or a transient instruction the L1 vetoes missed
  (meaning-level judgment regex can't do).
- **fact** = one short, atomic, **self-contained** statement with dangling referents resolved from context.
  If it can't be made self-contained → `keep=false`.
- Reuse the **live taxonomy** for `category` (never a fixed classifier — master §7).

### 4.2 Integration constraints (respect existing architecture)
- The pipeline runs in a **background worker thread** that deliberately avoids cold model loads
  (`refiner.set_rules_only(True)`; see `refiner.py:_get_model_and_embeddings`). The distiller must follow
  the **same prewarm-or-skip discipline**: if the model isn't resident, **skip distillation and fall back
  to L1-gated insert** — never block the worker.
- Gate behind a config flag (mirror the MiniLM refiner): `DISTILLER_ENABLED` (default off until validated).
- Only runs on candidates that **already passed L1** → bounded call volume.
- **Degradation (master §9.3):** model unavailable/invalid → gate-only insert, but **L1 vetoes still run**.
  Worst case becomes "store a slightly rough genuine fact", never "store a transient fragment".

### 4.3 Model choice
Local small instruct model (master §9.3 specifies ~8B, local, for data sovereignty). Log every
`(clause → verdict/fact)` pair for future distillation of the judge itself (master §9.6).

---

## 5. Layer 3 — tightened probationary lifecycle

**Files:** `thyra/formation/dedup.py` (`insert_as_probationary`), `thyra/consolidation/decay.py` /
`worker.py`.

- **Weak-admit tagging:** when an admit passed on **weak signal alone** (no `_DIRECTIVE_RE`, no confirmed
  reference, salience just over threshold), insert it on the **steepest episodic horizon**
  (`DECAY_EPISODIC = 0.15`, ~5-day half-life) regardless of category, so an unused borderline admit is gone
  in **days, not weeks**. Strong-signal admits keep their normal probationary tier.
- **A-Mem-style auto-purge:** in the nightly sweep, probationary + `use_count == 0` + age past a short
  window → archive immediately (don't wait for the slow strength curve to cross the archive threshold).
- This does **not** replace graduation (master §4) — a weak admit that *is* used still graduates normally.

---

## 6. Layer 4 — precision eval harness

**File:** extend `tests/test_noise_gate.py` (add a `TestTransienceGate` class).

- **Labeled fixtures.** Add `TRANSIENT_SAMPLES` (must NOT form) and keep `GENUINE_SAMPLES` (must form). Seed
  `TRANSIENT_SAMPLES` with the real leak + siblings:
  ```python
  TRANSIENT_SAMPLES = [
      "I dont see it anywhere and i want to remove all of those instructions",
      "can you delete that for me",
      "change this to use the other approach",
      "get rid of those",
      "undo that last change",
      "fix this please",
  ]
  ```
- **Assertions:** every `TRANSIENT_SAMPLES` entry → `compute_salience(...) == 0.0` **and**
  `run_formation_pipeline` creates 0 memories; every `GENUINE_SAMPLES` entry still forms (no regression).
- **Precision metric:** add a test that runs both sets and asserts formation **precision == 1.0** on the
  transient set (zero false admits) while **recall** on the genuine set stays at its current level.
- Guards against over-aggressive vetoes silently killing real facts.

---

## 7. Config changes (`thyra/config.py`)

| Const | Action | Notes |
|---|---|---|
| `DISTILLER_ENABLED` | **add**, default `False` | L2 master flag (mirror refiner flag) |
| `WEAK_ADMIT_DECAY` | **add** = `DECAY_EPISODIC (0.15)` | L3 steep horizon for weak-signal admits |
| `SALIENCE_THRESHOLD` | **review** (currently `0.32`) | vetoes do the heavy lifting now; consider nudging back toward master's `0.55` once L1+L4 land |

Vetoes themselves are pattern-based (no numeric knob) — they are hard rejects by design (master §12).

---

## 8. Cleanup of existing junk

- Delete `m_43eee26110844cdc` (the triggering fragment).
- Re-run `scripts/cleanup_junk.py` after L1 lands; consider extending it to flag existing stored memories
  that the **new vetoes** would now reject (one-time backfill sweep over the live DB).

---

## 9. Acceptance criteria

1. `compute_salience("…remove all of those instructions") == 0.0`.
2. `run_formation_pipeline` creates **0** memories for every `TRANSIENT_SAMPLES` entry.
3. All `GENUINE_SAMPLES` still form (zero regression) — `test_noise_gate.py` green.
4. `ruff format` clean before commit (project constraint).
5. L2 behind `DISTILLER_ENABLED=False` by default; enabling it drops additional non-facts L1 missed,
   measured against the eval set, with **no** new false negatives on genuine facts.

---

## 10. Open questions

- **Distiller model:** which local ~8B instruct model, and is it already in `H:\HuggingFace` (the refiner's
  cache)? Reuse the same prewarm path or a separate one?
- **Backfill scope:** do we retroactively veto-sweep the existing live DB, or only gate new formation?
- **Threshold reconciliation:** master says `0.55`; code runs `0.32`. Once vetoes land, do we raise the
  code threshold toward the master value, or keep it low and let vetoes carry precision?
