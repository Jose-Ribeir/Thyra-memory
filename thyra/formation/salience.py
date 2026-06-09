"""Deterministic salience + novelty gate for automatic memory formation."""

from __future__ import annotations

import re
import sqlite3
from typing import Sequence

from thyra.config import (
    NOVELTY_THRESHOLD,
    SALIENCE_THRESHOLD,
    AGENT_SALIENCE_THRESHOLD,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)

# ── Salience signals ───────────────────────────────────────────────────────────

_DIRECTIVE_RE = re.compile(
    r"\b(?:remember|always|never|from now on|make sure|please note|important|note that|"
    r"keep in mind|don't forget|be aware|going forward)\b",
    re.IGNORECASE,
)
_SELF_DISCLOSURE_RE = re.compile(
    r"\bI(?:'m| am| prefer| like| hate| love| need| want| use| work| have|"
    r" always| never| tend to| usually| often| sometimes)\b",
    re.IGNORECASE,
)
_NAMED_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_CORRECTION_RE = re.compile(
    r"\b(?:actually|that'?s wrong|no[,.]?\s+(?:I|it|that)|I meant|correction|"
    r"not quite|to be precise|let me correct)\b",
    re.IGNORECASE,
)
_ANTI_SIGNAL_RE = re.compile(
    r"^(?:ok|okay|sure|thanks|thank you|got it|sounds good|great|perfect|"
    r"alright|cool|understood|exactly|right|yes|no|yep|nope|hmm|mmh)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"[?]|^\s*(?:what|who|where|when|why|how|is|are|do|does|can|could|would|will)\b",
    re.IGNORECASE,
)
# Specificity markers: proper nouns, file paths, first/second-person referents, project nouns.
# A question containing these is user-specific, not a generic factual lookup.
_SPECIFIC_TOKEN_RE = re.compile(
    r"\b(?:our|my|your|this|that)\s+\w+"  # possessive/demonstrative + noun ("the" removed — too broad)
    r"|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"  # multi-word proper noun
    r"|[A-Za-z]:\\[^\s]{3,}"  # Windows path
    r"|/[a-z][a-zA-Z0-9_/.-]{3,}"  # Unix path
    r"|\b\w+\.(?:py|ts|js|json|yaml|toml|md)\b",  # filename
)

# ── Agent self-state narration veto (source="agent" only) ────────────────────
# Matches Claude transitional sentences that describe its own cognitive state
# rather than stating a durable fact: "I have the full picture now", "I'm ready",
# "I understand the problem". These are session-local narration, not memories.
# Applied only when source="agent" — user self-disclosures ("I have 10 years of
# experience") are genuine and must not be filtered.
_AGENT_SELF_STATE_RE = re.compile(
    r"^I(?:'ve| have) (?:the |a |now\b|everything\b|enough\b|all )"
    r"|^I(?:'m| am) (?:now\b|ready\b|able )"
    r"|^I (?:understand|see|realize|notice) (?:the |this |now\b|that (?:I|the|this))",
    re.IGNORECASE,
)

# ── Research / agent-work signals ─────────────────────────────────────────────
# Captures what the agent FOUND, FIXED, CONFIRMED, or DECIDED — the kind of
# cross-session knowledge that disappears between context compressions.
_AGENT_FINDING_RE = re.compile(
    r"\b(?:"
    r"root cause|the cause|caused by|the reason|"
    r"turns out|it turns out|found that|found out|we found|we discovered|"
    r"discovered that|confirmed that|we confirmed|we verified|verified that|"
    r"the issue is|the issue was|the bug is|the bug was|the problem is|the problem was|"
    r"the fix is|the fix was|fixed by|fixed by|resolved by|the solution is|"
    r"the key insight|key finding|we learned|we established|"
    r"stored at|located at|lives in|resides at|defined in|configured in|written to|"
    r"the path is|the file is|the mapping is|maps to|"
    r"we use|we decided|the decision|the approach|the architecture"
    r")\b",
    re.IGNORECASE,
)

# Matches file system paths, technical suffixes, and env-var-style constants
# that signal a concrete reference worth remembering.
# NOTE: do NOT add re.IGNORECASE here — the ALL_CAPS constant pattern must be
# truly case-sensitive so it doesn't match every ordinary English word.
_TECHNICAL_REF_RE = re.compile(
    r"[A-Za-z]:\\[^\s,)>\"']{4,}"  # Windows paths  C:\foo\bar
    r"|/[a-z][a-zA-Z0-9_/.-]{4,}"  # Unix paths     /home/user/file
    r"|\b\w+\.(?:py|ts|js|json|yaml|toml|md|sh|ps1|sql|env)\b"  # file extensions (.py etc.)
    r"|\b(?:localhost|127\.0\.0\.1):\d{2,5}\b"  # local ports    localhost:8080
    r"|\b[a-z][a-z0-9_]+\.[a-z][a-z0-9_.]+\(\)"  # method calls   foo.bar()
    r"|\b[A-Z][A-Z0-9_]{3,}\b",  # ALL_CAPS consts THYRA_DB_PATH
)

# ── L1 deterministic vetoes (transience + self-containedness) ────────────────────
# These make salience *subtractive*: a veto hard-rejects a clause (returns 0.0)
# regardless of any positive signal.  They catch the obvious non-facts cheaply —
# single-turn requests and dangling referents — that surface-form scoring let through.
# Master schematic §9.2.

# Transient single-turn request/action about the live session (NOT a standing fact).
_TRANSIENT_REQUEST_RE = re.compile(
    r"\b(?:remove|delete|undo|revert|get rid of|change (?:that|this|it|those)|"
    r"fix (?:this|that|it)|can you|could you|please (?:remove|delete|change|fix)|"
    r"i want (?:you )?to (?:remove|delete|change|undo|stop|get rid))\b",
    re.IGNORECASE,
)
# "Durable framing" that rescues an otherwise-transient clause (standing preference
# or policy rather than a one-off ask).
_DURABLE_FRAMING_RE = re.compile(
    r"\b(?:from now on|always|never|i prefer|i (?:usually|tend to)|going forward|by default)\b",
    re.IGNORECASE,
)
# Unresolved deixis: a pronoun/demonstrative referent with nothing to bind it to.
_DEIXIS_RE = re.compile(
    r"\b(?:those|that|this|these|it|them|here|there|this one)\b", re.IGNORECASE
)
# ...with NO concrete anchor present (proper noun / path / filename / quoted / number /
# ALL_CAPS const) that would let the referent resolve.
_CONCRETE_ANCHOR_RE = re.compile(
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"  # multi-word proper noun
    r"|[A-Za-z]:\\[^\s]{3,}|/[a-z][\w/.-]{3,}"  # path
    r"|\b\w+\.(?:py|ts|js|json|yaml|toml|md)\b"  # filename
    r"|\"[^\"]{3,}\"|'[^']{3,}'"  # quoted term
    r"|\b\d+\b|\b[A-Z][A-Z0-9_]{3,}\b",  # number / ALL_CAPS const
)


def _is_vetoed(clause: str) -> bool:
    """True when a deterministic L1 veto hard-rejects the clause as a non-fact.

    Two vetoes, both conservative (precision over recall on the veto itself — a
    missed transient is cleaned later by L2/L3; a wrongly-vetoed genuine fact is
    lost). When unsure, do NOT veto.

    1. Transience: a single-turn request/action ("remove those", "fix this") with
       no durable framing ("always", "from now on", "I prefer").
    2. Self-containedness: a dangling deictic referent ("those", "that") with no
       concrete anchor to bind it AND no durable framing.  Directive ("remember",
       "never") and agent-finding ("root cause", "the fix is") signals also rescue
       the clause — those carry their own durability, and "that" is a complementizer
       there ("the root cause is that …"), not a dangling demonstrative.
    """
    # Transience veto — single-turn request with no durable framing → not a memory.
    if _TRANSIENT_REQUEST_RE.search(clause) and not _DURABLE_FRAMING_RE.search(clause):
        return True
    # Self-containedness veto — dangling referent with no concrete anchor.
    if (
        _DEIXIS_RE.search(clause)
        and not _CONCRETE_ANCHOR_RE.search(clause)
        and not _DURABLE_FRAMING_RE.search(clause)
        and not _DIRECTIVE_RE.search(clause)
        and not _AGENT_FINDING_RE.search(clause)
    ):
        return True
    return False


# Matches clauses whose only directive signal is "always"/"never" with no other
# unambiguous directive word. Used to gate the directive score by source: agent
# observations ("The loop never ran") must not score as directives.
_ALWAYS_NEVER_ONLY_RE = re.compile(
    r"\b(?:always|never)\b",
    re.IGNORECASE,
)
_STRONG_DIRECTIVE_RE = re.compile(
    r"\b(?:remember|from now on|make sure|please note|important|note that|"
    r"keep in mind|don't forget|be aware|going forward)\b",
    re.IGNORECASE,
)

# Weights
# Note: _SELF_DISCLOSURE_W must be >= SALIENCE_THRESHOLD so that plain
# preference statements ("I prefer X", "I use Y") can form memories on their own.
_DIRECTIVE_W = 0.40
_SELF_DISCLOSURE_W = 0.42
_NAMED_ENTITY_W = 0.20
_CORRECTION_W = 0.30
_AGENT_FINDING_W = 0.40  # same weight as a directive — these are durable facts
_TECHNICAL_REF_W = 0.25  # concrete reference boosts confidence
# Reward for a specific-context question (proper noun / path / possessive present).
# Without this, "where is the master plan" has no signal and scores 0 even after
# the penalty escape. The weight sits just above SALIENCE_THRESHOLD so a bare
# specific question barely passes; any additional signal compounds it.
_SPECIFIC_QUESTION_W = 0.35


def compute_salience(clause: str, source: str = "user") -> float:
    """Score a clause 0–1 on how likely it contains a durable persistent fact.

    ``source`` is "user" (default) for user utterances or "agent" for sentences
    extracted from assistant responses. Agent-sourced text uses a lower base
    threshold (research findings don't need "I always…" framing to be valuable).
    """
    is_question = bool(_QUESTION_RE.search(clause))
    has_specific = is_question and bool(_SPECIFIC_TOKEN_RE.search(clause))
    if is_question and not has_specific:
        # Generic factual lookups ("what is a hashmap") have no memory value.
        return 0.0
    if _ANTI_SIGNAL_RE.match(clause.strip()):
        return 0.0
    # L1 deterministic vetoes: transient single-turn requests and dangling deictic
    # referents are not durable facts — hard-reject before any additive scoring.
    # Runs for both source="user" and source="agent".
    if _is_vetoed(clause):
        return 0.0
    # Agent self-state narration: Claude describing its own cognitive state
    # ("I have the full picture now", "I'm ready", "I understand the problem")
    # is session-local narration, not a durable cross-session fact.
    if source == "agent" and _AGENT_SELF_STATE_RE.match(clause.strip()):
        return 0.0

    score = 0.0
    if has_specific:
        # Context-bearing question: reward the specificity. No penalty applied —
        # the question form is irrelevant when the clause references something real.
        score += _SPECIFIC_QUESTION_W
    if _DIRECTIVE_RE.search(clause):
        # "always"/"never" in agent text are almost always observational
        # ("The hook never fires"), not directives. Only apply the directive
        # score weight when the clause has an unambiguous directive keyword
        # (remember, make sure, …) OR when the source is the user.
        if (
            _STRONG_DIRECTIVE_RE.search(clause)
            or source == "user"
            or not _ALWAYS_NEVER_ONLY_RE.search(clause)
        ):
            score += _DIRECTIVE_W
    if _SELF_DISCLOSURE_RE.search(clause):
        score += _SELF_DISCLOSURE_W
    if _NAMED_ENTITY_RE.search(clause):
        score += _NAMED_ENTITY_W
    if _CORRECTION_RE.search(clause):
        score += _CORRECTION_W
    m_finding = _AGENT_FINDING_RE.search(clause)
    if m_finding:
        # Require ≥ 3 words after the trigger phrase so short status updates
        # ("The problem is clear", "The fix is done") don't score as findings.
        if len(clause[m_finding.end() :].strip().split()) >= 3:
            score += _AGENT_FINDING_W
    if _TECHNICAL_REF_RE.search(clause):
        score += _TECHNICAL_REF_W
    # Agent-sourced text gets a small base bump: a sentence the assistant
    # chose to state as a conclusion carries implicit salience.
    if source == "agent" and score == 0.0 and len(clause) > 30:
        score = 0.10  # minimal floor — still filtered by threshold
    return min(1.0, score)


def compute_novelty(
    conn: sqlite3.Connection,
    clause: str,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> float:
    """Score novelty: how much of this clause's vocabulary is new to the agent."""
    from thyra.recall.cue_extractor import extract_raw_cues
    from thyra.config import DISCRIMINABILITY_FLOOR

    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchone()
    M = row[0] if row else 0
    if M == 0:
        return 1.0  # any clause is novel when there are no memories

    cues = extract_raw_cues(clause, max_cues=8)
    if not cues:
        return 0.5

    # Novelty = average rarity of the cues
    total = 0.0
    for cue in cues:
        df_row = conn.execute(
            "SELECT df FROM cue_nodes WHERE cue_id=? AND user_id=? AND agent_id=?",
            (cue, user_id, agent_id),
        ).fetchone()
        df = df_row["df"] if df_row else 0
        import math

        rarity = math.log(1 + M / max(1, df)) / math.log(1 + M)
        total += max(DISCRIMINABILITY_FLOOR, rarity)
    return total / len(cues)


def _split_to_clauses(text: str) -> list[str]:
    """Split text into sentence-level clauses."""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    clauses = []
    for s in sentences:
        # Further split on conjunctions for compound sentences
        parts = re.split(r"\s*[,;]\s*(?:and|but|so|yet|or)\s+", s, flags=re.IGNORECASE)
        clauses.extend(p.strip() for p in parts if p.strip() and len(p.strip()) > 10)
    return clauses


def extract_salient_clauses(
    text: str,
    conn: sqlite3.Connection,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
    source: str = "user",
) -> list[dict]:
    """Return clauses that pass the salience + novelty gate.

    Each result is a dict with keys: clause, salience, novelty, is_correction.
    ``source`` is "user" or "agent" — agent text uses the same threshold but
    benefits from the agent-finding and technical-reference signals.
    """
    threshold = AGENT_SALIENCE_THRESHOLD if source == "agent" else SALIENCE_THRESHOLD
    clauses = _split_to_clauses(text)
    results = []
    for clause in clauses:
        sal = compute_salience(clause, source=source)
        if sal < threshold:
            continue
        nov = compute_novelty(conn, clause, user_id, agent_id)
        if nov < NOVELTY_THRESHOLD:
            # Still pass corrections through (novelty can be low for rewrites)
            if not _CORRECTION_RE.search(clause):
                continue
        results.append(
            {
                "clause": clause,
                "salience": sal,
                "novelty": nov,
                "is_correction": bool(_CORRECTION_RE.search(clause)),
            }
        )
    return results
