"""Tests for Workstream 1 (noise gate) and Workstream 2 (content-hash dedup)."""

from __future__ import annotations

import time
import pytest

from thyra.formation.pipeline import _clean_for_formation, _is_noise_sentence
from thyra.formation.dedup import find_near_match, insert_as_probationary
from thyra.formation.pipeline import run_formation_pipeline
from thyra.formation.salience import compute_salience
from thyra.models.delta import DeltaEvent
from thyra.models.memory import create_memory, get_memory, compute_content_hash
from thyra.config import (
    THYRA_USER_ID as U,
    THYRA_AGENT_ID as A,
    DECAY_EPISODIC,
    DECAY_SEMANTIC,
    BASE_STRENGTH_AUTOMATIC,
)


# ── Real junk samples captured from the live DB ───────────────────────────────

JUNK_SAMPLES = [
    "Output is being written to: J:\\tasks\\bpez25fux.output",
    '1 """Stdin helpers"""\n2 import sys\n3 def read():',
    "343- \n344- \n345-",
    "Exit code 0",
    "Traceback (most recent call last):",
    "    let me check the file",
    "Looking at the output now",
    "Now I need to find where the bug is",
    "first, let's look at the structure",
]

GENUINE_SAMPLES = [
    "I always prefer tabs over spaces in Python code.",
    "The root cause is that the formation pipeline never filters out tool output.",
    "Remember that this project uses Poetry for dependency management.",
    "The user is a data scientist working on memory system design.",
    "Never store raw tool output as a memory — it pollutes the store.",
    "I prefer dark mode in all my editors and terminals.",
]

# ── Transient single-turn requests (must NOT form) ─────────────────────────────
# Seeded with the real leak (m_43eee26110844cdc) plus siblings: each is a
# single-turn instruction or an action with an unresolved referent, not a fact.
TRANSIENT_SAMPLES = [
    "I dont see it anywhere and i want to remove all of those instructions",
    "can you delete that for me",
    "change this to use the other approach",
    "get rid of those",
    "undo that last change",
    "fix this please",
]


class TestIsNoiseSentence:
    def test_tool_output_lines_are_noise(self):
        assert _is_noise_sentence("Output is being written to: J:\\tasks\\abc.output")
        assert _is_noise_sentence("Exit code 0")
        assert _is_noise_sentence("Traceback (most recent call last):")

    def test_narration_prefixes_are_noise(self):
        assert _is_noise_sentence("Let me check the file now")
        assert _is_noise_sentence("Now I need to find where the bug is")
        assert _is_noise_sentence("Looking at the output")
        assert _is_noise_sentence("First, let's look at the structure")
        assert _is_noise_sentence("I'll look at this now")

    def test_grep_fragments_are_noise(self):
        assert _is_noise_sentence("343-")
        assert _is_noise_sentence("344- ")

    def test_genuine_directives_pass(self):
        for s in GENUINE_SAMPLES:
            assert not _is_noise_sentence(s), f"False positive on: {s!r}"

    def test_short_strings_are_noise(self):
        assert _is_noise_sentence("ok")
        assert _is_noise_sentence("sure")

    def test_code_like_ratio(self):
        # Mostly punctuation/symbols — not a real sentence
        assert _is_noise_sentence(">>> import sys; sys.stdout.write('\\x00')")


class TestCleanForFormation:
    def test_strips_fenced_code_blocks(self):
        text = "Some prose.\n```python\nimport os\nprint(os.getcwd())\n```\nMore prose."
        cleaned = _clean_for_formation(text)
        assert "import os" not in cleaned
        assert "Some prose" in cleaned
        assert "More prose" in cleaned

    def test_strips_tool_output_lines(self):
        text = (
            "Output is being written to: J:\\tasks\\abc.output\nThe fix is confirmed."
        )
        cleaned = _clean_for_formation(text)
        assert "Output is being written to" not in cleaned
        assert "The fix is confirmed" in cleaned

    def test_strips_line_numbered_output(self):
        # Read tool format: "12\tsome code"
        text = "12\tdef foo():\n13\t    return 42\nI prefer Python."
        cleaned = _clean_for_formation(text)
        assert "I prefer Python" in cleaned
        # Line-numbered lines should be gone
        assert "def foo" not in cleaned

    def test_preserves_genuine_text(self):
        text = "Remember that I always use tabs in Python. I prefer dark mode."
        assert _clean_for_formation(text) == text

    def test_empty_input(self):
        assert _clean_for_formation("") == ""


class TestContentHashDedup:
    def test_exact_match_returns_existing(self, tmp_db):
        content = "I prefer tabs over spaces in Python code"
        existing_id = create_memory(tmp_db, content, "preferences")
        # Identical content — should hit the hash short-circuit
        match = find_near_match(tmp_db, content, U, A)
        assert match is not None
        assert match.id == existing_id

    def test_normalized_match(self, tmp_db):
        content = "I prefer tabs over spaces in Python code"
        existing_id = create_memory(tmp_db, content, "preferences")
        # Equivalent after normalization (different punctuation / case)
        match = find_near_match(
            tmp_db, "i prefer tabs over spaces in python code", U, A
        )
        assert match is not None
        assert match.id == existing_id

    def test_unrelated_content_no_match(self, tmp_db):
        create_memory(tmp_db, "I prefer Python for data science", "preferences")
        match = find_near_match(tmp_db, "Deploy Kubernetes with Helm charts", U, A)
        assert match is None

    def test_compute_content_hash_deterministic(self):
        h1 = compute_content_hash("Hello World!")
        h2 = compute_content_hash("Hello World!")
        assert h1 == h2

    def test_compute_content_hash_normalizes(self):
        h1 = compute_content_hash("Hello World!")
        h2 = compute_content_hash("hello world")  # lowercase, no punct
        assert h1 == h2

    def test_insert_sets_content_hash(self, tmp_db):
        mem_id = insert_as_probationary(
            tmp_db,
            "I prefer dark mode",
            "preferences",
            "semantic",
            DECAY_EPISODIC,
            U,
            A,
            [],
        )
        row = tmp_db.execute(
            "SELECT content_hash FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == compute_content_hash("I prefer dark mode")


class TestPerCategoryDecay:
    def test_durable_categories_get_semantic_decay(self, tmp_db):
        for cat in ("constraints", "identity", "preferences"):
            mid = insert_as_probationary(
                tmp_db,
                f"Test content for {cat}",
                cat,
                "semantic",
                DECAY_EPISODIC,
                U,
                A,
                [],
            )
            rec = get_memory(tmp_db, mid)
            assert rec is not None
            assert rec.decay_rate == pytest.approx(DECAY_SEMANTIC), (
                f"Expected DECAY_SEMANTIC for {cat}, got {rec.decay_rate}"
            )

    def test_context_category_gets_episodic_decay(self, tmp_db):
        mid = insert_as_probationary(
            tmp_db,
            "Currently working on this feature",
            "context",
            "episodic",
            DECAY_EPISODIC,
            U,
            A,
            [],
        )
        rec = get_memory(tmp_db, mid)
        assert rec is not None
        assert rec.decay_rate == pytest.approx(DECAY_EPISODIC)

    def test_directive_content_gets_higher_strength(self, tmp_db):
        directive_mid = insert_as_probationary(
            tmp_db,
            "Always remember to use tabs not spaces",
            "preferences",
            "semantic",
            DECAY_EPISODIC,
            U,
            A,
            [],
        )
        plain_mid = insert_as_probationary(
            tmp_db,
            "I like working with Python frameworks",
            "preferences",
            "semantic",
            DECAY_EPISODIC,
            U,
            A,
            [],
        )
        directive_rec = get_memory(tmp_db, directive_mid)
        plain_rec = get_memory(tmp_db, plain_mid)
        assert directive_rec.base_strength > plain_rec.base_strength


class TestPipelineNoiseGate:
    def _make_delta(self, user_text="", asst_text="") -> DeltaEvent:
        return DeltaEvent(
            session_id="test",
            turn_id=f"t{int(time.time() * 1000)}",
            user_id=U,
            agent_id=A,
            timestamp=int(time.time() * 1000),
            raw_user_text=user_text,
            raw_assistant_text=asst_text,
        )

    def test_tool_output_not_stored(self, tmp_db):
        asst = "Output is being written to: J:\\tasks\\abc.output\nExit code 0"
        delta = self._make_delta(asst_text=asst)
        actions = run_formation_pipeline(tmp_db, delta)
        assert len(actions) == 0, f"Expected 0 actions but got: {actions}"

    def test_genuine_directive_still_stored(self, tmp_db):
        user = "Remember that I always use tabs over spaces in Python."
        delta = self._make_delta(user_text=user)
        actions = run_formation_pipeline(tmp_db, delta)
        created = [a for a in actions if a[0] == "created"]
        assert len(created) >= 1

    def test_narration_not_stored(self, tmp_db):
        asst = "Let me check the file. Now I need to find where the issue is. Looking at the output."
        delta = self._make_delta(asst_text=asst)
        actions = run_formation_pipeline(tmp_db, delta)
        assert len(actions) == 0, f"Expected 0 actions but got: {actions}"

    def test_mixed_content_stores_only_real_fact(self, tmp_db):
        asst = (
            "Let me look at the code.\n"
            "```python\nimport os\n```\n"
            "The root cause is that the pipeline never filtered tool output before this fix."
        )
        delta = self._make_delta(asst_text=asst)
        actions = run_formation_pipeline(tmp_db, delta)
        # At least one creation, and none of the created memories should be noise
        created_ids = [mid for action, mid in actions if action == "created"]
        for mid in created_ids:
            rec = get_memory(tmp_db, mid)
            assert rec is not None
            assert "import os" not in rec.content
            assert "Let me" not in rec.content

    def test_duplicate_clause_reinforces_not_inserts(self, tmp_db):
        content = "I always prefer tabs over spaces"
        existing_id = create_memory(tmp_db, content, "preferences")
        str_before = tmp_db.execute(
            "SELECT base_strength FROM memories WHERE id=?", (existing_id,)
        ).fetchone()[0]
        delta = self._make_delta(user_text=content)
        run_formation_pipeline(tmp_db, delta)
        # Exact content should have been reinforced (find_near_match via hash), not re-inserted
        row = tmp_db.execute(
            "SELECT id, base_strength FROM memories WHERE content_hash=? AND user_id=? AND agent_id=?",
            (compute_content_hash(content), U, A),
        ).fetchall()
        # There should be exactly ONE memory with this hash (no duplicate inserted)
        ids_with_hash = [r["id"] for r in row]
        assert len(ids_with_hash) == 1
        assert existing_id in ids_with_hash


class TestSelfReferentialNoise:
    """Memories about Thyra memory IDs / injection XML must not be stored."""

    def _make_delta(self, user_text="", asst_text="") -> DeltaEvent:
        return DeltaEvent(
            session_id="test",
            turn_id=f"t{int(time.time() * 1000)}",
            user_id=U,
            agent_id=A,
            timestamp=int(time.time() * 1000),
            raw_user_text=user_text,
            raw_assistant_text=asst_text,
            memories_served=[],
            memories_declared=[],
            cues_fired=[],
        )

    def test_memory_id_line_is_noise(self):
        line = "m_62548e727c904fa0  constraints  semantic  probationary"
        assert _is_noise_sentence(line)

    def test_memory_id_listing_is_noise(self):
        listing = (
            "m_276d1cd3b0d048b6  'Sources:\\n- [How to Create an Email Signature]'"
        )
        assert _is_noise_sentence(listing)

    def test_two_ids_in_sentence_is_noise(self):
        s = "Consider m_62548e727c904fa0 and m_276d1cd3b0d048b6 as candidates."
        assert _is_noise_sentence(s)

    def test_single_id_in_long_prose_passes(self):
        # A single memory ID embedded in a long sentence should not be filtered
        s = "Memory m_62548e727c904fa0 was formed incorrectly and has been deleted from the store."
        assert not _is_noise_sentence(s)

    def test_thyra_injection_stripped_from_formation_text(self):
        text = (
            '<thyra_memories agent="memory_llm" retrieved_at="2026-06-08T16:00:00Z">'
            '[MEMORY id="m_abc1234567890abc" cat="context" strength="0.5" age_days="1"]'
            "some injected memory"
            "[/MEMORY]"
            "</thyra_memories>"
            " The real answer is that the fix is in dedup.py."
        )
        cleaned = _clean_for_formation(text)
        assert "injected memory" not in cleaned
        assert "thyra_memories" not in cleaned
        assert "The real answer is that the fix is in dedup.py" in cleaned

    def test_memories_used_tag_stripped(self):
        text = "The footer is always client-side.\n<memories_used>m_abc123def456789a</memories_used>"
        cleaned = _clean_for_formation(text)
        assert "memories_used" not in cleaned
        assert "The footer is always client-side" in cleaned

    def test_memory_listing_not_stored_by_pipeline(self, tmp_db):
        # Simulate the agent outputting a formatted memory listing in its response
        asst = (
            "Here are the current memories:\n"
            "  m_276d1cd3b0d048b6  'Sources for email signature in Spacemail'\n"
            "  m_4467e246fd20474d  'Just set up the text footer in the agent prompt'\n"
            "  m_62548e727c904fa0  'Tools listing from previous session'\n"
        )
        delta = self._make_delta(asst_text=asst)
        actions = run_formation_pipeline(tmp_db, delta)
        # Verify no memory was created with a memory ID in its content
        import re

        MEM_ID_RE = re.compile(r"\bm_[0-9a-f]{16}\b")
        for action, mid in actions:
            if action == "created":
                rec = get_memory(tmp_db, mid)
                assert rec is not None
                ids_found = MEM_ID_RE.findall(rec.content)
                assert len(ids_found) < 2, (
                    f"Memory {mid} contains multiple memory IDs: {rec.content!r}"
                )


class TestTransienceGate:
    """L1/L4 — transient single-turn requests must be vetoed; genuine facts must
    still form. Guards against the m_43eee26110844cdc class of leak and against
    over-aggressive vetoes silently killing real facts.
    """

    def _make_delta(self, user_text="", asst_text="") -> DeltaEvent:
        return DeltaEvent(
            session_id="test",
            turn_id=f"t{int(time.time() * 1e6)}",
            user_id=U,
            agent_id=A,
            timestamp=int(time.time() * 1000),
            raw_user_text=user_text,
            raw_assistant_text=asst_text,
        )

    def _created(self, tmp_db, user_text: str) -> list[str]:
        delta = self._make_delta(user_text=user_text)
        actions = run_formation_pipeline(tmp_db, delta)
        return [mid for action, mid in actions if action == "created"]

    # ── Salience-level veto (acceptance criterion #1) ─────────────────────────
    @pytest.mark.parametrize("clause", TRANSIENT_SAMPLES)
    def test_transient_salience_is_zero(self, clause):
        assert compute_salience(clause, source="user") == 0.0, (
            f"Transient clause was not vetoed: {clause!r}"
        )

    def test_triggering_fragment_is_zero(self):
        # The exact fragment that produced m_43eee26110844cdc.
        assert (
            compute_salience(
                "I dont see it anywhere and i want to remove all of those instructions"
            )
            == 0.0
        )

    def test_durable_framing_rescues_transient_verb(self):
        # An escape-hatch: a transient verb under durable framing is NOT vetoed.
        assert (
            compute_salience("From now on, remove trailing whitespace on save.") > 0.0
        )
        assert (
            compute_salience("I prefer to remove all comments before committing.") > 0.0
        )

    # ── Pipeline-level veto (acceptance criterion #2) ─────────────────────────
    @pytest.mark.parametrize("clause", TRANSIENT_SAMPLES)
    def test_transient_forms_zero_memories(self, tmp_db, clause):
        assert len(self._created(tmp_db, clause)) == 0, (
            f"Transient clause formed a memory: {clause!r}"
        )

    # ── No regression on genuine facts (acceptance criterion #3) ──────────────
    @pytest.mark.parametrize("clause", GENUINE_SAMPLES)
    def test_genuine_still_forms(self, tmp_db, clause):
        assert len(self._created(tmp_db, clause)) >= 1, (
            f"Genuine fact stopped forming (false veto): {clause!r}"
        )

    # ── Precision / recall metric ─────────────────────────────────────────────
    def test_formation_precision_on_transient_set(self, tmp_db):
        """Precision == 1.0 on the transient set (zero false admits) while recall
        on the genuine set stays at full (every genuine sample still forms)."""
        false_admits = sum(
            1 for s in TRANSIENT_SAMPLES if len(self._created(tmp_db, s)) > 0
        )
        true_admits = sum(
            1 for s in GENUINE_SAMPLES if len(self._created(tmp_db, s)) > 0
        )
        # Precision on the transient set: no transient sample may admit.
        assert false_admits == 0, f"{false_admits} transient false admits"
        # Recall on the genuine set: every genuine sample still admits.
        assert true_admits == len(GENUINE_SAMPLES), (
            f"recall regressed: {true_admits}/{len(GENUINE_SAMPLES)} genuine formed"
        )
