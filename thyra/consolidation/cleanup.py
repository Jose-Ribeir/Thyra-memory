"""Post-usage junk cleanup pass.

Called by the background worker after each delta event (rate-limited to
CLEANUP_INTERVAL_HOURS per user/agent pair) to remove noise that slipped
past the formation gate.

Foreign-key cascades (cue_edges ON DELETE CASCADE, association_edges ON DELETE
CASCADE) remove dependent rows automatically when a memory is deleted.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("thyra.cleanup")

_MEM_ID_RE = re.compile(r"\bm_[0-9a-f]{16}\b")


def _classify_junk_reasons(content: str) -> list[str]:
    """Return non-empty list of reason strings if content is junk, else []."""
    from thyra.formation.pipeline import _clean_for_formation, _is_noise_sentence
    from thyra.formation.salience import _is_vetoed, _split_to_clauses

    reasons: list[str] = []

    # Memory-ID listing (content is mostly a list of IDs, not prose)
    ids_found = _MEM_ID_RE.findall(content)
    if ids_found:
        non_id_text = _MEM_ID_RE.sub("", content).strip()
        if len(non_id_text) < 50 or len(ids_found) >= 2:
            reasons.append("memory-id-listing")
    if reasons:
        return reasons  # no need to check further

    cleaned = _clean_for_formation(content)
    if len(content) > 20 and len(cleaned.strip()) < len(content.strip()) * 0.5:
        reasons.append("tool-output-dominant")

    sentences = [s.strip() for s in cleaned.split("\n") if s.strip()]
    if not sentences:
        sentences = [content.strip()]
    if sentences and all(_is_noise_sentence(s) for s in sentences if s):
        reasons.append("all-sentences-noise")

    if content and sum(1 for c in content if c.isalpha()) / len(content) < 0.35:
        reasons.append("code-like-ratio")

    clauses = [c for c in _split_to_clauses(content) if c] or [content.strip()]
    if clauses and all(_is_vetoed(c) for c in clauses):
        reasons.append("transient-veto")

    return reasons


def run_junk_cleanup(conn, user_id: str, agent_id: str) -> int:
    """Delete junk memories for (user_id, agent_id). Returns count deleted.

    Pass 1 — exact duplicates: keep the highest-strength copy (rows are fetched
    DESC by base_strength; first occurrence wins).
    Pass 2 — content heuristics: noise, tool output, transient requests.

    Cascades (ON DELETE CASCADE) remove dependent cue_edges and
    association_edges automatically.
    """
    from thyra.models.memory import compute_content_hash

    rows = conn.execute(
        "SELECT id, content, content_hash "
        "FROM memories WHERE user_id=? AND agent_id=? AND archived=0 "
        "ORDER BY base_strength DESC",
        (user_id, agent_id),
    ).fetchall()

    to_delete: list[str] = []
    seen_hashes: dict[str, str] = {}

    for row in rows:
        chash = row["content_hash"] or compute_content_hash(row["content"])
        if chash in seen_hashes:
            to_delete.append(row["id"])
        else:
            seen_hashes[chash] = row["id"]

    flagged = set(to_delete)
    for row in rows:
        if row["id"] not in flagged and _classify_junk_reasons(row["content"]):
            to_delete.append(row["id"])

    if not to_delete:
        return 0

    placeholders = ",".join("?" * len(to_delete))
    with conn:
        conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", to_delete)

    log.info("Junk cleanup: deleted %d for %s:%s", len(to_delete), user_id, agent_id)
    return len(to_delete)
