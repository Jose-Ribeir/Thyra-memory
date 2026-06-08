"""Cleanup junk memories from the Thyra database.

Scans the active memory store for:
  - Tool/machine output (line-numbered code, task output headers, tracebacks)
  - Procedural narration (agent thinking aloud rather than stating facts)
  - Code dumps (high non-alphabetic character ratio)
  - Bare filesystem paths with no prose
  - Exact-content duplicates (by content_hash — keeps highest base_strength)

Usage:
    python scripts/cleanup_junk.py                 # dry-run (preview only)
    python scripts/cleanup_junk.py --apply         # hard delete matching rows
    python scripts/cleanup_junk.py --all-namespaces  # all user/agent pairs
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

from thyra.config import THYRA_DB_PATH, THYRA_USER_ID, THYRA_AGENT_ID
from thyra.db.connection import DBConnection
from thyra.formation.pipeline import _is_noise_sentence, _clean_for_formation
from thyra.formation.salience import _is_vetoed, _split_to_clauses
from thyra.models.memory import compute_content_hash

_MEM_ID_RE = re.compile(r"\bm_[0-9a-f]{16}\b")


def _classify_junk_reasons(content: str) -> list[str]:
    """Return list of reason strings if this content is junk, else []."""
    reasons = []
    # Memory ID listing: content contains multiple Thyra memory IDs
    ids_found = _MEM_ID_RE.findall(content)
    if ids_found:
        non_id_text = _MEM_ID_RE.sub("", content).strip()
        if len(non_id_text) < 50 or len(ids_found) >= 2:
            reasons.append("memory-id-listing")
    if reasons:
        return reasons  # no need to check further
    cleaned = _clean_for_formation(content)
    # If cleaning stripped >50% of the content, the original was mostly tool output
    if len(content) > 20 and len(cleaned.strip()) < len(content.strip()) * 0.5:
        reasons.append("tool-output-dominant")
    # Check each sentence
    sentences = [s.strip() for s in cleaned.split("\n") if s.strip()]
    if not sentences:
        sentences = [content.strip()]
    all_noise = all(_is_noise_sentence(s) for s in sentences if s)
    if all_noise and sentences:
        reasons.append("all-sentences-noise")
    # High non-alphabetic ratio on full content
    alpha = sum(1 for c in content if c.isalpha())
    if content and alpha / len(content) < 0.35:
        reasons.append("code-like-ratio")
    # Backfill the L1 transience / self-containedness vetoes over the live DB:
    # flag a memory whose every clause the new vetoes would now reject (a transient
    # single-turn request or a dangling deictic referent stored as a "fact").
    clauses = [c for c in _split_to_clauses(content) if c] or [content.strip()]
    if clauses and all(_is_vetoed(c) for c in clauses):
        reasons.append("transient-veto")
    return reasons


def run_cleanup(
    conn,
    user_id: str,
    agent_id: str,
    apply: bool,
) -> dict:
    rows = conn.execute(
        "SELECT id, content, base_strength, category, content_hash, probationary "
        "FROM memories WHERE user_id=? AND agent_id=? AND archived=0 "
        "ORDER BY base_strength DESC",
        (user_id, agent_id),
    ).fetchall()

    to_delete: list[tuple[str, str]] = []  # (memory_id, reason)
    seen_hashes: dict[str, str] = {}  # hash -> best id (keep this one)

    # Pass 1: identify exact duplicates (keep highest base_strength row — already ordered DESC)
    for row in rows:
        chash = row["content_hash"]
        if not chash:
            chash = compute_content_hash(row["content"])
        if chash in seen_hashes:
            to_delete.append((row["id"], "exact-duplicate"))
        else:
            seen_hashes[chash] = row["id"]

    # Pass 2: identify junk content
    already_flagged = {mid for mid, _ in to_delete}
    for row in rows:
        if row["id"] in already_flagged:
            continue
        reasons = _classify_junk_reasons(row["content"])
        if reasons:
            to_delete.append((row["id"], "+".join(reasons)))

    # Report
    print(f"\n[{user_id}/{agent_id}] Active memories: {len(rows)}")
    print(f"  Would {'delete' if apply else 'delete (dry-run)'}: {len(to_delete)}")
    if to_delete:
        print("\n  Rows flagged:")
        for mid, reason in to_delete:
            row = next((r for r in rows if r["id"] == mid), None)
            snippet = (row["content"][:70].replace("\n", " ")) if row else "?"
            # Encode safely for Windows console (replace unrepresentable chars)
            safe_snippet = snippet.encode(
                sys.stdout.encoding or "utf-8", errors="replace"
            ).decode(sys.stdout.encoding or "utf-8")
            print(f"    {mid}  [{reason}]  {safe_snippet!r}")

    if apply and to_delete:
        ids = [mid for mid, _ in to_delete]
        placeholders = ",".join("?" * len(ids))
        # cue_edges links to memories via src_id (memory_id is cue_node side)
        try:
            conn.execute(f"DELETE FROM cue_edges WHERE src_id IN ({placeholders})", ids)
        except Exception:
            pass
        try:
            conn.execute(f"DELETE FROM cue_edges WHERE dst_id IN ({placeholders})", ids)
        except Exception:
            pass
        try:
            conn.execute(
                f"DELETE FROM association_edges WHERE memory_a IN ({placeholders})", ids
            )
            conn.execute(
                f"DELETE FROM association_edges WHERE memory_b IN ({placeholders})", ids
            )
        except Exception:
            pass
        conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
        conn.commit()
        print(f"\n  Hard-deleted {len(ids)} memories.")
    elif not apply:
        print("\n  (Dry-run — pass --apply to execute)")

    return {"flagged": len(to_delete), "applied": apply}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean junk memories from Thyra DB")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (default is dry-run preview)",
    )
    parser.add_argument(
        "--all-namespaces",
        action="store_true",
        help="Sweep all user/agent pairs instead of just the default",
    )
    parser.add_argument("--db", default=THYRA_DB_PATH, help="Path to thyra.db")
    args = parser.parse_args()

    conn = DBConnection.get(args.db)

    if args.all_namespaces:
        pairs = conn.execute(
            "SELECT DISTINCT user_id, agent_id FROM memories WHERE archived=0"
        ).fetchall()
    else:
        pairs = [(THYRA_USER_ID, THYRA_AGENT_ID)]

    total_flagged = 0
    for row in pairs:
        uid = row[0] if isinstance(row, (list, tuple)) else row["user_id"]
        aid = row[1] if isinstance(row, (list, tuple)) else row["agent_id"]
        result = run_cleanup(conn, uid, aid, apply=args.apply)
        total_flagged += result["flagged"]

    print(f"\nTotal flagged: {total_flagged}")
    if not args.apply:
        print("Re-run with --apply to delete.")


if __name__ == "__main__":
    main()
