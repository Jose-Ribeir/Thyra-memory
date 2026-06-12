"""Drain the delta queue and run the nightly sweep for all pairs, then exit.

Usage:
    python scripts/run_worker_once.py

Run this as a Windows Scheduled Task (every 15 min) to ensure consolidation
and nightly maintenance happen even when the MCP server is not running.
"""

from __future__ import annotations

import datetime
import io
import sys
import time

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import sqlite3

from thyra.config import THYRA_DB_PATH
from thyra.consolidation.drain import drain_queue, run_nightly_for_all


def main() -> None:
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] run_worker_once started")

    # Step 1: drain any pending delta files
    start = time.time()
    n = drain_queue(THYRA_DB_PATH, budget_ms=None)
    print(f"  Drained {n} delta(s) in {time.time() - start:.2f}s")

    # Step 2: nightly sweep for all overdue pairs
    conn = sqlite3.connect(THYRA_DB_PATH)
    conn.row_factory = sqlite3.Row
    pairs = conn.execute(
        "SELECT DISTINCT user_id, agent_id, COUNT(*) as mem_count "
        "FROM memories WHERE archived=0 GROUP BY user_id, agent_id"
    ).fetchall()
    conn.close()

    if not pairs:
        print("  No active memories — nothing to sweep.")
        return

    print(f"\n  Checking {len(pairs)} namespace(s) for overdue nightly sweep...")

    results = run_nightly_for_all(THYRA_DB_PATH)
    for key, summary in results.items():
        if summary.get("skipped"):
            print(f"    {key}: recent sweep, skipped")
        elif summary.get("error"):
            print(f"    {key}: ERROR — {summary['error']}")
        else:
            non_zero = {k: v for k, v in summary.items() if v}
            summary_str = (
                ", ".join(f"{k}: {v}" for k, v in non_zero.items()) or "nothing to do"
            )
            print(f"    {key}: {summary_str}")

    print(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] done")


if __name__ == "__main__":
    main()
