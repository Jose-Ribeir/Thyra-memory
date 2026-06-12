"""Force the nightly sweep for every active (user_id, agent_id) pair.

Usage:
    python scripts/run_nightly.py

Useful when the worker has been offline or you want to force an immediate
sweep across all namespaces without waiting for the 24h clock or a delta event.
"""

from __future__ import annotations

import io
import sqlite3
import sys
import time

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from thyra.config import THYRA_DB_PATH
from thyra.consolidation.nightly import run_nightly_sweep
from thyra.models.memory import get_flag


def main() -> None:
    conn = sqlite3.connect(THYRA_DB_PATH)
    conn.row_factory = sqlite3.Row

    pairs = conn.execute(
        "SELECT DISTINCT user_id, agent_id, COUNT(*) as mem_count "
        "FROM memories WHERE archived=0 GROUP BY user_id, agent_id"
    ).fetchall()

    if not pairs:
        print("No active memories found — nothing to sweep.")
        return

    print(f"Found {len(pairs)} active namespace(s):\n")

    total_start = time.time()
    for row in pairs:
        user_id, agent_id, mem_count = row["user_id"], row["agent_id"], row["mem_count"]

        last_ms = float(get_flag(conn, "last_nightly", user_id, agent_id, default="0"))
        if last_ms:
            import datetime

            last_str = datetime.datetime.fromtimestamp(last_ms / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            last_str = "never"

        print(
            f"  {user_id}:{agent_id}  ({mem_count} active memories, last sweep: {last_str})"
        )
        start = time.time()
        summary = run_nightly_sweep(conn, user_id, agent_id)
        elapsed = time.time() - start

        non_zero = {k: v for k, v in summary.items() if v > 0}
        summary_str = (
            ", ".join(f"{k}: {v}" for k, v in non_zero.items())
            if non_zero
            else "nothing to do"
        )
        print(f"    done in {elapsed:.2f}s — {summary_str}\n")

    print(f"All namespaces swept in {time.time() - total_start:.2f}s.")


if __name__ == "__main__":
    main()
