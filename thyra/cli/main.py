"""Thyra CLI — admin interface for the memory system.

Usage:
    python -m thyra.cli stats
    python -m thyra.cli list [--cat <category>] [--archived]
    python -m thyra.cli search <query>
    python -m thyra.cli delete <memory_id>
    python -m thyra.cli export [--format json|csv]
    python -m thyra.cli toggle <on|off>
    python -m thyra.cli toggle-formation <on|off>
    python -m thyra.cli reset
    python -m thyra.cli nightly
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from thyra.config import THYRA_DB_PATH, THYRA_USER_ID, THYRA_AGENT_ID
from thyra.db.connection import DBConnection
from thyra.models.memory import get_flag, set_flag


def _conn():
    return DBConnection.get(THYRA_DB_PATH)


def cmd_stats(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    total = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (u, a),
    ).fetchone()[0]
    archived = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=1",
        (u, a),
    ).fetchone()[0]
    avg_strength = (
        conn.execute(
            "SELECT AVG(base_strength) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
            (u, a),
        ).fetchone()[0]
        or 0.0
    )
    by_cat = conn.execute(
        "SELECT category, COUNT(*) AS n FROM memories WHERE user_id=? AND agent_id=? AND archived=0 "
        "GROUP BY category ORDER BY n DESC",
        (u, a),
    ).fetchall()

    sys_on = get_flag(conn, "system_enabled", u, a)
    form_on = get_flag(conn, "formation_enabled", u, a)
    last_nightly = int(get_flag(conn, "last_nightly", u, a) or 0)
    max_mem_raw = get_flag(conn, "max_memories", u, a)
    max_mem = int(max_mem_raw or "0")
    max_mem_label = "unlimited" if max_mem == 0 else str(max_mem)

    print(f"Thyra Memory Stats  [{u} / {a}]")
    print(f"  Active memories : {total}")
    print(f"  Archived        : {archived}")
    print(f"  Avg strength    : {avg_strength:.4f}")
    print(f"  System enabled  : {sys_on}")
    print(f"  Formation enabled: {form_on}")
    print(f"  Max per turn    : {max_mem_label}")
    if last_nightly:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_nightly / 1000))
        print(f"  Last nightly    : {ts}")
    else:
        print("  Last nightly    : never")
    print("\n  By category:")
    for row in by_cat:
        print(f"    {row['category']:<20} {row['n']}")


def cmd_list(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    archived_filter = 1 if args.archived else 0
    if args.cat:
        rows = conn.execute(
            "SELECT id, category, base_strength, content, probationary, archived "
            "FROM memories WHERE user_id=? AND agent_id=? AND archived=? AND category=? "
            "ORDER BY base_strength DESC LIMIT 100",
            (u, a, archived_filter, args.cat),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, category, base_strength, content, probationary, archived "
            "FROM memories WHERE user_id=? AND agent_id=? AND archived=? "
            "ORDER BY base_strength DESC LIMIT 100",
            (u, a, archived_filter),
        ).fetchall()
    for row in rows:
        prob = " [P]" if row["probationary"] else ""
        snippet = row["content"][:70].replace("\n", " ")
        print(
            f"  {row['id']}  str={row['base_strength']:.3f}  cat={row['category']}{prob}  {snippet!r}"
        )


def cmd_search(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    query = " ".join(args.query)
    rows = conn.execute(
        """SELECT m.id, m.category, m.base_strength, m.content
           FROM memory_fts fts
           JOIN memories m ON fts.rowid = m.memory_int_id
           WHERE fts.content MATCH ? AND m.user_id=? AND m.agent_id=?
           ORDER BY rank LIMIT 20""",
        (query, u, a),
    ).fetchall()
    if not rows:
        print("No results.")
        return
    for row in rows:
        snippet = row["content"][:80].replace("\n", " ")
        print(
            f"  {row['id']}  str={row['base_strength']:.3f}  cat={row['category']}  {snippet!r}"
        )


def cmd_delete(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    mem_id = args.memory_id
    row = conn.execute(
        "SELECT content FROM memories WHERE id=? AND user_id=? AND agent_id=?",
        (mem_id, u, a),
    ).fetchone()
    if not row:
        print(f"Memory {mem_id} not found.")
        sys.exit(1)
    snippet = row["content"][:60]
    confirm = input(f"Delete {mem_id} ({snippet!r})? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return
    with conn:
        conn.execute(
            "DELETE FROM memories WHERE id=? AND user_id=? AND agent_id=?",
            (mem_id, u, a),
        )
    print(f"Deleted {mem_id}.")


def cmd_export(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    rows = conn.execute(
        "SELECT id, category, memory_type, base_strength, decay_rate, content, "
        "probationary, archived, created_at, last_access, use_count "
        "FROM memories WHERE user_id=? AND agent_id=? ORDER BY created_at",
        (u, a),
    ).fetchall()
    if args.format == "csv":
        import csv, io

        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=list(dict(rows[0]).keys()) if rows else [])
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))
        print(out.getvalue(), end="")
    else:
        print(json.dumps([dict(r) for r in rows], indent=2))


def cmd_toggle(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    value = "true" if args.state == "on" else "false"
    set_flag(conn, "system_enabled", value, u, a)
    conn.commit()
    print(f"System memory {'enabled' if value == 'true' else 'disabled'}.")


def cmd_toggle_formation(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    value = "true" if args.state == "on" else "false"
    set_flag(conn, "formation_enabled", value, u, a)
    conn.commit()
    print(f"Memory formation {'enabled' if value == 'true' else 'disabled'}.")


def cmd_reset(args) -> None:
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    confirm = input(
        f"This will DELETE ALL memories for {u}/{a}. Type 'yes' to confirm: "
    )
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return
    conn = _conn()
    with conn:
        conn.execute("DELETE FROM memories WHERE user_id=? AND agent_id=?", (u, a))
        conn.execute("DELETE FROM cue_nodes WHERE user_id=? AND agent_id=?", (u, a))
        conn.execute("DELETE FROM cue_edges WHERE user_id=? AND agent_id=?", (u, a))
        conn.execute(
            "DELETE FROM association_edges WHERE user_id=? AND agent_id=?", (u, a)
        )
        conn.execute(
            "DELETE FROM situation_edges WHERE user_id=? AND agent_id=?", (u, a)
        )
        conn.execute("DELETE FROM turn_log WHERE user_id=? AND agent_id=?", (u, a))
        conn.execute("DELETE FROM processed_turns")
    conn.commit()
    print("All memories deleted.")


def cmd_nightly(args) -> None:
    from thyra.consolidation.nightly import run_nightly_sweep

    conn = _conn()
    print(f"Running nightly sweep for {THYRA_USER_ID}/{THYRA_AGENT_ID}...")
    summary = run_nightly_sweep(conn, THYRA_USER_ID, THYRA_AGENT_ID)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("Done.")


def cmd_cleanup(args) -> None:
    """Remove junk memories: tool output, code dumps, narration, exact duplicates."""
    from scripts.cleanup_junk import run_cleanup

    conn = _conn()
    if args.all_namespaces:
        pairs = conn.execute(
            "SELECT DISTINCT user_id, agent_id FROM memories WHERE archived=0"
        ).fetchall()
    else:
        pairs = [(THYRA_USER_ID, THYRA_AGENT_ID)]

    total = 0
    for row in pairs:
        uid = row[0] if isinstance(row, (list, tuple)) else row["user_id"]
        aid = row[1] if isinstance(row, (list, tuple)) else row["agent_id"]
        result = run_cleanup(conn, uid, aid, apply=args.apply)
        total += result["flagged"]

    print(f"\nTotal flagged: {total}")
    if not args.apply:
        print("Re-run with --apply to delete.")


def cmd_set_max_memories(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    count = args.count
    if count < 0:
        print("Error: count must be 0 or positive (0 = unlimited).")
        sys.exit(1)
    set_flag(conn, "max_memories", str(count), u, a)
    conn.commit()
    label = "unlimited" if count == 0 else str(count)
    print(f"Max memories per turn set to: {label}")


def cmd_get_max_memories(args) -> None:
    conn = _conn()
    u, a = THYRA_USER_ID, THYRA_AGENT_ID
    raw = get_flag(conn, "max_memories", u, a)
    count = int(raw or "0")
    label = "unlimited" if count == 0 else str(count)
    print(f"Max memories per turn: {label}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thyra", description="Thyra memory admin CLI")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("stats", help="Show memory statistics")

    lst = sub.add_parser("list", help="List memories")
    lst.add_argument("--cat", help="Filter by category")
    lst.add_argument("--archived", action="store_true", help="Show archived memories")

    srch = sub.add_parser("search", help="Full-text search")
    srch.add_argument("query", nargs="+", help="Search terms")

    dlt = sub.add_parser("delete", help="Hard-delete a memory by ID")
    dlt.add_argument("memory_id")

    exp = sub.add_parser("export", help="Export all memories")
    exp.add_argument("--format", choices=["json", "csv"], default="json")

    tog = sub.add_parser("toggle", help="Enable or disable the memory system")
    tog.add_argument("state", choices=["on", "off"])

    togf = sub.add_parser("toggle-formation", help="Enable or disable auto-formation")
    togf.add_argument("state", choices=["on", "off"])

    smm = sub.add_parser(
        "set-max-memories",
        help="Cap how many memories inject per turn (0 = unlimited)",
    )
    smm.add_argument("count", type=int, help="Max memories (0 = unlimited)")

    sub.add_parser("get-max-memories", help="Show the current max-memories setting")

    sub.add_parser("reset", help="Delete ALL memories (destructive)")
    sub.add_parser("nightly", help="Run the nightly sweep now")

    cln = sub.add_parser(
        "cleanup", help="Preview and remove junk memories (noise, duplicates)"
    )
    cln.add_argument(
        "--apply", action="store_true", help="Hard-delete (default: dry-run)"
    )
    cln.add_argument(
        "--all-namespaces", action="store_true", help="All user/agent pairs"
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "stats": cmd_stats,
        "list": cmd_list,
        "search": cmd_search,
        "delete": cmd_delete,
        "export": cmd_export,
        "toggle": cmd_toggle,
        "toggle-formation": cmd_toggle_formation,
        "set-max-memories": cmd_set_max_memories,
        "get-max-memories": cmd_get_max_memories,
        "reset": cmd_reset,
        "nightly": cmd_nightly,
        "cleanup": cmd_cleanup,
    }
    if args.command not in commands:
        parser.print_help()
        sys.exit(0)
    commands[args.command](args)


if __name__ == "__main__":
    main()
