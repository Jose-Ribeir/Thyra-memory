"""Admin MCP tools: save, stats, search, toggle, delete."""

from __future__ import annotations

import pathlib
import time
from typing import Any


def _cwd_to_project_dir(cwd: str) -> str:
    """Convert a Windows project cwd to the Claude projects directory name.

    Claude Code converts the project path by replacing path separators:
      J:\\codigo\\Memory_llm  →  J--codigo-Memory-llm
    """
    name = cwd.replace("/", "\\")  # normalize to backslash
    name = name.replace(":\\", "--")  # drive :\ → --
    name = name.replace("\\", "-")  # remaining \ → -
    name = name.replace("_", "-")  # underscore → -
    return name


def _agent_id_from_project_dir_name(proj_dir_name: str) -> str:
    """Derive agent_id by reverse-engineering a Claude project directory name.

    Claude Code encodes the cwd as a directory name under ~/.claude/projects/:
      J:\\codigo\\thyra-ai  →  J--codigo-thyra-ai   (:\\ → --, \\ → -, _ → -)

    Since both path separators and name characters become '-', the encoding is
    lossy. We recover the original path by brute-force: for each '-' in the
    path portion try treating it as '\\', '-', or '_', and check which candidate
    actually exists on disk. Once we find a live directory, resolve_project_id()
    gives us the agent_id.

    Capped at 6 hyphens (3^6 = 729 checks) to stay fast.
    """
    import itertools
    import os

    from thyra.hooks._project import resolve_project_id

    try:
        if "--" not in proj_dir_name:
            return ""
        drive_part, rest = proj_dir_name.split("--", 1)
        drive = drive_part + ":\\"

        hyphen_positions = [i for i, c in enumerate(rest) if c == "-"]
        n = len(hyphen_positions)
        if n > 6:
            return ""

        for choices in itertools.product(["\\", "-", "_"], repeat=n):
            chars = list(rest)
            for pos, ch in zip(hyphen_positions, choices):
                chars[pos] = ch
            candidate = drive + "".join(chars)
            if os.path.isdir(candidate):
                agent_id = resolve_project_id(candidate)
                if agent_id and agent_id not in ("global", "unknown"):
                    return agent_id
    except Exception:
        pass
    return ""


def _find_transcript(cwd: str) -> str:
    """Return path to the most recently modified .jsonl for the given project cwd."""
    projects_dir = pathlib.Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return ""

    if cwd:
        dir_name = _cwd_to_project_dir(cwd)
        candidate = projects_dir / dir_name
        if candidate.is_dir():
            recent = sorted(
                candidate.glob("*.jsonl"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if recent:
                return str(recent[0])

    # Fallback: most recently modified .jsonl across all project dirs
    all_jsonl = sorted(
        projects_dir.rglob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return str(all_jsonl[0]) if all_jsonl else ""


def register_admin_tools(mcp) -> None:
    @mcp.tool()
    def thyra_save(
        content: str,
        category: str = "context",
        memory_type: str = "explicit",
    ) -> dict[str, Any]:
        """Explicitly save a fact to long-term memory.

        Use when the user asks you to 'remember' something or when you identify
        an important persistent fact. category must be one of the 15 seed categories
        (constraints, identity, preferences, relationships, tasks, goals, context,
        skills, habits, knowledge, events, communication, health, finance, routines).
        """
        try:
            from thyra.db.connection import get_conn
            from thyra.models.memory import (
                create_memory,
                get_flag,
                update_memory_strength,
            )
            from thyra.recall.cache import HOT_CACHE
            from thyra.config import (
                THYRA_USER_ID,
                DECAY_EXPLICIT,
                SURFACED_BOOST,
                STRENGTH_CAP,
            )
            from thyra.server.tools._context import get_current_project_id

            agent_id = get_current_project_id()
            conn = get_conn()
            if (
                get_flag(conn, "system_enabled", THYRA_USER_ID, agent_id).lower()
                != "true"
            ):
                return {"success": False, "error": "Memory system is disabled"}

            # Dedup: if a near-identical memory already exists, reinforce it
            # rather than creating a duplicate (same check as the formation pipeline).
            from thyra.formation.dedup import find_near_match

            existing = find_near_match(
                conn, content, THYRA_USER_ID, agent_id, fast=True
            )
            if existing:
                import time as _time

                new_strength = min(
                    STRENGTH_CAP, existing.base_strength + SURFACED_BOOST
                )
                update_memory_strength(
                    conn,
                    existing.id,
                    new_strength,
                    int(_time.time() * 1000),
                    THYRA_USER_ID,
                    agent_id,
                )
                HOT_CACHE.invalidate(f"snapshot:{THYRA_USER_ID}:{agent_id}")
                return {
                    "success": True,
                    "memory_id": existing.id,
                    "category": existing.category,
                    "action": "reinforced",
                    "note": "Near-duplicate found — existing memory reinforced instead of creating duplicate.",
                }

            mem_id = create_memory(
                conn,
                content,
                category=category,
                memory_type=memory_type,
                decay_rate=DECAY_EXPLICIT,
                user_id=THYRA_USER_ID,
                agent_id=agent_id,
            )
            HOT_CACHE.invalidate(f"snapshot:{THYRA_USER_ID}:{agent_id}")
            return {
                "success": True,
                "memory_id": mem_id,
                "category": category,
                "action": "created",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def thyra_stats() -> dict[str, Any]:
        """Return memory system statistics: counts, avg strength, category breakdown."""
        try:
            from thyra.db.connection import get_conn
            from thyra.config import THYRA_USER_ID as U
            from thyra.server.tools._context import get_current_project_id

            A = get_current_project_id()

            conn = get_conn()
            total = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
                (U, A),
            ).fetchone()[0]
            archived = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=1",
                (U, A),
            ).fetchone()[0]
            avg_strength = (
                conn.execute(
                    "SELECT AVG(base_strength) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
                    (U, A),
                ).fetchone()[0]
                or 0.0
            )
            by_cat = conn.execute(
                """SELECT category, COUNT(*) as cnt FROM memories
                   WHERE user_id=? AND agent_id=? AND archived=0
                   GROUP BY category ORDER BY cnt DESC""",
                (U, A),
            ).fetchall()
            cue_count = conn.execute(
                "SELECT COUNT(*) FROM cue_edges WHERE user_id=? AND agent_id=?",
                (U, A),
            ).fetchone()[0]
            assoc_count = conn.execute(
                "SELECT COUNT(*) FROM association_edges WHERE user_id=? AND agent_id=?",
                (U, A),
            ).fetchone()[0]
            return {
                "active_memories": total,
                "archived_memories": archived,
                "avg_strength": round(avg_strength, 3),
                "cue_edges": cue_count,
                "association_edges": assoc_count,
                "by_category": {r["category"]: r["cnt"] for r in by_cat},
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def thyra_search(query: str, limit: int = 10) -> dict[str, Any]:
        """Search memories by keyword (FTS5 full-text search)."""
        try:
            from thyra.db.connection import get_conn
            from thyra.config import THYRA_USER_ID as U
            from thyra.server.tools._context import get_current_project_id
            import math
            import time

            A = get_current_project_id()

            conn = get_conn()
            # Sanitize for FTS5 MATCH: wrap each token as a quoted phrase to prevent
            # special-character injection (hyphens, parens, stars crash FTS5 MATCH).
            tokens = query.strip().split()
            if not tokens:
                return {"results": [], "count": 0}
            fts_query = " ".join('"' + t.replace('"', '""') + '"' for t in tokens)
            rows = conn.execute(
                """SELECT m.id, m.content, m.category, m.base_strength, m.decay_rate,
                          m.last_access, m.archived
                   FROM memories m
                   JOIN memory_fts f ON m.memory_int_id = f.rowid
                   WHERE memory_fts MATCH ? AND m.user_id=? AND m.agent_id=?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, U, A, limit),
            ).fetchall()
            now = int(time.time() * 1000)
            results = []
            for r in rows:
                days = (now - r["last_access"]) / 86_400_000
                level = r["base_strength"] * math.exp(-r["decay_rate"] * days)
                results.append(
                    {
                        "id": r["id"],
                        "content": r["content"][:120],
                        "category": r["category"],
                        "strength": round(level, 3),
                        "archived": bool(r["archived"]),
                    }
                )
            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def thyra_delete_memory(memory_id: str) -> dict[str, Any]:
        """Hard-delete a specific memory by ID. This is permanent."""
        try:
            from thyra.db.connection import get_conn
            from thyra.models.memory import delete_memory
            from thyra.recall.cache import HOT_CACHE
            from thyra.config import THYRA_USER_ID
            from thyra.server.tools._context import get_current_project_id

            agent_id = get_current_project_id()
            conn = get_conn()
            delete_memory(conn, memory_id, THYRA_USER_ID, agent_id)
            HOT_CACHE.invalidate(f"snapshot:{THYRA_USER_ID}:{agent_id}")
            return {"success": True, "deleted": memory_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def thyra_toggle_system(enabled: bool) -> dict[str, Any]:
        """Enable or disable the entire memory system."""
        try:
            from thyra.db.connection import get_conn
            from thyra.models.memory import set_flag
            from thyra.config import THYRA_USER_ID
            from thyra.server.tools._context import get_current_project_id

            agent_id = get_current_project_id()
            conn = get_conn()
            set_flag(
                conn,
                "system_enabled",
                "true" if enabled else "false",
                THYRA_USER_ID,
                agent_id,
            )
            return {"success": True, "system_enabled": enabled}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def thyra_toggle_formation(enabled: bool) -> dict[str, Any]:
        """Enable or disable automatic memory formation from conversation."""
        try:
            from thyra.db.connection import get_conn
            from thyra.models.memory import set_flag
            from thyra.config import THYRA_USER_ID
            from thyra.server.tools._context import get_current_project_id

            agent_id = get_current_project_id()
            conn = get_conn()
            set_flag(
                conn,
                "formation_enabled",
                "true" if enabled else "false",
                THYRA_USER_ID,
                agent_id,
            )
            return {"success": True, "formation_enabled": enabled}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def thyra_set_max_memories(count: int) -> dict[str, Any]:
        """Set the maximum number of memories injected per turn.

        count=0 means unlimited (the token budget governs).
        Useful to reduce context noise when memories are overwhelming.
        Typical values: 3–10. Changes take effect on the next turn.
        """
        try:
            if count < 0:
                return {"success": False, "error": "count must be 0 or positive"}

            from thyra.db.connection import get_conn
            from thyra.models.memory import set_flag
            from thyra.config import THYRA_USER_ID
            from thyra.server.tools._context import get_current_project_id

            agent_id = get_current_project_id()
            conn = get_conn()
            set_flag(conn, "max_memories", str(count), THYRA_USER_ID, agent_id)
            label = "unlimited" if count == 0 else str(count)
            return {"success": True, "max_memories": count, "effective": label}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def thyra_get_max_memories() -> dict[str, Any]:
        """Return the current max_memories setting (0 = unlimited)."""
        try:
            from thyra.db.connection import get_conn
            from thyra.models.memory import get_flag
            from thyra.config import THYRA_USER_ID
            from thyra.server.tools._context import get_current_project_id

            agent_id = get_current_project_id()
            conn = get_conn()
            raw = get_flag(conn, "max_memories", THYRA_USER_ID, agent_id)
            count = int(raw or "0")
            return {
                "max_memories": count,
                "effective": "unlimited" if count == 0 else str(count),
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def thyra_end_turn(memories_used: str = "") -> dict[str, Any]:
        """Trigger memory formation for the current turn — for CCD mode.

        In Claude Desktop App (CCD) the Stop hook never fires, so memories are
        not formed automatically. Call this at the very end of each response to
        run the same formation + reinforcement pipeline the hook would run.

        The delta event is written to delta_queue/ and processed by the
        background worker — this call returns immediately (no ML work in-band).

        memories_used: comma-separated memory IDs from your <memories_used>
        block, e.g. "m_abc123,m_def456". Leave empty if no memories were used.
        """
        import json
        import os
        import pathlib
        import tempfile
        import uuid

        try:
            from thyra.config import THYRA_DB_PATH, THYRA_USER_ID as U
            from thyra.server.tools._context import get_current_project_id

            agent_id = get_current_project_id()

            # Read context file for cwd / session_id
            ctx_path = os.path.join(
                os.environ.get("TEMP", tempfile.gettempdir()),
                "thyra_ctx_latest.json",
            )
            ctx: dict = {}
            if os.path.exists(ctx_path):
                with open(ctx_path, encoding="utf-8") as _f:
                    ctx = json.load(_f)
            cwd = ctx.get("cwd", "")
            # Use the same "unknown" fallback as pre_turn.py and stop_hook.py so
            # the state file name is consistent even when CCD sends session_id:"".
            session_id = ctx.get("session_id") or "unknown"

            # Load served_ids from the turn state file written by recall_pipeline.
            # This is critical for the anti-spoofing check in apply_reinforcement:
            #   valid_ids = declared_set & served_set
            # Without served_ids, the intersection is always empty and no memories
            # are ever reinforced or graduated — the <memories_used> tag does nothing.
            #
            # State files live in the thyra DATA directory (not TEMP) so the MCP
            # server and the hook subprocess always resolve to the same path even if
            # their %TEMP% environments differ in CCD mode.
            served_ids: list[str] = []
            cues_fired: list[str] = []
            turn_id = f"end-turn:{uuid.uuid4().hex[:8]}:{int(time.time() * 1000)}"

            from thyra.recall.intent import _turn_state_path

            state_path = _turn_state_path(session_id)
            if os.path.exists(state_path):
                try:
                    with open(state_path, encoding="utf-8") as _sf:
                        state = json.load(_sf)
                    served_ids = state.get("served_ids", [])
                    cues_fired = state.get("cues_fired", [])
                    # Use the same turn_id that recall_pipeline wrote so the
                    # processed_turns idempotency check can detect duplicates.
                    turn_id = state.get("turn_id", turn_id)
                    # Consume the file — prevents _maybe_flush_prev_turn from
                    # re-queuing the same delta on the next turn.
                    try:
                        os.unlink(state_path)
                    except Exception:
                        pass
                except Exception:
                    pass

            # Find transcript and extract last messages
            transcript_path = _find_transcript(cwd)
            user_text = ""
            assistant_text = ""
            if transcript_path:
                from thyra.hooks.stop_hook import _extract_last_messages

                assistant_text, user_text = _extract_last_messages(transcript_path)

            # Parse declared IDs passed explicitly by the model
            declared_ids = [
                x.strip()
                for x in memories_used.split(",")
                if x.strip().startswith("m_")
            ]

            now_ms = int(time.time() * 1000)
            delta = {
                "session_id": session_id,
                "turn_id": turn_id,
                "user_id": U,
                "agent_id": agent_id,
                "timestamp": now_ms,
                "memories_served": served_ids,  # populated from turn state
                "memories_declared": declared_ids,
                "cues_fired": cues_fired,
                "raw_user_text": user_text,
                "raw_assistant_text": assistant_text,
                "correction_flag": False,
            }

            # Write to delta_queue — background worker handles the ML pipeline.
            # This returns instantly; no synchronous ML work in the MCP call.
            queue_dir = pathlib.Path(THYRA_DB_PATH).parent / "delta_queue"
            queue_dir.mkdir(parents=True, exist_ok=True)
            suffix = uuid.uuid4().hex[:8]
            fname = queue_dir / f"{now_ms}_{suffix}.json"
            tmp = fname.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as _f:
                json.dump(delta, _f)
            tmp.rename(fname)

            return {
                "success": True,
                "queued": fname.name,
                "agent_id": agent_id,
                "transcript_found": bool(transcript_path),
                "memories_declared_used": declared_ids,
                "memories_served_this_turn": served_ids,
                "user_text_chars": len(user_text),
                "assistant_text_chars": len(assistant_text),
            }
        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
