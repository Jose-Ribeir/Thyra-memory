"""Recall-related MCP tools: status, archived recall, lock/unlock."""

from __future__ import annotations

from typing import Any


def register_recall_tools(mcp) -> None:
    @mcp.tool()
    def thyra_init_session(cwd: str = "") -> dict:
        """Initialize Thyra memory context for this conversation and return relevant memories.

        CALL THIS AT THE START OF EVERY NEW CONVERSATION (CCD / Claude Desktop App).

        Pass the current project working directory so memories are scoped to the
        right repo. Returns the full memories XML block — read it and treat it
        exactly like an injected <thyra_memories> block.

        Args:
            cwd: Project working directory, e.g. 'J:\\\\codigo\\\\thyra-ai'
        """
        import json
        import os
        import tempfile
        import time
        import uuid

        from thyra.hooks._project import resolve_project_id
        from thyra.db.connection import get_conn
        from thyra.server.tools._context import write_context
        from thyra.config import THYRA_USER_ID as U

        try:
            agent_id = resolve_project_id(cwd.strip()) if cwd.strip() else "global"

            # Preserve the real session_id written by the UserPromptSubmit hook so
            # thyra_end_turn can find the correct turn-state file.  Only fall back
            # to a synthetic id when no prior context exists (first ever turn).
            ctx_path = os.path.join(
                os.environ.get("TEMP", tempfile.gettempdir()),
                "thyra_ctx_latest.json",
            )
            existing_session_id = ""
            try:
                with open(ctx_path, encoding="utf-8") as _f:
                    existing_session_id = json.load(_f).get("session_id", "")
            except Exception:
                pass
            session_id = existing_session_id or f"mcp-init:{int(time.time() * 1000)}"

            # Update context file so all subsequent MCP tool calls in this
            # conversation automatically use the correct project namespace.
            write_context(session_id, agent_id, cwd)

            conn = get_conn()

            # Direct query — skip the full scoring pipeline to avoid 1.5s cold-start.
            # Session init just needs all strong always-on memories, not cue scoring.
            rows = conn.execute(
                """SELECT id, content, category, base_strength, decay_rate, last_access
                   FROM memories
                   WHERE user_id=? AND agent_id=? AND archived=0
                   ORDER BY base_strength DESC
                   LIMIT 20""",
                (U, agent_id),
            ).fetchall()

            import math
            import datetime

            now_ms = int(time.time() * 1000)
            served_ids = []
            lines = []
            for r in rows:
                age_days = max(0.0, (now_ms - r["last_access"]) / 86_400_000)
                level = r["base_strength"] * math.exp(-r["decay_rate"] * age_days)
                if level < 0.05:
                    continue  # skip nearly-forgotten memories
                served_ids.append(r["id"])
                lines.append(
                    f'[MEMORY id="{r["id"]}" cat="{r["category"]}" '
                    f'strength="{r["base_strength"]:.2f}" age_days="{int(age_days)}"]\n'
                    f"{r['content']}\n[/MEMORY]"
                )

            if lines:
                ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                xml = (
                    f'<thyra_memories agent="{agent_id}" retrieved_at="{ts}">\n'
                    + "\n".join(lines)
                    + "\n</thyra_memories>"
                )
            else:
                xml = ""

            # Store turn state so thyra_end_turn / the auto-monitor can recover
            # served_ids for the anti-spoofing check and cue-edge updates.
            # No cues were fired (direct query, no scoring), so cues_fired=[].
            # This is still essential — without it served_ids stays [] and
            # <memories_used> tags never reinforce anything.
            turn_id = f"init:{uuid.uuid4().hex[:8]}:{now_ms}"
            from thyra.recall.intent import _store_turn_state

            _store_turn_state(session_id, turn_id, served_ids, [])

            return {
                "project_id": agent_id,
                "memories_xml": xml,
                "memory_count": len(served_ids),
                "note": (
                    "Treat memories_xml exactly like an injected <thyra_memories> block. "
                    "Include relevant memories in your reasoning and end your response "
                    "with <memories_used>id1,id2</memories_used> or <memories_used></memories_used>."
                ),
            }
        except Exception as e:
            return {
                "error": str(e),
                "project_id": "unknown",
                "memories_xml": "",
                "memory_count": 0,
            }

    @mcp.tool()
    def thyra_status() -> dict[str, Any]:
        """Check memory system health and current settings."""
        try:
            from thyra.db.connection import get_conn
            from thyra.models.memory import get_flag, count_active
            from thyra.config import THYRA_USER_ID as U
            from thyra.server.tools._context import get_current_project_id

            A = get_current_project_id()
            conn = get_conn()
            import time as _t
            from thyra.config import THYRA_AGENT_ID as _GA_DEFAULT

            now_ms = int(_t.time() * 1000)
            last_ping = int(
                get_flag(conn, "last_monitor_ping", U, _GA_DEFAULT, default="0") or "0"
            )
            monitor_ok = last_ping > 0 and (now_ms - last_ping) < 60_000
            return {
                "system_enabled": get_flag(conn, "system_enabled", U, A) == "true",
                "formation_enabled": get_flag(conn, "formation_enabled", U, A)
                == "true",
                "active_memories": count_active(conn, U, A),
                "user_id": U,
                "agent_id": A,
                "monitor_ok": monitor_ok,
                "last_monitor_ping": last_ping,
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def thyra_recall_archived(query: str, limit: int = 5) -> dict[str, Any]:
        """Fetch archived (forgotten) memories matching a keyword query.

        Use when the user asks about something that may have been forgotten
        (e.g. 'remember when we tried X?'). Found memories will be resurrected
        if their cue activation is strong enough.
        """
        try:
            from thyra.db.connection import get_conn
            from thyra.config import THYRA_USER_ID as U
            from thyra.server.tools._context import get_current_project_id
            import math, time

            A = get_current_project_id()

            conn = get_conn()
            rows = conn.execute(
                """SELECT m.id, m.content, m.category, m.base_strength, m.archived_at
                   FROM memories m
                   JOIN memory_fts f ON m.memory_int_id = f.rowid
                   WHERE memory_fts MATCH ? AND m.user_id=? AND m.agent_id=? AND m.archived=1
                   ORDER BY rank
                   LIMIT ?""",
                (query, U, A, limit),
            ).fetchall()
            results = []
            now = int(time.time() * 1000)
            for r in rows:
                days_archived = (now - (r["archived_at"] or now)) // 86_400_000
                results.append(
                    {
                        "id": r["id"],
                        "content": r["content"][:200],
                        "category": r["category"],
                        "days_archived": days_archived,
                    }
                )
            return {"archived_matches": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def thyra_lock_memory(memory_id: str, token: str) -> dict[str, Any]:
        """Encrypt a memory's content at rest. Requires a user-provided token.

        The token is never stored. Content is only decryptable with the same token.
        No lexical cues will be extracted from locked memory content.
        """
        try:
            from thyra.db.connection import get_conn
            from thyra.locks.crypto import MemoryLocker
            from thyra.recall.cache import HOT_CACHE
            from thyra.config import THYRA_USER_ID as U
            from thyra.server.tools._context import get_current_project_id

            A = get_current_project_id()

            conn = get_conn()
            row = conn.execute(
                "SELECT content, locked FROM memories WHERE id=? AND user_id=? AND agent_id=?",
                (memory_id, U, A),
            ).fetchone()
            if not row:
                return {"success": False, "error": "Memory not found"}
            if row["locked"]:
                return {"success": False, "error": "Memory is already locked"}

            locker = MemoryLocker()
            original_content = row["content"]
            encrypted = locker.encrypt(original_content, token, memory_id, A)

            # Get the memory_int_id for FTS5 operations before updating
            int_row = conn.execute(
                "SELECT memory_int_id FROM memories WHERE id=? AND user_id=? AND agent_id=?",
                (memory_id, U, A),
            ).fetchone()
            memory_int_id = int_row["memory_int_id"] if int_row else None

            with conn:
                conn.execute(
                    "UPDATE memories SET content=?, locked=1 WHERE id=? AND user_id=? AND agent_id=?",
                    (encrypted, memory_id, U, A),
                )
                # The FTS5 trigger fires on UPDATE and indexes the encrypted blob.
                # Manually clear it by deleting then re-inserting with empty content.
                if memory_int_id is not None:
                    conn.execute(
                        "INSERT INTO memory_fts(memory_fts, rowid, content) VALUES ('delete', ?, ?)",
                        (memory_int_id, original_content),
                    )
                    conn.execute(
                        "INSERT INTO memory_fts(rowid, content) VALUES (?, '')",
                        (memory_int_id,),
                    )
                # Remove all cue edges for this memory (no cues from encrypted content)
                conn.execute(
                    "DELETE FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?",
                    (memory_id, U, A),
                )
            HOT_CACHE.invalidate(f"snapshot:{U}:{A}")
            return {"success": True, "memory_id": memory_id, "locked": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def thyra_unlock_memory(memory_id: str, token: str) -> dict[str, Any]:
        """Decrypt a locked memory for this response only. Content is NOT stored back.

        Returns the plaintext content. You must ask the user for their token.
        """
        try:
            from thyra.db.connection import get_conn
            from thyra.locks.crypto import MemoryLocker
            from thyra.config import THYRA_USER_ID as U
            from thyra.server.tools._context import get_current_project_id

            A = get_current_project_id()

            conn = get_conn()
            row = conn.execute(
                "SELECT content, locked, category FROM memories WHERE id=? AND user_id=? AND agent_id=?",
                (memory_id, U, A),
            ).fetchone()
            if not row:
                return {"success": False, "error": "Memory not found"}
            if not row["locked"]:
                return {"success": False, "error": "Memory is not locked"}

            locker = MemoryLocker()
            plaintext = locker.decrypt(row["content"], token, memory_id, A)
            return {
                "success": True,
                "memory_id": memory_id,
                "category": row["category"],
                "content": plaintext,
                "note": "This content is for this response only and has not been stored in plaintext.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
