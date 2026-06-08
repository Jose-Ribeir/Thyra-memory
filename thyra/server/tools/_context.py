"""Resolve the current project context for MCP tool calls.

pre_turn.py writes TEMP/thyra_ctx_latest.json before every message.
MCP tools read it here so they operate on the same repo-scoped namespace
as the recall/stop hooks, without needing the cwd passed as a parameter.
"""

from __future__ import annotations

import json
import os
import tempfile


def write_context(session_id: str, project_id: str, cwd: str) -> None:
    """Write project context to temp file for cross-process communication.

    Called by thyra_init_session (MCP tool) so that subsequent MCP tool calls
    in the same conversation use the correct repo-scoped namespace even when
    the UserPromptSubmit hook doesn't fire (e.g. CCD Desktop App mode).
    """
    try:
        ctx = {"project_id": project_id, "session_id": session_id, "cwd": cwd}
        path = os.path.join(
            os.environ.get("TEMP", tempfile.gettempdir()),
            "thyra_ctx_latest.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ctx, f)
    except Exception:
        pass


def get_current_project_id() -> str:
    """Return the project ID for the active Claude Code session.

    Reads thyra_ctx_latest.json. When the stored project_id is "global" or
    "unknown" (i.e. the context was last written with an empty cwd), tries to
    recover a more specific ID from the most recently modified transcript path
    before falling back to the env var / "global".
    """
    cwd_from_ctx = ""
    try:
        path = os.path.join(
            os.environ.get("TEMP", tempfile.gettempdir()),
            "thyra_ctx_latest.json",
        )
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                ctx = json.load(f)
            pid = ctx.get("project_id", "")
            cwd_from_ctx = ctx.get("cwd", "")
            if pid and pid not in ("global", "unknown"):
                return pid
    except Exception:
        pass

    # Fallback: try to recover from the most recent transcript path.
    # The transcript lives at ~/.claude/projects/<proj-dir>/<session>.jsonl.
    # <proj-dir> encodes the cwd, so we can reverse-engineer the project slug.
    try:
        import pathlib
        from thyra.server.tools.admin_tools import (
            _agent_id_from_project_dir_name,
            _find_transcript,
        )

        # If we have a cwd from the context file, try it first
        if cwd_from_ctx:
            from thyra.hooks._project import resolve_project_id

            pid = resolve_project_id(cwd_from_ctx)
            if pid and pid not in ("global", "unknown"):
                return pid

        transcript = _find_transcript(cwd_from_ctx)
        if transcript:
            proj_dir_name = pathlib.Path(transcript).parent.name
            recovered = _agent_id_from_project_dir_name(proj_dir_name)
            if recovered and recovered not in ("global", "unknown"):
                return recovered
    except Exception:
        pass

    return os.environ.get("THYRA_AGENT_ID", "global")
