"""UserPromptSubmit hook: run recall_pipeline, return additionalContext JSON.

Claude Code pipes a JSON event to stdin; this script prints the additionalContext
JSON to stdout and exits 0. It must NEVER exit non-zero or raise.
"""

import json
import os
import pathlib
import sys
import tempfile
import time
import uuid

from thyra.hooks._project import resolve_project_id
from thyra.hooks._stdin import read_stdin_text


def _write_turn_state(session_id: str, turn_id: str) -> None:
    """Write a minimal turn-state file unconditionally.

    recall_pipeline() may overwrite this with actual served_ids if it finds
    memories. If it finds none, this file still exists — so _maybe_flush_prev_turn()
    can detect on the *next* turn that the Stop hook didn't fire (CCD) and queue
    the delta for formation. Without this, an empty DB can never bootstrap its
    first memory because _store_turn_state() in recall_pipeline() is only called
    when memories are actually selected.

    File lives in the thyra data dir (not TEMP) so the hook subprocess and the
    MCP server always resolve to the same path regardless of process environment.
    """
    try:
        from thyra.hooks.stop_hook import _turn_state_path

        state = {"turn_id": turn_id, "served_ids": [], "cues_fired": []}
        with open(_turn_state_path(session_id), "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


def _write_context(session_id: str, project_id: str, cwd: str) -> None:
    try:
        path = os.path.join(
            os.environ.get("TEMP", tempfile.gettempdir()),
            "thyra_ctx_latest.json",
        )
        # Don't overwrite a specific project_id with "global".
        # In CCD mode the hook sometimes fires with cwd="" which resolves to
        # "global", clobbering the correct project written at server startup.
        # Keep the existing project_id unless the new one is more specific.
        if project_id in ("global", "unknown"):
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        prev = json.load(f)
                    prev_pid = prev.get("project_id", "")
                    if prev_pid and prev_pid not in ("global", "unknown", ""):
                        # Keep the existing specific project_id; only update session_id
                        ctx = {
                            "project_id": prev_pid,
                            "session_id": session_id,
                            "cwd": prev.get("cwd", cwd),
                        }
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(ctx, f)
                        return
                except Exception:
                    pass
        ctx = {"project_id": project_id, "session_id": session_id, "cwd": cwd}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ctx, f)
    except Exception:
        pass


def _maybe_flush_prev_turn() -> None:
    """CCD fallback: if the Stop hook didn't fire last turn, run formation now.

    The Stop hook consumes (reads + deletes) thyra_{session_id}.json when it runs.
    If that file still exists at the start of the next turn, the hook never fired
    (CCD mode). We enqueue the delta here so formation still happens — one turn late,
    but reliably, without requiring any MCP tool call.
    """
    try:
        ctx_path = os.path.join(
            os.environ.get("TEMP", tempfile.gettempdir()),
            "thyra_ctx_latest.json",
        )
        if not os.path.exists(ctx_path):
            return
        with open(ctx_path, encoding="utf-8") as f:
            prev_ctx = json.load(f)

        prev_session_id = prev_ctx.get("session_id", "") or "unknown"
        prev_cwd = prev_ctx.get("cwd", "")

        from thyra.config import THYRA_DB_PATH
        from thyra.hooks.stop_hook import (
            _detect_correction,
            _enqueue_delta,
            _extract_last_messages,
            _load_turn_state,
            _parse_declared,
            _turn_state_path,
        )

        # If state file is gone, the Stop hook (or thyra_end_turn) already fired.
        if not os.path.exists(_turn_state_path(prev_session_id)):
            return

        served_ids, turn_id, cues_fired = _load_turn_state(prev_session_id)

        # Locate transcript for the previous session.
        transcript_path = ""
        projects = pathlib.Path.home() / ".claude" / "projects"
        if prev_session_id and prev_session_id != "unknown":
            for pdir in projects.iterdir():
                if pdir.is_dir():
                    candidate = pdir / f"{prev_session_id}.jsonl"
                    if candidate.exists():
                        transcript_path = str(candidate)
                        break
        if not transcript_path:
            agent_slug = resolve_project_id(prev_cwd).lower()
            for pdir in projects.iterdir():
                if pdir.is_dir() and agent_slug in pdir.name.lower():
                    recent = sorted(
                        pdir.glob("*.jsonl"),
                        key=lambda f: f.stat().st_mtime,
                        reverse=True,
                    )
                    if recent:
                        transcript_path = str(recent[0])
                        break

        assistant_text, user_text = _extract_last_messages(transcript_path)
        declared_ids = _parse_declared(assistant_text)

        db_path = os.environ.get("THYRA_DB_PATH", THYRA_DB_PATH)
        _flush_agent_id = resolve_project_id(prev_cwd)
        if _flush_agent_id in ("", "global", "unknown"):
            return  # unresolved project — skip rather than pollute the global namespace
        delta = {
            "session_id": prev_session_id,
            "turn_id": turn_id,
            "user_id": os.environ.get("THYRA_USER_ID", "default"),
            "agent_id": _flush_agent_id,
            "timestamp": int(time.time() * 1000),
            "memories_served": served_ids,
            "memories_declared": declared_ids,
            "cues_fired": cues_fired,
            "raw_user_text": user_text,
            "raw_assistant_text": assistant_text,
            "correction_flag": _detect_correction(user_text),
        }
        _enqueue_delta(delta, db_path)
    except Exception:
        pass  # must never fail or block recall


def main() -> None:
    try:
        raw = read_stdin_text()
        event = json.loads(raw) if raw.strip() else {}

        prompt = event.get("prompt", "")
        session_id = event.get("session_id", "unknown")
        cwd = event.get("cwd", "")
        turn_id = f"{session_id}:{int(time.time() * 1000)}:{uuid.uuid4().hex[:6]}"

        user_id = os.environ.get("THYRA_USER_ID", "default")
        agent_id = resolve_project_id(cwd)

        # CCD: when cwd="" the hook receives no path, so resolve_project_id falls
        # back to "global".  First try the session_id — each transcript lives at
        # ~/.claude/projects/<proj-dir>/<session_id>.jsonl, so this lookup is
        # unambiguous and immune to stale context from a previous project.
        if agent_id in ("global", "unknown"):
            if session_id not in ("unknown", ""):
                try:
                    from thyra.server.tools.admin_tools import (
                        _agent_id_from_project_dir_name,
                    )

                    _projects_dir = pathlib.Path.home() / ".claude" / "projects"
                    for _pdir in _projects_dir.iterdir():
                        if _pdir.is_dir() and (_pdir / f"{session_id}.jsonl").exists():
                            _recovered = _agent_id_from_project_dir_name(_pdir.name)
                            if _recovered and _recovered not in ("global", "unknown"):
                                agent_id = _recovered
                                break
                except Exception:
                    pass

        # Fallback: read the last-written project context file.  This can be
        # stale when the previous session was in a different project, so it is
        # only used when the session_id lookup above found nothing.
        if agent_id in ("global", "unknown"):
            _ctx_p = os.path.join(
                os.environ.get("TEMP", tempfile.gettempdir()),
                "thyra_ctx_latest.json",
            )
            try:
                with open(_ctx_p, encoding="utf-8") as _f:
                    _saved = json.load(_f).get("project_id", "")
                if _saved and _saved not in ("global", "unknown", ""):
                    agent_id = _saved
            except Exception:
                pass

        # CCD: flush formation for the previous turn before overwriting context.
        _maybe_flush_prev_turn()

        _write_context(session_id, agent_id, cwd)
        _write_turn_state(session_id, turn_id)

        # Lazy import keeps startup fast when there's nothing to recall
        from thyra.db.connection import get_conn
        from thyra.recall.intent import recall_pipeline

        conn = get_conn()
        xml, served_ids = recall_pipeline(
            conn, user_id, agent_id, prompt, session_id, turn_id
        )

        # CCD fallback: if project could not be resolved (agent_id stayed "global"),
        # inject a mandatory reminder so Claude always calls thyra_init_session —
        # otherwise additionalContext is "" and there is zero signal to trigger it.
        if agent_id in ("global", "unknown"):
            ctx_cwd = cwd
            if not ctx_cwd:
                try:
                    _ctx_p = os.path.join(
                        os.environ.get("TEMP", tempfile.gettempdir()),
                        "thyra_ctx_latest.json",
                    )
                    with open(_ctx_p, encoding="utf-8") as _f:
                        ctx_cwd = json.load(_f).get("cwd", "")
                except Exception:
                    pass
            cwd_hint = (
                f"cwd='{ctx_cwd}'" if ctx_cwd else "cwd='<project-working-directory>'"
            )
            init_reminder = (
                f"[THYRA-INIT-REQUIRED] CCD mode: project scope not resolved. "
                f"Your FIRST action MUST be thyra_init_session({cwd_hint}) "
                f"before answering the user — do not skip this."
            )
            xml = (init_reminder + "\n" + xml) if xml else init_reminder

        output = {"additionalContext": xml}

    except Exception:
        output = {"additionalContext": ""}

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
