"""Stop hook: parse memories_used block from response, write delta event to queue.

Claude Code pipes a JSON event to stdin after each response. This script
queues a DeltaEvent file for the background consolidation worker. It must
NEVER exit non-zero or raise.
"""

import json
import os
import pathlib
import re
import sys
import tempfile
import time
import uuid

from thyra.hooks._project import resolve_project_id
from thyra.hooks._stdin import read_stdin_text

MEMORIES_USED_RE = re.compile(
    r"<memories_used>\s*(.*?)\s*</memories_used>",
    re.DOTALL | re.IGNORECASE,
)


def main() -> None:
    try:
        raw = read_stdin_text()
        event = json.loads(raw) if raw.strip() else {}

        session_id = event.get("session_id", "unknown")
        transcript_path = event.get("transcript_path", "")
        cwd = event.get("cwd", "")
        user_id = os.environ.get("THYRA_USER_ID", "default")

        # CCD fallback: Desktop App doesn't populate cwd or transcript_path in the
        # stop-hook event. Recover both from the context file written by
        # thyra_init_session (or by a successful UserPromptSubmit run).
        if not cwd:
            try:
                ctx_path = os.path.join(
                    os.environ.get("TEMP", tempfile.gettempdir()),
                    "thyra_ctx_latest.json",
                )
                with open(ctx_path, encoding="utf-8") as _cf:
                    _ctx = json.load(_cf)
                cwd = _ctx.get("cwd", "")
            except Exception:
                pass

        agent_id = resolve_project_id(cwd)
        from thyra.config import THYRA_DB_PATH as _default_db_path

        db_path = os.environ.get("THYRA_DB_PATH", _default_db_path)

        # Diagnostic: capture raw event for debugging (written to a known temp path).
        try:
            _diag = {"raw": raw[:500], "event": event, "cwd_recovered": cwd}
            _diag_path = os.path.join(
                os.environ.get("TEMP", tempfile.gettempdir()),
                "thyra_last_stop_event.json",
            )
            with open(_diag_path, "w", encoding="utf-8") as _df:
                json.dump(_diag, _df)
        except Exception:
            pass

        # CCD fallback: transcript_path not sent — find the .jsonl by session_id.
        # Also handles the "unknown" session_id case by finding the most recently
        # modified .jsonl in the project-matching directory.
        if not transcript_path:
            try:
                import pathlib as _pl

                _projects = _pl.Path(os.path.expanduser("~/.claude/projects"))
                # Pass 1: exact match by session_id (when CCD sends a UUID)
                if session_id and session_id != "unknown":
                    for _pdir in _projects.iterdir():
                        if _pdir.is_dir():
                            _candidate = _pdir / f"{session_id}.jsonl"
                            if _candidate.exists():
                                transcript_path = str(_candidate)
                                break
                # Pass 2: most-recently-modified .jsonl in the project dir
                # (works even when session_id is "unknown")
                if not transcript_path and agent_id and agent_id != "global":
                    _slug = agent_id.lower()
                    for _pdir in _projects.iterdir():
                        if _pdir.is_dir() and _slug in _pdir.name.lower():
                            _recent = sorted(
                                _pdir.glob("*.jsonl"),
                                key=lambda _f: _f.stat().st_mtime,
                                reverse=True,
                            )
                            if _recent:
                                transcript_path = str(_recent[0])
                                break
            except Exception:
                pass

        assistant_text, user_text = _extract_last_messages(transcript_path)
        tool_activity = _extract_tool_activity(transcript_path)
        declared_ids = _parse_declared(assistant_text)
        served_ids, turn_id, cues_fired = _load_turn_state(session_id)

        delta = {
            "session_id": session_id,
            "turn_id": turn_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "timestamp": int(time.time() * 1000),
            "memories_served": served_ids,
            "memories_declared": declared_ids,
            "cues_fired": cues_fired,
            "raw_user_text": user_text,
            "raw_assistant_text": assistant_text,
            "tool_activity": tool_activity,
            "correction_flag": _detect_correction(user_text),
        }
        _enqueue_delta(delta, db_path)

    except Exception:
        pass  # stop hook must never surface errors

    sys.exit(0)


def _content_to_text(content) -> str:
    if isinstance(content, list):
        # Prefer text blocks — these are the user's actual speech.
        # Tool-result blocks travel as user-role messages but are not user speech;
        # mixing them in causes tool output to be treated as user text for formation.
        # Only fall back to tool_result content when there are no text blocks at all.
        text_parts = [
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        if text_parts:
            return " ".join(p for p in text_parts if p)
        tool_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content", "")
            if isinstance(inner, list):
                tool_parts.extend(
                    str(c.get("text", ""))
                    for c in inner
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            else:
                tool_parts.append(str(inner))
        return " ".join(p for p in tool_parts if p)
    return str(content or "")


def _normalize_transcript_record(record: dict) -> tuple[str, str] | None:
    """Return (role, text) for a Claude Code transcript line, if it is a chat turn."""
    if not isinstance(record, dict):
        return None

    msg = record.get("message")
    if isinstance(msg, dict) and msg.get("role") in {"user", "assistant"}:
        return msg["role"], _content_to_text(msg.get("content", ""))

    role = record.get("role")
    if role in {"user", "assistant"}:
        return role, _content_to_text(record.get("content", ""))

    return None


def _load_transcript_messages(transcript_path: str) -> list[tuple[str, str]]:
    """Load chat turns from Claude Code .jsonl (or legacy JSON array) transcripts."""
    messages: list[tuple[str, str]] = []
    with open(transcript_path, encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return messages

    # Claude Code stores one JSON object per line (.jsonl).
    if "\n" in raw:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = _normalize_transcript_record(record)
            if normalized:
                messages.append(normalized)
        return messages

    # Legacy smoke-test format: a single JSON array/object.
    data = json.loads(raw)
    if isinstance(data, list):
        source = data
    else:
        source = data.get("messages", []) if isinstance(data, dict) else []
    for item in source:
        normalized = _normalize_transcript_record(
            item if isinstance(item, dict) else {}
        )
        if normalized:
            messages.append(normalized)
    return messages


def _extract_last_messages(transcript_path: str) -> tuple[str, str]:
    """Return (last_assistant_text, last_user_question) from transcript.

    Skips tool-result injections when searching for the user's actual message.
    In tool-heavy sessions the last 'user' entry in the JSONL is a tool result,
    not the human's question — so we walk back until we find a plain-text user
    message or exhaust the recent window.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return ("", "")
    try:
        messages = _load_transcript_messages(transcript_path)
        assistant_text = ""
        user_text = ""
        # Walk backwards through the last 40 messages to stay fast on large logs
        window = messages[-40:] if len(messages) > 40 else messages
        for role, text in reversed(window):
            if role == "assistant" and not assistant_text:
                # Skip assistant entries that are pure tool-use listings
                if text.strip():
                    assistant_text = text
            elif role == "user" and not user_text:
                # Skip entries that look like tool-result injections:
                # they start with known tool-output patterns or are very long
                # structured blobs. Prefer the shorter, natural-language entry.
                stripped = text.strip()
                if stripped and not _is_tool_result_text(stripped):
                    user_text = text
            if assistant_text and user_text:
                break
        return (assistant_text, user_text)
    except Exception:
        return ("", "")


def _is_tool_result_text(text: str) -> bool:
    """Heuristic: return True if this 'user' entry is likely a tool result injection.

    Tool results tend to be JSON blobs, long stack traces, command output, or
    start with common shell/code output markers. Human questions are short and
    in natural language.
    """
    # Very long entries are almost certainly tool output
    if len(text) > 2000:
        return True
    # Starts with common tool-output markers
    tool_markers = (
        "{",
        "[",
        "Error:",
        "Traceback",
        "stdout:",
        "stderr:",
        "OK\n",
        "Files before:",
        "Queue",
        "Total lines:",
        "===",
        "Killed ",
        "Sending ",
        "Running ",
        "Starting ",
        "Exit code",
    )
    for marker in tool_markers:
        if text.startswith(marker):
            return True
    return False


def _extract_tool_activity(transcript_path: str, max_chars: int = 4000) -> str:
    """Concatenate this turn's tool-use names/inputs and tool-result text.

    Used as corroboration evidence for reinforcement: a declared memory that
    shaped a tool call (search query, file path, command) leaves a trace here
    even when it never appears in the assistant's prose.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return ""
        lines_list = raw.splitlines() if chr(10) in raw else [raw]
        parts: list[str] = []
        # Only the recent window -- the current turn's activity lives at the tail.
        for line in lines_list[-40:]:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            msg = record.get("message")
            content_val = (
                msg.get("content") if isinstance(msg, dict) else record.get("content")
            )
            if not isinstance(content_val, list):
                continue
            for block in content_val:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    parts.append(str(block.get("name", "")))
                    inp = block.get("input")
                    if isinstance(inp, dict):
                        parts.append(" ".join(str(v) for v in inp.values())[:1000])
                    elif inp:
                        parts.append(str(inp)[:1000])
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        parts.extend(
                            str(c.get("text", ""))[:1000]
                            for c in inner
                            if isinstance(c, dict) and c.get("type") == "text"
                        )
                    elif inner:
                        parts.append(str(inner)[:1000])
        return " ".join(p for p in parts if p)[:max_chars]
    except Exception:
        return ""


def _parse_declared(assistant_text: str) -> list[str]:
    m = MEMORIES_USED_RE.search(assistant_text)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip().startswith("m_")]


def _turn_state_path(session_id: str) -> str:
    """Canonical path for a turn's state file — data dir, not TEMP.

    Mirrors thyra.recall.intent._turn_state_path exactly. Duplicated here so
    the hook scripts (which run as standalone subprocesses) don't need to import
    the full thyra package just to resolve a path.
    """
    import pathlib
    from thyra.config import THYRA_DB_PATH

    safe_sid = (session_id or "unknown").replace("/", "_").replace("\\", "_")
    data_dir = pathlib.Path(THYRA_DB_PATH).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / f"turn_state_{safe_sid}.json")


def _load_turn_state(session_id: str) -> tuple[list[str], str, list[str]]:
    try:
        state_path = _turn_state_path(session_id)
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            try:
                os.unlink(state_path)
            except Exception:
                pass
            return (
                state.get("served_ids", []),
                state.get("turn_id", ""),
                state.get("cues_fired", []),
            )
    except Exception:
        pass
    return ([], f"unknown:{int(time.time() * 1000)}", [])


def _detect_correction(user_text: str) -> bool:
    if not user_text:
        return False
    correction_markers = ("actually", "no,", "that's wrong", "i meant", "correction")
    lower = user_text.lower()
    return any(m in lower for m in correction_markers)


def _enqueue_delta(data: dict, db_path: str) -> None:
    queue_dir = pathlib.Path(db_path).parent / "delta_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    ts = data["timestamp"]
    suffix = uuid.uuid4().hex[:8]
    fname = queue_dir / f"{ts}_{suffix}.json"
    tmp = fname.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.rename(fname)


if __name__ == "__main__":
    main()
