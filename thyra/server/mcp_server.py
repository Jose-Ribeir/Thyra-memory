"""FastMCP server entry point.

Starts the BackgroundWorker consolidation thread at import time, then
serves MCP tools over stdio transport.

Memory injection is fully automatic:
  - The UserPromptSubmit hook (pre_turn.py) runs before every message, scores
    memories for the current project, and injects them as additionalContext.
  - A background transcript monitor thread polls the Claude transcript file
    every 10 seconds and queues a delta event whenever a new completed turn
    is detected, so memory formation is also fully automatic.

Note: we do NOT inject memories into mcp.instructions at startup.  The MCP
server is often launched from C:\\Windows\\System32 (CCD), so the CWD is
unreliable and any startup injection would use a stale/wrong project context.
All recall goes through the per-turn hook which resolves the project from the
session_id → transcript path lookup.
"""

from __future__ import annotations

# HuggingFace cache — must be set BEFORE any other imports that read it.
# sentence_transformers reads HF_HOME when the module is first imported, so
# _ensure_hf_cache() inside _get_model_and_embeddings() fires too late.
# H:\HuggingFace is the designated persistent cache location.
# HF_HUB_OFFLINE=1 prevents the online version-check that hangs when the
# network is slow — the model is already cached locally.
import os as _os

_os.environ["HF_HOME"] = r"H:\HuggingFace"
_os.environ.setdefault("HF_HUB_OFFLINE", "1")

import threading

from fastmcp import FastMCP

_BASE_INSTRUCTIONS = (
    "Thyra Adaptive Long-Memory. Provides persistent, time-decaying memory "
    "for Claude Code sessions. Use thyra_save to explicitly persist a fact. "
    "The system forms memories automatically from conversation.\n\n"
    "At the end of every response include ONE of:\n"
    "  <memories_used>m_id1,m_id2</memories_used>  — IDs of memories you relied on\n"
    "  <memories_used></memories_used>              — if you used none\n"
    "This tag is parsed by the formation pipeline for reinforcement.\n\n"
    "CCD (Claude Desktop App): the Stop hook does NOT fire in CCD.\n"
    "1. At the START of every new conversation call:\n"
    "     thyra_init_session(cwd='<project working directory>')\n"
    "   You can read the CWD from the path of your CLAUDE.md shown in your "
    "system context (e.g. 'Contents of J:\\\\codigo\\\\thyra-ai\\\\CLAUDE.md' "
    "means cwd='J:\\\\codigo\\\\thyra-ai'). This scopes all memories to the "
    "correct project and returns the relevant memories — treat the returned "
    "memories_xml exactly like an injected <thyra_memories> block.\n"
    "2. At the END of every response call thyra_end_turn(memories_used='...')."
)

mcp = FastMCP(
    name="thyra-memory",
    instructions=_BASE_INSTRUCTIONS,
)

# Register all tools
from thyra.server.tools.recall_tools import register_recall_tools  # noqa: E402
from thyra.server.tools.admin_tools import register_admin_tools  # noqa: E402

register_recall_tools(mcp)
register_admin_tools(mcp)


# ── helpers ────────────────────────────────────────────────────────────────────


def _start_worker() -> None:
    try:
        import logging

        # Pre-import AND pre-call every module the worker thread needs, BEFORE
        # spawning any background threads.  sentence_transformers (loaded in the
        # prewarm sub-thread) can hold Python's per-module import lock for 20+
        # seconds.  If the worker thread tries to import the same dependency (e.g.
        # NLTK's PorterStemmer, which is imported lazily on first normalize_cue call)
        # while the prewarm sub-thread holds that import lock, the worker blocks
        # indefinitely.  Calling the functions here (main thread, before any threads
        # start) forces Python's import machinery to fully resolve every lazy import
        # they trigger — so the import locks are released before threads start.
        try:
            from thyra.recall.morphology import normalize_cue as _nc

            _nc("warmup")  # forces NLTK PorterStemmer import + stemmer init
        except Exception as _pre_err:
            logging.getLogger("thyra").debug("Pre-import normalize_cue: %s", _pre_err)
        try:
            from thyra.recall.cue_extractor import extract_raw_cues as _ec

            _ec(
                "warmup test text for thyra initialization"
            )  # forces cue extraction init
        except Exception as _pre_err:
            logging.getLogger("thyra").debug(
                "Pre-import extract_raw_cues: %s", _pre_err
            )
        try:
            from thyra.formation.keyphrase import extract_keyphrases as _kp

            _kp("")  # short-circuits for empty string but triggers module init
        except Exception as _pre_err:
            logging.getLogger("thyra").debug(
                "Pre-import extract_keyphrases: %s", _pre_err
            )

        from thyra.consolidation.worker import BackgroundWorker

        worker = BackgroundWorker()

        # Pre-warm the sentence-transformer model before the worker processes
        # its first event.  We use a threading.Event so the worker can wait
        # until prewarm finishes (or times out) rather than racing with it.
        model_ready = threading.Event()

        def _prewarm() -> None:
            try:
                from thyra.formation.refiner import _get_model_and_embeddings

                _get_model_and_embeddings()
                # L2 distiller: prewarm here (outside the worker thread) so the
                # worker can use it via the rules-only fast path without ever
                # blocking on a cold load.  No-op unless DISTILLER_ENABLED.
                from thyra.formation.distiller import prewarm as _distiller_prewarm

                _distiller_prewarm()
            except Exception:
                pass
            finally:
                model_ready.set()

        threading.Thread(
            target=_prewarm, daemon=True, name="thyra-model-prewarm"
        ).start()

        # Patch the worker's run() to wait for prewarm on its first iteration.
        _orig_run = worker.run

        def _run_with_prewarm() -> None:
            model_ready.wait(timeout=20)  # wait up to 20s for model to load
            _orig_run()

        t = threading.Thread(
            target=_run_with_prewarm, daemon=True, name="thyra-consolidation"
        )
        t.start()
    except Exception as e:
        import logging

        logging.getLogger("thyra").warning("BackgroundWorker failed to start: %s", e)


def _start_tray() -> None:
    try:
        from thyra.tray import _start_tray as _tray

        _tray()
    except Exception as e:
        import logging

        logging.getLogger("thyra").debug("Tray icon failed to start: %s", e)


def _start_dashboard() -> None:
    try:
        from thyra.config import (
            DASHBOARD_AUTO_OPEN,
            DASHBOARD_ENABLED,
            DASHBOARD_HOST,
            DASHBOARD_PORT,
        )

        if not DASHBOARD_ENABLED:
            return

        from thyra.dashboard.runner import start_dashboard

        _thread, _slot = start_dashboard(host=DASHBOARD_HOST, port=DASHBOARD_PORT)

        if DASHBOARD_AUTO_OPEN:
            url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"

            def _open_browser() -> None:
                """Wait for uvicorn to be ready, then open the browser.

                Uses a subprocess (Serena-style) so that any output from the
                browser launcher can never corrupt the MCP stdio pipe.
                """
                import subprocess
                import sys
                import time

                # Poll until uvicorn signals it is ready (up to 10 s).
                # The for…else pattern: the else block only runs when the loop
                # exits WITHOUT a break, i.e. uvicorn never became ready
                # (port already in use from a previous process).  In that case
                # a browser is already open — don't open a second one.
                for _ in range(100):
                    if _slot and getattr(_slot[0], "started", False):
                        break
                    time.sleep(0.1)
                else:
                    return  # timeout — uvicorn didn't start, skip browser open

                try:
                    subprocess.Popen(  # noqa: S603
                        [
                            sys.executable,
                            "-c",
                            f"import webbrowser; webbrowser.open({url!r})",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        close_fds=True,
                    )
                except Exception as exc:
                    import logging

                    logging.getLogger("thyra").debug("Could not open browser: %s", exc)

            threading.Thread(
                target=_open_browser, daemon=True, name="thyra-browser-open"
            ).start()
    except Exception as e:
        import logging

        logging.getLogger("thyra").warning("Dashboard failed to start: %s", e)


def _is_system_cwd(cwd: str) -> bool:
    """Return True if *cwd* is a Windows system directory, not a user project."""
    import os

    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    low = cwd.lower()
    return low.startswith(system_root.lower()) or low.startswith(r"c:\program files")


def _resolve_startup_agent_id() -> tuple[str, str]:
    """Return (agent_id, cwd) for startup, skipping system-directory CWDs.

    When Claude Desktop App launches the MCP server its working directory is
    often ``C:\\Windows\\System32``, which is not a user project.  In that
    case we fall back to the project_id recorded in the last context file so
    that memories are loaded and queued for the real active project.
    """
    import json
    import os
    import tempfile

    from thyra.hooks._project import resolve_project_id

    cwd = os.getcwd()
    if not _is_system_cwd(cwd):
        return resolve_project_id(cwd), cwd

    # System CWD — read the last known good context file
    ctx_path = os.path.join(
        os.environ.get("TEMP", tempfile.gettempdir()),
        "thyra_ctx_latest.json",
    )
    try:
        with open(ctx_path, encoding="utf-8") as _f:
            ctx = json.load(_f)
        pid = ctx.get("project_id", "")
        saved_cwd = ctx.get("cwd", "")
        # Accept any project_id that looks like a real project slug
        if pid and pid not in ("global", "system32", "windows", "programfiles"):
            return pid, saved_cwd
    except Exception:
        pass

    return os.environ.get("THYRA_AGENT_ID", "global"), ""


def _init_context() -> None:
    """Write project context from CWD at server startup.

    Skips writing when the MCP server was launched from a Windows system
    directory (C:\\Windows\\System32, etc.) AND a specific (non-global) project
    context already exists from a prior session — we don't want to overwrite
    a valid project context with a system-directory slug.

    When the server IS in a real project directory, always writes the correct
    project_id so subsequent calls to get_current_project_id() don't fall
    through to "global".
    """
    try:
        import os
        import time

        from thyra.hooks._project import resolve_project_id
        from thyra.server.tools._context import write_context

        cwd = os.getcwd()
        if _is_system_cwd(cwd):
            import logging

            logging.getLogger("thyra").debug(
                "Skipping context write for system CWD: %s", cwd
            )
            return

        project_id = resolve_project_id(cwd)
        # Only skip writing if the resolved project is "global" / "unknown"
        # AND a better value already exists in the file (startup from a non-project
        # directory like the user's home folder).
        if project_id in ("global", "unknown"):
            import json, tempfile

            _ctx_path = os.path.join(
                os.environ.get("TEMP", tempfile.gettempdir()), "thyra_ctx_latest.json"
            )
            if os.path.exists(_ctx_path):
                try:
                    with open(_ctx_path) as _f:
                        prev = json.load(_f)
                    if prev.get("project_id", "") not in ("global", "unknown", ""):
                        return  # keep the existing specific project
                except Exception:
                    pass
        write_context(
            session_id=f"mcp-startup:{int(time.time() * 1000)}",
            project_id=project_id,
            cwd=cwd,
        )
    except Exception as e:
        import logging

        logging.getLogger("thyra").debug("Could not write startup context: %s", e)


def _start_transcript_monitor(initial_agent_id: str) -> None:
    """Start a daemon thread that auto-queues delta events from the transcript.

    Every 10 seconds the thread:
      1. Re-reads thyra_ctx_latest.json for the current project + transcript path.
      2. Checks if the transcript file has been modified since last check.
      3. Extracts the last completed (assistant, user) turn.
      4. If it's a new turn (content hash changed), writes a delta event to
         delta_queue/ so the BackgroundWorker can run the formation pipeline.

    This makes thyra_end_turn calls unnecessary — formation happens automatically
    within ~10 seconds after Claude finishes each response.
    """

    def _run() -> None:
        import hashlib
        import json
        import os
        import tempfile
        import time
        import uuid

        from thyra.config import THYRA_DB_PATH, THYRA_USER_ID as U
        from thyra.hooks.stop_hook import (
            _detect_correction,
            _enqueue_delta,
            _extract_last_messages,
            _parse_declared,
        )
        from thyra.server.tools.admin_tools import (
            _agent_id_from_project_dir_name,
            _find_transcript,
        )

        last_mtime: float = 0.0
        last_hash: str = ""
        last_transcript: str = ""
        last_transcript_agent_id: str = ""  # agent_id resolved from transcript path
        _SETTLE_SECS = 8  # wait this many seconds after last write before queuing

        while True:
            time.sleep(10)
            try:
                # Re-read context on every iteration so project changes are picked up
                ctx_path = os.path.join(
                    os.environ.get("TEMP", tempfile.gettempdir()),
                    "thyra_ctx_latest.json",
                )
                ctx: dict = {}
                if os.path.exists(ctx_path):
                    with open(ctx_path, encoding="utf-8") as _f:
                        ctx = json.load(_f)

                cwd = ctx.get("cwd", "")
                raw_pid = ctx.get("project_id", "")
                # Fall back to initial_agent_id when the context file holds a
                # system-path slug (e.g. "system32") that isn't a real project.
                if raw_pid and raw_pid not in (
                    "global",
                    "system32",
                    "windows",
                    "programfiles",
                ):
                    ctx_agent_id = raw_pid
                else:
                    ctx_agent_id = initial_agent_id
                session_id = ctx.get("session_id", f"auto:{int(time.time() * 1000)}")

                transcript = _find_transcript(cwd)
                if not transcript or not os.path.exists(transcript):
                    continue

                # Reset state when transcript switches (new session starts).
                # Recompute agent_id from the transcript path so memories are always
                # tagged to the project that OWNS the transcript, not to whatever
                # stale cwd the context file happens to hold.  The context file can
                # lag by a whole session if the UserPromptSubmit hook didn't fire.
                if transcript != last_transcript:
                    last_transcript = transcript
                    last_mtime = 0.0
                    last_hash = ""
                    proj_dir_name = os.path.basename(os.path.dirname(transcript))
                    resolved = _agent_id_from_project_dir_name(proj_dir_name)
                    last_transcript_agent_id = resolved if resolved else ctx_agent_id

                agent_id = (
                    last_transcript_agent_id
                    if last_transcript_agent_id
                    else ctx_agent_id
                )

                mtime = os.path.getmtime(transcript)
                if mtime <= last_mtime:
                    continue  # nothing new written since last check

                # Settle-time guard: if the transcript was written very recently
                # Claude may still be mid-turn (writing tool results).  Wait
                # until it has been quiet for at least _SETTLE_SECS seconds to
                # avoid queueing partial turns or duplicate events.
                if time.time() - mtime < _SETTLE_SECS:
                    continue

                last_mtime = mtime

                assistant_text, user_text = _extract_last_messages(transcript)
                if not assistant_text:
                    continue

                # Deduplicate — only queue when the assistant text is genuinely new
                h = hashlib.md5(assistant_text[:500].encode()).hexdigest()
                if h == last_hash:
                    continue
                last_hash = h

                declared = _parse_declared(assistant_text)
                now_ms = int(time.time() * 1000)

                # Load served_ids from the turn state file so the
                # anti-spoofing check in apply_reinforcement can work:
                #   valid_ids = declared_set & served_set
                # Without served_ids the intersection is always empty and
                # <memories_used> never reinforces anything.
                served_ids: list = []
                cues_fired_state: list = []
                turn_id_state = f"auto-monitor:{uuid.uuid4().hex[:8]}:{now_ms}"
                # Use the canonical data-dir path written by intent.py._store_turn_state,
                # NOT a TEMP path — they resolve to different locations in CCD mode and
                # the TEMP path is never written, leaving served_ids/cues_fired always [].
                from thyra.recall.intent import _turn_state_path as _tsp

                _state_path = _tsp(session_id)
                if os.path.exists(_state_path):
                    try:
                        with open(_state_path, encoding="utf-8") as _sf:
                            _state = json.load(_sf)
                        served_ids = _state.get("served_ids", [])
                        cues_fired_state = _state.get("cues_fired", [])
                        turn_id_state = _state.get("turn_id", turn_id_state)
                        try:
                            os.unlink(_state_path)
                        except Exception:
                            pass
                    except Exception:
                        pass

                delta = {
                    "session_id": session_id,
                    "turn_id": turn_id_state,
                    "user_id": U,
                    "agent_id": agent_id,
                    "timestamp": now_ms,
                    "memories_served": served_ids,
                    "memories_declared": declared,
                    "cues_fired": cues_fired_state,
                    "raw_user_text": user_text,
                    "raw_assistant_text": assistant_text,
                    "correction_flag": _detect_correction(user_text),
                }
                _enqueue_delta(delta, THYRA_DB_PATH)

            except Exception:
                pass  # monitor thread must never crash

    t = threading.Thread(target=_run, daemon=True, name="thyra-transcript-monitor")
    t.start()


if __name__ == "__main__":
    # 1. Write startup context (skips system-directory CWDs automatically)
    _init_context()

    # 2. Resolve agent_id for the transcript monitor.
    #    _resolve_startup_agent_id() falls back to the last known good project
    #    when the MCP server was launched from a system directory.
    try:
        _agent_id, _agent_cwd = _resolve_startup_agent_id()
    except Exception:
        _agent_id = "claude-code-global"

    # 3. Start background workers
    #    Note: memory recall is handled entirely by the UserPromptSubmit hook
    #    (pre_turn.py) which runs per-turn with the correct session-scoped
    #    agent_id.  We do NOT inject memories into mcp.instructions at startup
    #    because the server CWD is unreliable (often C:\Windows\System32 in CCD)
    #    and any startup injection would use a stale/wrong project context.
    _start_worker()
    _start_transcript_monitor(_agent_id)
    _start_dashboard()
    _start_tray()

    # 4. Run MCP server — this blocks until Claude closes the connection
    mcp.run(transport="stdio")
