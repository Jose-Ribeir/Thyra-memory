"""Start the dashboard HTTP server in a daemon thread."""

from __future__ import annotations

import atexit
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uvicorn as _uvicorn

log = logging.getLogger("thyra.dashboard")

# Module-level reference so atexit can reach the server.
_server: _uvicorn.Server | None = None


def start_dashboard(
    host: str, port: int
) -> tuple[threading.Thread, "_uvicorn.Server | None"]:
    """Launch uvicorn in a daemon thread.  Returns (thread, server).

    The server object is also stored module-globally so the atexit handler can
    signal it to stop — this frees the port cleanly instead of waiting for the
    OS to reclaim it when the process is killed.

    stdout / access_log are suppressed because the MCP server owns stdio.
    """
    global _server

    # Shared slot: _run() fills this once uvicorn is constructed.
    _slot: list[_uvicorn.Server] = []

    def _run() -> None:
        global _server
        try:
            import uvicorn

            from thyra.dashboard import log_handler
            from thyra.dashboard.server import app

            log_handler.install()

            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
                lifespan="off",
            )
            server = uvicorn.Server(config)
            _server = server
            _slot.append(server)
            log.info("Dashboard starting on http://%s:%d", host, port)
            server.run()
        except Exception as exc:
            log.warning("Dashboard failed to start: %s", exc)

    t = threading.Thread(target=_run, daemon=True, name="thyra-dashboard")
    t.start()

    # Register graceful shutdown so the port is freed before the process dies.
    atexit.register(_shutdown_server)

    # Return the slot list — caller can poll _slot[0] once uvicorn fills it.
    return t, _slot


def _shutdown_server() -> None:
    """Tell uvicorn to stop accepting connections.  Called by atexit."""
    global _server
    if _server is not None:
        try:
            _server.should_exit = True
        except Exception:
            pass
