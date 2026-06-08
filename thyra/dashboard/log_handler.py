"""In-memory ring-buffer log handler for the dashboard live log view."""

from __future__ import annotations

import collections
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

from thyra.config import DASHBOARD_LOG_BUFFER_SIZE


@dataclass
class LogEntry:
    ts: float
    level: str
    logger: str
    message: str


class DashboardLogHandler(logging.Handler):
    """Thread-safe ring buffer attached to the 'thyra' logger."""

    def __init__(self, maxlen: int = DASHBOARD_LOG_BUFFER_SIZE) -> None:
        super().__init__()
        self._buf: collections.deque[LogEntry] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = LogEntry(
                ts=record.created,
                level=record.levelname,
                logger=record.name,
                message=self.format(record),
            )
            with self._lock:
                self._buf.append(entry)
        except Exception:
            self.handleError(record)

    def recent(self, since_ts: float = 0.0, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            entries = [asdict(e) for e in self._buf if e.ts > since_ts]
        return entries[-limit:]


_handler: DashboardLogHandler | None = None


def install() -> DashboardLogHandler:
    """Attach handler to the 'thyra' root logger. Idempotent."""
    global _handler
    if _handler is not None:
        return _handler
    _handler = DashboardLogHandler()
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(fmt)
    root = logging.getLogger("thyra")
    root.addHandler(_handler)
    root.setLevel(logging.DEBUG)
    return _handler


def get_handler() -> DashboardLogHandler | None:
    return _handler
