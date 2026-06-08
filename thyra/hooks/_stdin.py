"""Stdin helpers for Claude Code hook scripts (Windows-safe)."""

from __future__ import annotations

import sys
import threading

STDIN_TIMEOUT_SECONDS = 0.5


def read_stdin_text(timeout_seconds: float = STDIN_TIMEOUT_SECONDS) -> str:
    """Read hook stdin with a hard timeout so a missing pipe cannot hang CC.

    Reads from the binary buffer and decodes as utf-8-sig so that the UTF-8
    BOM that PowerShell 5.1 prepends is stripped regardless of the platform's
    default text encoding (cp1252 on Windows).
    """
    result: list[str] = []

    def _reader() -> None:
        try:
            raw_bytes = sys.stdin.buffer.read()
            result.append(raw_bytes.decode("utf-8-sig"))
        except Exception:
            result.append("")

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    return result[0] if result else ""
