"""Windows system-tray icon for the Thyra MCP server.

Provides pause/resume, auto-formation toggle, max-memories submenu,
open-dashboard, and quit-server — all without touching stdout/stderr
so the MCP stdio pipe is never corrupted.

Guard: if pystray or Pillow are unavailable, _start_tray() returns
silently so the MCP server still works.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("thyra.tray")


def _make_icon_image():
    """Generate a small 64×64 PIL image with a 'T' glyph on a dark background."""
    from PIL import Image, ImageDraw, ImageFont

    size = 64
    img = Image.new("RGBA", (size, size), (30, 30, 40, 255))
    draw = ImageDraw.Draw(img)
    # Draw a simple "T" — don't rely on a system font being present.
    # Manual pixel-art T at scale: top bar and vertical bar.
    margin = 10
    bar_h = 10
    stem_w = 10
    # Top horizontal bar
    draw.rectangle(
        [margin, margin, size - margin, margin + bar_h], fill=(120, 180, 255, 255)
    )
    # Vertical stem
    cx = size // 2
    draw.rectangle(
        [cx - stem_w // 2, margin, cx + stem_w // 2, size - margin],
        fill=(120, 180, 255, 255),
    )
    return img


def _start_tray() -> None:
    """Start the tray icon in a daemon thread.  Safe to call from the main thread."""
    from thyra.config import THYRA_TRAY_ENABLED

    if not THYRA_TRAY_ENABLED:
        return

    try:
        import pystray
    except ImportError:
        log.debug("pystray not installed — tray icon disabled")
        return
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        log.debug("Pillow not installed — tray icon disabled")
        return

    def _run_tray() -> None:
        try:
            _run_tray_inner()
        except Exception as exc:
            log.debug("Tray icon error: %s", exc)

    t = threading.Thread(target=_run_tray, daemon=True, name="thyra-tray")
    t.start()


def _run_tray_inner() -> None:
    import os
    import webbrowser

    import pystray

    from thyra.config import (
        DASHBOARD_HOST,
        DASHBOARD_PORT,
        THYRA_USER_ID,
        THYRA_AGENT_ID,
    )
    from thyra.db.connection import get_conn
    from thyra.models.memory import get_flag, set_flag
    from thyra.recall.cache import HOT_CACHE

    def _get(flag: str, default: str = "true") -> str:
        try:
            return get_flag(get_conn(), flag, THYRA_USER_ID, THYRA_AGENT_ID, default)
        except Exception:
            return default

    def _set(flag: str, value: str) -> None:
        try:
            conn = get_conn()
            set_flag(conn, flag, value, THYRA_USER_ID, THYRA_AGENT_ID)
            HOT_CACHE.invalidate(f"snapshot:{THYRA_USER_ID}:{THYRA_AGENT_ID}")
        except Exception as exc:
            log.debug("Tray set_flag error: %s", exc)

    def _checked(flag: str, default: str = "true") -> bool:
        return _get(flag, default).lower() == "true"

    # ── Menu item callbacks ────────────────────────────────────────────────────

    def on_toggle_system(icon, item) -> None:
        new_val = "false" if _checked("system_enabled") else "true"
        _set("system_enabled", new_val)
        icon.update_menu()

    def on_toggle_formation(icon, item) -> None:
        new_val = "false" if _checked("formation_enabled") else "true"
        _set("formation_enabled", new_val)
        icon.update_menu()

    def _set_max(n: int):
        def handler(icon, item) -> None:
            _set("max_memories", str(n))
            icon.update_menu()

        return handler

    def on_open_dashboard(icon, item) -> None:
        url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
        try:
            # Spawn subprocess so browser never writes to our stdio pipe.
            import subprocess, sys

            subprocess.Popen(
                [sys.executable, "-c", f"import webbrowser; webbrowser.open({url!r})"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
        except Exception as exc:
            log.debug("Tray open dashboard error: %s", exc)

    def on_quit(icon, item) -> None:
        _set("system_enabled", "false")
        icon.stop()
        try:
            from thyra.dashboard.runner import _shutdown_server

            _shutdown_server()
        except Exception:
            pass
        os._exit(0)

    # ── Dynamic checked state ──────────────────────────────────────────────────

    def _system_checked(item) -> bool:
        return _checked("system_enabled")

    def _formation_checked(item) -> bool:
        return _checked("formation_enabled")

    def _max_checked(n: int):
        def check(item) -> bool:
            return _get("max_memories", "0") == str(n)

        return check

    # ── Build menu ─────────────────────────────────────────────────────────────

    max_submenu = pystray.Menu(
        pystray.MenuItem("Unlimited", _set_max(0), checked=_max_checked(0), radio=True),
        pystray.MenuItem("3", _set_max(3), checked=_max_checked(3), radio=True),
        pystray.MenuItem("5", _set_max(5), checked=_max_checked(5), radio=True),
        pystray.MenuItem("10", _set_max(10), checked=_max_checked(10), radio=True),
    )

    menu = pystray.Menu(
        pystray.MenuItem("Thyra Memory", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Memory system", on_toggle_system, checked=_system_checked),
        pystray.MenuItem(
            "Auto-formation", on_toggle_formation, checked=_formation_checked
        ),
        pystray.MenuItem("Max memories per turn ►", max_submenu),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open dashboard", on_open_dashboard),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit server", on_quit),
    )

    icon = pystray.Icon("thyra", _make_icon_image(), "Thyra Memory", menu)
    icon.run()
