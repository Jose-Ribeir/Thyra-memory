"""Derive a stable project ID from a working directory path."""

from __future__ import annotations

import os
import re

_PROJECT_MARKERS = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "CMakeLists.txt",
    }
)


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9_-]", "-", s.lower()).strip("-")
    return s or "unknown"


def resolve_project_id(cwd: str) -> str:
    """Return a stable slug identifying the project that owns *cwd*.

    Search order:
    1. Walk up looking for .git (handles both normal repos and worktrees).
    2. Walk up looking for common project root markers (pyproject.toml etc.).
    3. Fall back to the basename of cwd itself.
    4. Final fallback: THYRA_AGENT_ID env var or "global".
    """
    if not cwd:
        return os.environ.get("THYRA_AGENT_ID", "global")

    path = os.path.normpath(cwd)

    # Pass 1: git root
    p = path
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return _slug(os.path.basename(p))
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent

    # Pass 2: generic project root markers
    p = path
    while True:
        for marker in _PROJECT_MARKERS:
            if os.path.exists(os.path.join(p, marker)):
                return _slug(os.path.basename(p))
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent

    # Pass 3: basename of the supplied cwd
    base = _slug(os.path.basename(path))
    return base or os.environ.get("THYRA_AGENT_ID", "global")
