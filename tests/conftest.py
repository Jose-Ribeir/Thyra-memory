"""Shared pytest fixtures."""

import os
import sqlite3
import tempfile
import pytest

from thyra.db.connection import DBConnection


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Provide a fresh temporary database for each test."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("THYRA_DB_PATH", db_file)

    # Patch the module-level default so DBConnection picks up the env var
    import thyra.config as cfg

    monkeypatch.setattr(cfg, "THYRA_DB_PATH", db_file)

    # Reset thread-local connection so it rebuilds with the new path
    DBConnection._local = type(DBConnection._local)()

    # Clear module-level hot cache so tests don't see each other's snapshots
    from thyra.recall.cache import HOT_CACHE

    HOT_CACHE.clear()

    conn = DBConnection.get(db_file)
    yield conn

    DBConnection.close()
    DBConnection._local = type(DBConnection._local)()
    HOT_CACHE.clear()
