"""Thread-local SQLite connection pool with WAL mode and FK enforcement."""

import os
import sqlite3
import threading
from thyra.config import THYRA_DB_PATH


class DBConnection:
    _local = threading.local()

    @classmethod
    def get(cls, db_path: str | None = None) -> sqlite3.Connection:
        if not hasattr(cls._local, "conn") or cls._local.conn is None:
            path = db_path or THYRA_DB_PATH
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA cache_size=-32000")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA temp_store=MEMORY")
            cls._local.conn = conn
            from thyra.db.migrations import migrate

            migrate(conn)
        return cls._local.conn

    @classmethod
    def close(cls) -> None:
        if hasattr(cls._local, "conn") and cls._local.conn:
            cls._local.conn.close()
            cls._local.conn = None


def get_conn(db_path: str | None = None) -> sqlite3.Connection:
    return DBConnection.get(db_path)
