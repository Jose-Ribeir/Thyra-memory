"""Hot cache: in-memory snapshot with TTL and invalidation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from thyra.config import HOT_CACHE_TTL_SECONDS


@dataclass
class _CacheEntry:
    payload: Any
    expires_at: float


class HotCache:
    """Thread-safe dict-based cache with per-key TTL and manual invalidation."""

    def __init__(self, ttl: int = HOT_CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._store: dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return None
            return entry.payload

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(
                payload=value,
                expires_at=time.monotonic() + self._ttl,
            )

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> None:
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# Module-level singleton shared by the whole process
HOT_CACHE = HotCache()
