"""Search result cache: in-memory dict or SQLite backend."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from tinirag.config import CONFIG_DIR

CACHE_DB = CONFIG_DIR / "cache.db"
_NOISE = {"a", "an", "the", "is", "of"}


def normalize_for_cache(keywords: str) -> str:
    """Normalize keyword string so rephrased queries hit the same cache entry (OPT-06)."""
    s = keywords.lower()
    s = re.sub(r"[^\w\s]", "", s)
    words = sorted(s.split())
    words = [w for w in words if w not in _NOISE]
    return " ".join(words)


def make_cache_key(keywords: str) -> str:
    """Return 32-char (128-bit) SHA-256 hex key (BUG-08: 16-char has collision risk)."""
    normalized = normalize_for_cache(keywords)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


class MemoryCache:
    """Simple in-process dict cache with TTL."""

    def __init__(self, ttl_minutes: int = 10) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_minutes * 60

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, data = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return data

    def set(self, key: str, data: Any) -> None:
        self._store[key] = (time.time(), data)

    def clear(self) -> None:
        self._store.clear()


class SQLiteCache:
    """SQLite-backed cache with TTL, persists across runs."""

    def __init__(self, db_path: Path = CACHE_DB, ttl_minutes: int = 10) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = str(db_path)
        self._ttl = ttl_minutes * 60
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, ts REAL, data TEXT)"
            )

    def get(self, key: str) -> Any | None:
        with sqlite3.connect(self._db) as con:
            row = con.execute("SELECT ts, data FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        ts, raw = row
        if time.time() - ts > self._ttl:
            with sqlite3.connect(self._db) as con:
                con.execute("DELETE FROM cache WHERE key = ?", (key,))
            return None
        return json.loads(raw)

    def set(self, key: str, data: Any) -> None:
        with sqlite3.connect(self._db) as con:
            con.execute(
                "INSERT OR REPLACE INTO cache (key, ts, data) VALUES (?, ?, ?)",
                (key, time.time(), json.dumps(data)),
            )

    def clear(self) -> None:
        with sqlite3.connect(self._db) as con:
            con.execute("DELETE FROM cache")


def make_cache(backend: str = "sqlite", ttl_minutes: int = 10) -> MemoryCache | SQLiteCache:
    """Factory: return the right cache backend."""
    if backend == "memory":
        return MemoryCache(ttl_minutes=ttl_minutes)
    return SQLiteCache(ttl_minutes=ttl_minutes)
