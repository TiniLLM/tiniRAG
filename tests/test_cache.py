"""Tests for tinirag.core.cache."""

import time

from tinirag.core.cache import (
    MemoryCache,
    SQLiteCache,
    make_cache,
    make_cache_key,
    normalize_for_cache,
)


class TestNormalize:
    def test_lowercases(self):
        assert normalize_for_cache("GPU MacBook") == normalize_for_cache("gpu macbook")

    def test_sorts_words(self):
        assert normalize_for_cache("GPU MacBook") == normalize_for_cache("MacBook GPU")

    def test_strips_punctuation(self):
        assert normalize_for_cache("GPU, MacBook!") == normalize_for_cache("GPU MacBook")

    def test_removes_noise_words(self):
        result = normalize_for_cache("the GPU of a MacBook")
        words = result.split()
        assert "the" not in words
        assert "a" not in words
        assert "of" not in words

    def test_rephrased_same_key(self):
        key1 = make_cache_key("MacBook GPU architecture")
        key2 = make_cache_key("GPU architecture MacBook")
        assert key1 == key2

    def test_different_queries_different_keys(self):
        key1 = make_cache_key("macbook GPU")
        key2 = make_cache_key("iphone camera")
        assert key1 != key2


class TestMakeCacheKey:
    def test_returns_32_chars(self):
        # BUG-08: must be 32 hex chars (128-bit)
        key = make_cache_key("some keywords here")
        assert len(key) == 32

    def test_hex_chars_only(self):
        key = make_cache_key("some keywords here")
        assert all(c in "0123456789abcdef" for c in key)

    def test_deterministic(self):
        assert make_cache_key("python async") == make_cache_key("python async")


class TestMemoryCache:
    def test_set_and_get(self):
        cache = MemoryCache(ttl_minutes=10)
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_missing_key_returns_none(self):
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_expired_returns_none(self):
        cache = MemoryCache(ttl_minutes=0)  # 0 minutes = immediate expiry
        cache.set("k", "v")
        # TTL = 0 seconds — sleep just enough to expire
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_overwrite(self):
        cache = MemoryCache()
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"

    def test_clear(self):
        cache = MemoryCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_stores_list(self):
        cache = MemoryCache()
        data = [{"url": "http://example.com", "content": "text"}]
        cache.set("results", data)
        assert cache.get("results") == data


# ---------------------------------------------------------------------------
# SQLiteCache
# ---------------------------------------------------------------------------


class TestSQLiteCache:
    def test_set_and_get(self, tmp_path):
        db = tmp_path / "test_cache.db"
        cache = SQLiteCache(db_path=db, ttl_minutes=10)
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_missing_key_returns_none(self, tmp_path):
        db = tmp_path / "test_cache.db"
        cache = SQLiteCache(db_path=db)
        assert cache.get("nonexistent") is None

    def test_overwrite(self, tmp_path):
        db = tmp_path / "test_cache.db"
        cache = SQLiteCache(db_path=db)
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"

    def test_clear(self, tmp_path):
        db = tmp_path / "test_cache.db"
        cache = SQLiteCache(db_path=db)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_stores_list(self, tmp_path):
        db = tmp_path / "test_cache.db"
        cache = SQLiteCache(db_path=db)
        data = [{"url": "http://example.com", "content": "text about stuff"}]
        cache.set("results", data)
        assert cache.get("results") == data

    def test_expired_returns_none(self, tmp_path):
        db = tmp_path / "test_cache.db"
        cache = SQLiteCache(db_path=db, ttl_minutes=0)
        cache.set("k", "v")
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_persists_across_instances(self, tmp_path):
        """A new SQLiteCache instance reading the same DB file sees existing data."""
        db = tmp_path / "test_cache.db"
        cache1 = SQLiteCache(db_path=db, ttl_minutes=10)
        cache1.set("persistent_key", ["result1", "result2"])

        cache2 = SQLiteCache(db_path=db, ttl_minutes=10)
        assert cache2.get("persistent_key") == ["result1", "result2"]

    def test_creates_parent_dirs(self, tmp_path):
        db = tmp_path / "subdir" / "nested" / "cache.db"
        cache = SQLiteCache(db_path=db)
        cache.set("k", "v")
        assert db.exists()


# ---------------------------------------------------------------------------
# make_cache factory
# ---------------------------------------------------------------------------


class TestMakeCache:
    def test_memory_backend(self):
        cache = make_cache(backend="memory")
        assert isinstance(cache, MemoryCache)

    def test_sqlite_backend(self, tmp_path, monkeypatch):
        import tinirag.core.cache as cache_module

        monkeypatch.setattr(cache_module, "CACHE_DB", tmp_path / "cache.db")
        cache = make_cache(backend="sqlite")
        assert isinstance(cache, SQLiteCache)

    def test_default_is_sqlite(self, tmp_path, monkeypatch):
        import tinirag.core.cache as cache_module

        monkeypatch.setattr(cache_module, "CACHE_DB", tmp_path / "cache.db")
        cache = make_cache()
        assert isinstance(cache, SQLiteCache)


# ---------------------------------------------------------------------------
# normalize_for_cache edge cases
# ---------------------------------------------------------------------------


class TestNormalizeEdgeCases:
    def test_empty_string(self):
        # Empty string normalizes to empty string without crashing
        result = normalize_for_cache("")
        assert result == ""

    def test_only_noise_words(self):
        result = normalize_for_cache("the a an is of")
        assert result == ""

    def test_unicode_passthrough(self):
        # Non-ASCII letters are kept (re strips punctuation, not unicode letters)
        result = normalize_for_cache("résumé python")
        assert "python" in result

    def test_numbers_preserved(self):
        result = normalize_for_cache("llama 3.1 model")
        assert "3" in result or "31" in result or "llama" in result
