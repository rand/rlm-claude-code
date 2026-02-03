"""
Unit tests for cache module.

Implements: Spec §8.1 Phase 3 - Caching tests
"""

import tempfile
import time
from pathlib import Path

import pytest

from src.cache import (
    CacheEntry,
    CacheStats,
    ContextCache,
    LRUCache,
    REPLStateCache,
    SummarizationCache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self):
        """Can create cache entry."""
        entry = CacheEntry(value="test")
        assert entry.value == "test"
        assert entry.access_count == 0

    def test_access_increments_count(self):
        """Access increments count and updates timestamp."""
        entry = CacheEntry(value="test")
        original_time = entry.last_accessed

        time.sleep(0.01)
        result = entry.access()

        assert result == "test"
        assert entry.access_count == 1
        assert entry.last_accessed > original_time

    def test_is_expired(self):
        """Can check if entry is expired."""
        entry = CacheEntry(value="test")
        entry.created_at = time.time() - 100  # 100 seconds ago

        assert entry.is_expired(50) is True
        assert entry.is_expired(200) is False


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_put_and_get(self):
        """Can put and get values."""
        cache: LRUCache[str] = LRUCache()
        cache.put("key1", "value1")

        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        """Get missing key returns None."""
        cache: LRUCache[str] = LRUCache()
        assert cache.get("missing") is None

    def test_get_expired_returns_none(self):
        """Get expired entry returns None."""
        cache: LRUCache[str] = LRUCache(default_ttl_s=0.01)
        cache.put("key1", "value1")

        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_evicts_lru_on_max_entries(self):
        """Evicts LRU entry when max entries reached."""
        cache: LRUCache[str] = LRUCache(max_entries=2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_evicts_on_max_size(self):
        """Evicts entries when max size reached."""
        cache: LRUCache[str] = LRUCache(max_size_bytes=100)
        cache.put("key1", "a" * 50, size_bytes=50)
        cache.put("key2", "b" * 50, size_bytes=50)

        # Add key3, should evict to make room
        cache.put("key3", "c" * 50, size_bytes=50)

        assert cache.get("key3") == "c" * 50

    def test_invalidate(self):
        """Can invalidate entry."""
        cache: LRUCache[str] = LRUCache()
        cache.put("key1", "value1")

        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        assert cache.invalidate("key1") is False

    def test_clear(self):
        """Can clear all entries."""
        cache: LRUCache[str] = LRUCache()
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats_tracking(self):
        """Tracks hits, misses, evictions."""
        cache: LRUCache[str] = LRUCache(max_entries=2)
        cache.put("key1", "value1")

        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_hit_rate(self):
        """Calculates hit rate correctly."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 0.7


class TestSummarizationCache:
    """Tests for SummarizationCache class."""

    def test_put_and_get_summary(self):
        """Can cache and retrieve summaries."""
        cache = SummarizationCache()
        content = "This is a long piece of content that needs summarization."
        summary = "Content summary."

        cache.put_summary(content, summary)
        assert cache.get_summary(content) == summary

    def test_different_summary_types(self):
        """Can cache different summary types for same content."""
        cache = SummarizationCache()
        content = "Some content"

        cache.put_summary(content, "Brief summary", "brief")
        cache.put_summary(content, "Detailed summary", "detailed")

        assert cache.get_summary(content, "brief") == "Brief summary"
        assert cache.get_summary(content, "detailed") == "Detailed summary"

    def test_cache_miss(self):
        """Returns None for uncached content."""
        cache = SummarizationCache()
        assert cache.get_summary("uncached content") is None


class TestContextCache:
    """Tests for ContextCache class."""

    def test_cache_file(self):
        """Can cache file content."""
        cache = ContextCache()
        cache.put_file("/path/to/file.py", "file content")

        assert cache.get_file("/path/to/file.py") == "file content"

    def test_cache_tool_output(self):
        """Can cache tool output."""
        cache = ContextCache()
        output = {"result": "success", "data": [1, 2, 3]}

        cache.put_tool_output("bash", "input_hash_123", output)
        assert cache.get_tool_output("bash", "input_hash_123") == output

    def test_invalidate_file(self):
        """Can invalidate cached file."""
        cache = ContextCache()
        cache.put_file("/path/to/file.py", "content")

        assert cache.invalidate_file("/path/to/file.py") is True
        assert cache.get_file("/path/to/file.py") is None

    def test_clear(self):
        """Can clear all caches."""
        cache = ContextCache()
        cache.put_file("/file.py", "content")
        cache.put_tool_output("bash", "hash", {"result": "ok"})

        cache.clear()

        assert cache.get_file("/file.py") is None
        assert cache.get_tool_output("bash", "hash") is None


class TestREPLStateCache:
    """Tests for REPLStateCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load_state(self, temp_cache_dir):
        """Can save and load REPL state."""
        cache = REPLStateCache(cache_dir=temp_cache_dir)
        state = {"variables": {"x": 1, "y": 2}, "history": ["cmd1", "cmd2"]}

        cache.save_state("session1", state)
        loaded = cache.load_state("session1")

        assert loaded == state

    def test_load_nonexistent_returns_none(self, temp_cache_dir):
        """Load nonexistent session returns None."""
        cache = REPLStateCache(cache_dir=temp_cache_dir)
        assert cache.load_state("nonexistent") is None

    def test_clear_state(self, temp_cache_dir):
        """Can clear session state."""
        cache = REPLStateCache(cache_dir=temp_cache_dir)
        cache.save_state("session1", {"data": "value"})

        assert cache.clear_state("session1") is True
        assert cache.load_state("session1") is None

    def test_list_sessions(self, temp_cache_dir):
        """Can list cached sessions."""
        cache = REPLStateCache(cache_dir=temp_cache_dir)
        cache.save_state("session1", {})
        cache.save_state("session2", {})

        sessions = cache.list_sessions()
        assert "session1" in sessions
        assert "session2" in sessions

    def test_cleanup_old(self, temp_cache_dir):
        """Can clean up old cached states."""
        cache = REPLStateCache(cache_dir=temp_cache_dir)
        cache.save_state("old_session", {})

        # Manually make it old
        state_file = temp_cache_dir / "old_session.json"
        import json

        with open(state_file) as f:
            data = json.load(f)
        data["saved_at"] = time.time() - (10 * 24 * 60 * 60)  # 10 days ago
        with open(state_file, "w") as f:
            json.dump(data, f)

        cache.save_state("new_session", {})

        cleaned = cache.cleanup_old(max_age_days=7)

        assert cleaned == 1
        assert cache.load_state("old_session") is None
        assert cache.load_state("new_session") is not None

    def test_memory_cache_used_first(self, temp_cache_dir):
        """Memory cache is checked before disk."""
        cache = REPLStateCache(cache_dir=temp_cache_dir)
        state = {"key": "value"}

        cache.save_state("session1", state)

        # Modify disk file
        state_file = temp_cache_dir / "session1.json"
        import json

        with open(state_file, "w") as f:
            json.dump(
                {"session_id": "session1", "saved_at": time.time(), "state": {"key": "modified"}}, f
            )

        # Should still get memory cached version
        loaded = cache.load_state("session1")
        assert loaded == state
