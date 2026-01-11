"""
Caching layer for RLM-Claude-Code.

Implements: Spec §8.1 Phase 3 - Caching
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with metadata."""

    value: T
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def access(self) -> T:
        """Record access and return value."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value

    def is_expired(self, max_age_s: float) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > max_age_s


@dataclass
class CacheStats:
    """Statistics about cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """
    LRU cache with size limits and TTL.

    Implements: Spec §8.1 Frequently-accessed context caching
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_size_bytes: int = 10 * 1024 * 1024,  # 10MB
        default_ttl_s: float = 3600.0,  # 1 hour
    ):
        """
        Initialize LRU cache.

        Args:
            max_entries: Maximum number of entries
            max_size_bytes: Maximum total size in bytes
            default_ttl_s: Default time-to-live in seconds
        """
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self.default_ttl_s = default_ttl_s

        self._cache: dict[str, CacheEntry[T]] = {}
        self._stats = CacheStats()

    def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        if entry.is_expired(self.default_ttl_s):
            self._evict(key)
            self._stats.misses += 1
            return None

        self._stats.hits += 1
        return entry.access()

    def put(self, key: str, value: T, size_bytes: int | None = None) -> None:
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Size of value in bytes (estimated if not provided)
        """
        # Estimate size if not provided
        if size_bytes is None:
            size_bytes = self._estimate_size(value)

        # Evict if necessary to make room
        self._ensure_capacity(size_bytes)

        # Create entry
        entry = CacheEntry(
            value=value,
            size_bytes=size_bytes,
        )

        # Update stats
        if key in self._cache:
            old_entry = self._cache[key]
            self._stats.total_size_bytes -= old_entry.size_bytes
        else:
            self._stats.entry_count += 1

        self._cache[key] = entry
        self._stats.total_size_bytes += size_bytes

    def invalidate(self, key: str) -> bool:
        """
        Remove entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was removed
        """
        if key in self._cache:
            self._evict(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()
        self._stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def _evict(self, key: str) -> None:
        """Evict a specific key."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.total_size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            self._stats.evictions += 1

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        self._evict(lru_key)

    def _ensure_capacity(self, needed_bytes: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Evict by count
        while len(self._cache) >= self.max_entries:
            self._evict_lru()

        # Evict by size
        while self._stats.total_size_bytes + needed_bytes > self.max_size_bytes:
            if not self._cache:
                break
            self._evict_lru()

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        elif isinstance(value, dict):
            return len(json.dumps(value, default=str).encode("utf-8"))
        elif isinstance(value, list):
            return sum(self._estimate_size(item) for item in value)
        else:
            return len(str(value).encode("utf-8"))


class SummarizationCache:
    """
    Cache for context summarizations.

    Implements: Spec §8.1 Summarization cache

    Caches summaries of large contexts to avoid re-summarizing
    the same content multiple times.
    """

    def __init__(
        self,
        max_entries: int = 50,
        max_size_bytes: int = 5 * 1024 * 1024,  # 5MB
        ttl_s: float = 1800.0,  # 30 minutes
    ):
        """
        Initialize summarization cache.

        Args:
            max_entries: Maximum cached summaries
            max_size_bytes: Maximum cache size
            ttl_s: Summary TTL in seconds
        """
        self._cache: LRUCache[str] = LRUCache(
            max_entries=max_entries,
            max_size_bytes=max_size_bytes,
            default_ttl_s=ttl_s,
        )

    def get_summary(self, content: str, summary_type: str = "default") -> str | None:
        """
        Get cached summary for content.

        Args:
            content: Original content
            summary_type: Type of summary (e.g., "default", "brief", "detailed")

        Returns:
            Cached summary or None
        """
        key = self._make_key(content, summary_type)
        return self._cache.get(key)

    def put_summary(self, content: str, summary: str, summary_type: str = "default") -> None:
        """
        Cache a summary.

        Args:
            content: Original content
            summary: Generated summary
            summary_type: Type of summary
        """
        key = self._make_key(content, summary_type)
        self._cache.put(key, summary)

    def _make_key(self, content: str, summary_type: str) -> str:
        """Create cache key from content hash and type."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{summary_type}:{content_hash}"

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()


class ContextCache:
    """
    Cache for frequently-accessed context items.

    Implements: Spec §8.1 Frequently-accessed context
    """

    def __init__(
        self,
        max_files: int = 100,
        max_tool_outputs: int = 50,
        max_size_bytes: int = 20 * 1024 * 1024,  # 20MB
    ):
        """
        Initialize context cache.

        Args:
            max_files: Maximum cached files
            max_tool_outputs: Maximum cached tool outputs
            max_size_bytes: Maximum total cache size
        """
        self._files: LRUCache[str] = LRUCache(
            max_entries=max_files,
            max_size_bytes=max_size_bytes // 2,
        )
        self._tool_outputs: LRUCache[dict[str, Any]] = LRUCache(
            max_entries=max_tool_outputs,
            max_size_bytes=max_size_bytes // 2,
        )

    def get_file(self, path: str) -> str | None:
        """Get cached file content."""
        return self._files.get(path)

    def put_file(self, path: str, content: str) -> None:
        """Cache file content."""
        self._files.put(path, content)

    def get_tool_output(self, tool_name: str, input_hash: str) -> dict[str, Any] | None:
        """Get cached tool output."""
        key = f"{tool_name}:{input_hash}"
        return self._tool_outputs.get(key)

    def put_tool_output(
        self, tool_name: str, input_hash: str, output: dict[str, Any]
    ) -> None:
        """Cache tool output."""
        key = f"{tool_name}:{input_hash}"
        self._tool_outputs.put(key, output)

    def invalidate_file(self, path: str) -> bool:
        """Invalidate cached file."""
        return self._files.invalidate(path)

    def clear(self) -> None:
        """Clear all caches."""
        self._files.clear()
        self._tool_outputs.clear()


class REPLStateCache:
    """
    Persistent REPL state cache.

    Implements: Spec §8.1 REPL state persistence
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize REPL state cache.

        Args:
            cache_dir: Directory for persistent storage
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".claude" / "rlm-repl-cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for current session
        self._memory_cache: dict[str, dict[str, Any]] = {}

    def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        """
        Save REPL state.

        Args:
            session_id: Session identifier
            state: REPL state to save (must be JSON-serializable)
        """
        # Save to memory
        self._memory_cache[session_id] = state.copy()

        # Save to disk
        state_file = self.cache_dir / f"{session_id}.json"
        with open(state_file, "w") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "saved_at": time.time(),
                    "state": state,
                },
                f,
                indent=2,
                default=str,
            )

    def load_state(self, session_id: str) -> dict[str, Any] | None:
        """
        Load REPL state.

        Args:
            session_id: Session identifier

        Returns:
            REPL state or None if not found
        """
        # Check memory first
        if session_id in self._memory_cache:
            return self._memory_cache[session_id].copy()

        # Check disk
        state_file = self.cache_dir / f"{session_id}.json"
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
                state = data.get("state", {})
                self._memory_cache[session_id] = state
                return state.copy()

        return None

    def clear_state(self, session_id: str) -> bool:
        """
        Clear REPL state.

        Args:
            session_id: Session identifier

        Returns:
            True if state was cleared
        """
        cleared = False

        if session_id in self._memory_cache:
            del self._memory_cache[session_id]
            cleared = True

        state_file = self.cache_dir / f"{session_id}.json"
        if state_file.exists():
            state_file.unlink()
            cleared = True

        return cleared

    def list_sessions(self) -> list[str]:
        """List all cached session IDs."""
        sessions = set(self._memory_cache.keys())
        for state_file in self.cache_dir.glob("*.json"):
            sessions.add(state_file.stem)
        return sorted(sessions)

    def cleanup_old(self, max_age_days: int = 7) -> int:
        """
        Clean up old cached states.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of states cleaned up
        """
        cleaned = 0
        max_age_s = max_age_days * 24 * 60 * 60
        now = time.time()

        for state_file in self.cache_dir.glob("*.json"):
            try:
                with open(state_file) as f:
                    data = json.load(f)
                saved_at = data.get("saved_at", 0)
                if now - saved_at > max_age_s:
                    state_file.unlink()
                    session_id = state_file.stem
                    self._memory_cache.pop(session_id, None)
                    cleaned += 1
            except (json.JSONDecodeError, OSError):
                # Remove corrupted files
                state_file.unlink()
                cleaned += 1

        return cleaned


# Global cache instances (lazily initialized)
_summarization_cache: SummarizationCache | None = None
_context_cache: ContextCache | None = None
_repl_state_cache: REPLStateCache | None = None


def get_summarization_cache() -> SummarizationCache:
    """Get global summarization cache."""
    global _summarization_cache
    if _summarization_cache is None:
        _summarization_cache = SummarizationCache()
    return _summarization_cache


def get_context_cache() -> ContextCache:
    """Get global context cache."""
    global _context_cache
    if _context_cache is None:
        _context_cache = ContextCache()
    return _context_cache


def get_repl_state_cache() -> REPLStateCache:
    """Get global REPL state cache."""
    global _repl_state_cache
    if _repl_state_cache is None:
        _repl_state_cache = REPLStateCache()
    return _repl_state_cache


__all__ = [
    "CacheEntry",
    "CacheStats",
    "ContextCache",
    "LRUCache",
    "REPLStateCache",
    "SummarizationCache",
    "get_context_cache",
    "get_repl_state_cache",
    "get_summarization_cache",
]
