"""
Prompt caching integration for Anthropic API.

Implements: SPEC-08.10-08.15 Prompt Caching Integration
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheablePrompt:
    """
    A prompt structured for optimal cache hits.

    Implements: SPEC-08.10
    """

    system_content: str
    system_cacheable: bool
    cacheable_content: str
    query_content: str
    cache_key: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    def to_anthropic_messages(self) -> dict[str, Any]:
        """
        Convert to Anthropic API format with cache_control.

        Returns:
            Dict with 'system' and 'messages' keys in Anthropic format.
        """
        result: dict[str, Any] = {}

        # System prompt with cache control
        if self.system_content:
            if self.system_cacheable:
                result["system"] = [
                    {
                        "type": "text",
                        "text": self.system_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                result["system"] = self.system_content
        else:
            result["system"] = None

        # Build user message content blocks
        content_blocks: list[dict[str, Any]] = []

        # Cacheable content first (with cache_control)
        if self.cacheable_content:
            content_blocks.append(
                {
                    "type": "text",
                    "text": self.cacheable_content,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        # Conversation history (if any)
        if self.conversation_history:
            history_text = "\n\n".join(
                f"[{msg['role'].upper()}]: {msg['content']}" for msg in self.conversation_history
            )
            content_blocks.append(
                {
                    "type": "text",
                    "text": f"\n\n--- Conversation History ---\n{history_text}\n",
                }
            )

        # Query-specific content last (no cache_control)
        if self.query_content:
            content_blocks.append(
                {
                    "type": "text",
                    "text": f"\n\n--- Current Query ---\n{self.query_content}",
                }
            )

        result["messages"] = [{"role": "user", "content": content_blocks}]

        return result


@dataclass
class StructuredPrompt:
    """Structured prompt with separate cacheable and non-cacheable parts."""

    cacheable_content: str
    query_content: str
    system_content: str = ""
    system_cacheable: bool = True


@dataclass
class CachePrefix:
    """A cached prefix entry."""

    prefix_id: str
    content: str
    context_hash: str
    created_at: float
    last_used: float
    use_count: int = 1


class CachePrefixRegistry:
    """
    Registry for cache prefixes used in recursive calls.

    Implements: SPEC-08.13
    """

    def __init__(self, ttl_seconds: float = 300.0):
        """
        Initialize cache prefix registry.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 5 minutes)
        """
        self.ttl_seconds = ttl_seconds
        self._prefixes: dict[str, CachePrefix] = {}
        self._hash_index: dict[str, str] = {}  # hash -> prefix_id
        self._counter = 0

    def register(self, content: str, context_hash: str) -> str:
        """
        Register a cache prefix.

        Args:
            content: The cacheable content
            context_hash: Hash of the context for lookup

        Returns:
            Prefix ID for reference
        """
        # Check if already registered
        if context_hash in self._hash_index:
            prefix_id = self._hash_index[context_hash]
            if prefix_id in self._prefixes:
                prefix = self._prefixes[prefix_id]
                prefix.last_used = time.time()
                prefix.use_count += 1
                return prefix_id

        # Create new prefix
        self._counter += 1
        prefix_id = f"prefix_{self._counter}"

        prefix = CachePrefix(
            prefix_id=prefix_id,
            content=content,
            context_hash=context_hash,
            created_at=time.time(),
            last_used=time.time(),
        )

        self._prefixes[prefix_id] = prefix
        self._hash_index[context_hash] = prefix_id

        return prefix_id

    def get(self, prefix_id: str) -> CachePrefix | None:
        """Get prefix by ID."""
        prefix = self._prefixes.get(prefix_id)
        if prefix and not self._is_expired(prefix):
            prefix.last_used = time.time()
            return prefix
        return None

    def find_by_hash(self, context_hash: str) -> CachePrefix | None:
        """Find prefix by context hash."""
        prefix_id = self._hash_index.get(context_hash)
        if prefix_id:
            return self.get(prefix_id)
        return None

    def _is_expired(self, prefix: CachePrefix) -> bool:
        """Check if prefix is expired."""
        if self.ttl_seconds <= 0:
            return True
        return (time.time() - prefix.last_used) > self.ttl_seconds

    def cleanup_expired(self) -> int:
        """Remove expired prefixes. Returns count removed."""
        expired = [pid for pid, p in self._prefixes.items() if self._is_expired(p)]

        for prefix_id in expired:
            prefix = self._prefixes.pop(prefix_id, None)
            if prefix:
                self._hash_index.pop(prefix.context_hash, None)

        return len(expired)

    def count(self) -> int:
        """Return count of active prefixes."""
        return len(self._prefixes)


@dataclass
class CacheMetrics:
    """
    Metrics for cache hit/miss tracking.

    Implements: SPEC-08.14
    """

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    tokens_cached: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def estimated_savings_ratio(self) -> float:
        """
        Estimate cost savings ratio from caching.

        Cache reads are 90% cheaper than regular input tokens.
        """
        if self.tokens_saved == 0:
            return 0.0
        # Savings = tokens_saved * 0.9 (90% discount on cached tokens)
        return 0.9 * (self.tokens_saved / (self.tokens_saved + self.tokens_cached + 1))

    def record_hit(self, tokens_saved: int) -> None:
        """Record a cache hit."""
        self.hits += 1
        self.tokens_saved += tokens_saved

    def record_miss(self, tokens_cached: int) -> None:
        """Record a cache miss (new cache creation)."""
        self.misses += 1
        self.tokens_cached += tokens_cached

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.tokens_cached = 0


class PromptCacheManager:
    """
    Manager for prompt caching optimization.

    Implements: SPEC-08.10-08.15

    Structures prompts for optimal cache hits and tracks metrics.
    """

    def __init__(
        self,
        min_cache_tokens: int = 1024,
        ttl_seconds: float = 300.0,
    ):
        """
        Initialize prompt cache manager.

        Args:
            min_cache_tokens: Minimum tokens to enable caching (default: 1024)
            ttl_seconds: TTL for cache prefix registry entries
        """
        self.min_cache_tokens = min_cache_tokens
        self.registry = CachePrefixRegistry(ttl_seconds=ttl_seconds)
        self.metrics = CacheMetrics()

    def build_prompt(
        self,
        query: str,
        shared_context: str | dict[str, str] | None = None,
        system: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> CacheablePrompt:
        """
        Build a cache-optimized prompt.

        Implements: SPEC-08.10-08.12

        Args:
            query: The user's query (placed last, not cached)
            shared_context: Shared context - string or dict of filename->content
            system: System prompt (cached)
            conversation_history: Previous conversation turns

        Returns:
            CacheablePrompt structured for optimal caching
        """
        # Build cacheable content from shared context
        if shared_context is None:
            cacheable_content = ""
        elif isinstance(shared_context, dict):
            # Multiple files - combine them
            parts = []
            for filename, content in shared_context.items():
                parts.append(f"--- {filename} ---\n{content}")
            cacheable_content = "\n\n".join(parts)
        else:
            cacheable_content = shared_context

        # Generate cache key from cacheable content
        cache_key = self._compute_cache_key(cacheable_content, system or "")

        # Register with prefix registry for reuse
        if cacheable_content:
            self.registry.register(
                content=cacheable_content,
                context_hash=cache_key,
            )

        return CacheablePrompt(
            system_content=system or "",
            system_cacheable=bool(system),
            cacheable_content=cacheable_content,
            query_content=query,
            cache_key=cache_key,
            conversation_history=conversation_history or [],
        )

    def _compute_cache_key(self, content: str, system: str) -> str:
        """Compute cache key from content."""
        combined = f"{system}|{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def record_response_metrics(self, usage: dict[str, int]) -> None:
        """
        Record metrics from API response usage data.

        Args:
            usage: API response usage dict with cache token info
        """
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)

        if cache_read > 0:
            self.metrics.record_hit(tokens_saved=cache_read)

        if cache_creation > 0:
            self.metrics.record_miss(tokens_cached=cache_creation)


def build_cacheable_prompt(
    query: str,
    shared_context: str | dict[str, str] | None = None,
    system: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> CacheablePrompt:
    """
    Convenience function to build a cacheable prompt.

    Implements: SPEC-08.10-08.12

    Args:
        query: The user's query
        shared_context: Shared context (files, code, etc.)
        system: System prompt
        conversation_history: Previous turns

    Returns:
        CacheablePrompt optimized for caching
    """
    manager = PromptCacheManager()
    return manager.build_prompt(
        query=query,
        shared_context=shared_context,
        system=system,
        conversation_history=conversation_history,
    )


__all__ = [
    "CacheablePrompt",
    "CacheMetrics",
    "CachePrefix",
    "CachePrefixRegistry",
    "PromptCacheManager",
    "StructuredPrompt",
    "build_cacheable_prompt",
]
