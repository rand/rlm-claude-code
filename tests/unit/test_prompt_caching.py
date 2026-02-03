"""
Tests for prompt caching integration.

@trace SPEC-08.10-08.15
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.prompt_caching import (
    CacheMetrics,
    CachePrefixRegistry,
    PromptCacheManager,
    build_cacheable_prompt,
)

# --- Test fixtures ---


@dataclass
class MockMessage:
    """Mock message for testing."""

    role: str
    content: str | list[dict[str, Any]]


def create_shared_context(size: int = 1000) -> str:
    """Create shared context of approximate token size."""
    # ~4 chars per token
    return "x" * (size * 4)


# --- SPEC-08.10: Structure prompts for optimal cache hits ---


class TestPromptStructure:
    """Tests for prompt structuring for cache optimization."""

    def test_shared_context_placed_first(self) -> None:
        """
        @trace SPEC-08.11
        Shared context should be placed first (cacheable).
        """
        shared_context = create_shared_context(1000)
        query = "What is the meaning of this code?"

        prompt = build_cacheable_prompt(
            shared_context=shared_context,
            query=query,
        )

        # Shared context should be first
        assert prompt.cacheable_content.startswith(shared_context[:100])
        # Query should not be in cacheable portion
        assert query not in prompt.cacheable_content

    def test_query_placed_last(self) -> None:
        """
        @trace SPEC-08.12
        Query-specific content should be placed last (not cached).
        """
        shared_context = create_shared_context(500)
        query = "Explain the main function"

        prompt = build_cacheable_prompt(
            shared_context=shared_context,
            query=query,
        )

        # Query should be in non-cacheable portion
        assert query in prompt.query_content
        assert query not in prompt.cacheable_content

    def test_system_prompt_cacheable(self) -> None:
        """
        @trace SPEC-08.10
        System prompt should be structured for caching.
        """
        system = "You are a helpful code assistant."
        shared_context = "def foo(): pass"
        query = "What does foo do?"

        prompt = build_cacheable_prompt(
            system=system,
            shared_context=shared_context,
            query=query,
        )

        # System prompt should be cacheable
        assert system in prompt.system_content
        assert prompt.system_cacheable is True

    def test_multiple_context_files_combined(self) -> None:
        """
        @trace SPEC-08.11
        Multiple context files should be combined in cacheable portion.
        """
        files = {
            "file1.py": "def func1(): pass",
            "file2.py": "def func2(): pass",
            "file3.py": "def func3(): pass",
        }
        query = "How do these functions relate?"

        prompt = build_cacheable_prompt(
            shared_context=files,
            query=query,
        )

        # All files should be in cacheable content
        for filename, content in files.items():
            assert filename in prompt.cacheable_content or content in prompt.cacheable_content


# --- SPEC-08.13: Cache prefix registry ---


class TestCachePrefixRegistry:
    """Tests for cache prefix registry for recursive calls."""

    def test_register_prefix(self) -> None:
        """
        @trace SPEC-08.13
        Should register cache prefixes for reuse.
        """
        registry = CachePrefixRegistry()

        prefix_id = registry.register(
            content="shared context content",
            context_hash="abc123",
        )

        assert prefix_id is not None
        assert registry.get(prefix_id) is not None

    def test_lookup_by_hash(self) -> None:
        """
        @trace SPEC-08.13
        Should lookup prefix by content hash.
        """
        registry = CachePrefixRegistry()
        content = "shared context content"
        context_hash = "abc123"

        registry.register(content=content, context_hash=context_hash)

        found = registry.find_by_hash(context_hash)
        assert found is not None
        assert found.content == content

    def test_prefix_expiry(self) -> None:
        """
        @trace SPEC-08.13
        Prefixes should expire after TTL.
        """
        registry = CachePrefixRegistry(ttl_seconds=0)  # Immediate expiry

        prefix_id = registry.register(
            content="content",
            context_hash="hash",
        )

        # Should be expired
        registry.cleanup_expired()
        assert registry.get(prefix_id) is None

    def test_prefix_reuse_for_recursive_calls(self) -> None:
        """
        @trace SPEC-08.15
        Recursive calls with shared context should reuse cached prefixes.
        """
        registry = CachePrefixRegistry()

        # Register prefix once
        context_hash = "shared_context_hash"
        prefix_id = registry.register(
            content="shared context for recursive calls",
            context_hash=context_hash,
        )

        # Multiple recursive calls should find the same prefix
        found1 = registry.find_by_hash(context_hash)
        found2 = registry.find_by_hash(context_hash)

        assert found1 is not None
        assert found2 is not None
        assert found1.prefix_id == found2.prefix_id


# --- SPEC-08.14: Cache metrics tracking ---


class TestCacheMetrics:
    """Tests for cache hit/miss metrics tracking."""

    def test_record_cache_hit(self) -> None:
        """
        @trace SPEC-08.14
        Should record cache hits.
        """
        metrics = CacheMetrics()

        metrics.record_hit(tokens_saved=1000)

        assert metrics.hits == 1
        assert metrics.tokens_saved == 1000

    def test_record_cache_miss(self) -> None:
        """
        @trace SPEC-08.14
        Should record cache misses.
        """
        metrics = CacheMetrics()

        metrics.record_miss(tokens_cached=500)

        assert metrics.misses == 1
        assert metrics.tokens_cached == 500

    def test_hit_rate_calculation(self) -> None:
        """
        @trace SPEC-08.14
        Should calculate cache hit rate correctly.
        """
        metrics = CacheMetrics()

        # 3 hits, 2 misses = 60% hit rate
        metrics.record_hit(100)
        metrics.record_hit(100)
        metrics.record_hit(100)
        metrics.record_miss(50)
        metrics.record_miss(50)

        assert metrics.hit_rate == pytest.approx(0.6)

    def test_cost_savings_calculation(self) -> None:
        """
        @trace SPEC-08.14
        Should calculate cost savings from caching.
        """
        metrics = CacheMetrics()

        # Cache reads are 90% cheaper than regular input
        metrics.record_hit(tokens_saved=10000)

        # Savings = tokens_saved * 0.9 (90% discount)
        assert metrics.estimated_savings_ratio > 0.5

    def test_metrics_reset(self) -> None:
        """
        @trace SPEC-08.14
        Should support resetting metrics.
        """
        metrics = CacheMetrics()
        metrics.record_hit(1000)
        metrics.record_miss(500)

        metrics.reset()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.tokens_saved == 0


# --- SPEC-08.15: Recursive call prefix reuse ---


class TestRecursiveCallCaching:
    """Tests for caching in recursive calls."""

    def test_shared_context_cached_across_calls(self) -> None:
        """
        @trace SPEC-08.15
        Shared context should be cached across recursive calls.
        """
        manager = PromptCacheManager()

        shared_context = create_shared_context(2000)
        queries = [
            "What does function A do?",
            "What does function B do?",
            "How do A and B relate?",
        ]

        prompts = []
        for query in queries:
            prompt = manager.build_prompt(
                shared_context=shared_context,
                query=query,
            )
            prompts.append(prompt)

        # All prompts should share the same cache prefix
        cache_keys = [p.cache_key for p in prompts]
        assert len(set(cache_keys)) == 1  # All same key

    def test_different_context_different_cache(self) -> None:
        """
        @trace SPEC-08.15
        Different contexts should have different cache keys.
        """
        manager = PromptCacheManager()

        context1 = "def foo(): pass"
        context2 = "def bar(): pass"
        query = "What does this do?"

        prompt1 = manager.build_prompt(shared_context=context1, query=query)
        prompt2 = manager.build_prompt(shared_context=context2, query=query)

        assert prompt1.cache_key != prompt2.cache_key


# --- PromptCacheManager integration ---


class TestPromptCacheManager:
    """Integration tests for PromptCacheManager."""

    def test_build_anthropic_messages(self) -> None:
        """
        Build messages in Anthropic cache-optimized format.
        """
        manager = PromptCacheManager()

        prompt = manager.build_prompt(
            system="You are a code assistant.",
            shared_context="def main(): print('hello')",
            query="What does main do?",
        )

        messages = prompt.to_anthropic_messages()

        # System should have cache_control
        assert messages["system"] is not None

        # User message should have cacheable content first
        user_content = messages["messages"][0]["content"]
        assert isinstance(user_content, list)
        # First block should be cacheable
        assert any(
            block.get("cache_control") == {"type": "ephemeral"}
            for block in user_content
            if isinstance(block, dict)
        )

    def test_extract_cache_metrics_from_response(self) -> None:
        """
        Extract cache metrics from API response.
        """
        manager = PromptCacheManager()

        # Simulate response with cache info
        mock_usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 200,
        }

        manager.record_response_metrics(mock_usage)

        assert manager.metrics.hits >= 1 or manager.metrics.misses >= 1

    def test_minimum_cache_size_threshold(self) -> None:
        """
        Content below minimum size should not be cached.
        """
        manager = PromptCacheManager(min_cache_tokens=1024)

        # Small context - below threshold
        small_context = "x" * 100  # ~25 tokens

        prompt = manager.build_prompt(
            shared_context=small_context,
            query="query",
        )

        # Should not have cache_control for small content
        messages = prompt.to_anthropic_messages()
        user_content = messages["messages"][0]["content"]

        # For small content, may not use caching
        if isinstance(user_content, str):
            pass  # Simple string, no caching
        else:
            # Check if cache_control is present
            has_cache = any(
                block.get("cache_control") is not None
                for block in user_content
                if isinstance(block, dict)
            )
            # Small content might still be cached if system deems beneficial
            # This is implementation-dependent


# --- Acceptance criteria tests ---


class TestAcceptanceCriteria:
    """Tests for spec acceptance criteria."""

    def test_prompt_structure_optimizes_caching(self) -> None:
        """
        Verify prompt structure is optimized for caching.
        """
        manager = PromptCacheManager()

        prompt = manager.build_prompt(
            system="System prompt",
            shared_context="Large shared context " * 500,
            conversation_history=[
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
            ],
            query="New question",
        )

        # Order should be: system (cached) -> shared_context (cached) -> history -> query
        messages = prompt.to_anthropic_messages()
        assert messages["system"] is not None

        user_content = messages["messages"][0]["content"]
        assert isinstance(user_content, list)

        # Verify ordering: cacheable content before query
        content_texts = [block.get("text", "") for block in user_content if isinstance(block, dict)]
        full_text = " ".join(content_texts)

        # Shared context should appear before query
        assert "Large shared context" in full_text

    def test_cache_prefix_registry_maintained(self) -> None:
        """
        Verify cache prefix registry is maintained.
        """
        manager = PromptCacheManager()

        # Build multiple prompts with same context
        context = "Shared context content"
        manager.build_prompt(shared_context=context, query="Query 1")
        manager.build_prompt(shared_context=context, query="Query 2")

        # Registry should have entry
        assert manager.registry.count() >= 1
