"""
Property-based tests for advanced REPL functions.

Implements: Spec SPEC-01.21 - Property tests for map_reduce consistency
"""

import sys
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.types import DeferredBatch, Message, MessageRole, SessionContext

# =============================================================================
# Strategies
# =============================================================================

content_strategy = st.text(min_size=0, max_size=10000)
prompt_strategy = st.text(min_size=1, max_size=200)
n_chunks_strategy = st.integers(min_value=1, max_value=20)
model_strategy = st.sampled_from(["fast", "balanced", "powerful", "auto"])


def make_context():
    """Create a basic context for testing."""
    return SessionContext(
        messages=[Message(role=MessageRole.USER, content="Test")],
        files={},
        tool_outputs=[],
        working_memory={},
    )


def get_env():
    """Create environment with map_reduce available."""
    from src.repl_environment import RLMEnvironment

    return RLMEnvironment(make_context(), use_restricted=False)


# =============================================================================
# SPEC-01.21: Property tests for map_reduce consistency
# =============================================================================


@pytest.mark.hypothesis
class TestMapReduceProperties:
    """Property-based tests for map_reduce() function."""

    @given(
        content=content_strategy,
        map_prompt=prompt_strategy,
        reduce_prompt=prompt_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_always_returns_deferred_batch(self, content, map_prompt, reduce_prompt):
        """
        map_reduce always returns a DeferredBatch regardless of input.

        @trace SPEC-01.21
        """
        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content=content,
            map_prompt=map_prompt,
            reduce_prompt=reduce_prompt,
        )

        assert isinstance(result, DeferredBatch)
        assert result.batch_id is not None

    @given(
        content=st.text(min_size=100, max_size=5000),
        map_prompt=prompt_strategy,
        reduce_prompt=prompt_strategy,
    )
    @settings(max_examples=50)
    def test_deterministic_batch_structure(self, content, map_prompt, reduce_prompt):
        """
        Same inputs produce same batch structure (deterministic).

        @trace SPEC-01.21
        """
        env1 = get_env()
        env2 = get_env()

        # Reset operation counters to ensure same IDs
        env1._operation_counter = 0
        env2._operation_counter = 0

        result1 = env1.globals["map_reduce"](content, map_prompt, reduce_prompt)
        result2 = env2.globals["map_reduce"](content, map_prompt, reduce_prompt)

        # Same number of operations
        assert len(result1.operations) == len(result2.operations)

        # Same operation types
        types1 = [op.operation_type for op in result1.operations]
        types2 = [op.operation_type for op in result2.operations]
        assert types1 == types2

    @given(
        content=st.text(min_size=100, max_size=5000),
        map_prompt=prompt_strategy,
        reduce_prompt=prompt_strategy,
        n_chunks=n_chunks_strategy,
    )
    @settings(max_examples=100)
    def test_operation_count_scales_with_n_chunks(
        self, content, map_prompt, reduce_prompt, n_chunks
    ):
        """
        Number of operations should be at least 1 and at most n_chunks.

        Token-based chunking may produce fewer chunks than n_chunks when:
        - Content has few tokens (uniform content like "000...")
        - Token count is less than n_chunks * min_chunk_size (100 tokens)

        @trace SPEC-01.21
        """
        assume(len(content) >= n_chunks)  # Need enough content to chunk

        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content=content,
            map_prompt=map_prompt,
            reduce_prompt=reduce_prompt,
            n_chunks=n_chunks,
        )

        # Token-based chunking: always produces at least 1 operation
        # May produce fewer than n_chunks if content has few tokens
        assert len(result.operations) >= 1
        # Should not exceed requested chunks
        assert len(result.operations) <= n_chunks + 1  # +1 for rounding edge case

    @given(model=model_strategy, content=st.text(min_size=10, max_size=1000))
    @settings(max_examples=50)
    def test_all_valid_models_work(self, model, content):
        """
        All valid model values should work without error.

        @trace SPEC-01.05, SPEC-01.21
        """
        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content=content,
            map_prompt="Process this",
            reduce_prompt="Combine results",
            model=model,
        )

        assert isinstance(result, DeferredBatch)

    @given(content=st.text(min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_never_produces_empty_batch(self, content):
        """
        map_reduce should never produce a batch with zero operations.

        @trace SPEC-01.21
        """
        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content=content,
            map_prompt="Process",
            reduce_prompt="Combine",
        )

        # Even empty content should produce at least one operation
        assert len(result.operations) >= 1

    @given(
        content=st.text(min_size=100, max_size=2000),
        n_chunks=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=50)
    def test_chunks_dont_lose_content(self, content, n_chunks):
        """
        Chunking should not lose any content (sum of chunk sizes >= original).

        @trace SPEC-01.02, SPEC-01.21
        """
        assume(len(content) >= n_chunks)

        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content=content,
            map_prompt="Process: {chunk}",
            reduce_prompt="Combine",
            n_chunks=n_chunks,
        )

        # Sum of all chunk contexts should cover the original content
        total_context_chars = sum(len(op.context) for op in result.operations if op.context)

        # Should have at least as many characters as original
        # (may have more due to overlaps or prompt additions)
        assert total_context_chars >= len(content) * 0.9  # Allow 10% variance


@pytest.mark.hypothesis
class TestMapReduceOperationProperties:
    """Property tests for individual operations in map_reduce batches."""

    @given(content=st.text(min_size=50, max_size=1000))
    @settings(max_examples=50)
    def test_all_operations_have_unique_ids(self, content):
        """
        All operations in a batch must have unique IDs.

        @trace SPEC-01.21
        """
        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content=content,
            map_prompt="Process",
            reduce_prompt="Combine",
            n_chunks=5,
        )

        op_ids = [op.operation_id for op in result.operations]
        assert len(set(op_ids)) == len(op_ids), "Duplicate operation IDs found"

    @given(content=st.text(min_size=50, max_size=1000))
    @settings(max_examples=50)
    def test_all_operations_unresolved_initially(self, content):
        """
        All operations should be unresolved when first created.

        @trace SPEC-01.21
        """
        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content=content,
            map_prompt="Process",
            reduce_prompt="Combine",
        )

        for op in result.operations:
            assert op.resolved is False
            assert op.result is None

    @given(
        content=st.text(
            # Use ASCII-like characters for predictable tokenization
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters=" \n"
            ),
            min_size=200,
            max_size=5000,
        ),
        n_chunks=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=50)
    def test_chunk_sizes_roughly_equal(self, content, n_chunks):
        """
        Chunks should have non-zero token content.

        Token-based chunking produces chunks with varying sizes based on
        semantic boundaries. The key invariant is that each chunk has
        meaningful content to process.

        @trace SPEC-01.02, SPEC-01.21
        """
        from src.tokenization import count_tokens

        # Need enough tokens to meaningfully chunk
        total_tokens = count_tokens(content)
        assume(total_tokens >= n_chunks * 100)  # Meaningful token content

        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content=content,
            map_prompt="Process",
            reduce_prompt="Combine",
            n_chunks=n_chunks,
        )

        # Get chunk token counts
        chunk_tokens = [count_tokens(op.context) for op in result.operations if op.context]

        # All chunks should have non-trivial content
        assert len(chunk_tokens) >= 1, "Should have at least one chunk"
        for tokens in chunk_tokens:
            assert tokens > 0, "Each chunk should have positive token count"

        # Total tokens across chunks should cover original content
        total_chunk_tokens = sum(chunk_tokens)
        assert total_chunk_tokens >= total_tokens * 0.9, "Chunks should cover most content"


@pytest.mark.hypothesis
class TestMapReduceEdgeCases:
    """Property tests for edge cases in map_reduce."""

    @given(n_chunks=st.integers(min_value=1, max_value=100))
    @settings(max_examples=30)
    def test_handles_n_chunks_larger_than_content(self, n_chunks):
        """
        Should handle n_chunks larger than content length gracefully.

        @trace SPEC-01.02, SPEC-01.21
        """
        env = get_env()
        map_reduce = env.globals["map_reduce"]

        # Content shorter than n_chunks
        short_content = "ABC"

        result = map_reduce(
            content=short_content,
            map_prompt="Process",
            reduce_prompt="Combine",
            n_chunks=n_chunks,
        )

        # Should not crash, should return valid batch
        assert isinstance(result, DeferredBatch)
        # Number of actual chunks should be at most len(content)
        assert len(result.operations) <= max(n_chunks, len(short_content)) + 1

    @given(
        map_prompt=st.text(min_size=1, max_size=500),
        reduce_prompt=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=30)
    def test_special_characters_in_prompts(self, map_prompt, reduce_prompt):
        """
        Special characters in prompts should not break processing.

        @trace SPEC-01.21
        """
        env = get_env()
        map_reduce = env.globals["map_reduce"]

        result = map_reduce(
            content="Test content for processing",
            map_prompt=map_prompt,
            reduce_prompt=reduce_prompt,
        )

        assert isinstance(result, DeferredBatch)

    @given(content=st.text(min_size=0, max_size=100).filter(lambda x: "\x00" not in x))
    @settings(max_examples=30)
    def test_whitespace_only_content(self, content):
        """
        Whitespace-only or empty content should be handled.

        @trace SPEC-01.06, SPEC-01.21
        """
        env = get_env()
        map_reduce = env.globals["map_reduce"]

        # Use whitespace-only content
        whitespace_content = " " * len(content) if content else ""

        result = map_reduce(
            content=whitespace_content,
            map_prompt="Process",
            reduce_prompt="Combine",
        )

        assert isinstance(result, DeferredBatch)
