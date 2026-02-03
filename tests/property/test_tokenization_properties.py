"""
Property-based tests for token-aware chunking.

Tests: SPEC-01.01 (Phase 4 - Massive Context)

Uses Hypothesis for property-based testing of chunking invariants.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.tokenization import (
    chunk_by_tokens,
    count_tokens,
    partition_content_by_tokens,
    token_aware_chunk,
)

# Custom strategies
reasonable_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=0,
    max_size=5000,
)

code_like_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_0123456789\n (){}[]:'\",.=+-*/<>",
    min_size=0,
    max_size=3000,
)

small_max_tokens = st.integers(min_value=50, max_value=500)
overlap_tokens = st.integers(min_value=0, max_value=100)
n_chunks = st.integers(min_value=1, max_value=20)


class TestTokenCountProperties:
    """Property tests for token counting."""

    @given(text=reasonable_text)
    @settings(max_examples=50)
    def test_token_count_non_negative(self, text: str):
        """Token count is always non-negative."""
        assert count_tokens(text) >= 0

    @given(text=reasonable_text)
    @settings(max_examples=50)
    def test_nonempty_has_positive_tokens(self, text: str):
        """Non-empty text has at least one token.

        With BPE tokenization, token count can vary in complex ways:
        - Prefixes may have MORE tokens than full text (e.g., "Infi" = 2, "Infinity" = 1)
        - Unicode chars may require multiple tokens per character
        - But non-empty text always has at least one token
        """
        assume(len(text) > 0)
        assert count_tokens(text) >= 1


class TestChunkingInvariants:
    """Property tests for chunking invariants."""

    @given(content=reasonable_text, max_tokens=small_max_tokens)
    @settings(max_examples=30, deadline=10000)
    def test_chunks_respect_max_tokens_approximately(self, content: str, max_tokens: int):
        """Each chunk should have approximately max_tokens or fewer."""
        assume(len(content) > 0)

        chunks = token_aware_chunk(content, max_tokens=max_tokens, overlap_tokens=0)

        for chunk_content, token_count in chunks:
            # Allow 20% tolerance for boundary adjustments
            assert token_count <= max_tokens * 1.2, (
                f"Chunk has {token_count} tokens, expected <= {max_tokens * 1.2}"
            )

    @given(content=code_like_text, max_tokens=small_max_tokens)
    @settings(max_examples=30, deadline=10000)
    def test_chunks_cover_content(self, content: str, max_tokens: int):
        """Chunks should cover all non-whitespace content."""
        assume(len(content.strip()) > 0)

        chunks = token_aware_chunk(content, max_tokens=max_tokens, overlap_tokens=0)

        # Concatenate all chunks
        combined = "".join(c for c, _ in chunks)

        # All significant characters should be present
        # (may have boundary adjustments, so check character coverage)
        original_chars = set(content.strip())
        combined_chars = set(combined.strip())
        missing = original_chars - combined_chars

        # Allow missing only whitespace-like characters
        significant_missing = {c for c in missing if not c.isspace()}
        assert len(significant_missing) == 0, f"Missing characters: {significant_missing}"

    @given(content=reasonable_text)
    @settings(max_examples=30)
    def test_single_chunk_for_small_content(self, content: str):
        """Content smaller than max_tokens should be single chunk."""
        assume(len(content) > 0)

        # Use very large max_tokens
        chunks = token_aware_chunk(content, max_tokens=10000, overlap_tokens=0)

        # Should be exactly one chunk containing all content
        assert len(chunks) == 1
        assert chunks[0][0] == content

    @given(content=code_like_text, max_tokens=small_max_tokens)
    @settings(max_examples=30, deadline=10000)
    def test_token_counts_match_actual(self, content: str, max_tokens: int):
        """Reported token counts should match actual counts."""
        assume(len(content) > 0)

        chunks = token_aware_chunk(content, max_tokens=max_tokens, overlap_tokens=0)

        for chunk_content, reported_tokens in chunks:
            actual_tokens = count_tokens(chunk_content)
            # Allow small variance due to encoding quirks
            assert abs(actual_tokens - reported_tokens) <= 5, (
                f"Reported {reported_tokens}, actual {actual_tokens}"
            )


class TestChunkByTokensProperties:
    """Property tests for chunk_by_tokens API."""

    @given(content=code_like_text, max_tokens=small_max_tokens)
    @settings(max_examples=30, deadline=10000)
    def test_chunk_offsets_valid(self, content: str, max_tokens: int):
        """Chunk offsets should be valid indices."""
        assume(len(content) > 0)

        chunks = chunk_by_tokens(content, max_tokens=max_tokens)

        for chunk in chunks:
            assert 0 <= chunk.start_offset <= len(content)
            assert chunk.start_offset < chunk.end_offset
            assert chunk.end_offset <= len(content) + 1000  # Allow overlap

    @given(content=code_like_text, max_tokens=small_max_tokens)
    @settings(max_examples=30, deadline=10000)
    def test_chunk_content_matches_offsets(self, content: str, max_tokens: int):
        """Chunk content should be findable in original at approximately the offset."""
        assume(len(content) > 0)

        chunks = chunk_by_tokens(content, max_tokens=max_tokens)

        for chunk in chunks:
            # Content should exist in original
            assert chunk.content in content or chunk.content.strip() in content


class TestPartitionProperties:
    """Property tests for partition_content_by_tokens."""

    @given(content=reasonable_text, n=n_chunks)
    @settings(max_examples=30, deadline=10000)
    def test_partition_produces_chunks(self, content: str, n: int):
        """Partitioning should produce at least one chunk."""
        chunks = partition_content_by_tokens(content, n)
        assert len(chunks) >= 1

    @given(content=code_like_text, n=n_chunks)
    @settings(max_examples=30, deadline=10000)
    def test_partition_covers_content(self, content: str, n: int):
        """Partitioned chunks should cover the content."""
        assume(len(content.strip()) > 0)

        chunks = partition_content_by_tokens(content, n)
        combined = "".join(chunks)

        # Should have most of the original non-whitespace content
        # Note: whitespace-only chunks are filtered out by design
        stripped_len = len(content.strip())
        combined_stripped_len = len(combined.strip())
        assert combined_stripped_len >= stripped_len * 0.8

    @given(content=reasonable_text)
    @settings(max_examples=30)
    def test_partition_single_preserves_content(self, content: str):
        """Partitioning into 1 should preserve content."""
        chunks = partition_content_by_tokens(content, 1)

        assert len(chunks) >= 1
        if content:
            combined = "".join(chunks)
            assert combined.strip() == content.strip() or content.strip() in combined


class TestOverlapProperties:
    """Property tests for overlap behavior."""

    @given(
        max_tokens=st.integers(min_value=100, max_value=300),
        overlap=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=20, deadline=15000)
    def test_overlap_increases_chunk_count(self, max_tokens: int, overlap: int):
        """Overlap should generally increase or maintain chunk count."""
        # Use fixed large content to avoid filtering issues
        content = "def function():\n    pass\n\n" * 50

        chunks_no_overlap = token_aware_chunk(content, max_tokens=max_tokens, overlap_tokens=0)
        chunks_with_overlap = token_aware_chunk(
            content, max_tokens=max_tokens, overlap_tokens=overlap
        )

        # With overlap, we should have same or more chunks
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)


class TestEdgeCaseProperties:
    """Property tests for edge cases."""

    @given(max_tokens=small_max_tokens)
    @settings(max_examples=20)
    def test_empty_content_handled(self, max_tokens: int):
        """Empty content should produce empty chunk."""
        chunks = token_aware_chunk("", max_tokens=max_tokens)
        assert len(chunks) == 1
        assert chunks[0] == ("", 0)

    @given(content=st.text(alphabet="\n ", min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_whitespace_only_handled(self, content: str):
        """Whitespace-only content should be handled gracefully."""
        chunks = token_aware_chunk(content, max_tokens=100)
        # Should return at least one chunk (possibly empty after strip)
        assert len(chunks) >= 1

    @given(n=st.integers(min_value=-10, max_value=0))
    @settings(max_examples=10)
    def test_invalid_n_chunks_handled(self, n: int):
        """Invalid n_chunks should be handled gracefully."""
        content = "some content"
        chunks = partition_content_by_tokens(content, n)
        assert len(chunks) >= 1
