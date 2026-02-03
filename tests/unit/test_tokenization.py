"""
Tests for token-aware chunking module.

Tests: SPEC-01.01 (Phase 4 - Massive Context)
"""

from __future__ import annotations

from src.tokenization import (
    Chunk,
    ChunkingConfig,
    chunk_by_tokens,
    count_tokens,
    detect_language,
    find_semantic_boundaries,
    iter_chunks,
    partition_content_by_tokens,
    token_aware_chunk,
)


class TestCountTokens:
    """Tests for token counting."""

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        tokens = count_tokens("hello")
        assert tokens >= 1

    def test_sentence(self):
        tokens = count_tokens("Hello, world! This is a test sentence.")
        assert tokens >= 5  # At least a few tokens

    def test_code_snippet(self):
        code = "def hello():\n    return 'world'"
        tokens = count_tokens(code)
        assert tokens >= 5

    def test_unicode_content(self):
        tokens = count_tokens("こんにちは世界 🌍")
        assert tokens >= 1

    def test_large_content(self):
        content = "word " * 10000
        tokens = count_tokens(content)
        # Should be roughly 10000 tokens (one per word + spaces)
        assert 5000 <= tokens <= 20000


class TestDetectLanguage:
    """Tests for language detection."""

    def test_detect_python_function(self):
        code = """
def hello_world():
    print("Hello")
"""
        assert detect_language(code) == "python"

    def test_detect_python_class(self):
        code = """
class MyClass:
    def __init__(self):
        pass
"""
        assert detect_language(code) == "python"

    def test_detect_javascript_function(self):
        code = """
function hello() {
    console.log("Hello");
}
"""
        assert detect_language(code) == "javascript"

    def test_detect_javascript_const(self):
        code = """
const hello = () => {
    console.log("Hello");
};
"""
        assert detect_language(code) == "javascript"

    def test_detect_typescript(self):
        code = """
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}`;
}
"""
        assert detect_language(code) == "typescript"

    def test_detect_unknown_returns_none(self):
        content = "Just some plain text without any code patterns."
        assert detect_language(content) is None


class TestFindSemanticBoundaries:
    """Tests for semantic boundary detection."""

    def test_finds_python_function_boundaries(self):
        code = """def first():
    pass

def second():
    pass
"""
        boundaries = find_semantic_boundaries(code, "python")
        assert 0 in boundaries  # Start
        assert len(boundaries) >= 2  # At least start + one function

    def test_finds_python_class_boundaries(self):
        code = """class First:
    pass

class Second:
    pass
"""
        boundaries = find_semantic_boundaries(code, "python")
        assert len(boundaries) >= 2

    def test_finds_decorator_boundaries(self):
        code = """@decorator
def decorated():
    pass

def undecorated():
    pass
"""
        boundaries = find_semantic_boundaries(code, "python")
        # Should find boundary at decorator
        assert len(boundaries) >= 2

    def test_finds_javascript_boundaries(self):
        code = """function first() {
}

function second() {
}
"""
        boundaries = find_semantic_boundaries(code, "javascript")
        assert len(boundaries) >= 2

    def test_generic_paragraph_boundaries(self):
        content = """First paragraph.

Second paragraph.

Third paragraph.
"""
        boundaries = find_semantic_boundaries(content, "generic")
        assert len(boundaries) >= 1

    def test_auto_detects_language(self):
        code = """def hello():
    pass
"""
        boundaries = find_semantic_boundaries(code)  # No language specified
        assert len(boundaries) >= 1


class TestTokenAwareChunk:
    """Tests for the main chunking function."""

    def test_empty_content_returns_empty_chunk(self):
        chunks = token_aware_chunk("")
        assert chunks == [("", 0)]

    def test_small_content_single_chunk(self):
        content = "Hello, world!"
        chunks = token_aware_chunk(content, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0][0] == content

    def test_respects_max_tokens(self):
        # Create content that's definitely larger than max_tokens
        content = "word " * 5000  # ~5000 tokens
        chunks = token_aware_chunk(content, max_tokens=1000, overlap_tokens=0)

        for chunk_content, token_count in chunks:
            # Allow some tolerance for boundary adjustment
            assert token_count <= 1200, f"Chunk has {token_count} tokens, expected <= 1200"

    def test_preserves_total_content(self):
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n" * 100
        chunks = token_aware_chunk(content, max_tokens=100, overlap_tokens=0)

        # Without overlap, concatenated chunks should cover all content
        # (may have boundary adjustments, but shouldn't lose content)
        total_chars = sum(len(c) for c, _ in chunks)
        assert total_chars >= len(content) * 0.9  # At least 90% preserved

    def test_overlap_creates_more_chunks(self):
        content = "word " * 2000
        chunks_no_overlap = token_aware_chunk(content, max_tokens=500, overlap_tokens=0)
        chunks_with_overlap = token_aware_chunk(content, max_tokens=500, overlap_tokens=100)

        # With overlap, we expect more chunks
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_custom_boundary_patterns(self):
        content = "## Section 1\nContent 1\n## Section 2\nContent 2\n" * 20
        chunks = token_aware_chunk(
            content,
            max_tokens=200,
            boundary_patterns=[r"^## "],
        )
        assert len(chunks) >= 2

    def test_returns_token_counts(self):
        content = "Hello world " * 100
        chunks = token_aware_chunk(content, max_tokens=100)

        for chunk_content, token_count in chunks:
            # Token count should be positive for non-empty chunks
            if chunk_content.strip():
                assert token_count > 0

    def test_handles_code_with_semantic_boundaries(self):
        code = (
            """
def function_one():
    # Lots of code here
    pass

def function_two():
    # More code
    pass

def function_three():
    # Even more code
    pass
"""
            * 10
        )  # Make it large enough to require chunking

        chunks = token_aware_chunk(code, max_tokens=100, overlap_tokens=0)

        # Should have multiple chunks (391 tokens / 100 max = at least 4)
        assert len(chunks) >= 2

        # Each chunk should be valid Python (no mid-function cuts ideally)
        for chunk_content, _ in chunks:
            assert chunk_content  # Non-empty


class TestChunkByTokens:
    """Tests for high-level chunking API."""

    def test_returns_chunk_objects(self):
        content = "Hello world " * 100
        chunks = chunk_by_tokens(content, max_tokens=100)

        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_has_token_count(self):
        content = "Hello world " * 50
        chunks = chunk_by_tokens(content, max_tokens=100)

        for chunk in chunks:
            assert chunk.token_count > 0

    def test_chunk_has_offsets(self):
        content = "Hello world " * 50
        chunks = chunk_by_tokens(content, max_tokens=100)

        for chunk in chunks:
            assert chunk.start_offset >= 0
            assert chunk.end_offset > chunk.start_offset
            assert chunk.end_offset <= len(content) + 1000  # Allow for overlap

    def test_chunk_has_boundary_type(self):
        code = (
            """def hello():
    pass
"""
            * 20
        )

        chunks = chunk_by_tokens(code, max_tokens=100)

        for chunk in chunks:
            assert chunk.boundary_type in {"function", "class", "paragraph", "line", "none"}

    def test_detects_function_boundary(self):
        code = """def hello():
    return "world"
"""
        chunks = chunk_by_tokens(code, max_tokens=1000)
        assert chunks[0].boundary_type == "function"

    def test_detects_class_boundary(self):
        code = """class MyClass:
    pass
"""
        chunks = chunk_by_tokens(code, max_tokens=1000)
        assert chunks[0].boundary_type == "class"


class TestIterChunks:
    """Tests for lazy chunk iteration."""

    def test_yields_chunks(self):
        content = "word " * 200
        chunks = list(iter_chunks(content, max_tokens=200))
        assert len(chunks) >= 1

    def test_yields_chunk_objects(self):
        content = "word " * 200
        for chunk in iter_chunks(content, max_tokens=200):
            assert isinstance(chunk, Chunk)
            break  # Just test first


class TestPartitionContentByTokens:
    """Tests for drop-in replacement function."""

    def test_empty_content(self):
        result = partition_content_by_tokens("", 5)
        assert result == [""]

    def test_single_chunk_request(self):
        content = "Hello, world!"
        result = partition_content_by_tokens(content, 1)
        assert len(result) >= 1
        assert "".join(result).strip() == content.strip()

    def test_multiple_chunks(self):
        content = "word " * 200
        result = partition_content_by_tokens(content, 5)
        assert len(result) >= 1

    def test_handles_n_chunks_larger_than_content(self):
        content = "hi"
        result = partition_content_by_tokens(content, 100)
        assert len(result) >= 1
        assert result[0] == content

    def test_preserves_content(self):
        content = "Line 1\nLine 2\nLine 3\n" * 30
        result = partition_content_by_tokens(content, 5)

        # Concatenated result should have most of the original content
        combined = "".join(result)
        # Allow some variance due to boundary adjustments
        assert len(combined) >= len(content) * 0.8


class TestChunkingConfig:
    """Tests for chunking configuration."""

    def test_default_config(self):
        config = ChunkingConfig()
        assert config.max_tokens == 4000
        assert config.overlap_tokens == 200
        assert config.prefer_semantic_boundaries is True

    def test_custom_config(self):
        config = ChunkingConfig(
            max_tokens=2000,
            overlap_tokens=100,
            language="python",
        )
        assert config.max_tokens == 2000
        assert config.overlap_tokens == 100
        assert config.language == "python"

    def test_default_boundary_patterns_populated(self):
        config = ChunkingConfig()
        assert "python" in config.boundary_patterns
        assert "javascript" in config.boundary_patterns
        assert "generic" in config.boundary_patterns


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_line_no_newlines(self):
        # Single very long line
        content = "x" * 10000
        chunks = token_aware_chunk(content, max_tokens=1000)
        assert len(chunks) >= 1
        for _, token_count in chunks:
            assert token_count <= 1500  # Some tolerance

    def test_only_newlines(self):
        content = "\n" * 100
        chunks = token_aware_chunk(content, max_tokens=10)
        assert len(chunks) >= 1

    def test_mixed_languages(self):
        content = """
# Python
def hello():
    pass

// JavaScript
function world() {
}
"""
        chunks = token_aware_chunk(content, max_tokens=50)
        assert len(chunks) >= 1

    def test_binary_like_content(self):
        # Content with lots of special characters
        content = "".join(chr(i) for i in range(32, 127)) * 20
        chunks = token_aware_chunk(content, max_tokens=500)
        assert len(chunks) >= 1

    def test_unicode_heavy_content(self):
        content = "日本語テスト " * 100
        chunks = token_aware_chunk(content, max_tokens=500)
        assert len(chunks) >= 1


class TestPropertyInvariants:
    """Property-based invariant tests (simpler than full Hypothesis)."""

    def test_token_count_accuracy(self):
        """Token count in result matches actual count."""
        content = "Hello world! " * 100
        chunks = token_aware_chunk(content, max_tokens=200)

        for chunk_content, reported_tokens in chunks:
            actual_tokens = count_tokens(chunk_content)
            # Allow small variance due to encoding quirks
            assert abs(actual_tokens - reported_tokens) <= 5

    def test_chunks_not_empty_for_nonempty_input(self):
        """Non-empty input produces non-empty chunks."""
        content = "Some content here."
        chunks = token_aware_chunk(content, max_tokens=1000)
        assert len(chunks) >= 1
        assert any(c.strip() for c, _ in chunks)

    def test_chunk_offsets_valid(self):
        """Chunk offsets are within bounds."""
        content = "Test content " * 50
        chunks = chunk_by_tokens(content, max_tokens=100)

        for chunk in chunks:
            assert chunk.start_offset >= 0
            assert chunk.start_offset < chunk.end_offset
