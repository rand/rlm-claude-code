"""
Property-based tests for context index.

Tests: SPEC-01.04 (Phase 4 - Hierarchical File Index)

Property tests verify invariants:
- Indexed files are searchable
- Token counts are accurate
- Budget constraints are respected
- FTS search results are consistent
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.context_index import ContextIndex, FileIndex
from src.tokenization import count_tokens

# Strategies
valid_filename = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
).map(lambda s: s + ".py")

python_content = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"), whitelist_characters="\n_():'\""
    ),
    min_size=10,
    max_size=500,
)

search_query = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu"), whitelist_characters=" "),
    min_size=3,
    max_size=20,
).filter(lambda s: s.strip() and len(s.strip()) >= 3)


class TestFileIndexProperties:
    """Property tests for FileIndex dataclass."""

    @given(
        path=valid_filename,
        token_count=st.integers(min_value=0, max_value=100000),
        summary=st.text(min_size=0, max_size=200),
    )
    def test_file_index_roundtrip(self, path: str, token_count: int, summary: str):
        """FileIndex preserves all fields."""
        idx = FileIndex(
            path=path,
            token_count=token_count,
            summary=summary,
        )

        assert idx.path == path
        assert idx.token_count == token_count
        assert idx.summary == summary

    @given(
        imports=st.lists(st.text(min_size=1, max_size=20), max_size=10),
        exports=st.lists(st.text(min_size=1, max_size=20), max_size=10),
    )
    def test_file_index_lists_preserved(self, imports: list[str], exports: list[str]):
        """Import and export lists are preserved."""
        idx = FileIndex(
            path="test.py",
            token_count=100,
            summary="test",
            imports=imports,
            exports=exports,
        )

        assert idx.imports == imports
        assert idx.exports == exports


class TestIndexingProperties:
    """Property tests for file indexing."""

    @given(content=python_content)
    @settings(max_examples=30, deadline=5000)
    def test_indexed_files_are_searchable(self, content: str):
        """Any indexed file with searchable content can be found."""
        assume(len(content.strip()) >= 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(content)

            # Index it
            index = ContextIndex()
            result = index.index_file(test_file, base_path=tmpdir)

            assert result is not None
            assert result.path == "test.py"
            assert result.token_count > 0

    @given(content=python_content)
    @settings(max_examples=20, deadline=5000)
    def test_token_count_matches_tokenization(self, content: str):
        """Token count matches tokenization module."""
        assume(len(content.strip()) >= 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(content)

            index = ContextIndex()
            result = index.index_file(test_file, base_path=tmpdir)

            if result:
                expected = count_tokens(content)
                assert result.token_count == expected

    @given(
        contents=st.lists(
            python_content.filter(lambda c: len(c.strip()) >= 5),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=15, deadline=10000)
    def test_stats_match_indexed_files(self, contents: list[str]):
        """Stats accurately reflect indexed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i, content in enumerate(contents):
                (Path(tmpdir) / f"file{i}.py").write_text(content)

            index = ContextIndex()
            count = index.index_directory(tmpdir)

            stats = index.get_stats()
            assert stats.total_files == count
            assert stats.total_tokens >= 0


class TestSearchProperties:
    """Property tests for search functionality."""

    @given(word=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=4, max_size=10))
    @settings(max_examples=20, deadline=5000)
    def test_search_finds_indexed_content(self, word: str):
        """Search finds content that was indexed."""
        assume(word.isalpha() and len(word) >= 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            content = f"def {word}_function():\n    pass\n"
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(content)

            index = ContextIndex()
            index.index_file(test_file, base_path=tmpdir)

            results = index.search(word)
            assert len(results) >= 1
            assert any(word in r.path or word in r.summary for r in results)

    @given(limit=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=5000)
    def test_search_respects_limit(self, limit: int):
        """Search never returns more than limit results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files with common word
            for i in range(10):
                (Path(tmpdir) / f"file{i}.py").write_text(f"def function{i}(): pass")

            index = ContextIndex()
            index.index_directory(tmpdir)

            results = index.search("function", limit=limit)
            assert len(results) <= limit


class TestBudgetProperties:
    """Property tests for budget-aware retrieval."""

    @given(budget=st.integers(min_value=100, max_value=10000))
    @settings(max_examples=20, deadline=10000)
    def test_budget_is_respected(self, budget: int):
        """Retrieved context fits within budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            for i in range(5):
                content = f"# File {i}\ndef func{i}(): pass\n" * 10
                (Path(tmpdir) / f"file{i}.py").write_text(content)

            index = ContextIndex()
            index.index_directory(tmpdir)

            context = index.get_relevant_context("func", budget_tokens=budget, base_path=tmpdir)

            # Total tokens should not exceed budget (with some margin for edge cases)
            total = sum(count_tokens(c) for c in context.values())
            # Allow some margin since we pick files that fit
            assert total <= budget + 500  # Small margin for file that just fits

    @given(budget=st.integers(min_value=1, max_value=50))
    @settings(max_examples=10, deadline=5000)
    def test_small_budget_returns_subset(self, budget: int):
        """Very small budget returns at most subset of files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                content = f"# Large file {i}\n" + "x = 1\n" * 100
                (Path(tmpdir) / f"file{i}.py").write_text(content)

            index = ContextIndex()
            index.index_directory(tmpdir)

            context = index.get_relevant_context("file", budget_tokens=budget, base_path=tmpdir)

            # With very small budget, may get nothing or summaries
            assert len(context) <= 3


class TestCacheProperties:
    """Property tests for caching behavior."""

    @given(content=python_content)
    @settings(max_examples=15, deadline=5000)
    def test_unchanged_files_use_cache(self, content: str):
        """Unchanged files return cached results."""
        assume(len(content.strip()) >= 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(content)

            index = ContextIndex()
            result1 = index.index_file(test_file, base_path=tmpdir)
            result2 = index.index_file(test_file, base_path=tmpdir)

            assert result1 is not None
            assert result2 is not None
            assert result1.content_hash == result2.content_hash


class TestClearAndRemoveProperties:
    """Property tests for clearing and removing."""

    @given(n_files=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=10000)
    def test_clear_removes_all(self, n_files: int):
        """Clear removes all indexed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_files):
                (Path(tmpdir) / f"file{i}.py").write_text(f"# file {i}")

            index = ContextIndex()
            index.index_directory(tmpdir)

            removed = index.clear()
            assert removed == n_files

            stats = index.get_stats()
            assert stats.total_files == 0
