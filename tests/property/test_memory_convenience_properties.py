"""
Property-based tests for MemoryStore convenience methods.

Tests: API Surface Improvements (add_fact, add_experience, add_entity, find)

Property tests verify invariants:
- Convenience methods create nodes with correct types
- Parameters are properly passed through
- Search results match created content
- Confidence values are bounded [0, 1]
"""

from __future__ import annotations

import os
import tempfile

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.memory_store import MemoryStore

# Strategies
content_text = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")),
    min_size=5,
    max_size=200,
).filter(lambda s: s.strip())

confidence_value = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

tier_value = st.sampled_from(["task", "session", "longterm", "archive"])

outcome_text = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")),
    min_size=3,
    max_size=50,
).filter(lambda s: s.strip())

entity_type_value = st.sampled_from(["class", "function", "module", "variable", None])


def create_temp_store() -> tuple[MemoryStore, str]:
    """Create a fresh MemoryStore with a temp database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = MemoryStore(db_path=path)
    return store, path


def cleanup_store(path: str) -> None:
    """Clean up temporary database files."""
    for suffix in ["", "-wal", "-shm"]:
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


class TestAddFactProperties:
    """Property tests for add_fact convenience method."""

    @given(content=content_text)
    @settings(max_examples=50)
    def test_add_fact_creates_fact_node(self, content: str):
        """add_fact always creates a node with type='fact'."""
        store, path = create_temp_store()
        try:
            node_id = store.add_fact(content)
            node = store.get_node(node_id)
            assert node is not None
            assert node.type == "fact"
            assert node.content == content
        finally:
            cleanup_store(path)

    @given(content=content_text, confidence=confidence_value)
    @settings(max_examples=50)
    def test_add_fact_preserves_confidence(self, content: str, confidence: float):
        """add_fact preserves the confidence value."""
        store, path = create_temp_store()
        try:
            node_id = store.add_fact(content, confidence=confidence)
            node = store.get_node(node_id)
            assert node.confidence == confidence
        finally:
            cleanup_store(path)

    @given(content=content_text, tier=tier_value)
    @settings(max_examples=50)
    def test_add_fact_preserves_tier(self, content: str, tier: str):
        """add_fact preserves the tier value."""
        store, path = create_temp_store()
        try:
            node_id = store.add_fact(content, tier=tier)
            # include_archived=True to retrieve archived nodes
            node = store.get_node(node_id, include_archived=True)
            assert node is not None
            assert node.tier == tier
        finally:
            cleanup_store(path)

    @given(content=content_text)
    @settings(max_examples=30)
    def test_add_fact_default_confidence(self, content: str):
        """add_fact uses default confidence of 0.8."""
        store, path = create_temp_store()
        try:
            node_id = store.add_fact(content)
            node = store.get_node(node_id)
            assert node.confidence == 0.8
        finally:
            cleanup_store(path)


class TestAddExperienceProperties:
    """Property tests for add_experience convenience method."""

    @given(content=content_text, outcome=outcome_text)
    @settings(max_examples=50)
    def test_add_experience_creates_experience_node(self, content: str, outcome: str):
        """add_experience always creates a node with type='experience'."""
        store, path = create_temp_store()
        try:
            node_id = store.add_experience(content, outcome=outcome)
            node = store.get_node(node_id)
            assert node is not None
            assert node.type == "experience"
            assert node.content == content
        finally:
            cleanup_store(path)

    @given(content=content_text, outcome=outcome_text)
    @settings(max_examples=50)
    def test_add_experience_stores_outcome(self, content: str, outcome: str):
        """add_experience stores outcome in metadata."""
        store, path = create_temp_store()
        try:
            node_id = store.add_experience(content, outcome=outcome)
            node = store.get_node(node_id)
            assert node.metadata["outcome"] == outcome
        finally:
            cleanup_store(path)

    @given(content=content_text, outcome=outcome_text, success=st.booleans())
    @settings(max_examples=50)
    def test_add_experience_stores_success(self, content: str, outcome: str, success: bool):
        """add_experience stores success flag in metadata."""
        store, path = create_temp_store()
        try:
            node_id = store.add_experience(content, outcome=outcome, success=success)
            node = store.get_node(node_id)
            assert node.metadata["success"] == success
        finally:
            cleanup_store(path)

    @given(content=content_text, outcome=outcome_text, confidence=confidence_value)
    @settings(max_examples=50)
    def test_add_experience_preserves_confidence(
        self, content: str, outcome: str, confidence: float
    ):
        """add_experience preserves confidence value."""
        store, path = create_temp_store()
        try:
            node_id = store.add_experience(content, outcome=outcome, confidence=confidence)
            node = store.get_node(node_id)
            assert node.confidence == confidence
        finally:
            cleanup_store(path)


class TestAddEntityProperties:
    """Property tests for add_entity convenience method."""

    @given(name=content_text)
    @settings(max_examples=50)
    def test_add_entity_creates_entity_node(self, name: str):
        """add_entity always creates a node with type='entity'."""
        store, path = create_temp_store()
        try:
            node_id = store.add_entity(name)
            node = store.get_node(node_id)
            assert node is not None
            assert node.type == "entity"
            assert node.content == name
        finally:
            cleanup_store(path)

    @given(name=content_text, entity_type=entity_type_value)
    @settings(max_examples=50)
    def test_add_entity_stores_entity_type(self, name: str, entity_type: str | None):
        """add_entity stores entity_type in metadata when provided."""
        store, path = create_temp_store()
        try:
            node_id = store.add_entity(name, entity_type=entity_type)
            node = store.get_node(node_id)
            if entity_type is not None:
                assert node.metadata["entity_type"] == entity_type
        finally:
            cleanup_store(path)


class TestFindProperties:
    """Property tests for find convenience method."""

    @given(content=content_text)
    @settings(max_examples=30)
    def test_find_returns_created_content(self, content: str):
        """find() can retrieve content that was just added."""
        assume(len(content.split()) >= 1)  # Need at least one word

        store, path = create_temp_store()
        try:
            store.add_fact(content)

            # Search for the first word
            first_word = content.split()[0]
            assume(len(first_word) >= 3)  # FTS needs reasonable query

            results = store.find(first_word, k=10)

            # Should find at least one result
            assert len(results) >= 1
            # The content should be in results
            found_contents = [r.content for r in results]
            assert content in found_contents
        finally:
            cleanup_store(path)

    @given(k=st.integers(min_value=1, max_value=100))
    @settings(max_examples=30)
    def test_find_respects_k_limit(self, k: int):
        """find() never returns more than k results."""
        store, path = create_temp_store()
        try:
            # Create more nodes than k
            for i in range(k + 10):
                store.add_fact(f"Test content number {i} for search")

            results = store.find("content", k=k)
            assert len(results) <= k
        finally:
            cleanup_store(path)

    @given(
        content=content_text,
        confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_find_respects_min_confidence(self, content: str, confidence: float):
        """find() filters by min_confidence."""
        store, path = create_temp_store()
        try:
            # Create low and high confidence nodes
            store.add_fact(f"low {content}", confidence=0.3)
            store.add_fact(f"high {content}", confidence=0.9)

            # Search with min_confidence filter
            assume(len(content.split()) >= 1)
            first_word = content.split()[0]
            assume(len(first_word) >= 3)

            results = store.find(first_word, k=10, min_confidence=confidence)

            # All results should have confidence >= min_confidence
            for result in results:
                node = store.get_node(result.node_id)
                assert node.confidence >= confidence
        finally:
            cleanup_store(path)


class TestExecutionResultFormatOutput:
    """Property tests for ExecutionResult.format_output()."""

    @given(
        output=st.one_of(
            st.none(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(min_size=0, max_size=100),
            st.lists(st.integers(), max_size=10),
            st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=5),
        )
    )
    @settings(max_examples=100)
    def test_format_output_never_crashes(self, output):
        """format_output() handles any output type without crashing."""
        from src.types import ExecutionResult

        result = ExecutionResult(success=True, output=output)
        formatted = result.format_output()

        # Should return a string
        assert isinstance(formatted, str)

    @given(max_length=st.integers(min_value=10, max_value=10000))
    @settings(max_examples=50)
    def test_format_output_respects_max_length(self, max_length: int):
        """format_output() respects max_length parameter."""
        from src.types import ExecutionResult

        # Create a large output
        large_output = {"data": "x" * 5000}
        result = ExecutionResult(success=True, output=large_output)

        formatted = result.format_output(max_length=max_length)

        # Output should be truncated if needed
        # (may be slightly longer due to truncation message)
        assert len(formatted) <= max_length + 100

    @given(stdout=st.text(min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_format_output_includes_stdout(self, stdout: str):
        """format_output() includes stdout when present."""
        from src.types import ExecutionResult

        result = ExecutionResult(success=True, output=42, stdout=stdout)
        formatted = result.format_output()

        # If stdout is non-empty, it should appear in output
        if stdout.strip():
            assert stdout in formatted or "[stdout]" in formatted
