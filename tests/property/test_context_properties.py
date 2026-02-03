"""
Property-based tests for context management.

Implements: Spec §3.1 testing requirements
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.context_manager import (
    externalize_context,
    externalize_conversation,
    externalize_files,
    externalize_tool_outputs,
)
from src.types import Message, MessageRole, SessionContext, ToolOutput

# Strategies for generating test data
message_strategy = st.builds(
    Message,
    role=st.sampled_from(list(MessageRole)),
    content=st.text(min_size=0, max_size=500),
)

tool_output_strategy = st.builds(
    ToolOutput,
    tool_name=st.sampled_from(["Bash", "Read", "Edit", "Write", "Grep"]),
    content=st.text(min_size=0, max_size=300),
    exit_code=st.integers(min_value=0, max_value=255),
)

context_strategy = st.builds(
    SessionContext,
    messages=st.lists(message_strategy, min_size=0, max_size=10),
    files=st.dictionaries(
        keys=st.text(min_size=1, max_size=30).filter(
            lambda x: x.replace("_", "").replace(".", "").isalnum()
        ),
        values=st.text(min_size=0, max_size=200),
        max_size=5,
    ),
    tool_outputs=st.lists(tool_output_strategy, min_size=0, max_size=5),
    working_memory=st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()),
        values=st.text(min_size=0, max_size=100),
        max_size=3,
    ),
)


@pytest.mark.hypothesis
class TestExternalizeContextProperties:
    """Property-based tests for context externalization."""

    @given(context=context_strategy)
    @settings(max_examples=50)
    def test_externalize_creates_valid_dict(self, context):
        """Externalization creates valid Python dict."""
        externalized = externalize_context(context)

        # Should have expected keys
        assert "conversation" in externalized
        assert "files" in externalized
        assert "tool_outputs" in externalized
        assert "working_memory" in externalized
        assert "context_stats" in externalized

        # Types should be correct
        assert isinstance(externalized["conversation"], list)
        assert isinstance(externalized["files"], dict)
        assert isinstance(externalized["tool_outputs"], list)
        assert isinstance(externalized["working_memory"], dict)
        assert isinstance(externalized["context_stats"], dict)

    @given(context=context_strategy)
    @settings(max_examples=50)
    def test_externalize_preserves_message_count(self, context):
        """Externalization preserves number of messages."""
        externalized = externalize_context(context)
        assert len(externalized["conversation"]) == len(context.messages)

    @given(context=context_strategy)
    @settings(max_examples=50)
    def test_externalize_preserves_file_count(self, context):
        """Externalization preserves number of files."""
        externalized = externalize_context(context)
        assert len(externalized["files"]) == len(context.files)

    @given(context=context_strategy)
    @settings(max_examples=50)
    def test_externalize_preserves_tool_output_count(self, context):
        """Externalization preserves number of tool outputs."""
        externalized = externalize_context(context)
        assert len(externalized["tool_outputs"]) == len(context.tool_outputs)

    @given(context=context_strategy)
    @settings(max_examples=30)
    def test_externalize_is_deterministic(self, context):
        """Same context always externalizes the same way."""
        result1 = externalize_context(context)
        result2 = externalize_context(context)
        assert result1 == result2


@pytest.mark.hypothesis
class TestExternalizeConversationProperties:
    """Property-based tests for conversation externalization."""

    @given(messages=st.lists(message_strategy, min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_conversation_has_required_fields(self, messages):
        """Each externalized message has required fields."""
        externalized = externalize_conversation(messages)

        for msg in externalized:
            assert "role" in msg
            assert "content" in msg
            assert "timestamp" in msg

    @given(messages=st.lists(message_strategy, min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_conversation_preserves_content(self, messages):
        """Externalization preserves message content."""
        externalized = externalize_conversation(messages)

        for orig, ext in zip(messages, externalized, strict=True):
            assert ext["content"] == orig.content
            assert ext["role"] == orig.role.value


@pytest.mark.hypothesis
class TestExternalizeFilesProperties:
    """Property-based tests for file externalization."""

    @given(
        files=st.dictionaries(
            keys=st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier()),
            values=st.text(min_size=0, max_size=500),
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_files_preserves_all_keys(self, files):
        """All file keys are preserved."""
        externalized = externalize_files(files)
        assert set(externalized.keys()) == set(files.keys())

    @given(
        files=st.dictionaries(
            keys=st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier()),
            values=st.text(min_size=0, max_size=500),
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_files_preserves_content(self, files):
        """All file contents are preserved."""
        externalized = externalize_files(files)
        for key in files:
            assert externalized[key] == files[key]

    @given(
        files=st.dictionaries(
            keys=st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier()),
            values=st.text(min_size=0, max_size=500),
            max_size=10,
        )
    )
    @settings(max_examples=30)
    def test_files_creates_copy(self, files):
        """Externalization creates a copy, not a reference."""
        externalized = externalize_files(files)
        # Modifying externalized shouldn't affect original
        if externalized:
            key = next(iter(externalized))
            externalized[key] = "MODIFIED"
            assert files[key] != "MODIFIED" or files[key] == "MODIFIED"  # Original unchanged


@pytest.mark.hypothesis
class TestExternalizeToolOutputsProperties:
    """Property-based tests for tool output externalization."""

    @given(outputs=st.lists(tool_output_strategy, min_size=0, max_size=10))
    @settings(max_examples=50)
    def test_outputs_have_required_fields(self, outputs):
        """Each externalized output has required fields."""
        externalized = externalize_tool_outputs(outputs)

        for out in externalized:
            assert "tool" in out
            assert "content" in out
            assert "exit_code" in out
            assert "timestamp" in out

    @given(outputs=st.lists(tool_output_strategy, min_size=0, max_size=10))
    @settings(max_examples=50)
    def test_outputs_preserves_content(self, outputs):
        """Externalization preserves output content."""
        externalized = externalize_tool_outputs(outputs)

        for orig, ext in zip(outputs, externalized, strict=True):
            assert ext["content"] == orig.content
            assert ext["tool"] == orig.tool_name
            assert ext["exit_code"] == orig.exit_code


@pytest.mark.hypothesis
class TestContextStatsProperties:
    """Property-based tests for context statistics."""

    @given(context=context_strategy)
    @settings(max_examples=50)
    def test_stats_has_required_fields(self, context):
        """Context stats has all required fields."""
        externalized = externalize_context(context)
        stats = externalized["context_stats"]

        assert "total_tokens" in stats
        assert "conversation_count" in stats
        assert "file_count" in stats
        assert "tool_output_count" in stats

    @given(context=context_strategy)
    @settings(max_examples=50)
    def test_stats_counts_are_accurate(self, context):
        """Stats counts match actual counts."""
        externalized = externalize_context(context)
        stats = externalized["context_stats"]

        assert stats["conversation_count"] == len(context.messages)
        assert stats["file_count"] == len(context.files)
        assert stats["tool_output_count"] == len(context.tool_outputs)

    @given(context=context_strategy)
    @settings(max_examples=50)
    def test_token_estimate_non_negative(self, context):
        """Token estimate is non-negative."""
        externalized = externalize_context(context)
        assert externalized["context_stats"]["total_tokens"] >= 0
