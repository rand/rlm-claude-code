"""
Unit tests for context_manager module.

Implements: Spec §3 tests
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.context_manager import (
    create_micro_context,
    externalize_context,
    externalize_conversation,
    externalize_files,
    externalize_tool_outputs,
)
from src.types import Message, MessageRole, SessionContext, ToolOutput


class TestExternalizeConversation:
    """Tests for externalize_conversation function."""

    def test_empty_conversation(self):
        """Empty message list returns empty list."""
        result = externalize_conversation([])

        assert result == []

    def test_single_message(self):
        """Single message is externalized correctly."""
        messages = [Message(role=MessageRole.USER, content="Hello")]

        result = externalize_conversation(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[0]["timestamp"] is None

    def test_multiple_messages(self):
        """Multiple messages maintain order."""
        messages = [
            Message(role=MessageRole.USER, content="Question"),
            Message(role=MessageRole.ASSISTANT, content="Answer"),
            Message(role=MessageRole.USER, content="Follow-up"),
        ]

        result = externalize_conversation(messages)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["content"] == "Follow-up"

    def test_preserves_timestamp(self):
        """Timestamp is preserved in output."""
        messages = [Message(role=MessageRole.USER, content="Hi", timestamp=12345.0)]

        result = externalize_conversation(messages)

        assert result[0]["timestamp"] == 12345.0


class TestExternalizeFiles:
    """Tests for externalize_files function."""

    def test_empty_files(self):
        """Empty dict returns empty dict."""
        result = externalize_files({})

        assert result == {}

    def test_single_file(self):
        """Single file is copied correctly."""
        files = {"test.py": "print('hello')"}

        result = externalize_files(files)

        assert result == {"test.py": "print('hello')"}

    def test_files_are_copied(self):
        """Files dict is copied, not referenced."""
        files = {"test.py": "code"}

        result = externalize_files(files)
        result["new.py"] = "new code"

        assert "new.py" not in files


class TestExternalizeToolOutputs:
    """Tests for externalize_tool_outputs function."""

    def test_empty_outputs(self):
        """Empty list returns empty list."""
        result = externalize_tool_outputs([])

        assert result == []

    def test_single_output(self):
        """Single output is externalized correctly."""
        outputs = [ToolOutput(tool_name="bash", content="output", exit_code=0)]

        result = externalize_tool_outputs(outputs)

        assert len(result) == 1
        assert result[0]["tool"] == "bash"
        assert result[0]["content"] == "output"
        assert result[0]["exit_code"] == 0

    def test_multiple_outputs(self):
        """Multiple outputs maintain order."""
        outputs = [
            ToolOutput(tool_name="bash", content="first"),
            ToolOutput(tool_name="read", content="second"),
        ]

        result = externalize_tool_outputs(outputs)

        assert len(result) == 2
        assert result[0]["tool"] == "bash"
        assert result[1]["tool"] == "read"


class TestExternalizeContext:
    """Tests for externalize_context function."""

    def test_empty_context(self):
        """Empty context produces expected structure."""
        ctx = SessionContext()

        result = externalize_context(ctx)

        assert "conversation" in result
        assert "files" in result
        assert "tool_outputs" in result
        assert "working_memory" in result
        assert "context_stats" in result

    def test_context_stats(self):
        """Context stats are computed correctly."""
        ctx = SessionContext(
            messages=[Message(role=MessageRole.USER, content="test")],
            files={"a.py": "code", "b.py": "more"},
            tool_outputs=[ToolOutput(tool_name="bash", content="out")],
        )

        result = externalize_context(ctx)
        stats = result["context_stats"]

        assert stats["conversation_count"] == 1
        assert stats["file_count"] == 2
        assert stats["tool_output_count"] == 1
        assert stats["total_tokens"] > 0

    def test_working_memory_copied(self):
        """Working memory is copied, not referenced."""
        ctx = SessionContext(working_memory={"key": "value"})

        result = externalize_context(ctx)
        result["working_memory"]["new_key"] = "new_value"

        assert "new_key" not in ctx.working_memory

    def test_full_context_externalization(self):
        """Full context is externalized correctly."""
        ctx = SessionContext(
            messages=[
                Message(role=MessageRole.USER, content="Hello"),
                Message(role=MessageRole.ASSISTANT, content="Hi"),
            ],
            files={"main.py": "print('hello')"},
            tool_outputs=[ToolOutput(tool_name="bash", content="Hello")],
            working_memory={"state": "active"},
        )

        result = externalize_context(ctx)

        assert len(result["conversation"]) == 2
        assert "main.py" in result["files"]
        assert len(result["tool_outputs"]) == 1
        assert result["working_memory"]["state"] == "active"


class TestMicroModeContext:
    """Tests for MicroModeContext (SPEC-14.40-14.44)."""

    @pytest.fixture
    def session_context(self):
        """Create a session context for testing."""
        return SessionContext(
            messages=[Message(role=MessageRole.USER, content="test query")],
            files={"app.py": "print('hello')"},
            tool_outputs=[],
        )

    def test_create_micro_context(self, session_context):
        """Can create micro context with query and session context."""
        ctx = create_micro_context(
            query="What does this do?",
            session_context=session_context,
        )

        assert ctx.query == "What does this do?"
        assert ctx.prior_result is None

    def test_micro_context_query_accessible(self, session_context):
        """Query is directly accessible (SPEC-14.41)."""
        ctx = create_micro_context(query="my query", session_context=session_context)

        assert ctx.query == "my query"

    def test_micro_context_lazy_loads_context(self, session_context):
        """Context is lazy loaded (SPEC-14.44)."""
        ctx = create_micro_context(query="test", session_context=session_context)

        # Not loaded yet
        assert not ctx.is_context_loaded

        # Access context
        _ = ctx.context

        # Now loaded
        assert ctx.is_context_loaded

    def test_micro_context_context_has_expected_keys(self, session_context):
        """Externalized context has expected structure (SPEC-14.41)."""
        ctx = create_micro_context(query="test", session_context=session_context)

        context = ctx.context

        assert "conversation" in context
        assert "files" in context
        assert "tool_outputs" in context
        assert "working_memory" in context

    def test_micro_context_memory_with_loader(self, session_context):
        """Memory is loaded via provided loader (SPEC-14.41)."""
        memory_facts = [{"fact": "important", "confidence": 0.9}]

        ctx = create_micro_context(
            query="test",
            session_context=session_context,
            memory_loader=lambda: memory_facts,
        )

        assert ctx.memory == memory_facts

    def test_micro_context_memory_lazy_loaded(self, session_context):
        """Memory is lazy loaded (SPEC-14.44)."""
        load_count = [0]

        def memory_loader():
            load_count[0] += 1
            return [{"fact": "test"}]

        ctx = create_micro_context(
            query="test",
            session_context=session_context,
            memory_loader=memory_loader,
        )

        # Not loaded yet
        assert not ctx.is_memory_loaded
        assert load_count[0] == 0

        # Access memory
        _ = ctx.memory

        # Now loaded (only once)
        assert ctx.is_memory_loaded
        assert load_count[0] == 1

        # Second access doesn't reload
        _ = ctx.memory
        assert load_count[0] == 1

    def test_micro_context_memory_without_loader(self, session_context):
        """Memory is empty list without loader."""
        ctx = create_micro_context(query="test", session_context=session_context)

        assert ctx.memory == []

    def test_micro_context_prior_result(self, session_context):
        """Prior result is accessible (SPEC-14.41)."""
        ctx = create_micro_context(
            query="test",
            session_context=session_context,
            prior_result="previous answer",
        )

        assert ctx.prior_result == "previous answer"

    def test_micro_context_to_repl_vars(self, session_context):
        """to_repl_vars provides all expected variables (SPEC-14.40)."""
        ctx = create_micro_context(
            query="test query",
            session_context=session_context,
            memory_loader=lambda: [{"fact": "test"}],
            prior_result="prior",
        )

        repl_vars = ctx.to_repl_vars()

        assert "query" in repl_vars
        assert "context" in repl_vars
        assert "memory" in repl_vars
        assert "prior_result" in repl_vars

        assert repl_vars["query"] == "test query"
        assert repl_vars["prior_result"] == "prior"
        assert isinstance(repl_vars["context"], dict)
        assert isinstance(repl_vars["memory"], list)

    def test_micro_context_prior_result_default_empty_string(self, session_context):
        """Prior result defaults to empty string in repl vars."""
        ctx = create_micro_context(query="test", session_context=session_context)

        repl_vars = ctx.to_repl_vars()

        assert repl_vars["prior_result"] == ""


class TestMicroModeContextPerformance:
    """Performance tests for micro mode context (SPEC-14.43)."""

    def test_externalization_overhead_under_100ms(self):
        """Externalization overhead is under 100ms (SPEC-14.43)."""
        import time

        # Create a moderate-sized context
        session_context = SessionContext(
            messages=[Message(role=MessageRole.USER, content="x" * 1000) for _ in range(10)],
            files={f"file{i}.py": "y" * 1000 for i in range(10)},
            tool_outputs=[ToolOutput(tool_name="bash", content="z" * 1000) for _ in range(5)],
        )

        start = time.perf_counter()
        ctx = create_micro_context(query="test", session_context=session_context)
        _ = ctx.to_repl_vars()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be well under 100ms
        assert elapsed_ms < 100, f"Externalization took {elapsed_ms:.2f}ms, expected <100ms"
