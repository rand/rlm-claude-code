"""
Unit tests for types module.

Implements: Spec §3.1 tests
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.types import (
    ActivationDecision,
    ExecutionResult,
    Message,
    MessageRole,
    SessionContext,
    TaskComplexitySignals,
    ToolOutput,
    VerificationResult,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_user_message(self):
        """Can create a user message."""
        msg = Message(role=MessageRole.USER, content="Hello")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.timestamp is None
        assert msg.metadata is None

    def test_create_message_with_metadata(self):
        """Can create a message with all fields."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there",
            timestamp=1234567890.0,
            metadata={"tokens": 10},
        )

        assert msg.role == MessageRole.ASSISTANT
        assert msg.timestamp == 1234567890.0
        assert msg.metadata == {"tokens": 10}


class TestToolOutput:
    """Tests for ToolOutput dataclass."""

    def test_create_tool_output(self):
        """Can create a tool output."""
        output = ToolOutput(tool_name="bash", content="Hello World")

        assert output.tool_name == "bash"
        assert output.content == "Hello World"
        assert output.exit_code is None

    def test_create_tool_output_with_exit_code(self):
        """Can create a tool output with exit code."""
        output = ToolOutput(tool_name="bash", content="Error", exit_code=1)

        assert output.exit_code == 1


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_empty_context(self):
        """Can create empty context."""
        ctx = SessionContext()

        assert ctx.messages == []
        assert ctx.files == {}
        assert ctx.tool_outputs == []
        assert ctx.working_memory == {}

    def test_total_tokens_empty(self):
        """Empty context has 0 tokens."""
        ctx = SessionContext()

        assert ctx.total_tokens == 0

    def test_total_tokens_with_content(self):
        """Token count estimates from content."""
        ctx = SessionContext(
            messages=[Message(role=MessageRole.USER, content="x" * 100)],
            files={"test.py": "y" * 200},
            tool_outputs=[ToolOutput(tool_name="bash", content="z" * 100)],
        )

        # 400 chars total / 4 = 100 tokens estimate
        assert ctx.total_tokens == 100

    def test_active_modules_empty(self):
        """Empty files means no active modules."""
        ctx = SessionContext()

        assert ctx.active_modules == set()

    def test_active_modules_extracts_from_paths(self):
        """Active modules extracted from file paths."""
        ctx = SessionContext(
            files={
                "src/auth/login.py": "code",
                "src/api/routes.py": "code",
                "tests/test_auth.py": "code",
            }
        )

        assert ctx.active_modules == {"src", "tests"}

    def test_active_modules_handles_simple_paths(self):
        """Handles files without directory structure."""
        ctx = SessionContext(files={"README.md": "# Title", "setup.py": "config"})

        assert ctx.active_modules == {"README.md", "setup.py"}


class TestTaskComplexitySignals:
    """Tests for TaskComplexitySignals pydantic model."""

    def test_defaults_all_false(self):
        """All signals default to False."""
        signals = TaskComplexitySignals()

        assert signals.references_multiple_files is False
        assert signals.requires_cross_context_reasoning is False
        assert signals.involves_temporal_reasoning is False
        assert signals.asks_about_patterns is False
        assert signals.debugging_task is False
        assert signals.context_has_multiple_domains is False
        assert signals.recent_tool_outputs_large is False
        assert signals.conversation_has_state_changes is False
        assert signals.files_span_multiple_modules is False
        assert signals.previous_turn_was_confused is False
        assert signals.task_is_continuation is False

    def test_can_set_signals(self):
        """Can create with specific signals set."""
        signals = TaskComplexitySignals(
            debugging_task=True,
            recent_tool_outputs_large=True,
        )

        assert signals.debugging_task is True
        assert signals.recent_tool_outputs_large is True
        assert signals.references_multiple_files is False


class TestActivationDecision:
    """Tests for ActivationDecision pydantic model."""

    def test_create_activation_decision(self):
        """Can create activation decision."""
        decision = ActivationDecision(
            should_activate=True,
            reason="cross_context_reasoning",
            score=5,
        )

        assert decision.should_activate is True
        assert decision.reason == "cross_context_reasoning"
        assert decision.score == 5
        assert decision.signals is None

    def test_activation_with_signals(self):
        """Can include signals in decision."""
        signals = TaskComplexitySignals(debugging_task=True)
        decision = ActivationDecision(
            should_activate=True,
            reason="debugging",
            signals=signals,
        )

        assert decision.signals is not None
        assert decision.signals.debugging_task is True


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_result(self):
        """Can create successful execution result."""
        result = ExecutionResult(success=True, output=42, execution_time_ms=10.5)

        assert result.success is True
        assert result.output == 42
        assert result.error is None
        assert result.execution_time_ms == 10.5

    def test_failed_result(self):
        """Can create failed execution result."""
        result = ExecutionResult(success=False, error="SyntaxError")

        assert result.success is False
        assert result.output is None
        assert result.error == "SyntaxError"


class TestVerificationResult:
    """Tests for VerificationResult pydantic model."""

    def test_safe_result(self):
        """Can create safe verification result."""
        result = VerificationResult(safe=True)

        assert result.safe is True
        assert result.reason is None
        assert result.witness is None
        assert result.conflicts is None

    def test_unsafe_result_with_reason(self):
        """Can create unsafe result with reason."""
        result = VerificationResult(
            safe=False,
            reason="Type mismatch",
            conflicts=["constraint_1", "constraint_2"],
        )

        assert result.safe is False
        assert result.reason == "Type mismatch"
        assert result.conflicts == ["constraint_1", "constraint_2"]
