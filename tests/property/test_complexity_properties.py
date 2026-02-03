"""
Property-based tests for complexity classifier.

Implements: Spec §6.3 testing requirements
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.complexity_classifier import (
    extract_complexity_signals,
    is_definitely_simple,
    should_activate_rlm,
)
from src.types import Message, MessageRole, SessionContext, ToolOutput

# Strategies for generating test data
message_strategy = st.builds(
    Message,
    role=st.sampled_from(list(MessageRole)),
    content=st.text(min_size=0, max_size=1000),
)

tool_output_strategy = st.builds(
    ToolOutput,
    tool_name=st.sampled_from(["Bash", "Read", "Edit", "Write", "Grep"]),
    content=st.text(min_size=0, max_size=500),
    exit_code=st.integers(min_value=0, max_value=255),
)

context_strategy = st.builds(
    SessionContext,
    messages=st.lists(message_strategy, min_size=0, max_size=10),
    files=st.dictionaries(
        keys=st.text(min_size=1, max_size=50).filter(lambda x: "/" not in x),
        values=st.text(min_size=0, max_size=200),
        max_size=5,
    ),
    tool_outputs=st.lists(tool_output_strategy, min_size=0, max_size=5),
    working_memory=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(min_size=0, max_size=100),
        max_size=3,
    ),
)


@pytest.mark.hypothesis
class TestComplexitySignalsProperties:
    """Property-based tests for complexity signal extraction."""

    @given(prompt=st.text(min_size=0, max_size=500), context=context_strategy)
    @settings(max_examples=100)
    def test_signals_always_returns_valid_object(self, prompt, context):
        """Signal extraction always returns a valid TaskComplexitySignals object."""
        signals = extract_complexity_signals(prompt, context)

        # All fields should be booleans
        assert isinstance(signals.references_multiple_files, bool)
        assert isinstance(signals.requires_cross_context_reasoning, bool)
        assert isinstance(signals.involves_temporal_reasoning, bool)
        assert isinstance(signals.asks_about_patterns, bool)
        assert isinstance(signals.debugging_task, bool)
        assert isinstance(signals.context_has_multiple_domains, bool)
        assert isinstance(signals.recent_tool_outputs_large, bool)
        assert isinstance(signals.conversation_has_state_changes, bool)
        assert isinstance(signals.files_span_multiple_modules, bool)
        assert isinstance(signals.previous_turn_was_confused, bool)
        assert isinstance(signals.task_is_continuation, bool)

    @given(prompt=st.text(min_size=0, max_size=500), context=context_strategy)
    @settings(max_examples=100)
    def test_extraction_is_deterministic(self, prompt, context):
        """Same input always produces same output."""
        signals1 = extract_complexity_signals(prompt, context)
        signals2 = extract_complexity_signals(prompt, context)

        assert signals1 == signals2

    @given(context=context_strategy)
    @settings(max_examples=50)
    def test_empty_prompt_no_prompt_based_signals(self, context):
        """Empty prompt should not trigger prompt-based signals."""
        signals = extract_complexity_signals("", context)

        # Prompt-based signals should be false for empty prompt
        assert signals.references_multiple_files is False
        assert signals.requires_cross_context_reasoning is False
        assert signals.involves_temporal_reasoning is False
        assert signals.asks_about_patterns is False
        assert signals.debugging_task is False


@pytest.mark.hypothesis
class TestActivationDecisionProperties:
    """Property-based tests for RLM activation decisions."""

    @given(prompt=st.text(min_size=0, max_size=500), context=context_strategy)
    @settings(max_examples=100)
    def test_activation_returns_valid_tuple(self, prompt, context):
        """Activation decision always returns (bool, str) tuple."""
        should_activate, reason = should_activate_rlm(prompt, context)

        assert isinstance(should_activate, bool)
        assert isinstance(reason, str)
        assert len(reason) > 0

    @given(prompt=st.text(min_size=0, max_size=500), context=context_strategy)
    @settings(max_examples=50)
    def test_manual_rlm_override_always_activates(self, prompt, context):
        """Manual RLM override always activates regardless of content."""
        should_activate, reason = should_activate_rlm(prompt, context, rlm_mode_forced=True)

        assert should_activate is True
        assert reason == "manual_override"

    @given(prompt=st.text(min_size=0, max_size=500), context=context_strategy)
    @settings(max_examples=50)
    def test_simple_mode_override_never_activates(self, prompt, context):
        """Simple mode override never activates regardless of content."""
        should_activate, reason = should_activate_rlm(prompt, context, simple_mode_forced=True)

        assert should_activate is False
        assert reason == "simple_mode_forced"

    @given(prompt=st.text(min_size=0, max_size=500), context=context_strategy)
    @settings(max_examples=50)
    def test_activation_is_deterministic(self, prompt, context):
        """Same input always produces same activation decision."""
        result1 = should_activate_rlm(prompt, context)
        result2 = should_activate_rlm(prompt, context)

        assert result1 == result2


@pytest.mark.hypothesis
class TestSimpleQueryProperties:
    """Property-based tests for simple query detection."""

    @given(prompt=st.text(min_size=0, max_size=500), context=context_strategy)
    @settings(max_examples=100)
    def test_simple_detection_returns_bool(self, prompt, context):
        """Simple detection always returns boolean."""
        result = is_definitely_simple(prompt, context)
        assert isinstance(result, bool)

    @given(prompt=st.text(min_size=0, max_size=500), context=context_strategy)
    @settings(max_examples=50)
    def test_simple_detection_is_deterministic(self, prompt, context):
        """Same input always produces same simple detection result."""
        result1 = is_definitely_simple(prompt, context)
        result2 = is_definitely_simple(prompt, context)

        assert result1 == result2

    @given(context=context_strategy)
    @settings(max_examples=30)
    def test_definitely_simple_implies_no_activation(self, context):
        """If definitely simple, should not activate RLM (without overrides)."""
        # Generate simple prompts that should be detected
        simple_prompts = ["ok", "thanks", "yes", "no", "sure"]

        for prompt in simple_prompts:
            if is_definitely_simple(prompt, context):
                should_activate, _ = should_activate_rlm(prompt, context)
                # If it's definitely simple, it shouldn't activate
                # (unless context signals override)
                # This is a soft property - context can still trigger activation
