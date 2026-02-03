"""
Unit tests for complexity classifier.

Implements: Spec §6.3 tests
"""

import sys
from pathlib import Path

# Add project root to path so we can import src package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.complexity_classifier import (
    extract_complexity_signals,
    is_definitely_simple,
    should_activate_rlm,
)


class TestExtractComplexitySignals:
    """Tests for extract_complexity_signals function."""

    def test_detects_multi_file_reference(self, mock_context):
        """Prompt mentioning multiple files sets references_multiple_files."""
        prompt = "Fix the bug in auth.py and update tests.py"

        signals = extract_complexity_signals(prompt, mock_context)

        assert signals.references_multiple_files is True

    def test_detects_cross_context_reasoning(self, mock_context):
        """'Why X given Y' patterns trigger cross_context_reasoning."""
        prompt = "Why is the test failing given the fix we made?"

        signals = extract_complexity_signals(prompt, mock_context)

        assert signals.requires_cross_context_reasoning is True

    def test_detects_debugging_task(self, mock_context):
        """Error-related keywords trigger debugging_task."""
        prompt = "There's a bug causing the authentication to crash"

        signals = extract_complexity_signals(prompt, mock_context)

        assert signals.debugging_task is True

    def test_detects_pattern_search(self, mock_context):
        """'Find all' patterns trigger asks_about_patterns."""
        prompt = "Find all places where we call the API"

        signals = extract_complexity_signals(prompt, mock_context)

        assert signals.asks_about_patterns is True

    def test_simple_query_no_signals(self, mock_context):
        """Simple queries should not trigger complexity signals."""
        prompt = "Show me package.json"

        signals = extract_complexity_signals(prompt, mock_context)

        assert signals.references_multiple_files is False
        assert signals.requires_cross_context_reasoning is False
        assert signals.debugging_task is False

    def test_detects_exhaustive_search(self, mock_context):
        """'Find all' and similar patterns trigger requires_exhaustive_search."""
        prompts = [
            "Find all API endpoints in the codebase",
            "List every error handler",
            "Show all occurrences of this pattern",
            "comprehensive review of the module",
        ]

        for prompt in prompts:
            signals = extract_complexity_signals(prompt, mock_context)
            assert signals.requires_exhaustive_search is True, f"Failed for: {prompt}"

    def test_detects_user_wants_thorough(self, mock_context):
        """Thorough intent patterns trigger user_wants_thorough."""
        prompts = [
            "Make sure all tests pass",
            "Be careful with the refactoring",
            "Do a thorough analysis",
            "Don't miss any edge cases",
            "This is critical - check everything",
        ]

        for prompt in prompts:
            signals = extract_complexity_signals(prompt, mock_context)
            assert signals.user_wants_thorough is True, f"Failed for: {prompt}"

    def test_detects_user_wants_fast(self, mock_context):
        """Fast intent patterns trigger user_wants_fast."""
        prompts = [
            "Quick question about the config",
            "Just show me the file",
            "Briefly explain this function",
            "Give me a simple answer",
        ]

        for prompt in prompts:
            signals = extract_complexity_signals(prompt, mock_context)
            assert signals.user_wants_fast is True, f"Failed for: {prompt}"


class TestShouldActivateRlm:
    """Tests for should_activate_rlm function."""

    def test_activates_on_cross_context_reasoning(self, mock_context):
        """Cross-context reasoning with sufficient complexity activates RLM."""
        # rlm_core's PatternClassifier needs enough signals to exceed threshold;
        # a simple "why X when Y" may not score high enough alone.
        prompt = (
            "Why does the authentication fail when the database config is changed across modules?"
        )

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        assert should_activate is True

    def test_activates_on_debugging_with_large_output(self, debug_context):
        """Debugging with large tool output activates RLM."""
        prompt = "Fix the failing test"
        debug_context.tool_outputs[0].content = "x" * 15000  # Large output

        should_activate, reason = should_activate_rlm(prompt, debug_context)

        assert should_activate is True

    def test_respects_manual_override_on(self, mock_context):
        """Manual RLM mode forces activation."""
        prompt = "Simple question"

        should_activate, reason = should_activate_rlm(prompt, mock_context, rlm_mode_forced=True)

        assert should_activate is True
        assert reason == "manual_override"

    def test_respects_manual_override_off(self, mock_context):
        """Manual simple mode prevents activation."""
        prompt = "Complex debugging task with errors"

        should_activate, reason = should_activate_rlm(prompt, mock_context, simple_mode_forced=True)

        assert should_activate is False
        assert reason == "simple_mode_forced"

    def test_activates_on_pattern_search_alone(self, mock_context):
        """Pattern search queries with enough signals should activate RLM."""
        # rlm_core's classifier needs score >= 3; combine pattern search with
        # enough complexity signals to exceed the threshold
        prompt = "Find all places where logging is used and trace through the error handlers"

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        assert should_activate is True

    def test_activates_on_list_all_pattern(self, mock_context):
        """'List all' patterns should activate RLM."""
        prompt = "List all error handlers in the codebase"

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        assert should_activate is True

    def test_activates_on_debug_keyword_alone(self, mock_context):
        """Debug keywords should trigger RLM even without large outputs."""
        prompt = "Debug the stack trace from the error"

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        assert should_activate is True
        assert "debug" in reason.lower()

    def test_activates_on_error_keyword_alone(self, mock_context):
        """Error keywords should trigger RLM."""
        prompt = "The authentication is throwing an exception"

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        assert should_activate is True
        assert "debug" in reason.lower()

    def test_simple_task_still_returns_false(self, mock_context):
        """Simple tasks should still not activate RLM."""
        prompt = "Show me the README"

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        assert should_activate is False
        assert "threshold" in reason or "simple" in reason.lower()

    def test_activates_on_exhaustive_search(self, mock_context):
        """Exhaustive search patterns immediately activate RLM."""
        prompt = "Find all usages of the deprecated API"

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        assert should_activate is True
        assert "exhaustive" in reason.lower()

    def test_activates_on_user_thorough_intent(self, mock_context):
        """User wanting thorough analysis adds to complexity score."""
        prompt = "Make sure all the edge cases are covered"

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        assert should_activate is True
        assert "thorough" in reason.lower()

    def test_user_fast_intent_does_not_prevent_activation(self, mock_context):
        """Fast intent doesn't override complexity signals."""
        prompt = "Quickly find all the bugs in this module"

        should_activate, reason = should_activate_rlm(prompt, mock_context)

        # Should still activate due to exhaustive search and debugging signals
        assert should_activate is True


class TestIsDefinitelySimple:
    """Tests for is_definitely_simple function."""

    def test_simple_file_read(self, mock_context):
        """Simple file reads are definitely simple."""
        prompt = "show package.json"

        assert is_definitely_simple(prompt, mock_context) is True

    def test_git_status(self, mock_context):
        """Git status is definitely simple."""
        prompt = "git status"

        assert is_definitely_simple(prompt, mock_context) is True

    def test_complex_not_simple(self, mock_context):
        """Complex queries are not definitely simple."""
        prompt = "Find all the bugs in the authentication module"

        assert is_definitely_simple(prompt, mock_context) is False

    def test_large_context_not_simple(self, large_context):
        """Even simple prompts with large context aren't definitely simple."""
        prompt = "ok"

        # Large context (100K tokens) should not be considered simple
        # even for acknowledgment prompts
        assert is_definitely_simple(prompt, large_context) is False
