"""
Unit tests for auto_activation module.

Implements: Spec §8.1 Phase 4 - Auto-activation tests
"""

import pytest

from src.auto_activation import (
    ActivationDecision,
    ActivationStats,
    ActivationThresholds,
    AutoActivator,
    check_auto_activation,
    get_auto_activator,
)
from src.orchestration_schema import ExecutionMode, ToolAccessLevel
from src.types import SessionContext, ToolOutput
from src.user_preferences import UserPreferences


@pytest.fixture
def empty_context():
    """Create empty session context."""
    return SessionContext(
        messages=[],
        files={},
        tool_outputs=[],
    )


@pytest.fixture
def complex_context():
    """Create complex session context with multiple modules."""
    # Files as dict with content to trigger active_modules property
    return SessionContext(
        messages=[],
        files={
            "src/auth/handler.py": "x" * 10000,
            "src/api/routes.py": "y" * 10000,
            "src/db/models.py": "z" * 10000,
        },
        tool_outputs=[
            ToolOutput(tool_name="Read", content="x" * 5000),
            ToolOutput(tool_name="Grep", content="y" * 5000),
        ],
    )


@pytest.fixture
def large_context():
    """Create large token context."""
    # Large content to trigger high token count
    return SessionContext(
        messages=[],
        files={"large.py": "x" * 600000},  # ~150k tokens
        tool_outputs=[],
    )


class TestActivationDecision:
    """Tests for ActivationDecision dataclass."""

    def test_create_decision(self):
        """Can create activation decision."""
        decision = ActivationDecision(
            should_activate=True,
            reason="complex_task",
            confidence=0.85,
        )

        assert decision.should_activate is True
        assert decision.reason == "complex_task"
        assert decision.confidence == 0.85

    def test_to_dict(self):
        """Converts to dictionary."""
        decision = ActivationDecision(
            should_activate=True,
            reason="multi_file",
            confidence=0.9,
            decision_time_ms=5.5,
        )

        d = decision.to_dict()

        assert d["should_activate"] is True
        assert d["reason"] == "multi_file"
        assert d["confidence"] == 0.9
        assert d["decision_time_ms"] == 5.5


class TestActivationThresholds:
    """Tests for ActivationThresholds dataclass."""

    def test_default_thresholds(self):
        """Default thresholds are sensible."""
        thresholds = ActivationThresholds()

        assert thresholds.min_tokens_for_activation == 20_000
        assert thresholds.auto_activate_above_tokens == 100_000
        assert thresholds.min_complexity_score == 2

    def test_custom_thresholds(self):
        """Can create custom thresholds."""
        thresholds = ActivationThresholds(
            min_tokens_for_activation=10_000,
            auto_activate_above_tokens=50_000,
        )

        assert thresholds.min_tokens_for_activation == 10_000
        assert thresholds.auto_activate_above_tokens == 50_000


class TestActivationStats:
    """Tests for ActivationStats dataclass."""

    def test_record_activation(self):
        """Records activation decisions."""
        stats = ActivationStats()
        decision = ActivationDecision(
            should_activate=True,
            reason="complex_task",
            confidence=0.8,
            decision_time_ms=5.0,
        )

        stats.record(decision)

        assert stats.total_decisions == 1
        assert stats.activations == 1
        assert stats.skips == 0

    def test_record_skip(self):
        """Records skip decisions."""
        stats = ActivationStats()
        decision = ActivationDecision(
            should_activate=False,
            reason="simple_task",
            confidence=0.9,
            decision_time_ms=2.0,
        )

        stats.record(decision)

        assert stats.total_decisions == 1
        assert stats.activations == 0
        assert stats.skips == 1

    def test_record_override(self):
        """Records override decisions."""
        stats = ActivationStats()
        decision = ActivationDecision(
            should_activate=True,
            reason="manual_force",
            confidence=1.0,
        )

        stats.record(decision, was_override=True)

        assert stats.overrides == 1

    def test_tracks_reasons(self):
        """Tracks activation reasons."""
        stats = ActivationStats()

        stats.record(ActivationDecision(True, "complex_task", 0.8))
        stats.record(ActivationDecision(True, "complex_task", 0.9))
        stats.record(ActivationDecision(False, "simple_task", 0.95))

        assert stats.activation_reasons["complex_task"] == 2
        assert stats.activation_reasons["simple_task"] == 1

    def test_to_dict(self):
        """Converts to dictionary."""
        stats = ActivationStats()
        stats.record(ActivationDecision(True, "multi_file", 0.8))

        d = stats.to_dict()

        assert "total_decisions" in d
        assert "activation_rate" in d
        assert "top_reasons" in d


class TestAutoActivator:
    """Tests for AutoActivator class."""

    @pytest.fixture
    def activator(self):
        """Create activator with default settings."""
        return AutoActivator()

    def test_force_rlm(self, activator, empty_context):
        """Force RLM always activates."""
        decision = activator.should_activate(
            "simple query",
            empty_context,
            force_rlm=True,
        )

        assert decision.should_activate is True
        assert decision.reason == "manual_force"
        assert decision.confidence == 1.0

    def test_force_simple(self, activator, complex_context):
        """Force simple always skips."""
        decision = activator.should_activate(
            "complex multi-file query about auth.py and api.py",
            complex_context,
            force_simple=True,
        )

        assert decision.should_activate is False
        assert decision.reason == "manual_force_simple"

    def test_auto_activate_disabled(self, activator, complex_context):
        """Respects disabled auto-activation."""
        activator.preferences = UserPreferences(auto_activate=False)

        decision = activator.should_activate(
            "complex multi-file query",
            complex_context,
        )

        assert decision.should_activate is False
        assert decision.reason == "auto_activate_disabled"

    def test_simple_query_skips(self, activator, empty_context):
        """Simple queries skip activation."""
        decision = activator.should_activate(
            "ok",
            empty_context,
        )

        assert decision.should_activate is False

    def test_large_context_activates(self, activator, large_context):
        """Large context auto-activates (SPEC-14.20: escalates to thorough)."""
        decision = activator.should_activate(
            "any query",
            large_context,
        )

        assert decision.should_activate is True
        # SPEC-14.20: Large context triggers escalation to thorough mode
        assert "large_context" in decision.reason

    def test_complex_query_activates(self, activator, complex_context):
        """Complex queries activate."""
        decision = activator.should_activate(
            "Why does auth.py fail when api.py calls the handler?",
            complex_context,
        )

        assert decision.should_activate is True

    def test_debugging_with_large_output(self, activator):
        """Debugging with large output activates."""
        context = SessionContext(
            messages=[],
            files={"app.py": "x" * 10000},
            tool_outputs=[ToolOutput(tool_name="Bash", content="x" * 15000)],
        )

        decision = activator.should_activate(
            "Fix this error in the logs",
            context,
        )

        assert decision.should_activate is True

    def test_decision_time_tracked(self, activator, empty_context):
        """Decision time is tracked."""
        decision = activator.should_activate("query", empty_context)

        assert decision.decision_time_ms > 0

    def test_statistics_tracked(self, activator, empty_context):
        """Statistics are tracked."""
        activator.should_activate("query 1", empty_context)
        activator.should_activate("query 2", empty_context)

        stats = activator.get_statistics()

        assert stats["total_decisions"] == 2

    def test_reset_statistics(self, activator, empty_context):
        """Can reset statistics."""
        activator.should_activate("query", empty_context)
        activator.reset_statistics()

        stats = activator.get_statistics()

        assert stats["total_decisions"] == 0

    def test_callbacks_invoked(self, activator, empty_context):
        """Callbacks are invoked on decisions."""
        decisions = []
        activator.add_callback(lambda d: decisions.append(d))

        activator.should_activate("query", empty_context)

        assert len(decisions) == 1

    def test_remove_callback(self, activator, empty_context):
        """Can remove callbacks."""
        decisions = []
        callback = lambda d: decisions.append(d)

        activator.add_callback(callback)
        activator.remove_callback(callback)
        activator.should_activate("query", empty_context)

        assert len(decisions) == 0

    def test_callback_error_ignored(self, activator, empty_context):
        """Callback errors don't affect activation."""

        def bad_callback(d):
            raise ValueError("callback error")

        activator.add_callback(bad_callback)

        # Should not raise
        decision = activator.should_activate("query", empty_context)
        assert decision is not None


class TestCreatePlanFromDecision:
    """Tests for plan creation from decisions."""

    @pytest.fixture
    def activator(self):
        """Create activator."""
        return AutoActivator()

    def test_no_plan_for_skip(self, activator):
        """No plan created for skip decisions."""
        decision = ActivationDecision(
            should_activate=False,
            reason="simple_task",
            confidence=0.9,
        )

        plan = activator.create_plan_from_decision(decision)

        # SPEC-14: Returns bypass plan instead of None
        assert plan is not None
        assert plan.activate_rlm is False
        assert plan.activation_reason == "simple_task"

    def test_plan_created_for_activation(self, activator):
        """Plan created for activation decisions."""
        decision = ActivationDecision(
            should_activate=True,
            reason="complex_task",
            confidence=0.85,
        )

        plan = activator.create_plan_from_decision(decision)

        assert plan is not None
        assert plan.activate_rlm is True
        assert plan.activation_reason == "complex_task"

    def test_plan_respects_preferences(self, activator):
        """Plan respects user preferences for non-micro modes."""
        activator.preferences = UserPreferences(
            execution_mode=ExecutionMode.FAST,
            max_depth=1,
            tool_access=ToolAccessLevel.FULL,
        )

        # SPEC-14: Must set execution_mode to non-MICRO to test preference application
        decision = ActivationDecision(
            should_activate=True,
            reason="complex_task",
            confidence=0.9,
            execution_mode=ExecutionMode.BALANCED,  # Non-micro mode
        )

        plan = activator.create_plan_from_decision(decision)

        # Plan uses strategy-based planning for non-micro modes
        assert plan.depth_budget <= 1  # Respects max_depth preference

    def test_low_confidence_reduces_depth(self, activator):
        """Low confidence reduces depth budget."""
        activator.preferences = UserPreferences(max_depth=3)

        decision = ActivationDecision(
            should_activate=True,
            reason="uncertain_task",
            confidence=0.5,  # Low confidence
        )

        plan = activator.create_plan_from_decision(decision)

        assert plan.depth_budget == 1


class TestGetAutoActivator:
    """Tests for get_auto_activator function."""

    def test_returns_activator(self):
        """Returns an AutoActivator instance."""
        activator = get_auto_activator()

        assert isinstance(activator, AutoActivator)

    def test_returns_same_instance(self):
        """Returns same global instance."""
        a1 = get_auto_activator()
        a2 = get_auto_activator()

        assert a1 is a2


class TestCheckAutoActivation:
    """Tests for check_auto_activation convenience function."""

    def test_returns_decision(self, empty_context):
        """Returns ActivationDecision."""
        decision = check_auto_activation("query", empty_context)

        assert isinstance(decision, ActivationDecision)

    def test_respects_preferences(self, complex_context):
        """Respects passed preferences."""
        prefs = UserPreferences(auto_activate=False)

        decision = check_auto_activation(
            "complex query about auth.py",
            complex_context,
            preferences=prefs,
        )

        assert decision.should_activate is False


class TestIntegration:
    """Integration tests for auto-activation."""

    def test_full_workflow(self, complex_context):
        """Full activation workflow."""
        activator = AutoActivator()

        # Check activation
        decision = activator.should_activate(
            "Trace why auth.py fails when connecting to the database through api.py",
            complex_context,
        )

        # Should activate for cross-context reasoning
        assert decision.should_activate is True
        assert decision.signals is not None

        # Create plan
        plan = activator.create_plan_from_decision(decision)
        assert plan is not None
        assert plan.activate_rlm is True

        # Statistics updated
        stats = activator.get_statistics()
        assert stats["activations"] >= 1

    def test_multiple_queries_track_stats(self, empty_context, complex_context):
        """Multiple queries track statistics correctly."""
        activator = AutoActivator()

        # Simple query
        activator.should_activate("ok", empty_context)

        # Complex query
        activator.should_activate(
            "Why does the error occur when auth and api interact?",
            complex_context,
        )

        stats = activator.get_statistics()
        assert stats["total_decisions"] == 2
