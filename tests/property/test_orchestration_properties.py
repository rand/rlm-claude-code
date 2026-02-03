"""
Property-based tests for orchestration modules.

Tests invariants and properties of:
- OrchestrationSchema
- UserPreferences
- AutoActivation
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.auto_activation import (
    ActivationDecision,
    ActivationStats,
    ActivationThresholds,
    AutoActivator,
)
from src.orchestration_schema import (
    ExecutionMode,
    ModelTier,
    OrchestrationPlan,
    ToolAccessLevel,
)
from src.types import SessionContext
from src.user_preferences import PreferencesManager, UserPreferences

# Strategies for generating test data
execution_modes = st.sampled_from(list(ExecutionMode))
tool_access_levels = st.sampled_from(list(ToolAccessLevel))
model_tiers = st.sampled_from(list(ModelTier))

# Valid ranges for numeric fields
depth_budget = st.integers(min_value=0, max_value=10)
token_budget = st.integers(min_value=1000, max_value=1_000_000)
cost_budget = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


class TestOrchestrationPlanProperties:
    """Property tests for OrchestrationPlan."""

    @given(
        activate=st.booleans(),
        mode=execution_modes,
        tool_access=tool_access_levels,
        depth=depth_budget,
        max_tokens=token_budget,
    )
    @settings(max_examples=100)
    def test_plan_creation_valid(
        self,
        activate: bool,
        mode: ExecutionMode,
        tool_access: ToolAccessLevel,
        depth: int,
        max_tokens: int,
    ):
        """Plans can be created with any valid combination of parameters."""
        plan = OrchestrationPlan(
            activate_rlm=activate,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="test-model",
            execution_mode=mode,
            tool_access=tool_access,
            depth_budget=depth,
            max_tokens=max_tokens,
        )

        assert plan.activate_rlm == activate
        assert plan.execution_mode == mode
        assert plan.tool_access == tool_access
        assert plan.depth_budget == depth

    @given(mode=execution_modes)
    def test_execution_mode_round_trip(self, mode: ExecutionMode):
        """ExecutionMode can round-trip through string value."""
        value = mode.value
        restored = ExecutionMode(value)
        assert restored == mode

    @given(level=tool_access_levels)
    def test_tool_access_level_round_trip(self, level: ToolAccessLevel):
        """ToolAccessLevel can round-trip through string value."""
        value = level.value
        restored = ToolAccessLevel(value)
        assert restored == level


class TestUserPreferencesProperties:
    """Property tests for UserPreferences."""

    @given(
        mode=execution_modes,
        auto_activate=st.booleans(),
        budget=cost_budget,
        max_depth=depth_budget,
    )
    @settings(max_examples=100)
    def test_preferences_creation(
        self, mode: ExecutionMode, auto_activate: bool, budget: float, max_depth: int
    ):
        """Preferences can be created with valid parameters."""
        prefs = UserPreferences(
            execution_mode=mode,
            auto_activate=auto_activate,
            budget_dollars=budget,
            max_depth=max_depth,
        )

        assert prefs.execution_mode == mode
        assert prefs.auto_activate == auto_activate
        assert prefs.budget_dollars == budget
        assert prefs.max_depth == max_depth

    @given(
        mode=execution_modes,
        budget=cost_budget,
    )
    def test_preferences_to_dict_round_trip(self, mode: ExecutionMode, budget: float):
        """Preferences can round-trip through dict."""
        original = UserPreferences(
            execution_mode=mode,
            budget_dollars=budget,
        )

        d = original.to_dict()
        restored = UserPreferences.from_dict(d)

        assert restored.execution_mode == original.execution_mode
        assert restored.budget_dollars == original.budget_dollars

    @given(mode=execution_modes)
    def test_parse_mode_command(self, mode: ExecutionMode):
        """Mode command parsing works for all modes."""
        manager = PreferencesManager()

        message, params = manager.parse_command(f"mode {mode.value}")

        assert "Mode set to" in message or "mode" in message.lower()
        assert params.get("execution_mode") == mode.value

    @given(budget=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False))
    def test_parse_budget_command(self, budget: float):
        """Budget command parsing works for valid amounts."""
        manager = PreferencesManager()

        message, params = manager.parse_command(f"budget ${budget:.2f}")

        assert "Budget set" in message or "budget" in message.lower()
        assert "budget_dollars" in params

    @given(depth=st.integers(min_value=0, max_value=5))
    def test_parse_depth_command(self, depth: int):
        """Depth command parsing works for valid depths."""
        manager = PreferencesManager()

        message, params = manager.parse_command(f"depth {depth}")

        assert "depth" in message.lower()
        assert params.get("max_depth") == depth


class TestActivationDecisionProperties:
    """Property tests for ActivationDecision."""

    @given(
        should_activate=st.booleans(),
        conf=confidence,
    )
    def test_decision_creation(self, should_activate: bool, conf: float):
        """Decisions can be created with any valid confidence."""
        decision = ActivationDecision(
            should_activate=should_activate,
            reason="test",
            confidence=conf,
        )

        assert decision.should_activate == should_activate
        assert decision.confidence == conf
        assert 0.0 <= decision.confidence <= 1.0

    @given(
        decisions=st.lists(
            st.tuples(st.booleans(), st.text(min_size=1, max_size=20)),
            min_size=1,
            max_size=20,
        )
    )
    def test_stats_accumulation(self, decisions: list[tuple[bool, str]]):
        """Stats correctly accumulate decisions."""
        stats = ActivationStats()

        for should_activate, reason in decisions:
            decision = ActivationDecision(
                should_activate=should_activate,
                reason=reason,
                confidence=0.8,
            )
            stats.record(decision)

        assert stats.total_decisions == len(decisions)
        assert stats.activations + stats.skips == stats.total_decisions
        assert stats.activations == sum(1 for a, _ in decisions if a)


class TestActivationThresholdsProperties:
    """Property tests for ActivationThresholds."""

    @given(
        min_tokens=st.integers(min_value=1000, max_value=50000),
        auto_tokens=st.integers(min_value=50000, max_value=500000),
    )
    def test_thresholds_ordering(self, min_tokens: int, auto_tokens: int):
        """Auto-activate threshold should be >= min threshold."""
        assume(auto_tokens >= min_tokens)

        thresholds = ActivationThresholds(
            min_tokens_for_activation=min_tokens,
            auto_activate_above_tokens=auto_tokens,
        )

        assert thresholds.auto_activate_above_tokens >= thresholds.min_tokens_for_activation


class TestAutoActivatorProperties:
    """Property tests for AutoActivator."""

    @given(force_rlm=st.booleans(), force_simple=st.booleans())
    def test_force_flags_override(self, force_rlm: bool, force_simple: bool):
        """Force flags always override automatic decision."""
        activator = AutoActivator()
        context = SessionContext()

        if force_rlm:
            decision = activator.should_activate("any query", context, force_rlm=True)
            assert decision.should_activate is True
            assert decision.confidence == 1.0

        if force_simple:
            decision = activator.should_activate("any query", context, force_simple=True)
            assert decision.should_activate is False
            assert decision.confidence == 1.0

    @given(auto_activate=st.booleans())
    def test_auto_activate_preference_respected(self, auto_activate: bool):
        """Auto-activate preference is respected."""
        prefs = UserPreferences(auto_activate=auto_activate)
        activator = AutoActivator(preferences=prefs)
        context = SessionContext()

        decision = activator.should_activate("complex query", context)

        if not auto_activate:
            assert decision.should_activate is False
            assert decision.reason == "auto_activate_disabled"
