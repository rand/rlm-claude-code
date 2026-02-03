"""
Unit tests for orchestration_schema module.

Implements: Spec §8.1 Phase 2 - Orchestration Layer tests
"""

import pytest

from src.orchestration_schema import (
    MODE_DEFAULTS,
    STRATEGY_DEFAULTS,
    TIER_MODELS,
    DecisionConfidence,
    ExecutionMode,
    ExecutionStrategy,
    OrchestrationContext,
    OrchestrationPlan,
    PlanAdjustment,
    ToolAccessLevel,
)
from src.smart_router import ModelTier, QueryType


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_all_modes_exist(self):
        """All expected execution modes exist."""
        assert ExecutionMode.FAST
        assert ExecutionMode.BALANCED
        assert ExecutionMode.THOROUGH

    def test_mode_values(self):
        """Mode values are correct strings."""
        assert ExecutionMode.FAST.value == "fast"
        assert ExecutionMode.BALANCED.value == "balanced"
        assert ExecutionMode.THOROUGH.value == "thorough"


class TestToolAccessLevel:
    """Tests for ToolAccessLevel enum."""

    def test_all_levels_exist(self):
        """All expected tool access levels exist."""
        assert ToolAccessLevel.NONE
        assert ToolAccessLevel.REPL_ONLY
        assert ToolAccessLevel.READ_ONLY
        assert ToolAccessLevel.FULL

    def test_level_values(self):
        """Level values are correct strings."""
        assert ToolAccessLevel.NONE.value == "none"
        assert ToolAccessLevel.REPL_ONLY.value == "repl_only"
        assert ToolAccessLevel.READ_ONLY.value == "read_only"
        assert ToolAccessLevel.FULL.value == "full"


class TestModeDefaults:
    """Tests for MODE_DEFAULTS configuration."""

    def test_all_modes_have_defaults(self):
        """All execution modes have default configurations."""
        for mode in ExecutionMode:
            assert mode in MODE_DEFAULTS

    def test_fast_mode_defaults(self):
        """Fast mode has appropriate defaults."""
        defaults = MODE_DEFAULTS[ExecutionMode.FAST]

        assert defaults["depth_budget"] == 1
        assert defaults["model_tier"] == ModelTier.FAST
        assert defaults["tool_access"] == ToolAccessLevel.REPL_ONLY
        assert defaults["max_cost_dollars"] < 1.0

    def test_balanced_mode_defaults(self):
        """Balanced mode has appropriate defaults."""
        defaults = MODE_DEFAULTS[ExecutionMode.BALANCED]

        assert defaults["depth_budget"] == 2
        assert defaults["model_tier"] == ModelTier.BALANCED
        assert defaults["tool_access"] == ToolAccessLevel.READ_ONLY

    def test_thorough_mode_defaults(self):
        """Thorough mode has appropriate defaults."""
        defaults = MODE_DEFAULTS[ExecutionMode.THOROUGH]

        assert defaults["depth_budget"] == 3
        assert defaults["model_tier"] == ModelTier.POWERFUL
        assert defaults["tool_access"] == ToolAccessLevel.FULL
        assert defaults["max_cost_dollars"] >= 5.0


class TestOrchestrationPlan:
    """Tests for OrchestrationPlan dataclass."""

    def test_create_basic_plan(self):
        """Can create a basic orchestration plan."""
        plan = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
        )

        assert plan.activate_rlm is True
        assert plan.primary_model == "sonnet"
        assert plan.depth_budget == 2
        assert plan.execution_mode == ExecutionMode.BALANCED

    def test_total_token_budget(self):
        """Calculates total token budget correctly."""
        plan = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            depth_budget=2,
            tokens_per_depth=25_000,
        )

        assert plan.total_token_budget == 50_000

    def test_allows_recursion(self):
        """Checks recursion permission correctly."""
        plan_with_depth = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            depth_budget=2,
        )
        assert plan_with_depth.allows_recursion is True

        plan_no_depth = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            depth_budget=0,
        )
        assert plan_no_depth.allows_recursion is False

    def test_tool_access_properties(self):
        """Tool access properties work correctly."""
        # No tool access
        plan_none = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            tool_access=ToolAccessLevel.NONE,
        )
        assert plan_none.allows_tools is False
        assert plan_none.allows_file_read is False
        assert plan_none.allows_file_write is False

        # REPL only
        plan_repl = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            tool_access=ToolAccessLevel.REPL_ONLY,
        )
        assert plan_repl.allows_tools is True
        assert plan_repl.allows_file_read is False
        assert plan_repl.allows_file_write is False

        # Read only
        plan_read = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            tool_access=ToolAccessLevel.READ_ONLY,
        )
        assert plan_read.allows_tools is True
        assert plan_read.allows_file_read is True
        assert plan_read.allows_file_write is False

        # Full access
        plan_full = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            tool_access=ToolAccessLevel.FULL,
        )
        assert plan_full.allows_tools is True
        assert plan_full.allows_file_read is True
        assert plan_full.allows_file_write is True

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        plan = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="complexity_score:3",
            model_tier=ModelTier.POWERFUL,
            primary_model="opus",
            fallback_chain=["sonnet", "haiku"],
            depth_budget=3,
            execution_mode=ExecutionMode.THOROUGH,
            tool_access=ToolAccessLevel.FULL,
            query_type=QueryType.DEBUGGING,
            complexity_score=0.8,
            signals=["multi_file", "debugging"],
        )

        d = plan.to_dict()

        assert d["activate_rlm"] is True
        assert d["activation_reason"] == "complexity_score:3"
        assert d["model_tier"] == "powerful"
        assert d["primary_model"] == "opus"
        assert d["fallback_chain"] == ["sonnet", "haiku"]
        assert d["depth_budget"] == 3
        assert d["execution_mode"] == "thorough"
        assert d["tool_access"] == "full"
        assert d["query_type"] == "debugging"
        assert d["complexity_score"] == 0.8
        assert d["signals"] == ["multi_file", "debugging"]

    def test_bypass_factory(self):
        """Bypass factory creates non-activating plan."""
        plan = OrchestrationPlan.bypass("simple_task")

        assert plan.activate_rlm is False
        assert plan.activation_reason == "simple_task"
        assert plan.depth_budget == 0
        assert plan.tool_access == ToolAccessLevel.NONE
        assert plan.allows_recursion is False

    def test_from_mode_fast(self):
        """Creates plan from fast mode."""
        plan = OrchestrationPlan.from_mode(ExecutionMode.FAST)

        assert plan.activate_rlm is True
        assert plan.model_tier == ModelTier.FAST
        assert plan.depth_budget == 1
        assert plan.tool_access == ToolAccessLevel.REPL_ONLY

    def test_from_mode_balanced(self):
        """Creates plan from balanced mode."""
        plan = OrchestrationPlan.from_mode(ExecutionMode.BALANCED)

        assert plan.activate_rlm is True
        assert plan.model_tier == ModelTier.BALANCED
        assert plan.depth_budget == 2
        assert plan.tool_access == ToolAccessLevel.READ_ONLY

    def test_from_mode_thorough(self):
        """Creates plan from thorough mode."""
        plan = OrchestrationPlan.from_mode(ExecutionMode.THOROUGH)

        assert plan.activate_rlm is True
        assert plan.model_tier == ModelTier.POWERFUL
        assert plan.depth_budget == 3
        assert plan.tool_access == ToolAccessLevel.FULL

    def test_from_mode_with_available_models(self):
        """Respects available models when creating plan."""
        # Only haiku available for fast tier
        plan = OrchestrationPlan.from_mode(
            ExecutionMode.FAST,
            available_models=["haiku"],
        )
        assert plan.primary_model == "haiku"

        # sonnet not available, should pick from available
        plan = OrchestrationPlan.from_mode(
            ExecutionMode.BALANCED,
            available_models=["gpt-4o", "haiku"],
        )
        assert plan.primary_model == "gpt-4o"

    def test_from_mode_with_query_type(self):
        """Includes query type in plan."""
        plan = OrchestrationPlan.from_mode(
            ExecutionMode.BALANCED,
            query_type=QueryType.DEBUGGING,
        )

        assert plan.query_type == QueryType.DEBUGGING


class TestOrchestrationContext:
    """Tests for OrchestrationContext dataclass."""

    def test_create_basic_context(self):
        """Can create basic context."""
        ctx = OrchestrationContext(
            query="What is the main function?",
            context_tokens=5000,
        )

        assert ctx.query == "What is the main function?"
        assert ctx.context_tokens == 5000
        assert ctx.current_depth == 0

    def test_remaining_depth(self):
        """Calculates remaining depth correctly."""
        ctx = OrchestrationContext(query="test", current_depth=0)
        assert ctx.remaining_depth == 3

        ctx = OrchestrationContext(query="test", current_depth=1)
        assert ctx.remaining_depth == 2

        ctx = OrchestrationContext(query="test", current_depth=3)
        assert ctx.remaining_depth == 0

        # Cap at 0
        ctx = OrchestrationContext(query="test", current_depth=5)
        assert ctx.remaining_depth == 0

    def test_can_recurse(self):
        """Checks recursion capability correctly."""
        # Can recurse at depth 0
        ctx = OrchestrationContext(
            query="test",
            current_depth=0,
            budget_remaining_tokens=10_000,
        )
        assert ctx.can_recurse is True

        # Cannot recurse at max depth
        ctx = OrchestrationContext(
            query="test",
            current_depth=3,
            budget_remaining_tokens=10_000,
        )
        assert ctx.can_recurse is False

        # Cannot recurse with low budget
        ctx = OrchestrationContext(
            query="test",
            current_depth=0,
            budget_remaining_tokens=1_000,
        )
        assert ctx.can_recurse is False

    def test_forced_preferences(self):
        """Stores forced user preferences."""
        ctx = OrchestrationContext(
            query="test",
            forced_mode=ExecutionMode.FAST,
            forced_model="haiku",
            forced_rlm=True,
        )

        assert ctx.forced_mode == ExecutionMode.FAST
        assert ctx.forced_model == "haiku"
        assert ctx.forced_rlm is True


class TestPlanAdjustment:
    """Tests for PlanAdjustment dataclass."""

    def test_create_adjustment(self):
        """Can create plan adjustment."""
        adj = PlanAdjustment(
            reason="Budget exceeded",
            field_name="depth_budget",
            old_value=3,
            new_value=1,
        )

        assert adj.reason == "Budget exceeded"
        assert adj.field_name == "depth_budget"
        assert adj.old_value == 3
        assert adj.new_value == 1


class TestTierModels:
    """Tests for TIER_MODELS configuration."""

    def test_all_tiers_have_models(self):
        """All model tiers (except INHERIT) have associated models."""
        for tier in ModelTier:
            # INHERIT is a special tier that uses parent session model (SPEC-14.02)
            if tier == ModelTier.INHERIT:
                continue
            assert tier in TIER_MODELS
            assert len(TIER_MODELS[tier]) > 0

    def test_inherit_tier_not_in_tier_models(self):
        """INHERIT tier is intentionally not in TIER_MODELS (SPEC-14.02)."""
        assert ModelTier.INHERIT not in TIER_MODELS

    def test_fast_tier_models(self):
        """Fast tier includes fast models."""
        models = TIER_MODELS[ModelTier.FAST]
        assert "haiku" in models
        assert "gpt-4o-mini" in models

    def test_powerful_tier_models(self):
        """Powerful tier includes powerful models."""
        models = TIER_MODELS[ModelTier.POWERFUL]
        assert "opus" in models


class TestExecutionStrategy:
    """Tests for ExecutionStrategy enum (SPEC-12.06, SPEC-14.01)."""

    def test_all_strategies_exist(self):
        """All expected execution strategies exist."""
        assert ExecutionStrategy.MICRO  # SPEC-14.01
        assert ExecutionStrategy.DIRECT_RESPONSE
        assert ExecutionStrategy.DISCOVERY
        assert ExecutionStrategy.EXHAUSTIVE_SEARCH
        assert ExecutionStrategy.RECURSIVE_DEBUG
        assert ExecutionStrategy.MAP_REDUCE
        assert ExecutionStrategy.ARCHITECTURE
        assert ExecutionStrategy.CONTINUATION

    def test_strategy_values(self):
        """Strategy values are correct strings."""
        assert ExecutionStrategy.MICRO.value == "micro"  # SPEC-14.01
        assert ExecutionStrategy.DIRECT_RESPONSE.value == "direct"
        assert ExecutionStrategy.DISCOVERY.value == "discovery"
        assert ExecutionStrategy.EXHAUSTIVE_SEARCH.value == "exhaustive_search"
        assert ExecutionStrategy.RECURSIVE_DEBUG.value == "recursive_debug"
        assert ExecutionStrategy.MAP_REDUCE.value == "map_reduce"
        assert ExecutionStrategy.ARCHITECTURE.value == "architecture"
        assert ExecutionStrategy.CONTINUATION.value == "continuation"

    def test_strategy_count(self):
        """Eight strategies: seven JTBDs plus MICRO (SPEC-14.01)."""
        assert len(ExecutionStrategy) == 8


class TestStrategyDefaults:
    """Tests for STRATEGY_DEFAULTS configuration."""

    def test_all_strategies_have_defaults(self):
        """All execution strategies have default configurations."""
        for strategy in ExecutionStrategy:
            assert strategy in STRATEGY_DEFAULTS
            defaults = STRATEGY_DEFAULTS[strategy]
            assert "activate_rlm" in defaults
            assert "depth_budget" in defaults
            assert "model_tier" in defaults
            assert "tool_access" in defaults
            assert "hints" in defaults

    def test_direct_response_bypasses_rlm(self):
        """Direct response strategy bypasses RLM."""
        defaults = STRATEGY_DEFAULTS[ExecutionStrategy.DIRECT_RESPONSE]
        assert defaults["activate_rlm"] is False
        assert defaults["depth_budget"] == 0
        assert defaults["hints"] == []

    def test_recursive_debug_uses_powerful_model(self):
        """Recursive debug strategy uses powerful model with depth 3."""
        defaults = STRATEGY_DEFAULTS[ExecutionStrategy.RECURSIVE_DEBUG]
        assert defaults["activate_rlm"] is True
        assert defaults["depth_budget"] == 3
        assert defaults["model_tier"] == ModelTier.POWERFUL
        assert len(defaults["hints"]) > 0

    def test_exhaustive_search_uses_fast_model(self):
        """Exhaustive search uses fast model for enumeration."""
        defaults = STRATEGY_DEFAULTS[ExecutionStrategy.EXHAUSTIVE_SEARCH]
        assert defaults["activate_rlm"] is True
        assert defaults["model_tier"] == ModelTier.FAST
        assert defaults["execution_mode"] == ExecutionMode.THOROUGH

    def test_continuation_has_low_depth(self):
        """Continuation strategy has low depth since memory available."""
        defaults = STRATEGY_DEFAULTS[ExecutionStrategy.CONTINUATION]
        assert defaults["depth_budget"] == 1
        assert "memory_query()" in defaults["hints"][0]

    def test_all_strategies_have_hints(self):
        """All active strategies have REPL function hints."""
        for strategy in ExecutionStrategy:
            if strategy != ExecutionStrategy.DIRECT_RESPONSE:
                defaults = STRATEGY_DEFAULTS[strategy]
                assert len(defaults["hints"]) > 0


class TestOrchestrationPlanFromStrategy:
    """Tests for OrchestrationPlan.from_strategy() method."""

    def test_from_strategy_direct_response(self):
        """Direct response strategy creates bypass plan."""
        plan = OrchestrationPlan.from_strategy(
            ExecutionStrategy.DIRECT_RESPONSE,
            activation_reason="simple_query",
        )

        assert plan.activate_rlm is False
        assert plan.strategy == ExecutionStrategy.DIRECT_RESPONSE
        assert plan.strategy_hints == []
        assert plan.depth_budget == 0

    def test_from_strategy_discovery(self):
        """Discovery strategy creates balanced plan."""
        plan = OrchestrationPlan.from_strategy(
            ExecutionStrategy.DISCOVERY,
            activation_reason="explore_codebase",
        )

        assert plan.activate_rlm is True
        assert plan.strategy == ExecutionStrategy.DISCOVERY
        assert plan.depth_budget == 2
        assert plan.model_tier == ModelTier.BALANCED
        assert len(plan.strategy_hints) > 0
        assert "peek()" in plan.strategy_hints[0]

    def test_from_strategy_recursive_debug(self):
        """Recursive debug strategy creates thorough plan."""
        plan = OrchestrationPlan.from_strategy(
            ExecutionStrategy.RECURSIVE_DEBUG,
            activation_reason="multi_layer_error",
        )

        assert plan.activate_rlm is True
        assert plan.strategy == ExecutionStrategy.RECURSIVE_DEBUG
        assert plan.depth_budget == 3
        assert plan.model_tier == ModelTier.POWERFUL
        assert plan.execution_mode == ExecutionMode.THOROUGH

    def test_from_strategy_map_reduce(self):
        """Map reduce strategy for systematic analysis."""
        plan = OrchestrationPlan.from_strategy(
            ExecutionStrategy.MAP_REDUCE,
            activation_reason="security_review",
        )

        assert plan.activate_rlm is True
        assert plan.strategy == ExecutionStrategy.MAP_REDUCE
        assert "map_reduce()" in plan.strategy_hints[0]

    def test_from_strategy_with_available_models(self):
        """Strategy respects available models."""
        plan = OrchestrationPlan.from_strategy(
            ExecutionStrategy.ARCHITECTURE,
            available_models=["sonnet", "haiku"],  # No opus
        )

        # Should fall back to sonnet since opus not available
        assert plan.primary_model == "sonnet"

    def test_from_strategy_hints_are_copied(self):
        """Strategy hints are copied to avoid mutation."""
        plan1 = OrchestrationPlan.from_strategy(ExecutionStrategy.DISCOVERY)
        plan2 = OrchestrationPlan.from_strategy(ExecutionStrategy.DISCOVERY)

        # Modify one plan's hints
        plan1.strategy_hints.append("extra hint")

        # Other plan should be unaffected
        assert "extra hint" not in plan2.strategy_hints

    def test_from_strategy_serialization(self):
        """Strategy fields serialize correctly."""
        plan = OrchestrationPlan.from_strategy(
            ExecutionStrategy.EXHAUSTIVE_SEARCH,
            activation_reason="find_all_usages",
        )

        data = plan.to_dict()

        assert data["strategy"] == "exhaustive_search"
        assert isinstance(data["strategy_hints"], list)
        assert len(data["strategy_hints"]) > 0

    def test_bypass_uses_direct_response_strategy(self):
        """Bypass plan uses DIRECT_RESPONSE strategy."""
        plan = OrchestrationPlan.bypass("simple_task")

        assert plan.strategy == ExecutionStrategy.DIRECT_RESPONSE
        assert plan.strategy_hints == []


class TestDecisionConfidence:
    """Tests for DecisionConfidence dataclass (SPEC-12.07)."""

    def test_default_values(self):
        """Default values are medium confidence (0.7)."""
        conf = DecisionConfidence()

        assert conf.activation == 0.7
        assert conf.model_tier == 0.7
        assert conf.depth == 0.7
        assert conf.strategy == 0.7

    def test_custom_values(self):
        """Can set custom confidence values."""
        conf = DecisionConfidence(
            activation=0.9,
            model_tier=0.8,
            depth=0.5,
            strategy=0.6,
        )

        assert conf.activation == 0.9
        assert conf.model_tier == 0.8
        assert conf.depth == 0.5
        assert conf.strategy == 0.6

    def test_validation_rejects_negative(self):
        """Negative values are rejected."""
        with pytest.raises(ValueError, match="must be in"):
            DecisionConfidence(activation=-0.1)

    def test_validation_rejects_over_one(self):
        """Values over 1.0 are rejected."""
        with pytest.raises(ValueError, match="must be in"):
            DecisionConfidence(model_tier=1.5)

    def test_average(self):
        """Average computes correctly."""
        conf = DecisionConfidence(
            activation=0.8,
            model_tier=0.6,
            depth=0.4,
            strategy=0.6,
        )

        assert conf.average() == pytest.approx(0.6)

    def test_min_confidence(self):
        """Min confidence returns lowest dimension."""
        conf = DecisionConfidence(
            activation=0.8,
            model_tier=0.6,
            depth=0.3,  # Lowest
            strategy=0.5,
        )

        assert conf.min_confidence() == 0.3

    def test_max_confidence(self):
        """Max confidence returns highest dimension."""
        conf = DecisionConfidence(
            activation=0.9,  # Highest
            model_tier=0.6,
            depth=0.4,
            strategy=0.7,
        )

        assert conf.max_confidence() == 0.9

    def test_low_confidence_dimensions(self):
        """Low confidence dimensions identified correctly."""
        conf = DecisionConfidence(
            activation=0.8,  # Above threshold
            model_tier=0.4,  # Below
            depth=0.3,  # Below
            strategy=0.6,  # Above
        )

        low = conf.low_confidence_dimensions(threshold=0.5)

        assert "model_tier" in low
        assert "depth" in low
        assert "activation" not in low
        assert "strategy" not in low

    def test_low_confidence_dimensions_custom_threshold(self):
        """Custom threshold works correctly."""
        conf = DecisionConfidence(
            activation=0.65,
            model_tier=0.7,
            depth=0.75,
            strategy=0.8,
        )

        # At threshold 0.7, activation should be low
        low = conf.low_confidence_dimensions(threshold=0.7)
        assert "activation" in low
        assert len(low) == 1

    def test_to_dict(self):
        """Serialization to dict works correctly."""
        conf = DecisionConfidence(
            activation=0.85,
            model_tier=0.75,
            depth=0.65,
            strategy=0.55,
        )

        data = conf.to_dict()

        assert data["activation"] == 0.85
        assert data["model_tier"] == 0.75
        assert data["depth"] == 0.65
        assert data["strategy"] == 0.55

    def test_high_factory(self):
        """High confidence factory creates 0.9 values."""
        conf = DecisionConfidence.high()

        assert conf.activation == 0.9
        assert conf.model_tier == 0.9
        assert conf.depth == 0.9
        assert conf.strategy == 0.9

    def test_medium_factory(self):
        """Medium confidence factory creates 0.7 values."""
        conf = DecisionConfidence.medium()

        assert conf.activation == 0.7
        assert conf.model_tier == 0.7
        assert conf.depth == 0.7
        assert conf.strategy == 0.7

    def test_low_factory(self):
        """Low confidence factory creates 0.4 values."""
        conf = DecisionConfidence.low()

        assert conf.activation == 0.4
        assert conf.model_tier == 0.4
        assert conf.depth == 0.4
        assert conf.strategy == 0.4


class TestOrchestrationPlanDecisionConfidence:
    """Tests for decision_confidence in OrchestrationPlan (SPEC-12.07)."""

    def test_default_confidence(self):
        """Plan has medium confidence by default."""
        plan = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
        )

        assert plan.decision_confidence.average() == 0.7

    def test_bypass_has_high_confidence(self):
        """Bypass plan has high confidence."""
        plan = OrchestrationPlan.bypass("simple_task")

        assert plan.decision_confidence.activation == 0.9
        assert plan.decision_confidence.model_tier == 0.9

    def test_from_strategy_has_high_confidence(self):
        """Factory method from_strategy sets high confidence."""
        plan = OrchestrationPlan.from_strategy(
            ExecutionStrategy.RECURSIVE_DEBUG,
            activation_reason="debugging",
        )

        assert plan.decision_confidence.activation == 0.9
        assert plan.decision_confidence.strategy == 0.9

    def test_from_mode_has_high_confidence(self):
        """Factory method from_mode sets high confidence."""
        plan = OrchestrationPlan.from_mode(
            ExecutionMode.THOROUGH,
            activation_reason="user_selected",
        )

        assert plan.decision_confidence.average() == 0.9

    def test_to_dict_includes_confidence(self):
        """Serialization includes decision_confidence."""
        plan = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            decision_confidence=DecisionConfidence(
                activation=0.85,
                model_tier=0.75,
                depth=0.65,
                strategy=0.55,
            ),
        )

        data = plan.to_dict()

        assert "decision_confidence" in data
        assert data["decision_confidence"]["activation"] == 0.85
        assert data["decision_confidence"]["model_tier"] == 0.75
        assert data["decision_confidence"]["depth"] == 0.65
        assert data["decision_confidence"]["strategy"] == 0.55

    def test_confidence_can_be_updated(self):
        """Decision confidence can be modified after creation."""
        plan = OrchestrationPlan(
            activate_rlm=True,
            activation_reason="test",
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
        )

        # Update confidence
        plan.decision_confidence = DecisionConfidence.high()

        assert plan.decision_confidence.activation == 0.9
