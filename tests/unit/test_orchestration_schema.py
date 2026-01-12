"""
Unit tests for orchestration_schema module.

Implements: Spec §8.1 Phase 2 - Orchestration Layer tests
"""

import pytest

from src.orchestration_schema import (
    ExecutionMode,
    MODE_DEFAULTS,
    OrchestrationContext,
    OrchestrationPlan,
    PlanAdjustment,
    TIER_MODELS,
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
        """All model tiers have associated models."""
        for tier in ModelTier:
            assert tier in TIER_MODELS
            assert len(TIER_MODELS[tier]) > 0

    def test_fast_tier_models(self):
        """Fast tier includes fast models."""
        models = TIER_MODELS[ModelTier.FAST]
        assert "haiku" in models
        assert "gpt-4o-mini" in models

    def test_powerful_tier_models(self):
        """Powerful tier includes powerful models."""
        models = TIER_MODELS[ModelTier.POWERFUL]
        assert "opus" in models
