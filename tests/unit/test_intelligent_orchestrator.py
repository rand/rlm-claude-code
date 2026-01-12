"""
Unit tests for intelligent_orchestrator module.

Implements: Spec §8.1 Phase 2 - Orchestration Layer tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.intelligent_orchestrator import (
    IntelligentOrchestrator,
    OrchestratorConfig,
    create_orchestration_plan,
    ORCHESTRATOR_SYSTEM_PROMPT,
)
from src.orchestration_schema import (
    ExecutionMode,
    OrchestrationContext,
    OrchestrationPlan,
    ToolAccessLevel,
)
from src.smart_router import ModelTier, QueryType
from src.types import SessionContext


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = OrchestratorConfig()

        assert config.orchestrator_model == "haiku"
        assert config.timeout_ms == 5000
        assert config.max_tokens == 500
        assert config.use_fallback is True
        assert config.cache_enabled is True
        assert config.cache_size == 100

    def test_custom_config(self):
        """Can create custom config."""
        config = OrchestratorConfig(
            orchestrator_model="sonnet",
            timeout_ms=10000,
            use_fallback=False,
        )

        assert config.orchestrator_model == "sonnet"
        assert config.timeout_ms == 10000
        assert config.use_fallback is False


class TestIntelligentOrchestrator:
    """Tests for IntelligentOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock client."""
        return IntelligentOrchestrator(
            client=None,
            available_models=["sonnet", "haiku", "opus"],
        )

    def test_initialization(self, orchestrator):
        """Orchestrator initializes correctly."""
        assert orchestrator.available_models == ["sonnet", "haiku", "opus"]
        assert orchestrator.config.orchestrator_model == "haiku"
        assert orchestrator._stats["llm_decisions"] == 0

    def test_heuristic_orchestrate_simple_task(self, orchestrator):
        """Heuristic fallback handles simple tasks."""
        context = OrchestrationContext(
            query="read file.py",
            context_tokens=1000,
        )

        plan = orchestrator._heuristic_orchestrate("read file.py", context)

        # Simple task should not activate RLM
        assert plan.activate_rlm is False
        assert plan.activation_reason == "simple_task"

    def test_heuristic_orchestrate_complex_task(self, orchestrator):
        """Heuristic fallback handles complex tasks."""
        context = OrchestrationContext(
            query="why is auth.py failing when I import db.py and config.py?",
            context_tokens=50000,
        )

        plan = orchestrator._heuristic_orchestrate(
            "why is auth.py failing when I import db.py and config.py?",
            context,
        )

        # Complex debugging task should activate RLM
        assert plan.activate_rlm is True
        assert plan.primary_model in orchestrator.available_models

    def test_select_model_fast_tier(self, orchestrator):
        """Selects appropriate model for fast tier."""
        model = orchestrator._select_model(ModelTier.FAST)
        assert model == "haiku"

    def test_select_model_balanced_tier(self, orchestrator):
        """Selects appropriate model for balanced tier."""
        model = orchestrator._select_model(ModelTier.BALANCED)
        assert model == "sonnet"

    def test_select_model_powerful_tier(self, orchestrator):
        """Selects appropriate model for powerful tier."""
        model = orchestrator._select_model(ModelTier.POWERFUL)
        assert model == "opus"

    def test_select_model_unavailable(self):
        """Falls back when preferred model unavailable."""
        orchestrator = IntelligentOrchestrator(
            available_models=["haiku"],
        )
        model = orchestrator._select_model(ModelTier.POWERFUL)
        assert model == "haiku"  # Falls back to available

    def test_get_fallbacks(self, orchestrator):
        """Gets fallback models correctly."""
        fallbacks = orchestrator._get_fallbacks(ModelTier.BALANCED, "sonnet")
        assert len(fallbacks) <= 2
        assert "sonnet" not in fallbacks

    def test_cache_key_computation(self, orchestrator):
        """Cache key computation is consistent."""
        context = OrchestrationContext(query="test", context_tokens=5000)

        key1 = orchestrator._compute_cache_key("test query", context)
        key2 = orchestrator._compute_cache_key("test query", context)

        assert key1 == key2

    def test_cache_key_differs_for_different_queries(self, orchestrator):
        """Different queries get different cache keys."""
        context = OrchestrationContext(query="test", context_tokens=5000)

        key1 = orchestrator._compute_cache_key("query one", context)
        key2 = orchestrator._compute_cache_key("query two", context)

        assert key1 != key2

    def test_summarize_context(self, orchestrator):
        """Context summary is informative."""
        context = OrchestrationContext(
            query="test",
            context_tokens=50000,
            current_depth=1,
            budget_remaining_dollars=3.50,
            budget_remaining_tokens=75000,
        )

        summary = orchestrator._summarize_context(context)

        assert "50000" in summary or "50,000" in summary
        assert "depth: 1" in summary.lower()
        assert "3.50" in summary

    def test_statistics_tracking(self, orchestrator):
        """Statistics are tracked correctly."""
        stats = orchestrator.get_statistics()

        assert "llm_decisions" in stats
        assert "fallback_decisions" in stats
        assert "cache_hits" in stats
        assert "total_decisions" in stats


class TestParseDecision:
    """Tests for decision parsing."""

    @pytest.fixture
    def orchestrator(self):
        return IntelligentOrchestrator(available_models=["sonnet", "haiku", "opus"])

    def test_parse_valid_json(self, orchestrator):
        """Parses valid JSON response."""
        context = OrchestrationContext(query="test", context_tokens=5000)
        response = """{
            "activate_rlm": true,
            "activation_reason": "debugging task",
            "execution_mode": "balanced",
            "model_tier": "balanced",
            "depth_budget": 2,
            "tool_access": "read_only",
            "query_type": "debugging",
            "complexity_score": 0.7,
            "signals": ["multi_file", "error"]
        }"""

        plan = orchestrator._parse_decision(response, "test", context)

        assert plan.activate_rlm is True
        assert plan.activation_reason == "debugging task"
        assert plan.execution_mode == ExecutionMode.BALANCED
        assert plan.model_tier == ModelTier.BALANCED
        assert plan.depth_budget == 2
        assert plan.tool_access == ToolAccessLevel.READ_ONLY
        assert plan.query_type == QueryType.DEBUGGING
        assert plan.complexity_score == 0.7
        assert "multi_file" in plan.signals

    def test_parse_fast_mode(self, orchestrator):
        """Parses fast mode correctly."""
        context = OrchestrationContext(query="test", context_tokens=5000)
        response = """{
            "activate_rlm": false,
            "activation_reason": "simple query",
            "execution_mode": "fast",
            "model_tier": "fast",
            "depth_budget": 0,
            "tool_access": "none",
            "query_type": "factual",
            "complexity_score": 0.1,
            "signals": []
        }"""

        plan = orchestrator._parse_decision(response, "test", context)

        assert plan.activate_rlm is False
        assert plan.execution_mode == ExecutionMode.FAST
        assert plan.model_tier == ModelTier.FAST
        assert plan.depth_budget == 0
        assert plan.tool_access == ToolAccessLevel.NONE

    def test_parse_thorough_mode(self, orchestrator):
        """Parses thorough mode correctly."""
        context = OrchestrationContext(query="test", context_tokens=5000)
        response = """{
            "activate_rlm": true,
            "activation_reason": "complex analysis",
            "execution_mode": "thorough",
            "model_tier": "powerful",
            "depth_budget": 3,
            "tool_access": "full",
            "query_type": "architecture",
            "complexity_score": 0.95,
            "signals": ["codebase", "comprehensive"]
        }"""

        plan = orchestrator._parse_decision(response, "test", context)

        assert plan.activate_rlm is True
        assert plan.execution_mode == ExecutionMode.THOROUGH
        assert plan.model_tier == ModelTier.POWERFUL
        assert plan.depth_budget == 3
        assert plan.tool_access == ToolAccessLevel.FULL

    def test_parse_clamps_depth_budget(self, orchestrator):
        """Clamps depth budget to valid range."""
        context = OrchestrationContext(query="test", context_tokens=5000)
        response = """{
            "activate_rlm": true,
            "activation_reason": "test",
            "execution_mode": "balanced",
            "model_tier": "balanced",
            "depth_budget": 10,
            "tool_access": "read_only",
            "query_type": "unknown",
            "complexity_score": 0.5,
            "signals": []
        }"""

        plan = orchestrator._parse_decision(response, "test", context)
        assert plan.depth_budget == 3  # Clamped to max

    def test_parse_invalid_json_raises(self, orchestrator):
        """Raises on invalid JSON."""
        context = OrchestrationContext(query="test", context_tokens=5000)

        with pytest.raises(ValueError, match="No JSON found"):
            orchestrator._parse_decision("not json at all", "test", context)

    def test_parse_json_with_surrounding_text(self, orchestrator):
        """Extracts JSON from surrounding text."""
        context = OrchestrationContext(query="test", context_tokens=5000)
        response = """Here is my decision:
        {"activate_rlm": true, "activation_reason": "test", "execution_mode": "balanced", "model_tier": "balanced", "depth_budget": 2, "tool_access": "read_only", "query_type": "code", "complexity_score": 0.5, "signals": []}
        That's my analysis."""

        plan = orchestrator._parse_decision(response, "test", context)
        assert plan.activate_rlm is True


class TestForcedOverrides:
    """Tests for user-forced overrides."""

    @pytest.fixture
    def orchestrator(self):
        return IntelligentOrchestrator(available_models=["sonnet", "haiku"])

    @pytest.mark.asyncio
    async def test_forced_rlm_off(self, orchestrator):
        """Respects forced RLM off."""
        context = OrchestrationContext(
            query="complex query",
            context_tokens=50000,
            forced_rlm=False,
        )

        plan = await orchestrator.create_plan("complex query", context)

        assert plan.activate_rlm is False
        assert plan.activation_reason == "user_forced_off"

    @pytest.mark.asyncio
    async def test_forced_mode(self, orchestrator):
        """Respects forced execution mode."""
        context = OrchestrationContext(
            query="any query",
            context_tokens=5000,
            forced_mode=ExecutionMode.THOROUGH,
        )

        plan = await orchestrator.create_plan("any query", context)

        assert plan.activate_rlm is True
        assert plan.execution_mode == ExecutionMode.THOROUGH
        assert plan.activation_reason == "user_forced_mode"


class TestCaching:
    """Tests for decision caching."""

    def test_cache_stores_decision(self):
        """Cache stores decisions."""
        orchestrator = IntelligentOrchestrator(
            config=OrchestratorConfig(cache_enabled=True, cache_size=10),
        )

        plan = OrchestrationPlan.bypass("test")
        orchestrator._update_cache("key1", plan)

        assert "key1" in orchestrator._decision_cache
        assert orchestrator._decision_cache["key1"] == plan

    def test_cache_evicts_old_entries(self):
        """Cache evicts old entries when full."""
        orchestrator = IntelligentOrchestrator(
            config=OrchestratorConfig(cache_enabled=True, cache_size=4),
        )

        # Fill cache
        for i in range(5):
            plan = OrchestrationPlan.bypass(f"test{i}")
            orchestrator._update_cache(f"key{i}", plan)

        # Should have evicted some entries
        assert len(orchestrator._decision_cache) <= 4


class TestSystemPrompt:
    """Tests for orchestrator system prompt."""

    def test_prompt_contains_modes(self):
        """Prompt documents execution modes."""
        assert "fast" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "balanced" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "thorough" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_prompt_contains_tool_access_levels(self):
        """Prompt documents tool access levels."""
        assert "none" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "repl_only" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "read_only" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "full" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_prompt_contains_json_format(self):
        """Prompt shows expected JSON format."""
        assert "activate_rlm" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "execution_mode" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "depth_budget" in ORCHESTRATOR_SYSTEM_PROMPT


class TestSessionContextConversion:
    """Tests for SessionContext to OrchestrationContext conversion."""

    @pytest.mark.asyncio
    async def test_session_context_conversion(self):
        """SessionContext is converted to OrchestrationContext."""
        orchestrator = IntelligentOrchestrator(
            config=OrchestratorConfig(use_fallback=True),
            available_models=["sonnet"],
        )

        session_context = SessionContext(
            messages=[],
            files={"file.py": "content"},
            tool_outputs=[],
        )

        # Force heuristic path for testing
        orchestrator._client = None

        plan = await orchestrator.create_plan("simple query", session_context)

        # Should get a valid plan
        assert isinstance(plan, OrchestrationPlan)


class TestConvenienceFunction:
    """Tests for create_orchestration_plan function."""

    @pytest.mark.asyncio
    async def test_heuristic_only_mode(self):
        """Can use heuristics-only mode."""
        context = SessionContext(
            messages=[],
            files={},
            tool_outputs=[],
        )

        plan = await create_orchestration_plan(
            "simple query",
            context,
            use_llm=False,
        )

        assert isinstance(plan, OrchestrationPlan)
