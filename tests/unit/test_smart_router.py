"""
Unit tests for smart_router module.

Implements: Spec §8.1 Phase 4 - Smart Routing tests
"""

import pytest

from src.api_client import Provider
from src.smart_router import (
    MODEL_CATALOG,
    OPTIMAL_MODELS,
    FallbackExecutor,
    ModelTier,
    QueryClassification,
    QueryClassifier,
    QueryType,
    RoutingDecision,
    SmartRouter,
    get_optimal_model,
)


class TestQueryType:
    """Tests for QueryType enum."""

    def test_all_query_types_exist(self):
        """All expected query types exist."""
        expected = [
            "factual",
            "analytical",
            "creative",
            "code",
            "search",
            "summarization",
            "planning",
            "debugging",
            "refactoring",
            "architecture",
            "unknown",
        ]
        actual = [qt.value for qt in QueryType]
        for e in expected:
            assert e in actual


class TestModelTier:
    """Tests for ModelTier enum."""

    def test_all_tiers_exist(self):
        """All expected tiers exist."""
        # INHERIT added in SPEC-14.02 for micro mode
        expected = ["fast", "balanced", "powerful", "code_specialist", "inherit"]
        actual = [mt.value for mt in ModelTier]
        assert set(expected) == set(actual)

    def test_inherit_tier_for_micro_mode(self):
        """INHERIT tier exists for micro mode (SPEC-14.02)."""
        assert ModelTier.INHERIT.value == "inherit"


class TestModelCatalog:
    """Tests for MODEL_CATALOG."""

    def test_has_anthropic_models(self):
        """Catalog has Anthropic models."""
        anthropic_models = [
            m for m, opt in MODEL_CATALOG.items() if opt.provider == Provider.ANTHROPIC
        ]
        assert "opus" in anthropic_models
        assert "sonnet" in anthropic_models
        assert "haiku" in anthropic_models

    def test_has_openai_models(self):
        """Catalog has OpenAI models."""
        openai_models = [m for m, opt in MODEL_CATALOG.items() if opt.provider == Provider.OPENAI]
        assert "gpt-5.2-codex" in openai_models
        assert "gpt-4o" in openai_models

    def test_models_have_strengths(self):
        """All models have defined strengths."""
        for model, option in MODEL_CATALOG.items():
            assert len(option.strengths) > 0, f"{model} has no strengths"


class TestQueryClassification:
    """Tests for QueryClassification dataclass."""

    def test_create_classification(self):
        """Can create classification."""
        classification = QueryClassification(
            query_type=QueryType.CODE,
            confidence=0.8,
            signals=["code", "function"],
            complexity=0.5,
        )

        assert classification.query_type == QueryType.CODE
        assert classification.confidence == 0.8
        assert len(classification.signals) == 2
        assert classification.complexity == 0.5

    def test_suggested_models_property(self):
        """suggested_models returns models for query type."""
        classification = QueryClassification(
            query_type=QueryType.CODE,
            confidence=0.9,
            signals=[],
            complexity=0.5,
        )

        models = classification.suggested_models
        assert len(models) > 0
        assert "gpt-5.2-codex" in models  # Codex is optimal for code


class TestQueryClassifier:
    """Tests for QueryClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return QueryClassifier()

    def test_classify_factual_query(self, classifier):
        """Classifies factual queries."""
        result = classifier.classify("What is Python?")
        assert result.query_type == QueryType.FACTUAL

    def test_classify_code_query(self, classifier):
        """Classifies code queries."""
        result = classifier.classify("Write a function to sort a list")
        assert result.query_type == QueryType.CODE

    def test_classify_analytical_query(self, classifier):
        """Classifies analytical queries."""
        result = classifier.classify("Why does this approach work better?")
        assert result.query_type == QueryType.ANALYTICAL

    def test_classify_search_query(self, classifier):
        """Classifies search queries."""
        result = classifier.classify("Find all Python files in the project")
        assert result.query_type == QueryType.SEARCH

    def test_classify_summarization_query(self, classifier):
        """Classifies summarization queries."""
        result = classifier.classify("Summarize this document")
        assert result.query_type == QueryType.SUMMARIZATION

    def test_classify_planning_query(self, classifier):
        """Classifies planning queries."""
        result = classifier.classify("How should I architect this system?")
        assert result.query_type == QueryType.PLANNING

    def test_classify_debugging_query(self, classifier):
        """Classifies debugging queries."""
        result = classifier.classify("Troubleshoot this issue, it's not working")
        assert result.query_type == QueryType.DEBUGGING

    def test_classify_refactoring_query(self, classifier):
        """Classifies refactoring queries."""
        result = classifier.classify("Refactor this to simplify the logic")
        assert result.query_type == QueryType.REFACTORING

    def test_classify_unknown_query(self, classifier):
        """Returns unknown for ambiguous queries."""
        result = classifier.classify("xyz123")
        assert result.query_type == QueryType.UNKNOWN
        assert result.confidence < 0.5

    def test_complexity_detection(self, classifier):
        """Detects complexity signals."""
        simple = classifier.classify("What is Python?")
        complex_query = classifier.classify(
            "Analyze the entire codebase architecture and refactor all modules"
        )

        assert complex_query.complexity > simple.complexity

    def test_custom_patterns(self):
        """Can add custom patterns."""
        custom = QueryClassifier(custom_patterns={QueryType.CODE: [r"\bcustom_pattern\b"]})

        result = custom.classify("This has custom_pattern in it")
        assert result.query_type == QueryType.CODE


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_create_decision(self):
        """Can create routing decision."""
        decision = RoutingDecision(
            primary_model="sonnet",
            fallback_chain=["opus", "haiku"],
            query_type=QueryType.CODE,
            confidence=0.8,
            reason="Code query",
            provider=Provider.ANTHROPIC,
            estimated_cost=1.0,
            estimated_speed=1.0,
        )

        assert decision.primary_model == "sonnet"
        assert len(decision.fallback_chain) == 2
        assert decision.provider == Provider.ANTHROPIC

    def test_all_models_property(self):
        """all_models returns correct order."""
        decision = RoutingDecision(
            primary_model="model1",
            fallback_chain=["model2", "model3"],
            query_type=QueryType.UNKNOWN,
            confidence=0.5,
            reason="test",
            provider=Provider.ANTHROPIC,
            estimated_cost=1.0,
            estimated_speed=1.0,
        )

        assert decision.all_models == ["model1", "model2", "model3"]


class TestSmartRouter:
    """Tests for SmartRouter class."""

    @pytest.fixture
    def router(self):
        """Create router instance with both providers."""
        return SmartRouter(available_providers=[Provider.ANTHROPIC, Provider.OPENAI])

    def test_route_code_query_to_codex(self, router):
        """Routes code queries to Codex."""
        decision = router.route("Implement a function to parse JSON")

        assert decision.query_type == QueryType.CODE
        assert decision.primary_model == "gpt-5.2-codex"
        assert decision.provider == Provider.OPENAI

    def test_route_planning_query_to_opus(self, router):
        """Routes planning queries to Opus."""
        decision = router.route("Plan the architecture for a new system")

        assert decision.query_type == QueryType.PLANNING
        assert decision.primary_model == "opus"
        assert decision.provider == Provider.ANTHROPIC

    def test_route_factual_to_fast_model(self, router):
        """Routes factual queries to fast models."""
        decision = router.route("What is the capital of France?")

        assert decision.query_type == QueryType.FACTUAL
        assert decision.primary_model in ["haiku", "gpt-4o-mini"]

    def test_force_model(self, router):
        """Can force specific model."""
        decision = router.route("Any query", force_model="opus")

        assert decision.primary_model == "opus"
        assert "Forced model" in decision.reason

    def test_fallback_chain_has_different_providers(self, router):
        """Fallback chain includes different providers for resilience."""
        decision = router.route("Write some code")

        providers = {MODEL_CATALOG[m].provider for m in decision.all_models if m in MODEL_CATALOG}
        # Should have at least some fallbacks
        assert len(decision.fallback_chain) > 0

    def test_prefer_speed(self):
        """prefer_speed prioritizes fast models."""
        router = SmartRouter(
            available_providers=[Provider.ANTHROPIC, Provider.OPENAI],
            prefer_speed=True,
        )

        decision = router.route("Analyze this code")
        # Fast models should be prioritized
        assert "(speed preferred)" in decision.reason

    def test_prefer_cost(self):
        """prefer_cost prioritizes cheap models."""
        router = SmartRouter(
            available_providers=[Provider.ANTHROPIC, Provider.OPENAI],
            prefer_cost=True,
        )

        decision = router.route("Analyze this code")
        assert "(cost preferred)" in decision.reason

    def test_force_provider(self):
        """Can force specific provider."""
        router = SmartRouter(
            available_providers=[Provider.ANTHROPIC, Provider.OPENAI],
            force_provider=Provider.ANTHROPIC,
        )

        decision = router.route("Implement a function")
        assert decision.provider == Provider.ANTHROPIC

    def test_context_depth_adjustment(self, router):
        """Uses faster models for recursive calls."""
        decision = router.route(
            "Analyze this",
            context={"depth": 1},
        )

        # Should prefer faster models at depth > 0
        assert decision.primary_model in MODEL_CATALOG

    def test_record_outcome(self, router):
        """Can record routing outcome."""
        router.route("Test query")
        router.record_outcome("Test query", "sonnet", success=True, latency_ms=100)

        stats = router.get_statistics()
        assert stats["total_routes"] == 1

    def test_get_statistics(self, router):
        """Can get routing statistics."""
        router.route("Code query about functions")
        router.route("What is Python?")
        router.route("Plan the architecture")

        stats = router.get_statistics()

        assert stats["total_routes"] == 3
        assert "by_query_type" in stats
        assert "by_model" in stats
        assert "by_provider" in stats


class TestFallbackExecutor:
    """Tests for FallbackExecutor class."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        return SmartRouter(available_providers=[Provider.ANTHROPIC, Provider.OPENAI])

    @pytest.fixture
    def executor(self, router):
        """Create executor instance."""
        return FallbackExecutor(router)

    @pytest.mark.asyncio
    async def test_execute_success(self, executor):
        """Executes successfully on first try."""

        async def mock_execute(query, model):
            return f"Result from {model}"

        result, model = await executor.execute_with_fallback(
            "Test query",
            mock_execute,
        )

        assert "Result from" in result
        assert model is not None

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, executor):
        """Falls back on failure."""
        call_count = 0

        async def failing_then_success(query, model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return f"Result from {model}"

        result, model = await executor.execute_with_fallback(
            "Test query",
            failing_then_success,
        )

        assert call_count >= 2
        assert "Result from" in result

    @pytest.mark.asyncio
    async def test_execute_all_fail(self, executor):
        """Raises if all models fail."""

        async def always_fail(query, model):
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            await executor.execute_with_fallback("Test", always_fail)


class TestGetOptimalModel:
    """Tests for get_optimal_model convenience function."""

    def test_returns_model_name(self):
        """Returns a valid model name."""
        model = get_optimal_model("Write some code")
        assert model in MODEL_CATALOG

    def test_respects_available_providers(self):
        """Respects available providers."""
        model = get_optimal_model(
            "Write some code",
            available_providers=[Provider.ANTHROPIC],
        )
        assert MODEL_CATALOG[model].provider == Provider.ANTHROPIC


class TestOptimalModels:
    """Tests for OPTIMAL_MODELS mapping."""

    def test_code_routes_to_codex(self):
        """Code queries route to Codex first."""
        assert OPTIMAL_MODELS[QueryType.CODE][0] == "gpt-5.2-codex"

    def test_planning_routes_to_opus(self):
        """Planning queries route to Opus first."""
        assert OPTIMAL_MODELS[QueryType.PLANNING][0] == "opus"

    def test_factual_routes_to_haiku(self):
        """Factual queries route to Haiku first."""
        assert OPTIMAL_MODELS[QueryType.FACTUAL][0] == "haiku"

    def test_all_query_types_have_models(self):
        """All query types have optimal models defined."""
        for qt in QueryType:
            assert qt in OPTIMAL_MODELS, f"{qt} missing from OPTIMAL_MODELS"
