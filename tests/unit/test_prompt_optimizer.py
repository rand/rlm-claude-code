"""
Unit tests for prompt_optimizer module.

Implements: Spec §8.1 Phase 3 - Prompt Optimization tests
"""

import tempfile
from pathlib import Path

import pytest

from src.prompt_optimizer import (
    PromptLibrary,
    PromptResult,
    PromptType,
    PromptVariant,
    StrategySelector,
    StrategyType,
    VariantStats,
)


class TestPromptVariant:
    """Tests for PromptVariant dataclass."""

    def test_create_variant(self):
        """Can create prompt variant."""
        variant = PromptVariant(
            id="test_v1",
            prompt_type=PromptType.ROOT,
            template="Hello {name}!",
            description="Test prompt",
        )

        assert variant.id == "test_v1"
        assert variant.prompt_type == PromptType.ROOT


class TestVariantStats:
    """Tests for VariantStats dataclass."""

    def test_default_values(self):
        """Has expected default values."""
        stats = VariantStats(variant_id="test")

        assert stats.uses == 0
        assert stats.successes == 0
        assert stats.success_rate == 0.0

    def test_success_rate(self):
        """Calculates success rate correctly."""
        stats = VariantStats(
            variant_id="test",
            uses=10,
            successes=7,
            failures=3,
        )

        assert stats.success_rate == 0.7

    def test_avg_tokens(self):
        """Calculates average tokens correctly."""
        stats = VariantStats(
            variant_id="test",
            uses=5,
            total_tokens=5000,
        )

        assert stats.avg_tokens == 1000.0

    def test_feedback_score(self):
        """Calculates feedback score correctly."""
        stats = VariantStats(
            variant_id="test",
            positive_feedback=7,
            negative_feedback=3,
        )

        # (7 - 3) / 10 = 0.4
        assert stats.feedback_score == 0.4


class TestPromptLibrary:
    """Tests for PromptLibrary class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def library(self, temp_storage):
        """Create library with temp storage."""
        return PromptLibrary(storage_path=temp_storage)

    def test_loads_default_variants(self, library):
        """Loads default variants on init."""
        root_variants = library.get_variants_by_type(PromptType.ROOT)
        assert len(root_variants) >= 2

    def test_add_variant(self, library):
        """Can add custom variant."""
        variant = PromptVariant(
            id="custom_v1",
            prompt_type=PromptType.ROOT,
            template="Custom: {query}",
            description="Custom prompt",
        )

        library.add_variant(variant)
        assert library.get_variant("custom_v1") is not None

    def test_get_variants_by_type(self, library):
        """Gets variants filtered by type."""
        root_variants = library.get_variants_by_type(PromptType.ROOT)
        recursive_variants = library.get_variants_by_type(PromptType.RECURSIVE)

        assert all(v.prompt_type == PromptType.ROOT for v in root_variants)
        assert all(v.prompt_type == PromptType.RECURSIVE for v in recursive_variants)

    def test_select_variant_random(self, library):
        """Can select variant randomly."""
        variant = library.select_variant(PromptType.ROOT, strategy="random")
        assert variant.prompt_type == PromptType.ROOT

    def test_select_variant_epsilon_greedy(self, library):
        """Can select using epsilon-greedy strategy."""
        # Record some results to create preference
        library.record_result(
            PromptResult(
                variant_id="root_v1",
                success=True,
                tokens_used=1000,
                execution_time_ms=500,
            )
        )

        variant = library.select_variant(
            PromptType.ROOT,
            strategy="epsilon_greedy",
            epsilon=0.0,  # Always exploit
        )

        # Should select the one with recorded success
        assert variant is not None

    def test_render_prompt(self, library):
        """Can render prompt with variables."""
        variant = PromptVariant(
            id="test",
            prompt_type=PromptType.ROOT,
            template="Query: {query}\nContext: {context}",
            description="Test",
        )

        rendered = library.render_prompt(
            variant,
            query="What is this?",
            context="Some context here",
        )

        assert "What is this?" in rendered
        assert "Some context here" in rendered

    def test_render_prompt_missing_var(self, library):
        """Handles missing template variables."""
        variant = PromptVariant(
            id="test",
            prompt_type=PromptType.ROOT,
            template="Query: {query}",
            description="Test",
        )

        rendered = library.render_prompt(variant)  # No kwargs
        assert "Missing variable" in rendered

    def test_record_result(self, library):
        """Can record prompt results."""
        library.record_result(
            PromptResult(
                variant_id="root_v1",
                success=True,
                tokens_used=1500,
                execution_time_ms=800,
            )
        )

        stats = library.get_stats("root_v1")
        assert stats.uses == 1
        assert stats.successes == 1
        assert stats.total_tokens == 1500

    def test_record_multiple_results(self, library):
        """Aggregates multiple results correctly."""
        for i in range(5):
            library.record_result(
                PromptResult(
                    variant_id="root_v1",
                    success=i < 3,  # 3 successes, 2 failures
                    tokens_used=1000,
                    execution_time_ms=500,
                )
            )

        stats = library.get_stats("root_v1")
        assert stats.uses == 5
        assert stats.successes == 3
        assert stats.failures == 2

    def test_record_feedback(self, library):
        """Records user feedback."""
        library.record_result(
            PromptResult(
                variant_id="root_v1",
                success=True,
                tokens_used=1000,
                execution_time_ms=500,
                user_feedback=1,  # Positive
            )
        )
        library.record_result(
            PromptResult(
                variant_id="root_v1",
                success=True,
                tokens_used=1000,
                execution_time_ms=500,
                user_feedback=-1,  # Negative
            )
        )

        stats = library.get_stats("root_v1")
        assert stats.positive_feedback == 1
        assert stats.negative_feedback == 1

    def test_get_all_stats(self, library):
        """Gets stats for all variants."""
        all_stats = library.get_all_stats()
        assert len(all_stats) > 0

    def test_get_recommendations(self, library):
        """Gets recommendations for improvement."""
        # Record enough data for recommendations
        for i in range(15):
            library.record_result(
                PromptResult(
                    variant_id="root_v1",
                    success=i < 5,  # Low success rate
                    tokens_used=6000,  # High token usage
                    execution_time_ms=1000,
                    user_feedback=-1 if i < 8 else 1,  # Negative feedback
                )
            )

        recommendations = library.get_recommendations(PromptType.ROOT)

        # Should have recommendations for issues
        assert len(recommendations) > 0


class TestStrategySelector:
    """Tests for StrategySelector class."""

    @pytest.fixture
    def selector(self):
        """Create strategy selector."""
        return StrategySelector()

    def test_select_direct_strategy(self, selector):
        """Selects direct strategy for simple queries."""
        strategy = selector.select_strategy(
            "What is the purpose of this function?",
            context_size=5000,
        )

        assert strategy == StrategyType.DIRECT

    def test_select_decompose_strategy(self, selector):
        """Selects decompose for multi-step queries."""
        strategy = selector.select_strategy(
            "First find the bug, then fix it and update the tests",
            context_size=10000,
        )

        assert strategy == StrategyType.DECOMPOSE

    def test_select_search_strategy(self, selector):
        """Selects search-first for find queries."""
        strategy = selector.select_strategy(
            "Find all usages of the deprecated API",
            context_size=10000,
        )

        assert strategy == StrategyType.SEARCH_FIRST

    def test_select_summarize_strategy(self, selector):
        """Selects summarize-first for large contexts."""
        strategy = selector.select_strategy(
            "Analyze this code",
            context_size=60000,  # Large context
        )

        assert strategy == StrategyType.SUMMARIZE_FIRST

    def test_select_summarize_for_summary_query(self, selector):
        """Selects summarize for explicit summary requests."""
        strategy = selector.select_strategy(
            "Give me an overview of the authentication system",
            context_size=10000,
        )

        assert strategy == StrategyType.SUMMARIZE_FIRST

    def test_default_for_code_context(self, selector):
        """Defaults to decompose for code contexts."""
        strategy = selector.select_strategy(
            "Review this implementation",
            context_size=15000,
            has_code=True,
        )

        assert strategy == StrategyType.DECOMPOSE

    def test_default_for_small_context(self, selector):
        """Defaults to direct for small contexts."""
        strategy = selector.select_strategy(
            "Analyze this",
            context_size=5000,
            has_code=False,
        )

        assert strategy == StrategyType.DIRECT
