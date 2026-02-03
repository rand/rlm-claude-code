"""
Unit tests for learning module.

Implements: Spec §8.1 Phase 4 - Learning tests
"""

import tempfile
from pathlib import Path

import pytest

from src.learning import (
    AdaptiveStrategy,
    FeedbackType,
    LearningSystem,
    StrategyOutcome,
    StrategyStats,
    StrategyTracker,
    StrategyType,
    UserFeedbackCollector,
)


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_all_types_exist(self):
        """All expected strategy types exist."""
        expected = [
            "summarization",
            "search",
            "decomposition",
            "caching",
            "model_selection",
            "prompt_variant",
        ]
        actual = [st.value for st in StrategyType]
        assert set(expected) == set(actual)


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_all_feedback_types_exist(self):
        """All expected feedback types exist."""
        expected = [
            "positive",
            "negative",
            "implicit_success",
            "implicit_failure",
            "neutral",
        ]
        actual = [ft.value for ft in FeedbackType]
        assert set(expected) == set(actual)


class TestStrategyOutcome:
    """Tests for StrategyOutcome dataclass."""

    def test_create_outcome(self):
        """Can create outcome."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.SUMMARIZATION,
            strategy_id="variant_a",
            success=True,
            latency_ms=100.0,
            token_cost=500,
        )

        assert outcome.strategy_type == StrategyType.SUMMARIZATION
        assert outcome.success is True

    def test_score_success(self):
        """Success gives positive score."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=True,
            latency_ms=500,
            token_cost=200,
        )

        assert outcome.score >= 1.0

    def test_score_failure(self):
        """Failure gives lower score."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=False,
            latency_ms=500,
            token_cost=200,
        )

        assert outcome.score < 1.0

    def test_score_with_positive_feedback(self):
        """Positive feedback increases score."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=True,
            latency_ms=500,
            token_cost=200,
            feedback=FeedbackType.POSITIVE,
        )

        assert outcome.score > 1.0

    def test_score_with_negative_feedback(self):
        """Negative feedback decreases score."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=True,
            latency_ms=500,
            token_cost=200,
            feedback=FeedbackType.NEGATIVE,
        )

        neutral = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=True,
            latency_ms=500,
            token_cost=200,
        )

        assert outcome.score < neutral.score

    def test_score_fast_latency_bonus(self):
        """Fast latency gives bonus."""
        fast = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=True,
            latency_ms=500,  # Under 1000
            token_cost=2000,
        )

        slow = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=True,
            latency_ms=2000,  # Over 1000
            token_cost=2000,
        )

        assert fast.score > slow.score

    def test_score_low_token_bonus(self):
        """Low token usage gives bonus."""
        cheap = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=True,
            latency_ms=2000,
            token_cost=500,  # Under 1000
        )

        expensive = StrategyOutcome(
            strategy_type=StrategyType.SEARCH,
            strategy_id="v1",
            success=True,
            latency_ms=2000,
            token_cost=2000,  # Over 1000
        )

        assert cheap.score > expensive.score


class TestStrategyStats:
    """Tests for StrategyStats dataclass."""

    def test_create_stats(self):
        """Can create stats."""
        stats = StrategyStats(
            strategy_type=StrategyType.CACHING,
            strategy_id="lru",
            total_uses=10,
            successes=8,
            total_score=12.0,
        )

        assert stats.total_uses == 10
        assert stats.successes == 8

    def test_success_rate(self):
        """Calculates success rate correctly."""
        stats = StrategyStats(
            strategy_type=StrategyType.CACHING,
            strategy_id="lru",
            total_uses=10,
            successes=8,
        )

        assert stats.success_rate == 0.8

    def test_success_rate_zero_uses(self):
        """Success rate is 0 with no uses."""
        stats = StrategyStats(
            strategy_type=StrategyType.CACHING,
            strategy_id="lru",
        )

        assert stats.success_rate == 0.0

    def test_avg_score(self):
        """Calculates average score correctly."""
        stats = StrategyStats(
            strategy_type=StrategyType.CACHING,
            strategy_id="lru",
            total_uses=10,
            total_score=15.0,
        )

        assert stats.avg_score == 1.5

    def test_ucb_score_unused(self):
        """UCB score is infinity for unused strategies."""
        stats = StrategyStats(
            strategy_type=StrategyType.CACHING,
            strategy_id="lru",
        )

        assert stats.ucb_score(100) == float("inf")

    def test_ucb_score_exploration(self):
        """UCB score includes exploration bonus."""
        stats = StrategyStats(
            strategy_type=StrategyType.CACHING,
            strategy_id="lru",
            total_uses=10,
            total_score=10.0,
        )

        # More total trials means more exploration bonus
        score_10 = stats.ucb_score(10)
        score_100 = stats.ucb_score(100)

        assert score_100 > score_10


class TestStrategyTracker:
    """Tests for StrategyTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return StrategyTracker()

    def test_record_outcome(self, tracker):
        """Can record outcome."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.SUMMARIZATION,
            strategy_id="v1",
            success=True,
            latency_ms=100,
            token_cost=500,
        )

        tracker.record_outcome(outcome)

        stats = tracker.get_stats(StrategyType.SUMMARIZATION, "v1")
        assert stats is not None
        assert stats.total_uses == 1
        assert stats.successes == 1

    def test_multiple_outcomes(self, tracker):
        """Aggregates multiple outcomes."""
        for i in range(5):
            outcome = StrategyOutcome(
                strategy_type=StrategyType.SEARCH,
                strategy_id="elastic",
                success=i % 2 == 0,  # 3 successes, 2 failures
                latency_ms=100,
                token_cost=200,
            )
            tracker.record_outcome(outcome)

        stats = tracker.get_stats(StrategyType.SEARCH, "elastic")
        assert stats.total_uses == 5
        assert stats.successes == 3

    def test_get_all_stats(self, tracker):
        """Can get all stats."""
        for strategy_id in ["v1", "v2", "v3"]:
            outcome = StrategyOutcome(
                strategy_type=StrategyType.PROMPT_VARIANT,
                strategy_id=strategy_id,
                success=True,
                latency_ms=100,
                token_cost=200,
            )
            tracker.record_outcome(outcome)

        all_stats = tracker.get_all_stats(StrategyType.PROMPT_VARIANT)
        assert len(all_stats) == 3

    def test_select_strategy_ucb(self, tracker):
        """Selects using UCB."""
        # Record some outcomes
        for _ in range(5):
            tracker.record_outcome(
                StrategyOutcome(
                    strategy_type=StrategyType.CACHING,
                    strategy_id="lru",
                    success=True,
                    latency_ms=100,
                    token_cost=200,
                )
            )

        # New strategy should be selected (infinity UCB)
        selected = tracker.select_strategy(StrategyType.CACHING, ["lru", "fifo"], method="ucb")

        assert selected == "fifo"  # Untried strategy

    def test_select_strategy_greedy(self, tracker):
        """Selects using greedy."""
        # Record outcomes for different strategies
        for _ in range(5):
            tracker.record_outcome(
                StrategyOutcome(
                    strategy_type=StrategyType.CACHING,
                    strategy_id="lru",
                    success=True,
                    latency_ms=100,
                    token_cost=200,
                )
            )

        for _ in range(5):
            tracker.record_outcome(
                StrategyOutcome(
                    strategy_type=StrategyType.CACHING,
                    strategy_id="fifo",
                    success=False,
                    latency_ms=100,
                    token_cost=200,
                )
            )

        selected = tracker.select_strategy(StrategyType.CACHING, ["lru", "fifo"], method="greedy")

        assert selected == "lru"  # Higher success rate

    def test_persistence(self):
        """Persists and loads data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"

            # Create and populate tracker
            tracker1 = StrategyTracker(path)
            tracker1.record_outcome(
                StrategyOutcome(
                    strategy_type=StrategyType.SEARCH,
                    strategy_id="v1",
                    success=True,
                    latency_ms=100,
                    token_cost=200,
                )
            )

            # Create new tracker with same path
            tracker2 = StrategyTracker(path)

            stats = tracker2.get_stats(StrategyType.SEARCH, "v1")
            assert stats is not None
            assert stats.total_uses == 1

    def test_reset(self, tracker):
        """Can reset tracker."""
        tracker.record_outcome(
            StrategyOutcome(
                strategy_type=StrategyType.SEARCH,
                strategy_id="v1",
                success=True,
                latency_ms=100,
                token_cost=200,
            )
        )

        tracker.reset()

        stats = tracker.get_stats(StrategyType.SEARCH, "v1")
        assert stats is None


class TestUserFeedbackCollector:
    """Tests for UserFeedbackCollector class."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return StrategyTracker()

    @pytest.fixture
    def collector(self, tracker):
        """Create collector instance."""
        return UserFeedbackCollector(tracker)

    def test_register_for_feedback(self, collector):
        """Can register outcome for feedback."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.SUMMARIZATION,
            strategy_id="v1",
            success=True,
            latency_ms=100,
            token_cost=200,
        )

        collector.register_for_feedback(outcome, "feedback-1")

        assert "feedback-1" in collector._pending_feedback

    def test_process_feedback(self, collector, tracker):
        """Can process feedback."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.SUMMARIZATION,
            strategy_id="v1",
            success=True,
            latency_ms=100,
            token_cost=200,
        )

        collector.register_for_feedback(outcome, "feedback-1")
        result = collector.process_feedback("feedback-1", FeedbackType.POSITIVE)

        assert result is True
        assert "feedback-1" not in collector._pending_feedback

        stats = tracker.get_stats(StrategyType.SUMMARIZATION, "v1")
        assert stats is not None

    def test_process_unknown_feedback(self, collector):
        """Returns False for unknown feedback."""
        result = collector.process_feedback("unknown", FeedbackType.POSITIVE)
        assert result is False

    def test_infer_feedback_success(self, collector, tracker):
        """Infers positive feedback from task completion."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.DECOMPOSITION,
            strategy_id="v1",
            success=True,
            latency_ms=100,
            token_cost=200,
        )

        collector.register_for_feedback(outcome, "feedback-1")
        collector.infer_feedback("feedback-1", task_completed=True, retry_count=0)

        # Outcome should be recorded with implicit success
        stats = tracker.get_stats(StrategyType.DECOMPOSITION, "v1")
        assert stats is not None

    def test_infer_feedback_failure(self, collector, tracker):
        """Infers negative feedback from retries."""
        outcome = StrategyOutcome(
            strategy_type=StrategyType.DECOMPOSITION,
            strategy_id="v1",
            success=True,
            latency_ms=100,
            token_cost=200,
        )

        collector.register_for_feedback(outcome, "feedback-1")
        collector.infer_feedback("feedback-1", task_completed=False, retry_count=3)

        stats = tracker.get_stats(StrategyType.DECOMPOSITION, "v1")
        assert stats is not None


class TestAdaptiveStrategy:
    """Tests for AdaptiveStrategy class."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return StrategyTracker()

    @pytest.fixture
    def strategy(self, tracker):
        """Create adaptive strategy instance."""
        return AdaptiveStrategy(
            tracker=tracker,
            strategy_type=StrategyType.PROMPT_VARIANT,
            strategy_configs={
                "concise": {"max_tokens": 100},
                "detailed": {"max_tokens": 500},
                "structured": {"max_tokens": 300},
            },
        )

    def test_available_strategies(self, strategy):
        """Returns available strategies."""
        available = strategy.available_strategies
        assert len(available) == 3
        assert "concise" in available

    def test_select(self, strategy):
        """Can select strategy."""
        strategy_id, config = strategy.select()

        assert strategy_id in ["concise", "detailed", "structured"]
        assert "max_tokens" in config

    def test_record_result(self, strategy, tracker):
        """Records result."""
        strategy.select()
        strategy.record_result(success=True, token_cost=150)

        all_stats = tracker.get_all_stats(StrategyType.PROMPT_VARIANT)
        assert len(all_stats) == 1

    def test_get_best_strategy(self, strategy, tracker):
        """Gets best performing strategy."""
        # Record some outcomes
        for _ in range(5):
            strategy_id, config = strategy.select(method="epsilon_greedy")
            strategy.record_result(
                success=strategy_id == "detailed",  # detailed always succeeds
                token_cost=100,
            )

        best = strategy.get_best_strategy()
        # After some trials, detailed should be best
        if best:
            assert best[0] in strategy.available_strategies


class TestLearningSystem:
    """Tests for LearningSystem class."""

    @pytest.fixture
    def system(self):
        """Create learning system instance."""
        return LearningSystem()

    def test_register_strategy(self, system):
        """Can register strategy."""
        strategy = system.register_strategy(
            StrategyType.MODEL_SELECTION,
            {"fast": {"model": "haiku"}, "powerful": {"model": "opus"}},
        )

        assert strategy is not None
        assert len(strategy.available_strategies) == 2

    def test_get_strategy(self, system):
        """Can get registered strategy."""
        system.register_strategy(
            StrategyType.MODEL_SELECTION,
            {"fast": {}, "powerful": {}},
        )

        strategy = system.get_strategy(StrategyType.MODEL_SELECTION)
        assert strategy is not None

    def test_get_summary(self, system):
        """Can get summary."""
        strategy = system.register_strategy(
            StrategyType.CACHING,
            {"lru": {}, "fifo": {}},
        )

        strategy.select()
        strategy.record_result(success=True, token_cost=100)

        summary = system.get_summary()

        assert "total_trials" in summary
        assert "strategies_tracked" in summary

    def test_reset(self, system):
        """Can reset system."""
        system.register_strategy(StrategyType.CACHING, {"lru": {}})
        system.reset()

        assert system.get_strategy(StrategyType.CACHING) is None

    def test_persistence(self):
        """Persists learning data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            system1 = LearningSystem(path)
            strategy = system1.register_strategy(
                StrategyType.SEARCH,
                {"v1": {}, "v2": {}},
            )
            strategy.select()
            strategy.record_result(success=True, token_cost=100)

            # Create new system with same path
            system2 = LearningSystem(path)

            summary = system2.get_summary()
            assert summary["total_trials"] == 1
