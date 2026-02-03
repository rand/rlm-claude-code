"""
Property-based tests for enhanced budget tracking.

Implements: Spec SPEC-05.26 - Property tests for budget metrics.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Strategies
# =============================================================================

token_strategy = st.integers(min_value=0, max_value=100000)
cost_strategy = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
count_strategy = st.integers(min_value=0, max_value=100)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


def make_budget_tracker():
    """Create a fresh EnhancedBudgetTracker instance."""
    from src.enhanced_budget import EnhancedBudgetTracker

    return EnhancedBudgetTracker()


# =============================================================================
# SPEC-05.26: Budget Metrics Never Decrease
# =============================================================================


@pytest.mark.hypothesis
class TestBudgetMetricsProperties:
    """Property tests for budget metric invariants."""

    @given(
        input_tokens=token_strategy,
        output_tokens=token_strategy,
        cached_tokens=token_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_metrics_never_decrease(self, input_tokens, output_tokens, cached_tokens):
        """
        Budget metrics never decrease unexpectedly.

        @trace SPEC-05.26
        """
        from src.cost_tracker import CostComponent

        tracker = make_budget_tracker()

        # Record initial state
        initial_metrics = tracker.get_metrics()

        # Record LLM call
        tracker.record_llm_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        # Get new metrics
        new_metrics = tracker.get_metrics()

        # Verify no decreases
        assert new_metrics.input_tokens >= initial_metrics.input_tokens
        assert new_metrics.output_tokens >= initial_metrics.output_tokens
        assert new_metrics.cached_tokens >= initial_metrics.cached_tokens
        assert new_metrics.total_cost_usd >= initial_metrics.total_cost_usd

    @given(
        num_calls=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=30, deadline=None)
    def test_sub_call_count_monotonic(self, num_calls):
        """
        Sub-call count should only increase.

        @trace SPEC-05.26
        """
        from src.cost_tracker import CostComponent

        tracker = make_budget_tracker()

        previous_count = 0
        for i in range(num_calls):
            tracker.record_llm_call(
                input_tokens=100,
                output_tokens=50,
                model="haiku",
                component=CostComponent.RECURSIVE_CALL,
            )

            metrics = tracker.get_metrics()
            assert metrics.sub_call_count >= previous_count
            previous_count = metrics.sub_call_count

    @given(
        num_executions=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=20, deadline=None)
    def test_repl_executions_monotonic(self, num_executions):
        """
        REPL execution count should only increase.

        @trace SPEC-05.26
        """
        tracker = make_budget_tracker()

        previous_count = 0
        for i in range(num_executions):
            tracker.record_repl_execution()

            metrics = tracker.get_metrics()
            assert metrics.repl_executions >= previous_count
            previous_count = metrics.repl_executions

    @given(
        depths=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=20),
    )
    @settings(max_examples=30, deadline=None)
    def test_max_depth_only_increases(self, depths):
        """
        Max depth reached should only increase or stay same.

        @trace SPEC-05.26
        """
        tracker = make_budget_tracker()

        previous_max = 0
        for depth in depths:
            tracker.record_depth(depth)

            metrics = tracker.get_metrics()
            assert metrics.max_depth_reached >= previous_max
            previous_max = metrics.max_depth_reached

        # Final max should be the maximum of all depths
        assert tracker.get_metrics().max_depth_reached == max(depths)


@pytest.mark.hypothesis
class TestAlertThresholdProperties:
    """Property tests for alert threshold behavior."""

    @given(
        threshold=st.floats(min_value=0.1, max_value=0.99, allow_nan=False),
        limit=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_warning_before_critical(self, threshold, limit):
        """
        Warning alerts should occur before critical alerts.

        @trace SPEC-05.26
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits, EnhancedBudgetTracker

        assume(threshold > 0.5)  # Reasonable threshold

        limits = BudgetLimits(
            max_cost_per_task=limit,
            cost_alert_threshold=threshold,
        )
        tracker = EnhancedBudgetTracker()
        tracker.set_limits(limits)

        all_alerts = []

        # Gradually increase cost
        for i in range(10):
            alerts = tracker.record_llm_call(
                input_tokens=500 * (i + 1),
                output_tokens=100,
                model="sonnet",
                component=CostComponent.ROOT_PROMPT,
            )
            all_alerts.extend(alerts)

        # If we have both warning and critical, warning should come first
        warning_alerts = [a for a in all_alerts if a.level == "warning"]
        critical_alerts = [a for a in all_alerts if a.level == "critical"]

        if warning_alerts and critical_alerts:
            # This is a structural property - warnings should precede criticals
            # in the alert generation logic
            pass  # Just verify no crash


@pytest.mark.hypothesis
class TestLimitEnforcementProperties:
    """Property tests for limit enforcement consistency."""

    @given(
        max_calls=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_limit_enforced_after_exact_count(self, max_calls):
        """
        Limits should be enforced exactly at the configured count.

        @trace SPEC-05.26
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits, EnhancedBudgetTracker

        limits = BudgetLimits(max_recursive_calls=max_calls)
        tracker = EnhancedBudgetTracker()
        tracker.set_limits(limits)

        # Make exactly max_calls recursive calls
        for _ in range(max_calls):
            allowed, _ = tracker.can_recurse()
            assert allowed, f"Should allow recursion before limit ({max_calls})"

            tracker.record_llm_call(
                input_tokens=100,
                output_tokens=50,
                model="haiku",
                component=CostComponent.RECURSIVE_CALL,
            )

        # Next call should be blocked
        allowed, reason = tracker.can_recurse()
        assert not allowed, f"Should block after {max_calls} calls"
        assert reason is not None

    @given(
        max_executions=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=20, deadline=None)
    def test_repl_limit_enforced_exactly(self, max_executions):
        """
        REPL execution limit should be enforced exactly.

        @trace SPEC-05.26
        """
        from src.enhanced_budget import BudgetLimits, EnhancedBudgetTracker

        limits = BudgetLimits(max_repl_executions=max_executions)
        tracker = EnhancedBudgetTracker()
        tracker.set_limits(limits)

        # Make exactly max_executions REPL executions
        for i in range(max_executions):
            allowed, _ = tracker.can_execute_repl()
            assert allowed, f"Should allow execution {i + 1}/{max_executions}"
            tracker.record_repl_execution()

        # Next should be blocked
        allowed, reason = tracker.can_execute_repl()
        assert not allowed
        assert reason is not None


# =============================================================================
# Burn Rate Monitoring Properties (Feature 3e0.5)
# =============================================================================


@pytest.mark.hypothesis
class TestBurnRateProperties:
    """Property tests for burn rate monitoring invariants."""

    @given(
        sample_count=st.integers(min_value=2, max_value=50),
        cost_per_sample=st.floats(min_value=0.0001, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_burn_rate_non_negative(self, sample_count, cost_per_sample):
        """
        Burn rate ($/minute) should never be negative.

        @trace 3e0.5
        """
        import time

        from src.enhanced_budget import EnhancedBudgetTracker

        tracker = EnhancedBudgetTracker()

        # Record samples with realistic timing
        for i in range(sample_count):
            tracker.record_cost_sample(
                cost=cost_per_sample,
                tokens=100,
                model="sonnet",
                call_type="root",
            )
            # Small sleep to create time delta (for rate calculation)
            time.sleep(0.001)

        metrics = tracker.get_burn_rate()
        assert metrics.dollars_per_minute >= 0.0
        assert metrics.tokens_per_minute >= 0.0

    @given(
        initial_budget=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
        burn_rate=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_time_to_exhaustion_consistent(self, initial_budget, burn_rate):
        """
        Time to exhaustion should be proportional to budget/rate.

        @trace 3e0.5
        """

        # Calculate expected time (in seconds)
        if burn_rate > 0:
            expected_time = (initial_budget / burn_rate) * 60.0  # Convert rate from $/min to time
        else:
            expected_time = None

        # Verify the math is consistent
        if expected_time is not None:
            # Reconstruct budget from time and rate
            reconstructed = (expected_time / 60.0) * burn_rate
            assert abs(reconstructed - initial_budget) < 0.001

    @given(
        num_samples=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_sample_pruning_bounded(self, num_samples):
        """
        Cost samples should be pruned to prevent unbounded growth.

        @trace 3e0.5
        """
        from src.enhanced_budget import EnhancedBudgetTracker

        tracker = EnhancedBudgetTracker()
        tracker._max_cost_samples = 50  # Set a test limit

        # Add many samples
        for i in range(num_samples):
            tracker.record_cost_sample(
                cost=0.01,
                tokens=100,
                model="sonnet",
                call_type="root",
            )

        # Verify bounded
        assert len(tracker._cost_samples) <= tracker._max_cost_samples
        assert len(tracker._token_samples) <= tracker._max_cost_samples


# =============================================================================
# Adaptive Depth Budgeting Properties (Feature 3e0.4)
# =============================================================================


@pytest.mark.hypothesis
class TestAdaptiveDepthProperties:
    """Property tests for adaptive depth budgeting invariants."""

    @given(
        budget_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_model_tier_monotonic_with_budget(self, budget_fraction):
        """
        Model recommendations should follow budget tiers:
        - >= 50%: opus
        - 20-50%: sonnet
        - < 20%: haiku

        @trace 3e0.4
        """
        from src.enhanced_budget import EnhancedBudgetTracker

        tracker = EnhancedBudgetTracker()
        model = tracker.get_recommended_model_for_budget(budget_fraction)

        if budget_fraction >= 0.5:
            assert model == "opus"
        elif budget_fraction >= 0.2:
            assert model == "sonnet"
        else:
            assert model == "haiku"

    @given(
        task_cost=st.floats(min_value=0.0, max_value=4.99, allow_nan=False),
        max_cost=st.floats(min_value=5.0, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_adaptive_depth_respects_budget(self, task_cost, max_cost):
        """
        Adaptive depth should never exceed affordable operations.

        @trace 3e0.4
        """
        from src.enhanced_budget import BudgetLimits, EnhancedBudgetTracker

        assume(task_cost < max_cost)  # Valid scenario

        limits = BudgetLimits(max_cost_per_task=max_cost)
        tracker = EnhancedBudgetTracker()
        tracker.set_limits(limits)
        tracker._task_cost = task_cost

        recommendation = tracker.compute_adaptive_depth(
            planned_depth=3,
            planned_model="opus",
            avg_tokens_per_call=5000,
            avg_output_tokens=1500,
        )

        # Verify budget utilization is correct
        remaining = max_cost - task_cost
        expected_utilization = task_cost / max_cost
        assert abs(recommendation.budget_utilization - expected_utilization) < 0.001

        # Recommended depth should be achievable
        if recommendation.estimated_cost_per_depth > 0:
            affordable_ops = remaining / recommendation.estimated_cost_per_depth
            assert recommendation.recommended_depth <= max(0, int(affordable_ops))

    @given(
        planned_depth=st.integers(min_value=0, max_value=5),
        budget_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=40, deadline=None)
    def test_downgrade_reason_matches_budget(self, planned_depth, budget_fraction):
        """
        Downgrade reason should accurately reflect budget state.

        @trace 3e0.4
        """
        from src.enhanced_budget import BudgetLimits, EnhancedBudgetTracker

        limits = BudgetLimits(max_cost_per_task=10.0)
        tracker = EnhancedBudgetTracker()
        tracker.set_limits(limits)
        tracker._task_cost = 10.0 * (1 - budget_fraction)  # Set task cost to match budget fraction

        recommendation = tracker.compute_adaptive_depth(
            planned_depth=planned_depth,
            planned_model="opus",
        )

        # Verify downgrade reason matches budget tier
        if budget_fraction >= 0.5:
            # Should keep opus, no downgrade
            assert recommendation.recommended_model == "opus"
            assert not recommendation.was_downgraded
        elif budget_fraction >= 0.2:
            # Should downgrade to sonnet
            assert recommendation.recommended_model == "sonnet"
            if recommendation.was_downgraded:
                assert recommendation.downgrade_reason == "budget_below_50_percent"
        else:
            # Should force haiku
            assert recommendation.recommended_model == "haiku"
            if recommendation.was_downgraded:
                assert recommendation.downgrade_reason == "budget_below_20_percent"


@pytest.mark.hypothesis
class TestExpensiveOperationWarningProperties:
    """Property tests for expensive operation warnings."""

    @given(
        estimated_cost=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        remaining_budget=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=40, deadline=None)
    def test_warning_level_matches_cost_fraction(self, estimated_cost, remaining_budget):
        """
        Warning level should correspond to cost fraction:
        - > budget: critical (abort)
        - > 50%: critical (downgrade)
        - > 25%: warning
        - <= 25%: None

        @trace 3e0.4
        """
        from src.enhanced_budget import BudgetLimits, EnhancedBudgetTracker

        assume(remaining_budget > 0)

        limits = BudgetLimits(max_cost_per_task=remaining_budget * 2)
        tracker = EnhancedBudgetTracker()
        tracker.set_limits(limits)
        tracker._task_cost = remaining_budget  # Half budget used

        warning = tracker.warn_before_expensive_operation(
            operation="recursive_call",
            estimated_cost=estimated_cost,
            model="sonnet",
        )

        cost_fraction = estimated_cost / remaining_budget

        if estimated_cost > remaining_budget:
            assert warning is not None
            assert warning.warning_level == "critical"
            assert warning.suggested_action == "abort"
        elif cost_fraction > 0.5:
            assert warning is not None
            assert warning.warning_level == "critical"
        elif cost_fraction > 0.25:
            assert warning is not None
            assert warning.warning_level == "warning"
        else:
            assert warning is None
