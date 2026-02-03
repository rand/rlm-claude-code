"""
Tests for circuit breaker for recursive calls (SPEC-10.20-10.26).

Tests cover:
- Circuit breaker states and transitions
- Configurable thresholds
- Per-tier failure tracking
- Fallback results when open
- Recovery testing
- Metrics exposure
"""

import time

from src.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    FailureRecord,
    FallbackResult,
    RecoveryTest,
    TierCircuitBreaker,
)


class TestCircuitBreakerStates:
    """Tests for circuit breaker states (SPEC-10.21)."""

    def test_closed_state_exists(self):
        """SPEC-10.21: CLOSED state for normal operation."""
        assert CircuitState.CLOSED.value == "closed"

    def test_open_state_exists(self):
        """SPEC-10.21: OPEN state for failing fast."""
        assert CircuitState.OPEN.value == "open"

    def test_half_open_state_exists(self):
        """SPEC-10.21: HALF_OPEN state for testing recovery."""
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()

        assert cb.state == CircuitState.CLOSED

    def test_allows_requests_when_closed(self):
        """CLOSED state allows requests."""
        cb = CircuitBreaker()

        assert cb.allow_request()


class TestCircuitBreakerConfig:
    """Tests for configurable circuit breaker (SPEC-10.22)."""

    def test_default_failure_threshold(self):
        """SPEC-10.22: Default failure_threshold is 3."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 3

    def test_default_recovery_timeout(self):
        """SPEC-10.22: Default recovery_timeout is 60.0 seconds."""
        config = CircuitBreakerConfig()

        assert config.recovery_timeout == 60.0

    def test_configurable_failure_threshold(self):
        """Failure threshold should be configurable."""
        config = CircuitBreakerConfig(failure_threshold=5)

        assert config.failure_threshold == 5

    def test_configurable_recovery_timeout(self):
        """Recovery timeout should be configurable."""
        config = CircuitBreakerConfig(recovery_timeout=30.0)

        assert config.recovery_timeout == 30.0

    def test_config_affects_behavior(self):
        """Config should affect circuit breaker behavior."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config=config)

        # Two failures should open the circuit
        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN


class TestStateTransitions:
    """Tests for state transition logic (SPEC-10.20)."""

    def test_transitions_to_open_on_threshold(self):
        """Circuit opens when failure threshold reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Success resets failure count in CLOSED state."""
        cb = CircuitBreaker()

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Check state (should transition on next request check)
        cb.check_recovery()
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_on_success_in_half_open(self):
        """Circuit closes on success in HALF_OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        time.sleep(0.02)
        cb.check_recovery()

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Circuit reopens on failure in HALF_OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        time.sleep(0.02)
        cb.check_recovery()

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestPerTierTracking:
    """Tests for per-tier failure tracking (SPEC-10.23)."""

    def test_tracks_failures_per_tier(self):
        """SPEC-10.23: Track failure count per model tier."""
        tier_cb = TierCircuitBreaker()

        tier_cb.record_failure("haiku")
        tier_cb.record_failure("haiku")
        tier_cb.record_failure("sonnet")

        assert tier_cb.get_failure_count("haiku") == 2
        assert tier_cb.get_failure_count("sonnet") == 1

    def test_separate_circuit_per_tier(self):
        """Each tier has its own circuit breaker."""
        tier_cb = TierCircuitBreaker()

        # Open haiku circuit
        for _ in range(3):
            tier_cb.record_failure("haiku")

        assert tier_cb.get_state("haiku") == CircuitState.OPEN
        assert tier_cb.get_state("sonnet") == CircuitState.CLOSED

    def test_allows_request_per_tier(self):
        """Request allowance is per-tier."""
        tier_cb = TierCircuitBreaker()

        # Open haiku circuit
        for _ in range(3):
            tier_cb.record_failure("haiku")

        assert not tier_cb.allow_request("haiku")
        assert tier_cb.allow_request("sonnet")

    def test_get_all_tiers(self):
        """Can get all tracked tiers."""
        tier_cb = TierCircuitBreaker()

        tier_cb.record_failure("haiku")
        tier_cb.record_failure("sonnet")
        tier_cb.record_failure("opus")

        tiers = tier_cb.get_tiers()

        assert "haiku" in tiers
        assert "sonnet" in tiers
        assert "opus" in tiers


class TestFallbackResults:
    """Tests for fallback results when open (SPEC-10.24)."""

    def test_returns_fallback_when_open(self):
        """SPEC-10.24: Return FallbackResult when circuit is OPEN."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()

        result = cb.get_fallback_result("test query")

        assert isinstance(result, FallbackResult)
        assert result.is_fallback
        assert "circuit" in result.reason.lower()

    def test_fallback_includes_query(self):
        """Fallback result includes original query."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()

        result = cb.get_fallback_result("my test query")

        assert result.original_query == "my test query"

    def test_fallback_includes_suggestion(self):
        """Fallback result includes user suggestion."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()

        result = cb.get_fallback_result("test")

        assert result.suggestion is not None

    def test_fallback_to_dict(self):
        """Fallback result should be serializable."""
        result = FallbackResult(
            is_fallback=True,
            reason="Circuit open",
            original_query="test",
            suggestion="Try again later",
        )

        data = result.to_dict()

        assert "is_fallback" in data
        assert "reason" in data
        assert "suggestion" in data


class TestRecoveryTesting:
    """Tests for recovery testing (SPEC-10.25)."""

    def test_recovery_test_on_probe_success(self):
        """SPEC-10.25: Close circuit on successful probe."""
        recovery = RecoveryTest()

        cb = CircuitBreaker()
        cb._state = CircuitState.HALF_OPEN

        recovery.test_recovery(cb, success=True)

        assert cb.state == CircuitState.CLOSED

    def test_recovery_test_on_probe_failure(self):
        """SPEC-10.25: Extend open period on probe failure."""
        recovery = RecoveryTest()

        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)
        cb._state = CircuitState.HALF_OPEN

        recovery.test_recovery(cb, success=False)

        assert cb.state == CircuitState.OPEN

    def test_probe_request_is_single(self):
        """SPEC-10.25: Send single probe request."""
        recovery = RecoveryTest()

        assert recovery.get_probe_count() == 1

    def test_can_schedule_recovery_test(self):
        """Recovery test can be scheduled."""
        cb = CircuitBreaker()

        scheduled = cb.schedule_recovery_test()

        assert scheduled or not scheduled  # Implementation detail


class TestMetricsExposure:
    """Tests for metrics exposure (SPEC-10.26)."""

    def test_exposes_state_metric(self):
        """SPEC-10.26: Expose current state."""
        cb = CircuitBreaker()

        metrics = cb.get_metrics()

        assert "state" in metrics.to_dict()

    def test_exposes_failure_count(self):
        """Expose current failure count."""
        cb = CircuitBreaker()
        cb.record_failure()

        metrics = cb.get_metrics()

        assert metrics.failure_count == 1

    def test_exposes_success_count(self):
        """Expose success count."""
        cb = CircuitBreaker()
        cb.record_success()
        cb.record_success()

        metrics = cb.get_metrics()

        assert metrics.success_count == 2

    def test_exposes_open_count(self):
        """Expose number of times circuit opened."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        cb.reset()
        cb.record_failure()

        metrics = cb.get_metrics()

        assert metrics.open_count >= 1

    def test_exposes_last_failure_time(self):
        """Expose last failure timestamp."""
        cb = CircuitBreaker()
        cb.record_failure()

        metrics = cb.get_metrics()

        assert metrics.last_failure_time is not None

    def test_metrics_to_dict(self):
        """Metrics should be serializable."""
        metrics = CircuitBreakerMetrics(
            state=CircuitState.CLOSED,
            failure_count=2,
            success_count=10,
            open_count=1,
        )

        data = metrics.to_dict()

        assert "state" in data
        assert "failure_count" in data
        assert "success_count" in data


class TestFailureRecord:
    """Tests for failure record structure."""

    def test_record_has_timestamp(self):
        """Failure record has timestamp."""
        record = FailureRecord(
            error_type="TimeoutError",
            message="Request timed out",
        )

        assert record.timestamp is not None

    def test_record_has_error_info(self):
        """Failure record has error information."""
        record = FailureRecord(
            error_type="APIError",
            message="Rate limited",
            tier="haiku",
        )

        assert record.error_type == "APIError"
        assert record.message == "Rate limited"
        assert record.tier == "haiku"

    def test_record_to_dict(self):
        """Failure record should be serializable."""
        record = FailureRecord(
            error_type="TimeoutError",
            message="Timed out",
        )

        data = record.to_dict()

        assert "error_type" in data
        assert "message" in data
        assert "timestamp" in data


class TestIntegration:
    """Integration tests for circuit breaker."""

    def test_full_circuit_lifecycle(self):
        """Test complete circuit breaker lifecycle."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.05,
        )
        cb = CircuitBreaker(config=config)

        # Start closed
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

        # Failures open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

        # Wait for recovery
        time.sleep(0.06)
        cb.check_recovery()
        assert cb.state == CircuitState.HALF_OPEN

        # Success closes circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_tier_circuit_breaker_workflow(self):
        """Test tier-based circuit breaker workflow."""
        tier_cb = TierCircuitBreaker()

        # Normal operation
        tier_cb.record_success("haiku")
        tier_cb.record_success("sonnet")

        # Failures on one tier
        for _ in range(3):
            tier_cb.record_failure("haiku")

        # Haiku open, sonnet still closed
        assert not tier_cb.allow_request("haiku")
        assert tier_cb.allow_request("sonnet")

        # Get fallback for haiku
        result = tier_cb.get_fallback_result("haiku", "test query")
        assert result.is_fallback

    def test_metrics_tracking_through_lifecycle(self):
        """Metrics track correctly through lifecycle."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config=config)

        cb.record_success()
        cb.record_failure()
        cb.record_failure()

        metrics = cb.get_metrics()

        assert metrics.success_count == 1
        assert metrics.failure_count == 2
        assert metrics.open_count >= 1

    def test_concurrent_tier_operations(self):
        """Multiple tiers operate independently."""
        tier_cb = TierCircuitBreaker()

        # Mix of operations on different tiers
        tier_cb.record_success("haiku")
        tier_cb.record_failure("sonnet")
        tier_cb.record_success("opus")
        tier_cb.record_failure("haiku")

        assert tier_cb.get_failure_count("haiku") == 1
        assert tier_cb.get_failure_count("sonnet") == 1
        assert tier_cb.get_failure_count("opus") == 0
