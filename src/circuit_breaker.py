"""
Circuit breaker for recursive calls to prevent cascade failures.

Implements: SPEC-10.20-10.26

Provides circuit breaker pattern with CLOSED, OPEN, and HALF_OPEN states,
configurable thresholds, per-tier tracking, and fallback results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CircuitState(Enum):
    """
    Circuit breaker state.

    Implements: SPEC-10.21
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.

    Implements: SPEC-10.22
    """

    failure_threshold: int = 3
    recovery_timeout: float = 60.0


@dataclass
class FailureRecord:
    """Record of a failure event."""

    error_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "tier": self.tier,
        }


@dataclass
class FallbackResult:
    """
    Result returned when circuit is open.

    Implements: SPEC-10.24
    """

    is_fallback: bool
    reason: str
    original_query: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_fallback": self.is_fallback,
            "reason": self.reason,
            "original_query": self.original_query,
            "suggestion": self.suggestion,
        }


@dataclass
class CircuitBreakerMetrics:
    """
    Metrics for circuit breaker monitoring.

    Implements: SPEC-10.26
    """

    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    open_count: int = 0
    last_failure_time: datetime | None = None
    last_state_change: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "open_count": self.open_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change": (
                self.last_state_change.isoformat() if self.last_state_change else None
            ),
        }


class CircuitBreaker:
    """
    Circuit breaker for preventing cascade failures.

    Implements: SPEC-10.20-10.26
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._open_count = 0
        self._last_failure_time: float | None = None
        self._last_state_change = time.time()
        self._failure_records: list[FailureRecord] = []

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    def allow_request(self) -> bool:
        """
        Check if a request is allowed.

        Implements: SPEC-10.20

        Returns:
            True if request is allowed
        """
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.HALF_OPEN:
            return True  # Allow probe request
        else:  # OPEN
            return False

    def record_failure(
        self,
        error_type: str = "Unknown",
        message: str = "Request failed",
    ) -> None:
        """
        Record a failure.

        Implements: SPEC-10.20

        Args:
            error_type: Type of error
            message: Error message
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        self._failure_records.append(FailureRecord(error_type=error_type, message=message))

        if self._state == CircuitState.HALF_OPEN:
            # Failure in half-open reopens circuit
            self._transition_to(CircuitState.OPEN)
        elif self._failure_count >= self.config.failure_threshold:
            # Threshold reached, open circuit
            self._transition_to(CircuitState.OPEN)

    def record_success(self) -> None:
        """
        Record a success.

        Implements: SPEC-10.20
        """
        self._success_count += 1

        if self._state == CircuitState.HALF_OPEN:
            # Success in half-open closes circuit
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            # Success resets failure count
            self._failure_count = 0

    def check_recovery(self) -> None:
        """
        Check if circuit should transition to half-open.

        Implements: SPEC-10.25
        """
        if self._state != CircuitState.OPEN:
            return

        if self._last_failure_time is None:
            return

        elapsed = time.time() - self._last_failure_time
        if elapsed >= self.config.recovery_timeout:
            self._transition_to(CircuitState.HALF_OPEN)

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_state_change = time.time()

    def get_fallback_result(self, query: str) -> FallbackResult:
        """
        Get fallback result when circuit is open.

        Implements: SPEC-10.24

        Args:
            query: Original query

        Returns:
            FallbackResult with explanation
        """
        return FallbackResult(
            is_fallback=True,
            reason="Circuit breaker is open due to repeated failures",
            original_query=query,
            suggestion="The service is experiencing issues. Please try again later or use a different model tier.",
        )

    def schedule_recovery_test(self) -> bool:
        """
        Schedule a recovery test.

        Implements: SPEC-10.25

        Returns:
            True if recovery test was scheduled
        """
        # In a real implementation, this would schedule an async task
        return True

    def get_metrics(self) -> CircuitBreakerMetrics:
        """
        Get current metrics.

        Implements: SPEC-10.26

        Returns:
            CircuitBreakerMetrics with current state
        """
        return CircuitBreakerMetrics(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            open_count=self._open_count,
            last_failure_time=(
                datetime.fromtimestamp(self._last_failure_time) if self._last_failure_time else None
            ),
            last_state_change=datetime.fromtimestamp(self._last_state_change),
        )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if new_state == CircuitState.OPEN and self._state != CircuitState.OPEN:
            self._open_count += 1

        self._state = new_state
        self._last_state_change = time.time()


class RecoveryTest:
    """
    Recovery testing for circuit breakers.

    Implements: SPEC-10.25
    """

    def __init__(self) -> None:
        """Initialize recovery test."""
        self._probe_count = 1

    def get_probe_count(self) -> int:
        """
        Get number of probe requests.

        SPEC-10.25: Single probe request

        Returns:
            Number of probes (always 1)
        """
        return self._probe_count

    def test_recovery(self, circuit: CircuitBreaker, success: bool) -> None:
        """
        Test recovery with probe result.

        Implements: SPEC-10.25

        Args:
            circuit: Circuit breaker to test
            success: Whether probe succeeded
        """
        if success:
            circuit.record_success()
        else:
            circuit.record_failure()


class TierCircuitBreaker:
    """
    Circuit breaker with per-tier tracking.

    Implements: SPEC-10.23
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """
        Initialize tier circuit breaker.

        Args:
            config: Configuration for all tier breakers
        """
        self.config = config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

    def _get_breaker(self, tier: str) -> CircuitBreaker:
        """Get or create circuit breaker for tier."""
        if tier not in self._breakers:
            self._breakers[tier] = CircuitBreaker(self.config)
        return self._breakers[tier]

    def record_failure(
        self,
        tier: str,
        error_type: str = "Unknown",
        message: str = "Request failed",
    ) -> None:
        """
        Record a failure for a tier.

        Implements: SPEC-10.23

        Args:
            tier: Model tier
            error_type: Type of error
            message: Error message
        """
        breaker = self._get_breaker(tier)
        breaker.record_failure(error_type, message)

    def record_success(self, tier: str) -> None:
        """
        Record a success for a tier.

        Args:
            tier: Model tier
        """
        breaker = self._get_breaker(tier)
        breaker.record_success()

    def get_failure_count(self, tier: str) -> int:
        """
        Get failure count for a tier.

        Implements: SPEC-10.23

        Args:
            tier: Model tier

        Returns:
            Failure count
        """
        if tier not in self._breakers:
            return 0
        return self._breakers[tier].failure_count

    def get_state(self, tier: str) -> CircuitState:
        """
        Get circuit state for a tier.

        Args:
            tier: Model tier

        Returns:
            Circuit state
        """
        breaker = self._get_breaker(tier)
        return breaker.state

    def allow_request(self, tier: str) -> bool:
        """
        Check if request is allowed for a tier.

        Args:
            tier: Model tier

        Returns:
            True if request is allowed
        """
        breaker = self._get_breaker(tier)
        return breaker.allow_request()

    def get_fallback_result(self, tier: str, query: str) -> FallbackResult:
        """
        Get fallback result for a tier.

        Implements: SPEC-10.24

        Args:
            tier: Model tier
            query: Original query

        Returns:
            FallbackResult
        """
        breaker = self._get_breaker(tier)
        result = breaker.get_fallback_result(query)
        result.reason = f"Circuit breaker for {tier} is open due to repeated failures"
        return result

    def get_tiers(self) -> list[str]:
        """
        Get all tracked tiers.

        Returns:
            List of tier names
        """
        return list(self._breakers.keys())

    def get_all_metrics(self) -> dict[str, CircuitBreakerMetrics]:
        """
        Get metrics for all tiers.

        Implements: SPEC-10.26

        Returns:
            Dict of tier to metrics
        """
        return {tier: breaker.get_metrics() for tier, breaker in self._breakers.items()}


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitState",
    "FailureRecord",
    "FallbackResult",
    "RecoveryTest",
    "TierCircuitBreaker",
]
