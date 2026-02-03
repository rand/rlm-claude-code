"""
Execution guarantees for RLM operations.

Implements: SPEC-10.10-10.15 Execution Guarantees
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ViolationType(Enum):
    """Type of guarantee violation."""

    COST_EXCEEDED = "cost_exceeded"
    COST_WOULD_EXCEED = "cost_would_exceed"
    DURATION_EXCEEDED = "duration_exceeded"
    DURATION_WOULD_EXCEED = "duration_would_exceed"
    CALLS_EXCEEDED = "calls_exceeded"
    CALLS_WOULD_EXCEED = "calls_would_exceed"


@dataclass
class ExecutionGuarantees:
    """
    Hard execution boundaries for RLM operations.

    Implements: SPEC-10.11
    """

    max_cost_usd: float = 1.0
    max_duration_seconds: float = 300.0
    max_recursive_calls: int = 20


@dataclass
class GuaranteeViolation:
    """Record of a guarantee violation."""

    violation_type: ViolationType
    timestamp: float
    current_value: float | int
    limit_value: float | int
    overridden: bool = False
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    """Result of a guarantee check."""

    allowed: bool
    violation_type: ViolationType | None = None
    message: str = ""
    overridden: bool = False


@dataclass
class GracefulDegradationPlan:
    """
    Plan for graceful degradation when budget exhausted.

    Implements: SPEC-10.13
    """

    partial_result: str
    explanation: str
    recommendations: list[str]
    cost_spent_usd: float
    duration_spent_seconds: float
    recursive_calls_made: int


class GuaranteeChecker:
    """
    Checker for execution guarantees.

    Implements: SPEC-10.10-10.15

    Enforces hard boundaries on cost, duration, and recursive calls.
    Provides graceful degradation and override mechanisms.
    """

    def __init__(self, guarantees: ExecutionGuarantees):
        """
        Initialize guarantee checker.

        Args:
            guarantees: Execution guarantees to enforce
        """
        self.guarantees = guarantees
        self._start_time = time.time()
        self._current_cost = 0.0
        self._recursive_calls = 0
        self._violation_log: list[GuaranteeViolation] = []

    @property
    def current_cost(self) -> float:
        """Current accumulated cost."""
        return self._current_cost

    @property
    def recursive_calls_count(self) -> int:
        """Current recursive call count."""
        return self._recursive_calls

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time since checker start."""
        return time.time() - self._start_time

    def record_cost(self, cost: float) -> None:
        """Record cost of an operation."""
        self._current_cost += cost

    def record_recursive_call(self) -> None:
        """Record a recursive call."""
        self._recursive_calls += 1

    def can_proceed(self, override_confirmed: bool = False) -> bool:
        """
        Check if execution can proceed.

        Args:
            override_confirmed: If True, allow proceeding even if limits exceeded

        Returns:
            True if can proceed, False otherwise
        """
        result = self.check_can_proceed(override_confirmed=override_confirmed)
        return result.allowed

    def check_can_proceed(
        self,
        estimated_cost: float = 0.0,
        override_confirmed: bool = False,
    ) -> CheckResult:
        """
        Check if operation can proceed within guarantees.

        Implements: SPEC-10.12

        Args:
            estimated_cost: Estimated cost of the operation
            override_confirmed: If True, allow proceeding with override

        Returns:
            CheckResult with allowed status and any violation info
        """
        # Check cost
        if self._current_cost >= self.guarantees.max_cost_usd:
            violation = GuaranteeViolation(
                violation_type=ViolationType.COST_EXCEEDED,
                timestamp=time.time(),
                current_value=self._current_cost,
                limit_value=self.guarantees.max_cost_usd,
                overridden=override_confirmed,
            )
            self._violation_log.append(violation)

            if override_confirmed:
                return CheckResult(
                    allowed=True,
                    violation_type=ViolationType.COST_EXCEEDED,
                    message="Cost limit exceeded (override confirmed)",
                    overridden=True,
                )
            return CheckResult(
                allowed=False,
                violation_type=ViolationType.COST_EXCEEDED,
                message=f"Cost limit exceeded: ${self._current_cost:.2f} >= ${self.guarantees.max_cost_usd:.2f}",
            )

        # Check if estimated cost would exceed
        if self._current_cost + estimated_cost > self.guarantees.max_cost_usd:
            return CheckResult(
                allowed=False,
                violation_type=ViolationType.COST_WOULD_EXCEED,
                message="Estimated operation would exceed cost limit",
            )

        # Check duration
        elapsed = self.elapsed_seconds
        if elapsed >= self.guarantees.max_duration_seconds:
            violation = GuaranteeViolation(
                violation_type=ViolationType.DURATION_EXCEEDED,
                timestamp=time.time(),
                current_value=elapsed,
                limit_value=self.guarantees.max_duration_seconds,
                overridden=override_confirmed,
            )
            self._violation_log.append(violation)

            if override_confirmed:
                return CheckResult(
                    allowed=True,
                    violation_type=ViolationType.DURATION_EXCEEDED,
                    message="Duration limit exceeded (override confirmed)",
                    overridden=True,
                )
            return CheckResult(
                allowed=False,
                violation_type=ViolationType.DURATION_EXCEEDED,
                message=f"Duration limit exceeded: {elapsed:.1f}s >= {self.guarantees.max_duration_seconds:.1f}s",
            )

        # Check recursive calls
        if self._recursive_calls >= self.guarantees.max_recursive_calls:
            violation = GuaranteeViolation(
                violation_type=ViolationType.CALLS_EXCEEDED,
                timestamp=time.time(),
                current_value=self._recursive_calls,
                limit_value=self.guarantees.max_recursive_calls,
                overridden=override_confirmed,
            )
            self._violation_log.append(violation)

            if override_confirmed:
                return CheckResult(
                    allowed=True,
                    violation_type=ViolationType.CALLS_EXCEEDED,
                    message="Recursive calls limit exceeded (override confirmed)",
                    overridden=True,
                )
            return CheckResult(
                allowed=False,
                violation_type=ViolationType.CALLS_EXCEEDED,
                message=f"Recursive calls limit exceeded: {self._recursive_calls} >= {self.guarantees.max_recursive_calls}",
            )

        return CheckResult(allowed=True)

    def create_degradation_plan(
        self,
        partial_result: str,
        context: dict[str, Any],
    ) -> GracefulDegradationPlan:
        """
        Create graceful degradation plan when guarantees violated.

        Implements: SPEC-10.13

        Args:
            partial_result: Partial result achieved so far
            context: Additional context about the operation

        Returns:
            GracefulDegradationPlan with recommendations
        """
        # Determine violation type and create explanation
        explanation_parts = []
        recommendations = []

        if self._current_cost >= self.guarantees.max_cost_usd:
            explanation_parts.append(
                f"Cost limit reached: ${self._current_cost:.2f} of ${self.guarantees.max_cost_usd:.2f}"
            )
            recommendations.append("Consider increasing max_cost_usd limit")
            recommendations.append("Use a more cost-efficient model for sub-queries")

        if self.elapsed_seconds >= self.guarantees.max_duration_seconds:
            explanation_parts.append(
                f"Duration limit reached: {self.elapsed_seconds:.1f}s of {self.guarantees.max_duration_seconds:.1f}s"
            )
            recommendations.append("Consider increasing max_duration_seconds limit")
            recommendations.append("Break task into smaller sub-tasks")

        if self._recursive_calls >= self.guarantees.max_recursive_calls:
            explanation_parts.append(
                f"Recursive call limit reached: {self._recursive_calls} of {self.guarantees.max_recursive_calls}"
            )
            recommendations.append("Consider increasing max_recursive_calls limit")
            recommendations.append("Reduce problem decomposition depth")

        if not explanation_parts:
            explanation_parts.append("Execution stopped by guarantee checker")

        if not recommendations:
            recommendations.append("Review execution parameters")

        return GracefulDegradationPlan(
            partial_result=partial_result,
            explanation="; ".join(explanation_parts),
            recommendations=recommendations,
            cost_spent_usd=self._current_cost,
            duration_spent_seconds=self.elapsed_seconds,
            recursive_calls_made=self._recursive_calls,
        )

    def get_violation_log(self) -> list[GuaranteeViolation]:
        """
        Get log of all guarantee violations.

        Implements: SPEC-10.15

        Returns:
            List of GuaranteeViolation records
        """
        return self._violation_log.copy()

    def reset(self) -> None:
        """Reset checker state for new session."""
        self._start_time = time.time()
        self._current_cost = 0.0
        self._recursive_calls = 0
        self._violation_log.clear()


__all__ = [
    "CheckResult",
    "ExecutionGuarantees",
    "GracefulDegradationPlan",
    "GuaranteeChecker",
    "GuaranteeViolation",
    "ViolationType",
]
