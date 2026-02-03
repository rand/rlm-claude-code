"""
Tests for execution guarantees implementation.

@trace SPEC-10.10-10.15
"""

from __future__ import annotations

import time

from src.execution_guarantees import (
    ExecutionGuarantees,
    GracefulDegradationPlan,
    GuaranteeChecker,
    ViolationType,
)

# --- Test fixtures ---


def create_default_guarantees() -> ExecutionGuarantees:
    """Create default execution guarantees."""
    return ExecutionGuarantees()


def create_strict_guarantees() -> ExecutionGuarantees:
    """Create strict execution guarantees for testing."""
    return ExecutionGuarantees(
        max_cost_usd=0.10,
        max_duration_seconds=10.0,
        max_recursive_calls=5,
    )


# --- SPEC-10.10: Hard execution boundaries ---


class TestHardBoundaries:
    """Tests for hard execution boundary enforcement."""

    def test_cost_boundary_enforced(self) -> None:
        """
        @trace SPEC-10.10
        Cost boundary should be enforced.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.50)
        checker = GuaranteeChecker(guarantees)

        # Record costs approaching limit
        checker.record_cost(0.30)
        assert checker.can_proceed()

        checker.record_cost(0.15)
        assert checker.can_proceed()

        checker.record_cost(0.10)  # Over limit
        assert not checker.can_proceed()

    def test_duration_boundary_enforced(self) -> None:
        """
        @trace SPEC-10.10
        Duration boundary should be enforced.
        """
        guarantees = ExecutionGuarantees(max_duration_seconds=0.1)
        checker = GuaranteeChecker(guarantees)

        assert checker.can_proceed()

        # Simulate time passing
        time.sleep(0.15)

        assert not checker.can_proceed()

    def test_recursive_calls_boundary_enforced(self) -> None:
        """
        @trace SPEC-10.10
        Recursive calls boundary should be enforced.
        """
        guarantees = ExecutionGuarantees(max_recursive_calls=3)
        checker = GuaranteeChecker(guarantees)

        checker.record_recursive_call()
        checker.record_recursive_call()
        assert checker.can_proceed()

        checker.record_recursive_call()
        assert not checker.can_proceed()


# --- SPEC-10.11: ExecutionGuarantees configuration ---


class TestExecutionGuaranteesConfig:
    """Tests for ExecutionGuarantees configuration."""

    def test_default_values(self) -> None:
        """
        @trace SPEC-10.11
        Default values should match spec.
        """
        guarantees = ExecutionGuarantees()

        assert guarantees.max_cost_usd == 1.0
        assert guarantees.max_duration_seconds == 300.0
        assert guarantees.max_recursive_calls == 20

    def test_custom_values(self) -> None:
        """
        @trace SPEC-10.11
        Custom values should be configurable.
        """
        guarantees = ExecutionGuarantees(
            max_cost_usd=5.0,
            max_duration_seconds=600.0,
            max_recursive_calls=50,
        )

        assert guarantees.max_cost_usd == 5.0
        assert guarantees.max_duration_seconds == 600.0
        assert guarantees.max_recursive_calls == 50


# --- SPEC-10.12: Check before each operation ---


class TestGuaranteeChecking:
    """Tests for guarantee checking before operations."""

    def test_check_before_operation(self) -> None:
        """
        @trace SPEC-10.12
        Guarantees should be checked before each operation.
        """
        guarantees = create_strict_guarantees()
        checker = GuaranteeChecker(guarantees)

        # Should pass initially
        result = checker.check_can_proceed(estimated_cost=0.01)
        assert result.allowed

        # Record some usage
        checker.record_cost(0.08)
        checker.record_recursive_call()
        checker.record_recursive_call()

        # Should still pass
        result = checker.check_can_proceed(estimated_cost=0.01)
        assert result.allowed

        # Push over cost limit
        checker.record_cost(0.03)

        # Should fail
        result = checker.check_can_proceed(estimated_cost=0.01)
        assert not result.allowed
        assert result.violation_type == ViolationType.COST_EXCEEDED

    def test_proactive_check_with_estimate(self) -> None:
        """
        @trace SPEC-10.12
        Should proactively check if estimated operation would exceed limits.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.10)
        checker = GuaranteeChecker(guarantees)

        checker.record_cost(0.05)

        # Estimated operation would exceed limit
        result = checker.check_can_proceed(estimated_cost=0.08)
        assert not result.allowed
        assert result.violation_type == ViolationType.COST_WOULD_EXCEED

    def test_multiple_violation_types(self) -> None:
        """
        @trace SPEC-10.12
        Should detect multiple violation types.
        """
        guarantees = ExecutionGuarantees(
            max_cost_usd=0.10,
            max_duration_seconds=0.05,
            max_recursive_calls=2,
        )
        checker = GuaranteeChecker(guarantees)

        # Exceed recursive calls
        checker.record_recursive_call()
        checker.record_recursive_call()

        time.sleep(0.1)  # Exceed duration

        result = checker.check_can_proceed()

        # Should report at least one violation
        assert not result.allowed
        assert result.violation_type in [
            ViolationType.DURATION_EXCEEDED,
            ViolationType.CALLS_EXCEEDED,
        ]


# --- SPEC-10.13: Graceful degradation ---


class TestGracefulDegradation:
    """Tests for graceful degradation when budget exhausted."""

    def test_degradation_plan_structure(self) -> None:
        """
        @trace SPEC-10.13
        GracefulDegradationPlan should include all required fields.
        """
        plan = GracefulDegradationPlan(
            partial_result="Partial analysis completed",
            explanation="Cost limit reached after analyzing 5 files",
            recommendations=["Consider increasing budget", "Focus on fewer files"],
            cost_spent_usd=1.0,
            duration_spent_seconds=45.0,
            recursive_calls_made=15,
        )

        assert plan.partial_result is not None
        assert plan.explanation is not None
        assert len(plan.recommendations) > 0
        assert plan.cost_spent_usd >= 0
        assert plan.duration_spent_seconds >= 0

    def test_create_degradation_plan_on_violation(self) -> None:
        """
        @trace SPEC-10.13
        Should create degradation plan when guarantee violated.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.10)
        checker = GuaranteeChecker(guarantees)

        # Exceed cost
        checker.record_cost(0.12)

        plan = checker.create_degradation_plan(
            partial_result="Analysis incomplete",
            context={"files_analyzed": 3, "files_remaining": 7},
        )

        assert isinstance(plan, GracefulDegradationPlan)
        assert "cost" in plan.explanation.lower()
        assert plan.cost_spent_usd >= 0.10

    def test_degradation_recommendations(self) -> None:
        """
        @trace SPEC-10.13
        Degradation plan should include actionable recommendations.
        """
        guarantees = ExecutionGuarantees(max_recursive_calls=5)
        checker = GuaranteeChecker(guarantees)

        for _ in range(6):
            checker.record_recursive_call()

        plan = checker.create_degradation_plan(
            partial_result="Partial result",
            context={},
        )

        # Should have at least one recommendation
        assert len(plan.recommendations) >= 1
        # Recommendations should be actionable
        assert any(
            "recursive" in rec.lower() or "call" in rec.lower() or "limit" in rec.lower()
            for rec in plan.recommendations
        )


# --- SPEC-10.14: Override mechanism ---


class TestOverrideMechanism:
    """Tests for guarantee override with confirmation."""

    def test_override_allows_exceeding_limits(self) -> None:
        """
        @trace SPEC-10.14
        Override should allow exceeding limits with confirmation.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.10)
        checker = GuaranteeChecker(guarantees)

        checker.record_cost(0.12)

        # Without override, should not proceed
        assert not checker.can_proceed()

        # With override, should proceed
        assert checker.can_proceed(override_confirmed=True)

    def test_override_requires_explicit_confirmation(self) -> None:
        """
        @trace SPEC-10.14
        Override should require explicit confirmation.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.10)
        checker = GuaranteeChecker(guarantees)

        checker.record_cost(0.12)

        # Default is no override
        result = checker.check_can_proceed()
        assert not result.allowed

        # Explicit override
        result = checker.check_can_proceed(override_confirmed=True)
        assert result.allowed
        assert result.overridden

    def test_override_logged(self) -> None:
        """
        @trace SPEC-10.14
        Override usage should be logged.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.10)
        checker = GuaranteeChecker(guarantees)

        checker.record_cost(0.12)
        checker.check_can_proceed(override_confirmed=True)

        # Override should be in log
        log = checker.get_violation_log()
        assert any(entry.overridden for entry in log)


# --- SPEC-10.15: Violation logging ---


class TestViolationLogging:
    """Tests for guarantee violation logging."""

    def test_violation_logged_with_context(self) -> None:
        """
        @trace SPEC-10.15
        Violations should be logged with context.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.10)
        checker = GuaranteeChecker(guarantees)

        checker.record_cost(0.12)
        checker.check_can_proceed()

        log = checker.get_violation_log()
        assert len(log) >= 1

        violation = log[-1]
        assert violation.violation_type == ViolationType.COST_EXCEEDED
        assert violation.timestamp > 0
        assert violation.current_value is not None
        assert violation.limit_value is not None

    def test_multiple_violations_logged(self) -> None:
        """
        @trace SPEC-10.15
        Multiple violations should all be logged.
        """
        # Create separate checkers to test independent violations
        guarantees_cost = ExecutionGuarantees(max_cost_usd=0.10)
        checker_cost = GuaranteeChecker(guarantees_cost)

        guarantees_calls = ExecutionGuarantees(max_recursive_calls=3)
        checker_calls = GuaranteeChecker(guarantees_calls)

        # Trigger cost violation
        checker_cost.record_cost(0.15)
        checker_cost.check_can_proceed()

        # Trigger calls violation
        for _ in range(4):
            checker_calls.record_recursive_call()
        checker_calls.check_can_proceed()

        cost_log = checker_cost.get_violation_log()
        calls_log = checker_calls.get_violation_log()

        assert any(v.violation_type == ViolationType.COST_EXCEEDED for v in cost_log)
        assert any(v.violation_type == ViolationType.CALLS_EXCEEDED for v in calls_log)

    def test_violation_log_includes_timestamp(self) -> None:
        """
        @trace SPEC-10.15
        Violation log should include timestamps.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.05)
        checker = GuaranteeChecker(guarantees)

        before = time.time()
        checker.record_cost(0.10)
        checker.check_can_proceed()
        after = time.time()

        log = checker.get_violation_log()
        assert len(log) >= 1

        violation = log[-1]
        assert before <= violation.timestamp <= after


# --- Integration tests ---


class TestGuaranteeCheckerIntegration:
    """Integration tests for GuaranteeChecker."""

    def test_full_workflow(self) -> None:
        """
        Test complete workflow with guarantees.
        """
        guarantees = ExecutionGuarantees(
            max_cost_usd=0.50,
            max_duration_seconds=60.0,
            max_recursive_calls=10,
        )
        checker = GuaranteeChecker(guarantees)

        # Simulate operations
        for i in range(5):
            result = checker.check_can_proceed(estimated_cost=0.05)
            assert result.allowed, f"Operation {i} should be allowed"

            checker.record_cost(0.05)
            checker.record_recursive_call()

        # Should have used 0.25 of 0.50 budget
        assert checker.can_proceed()

        # Use more budget
        for i in range(5):
            checker.record_cost(0.05)
            checker.record_recursive_call()

        # Now at limit
        assert not checker.can_proceed()

        # Create degradation plan
        plan = checker.create_degradation_plan(
            partial_result="Completed 10 operations",
            context={"operations": 10},
        )

        assert plan.recursive_calls_made == 10
        assert plan.cost_spent_usd >= 0.49  # Account for floating point

    def test_reset_checker(self) -> None:
        """
        Test resetting checker for new session.
        """
        guarantees = ExecutionGuarantees(max_cost_usd=0.10)
        checker = GuaranteeChecker(guarantees)

        checker.record_cost(0.15)
        assert not checker.can_proceed()

        checker.reset()

        assert checker.can_proceed()
        assert checker.current_cost == 0.0
        assert checker.recursive_calls_count == 0
