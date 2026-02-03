"""
Tests for async executor implementation.

@trace SPEC-08.01-08.06
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.async_executor import (
    AsyncExecutor,
    PartialFailureResult,
    SpeculativeExecution,
)
from src.types import DeferredOperation

# --- Test fixtures ---


@dataclass
class MockOperation:
    """Mock operation for testing."""

    operation_id: str
    query: str
    delay_ms: float = 0
    should_fail: bool = False
    result: str = "success"


def create_deferred_op(
    op_id: str,
    query: str = "test query",
    context: str = "test context",
    operation_type: str = "recursive_query",
) -> DeferredOperation:
    """Create a DeferredOperation for testing."""
    return DeferredOperation(
        operation_id=op_id,
        operation_type=operation_type,
        query=query,
        context=context,
        spawn_repl=False,
    )


# --- SPEC-08.01: Fully asynchronous execution ---


class TestAsyncExecutor:
    """Tests for AsyncExecutor class."""

    @pytest.mark.asyncio
    async def test_parallel_execution_basic(self) -> None:
        """
        @trace SPEC-08.01
        Test that operations execute in parallel, not sequentially.
        """
        executor = AsyncExecutor(max_concurrency=10)

        # Create operations that track execution timing
        execution_times: list[float] = []
        start_time = asyncio.get_event_loop().time()

        async def mock_execute(op: DeferredOperation) -> str:
            execution_times.append(asyncio.get_event_loop().time() - start_time)
            await asyncio.sleep(0.1)  # Simulate work
            return f"result_{op.operation_id}"

        ops = [create_deferred_op(f"op_{i}") for i in range(5)]

        results = await executor.execute_parallel(ops, mock_execute)

        # All 5 ops should complete in ~0.1s (parallel), not ~0.5s (sequential)
        total_time = asyncio.get_event_loop().time() - start_time
        assert total_time < 0.3, f"Parallel execution took {total_time}s, expected <0.3s"

        # All operations should have started within a small window
        assert max(execution_times) - min(execution_times) < 0.05

        # All results should be present
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.success
            assert result.result == f"result_op_{i}"

    @pytest.mark.asyncio
    async def test_taskgroup_used(self) -> None:
        """
        @trace SPEC-08.01
        Verify that asyncio.TaskGroup is used for true parallelism.
        """
        executor = AsyncExecutor(max_concurrency=10)

        taskgroup_used = False

        original_taskgroup = asyncio.TaskGroup

        class TrackingTaskGroup(asyncio.TaskGroup):
            async def __aenter__(self):
                nonlocal taskgroup_used
                taskgroup_used = True
                return await super().__aenter__()

        # Monkey-patch to track usage
        asyncio.TaskGroup = TrackingTaskGroup

        try:
            ops = [create_deferred_op(f"op_{i}") for i in range(3)]

            async def mock_execute(op: DeferredOperation) -> str:
                return "ok"

            await executor.execute_parallel(ops, mock_execute)
            assert taskgroup_used, "asyncio.TaskGroup should be used for parallel execution"
        finally:
            asyncio.TaskGroup = original_taskgroup


# --- SPEC-08.02: Configurable max_concurrency ---


class TestMaxConcurrency:
    """Tests for max_concurrency configuration."""

    @pytest.mark.asyncio
    async def test_default_max_concurrency(self) -> None:
        """
        @trace SPEC-08.02
        Default max_concurrency should be 10.
        """
        executor = AsyncExecutor()
        assert executor.max_concurrency == 10

    @pytest.mark.asyncio
    async def test_custom_max_concurrency(self) -> None:
        """
        @trace SPEC-08.02
        max_concurrency should be configurable.
        """
        executor = AsyncExecutor(max_concurrency=5)
        assert executor.max_concurrency == 5

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        """
        @trace SPEC-08.02
        Operations should respect the concurrency limit.
        """
        max_concurrent = 3
        executor = AsyncExecutor(max_concurrency=max_concurrent)

        # Track concurrent operations
        current_concurrent = 0
        max_observed_concurrent = 0

        async def mock_execute(op: DeferredOperation) -> str:
            nonlocal current_concurrent, max_observed_concurrent
            current_concurrent += 1
            max_observed_concurrent = max(max_observed_concurrent, current_concurrent)
            await asyncio.sleep(0.05)  # Hold the slot
            current_concurrent -= 1
            return "ok"

        # Run 10 operations with max 3 concurrent
        ops = [create_deferred_op(f"op_{i}") for i in range(10)]
        await executor.execute_parallel(ops, mock_execute)

        assert max_observed_concurrent <= max_concurrent, (
            f"Max concurrent was {max_observed_concurrent}, should be <= {max_concurrent}"
        )


# --- SPEC-08.03: Speculative execution ---


class TestSpeculativeExecution:
    """Tests for speculative execution with winner-takes-all."""

    @pytest.mark.asyncio
    async def test_speculative_returns_first_success(self) -> None:
        """
        @trace SPEC-08.03
        Speculative execution should return the first successful result.
        """
        executor = AsyncExecutor(max_concurrency=10)

        async def fast_op() -> str:
            await asyncio.sleep(0.05)
            return "fast_result"

        async def slow_op() -> str:
            await asyncio.sleep(0.2)
            return "slow_result"

        spec = SpeculativeExecution(
            primary=fast_op,
            alternatives=[slow_op],
        )

        result = await executor.execute_speculative(spec)

        assert result.result == "fast_result"
        assert result.execution_time_ms < 150  # Should be fast

    @pytest.mark.asyncio
    async def test_speculative_cancels_losers(self) -> None:
        """
        @trace SPEC-08.03
        Speculative execution should cancel losing operations.
        """
        executor = AsyncExecutor(max_concurrency=10)

        slow_completed = False

        async def fast_op() -> str:
            await asyncio.sleep(0.05)
            return "fast"

        async def slow_op() -> str:
            nonlocal slow_completed
            await asyncio.sleep(1.0)
            slow_completed = True
            return "slow"

        spec = SpeculativeExecution(
            primary=fast_op,
            alternatives=[slow_op],
        )

        result = await executor.execute_speculative(spec)

        # Wait a bit to ensure slow op would have completed if not cancelled
        await asyncio.sleep(0.1)

        assert result.result == "fast"
        assert not slow_completed, "Slow operation should have been cancelled"

    @pytest.mark.asyncio
    async def test_speculative_fallback_on_primary_failure(self) -> None:
        """
        @trace SPEC-08.03
        If primary fails, should use alternative result.
        """
        executor = AsyncExecutor(max_concurrency=10)

        async def failing_primary() -> str:
            await asyncio.sleep(0.02)
            raise ValueError("Primary failed")

        async def working_alternative() -> str:
            await asyncio.sleep(0.05)
            return "alternative_worked"

        spec = SpeculativeExecution(
            primary=failing_primary,
            alternatives=[working_alternative],
        )

        result = await executor.execute_speculative(spec)

        assert result.success
        assert result.result == "alternative_worked"


# --- SPEC-08.04: Partial failure handling ---


class TestPartialFailures:
    """Tests for graceful partial failure handling."""

    @pytest.mark.asyncio
    async def test_partial_failures_reported(self) -> None:
        """
        @trace SPEC-08.04
        Partial failures should be reported, not raise exceptions.
        """
        executor = AsyncExecutor(max_concurrency=10)

        async def execute(op: DeferredOperation) -> str:
            if "fail" in op.operation_id:
                raise ValueError(f"Operation {op.operation_id} failed")
            return f"success_{op.operation_id}"

        ops = [
            create_deferred_op("op_success_1"),
            create_deferred_op("op_fail_1"),
            create_deferred_op("op_success_2"),
            create_deferred_op("op_fail_2"),
            create_deferred_op("op_success_3"),
        ]

        results = await executor.execute_parallel(ops, execute)

        # All operations should have results
        assert len(results) == 5

        # Check successful operations
        successes = [r for r in results if r.success]
        assert len(successes) == 3

        # Check failed operations
        failures = [r for r in results if not r.success]
        assert len(failures) == 2
        for f in failures:
            assert f.error is not None
            assert "failed" in f.error.lower()

    @pytest.mark.asyncio
    async def test_partial_failures_best_effort_result(self) -> None:
        """
        @trace SPEC-08.04
        Should synthesize best-effort result from successful operations.
        """
        executor = AsyncExecutor(max_concurrency=10)

        async def execute(op: DeferredOperation) -> str:
            if op.operation_id == "op_2":
                raise ValueError("Failed")
            return f"result_{op.operation_id}"

        ops = [create_deferred_op(f"op_{i}") for i in range(5)]

        results = await executor.execute_parallel(ops, execute)

        # Get aggregated result
        aggregated = executor.aggregate_results(results)

        assert isinstance(aggregated, PartialFailureResult)
        assert aggregated.successful_count == 4
        assert aggregated.failed_count == 1
        assert len(aggregated.results) == 4
        assert len(aggregated.errors) == 1


# --- SPEC-08.05: Budget constraints during parallel execution ---


class TestBudgetConstraints:
    """Tests for budget constraint enforcement during parallel execution."""

    @pytest.mark.asyncio
    async def test_budget_checked_before_execution(self) -> None:
        """
        @trace SPEC-08.05
        Budget should be checked before starting parallel operations.
        """
        budget_checker = MagicMock()
        budget_checker.check_can_proceed = MagicMock(return_value=True)
        budget_checker.record_cost = MagicMock()

        executor = AsyncExecutor(max_concurrency=10, budget_checker=budget_checker)

        ops = [create_deferred_op(f"op_{i}") for i in range(3)]

        async def mock_execute(op: DeferredOperation) -> str:
            return "ok"

        await executor.execute_parallel(ops, mock_execute)

        # Budget should have been checked
        assert budget_checker.check_can_proceed.called

    @pytest.mark.asyncio
    async def test_budget_exceeded_stops_execution(self) -> None:
        """
        @trace SPEC-08.05
        If budget exceeded, should stop and return partial results.
        """
        call_count = 0

        class MockBudgetChecker:
            def check_can_proceed(self, estimated_cost: float = 0.1) -> bool:
                nonlocal call_count
                call_count += 1
                # Allow first 2 checks, then deny
                return call_count <= 2

            def record_cost(self, cost: float) -> None:
                pass

        executor = AsyncExecutor(max_concurrency=2, budget_checker=MockBudgetChecker())

        execution_count = 0

        async def mock_execute(op: DeferredOperation) -> str:
            nonlocal execution_count
            execution_count += 1
            return f"result_{op.operation_id}"

        ops = [create_deferred_op(f"op_{i}") for i in range(5)]

        results = await executor.execute_parallel(ops, mock_execute)

        # Some should have executed before budget exceeded
        assert execution_count >= 1
        # Results should include budget-exceeded info for some operations
        assert any(r.error and "budget" in r.error.lower() for r in results if not r.success)

    @pytest.mark.asyncio
    async def test_budget_respected_during_parallel(self) -> None:
        """
        @trace SPEC-08.05
        Budget consumption should be tracked across parallel operations.
        """
        total_cost = 0.0

        class MockBudgetChecker:
            def __init__(self, max_cost: float):
                self.max_cost = max_cost
                self.current_cost = 0.0

            def check_can_proceed(self, estimated_cost: float = 0.1) -> bool:
                return self.current_cost + estimated_cost <= self.max_cost

            def record_cost(self, cost: float) -> None:
                nonlocal total_cost
                self.current_cost += cost
                total_cost += cost

        budget = MockBudgetChecker(max_cost=0.3)
        executor = AsyncExecutor(max_concurrency=10, budget_checker=budget)

        async def mock_execute(op: DeferredOperation) -> str:
            budget.record_cost(0.1)
            return "ok"

        ops = [create_deferred_op(f"op_{i}") for i in range(5)]

        results = await executor.execute_parallel(ops, mock_execute)

        # Should have stopped at budget limit
        successes = [r for r in results if r.success]
        assert len(successes) <= 3  # max_cost / cost_per_op


# --- Integration tests ---


class TestAsyncExecutorIntegration:
    """Integration tests for AsyncExecutor."""

    @pytest.mark.asyncio
    async def test_latency_improvement(self) -> None:
        """
        @trace SPEC-08.01
        Verify 3-5x latency improvement over sequential execution.
        """
        executor = AsyncExecutor(max_concurrency=10)

        op_count = 10
        op_delay = 0.1  # 100ms per operation

        async def mock_execute(op: DeferredOperation) -> str:
            await asyncio.sleep(op_delay)
            return f"result_{op.operation_id}"

        ops = [create_deferred_op(f"op_{i}") for i in range(op_count)]

        # Measure parallel execution time
        start = asyncio.get_event_loop().time()
        await executor.execute_parallel(ops, mock_execute)
        parallel_time = asyncio.get_event_loop().time() - start

        # Sequential would take: op_count * op_delay = 1.0s
        sequential_time = op_count * op_delay

        # Parallel should be at least 3x faster
        speedup = sequential_time / parallel_time
        assert speedup >= 3, f"Speedup was only {speedup}x, expected >= 3x"

    @pytest.mark.asyncio
    async def test_empty_operations_list(self) -> None:
        """Handle empty operations list gracefully."""
        executor = AsyncExecutor()

        async def mock_execute(op: DeferredOperation) -> str:
            return "ok"

        results = await executor.execute_parallel([], mock_execute)

        assert results == []

    @pytest.mark.asyncio
    async def test_single_operation(self) -> None:
        """Handle single operation correctly."""
        executor = AsyncExecutor()

        async def mock_execute(op: DeferredOperation) -> str:
            return "single_result"

        results = await executor.execute_parallel(
            [create_deferred_op("single")],
            mock_execute,
        )

        assert len(results) == 1
        assert results[0].success
        assert results[0].result == "single_result"
