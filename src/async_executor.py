"""
Async executor for parallel RLM operations.

Implements: SPEC-08.01-08.06 Asynchronous Recursive Calls
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from .types import DeferredOperation

T = TypeVar("T")


class BudgetChecker(Protocol):
    """Protocol for budget checking during parallel execution."""

    def check_can_proceed(self, estimated_cost: float = 0.1) -> bool:
        """Check if operation can proceed within budget."""
        ...

    def record_cost(self, cost: float) -> None:
        """Record cost after operation completion."""
        ...


@dataclass
class ExecutionResult:
    """Result of a single async operation."""

    operation_id: str
    success: bool
    result: str | None = None
    error: str | None = None
    execution_time_ms: float = 0.0


@dataclass
class PartialFailureResult:
    """Aggregated result when some operations fail."""

    successful_count: int
    failed_count: int
    results: list[str]
    errors: list[tuple[str, str]]  # (operation_id, error_message)

    @property
    def total_count(self) -> int:
        """Total number of operations."""
        return self.successful_count + self.failed_count


@dataclass
class SpeculativeExecution:
    """
    Configuration for speculative execution.

    Primary operation runs alongside alternatives. First to succeed wins,
    others are cancelled.
    """

    primary: Callable[[], Awaitable[str]]
    alternatives: list[Callable[[], Awaitable[str]]] = field(default_factory=list)


@dataclass
class SpeculativeResult:
    """Result from speculative execution."""

    success: bool
    result: str | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    winner_index: int = 0  # 0 = primary, 1+ = alternative index


class AsyncExecutor:
    """
    Async executor for parallel RLM operations.

    Implements: SPEC-08.01-08.06

    Features:
    - True parallelism via asyncio.TaskGroup (SPEC-08.01)
    - Configurable max_concurrency (SPEC-08.02)
    - Speculative execution with cancellation (SPEC-08.03)
    - Graceful partial failure handling (SPEC-08.04)
    - Budget constraint enforcement (SPEC-08.05)
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        budget_checker: BudgetChecker | None = None,
    ):
        """
        Initialize async executor.

        Args:
            max_concurrency: Maximum concurrent operations (default: 10)
            budget_checker: Optional budget checker for cost enforcement
        """
        self.max_concurrency = max_concurrency
        self.budget_checker = budget_checker
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def execute_parallel(
        self,
        operations: list[DeferredOperation],
        executor_fn: Callable[[DeferredOperation], Awaitable[str]],
    ) -> list[ExecutionResult]:
        """
        Execute operations in parallel with bounded concurrency.

        Implements: SPEC-08.01-08.02

        Args:
            operations: List of operations to execute
            executor_fn: Async function to execute each operation

        Returns:
            List of ExecutionResult for each operation
        """
        if not operations:
            return []

        results: list[ExecutionResult] = []
        semaphore = self._get_semaphore()
        budget_exceeded = False

        async def execute_one(op: DeferredOperation) -> ExecutionResult:
            """Execute single operation with concurrency control."""
            nonlocal budget_exceeded

            # Check budget before acquiring semaphore
            if self.budget_checker and not self.budget_checker.check_can_proceed():
                budget_exceeded = True
                return ExecutionResult(
                    operation_id=op.operation_id,
                    success=False,
                    error="Budget exceeded",
                )

            start_time = time.perf_counter()

            async with semaphore:
                # Re-check budget after acquiring semaphore
                if budget_exceeded:
                    return ExecutionResult(
                        operation_id=op.operation_id,
                        success=False,
                        error="Budget exceeded",
                    )

                try:
                    result = await executor_fn(op)
                    execution_time = (time.perf_counter() - start_time) * 1000

                    return ExecutionResult(
                        operation_id=op.operation_id,
                        success=True,
                        result=result,
                        execution_time_ms=execution_time,
                    )
                except Exception as e:
                    execution_time = (time.perf_counter() - start_time) * 1000
                    return ExecutionResult(
                        operation_id=op.operation_id,
                        success=False,
                        error=str(e),
                        execution_time_ms=execution_time,
                    )

        # Check overall budget before starting
        if self.budget_checker:
            self.budget_checker.check_can_proceed()

        # Use TaskGroup for true parallelism (SPEC-08.01)
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(execute_one(op)) for op in operations]

        # Collect results in order
        results = [task.result() for task in tasks]

        return results

    async def execute_speculative(
        self,
        spec: SpeculativeExecution,
    ) -> SpeculativeResult:
        """
        Execute with speculative alternatives, cancel losers.

        Implements: SPEC-08.03

        Runs primary and alternatives in parallel. First successful result wins,
        all other tasks are cancelled.

        Args:
            spec: Speculative execution configuration

        Returns:
            Result from winning operation
        """
        start_time = time.perf_counter()

        # Build list of all operations
        all_ops: list[Callable[[], Awaitable[str]]] = [spec.primary, *spec.alternatives]

        # Track results and completion
        result_future: asyncio.Future[tuple[int, str]] = asyncio.get_event_loop().create_future()
        tasks: list[asyncio.Task[None]] = []

        async def run_and_report(index: int, op: Callable[[], Awaitable[str]]) -> None:
            """Run operation and report result if first to succeed."""
            try:
                result = await op()
                # Try to set result (first to succeed wins)
                if not result_future.done():
                    result_future.set_result((index, result))
            except Exception:
                # Don't set exception - let other ops try
                pass

        # Start all operations
        for i, op in enumerate(all_ops):
            task = asyncio.create_task(run_and_report(i, op))
            tasks.append(task)

        try:
            # Wait for first success with timeout
            winner_index, result = await asyncio.wait_for(result_future, timeout=60.0)

            execution_time = (time.perf_counter() - start_time) * 1000

            return SpeculativeResult(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                winner_index=winner_index,
            )

        except TimeoutError:
            execution_time = (time.perf_counter() - start_time) * 1000
            return SpeculativeResult(
                success=False,
                error="All operations timed out",
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return SpeculativeResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )

        finally:
            # Cancel all remaining tasks (SPEC-08.03: cancel losers)
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Wait for cancellations to complete
            await asyncio.gather(*tasks, return_exceptions=True)

    def aggregate_results(
        self,
        results: list[ExecutionResult],
    ) -> PartialFailureResult | list[str]:
        """
        Aggregate results, handling partial failures.

        Implements: SPEC-08.04

        Args:
            results: List of execution results

        Returns:
            PartialFailureResult if any failures, otherwise list of results
        """
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        if not failures:
            return [r.result for r in successes if r.result is not None]

        return PartialFailureResult(
            successful_count=len(successes),
            failed_count=len(failures),
            results=[r.result for r in successes if r.result is not None],
            errors=[(r.operation_id, r.error or "Unknown error") for r in failures],
        )


class AsyncRLMOrchestrator:
    """
    Async orchestrator for RLM recursive calls.

    Wraps AsyncExecutor with RLM-specific functionality including
    trajectory events and recursive handler integration.
    """

    def __init__(
        self,
        executor: AsyncExecutor | None = None,
        max_concurrency: int = 10,
    ):
        """
        Initialize async RLM orchestrator.

        Args:
            executor: Optional pre-configured executor
            max_concurrency: Max concurrent operations if creating executor
        """
        self.executor = executor or AsyncExecutor(max_concurrency=max_concurrency)

    async def execute_recursive_calls(
        self,
        operations: list[DeferredOperation],
        recursive_handler: Any,
    ) -> dict[str, str]:
        """
        Execute recursive calls with proper RLM handling.

        Args:
            operations: Deferred operations to execute
            recursive_handler: RecursiveREPL instance for making calls

        Returns:
            Dict mapping operation_id to result
        """

        async def execute_one(op: DeferredOperation) -> str:
            """Execute single recursive call."""
            return await recursive_handler.recursive_query(
                query=op.query,
                context=op.context,
                spawn_repl=op.spawn_repl,
            )

        results = await self.executor.execute_parallel(operations, execute_one)

        # Build result map
        result_map: dict[str, str] = {}
        for r in results:
            if r.success and r.result is not None:
                result_map[r.operation_id] = r.result
            else:
                result_map[r.operation_id] = f"[Error: {r.error}]"
        return result_map

    async def execute_with_speculation(
        self,
        primary_op: DeferredOperation,
        alternative_ops: list[DeferredOperation],
        recursive_handler: Any,
    ) -> str:
        """
        Execute with speculative alternatives.

        Args:
            primary_op: Primary operation
            alternative_ops: Alternative approaches
            recursive_handler: RecursiveREPL for making calls

        Returns:
            Result from winning operation
        """

        async def make_call(op: DeferredOperation) -> str:
            return await recursive_handler.recursive_query(
                query=op.query,
                context=op.context,
                spawn_repl=op.spawn_repl,
            )

        spec = SpeculativeExecution(
            primary=lambda: make_call(primary_op),
            alternatives=[lambda o=op: make_call(o) for op in alternative_ops],
        )

        result = await self.executor.execute_speculative(spec)

        if result.success:
            return result.result or ""
        else:
            return f"[Error: {result.error}]"


__all__ = [
    "AsyncExecutor",
    "AsyncRLMOrchestrator",
    "BudgetChecker",
    "ExecutionResult",
    "PartialFailureResult",
    "SpeculativeExecution",
    "SpeculativeResult",
]
