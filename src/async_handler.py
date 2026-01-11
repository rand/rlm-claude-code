"""
Async handler for parallel recursive queries.

Implements: Spec §8.1 Phase 3 - Async Sub-calls
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .recursive_handler import RecursiveREPL


@dataclass
class BatchedQuery:
    """A query waiting to be batched with others."""

    query: str
    context: Any
    spawn_repl: bool = False
    max_tokens: int | None = None
    future: asyncio.Future[str] | None = None

    def __post_init__(self) -> None:
        """Create future if not provided and event loop exists."""
        if self.future is None:
            try:
                loop = asyncio.get_running_loop()
                self.future = loop.create_future()
            except RuntimeError:
                # No running event loop - future will be set later
                pass


@dataclass
class ParallelResult:
    """Result from parallel query execution."""

    results: list[str]
    errors: list[Exception | None]
    execution_time_ms: float
    cancelled_count: int = 0


class AsyncQueryHandler:
    """
    Handler for parallel and batched recursive queries.

    Implements: Spec §8.1 Phase 3 - Async Sub-calls

    Provides:
    - Parallel execution of independent queries
    - Request batching to reduce API overhead
    - Cancellation support for long-running queries
    - Timeout management
    """

    def __init__(
        self,
        repl: RecursiveREPL,
        max_concurrent: int = 5,
        batch_delay_ms: float = 50.0,
        default_timeout_s: float = 30.0,
    ):
        """
        Initialize async query handler.

        Args:
            repl: The RecursiveREPL instance to use for queries
            max_concurrent: Maximum concurrent queries
            batch_delay_ms: Delay before executing batch (for grouping)
            default_timeout_s: Default timeout for queries
        """
        self.repl = repl
        self.max_concurrent = max_concurrent
        self.batch_delay_ms = batch_delay_ms
        self.default_timeout_s = default_timeout_s

        # Batching state
        self._pending_batch: list[BatchedQuery] = []
        self._batch_task: asyncio.Task[None] | None = None
        self._batch_lock = asyncio.Lock()

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Cancellation tracking
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._cancelled = False

    async def parallel_queries(
        self,
        queries: list[tuple[str, Any]],
        spawn_repl: bool = False,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> ParallelResult:
        """
        Execute multiple queries in parallel.

        Implements: Spec §8.1 Parallel recursive queries

        Args:
            queries: List of (query, context) tuples
            spawn_repl: Whether to spawn REPL for each query
            max_tokens: Max tokens per query response
            timeout_s: Timeout for entire batch

        Returns:
            ParallelResult with results and any errors
        """
        import time

        start_time = time.time()
        timeout = timeout_s or self.default_timeout_s

        # Create tasks for each query
        tasks: list[asyncio.Task[str]] = []
        for query, context in queries:
            task = asyncio.create_task(
                self._execute_with_semaphore(
                    query, context, spawn_repl, max_tokens
                )
            )
            self._active_tasks.add(task)
            tasks.append(task)

        # Wait for all with timeout
        results: list[str] = []
        errors: list[Exception | None] = []
        cancelled_count = 0

        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                cancelled_count += 1

            # Collect results
            for task in tasks:
                if task in done:
                    try:
                        results.append(task.result())
                        errors.append(None)
                    except Exception as e:
                        results.append("")
                        errors.append(e)
                else:
                    results.append("")
                    errors.append(TimeoutError(f"Query timed out after {timeout}s"))

        except asyncio.CancelledError:
            # Cancel all tasks on external cancellation
            for task in tasks:
                task.cancel()
            raise

        finally:
            # Clean up tracking
            for task in tasks:
                self._active_tasks.discard(task)

        execution_time = (time.time() - start_time) * 1000

        return ParallelResult(
            results=results,
            errors=errors,
            execution_time_ms=execution_time,
            cancelled_count=cancelled_count,
        )

    async def _execute_with_semaphore(
        self,
        query: str,
        context: Any,
        spawn_repl: bool,
        max_tokens: int | None,
    ) -> str:
        """Execute a single query with concurrency control."""
        async with self._semaphore:
            if self._cancelled:
                raise asyncio.CancelledError("Handler was cancelled")
            return await self.repl.recursive_query(
                query, context, spawn_repl=spawn_repl, max_tokens=max_tokens
            )

    async def batched_query(
        self,
        query: str,
        context: Any,
        spawn_repl: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """
        Submit a query to be batched with others.

        Implements: Spec §8.1 Request batching

        Queries submitted within batch_delay_ms of each other
        will be executed together for efficiency.

        Args:
            query: The query string
            context: Context for the query
            spawn_repl: Whether to spawn REPL
            max_tokens: Max tokens for response

        Returns:
            Query response
        """
        # Create future in the current event loop
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        batched = BatchedQuery(
            query=query,
            context=context,
            spawn_repl=spawn_repl,
            max_tokens=max_tokens,
            future=future,
        )

        async with self._batch_lock:
            self._pending_batch.append(batched)

            # Start batch timer if not running
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._execute_batch())

        # Wait for result
        return await future

    async def _execute_batch(self) -> None:
        """Execute pending batch after delay."""
        # Wait for more queries to accumulate
        await asyncio.sleep(self.batch_delay_ms / 1000.0)

        async with self._batch_lock:
            if not self._pending_batch:
                return

            batch = self._pending_batch.copy()
            self._pending_batch.clear()

        # Execute all queries in parallel
        queries = [(b.query, b.context) for b in batch]

        # Use first query's settings (could be smarter)
        spawn_repl = batch[0].spawn_repl if batch else False
        max_tokens = batch[0].max_tokens if batch else None

        result = await self.parallel_queries(
            queries,
            spawn_repl=spawn_repl,
            max_tokens=max_tokens,
        )

        # Distribute results to futures
        for i, batched in enumerate(batch):
            if batched.future is not None:
                error = result.errors[i]
                if error is not None:
                    batched.future.set_exception(error)
                else:
                    batched.future.set_result(result.results[i])

    async def cancel_all(self) -> int:
        """
        Cancel all active queries.

        Implements: Spec §8.1 Cancellation support

        Returns:
            Number of tasks cancelled
        """
        self._cancelled = True
        cancelled_count = 0

        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
                cancelled_count += 1

        # Wait for cancellations to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        self._active_tasks.clear()
        self._cancelled = False

        return cancelled_count

    def reset(self) -> None:
        """Reset handler state for reuse."""
        self._pending_batch.clear()
        self._cancelled = False


class AsyncContextManager:
    """
    Async context manager for query batches.

    Usage:
        async with AsyncContextManager(handler) as batch:
            result1 = await batch.add("query1", context1)
            result2 = await batch.add("query2", context2)
        # All queries execute when exiting context
    """

    def __init__(self, handler: AsyncQueryHandler):
        self.handler = handler
        self._queries: list[tuple[str, Any, asyncio.Future[str]]] = []

    async def __aenter__(self) -> AsyncContextManager:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            # Cancel on exception
            await self.handler.cancel_all()
            return

        if not self._queries:
            return

        # Execute all collected queries
        queries = [(q, c) for q, c, _ in self._queries]
        result = await self.handler.parallel_queries(queries)

        # Set results on futures
        for i, (_, _, future) in enumerate(self._queries):
            if result.errors[i] is not None:
                future.set_exception(result.errors[i])
            else:
                future.set_result(result.results[i])

    def add(self, query: str, context: Any) -> asyncio.Future[str]:
        """
        Add a query to the batch.

        Args:
            query: The query string
            context: Context for the query

        Returns:
            Future that will contain the result
        """
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._queries.append((query, context, future))
        return future


async def run_with_timeout(
    coro: Coroutine[Any, Any, Any],
    timeout_s: float,
    on_timeout: Callable[[], Any] | None = None,
) -> Any:
    """
    Run a coroutine with timeout and optional cleanup.

    Args:
        coro: Coroutine to run
        timeout_s: Timeout in seconds
        on_timeout: Optional callback on timeout

    Returns:
        Coroutine result

    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except TimeoutError:
        if on_timeout:
            on_timeout()
        raise


__all__ = [
    "AsyncQueryHandler",
    "AsyncContextManager",
    "BatchedQuery",
    "ParallelResult",
    "run_with_timeout",
]
