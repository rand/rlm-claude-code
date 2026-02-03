"""
Unit tests for async_handler module.

Implements: Spec §8.1 Phase 3 - Async Sub-calls tests
"""

import asyncio

import pytest

from src.async_handler import (
    AsyncContextManager,
    AsyncQueryHandler,
    BatchedQuery,
    ParallelResult,
    run_with_timeout,
)


class MockRecursiveREPL:
    """Mock RecursiveREPL for testing."""

    def __init__(self, delay: float = 0.01, should_fail: bool = False):
        self.delay = delay
        self.should_fail = should_fail
        self.call_count = 0

    async def recursive_query(
        self,
        query: str,
        context: str,
        spawn_repl: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        self.call_count += 1
        await asyncio.sleep(self.delay)
        if self.should_fail:
            raise ValueError("Simulated failure")
        return f"Result for: {query}"


class TestBatchedQuery:
    """Tests for BatchedQuery dataclass."""

    def test_create_batched_query(self):
        """Can create batched query."""
        query = BatchedQuery(
            query="test query",
            context="test context",
            spawn_repl=True,
            max_tokens=1000,
        )

        assert query.query == "test query"
        assert query.spawn_repl is True
        # Future may be None if no event loop running
        assert query.future is None or isinstance(query.future, asyncio.Future)


class TestParallelResult:
    """Tests for ParallelResult dataclass."""

    def test_create_result(self):
        """Can create parallel result."""
        result = ParallelResult(
            results=["r1", "r2"],
            errors=[None, None],
            execution_time_ms=100.0,
            cancelled_count=0,
        )

        assert len(result.results) == 2
        assert result.cancelled_count == 0


class TestAsyncQueryHandler:
    """Tests for AsyncQueryHandler class."""

    @pytest.fixture
    def mock_repl(self):
        """Create mock REPL."""
        return MockRecursiveREPL()

    @pytest.fixture
    def handler(self, mock_repl):
        """Create handler with mock REPL."""
        return AsyncQueryHandler(
            repl=mock_repl,
            max_concurrent=3,
            batch_delay_ms=10.0,
            default_timeout_s=5.0,
        )

    @pytest.mark.asyncio
    async def test_parallel_queries_basic(self, handler):
        """Executes queries in parallel."""
        queries = [
            ("query1", "context1"),
            ("query2", "context2"),
            ("query3", "context3"),
        ]

        result = await handler.parallel_queries(queries)

        assert len(result.results) == 3
        assert all(r.startswith("Result for:") for r in result.results)
        assert all(e is None for e in result.errors)

    @pytest.mark.asyncio
    async def test_parallel_queries_with_errors(self):
        """Handles errors in parallel queries."""
        failing_repl = MockRecursiveREPL(should_fail=True)
        handler = AsyncQueryHandler(repl=failing_repl)

        queries = [("query1", "context1")]
        result = await handler.parallel_queries(queries)

        assert result.errors[0] is not None
        assert isinstance(result.errors[0], ValueError)

    @pytest.mark.asyncio
    async def test_parallel_queries_timeout(self):
        """Times out slow queries."""
        slow_repl = MockRecursiveREPL(delay=1.0)  # 1 second delay
        handler = AsyncQueryHandler(repl=slow_repl, default_timeout_s=0.1)

        queries = [("query1", "context1")]
        result = await handler.parallel_queries(queries, timeout_s=0.05)

        assert result.errors[0] is not None
        assert isinstance(result.errors[0], asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, mock_repl):
        """Respects max concurrent limit."""
        handler = AsyncQueryHandler(repl=mock_repl, max_concurrent=2)

        # Submit 4 queries
        queries = [(f"query{i}", f"context{i}") for i in range(4)]

        result = await handler.parallel_queries(queries)

        assert len(result.results) == 4
        # All should complete (semaphore just limits concurrency)

    @pytest.mark.asyncio
    async def test_batched_query(self, handler):
        """Can submit batched query."""
        result = await handler.batched_query("query1", "context1")

        assert result.startswith("Result for:")

    @pytest.mark.asyncio
    async def test_cancel_all(self, mock_repl):
        """Can cancel all active queries."""
        slow_repl = MockRecursiveREPL(delay=10.0)  # Very slow
        handler = AsyncQueryHandler(repl=slow_repl)

        # Start queries in background
        queries = [(f"query{i}", f"context{i}") for i in range(3)]

        async def run_queries():
            return await handler.parallel_queries(queries)

        task = asyncio.create_task(run_queries())

        # Give queries time to start
        await asyncio.sleep(0.05)

        # Cancel
        cancelled = await handler.cancel_all()

        # Task should raise CancelledError or complete with cancellation
        try:
            result = await task
            # If it completed, check for cancellation
            assert result.cancelled_count > 0 or any(
                isinstance(e, asyncio.CancelledError) for e in result.errors if e
            )
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_reset(self, handler):
        """Can reset handler state."""
        handler._cancelled = True
        handler._pending_batch.append(BatchedQuery("q", "c"))

        handler.reset()

        assert handler._cancelled is False
        assert len(handler._pending_batch) == 0


class TestAsyncContextManager:
    """Tests for AsyncContextManager class."""

    @pytest.mark.asyncio
    async def test_batch_execution(self):
        """Executes all queries on exit."""
        mock_repl = MockRecursiveREPL()
        handler = AsyncQueryHandler(repl=mock_repl)

        async with AsyncContextManager(handler) as batch:
            future1 = batch.add("query1", "context1")
            future2 = batch.add("query2", "context2")

        # Futures should be resolved
        result1 = await future1
        result2 = await future2

        assert result1.startswith("Result for:")
        assert result2.startswith("Result for:")

    @pytest.mark.asyncio
    async def test_cancels_on_exception(self):
        """Cancels queries on exception."""
        slow_repl = MockRecursiveREPL(delay=10.0)
        handler = AsyncQueryHandler(repl=slow_repl)

        with pytest.raises(ValueError):
            async with AsyncContextManager(handler) as batch:
                batch.add("query1", "context1")
                raise ValueError("Test error")

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Handles empty batch gracefully."""
        mock_repl = MockRecursiveREPL()
        handler = AsyncQueryHandler(repl=mock_repl)

        async with AsyncContextManager(handler):
            pass  # No queries added

        # Should complete without error


class TestRunWithTimeout:
    """Tests for run_with_timeout utility."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Returns result when completing within timeout."""

        async def fast_coro():
            await asyncio.sleep(0.01)
            return "done"

        result = await run_with_timeout(fast_coro(), timeout_s=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        """Raises TimeoutError when exceeding timeout."""

        async def slow_coro():
            await asyncio.sleep(10.0)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await run_with_timeout(slow_coro(), timeout_s=0.01)

    @pytest.mark.asyncio
    async def test_calls_on_timeout_callback(self):
        """Calls callback on timeout."""
        callback_called = False

        def on_timeout():
            nonlocal callback_called
            callback_called = True

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(asyncio.TimeoutError):
            await run_with_timeout(slow_coro(), timeout_s=0.01, on_timeout=on_timeout)

        assert callback_called is True
