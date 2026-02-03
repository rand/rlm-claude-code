"""
Tests for progress reporting interface.

Tests: SPEC-01.05 (Phase 4 - Progress Reporting)
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.progress import (
    CancellationToken,
    CancelledException,
    CompositeProgressCallback,
    ConsoleProgressCallback,
    NullProgressCallback,
    OperationType,
    ProgressCallback,
    ProgressContext,
    ProgressStats,
    ProgressUpdate,
    ThrottledProgressCallback,
    create_progress_context,
)


class MockProgressCallback:
    """Mock callback for testing."""

    def __init__(self) -> None:
        self.updates: list[ProgressUpdate] = []
        self.stats: ProgressStats | None = None
        self.errors: list[tuple[Exception, ProgressUpdate]] = []

    def on_progress(self, update: ProgressUpdate) -> None:
        self.updates.append(update)

    def on_complete(self, stats: ProgressStats) -> None:
        self.stats = stats

    def on_error(self, error: Exception, update: ProgressUpdate) -> None:
        self.errors.append((error, update))


class TestProgressUpdate:
    """Tests for ProgressUpdate dataclass."""

    def test_create_progress_update(self):
        """Create a progress update."""
        update = ProgressUpdate(
            operation=OperationType.INDEXING,
            current=5,
            total=10,
            message="Indexing files",
        )

        assert update.operation == OperationType.INDEXING
        assert update.current == 5
        assert update.total == 10
        assert update.message == "Indexing files"
        assert update.elapsed_ms == 0.0
        assert update.estimated_remaining_ms is None

    def test_progress_update_with_all_fields(self):
        """Create progress update with all fields."""
        update = ProgressUpdate(
            operation=OperationType.ANALYZING,
            current=3,
            total=10,
            message="Analyzing",
            elapsed_ms=500.0,
            estimated_remaining_ms=1000.0,
            detail="Processing file.py",
            depth=2,
        )

        assert update.elapsed_ms == 500.0
        assert update.estimated_remaining_ms == 1000.0
        assert update.detail == "Processing file.py"
        assert update.depth == 2


class TestCancellationToken:
    """Tests for CancellationToken."""

    def test_initial_state(self):
        """Token is not cancelled initially."""
        token = CancellationToken()
        assert token.is_cancelled is False

    def test_cancel(self):
        """Can cancel token."""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_check_not_cancelled(self):
        """Check does not raise when not cancelled."""
        token = CancellationToken()
        token.check()  # Should not raise

    def test_check_cancelled_raises(self):
        """Check raises when cancelled."""
        token = CancellationToken()
        token.cancel()

        with pytest.raises(CancelledException):
            token.check()

    @pytest.mark.asyncio
    async def test_wait_for_cancellation_timeout(self):
        """Wait times out if not cancelled."""
        token = CancellationToken()
        result = await token.wait_for_cancellation(timeout=0.01)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_cancellation_cancelled(self):
        """Wait returns True if cancelled."""
        token = CancellationToken()

        async def cancel_later():
            await asyncio.sleep(0.01)
            token.cancel()

        task = asyncio.create_task(cancel_later())
        result = await token.wait_for_cancellation(timeout=1.0)
        await task

        assert result is True


class TestProgressContext:
    """Tests for ProgressContext."""

    def test_context_manager(self):
        """ProgressContext works as context manager."""
        callback = MockProgressCallback()

        with ProgressContext(
            operation=OperationType.INDEXING,
            total=10,
            callback=callback,
        ) as ctx:
            ctx.advance(5)
            ctx.advance(5)

        assert len(callback.updates) == 3  # Initial + 2 advances
        assert callback.stats is not None
        assert callback.stats.completed_operations == 10

    def test_advance(self):
        """Advance increases progress."""
        callback = MockProgressCallback()

        with ProgressContext(
            operation=OperationType.INDEXING,
            total=10,
            callback=callback,
        ) as ctx:
            ctx.advance(3)
            ctx.advance(2, detail="Processing")

        # Check updates
        assert callback.updates[-1].current == 5
        assert callback.updates[-1].detail == "Processing"

    def test_set_progress(self):
        """Set_progress sets absolute progress."""
        callback = MockProgressCallback()

        with ProgressContext(
            operation=OperationType.INDEXING,
            total=10,
            callback=callback,
        ) as ctx:
            ctx.set_progress(7)

        assert callback.updates[-1].current == 7

    def test_cancellation_on_advance(self):
        """Advance checks cancellation."""
        callback = MockProgressCallback()
        token = CancellationToken()
        token.cancel()

        with (
            pytest.raises(CancelledException),
            ProgressContext(
                operation=OperationType.INDEXING,
                total=10,
                callback=callback,
                cancellation_token=token,
            ) as ctx,
        ):
            ctx.advance(1)

    def test_error_handling(self):
        """Errors are reported to callback."""
        callback = MockProgressCallback()

        with (
            pytest.raises(ValueError),
            ProgressContext(
                operation=OperationType.INDEXING,
                total=10,
                callback=callback,
            ) as ctx,
        ):
            ctx.advance(5)
            raise ValueError("Test error")

        assert len(callback.errors) == 1
        assert isinstance(callback.errors[0][0], ValueError)
        assert callback.stats is None  # Not completed normally

    def test_elapsed_time(self):
        """Elapsed time is tracked."""
        callback = MockProgressCallback()

        with ProgressContext(
            operation=OperationType.INDEXING,
            total=2,
            callback=callback,
        ) as ctx:
            time.sleep(0.01)
            ctx.advance(1)

        # Check elapsed time was recorded
        assert callback.updates[-1].elapsed_ms > 0

    def test_estimated_remaining(self):
        """Estimated remaining time is calculated."""
        callback = MockProgressCallback()

        with ProgressContext(
            operation=OperationType.INDEXING,
            total=10,
            callback=callback,
        ) as ctx:
            time.sleep(0.01)
            ctx.advance(5)

        # Should have estimate for remaining 5
        update = callback.updates[-1]
        assert update.estimated_remaining_ms is not None

    def test_message_template(self):
        """Custom message template is used."""
        callback = MockProgressCallback()

        with ProgressContext(
            operation=OperationType.ANALYZING,
            total=5,
            callback=callback,
            message_template="Processing {current} of {total}",
        ) as ctx:
            ctx.advance(3)

        assert "Processing 3 of 5" in callback.updates[-1].message

    def test_depth_tracking(self):
        """Depth is tracked for nested operations."""
        callback = MockProgressCallback()

        with ProgressContext(
            operation=OperationType.DECOMPOSING,
            total=3,
            callback=callback,
            depth=2,
        ) as ctx:
            ctx.advance(1)

        assert callback.updates[-1].depth == 2

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Works as async context manager."""
        callback = MockProgressCallback()

        async with ProgressContext(
            operation=OperationType.INDEXING,
            total=5,
            callback=callback,
        ) as ctx:
            ctx.advance(5)

        assert callback.stats is not None
        assert callback.stats.completed_operations == 5


class TestCreateProgressContext:
    """Tests for create_progress_context helper."""

    def test_with_callback(self):
        """Create with callback."""
        callback = MockProgressCallback()
        ctx = create_progress_context(
            OperationType.INDEXING,
            total=10,
            callback=callback,
        )

        with ctx:
            ctx.advance(5)

        assert len(callback.updates) > 0

    def test_with_function(self):
        """Create with simple function."""
        updates: list[ProgressUpdate] = []

        def on_progress(update: ProgressUpdate) -> None:
            updates.append(update)

        ctx = create_progress_context(
            OperationType.INDEXING,
            total=5,
            callback=on_progress,
        )

        with ctx:
            ctx.advance(3)

        assert len(updates) > 0

    def test_with_cancellation(self):
        """Create with cancellation token."""
        token = CancellationToken()
        ctx = create_progress_context(
            OperationType.INDEXING,
            total=10,
            cancellation_token=token,
        )

        token.cancel()

        with pytest.raises(CancelledException), ctx:
            ctx.advance(1)

    def test_custom_message_template(self):
        """Create with custom message template."""
        callback = MockProgressCallback()
        ctx = create_progress_context(
            OperationType.ANALYZING,
            total=5,
            callback=callback,
            message_template="Step {current}/{total}",
        )

        with ctx:
            ctx.advance(2)

        assert "Step 2/5" in callback.updates[-1].message


class TestNullProgressCallback:
    """Tests for NullProgressCallback."""

    def test_accepts_all_events(self):
        """Null callback accepts all events without error."""
        callback = NullProgressCallback()

        update = ProgressUpdate(
            operation=OperationType.INDEXING,
            current=5,
            total=10,
            message="Test",
        )

        callback.on_progress(update)
        callback.on_complete(ProgressStats())
        callback.on_error(ValueError("test"), update)
        # No assertions - just verify no exceptions


class TestConsoleProgressCallback:
    """Tests for ConsoleProgressCallback."""

    def test_creates_callback(self):
        """Can create console callback."""
        callback = ConsoleProgressCallback()
        assert callback is not None

    def test_with_options(self):
        """Can create with options."""
        callback = ConsoleProgressCallback(
            show_elapsed=False,
            show_eta=False,
            prefix="[TEST] ",
        )
        assert callback.prefix == "[TEST] "

    def test_is_progress_callback(self):
        """Is a valid ProgressCallback."""
        callback = ConsoleProgressCallback()
        assert isinstance(callback, ProgressCallback)


class TestCompositeProgressCallback:
    """Tests for CompositeProgressCallback."""

    def test_forwards_to_all(self):
        """Forwards events to all callbacks."""
        cb1 = MockProgressCallback()
        cb2 = MockProgressCallback()
        composite = CompositeProgressCallback([cb1, cb2])

        update = ProgressUpdate(
            operation=OperationType.INDEXING,
            current=5,
            total=10,
            message="Test",
        )

        composite.on_progress(update)

        assert len(cb1.updates) == 1
        assert len(cb2.updates) == 1

    def test_add_callback(self):
        """Can add callback."""
        cb = MockProgressCallback()
        composite = CompositeProgressCallback()
        composite.add(cb)

        update = ProgressUpdate(
            operation=OperationType.INDEXING,
            current=1,
            total=2,
            message="Test",
        )
        composite.on_progress(update)

        assert len(cb.updates) == 1

    def test_remove_callback(self):
        """Can remove callback."""
        cb = MockProgressCallback()
        composite = CompositeProgressCallback([cb])
        composite.remove(cb)

        update = ProgressUpdate(
            operation=OperationType.INDEXING,
            current=1,
            total=2,
            message="Test",
        )
        composite.on_progress(update)

        assert len(cb.updates) == 0


class TestThrottledProgressCallback:
    """Tests for ThrottledProgressCallback."""

    def test_throttles_updates(self):
        """Throttles frequent updates."""
        inner = MockProgressCallback()
        throttled = ThrottledProgressCallback(inner, interval_ms=100)

        # Rapid updates
        for i in range(10):
            update = ProgressUpdate(
                operation=OperationType.INDEXING,
                current=i,
                total=20,
                message="Test",
            )
            throttled.on_progress(update)

        # Should have throttled some updates
        assert len(inner.updates) < 10

    def test_always_reports_completion(self):
        """Always reports when current == total."""
        inner = MockProgressCallback()
        throttled = ThrottledProgressCallback(inner, interval_ms=100)

        # First update
        throttled.on_progress(
            ProgressUpdate(
                operation=OperationType.INDEXING,
                current=0,
                total=10,
                message="Test",
            )
        )

        # Completion update (should not be throttled)
        throttled.on_progress(
            ProgressUpdate(
                operation=OperationType.INDEXING,
                current=10,
                total=10,
                message="Test",
            )
        )

        assert len(inner.updates) == 2  # Both should be reported

    def test_always_reports_errors(self):
        """Always reports errors."""
        inner = MockProgressCallback()
        throttled = ThrottledProgressCallback(inner, interval_ms=100)

        update = ProgressUpdate(
            operation=OperationType.INDEXING,
            current=5,
            total=10,
            message="Test",
        )
        throttled.on_error(ValueError("test"), update)

        assert len(inner.errors) == 1


class TestProgressStats:
    """Tests for ProgressStats."""

    def test_default_values(self):
        """Default values are sensible."""
        stats = ProgressStats()

        assert stats.total_operations == 0
        assert stats.completed_operations == 0
        assert stats.failed_operations == 0
        assert stats.total_time_ms == 0.0
        assert stats.operations_per_second == 0.0

    def test_computed_in_context(self):
        """Stats are computed from context."""
        callback = MockProgressCallback()

        with ProgressContext(
            operation=OperationType.INDEXING,
            total=5,
            callback=callback,
        ) as ctx:
            ctx.advance(3)
            ctx.advance(2)

        stats = callback.stats
        assert stats is not None
        assert stats.total_operations == 5
        assert stats.completed_operations == 5
        assert stats.total_time_ms > 0


class TestOperationType:
    """Tests for OperationType enum."""

    def test_all_types_exist(self):
        """All expected operation types exist."""
        types = [
            OperationType.INDEXING,
            OperationType.SEARCHING,
            OperationType.CHUNKING,
            OperationType.ANALYZING,
            OperationType.DECOMPOSING,
            OperationType.SYNTHESIZING,
            OperationType.LLM_CALL,
            OperationType.BATCH_OPERATION,
            OperationType.MAP_REDUCE,
            OperationType.CONTEXT_LOAD,
        ]

        assert len(types) == 10

    def test_types_have_string_values(self):
        """Types have string values."""
        assert OperationType.INDEXING.value == "indexing"
        assert OperationType.LLM_CALL.value == "llm_call"
