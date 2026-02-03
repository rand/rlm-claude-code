"""
Property-based tests for progress reporting.

Tests: SPEC-01.05 (Phase 4 - Progress Reporting)

Property tests verify invariants:
- Progress updates track state correctly
- Cancellation is thread-safe and idempotent
- Throttling respects intervals
- Stats are accurately computed
"""

from __future__ import annotations

import time

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.progress import (
    CancellationToken,
    CancelledException,
    OperationType,
    ProgressContext,
    ProgressStats,
    ProgressUpdate,
    ThrottledProgressCallback,
)


class MockCallback:
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


class TestProgressUpdateProperties:
    """Property tests for ProgressUpdate dataclass."""

    @given(
        current=st.integers(min_value=0, max_value=10000),
        total=st.integers(min_value=1, max_value=10000),
        elapsed_ms=st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
        depth=st.integers(min_value=0, max_value=10),
    )
    def test_progress_update_preserves_fields(
        self, current: int, total: int, elapsed_ms: float, depth: int
    ):
        """ProgressUpdate preserves all fields."""
        update = ProgressUpdate(
            operation=OperationType.INDEXING,
            current=current,
            total=total,
            message="Test message",
            elapsed_ms=elapsed_ms,
            depth=depth,
        )

        assert update.current == current
        assert update.total == total
        assert update.elapsed_ms == elapsed_ms
        assert update.depth == depth
        assert update.operation == OperationType.INDEXING

    @given(
        current=st.integers(min_value=0, max_value=10000),
        total=st.integers(min_value=1, max_value=10000),
    )
    def test_progress_percentage_calculation(self, current: int, total: int):
        """Progress percentage is correctly calculable from update."""
        assume(total > 0)

        update = ProgressUpdate(
            operation=OperationType.ANALYZING,
            current=current,
            total=total,
            message="Test",
        )

        # Percentage should be calculable
        pct = (update.current / update.total) * 100
        assert 0 <= pct <= (current / total) * 100 + 0.01  # Allow tiny float error


class TestCancellationTokenProperties:
    """Property tests for CancellationToken."""

    @given(cancel_count=st.integers(min_value=0, max_value=100))
    def test_cancellation_is_idempotent(self, cancel_count: int):
        """Multiple cancellations have same effect as one."""
        token = CancellationToken()

        for _ in range(cancel_count):
            token.cancel()

        if cancel_count > 0:
            assert token.is_cancelled is True
        else:
            assert token.is_cancelled is False

    @given(st.booleans())
    def test_check_raises_iff_cancelled(self, should_cancel: bool):
        """Check raises CancelledException iff token is cancelled."""
        token = CancellationToken()

        if should_cancel:
            token.cancel()

        if should_cancel:
            raised = False
            try:
                token.check()
            except CancelledException:
                raised = True
            assert raised, "Should raise CancelledException when cancelled"
        else:
            token.check()  # Should not raise


class TestProgressContextProperties:
    """Property tests for ProgressContext."""

    @given(
        total=st.integers(min_value=1, max_value=1000),
        advances=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=20),
    )
    @settings(max_examples=30, deadline=5000)
    def test_advance_accumulates_correctly(self, total: int, advances: list[int]):
        """Advance accumulates progress correctly."""
        callback = MockCallback()

        with ProgressContext(
            operation=OperationType.CHUNKING,
            total=total,
            callback=callback,
        ) as ctx:
            for adv in advances:
                ctx.advance(adv)

        expected_sum = sum(advances)
        assert callback.updates[-1].current == expected_sum

    @given(
        total=st.integers(min_value=10, max_value=1000),
        progress_values=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=10),
    )
    @settings(max_examples=20, deadline=5000)
    def test_set_progress_sets_absolute(self, total: int, progress_values: list[int]):
        """set_progress sets absolute position, not relative."""
        callback = MockCallback()

        with ProgressContext(
            operation=OperationType.SEARCHING,
            total=total,
            callback=callback,
        ) as ctx:
            for val in progress_values:
                ctx.set_progress(val)

        # Final position should be last set value
        assert callback.updates[-1].current == progress_values[-1]

    @given(total=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=5000)
    def test_completed_operations_match_progress(self, total: int):
        """Stats completed_operations matches final progress."""
        callback = MockCallback()

        with ProgressContext(
            operation=OperationType.MAP_REDUCE,
            total=total,
            callback=callback,
        ) as ctx:
            ctx.advance(total)

        assert callback.stats is not None
        assert callback.stats.completed_operations == total
        assert callback.stats.total_operations == total

    @given(depth=st.integers(min_value=0, max_value=10))
    @settings(max_examples=20, deadline=5000)
    def test_depth_is_preserved(self, depth: int):
        """Depth is preserved in updates."""
        callback = MockCallback()

        with ProgressContext(
            operation=OperationType.DECOMPOSING,
            total=5,
            callback=callback,
            depth=depth,
        ) as ctx:
            ctx.advance(1)

        assert all(u.depth == depth for u in callback.updates)


class TestProgressStatsProperties:
    """Property tests for ProgressStats."""

    @given(
        total_ops=st.integers(min_value=0, max_value=10000),
        completed_ops=st.integers(min_value=0, max_value=10000),
        failed_ops=st.integers(min_value=0, max_value=1000),
        total_time_ms=st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
    def test_stats_preserve_all_fields(
        self, total_ops: int, completed_ops: int, failed_ops: int, total_time_ms: float
    ):
        """ProgressStats preserves all fields."""
        stats = ProgressStats(
            total_operations=total_ops,
            completed_operations=completed_ops,
            failed_operations=failed_ops,
            total_time_ms=total_time_ms,
        )

        assert stats.total_operations == total_ops
        assert stats.completed_operations == completed_ops
        assert stats.failed_operations == failed_ops
        assert stats.total_time_ms == total_time_ms


class TestThrottledCallbackProperties:
    """Property tests for ThrottledProgressCallback."""

    @given(
        interval_ms=st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
        n_updates=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=20, deadline=10000)
    def test_throttling_reduces_updates(self, interval_ms: float, n_updates: int):
        """Throttling reduces number of forwarded updates."""
        inner = MockCallback()
        throttled = ThrottledProgressCallback(inner, interval_ms=interval_ms)

        # Send rapid updates
        for i in range(n_updates):
            update = ProgressUpdate(
                operation=OperationType.INDEXING,
                current=i,
                total=n_updates + 10,  # Not completing
                message="Test",
            )
            throttled.on_progress(update)

        # Should have throttled some updates (unless interval is very small)
        # At minimum, first update should always be sent
        assert len(inner.updates) >= 1
        # If we sent many updates rapidly, should have throttled some
        if n_updates > 5 and interval_ms > 10:
            assert len(inner.updates) < n_updates

    @given(
        interval_ms=st.floats(min_value=100, max_value=1000, allow_nan=False, allow_infinity=False),
        n_updates=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=15, deadline=10000)
    def test_completion_always_reported(self, interval_ms: float, n_updates: int):
        """Completion (current == total) is always reported."""
        inner = MockCallback()
        throttled = ThrottledProgressCallback(inner, interval_ms=interval_ms)

        total = n_updates

        # Send updates including completion
        for i in range(n_updates + 1):
            update = ProgressUpdate(
                operation=OperationType.ANALYZING,
                current=i,
                total=total,
                message="Test",
            )
            throttled.on_progress(update)

        # Completion update should be in the forwarded updates
        completion_updates = [u for u in inner.updates if u.current >= u.total]
        assert len(completion_updates) >= 1, "Completion should always be reported"

    @given(n_errors=st.integers(min_value=1, max_value=10))
    @settings(max_examples=15, deadline=5000)
    def test_errors_always_reported(self, n_errors: int):
        """Errors are always reported, never throttled."""
        inner = MockCallback()
        throttled = ThrottledProgressCallback(inner, interval_ms=1000)  # Long interval

        for i in range(n_errors):
            update = ProgressUpdate(
                operation=OperationType.LLM_CALL,
                current=i,
                total=100,
                message="Test",
            )
            throttled.on_error(ValueError(f"Error {i}"), update)

        assert len(inner.errors) == n_errors, "All errors should be reported"


class TestOperationTypeProperties:
    """Property tests for OperationType enum."""

    @given(op_idx=st.integers(min_value=0, max_value=9))
    def test_operation_types_have_string_values(self, op_idx: int):
        """All operation types have non-empty string values."""
        ops = list(OperationType)
        assume(op_idx < len(ops))

        op = ops[op_idx]
        assert isinstance(op.value, str)
        assert len(op.value) > 0


class TestElapsedTimeProperties:
    """Property tests for elapsed time tracking."""

    @given(total=st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, deadline=5000)
    def test_elapsed_time_increases(self, total: int):
        """Elapsed time is non-decreasing across updates."""
        callback = MockCallback()

        with ProgressContext(
            operation=OperationType.BATCH_OPERATION,
            total=total,
            callback=callback,
        ) as ctx:
            for i in range(min(total, 5)):
                time.sleep(0.001)  # Small delay to ensure time passes
                ctx.advance(1)

        # Elapsed times should be non-decreasing
        elapsed_times = [u.elapsed_ms for u in callback.updates]
        for i in range(1, len(elapsed_times)):
            assert elapsed_times[i] >= elapsed_times[i - 1], "Elapsed time should not decrease"
