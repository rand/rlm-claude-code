"""
Unit tests for cost_tracker module.

Implements: Spec §8.1 Phase 3 - Cost Tracking tests
"""

from src.cost_tracker import (
    CostComponent,
    CostEstimate,
    CostTracker,
    TokenUsage,
    estimate_context_tokens,
    estimate_tokens,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_create_usage(self):
        """Can create token usage record."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            component=CostComponent.ROOT_PROMPT,
        )

        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.total_tokens == 1500

    def test_estimate_cost_sonnet(self):
        """Estimates cost correctly for Sonnet."""
        usage = TokenUsage(
            input_tokens=1_000_000,  # 1M input
            output_tokens=100_000,  # 100K output
            model="claude-sonnet-4-20250514",
        )

        # Sonnet: $3/1M input, $15/1M output
        expected = 3.0 + 1.5  # $3 + $1.50
        assert abs(usage.estimate_cost() - expected) < 0.01

    def test_estimate_cost_opus(self):
        """Estimates cost correctly for Opus."""
        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=100_000,
            model="claude-opus-4-5-20251101",
        )

        # Opus: $15/1M input, $75/1M output
        expected = 15.0 + 7.5
        assert abs(usage.estimate_cost() - expected) < 0.01


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_create_estimate(self):
        """Can create cost estimate."""
        estimate = CostEstimate(
            estimated_input_tokens=5000,
            estimated_output_tokens=2000,
            model="claude-sonnet-4-20250514",
            confidence=0.8,
            component=CostComponent.RECURSIVE_CALL,
        )

        assert estimate.estimated_total_tokens == 7000
        assert estimate.confidence == 0.8

    def test_estimated_cost(self):
        """Calculates estimated cost."""
        estimate = CostEstimate(
            estimated_input_tokens=100_000,
            estimated_output_tokens=50_000,
            model="claude-sonnet-4-20250514",
            confidence=0.7,
            component=CostComponent.ROOT_PROMPT,
        )

        # $0.30 input + $0.75 output
        expected = 0.30 + 0.75
        assert abs(estimate.estimated_cost - expected) < 0.01


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_record_usage(self):
        """Can record token usage."""
        tracker = CostTracker(budget_tokens=100_000)

        usage = tracker.record_usage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            component=CostComponent.ROOT_PROMPT,
        )

        assert usage.total_tokens == 1500
        assert tracker.total_tokens == 1500

    def test_track_multiple_usages(self):
        """Tracks multiple usages correctly."""
        tracker = CostTracker()

        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT)
        tracker.record_usage(2000, 1000, "haiku", CostComponent.RECURSIVE_CALL)

        assert tracker.total_tokens == 4500

    def test_estimate_cost(self):
        """Can estimate cost before execution."""
        tracker = CostTracker()

        estimate = tracker.estimate_cost(
            prompt_length=4000,  # ~1000 tokens
            expected_output_length=2000,  # ~500 tokens
            model="claude-sonnet-4-20250514",
            component=CostComponent.ROOT_PROMPT,
        )

        # Should include overhead
        assert estimate.estimated_input_tokens > 1000
        assert estimate.estimated_output_tokens >= 500

    def test_would_exceed_budget_tokens(self):
        """Detects when estimate would exceed token budget."""
        tracker = CostTracker(budget_tokens=1000)
        tracker.record_usage(800, 0, "sonnet", CostComponent.ROOT_PROMPT)

        estimate = CostEstimate(
            estimated_input_tokens=300,
            estimated_output_tokens=100,
            model="sonnet",
            confidence=0.8,
            component=CostComponent.RECURSIVE_CALL,
        )

        would_exceed, reason = tracker.would_exceed_budget(estimate)
        assert would_exceed is True
        assert "token budget" in reason

    def test_would_exceed_budget_dollars(self):
        """Detects when estimate would exceed dollar budget."""
        # Use high token budget so only cost budget triggers
        tracker = CostTracker(budget_tokens=10_000_000, budget_dollars=0.01)
        tracker.record_usage(1000, 500, "opus", CostComponent.ROOT_PROMPT)

        estimate = CostEstimate(
            estimated_input_tokens=100_000,
            estimated_output_tokens=50_000,
            model="opus",
            confidence=0.8,
            component=CostComponent.RECURSIVE_CALL,
        )

        would_exceed, reason = tracker.would_exceed_budget(estimate)
        assert would_exceed is True
        assert "cost budget" in reason

    def test_remaining_tokens(self):
        """Calculates remaining tokens correctly."""
        tracker = CostTracker(budget_tokens=10_000)
        tracker.record_usage(3000, 1000, "sonnet", CostComponent.ROOT_PROMPT)

        assert tracker.remaining_tokens == 6000

    def test_budget_warning_alert(self):
        """Emits warning when approaching budget."""
        alerts_received = []
        tracker = CostTracker(budget_tokens=1000, warning_threshold=0.8)
        tracker.on_alert(lambda a: alerts_received.append(a))

        # Use 850 tokens (85% of budget)
        tracker.record_usage(850, 0, "sonnet", CostComponent.ROOT_PROMPT)

        assert len(alerts_received) == 1
        assert alerts_received[0].severity == "warning"

    def test_budget_critical_alert(self):
        """Emits critical alert when exceeding budget."""
        alerts_received = []
        tracker = CostTracker(budget_tokens=1000)
        tracker.on_alert(lambda a: alerts_received.append(a))

        # Exceed budget
        tracker.record_usage(1100, 0, "sonnet", CostComponent.ROOT_PROMPT)

        critical_alerts = [a for a in alerts_received if a.severity == "critical"]
        assert len(critical_alerts) >= 1

    def test_breakdown_by_component(self):
        """Gets breakdown by component."""
        tracker = CostTracker()
        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT)
        tracker.record_usage(500, 200, "haiku", CostComponent.RECURSIVE_CALL)
        tracker.record_usage(300, 100, "haiku", CostComponent.RECURSIVE_CALL)

        breakdown = tracker.get_breakdown_by_component()

        assert breakdown["root_prompt"]["tokens"] == 1500
        assert breakdown["root_prompt"]["calls"] == 1
        assert breakdown["recursive_call"]["tokens"] == 1100
        assert breakdown["recursive_call"]["calls"] == 2

    def test_breakdown_by_model(self):
        """Gets breakdown by model."""
        tracker = CostTracker()
        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT)
        tracker.record_usage(2000, 1000, "haiku", CostComponent.RECURSIVE_CALL)

        breakdown = tracker.get_breakdown_by_model()

        assert breakdown["sonnet"]["total_tokens"] == 1500
        assert breakdown["haiku"]["total_tokens"] == 3000

    def test_get_summary(self):
        """Gets complete cost summary."""
        tracker = CostTracker(budget_tokens=10_000, budget_dollars=5.0)
        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT)

        summary = tracker.get_summary()

        assert summary["total_tokens"] == 1500
        assert summary["budget_tokens"] == 10_000
        assert "by_component" in summary
        assert "by_model" in summary

    def test_reset(self):
        """Can reset all tracking."""
        tracker = CostTracker()
        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT)

        tracker.reset()

        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0


class TestLatencyTracking:
    """Tests for latency tracking functionality."""

    def test_token_usage_latency_field(self):
        """TokenUsage has latency_ms field."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
            latency_ms=1500.0,
        )

        assert usage.latency_ms == 1500.0

    def test_tokens_per_second_calculation(self):
        """Calculates tokens per second from latency."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
            latency_ms=1000.0,  # 1 second
        )

        # 500 output tokens / 1 second = 500 tps
        assert usage.tokens_per_second == 500.0

    def test_tokens_per_second_zero_latency(self):
        """Returns zero when latency is zero."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
            latency_ms=0.0,
        )

        assert usage.tokens_per_second == 0.0

    def test_record_usage_with_latency(self):
        """Can record usage with latency."""
        tracker = CostTracker()

        usage = tracker.record_usage(
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
            latency_ms=2000.0,
        )

        assert usage.latency_ms == 2000.0
        assert tracker.total_latency_ms == 2000.0

    def test_total_latency_ms(self):
        """Tracks total latency across operations."""
        tracker = CostTracker()

        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT, latency_ms=1500.0)
        tracker.record_usage(500, 200, "haiku", CostComponent.RECURSIVE_CALL, latency_ms=800.0)

        assert tracker.total_latency_ms == 2300.0

    def test_average_latency_ms(self):
        """Calculates average latency per operation."""
        tracker = CostTracker()

        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT, latency_ms=1000.0)
        tracker.record_usage(500, 200, "haiku", CostComponent.RECURSIVE_CALL, latency_ms=500.0)

        assert tracker.average_latency_ms == 750.0

    def test_average_latency_empty(self):
        """Average latency is zero with no operations."""
        tracker = CostTracker()

        assert tracker.average_latency_ms == 0.0

    def test_average_tokens_per_second(self):
        """Calculates average throughput."""
        tracker = CostTracker()

        # 500 output tokens in 1000ms + 200 output tokens in 500ms = 700 tokens in 1.5s
        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT, latency_ms=1000.0)
        tracker.record_usage(500, 200, "haiku", CostComponent.RECURSIVE_CALL, latency_ms=500.0)

        # 700 tokens / 1.5 seconds ≈ 466.67 tps
        assert abs(tracker.average_tokens_per_second - 466.67) < 1.0

    def test_average_tokens_per_second_no_latency(self):
        """Returns zero when total latency is zero."""
        tracker = CostTracker()
        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT, latency_ms=0.0)

        assert tracker.average_tokens_per_second == 0.0

    def test_get_summary_includes_latency(self):
        """Summary includes latency statistics."""
        tracker = CostTracker()

        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT, latency_ms=1500.0)

        summary = tracker.get_summary()

        assert "latency" in summary
        assert summary["latency"]["total_ms"] == 1500.0
        assert summary["latency"]["average_ms"] == 1500.0
        assert summary["latency"]["throughput_tps"] > 0


class TestFormatReport:
    """Tests for format_report method."""

    def test_format_report_structure(self):
        """Report has expected structure."""
        tracker = CostTracker(budget_tokens=10_000, budget_dollars=1.0)
        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT, latency_ms=500.0)

        report = tracker.format_report()

        assert "Cost Report" in report
        assert "Tokens:" in report
        assert "Cost:" in report
        assert "Latency:" in report
        assert "Throughput:" in report
        assert "API Calls:" in report

    def test_format_report_model_breakdown(self):
        """Report includes model breakdown."""
        tracker = CostTracker()
        tracker.record_usage(1000, 500, "sonnet", CostComponent.ROOT_PROMPT)
        tracker.record_usage(500, 200, "haiku", CostComponent.RECURSIVE_CALL)

        report = tracker.format_report()

        assert "By Model" in report
        assert "sonnet" in report
        assert "haiku" in report

    def test_format_report_with_alerts(self):
        """Report includes alerts when budget exceeded."""
        tracker = CostTracker(budget_tokens=1000)
        tracker.record_usage(1100, 0, "sonnet", CostComponent.ROOT_PROMPT)

        report = tracker.format_report()

        assert "Alerts" in report or "exceeded" in report

    def test_format_report_empty_tracker(self):
        """Report works with no usage recorded."""
        tracker = CostTracker()

        report = tracker.format_report()

        assert "Cost Report" in report
        assert "Tokens: 0" in report


class TestEstimateTokens:
    """Tests for token estimation utilities."""

    def test_estimate_tokens_basic(self):
        """Estimates tokens from text using tiktoken."""
        text = "a" * 400  # 400 chars
        tokens = estimate_tokens(text)

        # tiktoken gives accurate counts, typically fewer than 4 chars/token heuristic
        # for repeated characters. The key is tokens > 0 and reasonable.
        assert tokens > 0
        assert tokens <= 400  # Can't be more tokens than characters

    def test_estimate_tokens_realistic_text(self):
        """Estimates tokens for realistic English text."""
        # Realistic text has ~4 chars/token on average
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = estimate_tokens(text)

        # Should be roughly len(text) / 4, within reasonable bounds
        expected_approx = len(text) // 4
        assert tokens > expected_approx * 0.5  # At least half of estimate
        assert tokens < expected_approx * 2.0  # At most double

    def test_estimate_context_tokens(self):
        """Estimates tokens for full context."""
        messages = [
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "Hi " * 100},
        ]
        files = {
            "file1.py": "x" * 400,
            "file2.py": "y" * 400,
        }
        tool_outputs = [
            {"tool": "bash", "content": "output " * 50},
        ]

        tokens = estimate_context_tokens(messages, files, tool_outputs)

        # Should account for content + overhead
        assert tokens > 0
        assert tokens > (600 + 300 + 350) // 4  # Minimum from content


# =============================================================================
# SPEC-14.60-14.65: Session Budget Tests
# =============================================================================


class TestSessionBudget:
    """Tests for SessionBudget (SPEC-14.60-14.65)."""

    def test_default_budget_500k_tokens(self):
        """Default budget is 500K tokens (SPEC-14.62)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget()

        assert budget.budget_tokens == 500_000

    def test_tracks_tokens_not_dollars(self):
        """Budget tracks in tokens not dollars (SPEC-14.61)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget()
        budget.record_tokens(1000, mode="micro")

        assert budget.total_tokens == 1000
        # No dollar tracking on SessionBudget

    def test_tracks_cumulative_session_cost(self):
        """Tracks cumulative session cost (SPEC-14.60)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget()
        budget.record_tokens(1000, mode="micro")
        budget.record_tokens(2000, mode="balanced")
        budget.record_tokens(3000, mode="thorough")

        assert budget.total_tokens == 6000

    def test_warning_at_50_percent(self):
        """Warning at 50% utilization (SPEC-14.63)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=10_000)
        warnings = budget.record_tokens(5000, mode="micro")

        assert len(warnings) == 1
        assert warnings[0].threshold_percent == 50

    def test_warning_at_75_percent(self):
        """Warning at 75% utilization (SPEC-14.63)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=10_000)
        budget.record_tokens(5000, mode="micro")  # 50%
        warnings = budget.record_tokens(2500, mode="micro")  # 75%

        assert len(warnings) == 1
        assert warnings[0].threshold_percent == 75

    def test_warning_at_90_percent(self):
        """Warning at 90% utilization (SPEC-14.63)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=10_000)
        budget.record_tokens(7500, mode="micro")  # 75%
        warnings = budget.record_tokens(1500, mode="micro")  # 90%

        assert len(warnings) == 1
        assert warnings[0].threshold_percent == 90

    def test_warnings_not_repeated(self):
        """Same warning threshold not emitted twice."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=10_000)
        warnings1 = budget.record_tokens(5000, mode="micro")  # 50%
        warnings2 = budget.record_tokens(100, mode="micro")  # Still 50%

        assert len(warnings1) == 1
        assert len(warnings2) == 0  # Not repeated

    def test_can_escalate_within_budget(self):
        """Can escalate if within budget (SPEC-14.64)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=500_000)
        budget.record_tokens(100_000, mode="micro")

        can, reason = budget.can_escalate("thorough")

        assert can is True
        assert reason is None

    def test_cannot_escalate_exceeds_budget(self):
        """Cannot escalate if would exceed budget (SPEC-14.64)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=50_000)
        budget.record_tokens(40_000, mode="micro")

        can, reason = budget.can_escalate("thorough")  # Needs 100K

        assert can is False
        assert "exceed budget" in reason

    def test_tracks_by_mode(self):
        """Tracks tokens by execution mode (SPEC-14.65)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget()
        budget.record_tokens(1000, mode="micro")
        budget.record_tokens(20000, mode="balanced")
        budget.record_tokens(50000, mode="thorough")

        breakdown = budget.get_mode_breakdown()

        assert breakdown["micro"]["tokens_used"] == 1000
        assert breakdown["balanced"]["tokens_used"] == 20000
        assert breakdown["thorough"]["tokens_used"] == 50000

    def test_mode_over_target_tracking(self):
        """Tracks when mode exceeds target (SPEC-14.65)."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget()
        budget.record_tokens(5000, mode="micro")  # Target is 2K

        breakdown = budget.get_mode_breakdown()

        assert breakdown["micro"]["over_target"] is True

    def test_warning_callback(self):
        """Warning callbacks are invoked."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=10_000)
        warnings_received = []
        budget.on_warning(lambda w: warnings_received.append(w))

        budget.record_tokens(5000, mode="micro")

        assert len(warnings_received) == 1
        assert warnings_received[0].threshold_percent == 50

    def test_get_summary(self):
        """get_summary returns expected structure."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=100_000)
        budget.record_tokens(50_000, mode="micro")

        summary = budget.get_summary()

        assert summary["total_tokens"] == 50_000
        assert summary["budget_tokens"] == 100_000
        assert summary["remaining_tokens"] == 50_000
        assert summary["utilization_percent"] == 50
        assert "by_mode" in summary

    def test_reset(self):
        """Reset clears all tracking."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=10_000)
        budget.record_tokens(5000, mode="micro")
        budget.reset()

        assert budget.total_tokens == 0
        assert budget.remaining_tokens == 10_000

    def test_get_available_for_mode(self):
        """get_available_for_mode returns correct value."""
        from src.cost_tracker import SessionBudget

        budget = SessionBudget(budget_tokens=100_000)
        budget.record_tokens(99_000, mode="micro")

        # Only 1K remaining, but balanced wants 25K
        available = budget.get_available_for_mode("balanced")

        assert available == 1000  # min(remaining, target)


class TestCreateSessionBudget:
    """Tests for create_session_budget function."""

    def test_creates_with_default_budget(self):
        """Creates with 500K default budget."""
        from src.cost_tracker import create_session_budget

        budget = create_session_budget()

        assert budget.budget_tokens == 500_000

    def test_creates_with_custom_budget(self):
        """Creates with custom budget."""
        from src.cost_tracker import create_session_budget

        budget = create_session_budget(budget_tokens=1_000_000)

        assert budget.budget_tokens == 1_000_000


class TestModeTokenTargets:
    """Tests for MODE_TOKEN_TARGETS constants."""

    def test_micro_target_under_2k(self):
        """Micro mode target is <2K tokens (SPEC-14.65)."""
        from src.cost_tracker import MODE_TOKEN_TARGETS

        assert MODE_TOKEN_TARGETS["micro"] == 2_000

    def test_balanced_target_under_25k(self):
        """Balanced mode target is <25K tokens (SPEC-14.65)."""
        from src.cost_tracker import MODE_TOKEN_TARGETS

        assert MODE_TOKEN_TARGETS["balanced"] == 25_000

    def test_thorough_target_under_100k(self):
        """Thorough mode target is <100K tokens (SPEC-14.65)."""
        from src.cost_tracker import MODE_TOKEN_TARGETS

        assert MODE_TOKEN_TARGETS["thorough"] == 100_000
