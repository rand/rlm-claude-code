"""
Unit tests for cost_tracker module.

Implements: Spec §8.1 Phase 3 - Cost Tracking tests
"""

import pytest

from src.cost_tracker import (
    BudgetAlert,
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
        """Estimates tokens from text."""
        text = "a" * 400  # 400 chars
        tokens = estimate_tokens(text)

        # ~4 chars per token
        assert tokens == 100

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
