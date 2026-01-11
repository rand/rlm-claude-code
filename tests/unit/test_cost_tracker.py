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
