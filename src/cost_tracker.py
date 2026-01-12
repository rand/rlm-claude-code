"""
Cost tracking for RLM-Claude-Code.

Implements: Spec §8.1 Phase 3 - Cost Tracking
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CostComponent(Enum):
    """Components that incur token costs."""

    ROOT_PROMPT = "root_prompt"
    RECURSIVE_CALL = "recursive_call"
    REPL_EXECUTION = "repl_execution"
    CONTEXT_LOAD = "context_load"
    SUMMARIZATION = "summarization"
    TOOL_OUTPUT = "tool_output"


# Token costs per model (per 1M tokens as of spec date)
# Input / Output costs in dollars
MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    # Fallbacks for older model names
    "opus": {"input": 15.0, "output": 75.0},
    "sonnet": {"input": 3.0, "output": 15.0},
    "haiku": {"input": 0.80, "output": 4.0},
}


@dataclass
class TokenUsage:
    """Token usage for a single operation."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    component: CostComponent = CostComponent.ROOT_PROMPT
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0  # API call latency in milliseconds

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second (throughput)."""
        if self.latency_ms > 0:
            return (self.output_tokens / self.latency_ms) * 1000
        return 0.0

    def estimate_cost(self) -> float:
        """Estimate cost in dollars."""
        costs = MODEL_COSTS.get(self.model, MODEL_COSTS.get("sonnet", {}))
        input_cost = (self.input_tokens / 1_000_000) * costs.get("input", 3.0)
        output_cost = (self.output_tokens / 1_000_000) * costs.get("output", 15.0)
        return input_cost + output_cost


@dataclass
class CostEstimate:
    """Pre-execution cost estimate."""

    estimated_input_tokens: int
    estimated_output_tokens: int
    model: str
    confidence: float  # 0.0 to 1.0
    component: CostComponent

    @property
    def estimated_total_tokens(self) -> int:
        """Total estimated tokens."""
        return self.estimated_input_tokens + self.estimated_output_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in dollars."""
        costs = MODEL_COSTS.get(self.model, MODEL_COSTS.get("sonnet", {}))
        input_cost = (self.estimated_input_tokens / 1_000_000) * costs.get("input", 3.0)
        output_cost = (self.estimated_output_tokens / 1_000_000) * costs.get("output", 15.0)
        return input_cost + output_cost


@dataclass
class BudgetAlert:
    """Alert when budget threshold is crossed."""

    threshold_name: str
    threshold_value: float
    current_value: float
    message: str
    severity: str  # "warning", "critical"
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """
    Track and estimate token costs across RLM operations.

    Implements: Spec §8.1 Phase 3 - Cost Tracking

    Provides:
    - Token counting per component
    - Cost estimation before execution
    - Budget alerts
    - Cost breakdowns by model and component
    """

    def __init__(
        self,
        budget_tokens: int = 100_000,
        budget_dollars: float = 5.0,
        warning_threshold: float = 0.8,  # 80% of budget
    ):
        """
        Initialize cost tracker.

        Args:
            budget_tokens: Maximum tokens per session
            budget_dollars: Maximum cost per session in dollars
            warning_threshold: Fraction of budget that triggers warning
        """
        self.budget_tokens = budget_tokens
        self.budget_dollars = budget_dollars
        self.warning_threshold = warning_threshold

        self._usage: list[TokenUsage] = []
        self._alerts: list[BudgetAlert] = []
        self._alert_callbacks: list[Callable[[BudgetAlert], None]] = []

        # Track by component
        self._component_tokens: dict[CostComponent, int] = dict.fromkeys(CostComponent, 0)
        self._model_tokens: dict[str, dict[str, int]] = {}

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        component: CostComponent,
        latency_ms: float = 0.0,
    ) -> TokenUsage:
        """
        Record token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model used
            component: Which component used the tokens
            latency_ms: API call latency in milliseconds

        Returns:
            TokenUsage record
        """
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            component=component,
            latency_ms=latency_ms,
        )
        self._usage.append(usage)

        # Update aggregates
        self._component_tokens[component] += usage.total_tokens

        if model not in self._model_tokens:
            self._model_tokens[model] = {"input": 0, "output": 0}
        self._model_tokens[model]["input"] += input_tokens
        self._model_tokens[model]["output"] += output_tokens

        # Check budgets
        self._check_budgets()

        return usage

    def estimate_cost(
        self,
        prompt_length: int,
        expected_output_length: int,
        model: str,
        component: CostComponent,
        include_overhead: bool = True,
    ) -> CostEstimate:
        """
        Estimate cost before execution.

        Implements: Spec §8.1 Cost estimation before execution

        Args:
            prompt_length: Length of prompt in characters
            expected_output_length: Expected output length in characters
            model: Model to use
            component: Component making the call
            include_overhead: Include system prompt overhead

        Returns:
            CostEstimate with token and dollar estimates
        """
        # Estimate tokens from characters (roughly 4 chars per token)
        estimated_input = prompt_length // 4

        # Add overhead for system prompts
        if include_overhead:
            overhead = self._get_overhead_tokens(component)
            estimated_input += overhead

        estimated_output = expected_output_length // 4

        # Confidence based on how well we can estimate
        confidence = 0.7  # Base confidence
        if component == CostComponent.RECURSIVE_CALL:
            confidence = 0.5  # Less predictable
        elif component == CostComponent.SUMMARIZATION:
            confidence = 0.8  # More predictable

        return CostEstimate(
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            model=model,
            confidence=confidence,
            component=component,
        )

    def _get_overhead_tokens(self, component: CostComponent) -> int:
        """Get estimated system prompt overhead by component."""
        overheads = {
            CostComponent.ROOT_PROMPT: 2000,  # RLM system prompt
            CostComponent.RECURSIVE_CALL: 500,  # Sub-call framing
            CostComponent.REPL_EXECUTION: 100,  # REPL context
            CostComponent.CONTEXT_LOAD: 0,  # No overhead
            CostComponent.SUMMARIZATION: 300,  # Summarization instructions
            CostComponent.TOOL_OUTPUT: 50,  # Tool framing
        }
        return overheads.get(component, 0)

    def would_exceed_budget(self, estimate: CostEstimate) -> tuple[bool, str | None]:
        """
        Check if estimate would exceed budget.

        Args:
            estimate: Cost estimate to check

        Returns:
            (would_exceed, reason) tuple
        """
        new_total_tokens = self.total_tokens + estimate.estimated_total_tokens
        if new_total_tokens > self.budget_tokens:
            return True, f"Would exceed token budget ({new_total_tokens} > {self.budget_tokens})"

        new_total_cost = self.total_cost + estimate.estimated_cost
        if new_total_cost > self.budget_dollars:
            return True, f"Would exceed cost budget (${new_total_cost:.2f} > ${self.budget_dollars:.2f})"

        return False, None

    def _check_budgets(self) -> None:
        """Check budgets and emit alerts."""
        # Token budget
        token_fraction = self.total_tokens / self.budget_tokens
        if token_fraction >= 1.0:
            self._emit_alert(
                "token_budget",
                self.budget_tokens,
                self.total_tokens,
                f"Token budget exceeded: {self.total_tokens:,} / {self.budget_tokens:,}",
                "critical",
            )
        elif token_fraction >= self.warning_threshold:
            self._emit_alert(
                "token_warning",
                self.budget_tokens * self.warning_threshold,
                self.total_tokens,
                f"Approaching token budget: {self.total_tokens:,} / {self.budget_tokens:,} ({token_fraction:.0%})",
                "warning",
            )

        # Cost budget
        cost_fraction = self.total_cost / self.budget_dollars
        if cost_fraction >= 1.0:
            self._emit_alert(
                "cost_budget",
                self.budget_dollars,
                self.total_cost,
                f"Cost budget exceeded: ${self.total_cost:.2f} / ${self.budget_dollars:.2f}",
                "critical",
            )
        elif cost_fraction >= self.warning_threshold:
            self._emit_alert(
                "cost_warning",
                self.budget_dollars * self.warning_threshold,
                self.total_cost,
                f"Approaching cost budget: ${self.total_cost:.2f} / ${self.budget_dollars:.2f} ({cost_fraction:.0%})",
                "warning",
            )

    def _emit_alert(
        self,
        name: str,
        threshold: float,
        current: float,
        message: str,
        severity: str,
    ) -> None:
        """Emit a budget alert."""
        # Avoid duplicate alerts
        for alert in self._alerts:
            if alert.threshold_name == name and alert.severity == severity:
                return

        alert = BudgetAlert(
            threshold_name=name,
            threshold_value=threshold,
            current_value=current,
            message=message,
            severity=severity,
        )
        self._alerts.append(alert)

        # Notify callbacks
        for callback in self._alert_callbacks:
            callback(alert)

    def on_alert(self, callback: Callable[[BudgetAlert], None]) -> None:
        """Register callback for budget alerts."""
        self._alert_callbacks.append(callback)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all operations."""
        return sum(u.total_tokens for u in self._usage)

    @property
    def total_cost(self) -> float:
        """Total cost in dollars."""
        return sum(u.estimate_cost() for u in self._usage)

    @property
    def remaining_tokens(self) -> int:
        """Remaining tokens in budget."""
        return max(0, self.budget_tokens - self.total_tokens)

    @property
    def remaining_budget(self) -> float:
        """Remaining dollar budget."""
        return max(0.0, self.budget_dollars - self.total_cost)

    @property
    def total_latency_ms(self) -> float:
        """Total latency across all operations."""
        return sum(u.latency_ms for u in self._usage)

    @property
    def average_latency_ms(self) -> float:
        """Average latency per operation."""
        if not self._usage:
            return 0.0
        return self.total_latency_ms / len(self._usage)

    @property
    def average_tokens_per_second(self) -> float:
        """Average output throughput (tokens/second)."""
        total_output = sum(u.output_tokens for u in self._usage)
        total_time_s = self.total_latency_ms / 1000
        if total_time_s > 0:
            return total_output / total_time_s
        return 0.0

    def get_breakdown_by_component(self) -> dict[str, dict[str, Any]]:
        """Get cost breakdown by component."""
        breakdown: dict[str, dict[str, Any]] = {}

        for component in CostComponent:
            component_usage = [u for u in self._usage if u.component == component]
            if component_usage:
                breakdown[component.value] = {
                    "tokens": sum(u.total_tokens for u in component_usage),
                    "cost": sum(u.estimate_cost() for u in component_usage),
                    "calls": len(component_usage),
                }

        return breakdown

    def get_breakdown_by_model(self) -> dict[str, dict[str, Any]]:
        """Get cost breakdown by model."""
        breakdown: dict[str, dict[str, Any]] = {}

        for model, tokens in self._model_tokens.items():
            costs = MODEL_COSTS.get(model, MODEL_COSTS.get("sonnet", {}))
            input_cost = (tokens["input"] / 1_000_000) * costs.get("input", 3.0)
            output_cost = (tokens["output"] / 1_000_000) * costs.get("output", 15.0)

            breakdown[model] = {
                "input_tokens": tokens["input"],
                "output_tokens": tokens["output"],
                "total_tokens": tokens["input"] + tokens["output"],
                "cost": input_cost + output_cost,
            }

        return breakdown

    def get_summary(self) -> dict[str, Any]:
        """Get complete cost summary."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "remaining_tokens": self.remaining_tokens,
            "remaining_budget": self.remaining_budget,
            "budget_tokens": self.budget_tokens,
            "budget_dollars": self.budget_dollars,
            "latency": {
                "total_ms": self.total_latency_ms,
                "average_ms": self.average_latency_ms,
                "throughput_tps": self.average_tokens_per_second,
            },
            "api_calls": len(self._usage),
            "by_component": self.get_breakdown_by_component(),
            "by_model": self.get_breakdown_by_model(),
            "alerts": [
                {
                    "name": a.threshold_name,
                    "message": a.message,
                    "severity": a.severity,
                }
                for a in self._alerts
            ],
        }

    def format_report(self) -> str:
        """
        Format a human-readable cost report.

        Implements: Spec §8.1 Display cost report in trajectory stream
        """
        summary = self.get_summary()
        lines = [
            "╭─ Cost Report ───────────────────────────────────────╮",
            f"│ Tokens: {summary['total_tokens']:,} / {summary['budget_tokens']:,} "
            f"({summary['total_tokens'] / summary['budget_tokens'] * 100:.1f}%)",
            f"│ Cost: ${summary['total_cost']:.4f} / ${summary['budget_dollars']:.2f}",
            f"│ Latency: {summary['latency']['total_ms']:.0f}ms "
            f"(avg {summary['latency']['average_ms']:.0f}ms/call)",
            f"│ Throughput: {summary['latency']['throughput_tps']:.1f} tokens/sec",
            f"│ API Calls: {summary['api_calls']}",
        ]

        # Add model breakdown
        if summary["by_model"]:
            lines.append("├─ By Model ──────────────────────────────────────────┤")
            for model, data in summary["by_model"].items():
                model_name = model.split("-")[1] if "-" in model else model
                lines.append(
                    f"│   {model_name}: {data['total_tokens']:,} tokens, ${data['cost']:.4f}"
                )

        # Add alerts
        if summary["alerts"]:
            lines.append("├─ Alerts ────────────────────────────────────────────┤")
            for alert in summary["alerts"]:
                icon = "⚠️" if alert["severity"] == "warning" else "🛑"
                lines.append(f"│ {icon} {alert['message']}")

        lines.append("╰─────────────────────────────────────────────────────╯")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all tracking."""
        self._usage.clear()
        self._alerts.clear()
        self._component_tokens = dict.fromkeys(CostComponent, 0)
        self._model_tokens.clear()


# Token estimation utilities


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses simple heuristic of ~4 characters per token.
    For more accurate counts, use a proper tokenizer.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def estimate_context_tokens(
    messages: list[dict[str, str]],
    files: dict[str, str],
    tool_outputs: list[dict[str, Any]],
) -> int:
    """
    Estimate tokens for full context.

    Args:
        messages: Conversation messages
        files: Cached files
        tool_outputs: Tool outputs

    Returns:
        Estimated total tokens
    """
    total = 0

    # Messages
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 10  # Role and formatting overhead

    # Files
    for content in files.values():
        total += estimate_tokens(content)
        total += 20  # Path and formatting overhead

    # Tool outputs
    for output in tool_outputs:
        total += estimate_tokens(str(output.get("content", "")))
        total += 15  # Tool name and formatting overhead

    return total


# Global tracker instance
_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


__all__ = [
    "BudgetAlert",
    "CostComponent",
    "CostEstimate",
    "CostTracker",
    "TokenUsage",
    "estimate_context_tokens",
    "estimate_tokens",
    "get_cost_tracker",
]
