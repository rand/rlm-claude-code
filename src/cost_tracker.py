"""
Cost tracking for RLM-Claude-Code.

Implements: Spec §8.1 Phase 3 - Cost Tracking
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

# tiktoken for accurate token counting (Claude uses cl100k_base encoding)
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False


@lru_cache(maxsize=1)
def _get_encoding() -> Any:
    """Get cached tiktoken encoding (cl100k_base, same as Claude/GPT-4)."""
    if not _TIKTOKEN_AVAILABLE:
        return None
    return tiktoken.get_encoding("cl100k_base")


class CostComponent(Enum):
    """Components that incur token costs."""

    ROOT_PROMPT = "root_prompt"
    RECURSIVE_CALL = "recursive_call"
    REPL_EXECUTION = "repl_execution"
    CONTEXT_LOAD = "context_load"
    SUMMARIZATION = "summarization"
    TOOL_OUTPUT = "tool_output"
    VERIFICATION = "verification"  # SPEC-16.27: Epistemic verification calls


# Token costs per model (per 1M tokens, Jan 2026 pricing)
# Input / Output costs in dollars
MODEL_COSTS: dict[str, dict[str, float]] = {
    # Claude 4.5 Opus (Jan 2026)
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0},
    # Claude 4 Sonnet (Jan 2026)
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    # Claude 3.5 Haiku (Jan 2026) - Note: Haiku 4.5 is more expensive than Haiku 3
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # Claude 3 Haiku (legacy, cheaper)
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Short aliases for convenience
    "opus": {"input": 15.0, "output": 75.0},
    "sonnet": {"input": 3.0, "output": 15.0},
    "haiku": {"input": 1.0, "output": 5.0},  # Default to Haiku 4.5 pricing
    "haiku-3": {"input": 0.25, "output": 1.25},  # Legacy Haiku 3
    # OpenAI models (for multi-provider support)
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o1": {"input": 15.0, "output": 60.0},
    "o3-mini": {"input": 1.10, "output": 4.40},
}

# Default model for cost calculations when model is unknown
DEFAULT_MODEL_FOR_COSTS = "sonnet"


def get_model_costs(model: str) -> dict[str, float]:
    """
    Get cost rates for a model.

    Args:
        model: Model name or alias

    Returns:
        Dict with 'input' and 'output' costs per 1M tokens
    """
    # Try exact match first
    if model in MODEL_COSTS:
        return MODEL_COSTS[model]

    # Try matching by model family (e.g., "claude-sonnet" matches "sonnet")
    model_lower = model.lower()
    for key, costs in MODEL_COSTS.items():
        if key in model_lower or model_lower in key:
            return costs

    # Default to sonnet pricing
    return MODEL_COSTS[DEFAULT_MODEL_FOR_COSTS]


def estimate_call_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """
    Estimate cost for a single API call.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name or alias

    Returns:
        Estimated cost in dollars
    """
    costs = get_model_costs(model)
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost


def compute_affordable_tokens(
    budget_dollars: float,
    model: str,
    input_output_ratio: float = 0.7,
) -> int:
    """
    Compute how many tokens can be afforded with a given budget.

    Args:
        budget_dollars: Available budget in dollars
        model: Model name or alias
        input_output_ratio: Expected ratio of input to total tokens (default 0.7)

    Returns:
        Total tokens (input + output) that can be afforded
    """
    costs = get_model_costs(model)

    # Weighted average cost per token
    input_weight = input_output_ratio
    output_weight = 1 - input_output_ratio
    avg_cost_per_token = (
        input_weight * costs["input"] + output_weight * costs["output"]
    ) / 1_000_000

    if avg_cost_per_token <= 0:
        return 1_000_000  # Fallback to 1M tokens if cost is zero

    return int(budget_dollars / avg_cost_per_token)


def compute_affordable_depth(
    budget_dollars: float,
    model: str,
    avg_tokens_per_call: int = 5000,
    avg_output_tokens: int = 1500,
) -> int:
    """
    Compute maximum affordable recursion depth.

    Args:
        budget_dollars: Available budget in dollars
        model: Model name or alias
        avg_tokens_per_call: Average input tokens per call
        avg_output_tokens: Average output tokens per call

    Returns:
        Maximum affordable depth (capped at 3)
    """
    cost_per_call = estimate_call_cost(avg_tokens_per_call, avg_output_tokens, model)
    if cost_per_call <= 0:
        return 3  # Max depth if cost is zero

    affordable_calls = int(budget_dollars / cost_per_call)
    # Assume ~3 calls per depth level on average
    affordable_depth = affordable_calls // 3

    return min(affordable_depth, 3)  # Cap at max depth 3


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
        return estimate_call_cost(self.input_tokens, self.output_tokens, self.model)


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
        return estimate_call_cost(
            self.estimated_input_tokens, self.estimated_output_tokens, self.model
        )


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
        prompt_text: str | None = None,
    ) -> CostEstimate:
        """
        Estimate cost before execution.

        Implements: Spec §8.1 Cost estimation before execution

        Args:
            prompt_length: Length of prompt in characters (used if prompt_text not provided)
            expected_output_length: Expected output length in characters
            model: Model to use
            component: Component making the call
            include_overhead: Include system prompt overhead
            prompt_text: Actual prompt text for accurate tiktoken counting

        Returns:
            CostEstimate with token and dollar estimates
        """
        # Use tiktoken for accurate counting if prompt_text provided
        if prompt_text is not None:
            estimated_input = estimate_tokens_accurate(prompt_text)
        else:
            # Fallback to character-based estimation
            estimated_input = prompt_length // 4

        # Add overhead for system prompts
        if include_overhead:
            overhead = self._get_overhead_tokens(component)
            estimated_input += overhead

        # Output estimation is always approximate (we don't know output yet)
        estimated_output = expected_output_length // 4

        # Confidence based on estimation method and component
        if prompt_text is not None and _TIKTOKEN_AVAILABLE:
            # Accurate input counting with tiktoken
            base_confidence = 0.85
        else:
            # Character-based estimation
            base_confidence = 0.6

        # Adjust for component predictability
        if component == CostComponent.RECURSIVE_CALL:
            confidence = base_confidence * 0.7  # Less predictable
        elif component == CostComponent.SUMMARIZATION:
            confidence = min(0.95, base_confidence * 1.1)  # More predictable
        else:
            confidence = base_confidence

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
            CostComponent.VERIFICATION: 200,  # SPEC-16.27: Verification prompt overhead
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
            return (
                True,
                f"Would exceed cost budget (${new_total_cost:.2f} > ${self.budget_dollars:.2f})",
            )

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
            cost = estimate_call_cost(tokens["input"], tokens["output"], model)
            breakdown[model] = {
                "input_tokens": tokens["input"],
                "output_tokens": tokens["output"],
                "total_tokens": tokens["input"] + tokens["output"],
                "cost": cost,
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


def estimate_tokens_accurate(text: str) -> int:
    """
    Accurately count tokens using tiktoken (cl100k_base encoding).

    Falls back to ~4 chars/token heuristic if tiktoken unavailable.

    Args:
        text: Text to tokenize

    Returns:
        Token count (accurate if tiktoken available, estimated otherwise)
    """
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    # Fallback: ~4 characters per token (conservative estimate)
    return len(text) // 4


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses tiktoken if available, otherwise ~4 characters per token.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return estimate_tokens_accurate(text)


def is_tiktoken_available() -> bool:
    """Check if tiktoken is available for accurate token counting."""
    return _TIKTOKEN_AVAILABLE


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


# =============================================================================
# SPEC-14.60-14.65: Session Budget Management
# =============================================================================

# Target tokens per execution mode (SPEC-14.65)
MODE_TOKEN_TARGETS: dict[str, int] = {
    "micro": 2_000,  # SPEC-14.65: Micro <2K tokens
    "balanced": 25_000,  # SPEC-14.65: Balanced <25K tokens
    "thorough": 100_000,  # SPEC-14.65: Thorough <100K tokens
}

# Budget warning thresholds (SPEC-14.63)
BUDGET_WARNING_THRESHOLDS: list[float] = [0.50, 0.75, 0.90]


@dataclass
class BudgetWarning:
    """
    Budget warning at utilization threshold.

    Implements: SPEC-14.63
    """

    threshold_percent: int
    tokens_used: int
    tokens_budget: int
    message: str


class SessionBudget:
    """
    Token-based session budget manager.

    Implements: SPEC-14.60-14.65

    Tracks cumulative session cost in tokens (not dollars), provides
    warnings at configurable thresholds, and prevents escalation if
    it would exceed budget.
    """

    def __init__(
        self,
        budget_tokens: int = 500_000,  # SPEC-14.62: Default 500K tokens
        warning_thresholds: list[float] | None = None,
    ):
        """
        Initialize session budget.

        Args:
            budget_tokens: Maximum tokens for the session (SPEC-14.62)
            warning_thresholds: Warning thresholds as fractions (SPEC-14.63)
        """
        self.budget_tokens = budget_tokens
        self.warning_thresholds = warning_thresholds or BUDGET_WARNING_THRESHOLDS

        # Track tokens by mode (SPEC-14.65)
        self._mode_tokens: dict[str, int] = {
            "micro": 0,
            "balanced": 0,
            "thorough": 0,
        }
        self._total_tokens = 0
        self._warnings_emitted: set[int] = set()
        self._warning_callbacks: list[Callable[[BudgetWarning], None]] = []

    def record_tokens(self, tokens: int, mode: str = "micro") -> list[BudgetWarning]:
        """
        Record token usage for the session.

        Implements: SPEC-14.60, SPEC-14.61

        Args:
            tokens: Number of tokens used
            mode: Execution mode ("micro", "balanced", "thorough")

        Returns:
            List of warnings triggered by this usage
        """
        self._total_tokens += tokens

        if mode in self._mode_tokens:
            self._mode_tokens[mode] += tokens

        # Check for warnings (SPEC-14.63)
        return self._check_warnings()

    def _check_warnings(self) -> list[BudgetWarning]:
        """Check and emit budget warnings (SPEC-14.63)."""
        warnings: list[BudgetWarning] = []
        utilization = self._total_tokens / self.budget_tokens

        for threshold in self.warning_thresholds:
            threshold_percent = int(threshold * 100)

            if utilization >= threshold and threshold_percent not in self._warnings_emitted:
                self._warnings_emitted.add(threshold_percent)

                warning = BudgetWarning(
                    threshold_percent=threshold_percent,
                    tokens_used=self._total_tokens,
                    tokens_budget=self.budget_tokens,
                    message=f"Session budget at {threshold_percent}%: "
                    f"{self._total_tokens:,} / {self.budget_tokens:,} tokens",
                )
                warnings.append(warning)

                # Notify callbacks
                for callback in self._warning_callbacks:
                    callback(warning)

        return warnings

    def can_escalate(self, target_mode: str) -> tuple[bool, str | None]:
        """
        Check if escalation to target mode is within budget.

        Implements: SPEC-14.64

        Args:
            target_mode: Mode to escalate to ("balanced", "thorough")

        Returns:
            (can_escalate, reason) tuple
        """
        if target_mode not in MODE_TOKEN_TARGETS:
            return False, f"Unknown mode: {target_mode}"

        target_tokens = MODE_TOKEN_TARGETS[target_mode]
        projected_total = self._total_tokens + target_tokens

        if projected_total > self.budget_tokens:
            return False, (
                f"Escalation to {target_mode} mode would exceed budget: "
                f"{projected_total:,} > {self.budget_tokens:,} tokens"
            )

        return True, None

    def get_available_for_mode(self, mode: str) -> int:
        """
        Get remaining tokens available for a mode.

        Args:
            mode: Execution mode

        Returns:
            Available tokens (may be negative if over budget)
        """
        remaining = self.budget_tokens - self._total_tokens
        target = MODE_TOKEN_TARGETS.get(mode, MODE_TOKEN_TARGETS["micro"])
        return min(remaining, target)

    def on_warning(self, callback: Callable[[BudgetWarning], None]) -> None:
        """Register callback for budget warnings."""
        self._warning_callbacks.append(callback)

    @property
    def total_tokens(self) -> int:
        """Total tokens used in session (SPEC-14.60)."""
        return self._total_tokens

    @property
    def remaining_tokens(self) -> int:
        """Remaining tokens in budget."""
        return max(0, self.budget_tokens - self._total_tokens)

    @property
    def utilization(self) -> float:
        """Budget utilization as fraction."""
        return self._total_tokens / self.budget_tokens

    @property
    def utilization_percent(self) -> int:
        """Budget utilization as percentage."""
        return int(self.utilization * 100)

    def get_mode_breakdown(self) -> dict[str, dict[str, Any]]:
        """
        Get token breakdown by execution mode.

        Implements: SPEC-14.65
        """
        return {
            mode: {
                "tokens_used": tokens,
                "target_tokens": MODE_TOKEN_TARGETS.get(mode, 0),
                "over_target": tokens > MODE_TOKEN_TARGETS.get(mode, float("inf")),
            }
            for mode, tokens in self._mode_tokens.items()
        }

    def get_summary(self) -> dict[str, Any]:
        """Get complete budget summary."""
        return {
            "total_tokens": self._total_tokens,
            "budget_tokens": self.budget_tokens,
            "remaining_tokens": self.remaining_tokens,
            "utilization_percent": self.utilization_percent,
            "warnings_emitted": sorted(self._warnings_emitted),
            "by_mode": self.get_mode_breakdown(),
        }

    def reset(self) -> None:
        """Reset session budget tracking."""
        self._total_tokens = 0
        self._mode_tokens = dict.fromkeys(self._mode_tokens, 0)
        self._warnings_emitted.clear()


def create_session_budget(
    budget_tokens: int | None = None,
) -> SessionBudget:
    """
    Create a session budget tracker.

    Implements: SPEC-14.62

    Args:
        budget_tokens: Custom budget (default 500K tokens)

    Returns:
        Configured SessionBudget instance
    """
    return SessionBudget(budget_tokens=budget_tokens or 500_000)


__all__ = [
    "BudgetAlert",
    "BudgetWarning",
    "CostComponent",
    "CostEstimate",
    "CostTracker",
    "DEFAULT_MODEL_FOR_COSTS",
    "MODEL_COSTS",
    "MODE_TOKEN_TARGETS",
    "BUDGET_WARNING_THRESHOLDS",
    "SessionBudget",
    "TokenUsage",
    "compute_affordable_depth",
    "compute_affordable_tokens",
    "create_session_budget",
    "estimate_call_cost",
    "estimate_context_tokens",
    "estimate_tokens",
    "estimate_tokens_accurate",
    "get_cost_tracker",
    "get_model_costs",
    "is_tiktoken_available",
]
