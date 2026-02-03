"""
Enhanced budget tracking for RLM-Claude-Code.

Implements: Spec SPEC-05 - Enhanced Budget Tracking
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .cost_tracker import CostComponent, estimate_call_cost
from .trajectory import TrajectoryEvent, TrajectoryEventType

if TYPE_CHECKING:
    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EnhancedBudgetMetrics:
    """
    Enhanced budget metrics.

    Implements: Spec SPEC-05.01-05
    """

    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    # Cost
    total_cost_usd: float = 0.0

    # Execution metrics
    recursion_depth: int = 0
    max_depth_reached: int = 0
    sub_call_count: int = 0
    repl_executions: int = 0

    # Time
    session_start: float = 0.0
    session_duration_seconds: float = 0.0
    wall_clock_seconds: float = 0.0


@dataclass
class BudgetLimits:
    """
    Budget limits configuration.

    Implements: Spec SPEC-05.06-10
    """

    max_cost_per_task: float = 5.0
    max_cost_per_session: float = 25.0
    max_tokens_per_call: int = 8000
    max_recursive_calls: int = 10
    max_repl_executions: int = 50
    cost_alert_threshold: float = 0.8
    token_alert_threshold: float = 0.75


@dataclass
class BudgetAlert:
    """
    Budget alert.

    Implements: Spec SPEC-05.14
    """

    level: str  # "warning" | "critical"
    message: str
    metric: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Burn Rate Data Classes (Feature 3e0.5)
# =============================================================================


@dataclass
class CostSample:
    """Single cost measurement for burn rate calculation."""

    timestamp: float
    cumulative_cost: float
    model: str
    call_type: str  # "root", "recursive", "repl"


@dataclass
class BurnRateMetrics:
    """Real-time burn rate tracking."""

    dollars_per_minute: float
    tokens_per_minute: float
    time_to_exhaustion_seconds: float | None  # None if not burning
    is_burning_fast: bool  # True if will exhaust budget in < 5 minutes
    measurement_window_seconds: float
    sample_count: int


@dataclass
class BurnRateAlert:
    """Alert for burn rate issues."""

    level: str  # "warning", "critical"
    message: str
    current_rate: float  # $/minute
    suggested_model: str | None
    time_to_exhaustion: float | None
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Adaptive Depth Data Classes (Feature 3e0.4)
# =============================================================================


@dataclass
class AdaptiveDepthRecommendation:
    """Depth recommendation based on budget state."""

    recommended_depth: int  # 0-3
    recommended_model: str
    original_model: str
    was_downgraded: bool
    downgrade_reason: str | None
    estimated_cost_per_depth: float
    budget_utilization: float  # 0.0-1.0 (fraction of budget used)


@dataclass
class DepthBudgetWarning:
    """Warning before expensive operation."""

    operation: str  # "recursive_call", "depth_increase"
    estimated_cost: float
    remaining_budget: float
    warning_level: str  # "info", "warning", "critical"
    message: str
    suggested_action: str  # "proceed", "downgrade_model", "reduce_depth", "abort"


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_LIMITS = BudgetLimits()


# =============================================================================
# EnhancedBudgetTracker Class
# =============================================================================


class EnhancedBudgetTracker:
    """
    Enhanced budget tracking with granular metrics, limits, and alerts.

    Implements: Spec SPEC-05

    Features:
    - Track cached_tokens, sub_call_count, repl_executions
    - Track wall_clock_seconds, max_depth_reached
    - Granular per-task and per-session limits
    - Alert system with warning/critical levels
    - Limit enforcement with force bypass
    - Configuration from file with per-mode overrides
    """

    def __init__(
        self,
        config_path: str | None = None,
        mode: str = "balanced",
    ):
        """
        Initialize enhanced budget tracker.

        Args:
            config_path: Optional path to config file
            mode: Operating mode (fast, balanced, thorough)
        """
        self.mode = mode
        self._limits = self._load_config(config_path, mode)
        self._metrics = EnhancedBudgetMetrics()
        self._alerts: list[BudgetAlert] = []
        self._emitted_alerts: set[str] = set()  # Prevent duplicate alerts
        self._alert_callbacks: list[Callable[[BudgetAlert], None]] = []
        self._alert_event_callbacks: list[Callable[[TrajectoryEvent], None]] = []

        # Task tracking
        self._current_task_id: str | None = None
        self._task_cost: float = 0.0

        # Timing
        self._timing_start: float | None = None

        # Burn rate tracking (Feature 3e0.5)
        self._cost_samples: list[CostSample] = []
        self._burn_rate_window_seconds: float = 60.0  # 1 minute sliding window
        self._burn_rate_callbacks: list[Callable[[BurnRateAlert], None]] = []
        self._max_cost_samples: int = 1000  # Prevent unbounded growth
        self._token_samples: list[tuple[float, int]] = []  # (timestamp, tokens)

    def _load_config(
        self,
        config_path: str | None,
        mode: str,
    ) -> BudgetLimits:
        """
        Load configuration from file or use defaults.

        Implements: Spec SPEC-05.20-21
        """
        limits = BudgetLimits()

        # Try loading from specified path
        paths_to_try = []
        if config_path:
            paths_to_try.append(Path(config_path))

        # Try default location
        default_path = Path.home() / ".claude" / "rlm-config.json"
        paths_to_try.append(default_path)

        config_data: dict[str, Any] = {}
        for path in paths_to_try:
            if path.exists():
                try:
                    with open(path) as f:
                        config_data = json.load(f)
                        break
                except (json.JSONDecodeError, OSError):
                    pass

        if "budget" not in config_data:
            return limits

        budget_config = config_data["budget"]

        # Apply base configuration
        if "max_cost_per_task" in budget_config:
            limits.max_cost_per_task = budget_config["max_cost_per_task"]
        if "max_cost_per_session" in budget_config:
            limits.max_cost_per_session = budget_config["max_cost_per_session"]
        if "max_tokens_per_call" in budget_config:
            limits.max_tokens_per_call = budget_config["max_tokens_per_call"]
        if "max_recursive_calls" in budget_config:
            limits.max_recursive_calls = budget_config["max_recursive_calls"]
        if "max_repl_executions" in budget_config:
            limits.max_repl_executions = budget_config["max_repl_executions"]
        if "cost_alert_threshold" in budget_config:
            limits.cost_alert_threshold = budget_config["cost_alert_threshold"]
        if "token_alert_threshold" in budget_config:
            limits.token_alert_threshold = budget_config["token_alert_threshold"]

        # Apply per-mode overrides (SPEC-05.21)
        if "modes" in budget_config and mode in budget_config["modes"]:
            mode_config = budget_config["modes"][mode]
            if "max_cost_per_task" in mode_config:
                limits.max_cost_per_task = mode_config["max_cost_per_task"]
            if "max_cost_per_session" in mode_config:
                limits.max_cost_per_session = mode_config["max_cost_per_session"]
            if "max_tokens_per_call" in mode_config:
                limits.max_tokens_per_call = mode_config["max_tokens_per_call"]
            if "max_recursive_calls" in mode_config:
                limits.max_recursive_calls = mode_config["max_recursive_calls"]
            if "max_repl_executions" in mode_config:
                limits.max_repl_executions = mode_config["max_repl_executions"]
            if "cost_alert_threshold" in mode_config:
                limits.cost_alert_threshold = mode_config["cost_alert_threshold"]
            if "token_alert_threshold" in mode_config:
                limits.token_alert_threshold = mode_config["token_alert_threshold"]

        return limits

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_limits(self, limits: BudgetLimits) -> None:
        """Set budget limits."""
        self._limits = limits

    def get_limits(self) -> BudgetLimits:
        """Get current budget limits."""
        return self._limits

    # =========================================================================
    # Recording Methods
    # =========================================================================

    def record_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        component: CostComponent,
        cached_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> list[BudgetAlert]:
        """
        Record LLM call and return any triggered alerts.

        Implements: Spec SPEC-05.01-02

        Args:
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens generated
            model: Model identifier (e.g., "claude-sonnet-4-20250514")
            component: Cost attribution component. Valid values:
                - CostComponent.ROOT_PROMPT: Main conversation turn
                - CostComponent.RECURSIVE_CALL: Sub-query/recursive LLM call
                - CostComponent.SUMMARIZATION: Context summarization
                - CostComponent.CONTEXT_LOAD: Loading context from memory
                - CostComponent.TOOL_OUTPUT: Processing tool outputs
            cached_tokens: Number of tokens served from cache (reduces cost)
            latency_ms: Call latency in milliseconds (for metrics)

        Returns:
            List of BudgetAlert objects if any limits exceeded

        Example:
            >>> from src.cost_tracker import CostComponent
            >>> tracker = EnhancedBudgetTracker()
            >>> alerts = tracker.record_llm_call(
            ...     input_tokens=1000,
            ...     output_tokens=500,
            ...     model="claude-sonnet-4-20250514",
            ...     component=CostComponent.ROOT_PROMPT,
            ... )
        """
        # Update token metrics
        self._metrics.input_tokens += input_tokens
        self._metrics.output_tokens += output_tokens
        self._metrics.cached_tokens += cached_tokens

        # Calculate cost using model-aware pricing
        call_cost = estimate_call_cost(input_tokens, output_tokens, model)

        self._metrics.total_cost_usd += call_cost
        self._task_cost += call_cost

        # Track sub-calls (SPEC-05.02)
        if component == CostComponent.RECURSIVE_CALL:
            self._metrics.sub_call_count += 1

        # Record burn rate sample (Feature 3e0.5)
        call_type = "recursive" if component == CostComponent.RECURSIVE_CALL else "root"
        self.record_cost_sample(
            cost=call_cost,
            tokens=input_tokens + output_tokens,
            model=model,
            call_type=call_type,
        )

        # Check burn rate and notify callbacks
        burn_alert = self.check_burn_rate()
        if burn_alert:
            for callback in self._burn_rate_callbacks:
                callback(burn_alert)

        # Check for alerts
        return self._check_and_emit_alerts()

    def record_repl_execution(self) -> list[BudgetAlert]:
        """
        Record REPL execution and return any triggered alerts.

        Implements: Spec SPEC-05.03
        """
        self._metrics.repl_executions += 1
        return self._check_and_emit_alerts()

    def record_depth(self, depth: int) -> None:
        """
        Record current recursion depth.

        Implements: Spec SPEC-05.05
        """
        self._metrics.recursion_depth = depth
        if depth > self._metrics.max_depth_reached:
            self._metrics.max_depth_reached = depth

    # =========================================================================
    # Timing Methods (SPEC-05.04)
    # =========================================================================

    def start_timing(self) -> None:
        """Start wall clock timing."""
        self._timing_start = time.time()
        if self._metrics.session_start == 0.0:
            self._metrics.session_start = self._timing_start

    def stop_timing(self) -> None:
        """Stop wall clock timing and update metrics."""
        if self._timing_start is not None:
            elapsed = time.time() - self._timing_start
            self._metrics.wall_clock_seconds += elapsed
            self._metrics.session_duration_seconds = time.time() - self._metrics.session_start
            self._timing_start = None

    # =========================================================================
    # Task Management (SPEC-05.25)
    # =========================================================================

    def start_task(self, task_id: str) -> None:
        """Start a new task."""
        self._current_task_id = task_id
        self._task_cost = 0.0

    def end_task(self) -> None:
        """End current task."""
        self._current_task_id = None

    # =========================================================================
    # Limit Checks (SPEC-05.16-19)
    # =========================================================================

    def can_make_llm_call(self, force: bool = False) -> tuple[bool, str | None]:
        """
        Check if LLM call is allowed.

        Implements: Spec SPEC-05.16, SPEC-05.19

        Args:
            force: Bypass limits for debugging

        Returns:
            (allowed, reason) tuple
        """
        if force:
            return True, None

        # Check task cost limit
        if self._task_cost >= self._limits.max_cost_per_task:
            return False, (
                f"Task cost limit exceeded: ${self._task_cost:.2f} >= "
                f"${self._limits.max_cost_per_task:.2f}"
            )

        # Check session cost limit
        if self._metrics.total_cost_usd >= self._limits.max_cost_per_session:
            return False, (
                f"Session cost limit exceeded: ${self._metrics.total_cost_usd:.2f} >= "
                f"${self._limits.max_cost_per_session:.2f}"
            )

        return True, None

    def can_recurse(self, force: bool = False) -> tuple[bool, str | None]:
        """
        Check if recursion is allowed.

        Implements: Spec SPEC-05.17, SPEC-05.19

        Args:
            force: Bypass limits for debugging

        Returns:
            (allowed, reason) tuple
        """
        if force:
            return True, None

        if self._metrics.sub_call_count >= self._limits.max_recursive_calls:
            return False, (
                f"Recursive call limit reached: {self._metrics.sub_call_count} >= "
                f"{self._limits.max_recursive_calls}"
            )

        return True, None

    def can_execute_repl(self, force: bool = False) -> tuple[bool, str | None]:
        """
        Check if REPL execution is allowed.

        Implements: Spec SPEC-05.18, SPEC-05.19

        Args:
            force: Bypass limits for debugging

        Returns:
            (allowed, reason) tuple
        """
        if force:
            return True, None

        if self._metrics.repl_executions >= self._limits.max_repl_executions:
            return False, (
                f"REPL execution limit reached: {self._metrics.repl_executions} >= "
                f"{self._limits.max_repl_executions}"
            )

        return True, None

    # =========================================================================
    # Alert System (SPEC-05.11-15)
    # =========================================================================

    def check_limits(self) -> list[BudgetAlert]:
        """
        Check all limits and return current alerts.

        Implements: Spec SPEC-05.11-13
        """
        return self._check_and_emit_alerts()

    def _check_and_emit_alerts(self) -> list[BudgetAlert]:
        """Check all limits and emit any new alerts."""
        new_alerts: list[BudgetAlert] = []

        # Cost alert (SPEC-05.11)
        cost_fraction = self._task_cost / self._limits.max_cost_per_task
        if cost_fraction >= 1.0:
            alert = self._create_alert(
                level="critical",
                metric="cost",
                current_value=self._task_cost,
                threshold=self._limits.max_cost_per_task,
                message=(
                    f"Task cost limit exceeded: ${self._task_cost:.2f} / "
                    f"${self._limits.max_cost_per_task:.2f}"
                ),
            )
            if alert:
                new_alerts.append(alert)
        elif cost_fraction >= self._limits.cost_alert_threshold:
            alert = self._create_alert(
                level="warning",
                metric="cost",
                current_value=self._task_cost,
                threshold=self._limits.max_cost_per_task * self._limits.cost_alert_threshold,
                message=(
                    f"Approaching task cost limit: ${self._task_cost:.2f} / "
                    f"${self._limits.max_cost_per_task:.2f} ({cost_fraction:.0%})"
                ),
            )
            if alert:
                new_alerts.append(alert)

        # Recursive call warning (SPEC-05.13)
        remaining_calls = self._limits.max_recursive_calls - self._metrics.sub_call_count
        if remaining_calls <= 0:
            alert = self._create_alert(
                level="critical",
                metric="recursive_calls",
                current_value=self._metrics.sub_call_count,
                threshold=self._limits.max_recursive_calls,
                message=(
                    f"Recursive call limit reached: {self._metrics.sub_call_count} / "
                    f"{self._limits.max_recursive_calls}"
                ),
            )
            if alert:
                new_alerts.append(alert)
        elif remaining_calls <= 2:
            alert = self._create_alert(
                level="warning",
                metric="recursive_calls",
                current_value=self._metrics.sub_call_count,
                threshold=self._limits.max_recursive_calls - 2,
                message=(
                    f"Approaching recursive call limit: {self._metrics.sub_call_count} / "
                    f"{self._limits.max_recursive_calls} ({remaining_calls} remaining)"
                ),
            )
            if alert:
                new_alerts.append(alert)

        # REPL execution warning
        remaining_repl = self._limits.max_repl_executions - self._metrics.repl_executions
        if remaining_repl <= 0:
            alert = self._create_alert(
                level="critical",
                metric="repl_executions",
                current_value=self._metrics.repl_executions,
                threshold=self._limits.max_repl_executions,
                message=(
                    f"REPL execution limit reached: {self._metrics.repl_executions} / "
                    f"{self._limits.max_repl_executions}"
                ),
            )
            if alert:
                new_alerts.append(alert)
        elif remaining_repl <= 5:
            alert = self._create_alert(
                level="warning",
                metric="repl_executions",
                current_value=self._metrics.repl_executions,
                threshold=self._limits.max_repl_executions - 5,
                message=(
                    f"Approaching REPL execution limit: {self._metrics.repl_executions} / "
                    f"{self._limits.max_repl_executions} ({remaining_repl} remaining)"
                ),
            )
            if alert:
                new_alerts.append(alert)

        return new_alerts

    def _create_alert(
        self,
        level: str,
        metric: str,
        current_value: float,
        threshold: float,
        message: str,
    ) -> BudgetAlert | None:
        """Create alert if not already emitted."""
        alert_key = f"{level}:{metric}"
        if alert_key in self._emitted_alerts:
            return None

        self._emitted_alerts.add(alert_key)

        alert = BudgetAlert(
            level=level,
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            message=message,
        )

        self._alerts.append(alert)

        # Notify callbacks
        for callback in self._alert_callbacks:
            callback(alert)

        # Emit as TrajectoryEvent (SPEC-05.15)
        event = TrajectoryEvent(
            type=TrajectoryEventType.BUDGET_ALERT,
            depth=0,
            content=message,
            metadata={
                "level": level,
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
            },
        )

        for callback in self._alert_event_callbacks:
            callback(event)

        return alert

    def on_alert(self, callback: Callable[[BudgetAlert], None]) -> None:
        """Register callback for budget alerts."""
        self._alert_callbacks.append(callback)

    def on_alert_event(self, callback: Callable[[TrajectoryEvent], None]) -> None:
        """Register callback for budget alert trajectory events."""
        self._alert_event_callbacks.append(callback)

    # =========================================================================
    # Metrics Access
    # =========================================================================

    def get_metrics(self) -> EnhancedBudgetMetrics:
        """
        Get current budget metrics.

        Returns:
            EnhancedBudgetMetrics with current values
        """
        # Update session duration if timing is active
        if self._timing_start is not None:
            elapsed = time.time() - self._timing_start
            self._metrics.wall_clock_seconds += elapsed
            self._timing_start = time.time()

        if self._metrics.session_start > 0:
            self._metrics.session_duration_seconds = time.time() - self._metrics.session_start

        return EnhancedBudgetMetrics(
            input_tokens=self._metrics.input_tokens,
            output_tokens=self._metrics.output_tokens,
            cached_tokens=self._metrics.cached_tokens,
            total_cost_usd=self._metrics.total_cost_usd,
            recursion_depth=self._metrics.recursion_depth,
            max_depth_reached=self._metrics.max_depth_reached,
            sub_call_count=self._metrics.sub_call_count,
            repl_executions=self._metrics.repl_executions,
            session_start=self._metrics.session_start,
            session_duration_seconds=self._metrics.session_duration_seconds,
            wall_clock_seconds=self._metrics.wall_clock_seconds,
        )

    def get_alerts(self) -> list[BudgetAlert]:
        """Get all alerts that have been emitted."""
        return self._alerts.copy()

    def reset(self) -> None:
        """Reset all tracking."""
        self._metrics = EnhancedBudgetMetrics()
        self._alerts.clear()
        self._emitted_alerts.clear()
        self._current_task_id = None
        self._task_cost = 0.0
        self._timing_start = None
        self._cost_samples.clear()
        self._token_samples.clear()

    # =========================================================================
    # Burn Rate Monitoring (Feature 3e0.5)
    # =========================================================================

    def record_cost_sample(
        self,
        cost: float,
        tokens: int,
        model: str,
        call_type: str,
    ) -> None:
        """
        Record a cost sample for burn rate tracking.

        Maintains sliding window of samples, pruning old entries.

        Args:
            cost: Cost of this call in dollars
            tokens: Total tokens used in this call
            model: Model name
            call_type: Type of call ("root", "recursive", "repl")
        """
        now = time.time()

        # Add new sample
        self._cost_samples.append(
            CostSample(
                timestamp=now,
                cumulative_cost=self._metrics.total_cost_usd,
                model=model,
                call_type=call_type,
            )
        )
        self._token_samples.append((now, tokens))

        # Prune old samples beyond window
        cutoff = now - self._burn_rate_window_seconds * 2  # Keep 2x window for smoothing
        self._cost_samples = [s for s in self._cost_samples if s.timestamp > cutoff]
        self._token_samples = [(t, n) for t, n in self._token_samples if t > cutoff]

        # Enforce max samples limit
        if len(self._cost_samples) > self._max_cost_samples:
            self._cost_samples = self._cost_samples[-self._max_cost_samples :]
        if len(self._token_samples) > self._max_cost_samples:
            self._token_samples = self._token_samples[-self._max_cost_samples :]

    def get_burn_rate(
        self,
        window_seconds: float | None = None,
    ) -> BurnRateMetrics:
        """
        Calculate current burn rate over specified window.

        Args:
            window_seconds: Measurement window (default: 60s)

        Returns:
            BurnRateMetrics with current velocity and projections
        """
        window = window_seconds or self._burn_rate_window_seconds
        now = time.time()
        cutoff = now - window

        # Filter samples within window
        window_samples = [s for s in self._cost_samples if s.timestamp > cutoff]
        window_tokens = [(t, n) for t, n in self._token_samples if t > cutoff]

        if len(window_samples) < 2:
            # Not enough data for rate calculation
            return BurnRateMetrics(
                dollars_per_minute=0.0,
                tokens_per_minute=0.0,
                time_to_exhaustion_seconds=None,
                is_burning_fast=False,
                measurement_window_seconds=window,
                sample_count=len(window_samples),
            )

        # Calculate cost rate ($/minute)
        first_sample = window_samples[0]
        last_sample = window_samples[-1]
        time_span = last_sample.timestamp - first_sample.timestamp

        if time_span <= 0:
            return BurnRateMetrics(
                dollars_per_minute=0.0,
                tokens_per_minute=0.0,
                time_to_exhaustion_seconds=None,
                is_burning_fast=False,
                measurement_window_seconds=window,
                sample_count=len(window_samples),
            )

        cost_delta = last_sample.cumulative_cost - first_sample.cumulative_cost
        dollars_per_minute = (cost_delta / time_span) * 60.0

        # Calculate token rate
        total_tokens_in_window = sum(n for _, n in window_tokens)
        tokens_per_minute = (total_tokens_in_window / time_span) * 60.0

        # Calculate time to exhaustion
        remaining_budget = self._limits.max_cost_per_task - self._task_cost
        time_to_exhaustion: float | None = None
        if dollars_per_minute > 0:
            time_to_exhaustion = (remaining_budget / dollars_per_minute) * 60.0

        # Determine if burning fast (< 5 minutes to exhaustion)
        is_burning_fast = time_to_exhaustion is not None and time_to_exhaustion < 300.0

        return BurnRateMetrics(
            dollars_per_minute=dollars_per_minute,
            tokens_per_minute=tokens_per_minute,
            time_to_exhaustion_seconds=time_to_exhaustion,
            is_burning_fast=is_burning_fast,
            measurement_window_seconds=window,
            sample_count=len(window_samples),
        )

    def check_burn_rate(self) -> BurnRateAlert | None:
        """
        Check if burn rate warrants an alert.

        Alert thresholds:
        - Warning: Will exhaust budget in < 5 minutes at current rate
        - Critical: Will exhaust budget in < 2 minutes at current rate

        Returns:
            BurnRateAlert if threshold exceeded, None otherwise
        """
        metrics = self.get_burn_rate()

        if metrics.time_to_exhaustion_seconds is None:
            return None

        # Critical: < 2 minutes to exhaustion
        if metrics.time_to_exhaustion_seconds < 120.0:
            suggested = self.suggest_model_for_burn_rate("opus", metrics)
            return BurnRateAlert(
                level="critical",
                message=(
                    f"Budget will be exhausted in {metrics.time_to_exhaustion_seconds:.0f}s "
                    f"at current rate of ${metrics.dollars_per_minute:.3f}/min"
                ),
                current_rate=metrics.dollars_per_minute,
                suggested_model=suggested[0] if suggested[0] != "opus" else None,
                time_to_exhaustion=metrics.time_to_exhaustion_seconds,
            )

        # Warning: < 5 minutes to exhaustion
        if metrics.time_to_exhaustion_seconds < 300.0:
            suggested = self.suggest_model_for_burn_rate("sonnet", metrics)
            return BurnRateAlert(
                level="warning",
                message=(
                    f"High burn rate: ${metrics.dollars_per_minute:.3f}/min. "
                    f"Budget exhaustion in {metrics.time_to_exhaustion_seconds:.0f}s"
                ),
                current_rate=metrics.dollars_per_minute,
                suggested_model=suggested[0] if suggested[0] != "sonnet" else None,
                time_to_exhaustion=metrics.time_to_exhaustion_seconds,
            )

        return None

    def suggest_model_for_burn_rate(
        self,
        current_model: str,
        metrics: BurnRateMetrics,
    ) -> tuple[str, str]:
        """
        Suggest model downgrade based on burn rate.

        Args:
            current_model: Currently active model
            metrics: Current burn rate metrics

        Returns:
            (suggested_model, reason) tuple
        """
        # Model cost ratios (relative to sonnet)
        # opus: ~5x, sonnet: 1x, haiku: ~0.3x
        model_lower = current_model.lower()

        if metrics.time_to_exhaustion_seconds is None:
            return current_model, "no_rate_data"

        # If burning very fast (< 2 min), suggest haiku
        if metrics.time_to_exhaustion_seconds < 120.0:
            if "haiku" not in model_lower:
                return "haiku", "critical_burn_rate"

        # If burning fast (< 5 min), suggest downgrade by one tier
        if metrics.time_to_exhaustion_seconds < 300.0:
            if "opus" in model_lower:
                return "sonnet", "high_burn_rate"
            if "sonnet" in model_lower:
                return "haiku", "high_burn_rate"

        return current_model, "acceptable_burn_rate"

    def on_burn_rate_alert(
        self,
        callback: Callable[[BurnRateAlert], None],
    ) -> None:
        """Register callback for burn rate alerts."""
        self._burn_rate_callbacks.append(callback)

    # =========================================================================
    # Adaptive Depth Budgeting (Feature 3e0.4)
    # =========================================================================

    def compute_adaptive_depth(
        self,
        planned_depth: int,
        planned_model: str,
        avg_tokens_per_call: int = 5000,
        avg_output_tokens: int = 1500,
    ) -> AdaptiveDepthRecommendation:
        """
        Compute affordable depth and model based on remaining budget.

        Implements: Feature 3e0.4 - Adaptive Depth Budgeting

        Rules:
        - budget >= 50%: Keep planned model
        - 20% <= budget < 50%: Downgrade one tier (opus->sonnet, sonnet->haiku)
        - budget < 20%: Force haiku

        Args:
            planned_depth: Originally planned depth (0-3)
            planned_model: Originally planned model
            avg_tokens_per_call: Expected input tokens per call
            avg_output_tokens: Expected output tokens per call

        Returns:
            AdaptiveDepthRecommendation with adjusted depth and model
        """
        remaining_budget = self._limits.max_cost_per_task - self._task_cost
        budget_fraction = remaining_budget / self._limits.max_cost_per_task

        # Get recommended model based on budget fraction
        recommended_model = self.get_recommended_model_for_budget(budget_fraction)
        was_downgraded = recommended_model.lower() != planned_model.lower()

        # Determine downgrade reason
        downgrade_reason: str | None = None
        if was_downgraded:
            if budget_fraction < 0.2:
                downgrade_reason = "budget_below_20_percent"
            elif budget_fraction < 0.5:
                downgrade_reason = "budget_below_50_percent"

        # Estimate cost per depth level with recommended model
        estimated_cost_per_depth = estimate_call_cost(
            avg_tokens_per_call, avg_output_tokens, recommended_model
        )

        # Calculate max affordable depth
        if estimated_cost_per_depth > 0:
            max_affordable_depth = int(remaining_budget / estimated_cost_per_depth)
            recommended_depth = min(planned_depth, max(0, max_affordable_depth))
        else:
            recommended_depth = planned_depth

        return AdaptiveDepthRecommendation(
            recommended_depth=recommended_depth,
            recommended_model=recommended_model,
            original_model=planned_model,
            was_downgraded=was_downgraded,
            downgrade_reason=downgrade_reason,
            estimated_cost_per_depth=estimated_cost_per_depth,
            budget_utilization=1.0 - budget_fraction,
        )

    def get_recommended_model_for_budget(
        self,
        budget_fraction: float,
    ) -> str:
        """
        Recommend model based on budget state.

        Rules:
        - budget >= 50%: Return "opus" (no downgrade)
        - 20% <= budget < 50%: Return "sonnet"
        - budget < 20%: Return "haiku"

        Args:
            budget_fraction: Fraction of original budget remaining (0.0-1.0)

        Returns:
            Model name string
        """
        if budget_fraction >= 0.5:
            return "opus"
        elif budget_fraction >= 0.2:
            return "sonnet"
        else:
            return "haiku"

    def warn_before_expensive_operation(
        self,
        operation: str,
        estimated_cost: float,
        model: str,
    ) -> DepthBudgetWarning | None:
        """
        Generate warning if operation would consume significant budget.

        Generates warning when:
        - Operation cost > 25% of remaining budget (warning)
        - Operation cost > 50% of remaining budget (critical)
        - Would exceed budget limit (critical)

        Args:
            operation: Type of operation ("recursive_call", "depth_increase")
            estimated_cost: Estimated cost in dollars
            model: Model being used

        Returns:
            DepthBudgetWarning if warning needed, None otherwise
        """
        remaining_budget = self._limits.max_cost_per_task - self._task_cost

        if remaining_budget <= 0:
            return DepthBudgetWarning(
                operation=operation,
                estimated_cost=estimated_cost,
                remaining_budget=remaining_budget,
                warning_level="critical",
                message="Budget exhausted. Cannot proceed with operation.",
                suggested_action="abort",
            )

        cost_fraction = estimated_cost / remaining_budget

        # Would exceed budget
        if estimated_cost > remaining_budget:
            return DepthBudgetWarning(
                operation=operation,
                estimated_cost=estimated_cost,
                remaining_budget=remaining_budget,
                warning_level="critical",
                message=(
                    f"Operation would exceed budget: ${estimated_cost:.4f} > "
                    f"${remaining_budget:.4f} remaining"
                ),
                suggested_action="abort",
            )

        # > 50% of remaining budget - critical
        if cost_fraction > 0.5:
            return DepthBudgetWarning(
                operation=operation,
                estimated_cost=estimated_cost,
                remaining_budget=remaining_budget,
                warning_level="critical",
                message=(
                    f"Operation will use {cost_fraction:.0%} of remaining budget "
                    f"(${estimated_cost:.4f} of ${remaining_budget:.4f})"
                ),
                suggested_action="downgrade_model",
            )

        # > 25% of remaining budget - warning
        if cost_fraction > 0.25:
            return DepthBudgetWarning(
                operation=operation,
                estimated_cost=estimated_cost,
                remaining_budget=remaining_budget,
                warning_level="warning",
                message=(
                    f"Operation will use {cost_fraction:.0%} of remaining budget "
                    f"(${estimated_cost:.4f} of ${remaining_budget:.4f})"
                ),
                suggested_action="proceed",
            )

        return None


__all__ = [
    "AdaptiveDepthRecommendation",
    "BudgetAlert",
    "BudgetLimits",
    "BurnRateAlert",
    "BurnRateMetrics",
    "CostSample",
    "DepthBudgetWarning",
    "EnhancedBudgetMetrics",
    "EnhancedBudgetTracker",
]
