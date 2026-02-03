"""
Auto-activation for RLM mode on complex tasks.

Implements: Spec §8.1 Phase 4 - Auto-activation
Implements: Spec SPEC-14.10-14.15 for always-on micro mode
Implements: Spec SPEC-14.30-14.34 for fast-path bypass

Automatically activates RLM mode when the context is complex enough,
integrating complexity classification, orchestration, and user preferences.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .complexity_classifier import (
    extract_complexity_signals,
    is_definitely_simple,
    should_activate_rlm,
)
from .orchestration_schema import ExecutionMode, ExecutionStrategy, OrchestrationPlan
from .types import SessionContext, TaskComplexitySignals
from .user_preferences import UserPreferences

# SPEC-14.31: Fast-path patterns for queries that skip RLM entirely
FAST_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(show|cat|read)\s+\S+$", re.IGNORECASE),  # File read
    re.compile(r"^git\s+(status|log|diff)", re.IGNORECASE),  # Git commands
    re.compile(r"^(yes|no|ok|thanks|thank you)$", re.IGNORECASE),  # Conversational
    re.compile(r"^what('s| is) in .+\?$", re.IGNORECASE),  # Simple file query
    re.compile(r"^(list|ls)\s+", re.IGNORECASE),  # Directory listing
    re.compile(r"^(run|execute)\s+", re.IGNORECASE),  # Command execution
    re.compile(r"^(cd|pwd|which)\s*", re.IGNORECASE),  # Navigation
]

# Activation mode type
ActivationMode = Literal["micro", "complexity", "always", "manual", "token"]


@dataclass
class ActivationDecision:
    """
    Result of an activation decision.

    Implements: SPEC-14.10-14.15 for always-on micro mode.
    """

    should_activate: bool
    reason: str
    confidence: float  # 0.0 to 1.0
    signals: TaskComplexitySignals | None = None
    plan: OrchestrationPlan | None = None
    decision_time_ms: float = 0.0
    # SPEC-14: Mode selection (micro, balanced, thorough, or bypass)
    execution_mode: ExecutionMode = ExecutionMode.MICRO
    is_fast_path: bool = False  # SPEC-14.32: True if fast-path bypass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_activate": self.should_activate,
            "reason": self.reason,
            "confidence": self.confidence,
            "decision_time_ms": self.decision_time_ms,
            "execution_mode": self.execution_mode.value,
            "is_fast_path": self.is_fast_path,
            "signals": {
                "references_multiple_files": self.signals.references_multiple_files,
                "requires_cross_context_reasoning": self.signals.requires_cross_context_reasoning,
                "debugging_task": self.signals.debugging_task,
            }
            if self.signals
            else None,
        }


def is_fast_path_query(prompt: str) -> tuple[bool, float]:
    """
    Check if a query matches fast-path bypass patterns.

    Implements: SPEC-14.30-14.34

    Args:
        prompt: User's prompt

    Returns:
        Tuple of (is_fast_path, confidence)
    """
    prompt_stripped = prompt.strip()

    for pattern in FAST_PATH_PATTERNS:
        if pattern.match(prompt_stripped):
            return True, 0.95  # SPEC-14.34: 0.95+ confidence required

    return False, 0.0


@dataclass
class ActivationThresholds:
    """
    Configurable thresholds for auto-activation.

    Implements: SPEC-14.20-14.25 for progressive escalation.
    """

    # Token thresholds
    min_tokens_for_activation: int = 20_000
    auto_activate_above_tokens: int = 100_000

    # Complexity score thresholds
    min_complexity_score: int = 2
    high_complexity_score: int = 4

    # Confidence thresholds
    min_confidence: float = 0.6

    # Context thresholds
    max_files_for_simple: int = 2
    max_modules_for_simple: int = 2

    # SPEC-14: Micro mode thresholds
    micro_max_tokens: int = 5_000  # SPEC-14.02
    fast_path_confidence: float = 0.95  # SPEC-14.34

    # SPEC-14.20-14.25: Escalation thresholds
    escalate_to_balanced_tokens: int = 10_000
    escalate_to_thorough_tokens: int = 50_000


@dataclass
class ActivationStats:
    """Statistics for activation decisions."""

    total_decisions: int = 0
    activations: int = 0
    skips: int = 0
    overrides: int = 0
    avg_decision_time_ms: float = 0.0
    activation_reasons: dict[str, int] = field(default_factory=dict)

    def record(self, decision: ActivationDecision, was_override: bool = False) -> None:
        """Record a decision."""
        self.total_decisions += 1

        if decision.should_activate:
            self.activations += 1
        else:
            self.skips += 1

        if was_override:
            self.overrides += 1

        # Update average time
        self.avg_decision_time_ms = (
            self.avg_decision_time_ms * (self.total_decisions - 1) + decision.decision_time_ms
        ) / self.total_decisions

        # Track reasons
        reason_key = decision.reason.split(":")[0]  # Get base reason
        self.activation_reasons[reason_key] = self.activation_reasons.get(reason_key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_decisions": self.total_decisions,
            "activations": self.activations,
            "skips": self.skips,
            "overrides": self.overrides,
            "activation_rate": self.activations / self.total_decisions
            if self.total_decisions > 0
            else 0.0,
            "avg_decision_time_ms": self.avg_decision_time_ms,
            "top_reasons": dict(sorted(self.activation_reasons.items(), key=lambda x: -x[1])[:5]),
        }


# Callback type for activation events
ActivationCallback = Callable[[ActivationDecision], None]


class AutoActivator:
    """
    Manages automatic RLM activation based on task complexity.

    Implements: Spec §8.1 Phase 4 - Auto-activation
    Implements: SPEC-14.10-14.15 for always-on micro mode
    Implements: SPEC-14.30-14.34 for fast-path bypass

    Features:
    - Fast heuristic-based decisions (<50ms)
    - Micro mode as default (SPEC-14.11)
    - Fast-path bypass for trivial queries (SPEC-14.30)
    - Progressive escalation based on complexity (SPEC-14.20)
    - Respects user preferences
    - Tracks decision statistics
    - Provides callbacks for integration
    """

    def __init__(
        self,
        preferences: UserPreferences | None = None,
        thresholds: ActivationThresholds | None = None,
        activation_mode: ActivationMode = "micro",
    ):
        """
        Initialize auto-activator.

        Args:
            preferences: User preferences (defaults to standard)
            thresholds: Activation thresholds (defaults to standard)
            activation_mode: Default activation mode (SPEC-14.12)
        """
        self.preferences = preferences or UserPreferences()
        self.thresholds = thresholds or ActivationThresholds()
        self.activation_mode = activation_mode
        self._stats = ActivationStats()
        self._callbacks: list[ActivationCallback] = []

    def should_activate(
        self,
        prompt: str,
        context: SessionContext,
        force_rlm: bool = False,
        force_simple: bool = False,
    ) -> ActivationDecision:
        """
        Decide whether to activate RLM mode.

        Implements: SPEC-14.10-14.15 for always-on micro mode

        Args:
            prompt: User's prompt
            context: Current session context
            force_rlm: Force RLM activation (full mode)
            force_simple: Force simple mode (bypass RLM)

        Returns:
            ActivationDecision with result and reasoning
        """
        start = time.time()

        # Check manual overrides first
        if force_rlm:
            decision = ActivationDecision(
                should_activate=True,
                reason="manual_force",
                confidence=1.0,
                execution_mode=ExecutionMode.BALANCED,
            )
            self._finalize_decision(decision, start, was_override=True)
            return decision

        if force_simple:
            decision = ActivationDecision(
                should_activate=False,
                reason="manual_force_simple",
                confidence=1.0,
                execution_mode=ExecutionMode.FAST,
            )
            self._finalize_decision(decision, start, was_override=True)
            return decision

        # Check if auto-activation is disabled (manual mode)
        if self.activation_mode == "manual" or not self.preferences.auto_activate:
            decision = ActivationDecision(
                should_activate=False,
                reason="auto_activate_disabled",
                confidence=1.0,
                execution_mode=ExecutionMode.FAST,
            )
            self._finalize_decision(decision, start)
            return decision

        # SPEC-14.30-14.34: Fast-path bypass for trivial queries
        is_fast_path, fast_confidence = is_fast_path_query(prompt)
        if is_fast_path and fast_confidence >= self.thresholds.fast_path_confidence:
            decision = ActivationDecision(
                should_activate=False,
                reason="fast_path_bypass",
                confidence=fast_confidence,
                is_fast_path=True,
                execution_mode=ExecutionMode.FAST,
            )
            self._finalize_decision(decision, start)
            return decision

        # SPEC-14.10-14.11: Micro mode is the default
        if self.activation_mode == "micro":
            return self._micro_mode_decision(prompt, context, start)

        # Legacy activation modes
        return self._complexity_mode_decision(prompt, context, start)

    def _micro_mode_decision(
        self,
        prompt: str,
        context: SessionContext,
        start_time: float,
    ) -> ActivationDecision:
        """
        Make activation decision for micro mode (always-on default).

        Implements: SPEC-14.10-14.15, SPEC-14.20-14.25

        Args:
            prompt: User's prompt
            context: Current session context
            start_time: Decision start time

        Returns:
            ActivationDecision with micro mode or escalated mode
        """
        # Extract complexity signals for escalation check
        signals = extract_complexity_signals(prompt, context)

        # SPEC-14.20-14.25: Check escalation triggers
        execution_mode, reason = self._check_escalation_triggers(prompt, signals, context)

        # Calculate confidence
        confidence = self._calculate_confidence(signals, context)

        # SPEC-14.10: Always activate in micro mode (unless fast-path)
        decision = ActivationDecision(
            should_activate=True,
            reason=reason,
            confidence=confidence,
            signals=signals,
            execution_mode=execution_mode,
        )

        self._finalize_decision(decision, start_time)
        return decision

    def _check_escalation_triggers(
        self,
        prompt: str,
        signals: TaskComplexitySignals,
        context: SessionContext,
    ) -> tuple[ExecutionMode, str]:
        """
        Check if micro mode should escalate to a higher mode.

        Implements: SPEC-14.20-14.25 - Progressive escalation

        Args:
            prompt: User's prompt
            signals: Extracted complexity signals
            context: Session context

        Returns:
            Tuple of (execution_mode, reason)
        """
        prompt_lower = prompt.lower()

        # SPEC-14.21: Immediate escalation to THOROUGH
        if any(kw in prompt_lower for kw in ["architecture", "design decision", "thorough"]):
            return ExecutionMode.THOROUGH, "escalate_thorough:architecture_or_user_request"

        # SPEC-14.21: Immediate escalation to BALANCED
        if signals.references_multiple_files:
            return ExecutionMode.BALANCED, "escalate_balanced:multi_file_reference"

        if signals.debugging_task:
            return ExecutionMode.BALANCED, "escalate_balanced:debugging_task"

        if any(kw in prompt_lower for kw in ["discover", "explore", "understand", "analyze"]):
            return ExecutionMode.BALANCED, "escalate_balanced:discovery_keywords"

        if signals.requires_cross_context_reasoning:
            return ExecutionMode.BALANCED, "escalate_balanced:cross_context_reasoning"

        # Large context triggers escalation
        if context.total_tokens >= self.thresholds.escalate_to_thorough_tokens:
            return ExecutionMode.THOROUGH, "escalate_thorough:large_context"

        if context.total_tokens >= self.thresholds.escalate_to_balanced_tokens:
            return ExecutionMode.BALANCED, "escalate_balanced:medium_context"

        # Default: stay in micro mode
        return ExecutionMode.MICRO, "micro_mode:default"

    def _complexity_mode_decision(
        self,
        prompt: str,
        context: SessionContext,
        start_time: float,
    ) -> ActivationDecision:
        """
        Make activation decision using legacy complexity-based mode.

        Args:
            prompt: User's prompt
            context: Current session context
            start_time: Decision start time

        Returns:
            ActivationDecision
        """
        # Fast path: definitely simple queries
        if is_definitely_simple(prompt, context):
            decision = ActivationDecision(
                should_activate=False,
                reason="definitely_simple",
                confidence=0.95,
                execution_mode=ExecutionMode.FAST,
            )
            self._finalize_decision(decision, start_time)
            return decision

        # Extract complexity signals
        signals = extract_complexity_signals(prompt, context)

        # Token-based auto-activation
        if context.total_tokens >= self.thresholds.auto_activate_above_tokens:
            decision = ActivationDecision(
                should_activate=True,
                reason="large_context",
                confidence=0.9,
                signals=signals,
                execution_mode=ExecutionMode.THOROUGH,
            )
            self._finalize_decision(decision, start_time)
            return decision

        # Use complexity classifier
        should_activate, reason = should_activate_rlm(prompt, context)

        # Calculate confidence based on signal strength
        confidence = self._calculate_confidence(signals, context)

        # Determine execution mode based on complexity
        if should_activate:
            if signals.requires_cross_context_reasoning or signals.debugging_task:
                exec_mode = ExecutionMode.BALANCED
            else:
                exec_mode = ExecutionMode.FAST
        else:
            exec_mode = ExecutionMode.FAST

        decision = ActivationDecision(
            should_activate=should_activate,
            reason=reason,
            confidence=confidence,
            signals=signals,
            execution_mode=exec_mode,
        )

        self._finalize_decision(decision, start_time)
        return decision

    def _calculate_confidence(
        self,
        signals: TaskComplexitySignals,
        context: SessionContext,
    ) -> float:
        """Calculate confidence in the activation decision."""
        confidence = 0.5  # Base confidence

        # Strong positive signals
        if signals.requires_cross_context_reasoning:
            confidence += 0.2
        if signals.debugging_task and signals.recent_tool_outputs_large:
            confidence += 0.15
        if signals.references_multiple_files:
            confidence += 0.1

        # Context size factors
        if context.total_tokens > 50_000:
            confidence += 0.1
        elif context.total_tokens < 10_000:
            confidence -= 0.1

        # Confusion indicator is strong
        if signals.previous_turn_was_confused:
            confidence += 0.15

        return min(1.0, max(0.0, confidence))

    def _finalize_decision(
        self,
        decision: ActivationDecision,
        start_time: float,
        was_override: bool = False,
    ) -> None:
        """Finalize decision with timing and callbacks."""
        decision.decision_time_ms = (time.time() - start_time) * 1000

        # Record stats
        self._stats.record(decision, was_override)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(decision)
            except Exception:
                pass  # Don't let callback errors affect activation

    def create_plan_from_decision(
        self,
        decision: ActivationDecision,
    ) -> OrchestrationPlan | None:
        """
        Create an orchestration plan from an activation decision.

        Implements: SPEC-14.01-14.02 for micro mode plans

        Args:
            decision: The activation decision

        Returns:
            OrchestrationPlan if activated, None otherwise
        """
        if not decision.should_activate:
            return OrchestrationPlan.bypass(reason=decision.reason)

        # SPEC-14.01-14.02: Create micro mode plan
        if decision.execution_mode == ExecutionMode.MICRO:
            primary_model = self.preferences.preferred_model or "sonnet"
            plan = OrchestrationPlan.micro(
                reason=decision.reason,
                parent_model=primary_model,
            )
            decision.plan = plan
            return plan

        # Non-micro modes: use strategy-based planning

        # Map execution mode to strategy
        strategy_map = {
            ExecutionMode.FAST: ExecutionStrategy.DIRECT_RESPONSE,
            ExecutionMode.BALANCED: ExecutionStrategy.DISCOVERY,
            ExecutionMode.THOROUGH: ExecutionStrategy.RECURSIVE_DEBUG,
        }
        strategy = strategy_map.get(decision.execution_mode, ExecutionStrategy.DISCOVERY)

        # Build plan from strategy
        plan = OrchestrationPlan.from_strategy(
            strategy=strategy,
            activation_reason=decision.reason,
        )

        # Apply user preferences
        if self.preferences.preferred_model:
            plan.primary_model = self.preferences.preferred_model

        if self.preferences.max_depth < plan.depth_budget:
            plan.depth_budget = self.preferences.max_depth

        if self.preferences.budget_tokens:
            plan.max_tokens = self.preferences.budget_tokens

        if self.preferences.budget_dollars:
            plan.max_cost_dollars = self.preferences.budget_dollars

        # Apply execution mode and tool access from preferences
        plan.execution_mode = self.preferences.execution_mode
        plan.tool_access = self.preferences.tool_access

        decision.plan = plan
        return plan

    def add_callback(self, callback: ActivationCallback) -> None:
        """Add a callback for activation events."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: ActivationCallback) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_statistics(self) -> dict[str, Any]:
        """Get activation statistics."""
        return self._stats.to_dict()

    def reset_statistics(self) -> None:
        """Reset activation statistics."""
        self._stats = ActivationStats()


# Global auto-activator instance
_global_activator: AutoActivator | None = None


def get_auto_activator() -> AutoActivator:
    """Get global auto-activator instance."""
    global _global_activator
    if _global_activator is None:
        _global_activator = AutoActivator()
    return _global_activator


def check_auto_activation(
    prompt: str,
    context: SessionContext,
    preferences: UserPreferences | None = None,
) -> ActivationDecision:
    """
    Convenience function to check if RLM should auto-activate.

    Args:
        prompt: User's prompt
        context: Current session context
        preferences: Optional user preferences

    Returns:
        ActivationDecision
    """
    activator = get_auto_activator()
    if preferences:
        activator.preferences = preferences
    return activator.should_activate(prompt, context)


__all__ = [
    "ActivationCallback",
    "ActivationDecision",
    "ActivationMode",
    "ActivationStats",
    "ActivationThresholds",
    "AutoActivator",
    "FAST_PATH_PATTERNS",
    "check_auto_activation",
    "get_auto_activator",
    "is_fast_path_query",
]
