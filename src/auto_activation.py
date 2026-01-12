"""
Auto-activation for RLM mode on complex tasks.

Implements: Spec §8.1 Phase 4 - Auto-activation

Automatically activates RLM mode when the context is complex enough,
integrating complexity classification, orchestration, and user preferences.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .complexity_classifier import (
    extract_complexity_signals,
    is_definitely_simple,
    should_activate_rlm,
)
from .orchestration_schema import ExecutionMode, OrchestrationPlan, ToolAccessLevel
from .types import SessionContext, TaskComplexitySignals
from .user_preferences import UserPreferences


@dataclass
class ActivationDecision:
    """Result of an activation decision."""

    should_activate: bool
    reason: str
    confidence: float  # 0.0 to 1.0
    signals: TaskComplexitySignals | None = None
    plan: OrchestrationPlan | None = None
    decision_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_activate": self.should_activate,
            "reason": self.reason,
            "confidence": self.confidence,
            "decision_time_ms": self.decision_time_ms,
            "signals": {
                "references_multiple_files": self.signals.references_multiple_files,
                "requires_cross_context_reasoning": self.signals.requires_cross_context_reasoning,
                "debugging_task": self.signals.debugging_task,
            } if self.signals else None,
        }


@dataclass
class ActivationThresholds:
    """Configurable thresholds for auto-activation."""

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
            (self.avg_decision_time_ms * (self.total_decisions - 1) + decision.decision_time_ms)
            / self.total_decisions
        )

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
            "activation_rate": self.activations / self.total_decisions if self.total_decisions > 0 else 0.0,
            "avg_decision_time_ms": self.avg_decision_time_ms,
            "top_reasons": dict(sorted(
                self.activation_reasons.items(),
                key=lambda x: -x[1]
            )[:5]),
        }


# Callback type for activation events
ActivationCallback = Callable[[ActivationDecision], None]


class AutoActivator:
    """
    Manages automatic RLM activation based on task complexity.

    Implements: Spec §8.1 Phase 4 - Auto-activation

    Features:
    - Fast heuristic-based decisions (<50ms)
    - Respects user preferences
    - Tracks decision statistics
    - Provides callbacks for integration
    """

    def __init__(
        self,
        preferences: UserPreferences | None = None,
        thresholds: ActivationThresholds | None = None,
    ):
        """
        Initialize auto-activator.

        Args:
            preferences: User preferences (defaults to standard)
            thresholds: Activation thresholds (defaults to standard)
        """
        self.preferences = preferences or UserPreferences()
        self.thresholds = thresholds or ActivationThresholds()
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

        Args:
            prompt: User's prompt
            context: Current session context
            force_rlm: Force RLM activation
            force_simple: Force simple mode

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
            )
            self._finalize_decision(decision, start, was_override=True)
            return decision

        if force_simple:
            decision = ActivationDecision(
                should_activate=False,
                reason="manual_force_simple",
                confidence=1.0,
            )
            self._finalize_decision(decision, start, was_override=True)
            return decision

        # Check if auto-activation is disabled
        if not self.preferences.auto_activate:
            decision = ActivationDecision(
                should_activate=False,
                reason="auto_activate_disabled",
                confidence=1.0,
            )
            self._finalize_decision(decision, start)
            return decision

        # Fast path: definitely simple queries
        if is_definitely_simple(prompt, context):
            decision = ActivationDecision(
                should_activate=False,
                reason="definitely_simple",
                confidence=0.95,
            )
            self._finalize_decision(decision, start)
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
            )
            self._finalize_decision(decision, start)
            return decision

        # Use complexity classifier
        should_activate, reason = should_activate_rlm(prompt, context)

        # Calculate confidence based on signal strength
        confidence = self._calculate_confidence(signals, context)

        decision = ActivationDecision(
            should_activate=should_activate,
            reason=reason,
            confidence=confidence,
            signals=signals,
        )

        self._finalize_decision(decision, start)
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

        Args:
            decision: The activation decision

        Returns:
            OrchestrationPlan if activated, None otherwise
        """
        if not decision.should_activate:
            return None

        # Build plan based on preferences and signals
        from .smart_router import ModelTier

        # Determine model tier from preferences
        model_tier = ModelTier.POWERFUL
        primary_model = self.preferences.preferred_model or "claude-sonnet-4-20250514"

        # Adjust depth based on confidence and preferences
        depth = min(self.preferences.max_depth, 2)
        if decision.confidence < 0.7:
            depth = 1  # Lower depth for uncertain activations

        plan = OrchestrationPlan(
            activate_rlm=True,
            activation_reason=decision.reason,
            model_tier=model_tier,
            primary_model=primary_model,
            depth_budget=depth,
            execution_mode=self.preferences.execution_mode,
            tool_access=self.preferences.tool_access,
            max_tokens=self.preferences.budget_tokens,
            max_cost_dollars=self.preferences.budget_dollars,
        )

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
    "ActivationStats",
    "ActivationThresholds",
    "AutoActivator",
    "check_auto_activation",
    "get_auto_activator",
]
