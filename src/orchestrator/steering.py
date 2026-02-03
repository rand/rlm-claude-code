"""
User interaction and steering support for RLM sessions.

Implements: SPEC-12.06, SPEC-11.10-11.16

Contains:
- SteeringPoint for decision points
- SteeringPointType for categorizing steering
- SteeringResponse for logging decisions
- InteractiveOrchestrator for user interaction
- Auto-steering policy
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class SteeringDecision(Enum):
    """Decisions available at steering points."""

    CONTINUE = "continue"
    STOP = "stop"
    ADJUST_DEPTH = "adjust_depth"
    ADJUST_MODEL = "adjust_model"
    PROVIDE_CONTEXT = "provide_context"
    SKIP_STEP = "skip_step"
    ABORT = "abort"
    REFINE = "refine"


class SteeringPointType(Enum):
    """
    Types of steering points.

    Implements: SPEC-11.11
    """

    BRANCH = "branch"  # Choose between exploration paths
    DEPTH = "depth"  # Adjust remaining depth budget
    ABORT = "abort"  # Cancel and return current results
    REFINE = "refine"  # Provide additional guidance


@dataclass
class SteeringPoint:
    """
    A point where user steering is available.

    Implements: SPEC-12.06, SPEC-11.12

    Captures the decision context and available options
    at points where user intervention might be valuable.
    """

    turn: int
    depth: int
    decision_type: str
    options: list[str]
    context: str
    # SPEC-11.12: Extended fields
    point_type: SteeringPointType = SteeringPointType.BRANCH
    default: str = ""
    timeout: float = 30.0
    current_state: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    point_id: str = field(default_factory=lambda: f"sp-{uuid.uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        """Set default if not provided."""
        if not self.default and self.options:
            self.default = self.options[0]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "point_id": self.point_id,
            "turn": self.turn,
            "depth": self.depth,
            "decision_type": self.decision_type,
            "point_type": self.point_type.value,
            "options": self.options,
            "default": self.default,
            "timeout": self.timeout,
            "context": self.context,
            "current_state": self.current_state,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }


@dataclass
class SteeringResponse:
    """
    Record of a steering decision.

    Implements: SPEC-11.15
    """

    point_id: str
    point_type: SteeringPointType
    decision: SteeringDecision
    selected_option: str
    timestamp: float
    source: str  # "user" or "auto"
    response_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "point_id": self.point_id,
            "point_type": self.point_type.value,
            "decision": self.decision.value,
            "selected_option": self.selected_option,
            "timestamp": self.timestamp,
            "source": self.source,
            "response_time_ms": self.response_time_ms,
        }


class SteeringCallback(Protocol):
    """Protocol for user steering callbacks."""

    def __call__(self, point: SteeringPoint) -> SteeringDecision:
        """
        Get user decision at a steering point.

        Args:
            point: The steering point context

        Returns:
            User's steering decision
        """
        ...


class AutoSteeringPolicy:
    """
    Automatic steering policy for non-interactive use.

    Implements: SPEC-12.06, SPEC-11.14

    Provides sensible defaults when user interaction is not available.
    """

    def __init__(
        self,
        max_turns_before_stop: int = 50,
        max_depth: int = 3,
        cost_threshold_usd: float = 1.0,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize auto-steering policy.

        Args:
            max_turns_before_stop: Maximum turns before forcing stop
            max_depth: Maximum recursion depth
            cost_threshold_usd: Maximum cost before stopping
            confidence_threshold: Minimum confidence to continue
        """
        self.max_turns_before_stop = max_turns_before_stop
        self.max_depth = max_depth
        self.cost_threshold_usd = cost_threshold_usd
        self.confidence_threshold = confidence_threshold

    def decide(self, point: SteeringPoint) -> SteeringDecision:
        """
        Make automatic steering decision.

        Args:
            point: The steering point context

        Returns:
            Automatic steering decision
        """
        # Check turn limit
        if point.turn >= self.max_turns_before_stop:
            return SteeringDecision.STOP

        # Check depth limit
        if point.depth >= self.max_depth:
            return SteeringDecision.STOP

        # Check cost if available
        cost = point.current_state.get("cost_usd", 0.0)
        if cost >= self.cost_threshold_usd:
            return SteeringDecision.STOP

        # Check confidence if available
        confidence = point.current_state.get("confidence", 1.0)
        if confidence < self.confidence_threshold:
            return SteeringDecision.ADJUST_DEPTH

        # Default: continue
        return SteeringDecision.CONTINUE

    def __call__(self, point: SteeringPoint) -> SteeringDecision:
        """Make policy callable for use as SteeringCallback."""
        return self.decide(point)


class InteractiveOrchestrator:
    """
    Orchestrator wrapper with user interaction support.

    Implements: SPEC-12.06, SPEC-11.10-11.16

    Provides:
    - Steering point detection
    - User callback integration
    - Auto-steering fallback
    - Timeout handling
    - Steering logging
    """

    def __init__(
        self,
        callback: SteeringCallback | None = None,
        auto_policy: AutoSteeringPolicy | None = None,
        low_confidence_threshold: float = 0.4,
        min_paths_for_steering: int = 2,
    ):
        """
        Initialize interactive orchestrator.

        Args:
            callback: Optional user steering callback
            auto_policy: Auto-steering policy for non-interactive use
            low_confidence_threshold: Threshold for low confidence steering
            min_paths_for_steering: Minimum paths to trigger branch steering
        """
        self.callback = callback
        self.auto_policy = auto_policy or AutoSteeringPolicy()
        self.low_confidence_threshold = low_confidence_threshold
        self.min_paths_for_steering = min_paths_for_steering
        self._steering_history: list[tuple[SteeringPoint, SteeringDecision]] = []
        self._steering_responses: list[SteeringResponse] = []

    # SPEC-11.13: Steering point presentation detection

    def should_present_steering(
        self,
        turn: int,
        depth: int,
        event: str,
        confidence: float | None = None,
        num_paths: int | None = None,
        cost_usd: float | None = None,
    ) -> bool:
        """
        Check if steering should be presented at this point.

        Implements: SPEC-11.13

        Args:
            turn: Current turn number
            depth: Current recursion depth
            event: Type of event triggering check
            confidence: Current confidence level
            num_paths: Number of available paths
            cost_usd: Accumulated cost

        Returns:
            True if steering should be presented
        """
        # Before recursion with multiple paths
        if event == "before_recursion" and num_paths and num_paths >= self.min_paths_for_steering:
            return True

        # After low-confidence intermediate results
        if event == "intermediate_result" and confidence is not None:
            if confidence < self.low_confidence_threshold:
                return True

        # When multiple viable paths exist
        if event == "path_selection" and num_paths and num_paths >= self.min_paths_for_steering:
            return True

        # Legacy checks for backward compatibility
        if depth > 0 and turn % 5 == 0:
            return True

        if cost_usd is not None and cost_usd > 0.5:
            return True

        if turn in {10, 20, 30, 40}:
            return True

        return False

    def should_steer(
        self,
        turn: int,
        depth: int,
        confidence: float | None = None,
        cost_usd: float | None = None,
    ) -> bool:
        """
        Check if steering should occur at this point (legacy method).

        Args:
            turn: Current turn number
            depth: Current recursion depth
            confidence: Optional current confidence
            cost_usd: Optional accumulated cost

        Returns:
            True if steering point should trigger
        """
        return self.should_present_steering(
            turn=turn,
            depth=depth,
            event="legacy",
            confidence=confidence,
            cost_usd=cost_usd,
        )

    # Steering point creation methods

    def create_branch_point(
        self,
        turn: int,
        depth: int,
        paths: list[tuple[str, float]],
        context: str,
    ) -> SteeringPoint:
        """
        Create a branch steering point.

        Implements: SPEC-11.11 (branch type)

        Args:
            turn: Current turn
            depth: Current depth
            paths: List of (path_description, value_estimate) tuples
            context: Context description

        Returns:
            SteeringPoint for branch decision
        """
        # Sort by value estimate, highest first
        sorted_paths = sorted(paths, key=lambda p: p[1], reverse=True)
        options = [p[0] for p in sorted_paths]
        default = options[0] if options else ""

        return SteeringPoint(
            turn=turn,
            depth=depth,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=options,
            default=default,
            context=context,
        )

    def create_depth_point(
        self,
        turn: int,
        depth: int,
        current_depth_budget: int,
        confidence: float,
        context: str,
    ) -> SteeringPoint:
        """
        Create a depth adjustment steering point.

        Implements: SPEC-11.11 (depth type)

        Args:
            turn: Current turn
            depth: Current depth
            current_depth_budget: Remaining depth budget
            confidence: Current confidence
            context: Context description

        Returns:
            SteeringPoint for depth decision
        """
        options = ["increase", "decrease", "maintain"]
        # Default based on confidence
        if confidence < 0.3:
            default = "increase"
        elif confidence > 0.8:
            default = "decrease"
        else:
            default = "maintain"

        return SteeringPoint(
            turn=turn,
            depth=depth,
            decision_type="depth",
            point_type=SteeringPointType.DEPTH,
            options=options,
            default=default,
            context=context,
            current_state={
                "depth_budget": current_depth_budget,
                "confidence": confidence,
            },
        )

    def create_abort_point(
        self,
        turn: int,
        depth: int,
        cost_so_far: float,
        progress_summary: str,
    ) -> SteeringPoint:
        """
        Create an abort steering point.

        Implements: SPEC-11.11 (abort type)

        Args:
            turn: Current turn
            depth: Current depth
            cost_so_far: Accumulated cost
            progress_summary: Summary of progress so far

        Returns:
            SteeringPoint for abort decision
        """
        return SteeringPoint(
            turn=turn,
            depth=depth,
            decision_type="abort",
            point_type=SteeringPointType.ABORT,
            options=["continue", "abort"],
            default="continue",
            context=f"Cost: ${cost_so_far:.3f}. {progress_summary}",
            current_state={"cost_usd": cost_so_far},
        )

    def create_refine_point(
        self,
        turn: int,
        depth: int,
        current_result: str,
        context: str,
    ) -> SteeringPoint:
        """
        Create a refine steering point.

        Implements: SPEC-11.11 (refine type)

        Args:
            turn: Current turn
            depth: Current depth
            current_result: Current result to potentially refine
            context: Context description

        Returns:
            SteeringPoint for refine decision
        """
        return SteeringPoint(
            turn=turn,
            depth=depth,
            decision_type="refine",
            point_type=SteeringPointType.REFINE,
            options=["accept", "refine"],
            default="accept",
            context=context,
            metadata={"current_result": current_result},
        )

    def create_steering_point(
        self,
        turn: int,
        depth: int,
        context: str,
        decision_type: str = "continue_or_stop",
        current_state: dict[str, Any] | None = None,
    ) -> SteeringPoint:
        """
        Create a steering point for user decision (legacy method).

        Args:
            turn: Current turn number
            depth: Current recursion depth
            context: Human-readable context description
            decision_type: Type of decision needed
            current_state: Current orchestration state

        Returns:
            SteeringPoint for user decision
        """
        # Determine options based on decision type
        if decision_type == "continue_or_stop":
            options = ["continue", "stop", "adjust_depth"]
        elif decision_type == "model_selection":
            options = ["haiku", "sonnet", "opus"]
        elif decision_type == "depth_adjustment":
            options = ["increase", "decrease", "maintain"]
        else:
            options = ["continue", "stop"]

        return SteeringPoint(
            turn=turn,
            depth=depth,
            decision_type=decision_type,
            options=options,
            context=context,
            current_state=current_state or {},
        )

    # Decision methods

    def get_decision(self, point: SteeringPoint) -> SteeringDecision:
        """
        Get steering decision from user or auto-policy.

        Implements: SPEC-11.15 (logging)

        Args:
            point: The steering point

        Returns:
            Steering decision
        """
        start_time = time.time()
        source = "auto"

        if self.callback is not None:
            try:
                decision = self.callback(point)
                source = "user"
            except Exception:
                # Fallback to auto on callback error
                decision = self.auto_policy.decide(point)
        else:
            decision = self.auto_policy.decide(point)

        response_time_ms = (time.time() - start_time) * 1000

        # Record decision in history
        self._steering_history.append((point, decision))

        # Record detailed response
        response = SteeringResponse(
            point_id=point.point_id,
            point_type=point.point_type,
            decision=decision,
            selected_option=self._decision_to_option(decision, point),
            timestamp=time.time(),
            source=source,
            response_time_ms=response_time_ms,
        )
        self._steering_responses.append(response)

        return decision

    def get_decision_with_timeout(self, point: SteeringPoint) -> SteeringDecision:
        """
        Get steering decision with timeout handling.

        Implements: SPEC-11.16

        Args:
            point: The steering point

        Returns:
            Steering decision (default if timeout)
        """
        if self.callback is None:
            return self.get_decision(point)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.callback, point)
            try:
                decision = future.result(timeout=point.timeout)
                source = "user"
            except (FuturesTimeoutError, Exception):
                # Timeout or error: use default via auto-policy
                decision = self._default_to_decision(point.default, point)
                source = "auto"

        response_time_ms = (time.time() - start_time) * 1000

        # Record decision
        self._steering_history.append((point, decision))

        response = SteeringResponse(
            point_id=point.point_id,
            point_type=point.point_type,
            decision=decision,
            selected_option=self._decision_to_option(decision, point),
            timestamp=time.time(),
            source=source,
            response_time_ms=response_time_ms,
        )
        self._steering_responses.append(response)

        return decision

    def _decision_to_option(self, decision: SteeringDecision, point: SteeringPoint) -> str:
        """Map decision to selected option string."""
        decision_map = {
            SteeringDecision.CONTINUE: "continue",
            SteeringDecision.STOP: "stop",
            SteeringDecision.ABORT: "abort",
            SteeringDecision.ADJUST_DEPTH: "adjust_depth",
            SteeringDecision.REFINE: "refine",
        }
        return decision_map.get(decision, point.default)

    def _default_to_decision(self, default: str, point: SteeringPoint) -> SteeringDecision:
        """Map default option to SteeringDecision."""
        default_map = {
            "continue": SteeringDecision.CONTINUE,
            "stop": SteeringDecision.STOP,
            "abort": SteeringDecision.ABORT,
            "adjust_depth": SteeringDecision.ADJUST_DEPTH,
            "increase": SteeringDecision.ADJUST_DEPTH,
            "decrease": SteeringDecision.ADJUST_DEPTH,
            "maintain": SteeringDecision.CONTINUE,
            "refine": SteeringDecision.REFINE,
            "accept": SteeringDecision.CONTINUE,
        }
        return default_map.get(default.lower(), self.auto_policy.decide(point))

    # History and logging methods

    def get_steering_history(self) -> list[tuple[SteeringPoint, SteeringDecision]]:
        """Get history of steering decisions."""
        return self._steering_history.copy()

    def get_steering_responses(self) -> list[SteeringResponse]:
        """
        Get detailed steering responses.

        Implements: SPEC-11.15

        Returns:
            List of SteeringResponse records
        """
        return self._steering_responses.copy()

    def clear_history(self) -> None:
        """Clear steering history."""
        self._steering_history.clear()
        self._steering_responses.clear()


__all__ = [
    "AutoSteeringPolicy",
    "InteractiveOrchestrator",
    "SteeringCallback",
    "SteeringDecision",
    "SteeringPoint",
    "SteeringPointType",
    "SteeringResponse",
]
