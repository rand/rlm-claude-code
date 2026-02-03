"""
Learning from user corrections to improve RLM classifier.

Implements: SPEC-11.20-11.25

Captures user corrections to RLM outputs, analyzes patterns,
and suggests classifier adjustments with confirmation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CorrectionType(Enum):
    """
    Type of user correction.

    Implements: SPEC-11.21
    """

    FACTUAL = "factual"
    INCOMPLETE = "incomplete"
    WRONG_APPROACH = "wrong_approach"
    OVER_COMPLEX = "over_complex"
    UNDER_COMPLEX = "under_complex"


@dataclass
class Correction:
    """
    A user correction record.

    Implements: SPEC-11.22
    """

    query: str
    rlm_output: str
    user_correction: str
    correction_type: CorrectionType
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "rlm_output": self.rlm_output,
            "user_correction": self.user_correction,
            "correction_type": self.correction_type.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ClassifierAdjustment:
    """
    Suggested classifier adjustment.

    Implements: SPEC-11.24
    """

    signal_adjustments: dict[str, float]
    threshold_adjustment: float
    reasoning: str
    is_confirmed: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "signal_adjustments": self.signal_adjustments,
            "threshold_adjustment": self.threshold_adjustment,
            "reasoning": self.reasoning,
            "is_confirmed": self.is_confirmed,
            "created_at": self.created_at.isoformat(),
        }


class CorrectionRecorder:
    """
    Record user corrections.

    Implements: SPEC-11.20, SPEC-11.22
    """

    def __init__(self) -> None:
        """Initialize recorder."""
        self._corrections: list[Correction] = []

    def record_correction(
        self,
        query: str,
        rlm_output: str,
        user_correction: str,
        correction_type: CorrectionType,
    ) -> Correction:
        """
        Record a user correction.

        Implements: SPEC-11.20

        Args:
            query: Original query
            rlm_output: RLM's output
            user_correction: User's correction
            correction_type: Type of correction

        Returns:
            Recorded Correction
        """
        correction = Correction(
            query=query,
            rlm_output=rlm_output,
            user_correction=user_correction,
            correction_type=correction_type,
        )
        self._corrections.append(correction)
        return correction

    def get_corrections(self) -> list[Correction]:
        """Get all recorded corrections."""
        return self._corrections.copy()

    def get_corrections_by_type(
        self,
        correction_type: CorrectionType,
    ) -> list[Correction]:
        """Get corrections of a specific type."""
        return [c for c in self._corrections if c.correction_type == correction_type]

    def clear(self) -> None:
        """Clear all corrections."""
        self._corrections.clear()


class CorrectionAnalyzer:
    """
    Analyze corrections to suggest classifier adjustments.

    Implements: SPEC-11.23
    """

    def __init__(self) -> None:
        """Initialize analyzer."""
        self._adjustment_per_correction = 0.05

    def analyze(self, corrections: list[Correction]) -> ClassifierAdjustment:
        """
        Analyze corrections and suggest adjustments.

        Implements: SPEC-11.23

        Args:
            corrections: List of corrections to analyze

        Returns:
            ClassifierAdjustment with suggestions
        """
        if not corrections:
            return ClassifierAdjustment(
                signal_adjustments={},
                threshold_adjustment=0.0,
                reasoning="No corrections to analyze",
            )

        # Count correction types
        over_complex_count = sum(
            1 for c in corrections if c.correction_type == CorrectionType.OVER_COMPLEX
        )
        under_complex_count = sum(
            1 for c in corrections if c.correction_type == CorrectionType.UNDER_COMPLEX
        )

        # Calculate threshold adjustment
        # SPEC-11.23: OVER_COMPLEX → raise threshold, UNDER_COMPLEX → lower
        threshold_adjustment = (
            over_complex_count - under_complex_count
        ) * self._adjustment_per_correction

        # Build reasoning
        reasoning_parts = []
        if over_complex_count > 0:
            reasoning_parts.append(
                f"{over_complex_count} over-complex corrections suggest raising threshold"
            )
        if under_complex_count > 0:
            reasoning_parts.append(
                f"{under_complex_count} under-complex corrections suggest lowering threshold"
            )

        reasoning = (
            "; ".join(reasoning_parts) if reasoning_parts else "No threshold-affecting corrections"
        )

        # Signal adjustments based on patterns
        signal_adjustments: dict[str, float] = {}

        # Analyze wrong approach corrections
        wrong_approach_count = sum(
            1 for c in corrections if c.correction_type == CorrectionType.WRONG_APPROACH
        )
        if wrong_approach_count > 2:
            signal_adjustments["strategy_weight"] = -0.1

        return ClassifierAdjustment(
            signal_adjustments=signal_adjustments,
            threshold_adjustment=threshold_adjustment,
            reasoning=reasoning,
        )


class UserCorrectionLearner:
    """
    Main interface for learning from user corrections.

    Implements: SPEC-11.20-11.25
    """

    def __init__(self) -> None:
        """Initialize learner."""
        self._recorder = CorrectionRecorder()
        self._analyzer = CorrectionAnalyzer()
        self._pending_adjustment: ClassifierAdjustment | None = None
        self._applied_adjustments: list[ClassifierAdjustment] = []
        self._adjustment_log: list[dict[str, Any]] = []

    def record_correction(
        self,
        query: str,
        rlm_output: str,
        user_correction: str,
        correction_type: CorrectionType,
    ) -> Correction:
        """
        Record a user correction and update pending adjustment.

        Args:
            query: Original query
            rlm_output: RLM's output
            user_correction: User's correction
            correction_type: Type of correction

        Returns:
            Recorded Correction
        """
        correction = self._recorder.record_correction(
            query=query,
            rlm_output=rlm_output,
            user_correction=user_correction,
            correction_type=correction_type,
        )

        # Re-analyze corrections
        self._pending_adjustment = self._analyzer.analyze(self._recorder.get_corrections())

        return correction

    def get_corrections(self) -> list[Correction]:
        """Get all recorded corrections."""
        return self._recorder.get_corrections()

    def get_pending_adjustment(self) -> ClassifierAdjustment | None:
        """
        Get pending adjustment awaiting confirmation.

        Implements: SPEC-11.25

        Returns:
            Pending adjustment or None
        """
        return self._pending_adjustment

    def confirm_adjustment(self) -> bool:
        """
        Confirm and apply pending adjustment.

        Implements: SPEC-11.25

        Returns:
            True if adjustment was applied
        """
        if self._pending_adjustment is None:
            return False

        self._pending_adjustment.is_confirmed = True
        self._applied_adjustments.append(self._pending_adjustment)

        # Log the adjustment
        self._adjustment_log.append(
            {
                "adjustment": self._pending_adjustment.to_dict(),
                "applied_at": datetime.now().isoformat(),
                "action": "confirmed",
            }
        )

        self._pending_adjustment = None
        self._recorder.clear()

        return True

    def reject_adjustment(self) -> bool:
        """
        Reject pending adjustment.

        Returns:
            True if adjustment was rejected
        """
        if self._pending_adjustment is None:
            return False

        # Log the rejection
        self._adjustment_log.append(
            {
                "adjustment": self._pending_adjustment.to_dict(),
                "rejected_at": datetime.now().isoformat(),
                "action": "rejected",
            }
        )

        self._pending_adjustment = None
        return True

    def get_applied_adjustments(self) -> list[ClassifierAdjustment]:
        """Get list of applied adjustments."""
        return self._applied_adjustments.copy()

    def get_adjustment_log(self) -> list[dict[str, Any]]:
        """
        Get adjustment log for auditability.

        Implements: SPEC-11.25

        Returns:
            List of log entries
        """
        return self._adjustment_log.copy()


__all__ = [
    "ClassifierAdjustment",
    "Correction",
    "CorrectionAnalyzer",
    "CorrectionRecorder",
    "CorrectionType",
    "UserCorrectionLearner",
]
