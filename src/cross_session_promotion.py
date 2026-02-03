"""
Cross-session memory promotion for automatic tier advancement.

Implements: SPEC-09.10-09.15

Tracks memory access across sessions and automatically promotes
memories based on cross-session access patterns, successful outcomes,
and confidence levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PromotionReason(Enum):
    """Reason for promotion decision."""

    SESSION_USE_COUNT = "session_use_count"
    CROSS_SESSION_ACCESS = "cross_session_access"
    STALENESS = "staleness"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class PromotionConfig:
    """
    Configuration for cross-session promotion.

    Implements: SPEC-09.11, SPEC-09.12
    """

    min_sessions: int = 3
    min_confidence: float = 0.8
    task_to_session_uses: int = 2
    archive_staleness_days: int = 90


@dataclass
class AccessRecord:
    """
    Record of a memory access event.

    Implements: SPEC-09.10
    """

    node_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }


@dataclass
class PromotionCandidate:
    """
    Candidate for promotion evaluation.

    Implements: SPEC-09.11
    """

    node_id: str
    session_count: int
    access_count: int
    success_count: int
    confidence: float
    current_tier: str
    content: str | None = None
    metadata: dict[str, Any] | None = None
    days_since_access: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "session_count": self.session_count,
            "access_count": self.access_count,
            "success_count": self.success_count,
            "confidence": self.confidence,
            "current_tier": self.current_tier,
            "content": self.content,
            "metadata": self.metadata,
            "days_since_access": self.days_since_access,
        }


@dataclass
class PromotionDecision:
    """
    Decision about whether to promote a memory.

    Implements: SPEC-09.12
    """

    node_id: str
    should_promote: bool
    target_tier: str | None = None
    rejection_reason: str | None = None
    preserved_content: str | None = None
    preserved_metadata: dict[str, Any] | None = None
    new_metadata: dict[str, Any] = field(default_factory=dict)
    is_manual: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "should_promote": self.should_promote,
            "target_tier": self.target_tier,
            "rejection_reason": self.rejection_reason,
            "is_manual": self.is_manual,
        }


@dataclass
class PromotionLogEntry:
    """Entry in the promotion log."""

    node_id: str
    from_tier: str
    to_tier: str
    reason: PromotionReason
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "from_tier": self.from_tier,
            "to_tier": self.to_tier,
            "reason": self.reason.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class PromotionLog:
    """
    Log of promotion decisions for auditability.

    Implements: SPEC-09.14
    """

    def __init__(self) -> None:
        """Initialize promotion log."""
        self._entries: list[PromotionLogEntry] = []

    def record_decision(
        self,
        node_id: str,
        from_tier: str,
        to_tier: str,
        reason: PromotionReason,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a promotion decision.

        Implements: SPEC-09.14

        Args:
            node_id: Node being promoted
            from_tier: Source tier
            to_tier: Target tier
            reason: Reason for promotion
            details: Additional details
        """
        entry = PromotionLogEntry(
            node_id=node_id,
            from_tier=from_tier,
            to_tier=to_tier,
            reason=reason,
            details=details or {},
        )
        self._entries.append(entry)

    def get_entries(self) -> list[PromotionLogEntry]:
        """Get all log entries."""
        return self._entries.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entries": [e.to_dict() for e in self._entries],
        }


class CrossSessionTracker:
    """
    Track memory access across sessions.

    Implements: SPEC-09.10
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._records: dict[str, list[AccessRecord]] = {}

    def record_access(
        self,
        node_id: str,
        session_id: str,
        success: bool = False,
    ) -> None:
        """
        Record a memory access.

        Implements: SPEC-09.10

        Args:
            node_id: Node that was accessed
            session_id: Session in which access occurred
            success: Whether access was associated with success
        """
        if node_id not in self._records:
            self._records[node_id] = []

        record = AccessRecord(
            node_id=node_id,
            session_id=session_id,
            success=success,
        )
        self._records[node_id].append(record)

    def get_access_count(self, node_id: str) -> int:
        """Get total access count for a node."""
        if node_id not in self._records:
            return 0
        return len(self._records[node_id])

    def get_session_count(self, node_id: str) -> int:
        """
        Get count of distinct sessions that accessed a node.

        Implements: SPEC-09.11
        """
        if node_id not in self._records:
            return 0

        sessions = {r.session_id for r in self._records[node_id]}
        return len(sessions)

    def get_success_count(self, node_id: str) -> int:
        """Get count of successful accesses for a node."""
        if node_id not in self._records:
            return 0

        return sum(1 for r in self._records[node_id] if r.success)

    def get_access_records(self, node_id: str) -> list[AccessRecord]:
        """Get all access records for a node."""
        return self._records.get(node_id, []).copy()

    def to_candidate(
        self,
        node_id: str,
        confidence: float,
        current_tier: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromotionCandidate:
        """
        Convert tracker data to a promotion candidate.

        Args:
            node_id: Node ID
            confidence: Current confidence level
            current_tier: Current tier
            content: Optional content
            metadata: Optional metadata

        Returns:
            PromotionCandidate with tracker data
        """
        return PromotionCandidate(
            node_id=node_id,
            session_count=self.get_session_count(node_id),
            access_count=self.get_access_count(node_id),
            success_count=self.get_success_count(node_id),
            confidence=confidence,
            current_tier=current_tier,
            content=content,
            metadata=metadata,
        )


class PromotionCriteria:
    """
    Evaluate promotion criteria for candidates.

    Implements: SPEC-09.11
    """

    def __init__(self, config: PromotionConfig | None = None) -> None:
        """
        Initialize criteria evaluator.

        Args:
            config: Promotion configuration
        """
        self.config = config or PromotionConfig()

    def meets_session_threshold(self, session_count: int) -> bool:
        """
        Check if session count meets threshold.

        Implements: SPEC-09.11 - Accessed in 3+ distinct sessions

        Args:
            session_count: Number of distinct sessions

        Returns:
            True if threshold met
        """
        return session_count >= self.config.min_sessions

    def has_successful_outcomes(self, success_count: int) -> bool:
        """
        Check if there are successful outcomes.

        Implements: SPEC-09.11 - Associated with successful outcomes

        Args:
            success_count: Number of successful outcomes

        Returns:
            True if has successful outcomes
        """
        return success_count > 0

    def has_high_confidence(self, confidence: float) -> bool:
        """
        Check if confidence is high enough.

        Implements: SPEC-09.11 - High confidence maintained over time

        Args:
            confidence: Current confidence level

        Returns:
            True if confidence is high enough
        """
        return confidence >= self.config.min_confidence

    def evaluate(self, candidate: PromotionCandidate) -> bool:
        """
        Evaluate all criteria for a candidate.

        Implements: SPEC-09.11

        Args:
            candidate: Promotion candidate

        Returns:
            True if all criteria met
        """
        return (
            self.meets_session_threshold(candidate.session_count)
            and self.has_successful_outcomes(candidate.success_count)
            and self.has_high_confidence(candidate.confidence)
        )


class CrossSessionPromoter:
    """
    Promote memories based on cross-session access patterns.

    Implements: SPEC-09.10-09.15
    """

    def __init__(self, config: PromotionConfig | None = None) -> None:
        """
        Initialize promoter.

        Args:
            config: Promotion configuration
        """
        self.config = config or PromotionConfig()
        self.criteria = PromotionCriteria(self.config)
        self.log = PromotionLog()

    def evaluate_promotion(
        self,
        candidate: PromotionCandidate,
    ) -> PromotionDecision:
        """
        Evaluate whether a candidate should be promoted.

        Implements: SPEC-09.12

        Args:
            candidate: Promotion candidate

        Returns:
            PromotionDecision with promotion details
        """
        # Determine promotion based on current tier
        if candidate.current_tier == "task":
            return self._evaluate_task_promotion(candidate)
        elif candidate.current_tier == "session":
            return self._evaluate_session_promotion(candidate)
        elif candidate.current_tier == "longterm":
            return self._evaluate_longterm_promotion(candidate)
        else:
            return PromotionDecision(
                node_id=candidate.node_id,
                should_promote=False,
                rejection_reason=f"Unknown tier: {candidate.current_tier}",
            )

    def _evaluate_task_promotion(
        self,
        candidate: PromotionCandidate,
    ) -> PromotionDecision:
        """
        Evaluate task → session promotion.

        SPEC-09.12: task → session after 2+ uses in session
        """
        if candidate.access_count >= self.config.task_to_session_uses:
            decision = PromotionDecision(
                node_id=candidate.node_id,
                should_promote=True,
                target_tier="session",
                preserved_content=candidate.content,
                preserved_metadata=candidate.metadata,
                new_metadata={
                    "promoted_from": "task",
                    "promoted_at": datetime.now().isoformat(),
                    "promotion_reason": "session_use_count",
                },
            )

            self.log.record_decision(
                node_id=candidate.node_id,
                from_tier="task",
                to_tier="session",
                reason=PromotionReason.SESSION_USE_COUNT,
                details={"access_count": candidate.access_count},
            )

            return decision

        return PromotionDecision(
            node_id=candidate.node_id,
            should_promote=False,
            rejection_reason=f"Access count {candidate.access_count} < {self.config.task_to_session_uses}",
        )

    def _evaluate_session_promotion(
        self,
        candidate: PromotionCandidate,
    ) -> PromotionDecision:
        """
        Evaluate session → longterm promotion.

        SPEC-09.12: session → longterm after 3+ cross-session accesses
        """
        if self.criteria.evaluate(candidate):
            decision = PromotionDecision(
                node_id=candidate.node_id,
                should_promote=True,
                target_tier="longterm",
                preserved_content=candidate.content,
                preserved_metadata=candidate.metadata,
                new_metadata={
                    "promoted_from": "session",
                    "promoted_at": datetime.now().isoformat(),
                    "promotion_reason": "cross_session_access",
                    "session_count": candidate.session_count,
                },
            )

            self.log.record_decision(
                node_id=candidate.node_id,
                from_tier="session",
                to_tier="longterm",
                reason=PromotionReason.CROSS_SESSION_ACCESS,
                details={
                    "session_count": candidate.session_count,
                    "success_count": candidate.success_count,
                    "confidence": candidate.confidence,
                },
            )

            return decision

        # Build rejection reason
        reasons = []
        if not self.criteria.meets_session_threshold(candidate.session_count):
            reasons.append(f"session_count {candidate.session_count} < {self.config.min_sessions}")
        if not self.criteria.has_successful_outcomes(candidate.success_count):
            reasons.append("no successful outcomes")
        if not self.criteria.has_high_confidence(candidate.confidence):
            reasons.append(f"confidence {candidate.confidence} < {self.config.min_confidence}")

        return PromotionDecision(
            node_id=candidate.node_id,
            should_promote=False,
            rejection_reason="; ".join(reasons),
        )

    def _evaluate_longterm_promotion(
        self,
        candidate: PromotionCandidate,
    ) -> PromotionDecision:
        """
        Evaluate longterm → archive promotion.

        SPEC-09.12: longterm → archive based on staleness (>90 days unused)
        """
        if candidate.days_since_access > self.config.archive_staleness_days:
            decision = PromotionDecision(
                node_id=candidate.node_id,
                should_promote=True,
                target_tier="archive",
                preserved_content=candidate.content,
                preserved_metadata=candidate.metadata,
                new_metadata={
                    "promoted_from": "longterm",
                    "promoted_at": datetime.now().isoformat(),
                    "promotion_reason": "staleness",
                    "days_since_access": candidate.days_since_access,
                },
            )

            self.log.record_decision(
                node_id=candidate.node_id,
                from_tier="longterm",
                to_tier="archive",
                reason=PromotionReason.STALENESS,
                details={"days_since_access": candidate.days_since_access},
            )

            return decision

        return PromotionDecision(
            node_id=candidate.node_id,
            should_promote=False,
            rejection_reason=f"days_since_access {candidate.days_since_access} <= {self.config.archive_staleness_days}",
        )

    def manual_promote(
        self,
        node_id: str,
        from_tier: str,
        to_tier: str,
        reason: str,
    ) -> PromotionDecision:
        """
        Manually override promotion criteria.

        Implements: SPEC-09.15

        Args:
            node_id: Node to promote
            from_tier: Source tier
            to_tier: Target tier
            reason: Reason for manual override

        Returns:
            PromotionDecision for manual promotion
        """
        decision = PromotionDecision(
            node_id=node_id,
            should_promote=True,
            target_tier=to_tier,
            is_manual=True,
            new_metadata={
                "promoted_from": from_tier,
                "promoted_at": datetime.now().isoformat(),
                "promotion_reason": "manual_override",
                "manual_reason": reason,
            },
        )

        self.log.record_decision(
            node_id=node_id,
            from_tier=from_tier,
            to_tier=to_tier,
            reason=PromotionReason.MANUAL_OVERRIDE,
            details={"manual_reason": reason},
        )

        return decision


__all__ = [
    "AccessRecord",
    "CrossSessionPromoter",
    "CrossSessionTracker",
    "PromotionCandidate",
    "PromotionConfig",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionLog",
    "PromotionReason",
]
