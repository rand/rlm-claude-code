"""
Verification feedback storage and processing.

Implements: SPEC-16.37 User feedback for verification

Allows users to mark verification results as correct/incorrect,
stores feedback in memory, and uses it to calibrate future verification.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class FeedbackType(str, Enum):
    """Type of user feedback on a verification result."""

    CORRECT = "correct"  # Verification was accurate
    FALSE_POSITIVE = "false_positive"  # Claim flagged but was actually correct
    FALSE_NEGATIVE = "false_negative"  # Claim not flagged but was incorrect
    INCORRECT = "incorrect"  # General incorrectness


@dataclass
class VerificationFeedback:
    """
    User feedback on a verification result.

    Implements: SPEC-16.37

    Attributes:
        id: Unique feedback ID
        claim_id: ID of the claim this feedback is for
        claim_text: Text of the claim
        feedback_type: Type of feedback (correct/incorrect/false_positive/false_negative)
        original_flag_reason: Original reason the claim was flagged (if any)
        user_note: Optional user explanation
        timestamp: When feedback was provided
        response_id: ID of the response containing this claim
        context_hash: Hash of evidence context for matching similar situations
    """

    id: str
    claim_id: str
    claim_text: str
    feedback_type: FeedbackType
    original_flag_reason: str | None = None
    user_note: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    response_id: str | None = None
    context_hash: str | None = None


@dataclass
class FeedbackStatistics:
    """
    Aggregated statistics from verification feedback.

    Used for calibration and reporting.
    """

    total_feedback: int = 0
    correct_count: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0
    incorrect_count: int = 0

    @property
    def accuracy_rate(self) -> float:
        """Percentage of verifications that were correct."""
        if self.total_feedback == 0:
            return 1.0
        return self.correct_count / self.total_feedback

    @property
    def false_positive_rate(self) -> float:
        """Rate of false positives (flagged but correct)."""
        if self.total_feedback == 0:
            return 0.0
        return self.false_positive_count / self.total_feedback

    @property
    def false_negative_rate(self) -> float:
        """Rate of false negatives (not flagged but incorrect)."""
        if self.total_feedback == 0:
            return 0.0
        return self.false_negative_count / self.total_feedback


# Schema for verification feedback table
FEEDBACK_SCHEMA_SQL = """
-- Verification feedback table (SPEC-16.37)
CREATE TABLE IF NOT EXISTS verification_feedback (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    feedback_type TEXT NOT NULL CHECK(feedback_type IN ('correct', 'false_positive', 'false_negative', 'incorrect')),
    original_flag_reason TEXT,
    user_note TEXT,
    timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    response_id TEXT,
    context_hash TEXT
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_feedback_claim ON verification_feedback(claim_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON verification_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON verification_feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_context_hash ON verification_feedback(context_hash);
"""


class FeedbackStore:
    """
    Persistent storage for verification feedback.

    Implements: SPEC-16.37

    Uses SQLite for storage, supports calibration queries.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize feedback store.

        Args:
            db_path: Path to SQLite database. If None, uses default.
        """
        if db_path is None:
            db_path = str(Path.home() / ".claude" / "rlm-verification-feedback.db")

        self.db_path = db_path
        self._ensure_directory()
        self._init_database()

    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self) -> None:
        """Initialize database with schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(FEEDBACK_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_feedback(
        self,
        claim_id: str,
        claim_text: str,
        feedback_type: FeedbackType | str,
        original_flag_reason: str | None = None,
        user_note: str | None = None,
        response_id: str | None = None,
        context_hash: str | None = None,
    ) -> str:
        """
        Add user feedback for a verification result.

        Args:
            claim_id: ID of the claim
            claim_text: Text of the claim
            feedback_type: Type of feedback
            original_flag_reason: Original flag reason (if claim was flagged)
            user_note: Optional user explanation
            response_id: ID of response containing this claim
            context_hash: Hash for matching similar contexts

        Returns:
            Feedback ID
        """
        # Convert string to enum if needed
        if isinstance(feedback_type, str):
            feedback_type = FeedbackType(feedback_type)

        feedback_id = str(uuid.uuid4())
        timestamp_ms = int(datetime.now().timestamp() * 1000)

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO verification_feedback
                (id, claim_id, claim_text, feedback_type, original_flag_reason,
                 user_note, timestamp, response_id, context_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    claim_id,
                    claim_text,
                    feedback_type.value,
                    original_flag_reason,
                    user_note,
                    timestamp_ms,
                    response_id,
                    context_hash,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return feedback_id

    def get_feedback(self, feedback_id: str) -> VerificationFeedback | None:
        """Get feedback by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM verification_feedback WHERE id = ?",
                (feedback_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_feedback(row)
        finally:
            conn.close()

    def get_feedback_for_claim(self, claim_id: str) -> list[VerificationFeedback]:
        """Get all feedback for a specific claim."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM verification_feedback WHERE claim_id = ? ORDER BY timestamp DESC",
                (claim_id,),
            )
            return [self._row_to_feedback(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_recent_feedback(
        self,
        limit: int = 100,
        feedback_type: FeedbackType | None = None,
    ) -> list[VerificationFeedback]:
        """
        Get recent feedback entries.

        Args:
            limit: Maximum entries to return
            feedback_type: Optional filter by type

        Returns:
            List of feedback entries
        """
        conn = self._get_connection()
        try:
            if feedback_type:
                cursor = conn.execute(
                    """
                    SELECT * FROM verification_feedback
                    WHERE feedback_type = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (feedback_type.value, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM verification_feedback ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            return [self._row_to_feedback(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_statistics(
        self,
        since_timestamp: datetime | None = None,
    ) -> FeedbackStatistics:
        """
        Get aggregated feedback statistics.

        Args:
            since_timestamp: Only include feedback after this time

        Returns:
            FeedbackStatistics object
        """
        conn = self._get_connection()
        try:
            if since_timestamp:
                ts_ms = int(since_timestamp.timestamp() * 1000)
                cursor = conn.execute(
                    """
                    SELECT feedback_type, COUNT(*) as count
                    FROM verification_feedback
                    WHERE timestamp >= ?
                    GROUP BY feedback_type
                    """,
                    (ts_ms,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT feedback_type, COUNT(*) as count
                    FROM verification_feedback
                    GROUP BY feedback_type
                    """
                )

            stats = FeedbackStatistics()
            for row in cursor.fetchall():
                ft = row["feedback_type"]
                count = row["count"]
                stats.total_feedback += count

                if ft == FeedbackType.CORRECT.value:
                    stats.correct_count = count
                elif ft == FeedbackType.FALSE_POSITIVE.value:
                    stats.false_positive_count = count
                elif ft == FeedbackType.FALSE_NEGATIVE.value:
                    stats.false_negative_count = count
                elif ft == FeedbackType.INCORRECT.value:
                    stats.incorrect_count = count

            return stats
        finally:
            conn.close()

    def get_similar_claim_feedback(
        self,
        claim_text: str,
        limit: int = 10,
    ) -> list[VerificationFeedback]:
        """
        Find feedback for similar claims using text search.

        Args:
            claim_text: Text to find similar claims for
            limit: Maximum results

        Returns:
            List of feedback for similar claims
        """
        # Simple word overlap for similarity (could be enhanced with embeddings)
        words = set(claim_text.lower().split())
        if not words:
            return []

        conn = self._get_connection()
        try:
            # Get all feedback and filter by word overlap
            cursor = conn.execute(
                "SELECT * FROM verification_feedback ORDER BY timestamp DESC LIMIT 1000"
            )

            results = []
            for row in cursor.fetchall():
                row_words = set(row["claim_text"].lower().split())
                overlap = len(words & row_words)
                if overlap >= min(2, len(words)):  # At least 2 words overlap
                    results.append((overlap, self._row_to_feedback(row)))

            # Sort by overlap and return top results
            results.sort(key=lambda x: x[0], reverse=True)
            return [r[1] for r in results[:limit]]
        finally:
            conn.close()

    def get_calibration_data(self) -> dict[str, Any]:
        """
        Get calibration data for adjusting verification thresholds.

        Returns:
            Dict with calibration recommendations based on feedback
        """
        stats = self.get_statistics()

        recommendations: list[dict[str, str]] = []

        # Generate calibration recommendations
        if stats.total_feedback >= 10:  # Need sufficient data
            if stats.false_positive_rate > 0.2:
                recommendations.append(
                    {
                        "type": "raise_threshold",
                        "reason": f"High false positive rate ({stats.false_positive_rate:.1%})",
                        "suggestion": "Consider raising support_threshold to reduce false positives",
                    }
                )

            if stats.false_negative_rate > 0.1:
                recommendations.append(
                    {
                        "type": "lower_threshold",
                        "reason": f"Elevated false negative rate ({stats.false_negative_rate:.1%})",
                        "suggestion": "Consider lowering support_threshold to catch more issues",
                    }
                )

        return {
            "total_feedback": stats.total_feedback,
            "accuracy_rate": stats.accuracy_rate,
            "false_positive_rate": stats.false_positive_rate,
            "false_negative_rate": stats.false_negative_rate,
            "recommendations": recommendations,
        }

    def _row_to_feedback(self, row: sqlite3.Row) -> VerificationFeedback:
        """Convert database row to VerificationFeedback."""
        return VerificationFeedback(
            id=row["id"],
            claim_id=row["claim_id"],
            claim_text=row["claim_text"],
            feedback_type=FeedbackType(row["feedback_type"]),
            original_flag_reason=row["original_flag_reason"],
            user_note=row["user_note"],
            timestamp=datetime.fromtimestamp(row["timestamp"] / 1000),
            response_id=row["response_id"],
            context_hash=row["context_hash"],
        )


def record_feedback(
    claim_id: str,
    claim_text: str,
    is_correct: bool,
    was_flagged: bool = False,
    flag_reason: str | None = None,
    user_note: str | None = None,
    db_path: str | None = None,
) -> str:
    """
    Convenience function to record verification feedback.

    Implements: SPEC-16.37 `/verify feedback` command support

    Args:
        claim_id: ID of the claim
        claim_text: Text of the claim
        is_correct: Whether the claim was actually correct
        was_flagged: Whether verification flagged this claim
        flag_reason: Original flag reason (if flagged)
        user_note: Optional user explanation

    Returns:
        Feedback ID
    """
    store = FeedbackStore(db_path)

    # Determine feedback type based on correctness and flag status
    if is_correct:
        if was_flagged:
            feedback_type = FeedbackType.FALSE_POSITIVE
        else:
            feedback_type = FeedbackType.CORRECT
    else:
        if was_flagged:
            feedback_type = FeedbackType.CORRECT  # Flag was right
        else:
            feedback_type = FeedbackType.FALSE_NEGATIVE

    return store.add_feedback(
        claim_id=claim_id,
        claim_text=claim_text,
        feedback_type=feedback_type,
        original_flag_reason=flag_reason,
        user_note=user_note,
    )


__all__ = [
    "FeedbackType",
    "VerificationFeedback",
    "FeedbackStatistics",
    "FeedbackStore",
    "record_feedback",
]
