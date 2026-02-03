"""
Tests for verification feedback storage and processing.

Implements: SPEC-16.37 User feedback for verification
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from src.epistemic.verification_feedback import (
    FeedbackStatistics,
    FeedbackStore,
    FeedbackType,
    VerificationFeedback,
    record_feedback,
)


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_feedback_type_values(self) -> None:
        """Test that all feedback types have correct values."""
        assert FeedbackType.CORRECT.value == "correct"
        assert FeedbackType.FALSE_POSITIVE.value == "false_positive"
        assert FeedbackType.FALSE_NEGATIVE.value == "false_negative"
        assert FeedbackType.INCORRECT.value == "incorrect"

    def test_feedback_type_from_string(self) -> None:
        """Test creating FeedbackType from string."""
        assert FeedbackType("correct") == FeedbackType.CORRECT
        assert FeedbackType("false_positive") == FeedbackType.FALSE_POSITIVE
        assert FeedbackType("false_negative") == FeedbackType.FALSE_NEGATIVE
        assert FeedbackType("incorrect") == FeedbackType.INCORRECT

    def test_invalid_feedback_type(self) -> None:
        """Test that invalid feedback type raises error."""
        with pytest.raises(ValueError):
            FeedbackType("invalid")


class TestVerificationFeedback:
    """Tests for VerificationFeedback dataclass."""

    def test_create_feedback(self) -> None:
        """Test creating a feedback record."""
        feedback = VerificationFeedback(
            id="fb-1",
            claim_id="c1",
            claim_text="The function returns an integer",
            feedback_type=FeedbackType.CORRECT,
        )
        assert feedback.id == "fb-1"
        assert feedback.claim_id == "c1"
        assert feedback.feedback_type == FeedbackType.CORRECT
        assert feedback.original_flag_reason is None
        assert feedback.user_note is None

    def test_feedback_with_all_fields(self) -> None:
        """Test creating feedback with all optional fields."""
        ts = datetime.now()
        feedback = VerificationFeedback(
            id="fb-2",
            claim_id="c2",
            claim_text="The API returns JSON",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            original_flag_reason="unsupported",
            user_note="The claim is actually correct",
            timestamp=ts,
            response_id="resp-1",
            context_hash="abc123",
        )
        assert feedback.original_flag_reason == "unsupported"
        assert feedback.user_note == "The claim is actually correct"
        assert feedback.timestamp == ts
        assert feedback.response_id == "resp-1"
        assert feedback.context_hash == "abc123"


class TestFeedbackStatistics:
    """Tests for FeedbackStatistics dataclass."""

    def test_empty_statistics(self) -> None:
        """Test statistics with no feedback."""
        stats = FeedbackStatistics()
        assert stats.total_feedback == 0
        assert stats.accuracy_rate == 1.0  # Default to 100% when no data
        assert stats.false_positive_rate == 0.0
        assert stats.false_negative_rate == 0.0

    def test_statistics_calculations(self) -> None:
        """Test statistics calculations."""
        stats = FeedbackStatistics(
            total_feedback=100,
            correct_count=80,
            false_positive_count=10,
            false_negative_count=5,
            incorrect_count=5,
        )
        assert stats.accuracy_rate == 0.8
        assert stats.false_positive_rate == 0.1
        assert stats.false_negative_rate == 0.05


class TestFeedbackStore:
    """Tests for FeedbackStore class."""

    @pytest.fixture
    def store(self) -> FeedbackStore:
        """Create a temporary feedback store."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = FeedbackStore(path)
        yield store
        os.unlink(path)

    def test_add_and_get_feedback(self, store: FeedbackStore) -> None:
        """Test adding and retrieving feedback."""
        feedback_id = store.add_feedback(
            claim_id="c1",
            claim_text="The function is pure",
            feedback_type=FeedbackType.CORRECT,
        )

        feedback = store.get_feedback(feedback_id)
        assert feedback is not None
        assert feedback.claim_id == "c1"
        assert feedback.claim_text == "The function is pure"
        assert feedback.feedback_type == FeedbackType.CORRECT

    def test_add_feedback_with_string_type(self, store: FeedbackStore) -> None:
        """Test adding feedback with string feedback type."""
        feedback_id = store.add_feedback(
            claim_id="c2",
            claim_text="The class is immutable",
            feedback_type="false_positive",  # String instead of enum
        )

        feedback = store.get_feedback(feedback_id)
        assert feedback is not None
        assert feedback.feedback_type == FeedbackType.FALSE_POSITIVE

    def test_get_nonexistent_feedback(self, store: FeedbackStore) -> None:
        """Test getting feedback that doesn't exist."""
        feedback = store.get_feedback("nonexistent-id")
        assert feedback is None

    def test_get_feedback_for_claim(self, store: FeedbackStore) -> None:
        """Test getting all feedback for a specific claim."""
        # Add multiple feedback entries for same claim
        store.add_feedback(
            claim_id="c1",
            claim_text="Claim text",
            feedback_type=FeedbackType.CORRECT,
        )
        store.add_feedback(
            claim_id="c1",
            claim_text="Claim text",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            user_note="Changed my mind",
        )
        store.add_feedback(
            claim_id="c2",
            claim_text="Other claim",
            feedback_type=FeedbackType.CORRECT,
        )

        feedback_list = store.get_feedback_for_claim("c1")
        assert len(feedback_list) == 2
        # Most recent first
        assert feedback_list[0].feedback_type == FeedbackType.FALSE_POSITIVE

    def test_get_recent_feedback(self, store: FeedbackStore) -> None:
        """Test getting recent feedback entries."""
        for i in range(5):
            store.add_feedback(
                claim_id=f"c{i}",
                claim_text=f"Claim {i}",
                feedback_type=FeedbackType.CORRECT,
            )

        recent = store.get_recent_feedback(limit=3)
        assert len(recent) == 3

    def test_get_recent_feedback_filtered(self, store: FeedbackStore) -> None:
        """Test getting recent feedback filtered by type."""
        store.add_feedback("c1", "Claim 1", FeedbackType.CORRECT)
        store.add_feedback("c2", "Claim 2", FeedbackType.FALSE_POSITIVE)
        store.add_feedback("c3", "Claim 3", FeedbackType.CORRECT)

        fp_feedback = store.get_recent_feedback(
            limit=10,
            feedback_type=FeedbackType.FALSE_POSITIVE,
        )
        assert len(fp_feedback) == 1
        assert fp_feedback[0].claim_id == "c2"

    def test_get_statistics(self, store: FeedbackStore) -> None:
        """Test getting aggregated statistics."""
        # Add various feedback types
        for _ in range(8):
            store.add_feedback("c", "Claim", FeedbackType.CORRECT)
        for _ in range(2):
            store.add_feedback("c", "Claim", FeedbackType.FALSE_POSITIVE)

        stats = store.get_statistics()
        assert stats.total_feedback == 10
        assert stats.correct_count == 8
        assert stats.false_positive_count == 2
        assert stats.accuracy_rate == 0.8
        assert stats.false_positive_rate == 0.2

    def test_get_statistics_with_timestamp(self, store: FeedbackStore) -> None:
        """Test getting statistics filtered by timestamp."""
        # Add some feedback
        store.add_feedback("c1", "Claim 1", FeedbackType.CORRECT)
        store.add_feedback("c2", "Claim 2", FeedbackType.FALSE_POSITIVE)

        # Get stats from future timestamp (should be empty)
        future = datetime.now() + timedelta(hours=1)
        stats = store.get_statistics(since_timestamp=future)
        assert stats.total_feedback == 0

    def test_get_similar_claim_feedback(self, store: FeedbackStore) -> None:
        """Test finding feedback for similar claims."""
        store.add_feedback(
            claim_id="c1",
            claim_text="The function returns an integer value",
            feedback_type=FeedbackType.CORRECT,
        )
        store.add_feedback(
            claim_id="c2",
            claim_text="The method processes string data",
            feedback_type=FeedbackType.FALSE_POSITIVE,
        )

        # Search for similar to "function returns integer"
        similar = store.get_similar_claim_feedback("function returns integer")
        assert len(similar) >= 1
        assert any(fb.claim_id == "c1" for fb in similar)

    def test_get_calibration_data(self, store: FeedbackStore) -> None:
        """Test getting calibration recommendations."""
        # Add enough data to generate recommendations
        for _ in range(8):
            store.add_feedback("c", "Claim", FeedbackType.CORRECT)
        for _ in range(3):
            store.add_feedback("c", "Claim", FeedbackType.FALSE_POSITIVE)

        calibration = store.get_calibration_data()
        assert calibration["total_feedback"] == 11
        assert "accuracy_rate" in calibration
        assert "recommendations" in calibration

    def test_calibration_high_false_positive(self, store: FeedbackStore) -> None:
        """Test calibration recommends raising threshold on high FP rate."""
        # 50% false positive rate
        for _ in range(5):
            store.add_feedback("c", "Claim", FeedbackType.CORRECT)
        for _ in range(5):
            store.add_feedback("c", "Claim", FeedbackType.FALSE_POSITIVE)

        calibration = store.get_calibration_data()
        assert any(r["type"] == "raise_threshold" for r in calibration["recommendations"])


class TestRecordFeedback:
    """Tests for record_feedback convenience function."""

    @pytest.fixture
    def db_path(self) -> str:
        """Create a temporary database path."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        os.unlink(path)

    def test_record_correct_unflagged(self, db_path: str) -> None:
        """Test recording correct claim that wasn't flagged."""
        feedback_id = record_feedback(
            claim_id="c1",
            claim_text="The API is RESTful",
            is_correct=True,
            was_flagged=False,
            db_path=db_path,
        )

        store = FeedbackStore(db_path)
        feedback = store.get_feedback(feedback_id)
        assert feedback is not None
        assert feedback.feedback_type == FeedbackType.CORRECT

    def test_record_correct_but_flagged(self, db_path: str) -> None:
        """Test recording correct claim that was wrongly flagged."""
        feedback_id = record_feedback(
            claim_id="c1",
            claim_text="The API is RESTful",
            is_correct=True,
            was_flagged=True,
            flag_reason="unsupported",
            db_path=db_path,
        )

        store = FeedbackStore(db_path)
        feedback = store.get_feedback(feedback_id)
        assert feedback is not None
        assert feedback.feedback_type == FeedbackType.FALSE_POSITIVE
        assert feedback.original_flag_reason == "unsupported"

    def test_record_incorrect_not_flagged(self, db_path: str) -> None:
        """Test recording incorrect claim that wasn't flagged."""
        feedback_id = record_feedback(
            claim_id="c1",
            claim_text="The API returns XML",
            is_correct=False,
            was_flagged=False,
            db_path=db_path,
        )

        store = FeedbackStore(db_path)
        feedback = store.get_feedback(feedback_id)
        assert feedback is not None
        assert feedback.feedback_type == FeedbackType.FALSE_NEGATIVE

    def test_record_incorrect_was_flagged(self, db_path: str) -> None:
        """Test recording incorrect claim that was correctly flagged."""
        feedback_id = record_feedback(
            claim_id="c1",
            claim_text="The API returns XML",
            is_correct=False,
            was_flagged=True,
            flag_reason="contradiction",
            db_path=db_path,
        )

        store = FeedbackStore(db_path)
        feedback = store.get_feedback(feedback_id)
        assert feedback is not None
        # Flag was correct
        assert feedback.feedback_type == FeedbackType.CORRECT

    def test_record_with_user_note(self, db_path: str) -> None:
        """Test recording feedback with user note."""
        feedback_id = record_feedback(
            claim_id="c1",
            claim_text="The function is O(n)",
            is_correct=False,
            was_flagged=False,
            user_note="Actually O(n^2) due to nested loops",
            db_path=db_path,
        )

        store = FeedbackStore(db_path)
        feedback = store.get_feedback(feedback_id)
        assert feedback is not None
        assert feedback.user_note == "Actually O(n^2) due to nested loops"


class TestFeedbackStoreEdgeCases:
    """Edge case tests for FeedbackStore."""

    @pytest.fixture
    def store(self) -> FeedbackStore:
        """Create a temporary feedback store."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = FeedbackStore(path)
        yield store
        os.unlink(path)

    def test_empty_statistics(self, store: FeedbackStore) -> None:
        """Test statistics with no feedback entries."""
        stats = store.get_statistics()
        assert stats.total_feedback == 0
        assert stats.accuracy_rate == 1.0  # Default

    def test_similar_claims_empty(self, store: FeedbackStore) -> None:
        """Test finding similar claims when store is empty."""
        similar = store.get_similar_claim_feedback("some claim text")
        assert similar == []

    def test_similar_claims_no_overlap(self, store: FeedbackStore) -> None:
        """Test finding similar claims with no word overlap."""
        store.add_feedback("c1", "alpha beta gamma", FeedbackType.CORRECT)
        similar = store.get_similar_claim_feedback("delta epsilon zeta")
        assert similar == []

    def test_feedback_with_special_characters(self, store: FeedbackStore) -> None:
        """Test feedback with special characters in text."""
        feedback_id = store.add_feedback(
            claim_id="c1",
            claim_text="The regex is /^[a-z]+$/i",
            feedback_type=FeedbackType.CORRECT,
            user_note="Contains 'special' \"characters\"",
        )

        feedback = store.get_feedback(feedback_id)
        assert feedback is not None
        assert "regex" in feedback.claim_text
        assert "special" in feedback.user_note

    def test_calibration_insufficient_data(self, store: FeedbackStore) -> None:
        """Test calibration with insufficient data."""
        # Only 5 entries, below threshold for recommendations
        for _ in range(5):
            store.add_feedback("c", "Claim", FeedbackType.FALSE_POSITIVE)

        calibration = store.get_calibration_data()
        # Should have no recommendations with < 10 entries
        assert calibration["recommendations"] == []

    def test_multiple_stores_same_db(self) -> None:
        """Test multiple store instances accessing same database."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            store1 = FeedbackStore(path)
            store2 = FeedbackStore(path)

            # Add via store1
            feedback_id = store1.add_feedback("c1", "Claim 1", FeedbackType.CORRECT)

            # Read via store2
            feedback = store2.get_feedback(feedback_id)
            assert feedback is not None
            assert feedback.claim_id == "c1"
        finally:
            os.unlink(path)
