"""
Tests for cross-session memory promotion (SPEC-09.10-09.15).

Tests cover:
- Cross-session access tracking
- Promotion criteria evaluation
- Automatic tier promotion
- Metadata preservation
- Promotion logging
- Manual promotion override
"""

from src.cross_session_promotion import (
    AccessRecord,
    CrossSessionPromoter,
    CrossSessionTracker,
    PromotionCandidate,
    PromotionConfig,
    PromotionCriteria,
    PromotionDecision,
    PromotionLog,
    PromotionReason,
)


class TestCrossSessionTracker:
    """Tests for cross-session access tracking (SPEC-09.10)."""

    def test_tracks_memory_access(self):
        """SPEC-09.10: Track memory access across sessions."""
        tracker = CrossSessionTracker()

        tracker.record_access(
            node_id="node-1",
            session_id="session-a",
        )

        assert tracker.get_access_count("node-1") == 1

    def test_tracks_multiple_sessions(self):
        """Track accesses from distinct sessions."""
        tracker = CrossSessionTracker()

        tracker.record_access("node-1", "session-a")
        tracker.record_access("node-1", "session-b")
        tracker.record_access("node-1", "session-c")

        assert tracker.get_session_count("node-1") == 3

    def test_deduplicates_same_session_accesses(self):
        """Multiple accesses from same session count as one session."""
        tracker = CrossSessionTracker()

        tracker.record_access("node-1", "session-a")
        tracker.record_access("node-1", "session-a")
        tracker.record_access("node-1", "session-a")

        assert tracker.get_session_count("node-1") == 1
        assert tracker.get_access_count("node-1") == 3

    def test_returns_zero_for_untracked_node(self):
        """Untracked nodes return zero counts."""
        tracker = CrossSessionTracker()

        assert tracker.get_access_count("unknown") == 0
        assert tracker.get_session_count("unknown") == 0

    def test_get_access_records(self):
        """Get all access records for a node."""
        tracker = CrossSessionTracker()

        tracker.record_access("node-1", "session-a")
        tracker.record_access("node-1", "session-b")

        records = tracker.get_access_records("node-1")

        assert len(records) == 2
        assert all(isinstance(r, AccessRecord) for r in records)


class TestPromotionCriteria:
    """Tests for promotion criteria evaluation (SPEC-09.11)."""

    def test_three_plus_sessions_criterion(self):
        """SPEC-09.11: Accessed in 3+ distinct sessions."""
        criteria = PromotionCriteria()

        # Not enough sessions
        assert not criteria.meets_session_threshold(session_count=2)

        # Meets threshold
        assert criteria.meets_session_threshold(session_count=3)
        assert criteria.meets_session_threshold(session_count=5)

    def test_successful_outcomes_criterion(self):
        """SPEC-09.11: Associated with successful outcomes."""
        criteria = PromotionCriteria()

        # No successful outcomes
        assert not criteria.has_successful_outcomes(success_count=0)

        # Has successful outcomes
        assert criteria.has_successful_outcomes(success_count=1)
        assert criteria.has_successful_outcomes(success_count=5)

    def test_high_confidence_criterion(self):
        """SPEC-09.11: High confidence maintained over time."""
        criteria = PromotionCriteria()

        # Low confidence
        assert not criteria.has_high_confidence(confidence=0.5)
        assert not criteria.has_high_confidence(confidence=0.7)

        # High confidence (default threshold 0.8)
        assert criteria.has_high_confidence(confidence=0.8)
        assert criteria.has_high_confidence(confidence=0.95)

    def test_configurable_thresholds(self):
        """Thresholds should be configurable."""
        config = PromotionConfig(
            min_sessions=5,
            min_confidence=0.9,
        )
        criteria = PromotionCriteria(config=config)

        assert not criteria.meets_session_threshold(session_count=4)
        assert criteria.meets_session_threshold(session_count=5)

        assert not criteria.has_high_confidence(confidence=0.85)
        assert criteria.has_high_confidence(confidence=0.9)

    def test_evaluate_all_criteria(self):
        """Evaluate all criteria at once."""
        criteria = PromotionCriteria()

        # All criteria met
        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
        )

        assert criteria.evaluate(candidate)

    def test_reject_if_any_criterion_fails(self):
        """Reject if any criterion fails."""
        criteria = PromotionCriteria()

        # Session count too low
        candidate1 = PromotionCandidate(
            node_id="node-1",
            session_count=2,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
        )
        assert not criteria.evaluate(candidate1)

        # No successful outcomes
        candidate2 = PromotionCandidate(
            node_id="node-2",
            session_count=3,
            access_count=5,
            success_count=0,
            confidence=0.85,
            current_tier="session",
        )
        assert not criteria.evaluate(candidate2)


class TestAutomaticPromotion:
    """Tests for automatic tier promotion (SPEC-09.12)."""

    def test_task_to_session_promotion(self):
        """SPEC-09.12: task → session after 2+ uses in session."""
        promoter = CrossSessionPromoter()

        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=1,
            access_count=2,
            success_count=1,
            confidence=0.8,
            current_tier="task",
        )

        decision = promoter.evaluate_promotion(candidate)

        assert decision.should_promote
        assert decision.target_tier == "session"

    def test_session_to_longterm_promotion(self):
        """SPEC-09.12: session → longterm after 3+ cross-session accesses."""
        promoter = CrossSessionPromoter()

        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
        )

        decision = promoter.evaluate_promotion(candidate)

        assert decision.should_promote
        assert decision.target_tier == "longterm"

    def test_longterm_to_archive_promotion(self):
        """SPEC-09.12: longterm → archive based on staleness (>90 days)."""
        promoter = CrossSessionPromoter()

        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.3,
            current_tier="longterm",
            days_since_access=91,
        )

        decision = promoter.evaluate_promotion(candidate)

        assert decision.should_promote
        assert decision.target_tier == "archive"

    def test_no_promotion_if_criteria_not_met(self):
        """No promotion if criteria not met."""
        promoter = CrossSessionPromoter()

        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=1,
            access_count=1,
            success_count=0,
            confidence=0.5,
            current_tier="task",
        )

        decision = promoter.evaluate_promotion(candidate)

        assert not decision.should_promote


class TestMetadataPreservation:
    """Tests for metadata preservation (SPEC-09.13)."""

    def test_preserves_original_content(self):
        """SPEC-09.13: Preserve original memory content."""
        promoter = CrossSessionPromoter()

        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
            content="Original content here",
            metadata={"key": "value"},
        )

        decision = promoter.evaluate_promotion(candidate)

        assert decision.preserved_content == "Original content here"

    def test_preserves_metadata(self):
        """SPEC-09.13: Preserve original metadata."""
        promoter = CrossSessionPromoter()

        original_metadata = {
            "created_by": "user-123",
            "source": "conversation",
            "tags": ["important"],
        }

        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
            metadata=original_metadata,
        )

        decision = promoter.evaluate_promotion(candidate)

        assert decision.preserved_metadata == original_metadata

    def test_adds_promotion_metadata(self):
        """Promotion should add tracking metadata."""
        promoter = CrossSessionPromoter()

        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
            metadata={},
        )

        decision = promoter.evaluate_promotion(candidate)

        assert "promoted_from" in decision.new_metadata
        assert "promoted_at" in decision.new_metadata


class TestPromotionLogging:
    """Tests for promotion logging (SPEC-09.14)."""

    def test_logs_promotion_decision(self):
        """SPEC-09.14: Log promotion decisions for auditability."""
        log = PromotionLog()

        log.record_decision(
            node_id="node-1",
            from_tier="session",
            to_tier="longterm",
            reason=PromotionReason.CROSS_SESSION_ACCESS,
        )

        entries = log.get_entries()

        assert len(entries) == 1
        assert entries[0].node_id == "node-1"
        assert entries[0].from_tier == "session"
        assert entries[0].to_tier == "longterm"

    def test_logs_include_timestamp(self):
        """Logs should include timestamp."""
        log = PromotionLog()

        log.record_decision(
            node_id="node-1",
            from_tier="task",
            to_tier="session",
            reason=PromotionReason.SESSION_USE_COUNT,
        )

        entries = log.get_entries()

        assert entries[0].timestamp is not None

    def test_logs_include_reason(self):
        """Logs should include promotion reason."""
        log = PromotionLog()

        log.record_decision(
            node_id="node-1",
            from_tier="session",
            to_tier="longterm",
            reason=PromotionReason.CROSS_SESSION_ACCESS,
            details={"session_count": 5},
        )

        entries = log.get_entries()

        assert entries[0].reason == PromotionReason.CROSS_SESSION_ACCESS
        assert entries[0].details["session_count"] == 5

    def test_log_to_dict(self):
        """Logs should be serializable."""
        log = PromotionLog()

        log.record_decision(
            node_id="node-1",
            from_tier="task",
            to_tier="session",
            reason=PromotionReason.SESSION_USE_COUNT,
        )

        data = log.to_dict()

        assert "entries" in data
        assert len(data["entries"]) == 1


class TestManualPromotionOverride:
    """Tests for manual promotion override (SPEC-09.15)."""

    def test_manual_promote(self):
        """SPEC-09.15: Support manual promotion override."""
        promoter = CrossSessionPromoter()

        decision = promoter.manual_promote(
            node_id="node-1",
            from_tier="task",
            to_tier="longterm",
            reason="User requested",
        )

        assert decision.should_promote
        assert decision.target_tier == "longterm"
        assert decision.is_manual

    def test_manual_promote_skips_criteria(self):
        """Manual promotion should skip automatic criteria."""
        promoter = CrossSessionPromoter()

        # Would not meet automatic criteria
        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=1,
            access_count=1,
            success_count=0,
            confidence=0.3,
            current_tier="task",
        )

        # Automatic would reject
        auto_decision = promoter.evaluate_promotion(candidate)
        assert not auto_decision.should_promote

        # Manual overrides
        manual_decision = promoter.manual_promote(
            node_id="node-1",
            from_tier="task",
            to_tier="longterm",
            reason="Important for project",
        )
        assert manual_decision.should_promote

    def test_manual_promote_logs_override(self):
        """Manual promotion should log the override."""
        promoter = CrossSessionPromoter()

        promoter.manual_promote(
            node_id="node-1",
            from_tier="session",
            to_tier="longterm",
            reason="Critical memory",
        )

        entries = promoter.log.get_entries()

        assert len(entries) == 1
        assert entries[0].reason == PromotionReason.MANUAL_OVERRIDE


class TestPromotionConfig:
    """Tests for promotion configuration."""

    def test_default_config(self):
        """Default configuration values."""
        config = PromotionConfig()

        assert config.min_sessions == 3
        assert config.min_confidence == 0.8
        assert config.task_to_session_uses == 2
        assert config.archive_staleness_days == 90

    def test_configurable_values(self):
        """Configuration values should be customizable."""
        config = PromotionConfig(
            min_sessions=5,
            min_confidence=0.9,
            task_to_session_uses=3,
            archive_staleness_days=60,
        )

        assert config.min_sessions == 5
        assert config.min_confidence == 0.9
        assert config.task_to_session_uses == 3
        assert config.archive_staleness_days == 60


class TestPromotionCandidate:
    """Tests for PromotionCandidate structure."""

    def test_candidate_has_required_fields(self):
        """Candidate should have all required fields."""
        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
        )

        assert candidate.node_id == "node-1"
        assert candidate.session_count == 3
        assert candidate.current_tier == "session"

    def test_candidate_optional_fields(self):
        """Candidate optional fields have defaults."""
        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
        )

        assert candidate.content is None
        assert candidate.metadata is None
        assert candidate.days_since_access == 0

    def test_candidate_to_dict(self):
        """Candidate should be serializable."""
        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=3,
            access_count=5,
            success_count=2,
            confidence=0.85,
            current_tier="session",
        )

        data = candidate.to_dict()

        assert "node_id" in data
        assert "session_count" in data
        assert "current_tier" in data


class TestPromotionDecision:
    """Tests for PromotionDecision structure."""

    def test_decision_approve(self):
        """Decision can approve promotion."""
        decision = PromotionDecision(
            node_id="node-1",
            should_promote=True,
            target_tier="longterm",
        )

        assert decision.should_promote
        assert decision.target_tier == "longterm"

    def test_decision_reject(self):
        """Decision can reject promotion."""
        decision = PromotionDecision(
            node_id="node-1",
            should_promote=False,
            rejection_reason="Criteria not met",
        )

        assert not decision.should_promote
        assert decision.rejection_reason is not None

    def test_decision_to_dict(self):
        """Decision should be serializable."""
        decision = PromotionDecision(
            node_id="node-1",
            should_promote=True,
            target_tier="longterm",
        )

        data = decision.to_dict()

        assert "node_id" in data
        assert "should_promote" in data


class TestAccessRecord:
    """Tests for AccessRecord structure."""

    def test_record_has_required_fields(self):
        """Access record should have required fields."""
        record = AccessRecord(
            node_id="node-1",
            session_id="session-a",
        )

        assert record.node_id == "node-1"
        assert record.session_id == "session-a"

    def test_record_has_timestamp(self):
        """Access record should have timestamp."""
        record = AccessRecord(
            node_id="node-1",
            session_id="session-a",
        )

        assert record.timestamp is not None

    def test_record_to_dict(self):
        """Access record should be serializable."""
        record = AccessRecord(
            node_id="node-1",
            session_id="session-a",
        )

        data = record.to_dict()

        assert "node_id" in data
        assert "session_id" in data
        assert "timestamp" in data


class TestIntegration:
    """Integration tests for cross-session promotion."""

    def test_full_promotion_workflow(self):
        """Test complete promotion workflow."""
        tracker = CrossSessionTracker()
        promoter = CrossSessionPromoter()

        # Track accesses across sessions
        for session in ["session-a", "session-b", "session-c"]:
            tracker.record_access("node-1", session)
            tracker.record_access("node-1", session, success=True)

        # Build candidate from tracker
        candidate = PromotionCandidate(
            node_id="node-1",
            session_count=tracker.get_session_count("node-1"),
            access_count=tracker.get_access_count("node-1"),
            success_count=tracker.get_success_count("node-1"),
            confidence=0.85,
            current_tier="session",
        )

        # Evaluate promotion
        decision = promoter.evaluate_promotion(candidate)

        assert decision.should_promote
        assert decision.target_tier == "longterm"

    def test_promotion_log_audit_trail(self):
        """Promotion log provides audit trail."""
        promoter = CrossSessionPromoter()

        # Multiple promotions
        for i in range(3):
            candidate = PromotionCandidate(
                node_id=f"node-{i}",
                session_count=3,
                access_count=5,
                success_count=2,
                confidence=0.85,
                current_tier="session",
            )
            promoter.evaluate_promotion(candidate)

        entries = promoter.log.get_entries()

        assert len(entries) == 3
        assert all(e.reason == PromotionReason.CROSS_SESSION_ACCESS for e in entries)

    def test_tracker_to_candidate_conversion(self):
        """Tracker data converts to candidate correctly."""
        tracker = CrossSessionTracker()

        # Simulate multiple sessions
        tracker.record_access("node-1", "session-a", success=True)
        tracker.record_access("node-1", "session-b", success=False)
        tracker.record_access("node-1", "session-c", success=True)

        candidate = tracker.to_candidate(
            node_id="node-1",
            confidence=0.9,
            current_tier="session",
        )

        assert candidate.session_count == 3
        assert candidate.access_count == 3
        assert candidate.success_count == 2
