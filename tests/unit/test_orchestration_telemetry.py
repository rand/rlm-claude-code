"""
Unit tests for orchestration telemetry.

Implements: Feature 3e0.6 - Telemetry-Driven Heuristics
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def telemetry(temp_dir):
    """Create a fresh OrchestrationTelemetry instance."""
    from src.orchestration_telemetry import OrchestrationTelemetry, TelemetryConfig

    config = TelemetryConfig(
        decisions_path=os.path.join(temp_dir, "decisions.jsonl"),
        feedback_path=os.path.join(temp_dir, "feedback.jsonl"),
    )
    return OrchestrationTelemetry(config=config)


# =============================================================================
# Data Class Tests
# =============================================================================


class TestHeuristicOutcome:
    """Tests for HeuristicOutcome dataclass."""

    def test_creation(self):
        """Test basic creation."""
        from src.orchestration_telemetry import HeuristicOutcome

        outcome = HeuristicOutcome(
            heuristic_name="discovery_required",
            triggered=True,
            predicted_rlm=True,
            query_id="test-123",
        )

        assert outcome.heuristic_name == "discovery_required"
        assert outcome.triggered is True
        assert outcome.predicted_rlm is True

    def test_to_dict(self):
        """Test dictionary conversion."""
        from src.orchestration_telemetry import HeuristicOutcome

        outcome = HeuristicOutcome(
            heuristic_name="debugging_deep",
            triggered=False,
            predicted_rlm=False,
        )

        d = outcome.to_dict()
        assert d["heuristic_name"] == "debugging_deep"
        assert d["triggered"] is False


class TestHeuristicAccuracy:
    """Tests for HeuristicAccuracy dataclass."""

    def test_precision_calculation(self):
        """Test precision: TP / (TP + FP)."""
        from src.orchestration_telemetry import HeuristicAccuracy

        acc = HeuristicAccuracy(
            heuristic_name="test",
            true_positives=8,
            false_positives=2,
        )

        assert acc.precision == 0.8  # 8 / (8 + 2)

    def test_recall_calculation(self):
        """Test recall: TP / (TP + FN)."""
        from src.orchestration_telemetry import HeuristicAccuracy

        acc = HeuristicAccuracy(
            heuristic_name="test",
            true_positives=6,
            false_negatives=4,
        )

        assert acc.recall == 0.6  # 6 / (6 + 4)

    def test_f1_score_calculation(self):
        """Test F1: 2 * (P * R) / (P + R)."""
        from src.orchestration_telemetry import HeuristicAccuracy

        acc = HeuristicAccuracy(
            heuristic_name="test",
            true_positives=8,
            false_positives=2,
            false_negatives=2,
        )

        # precision = 8/10 = 0.8
        # recall = 8/10 = 0.8
        # f1 = 2 * 0.8 * 0.8 / (0.8 + 0.8) = 0.8
        assert acc.f1_score == pytest.approx(0.8)

    def test_accuracy_calculation(self):
        """Test overall accuracy: (TP + TN) / total."""
        from src.orchestration_telemetry import HeuristicAccuracy

        acc = HeuristicAccuracy(
            heuristic_name="test",
            true_positives=40,
            true_negatives=40,
            false_positives=10,
            false_negatives=10,
        )

        assert acc.accuracy == 0.8  # 80 / 100

    def test_zero_division_handling(self):
        """Test that zero division returns 0.0."""
        from src.orchestration_telemetry import HeuristicAccuracy

        acc = HeuristicAccuracy(heuristic_name="test")

        assert acc.precision == 0.0
        assert acc.recall == 0.0
        assert acc.f1_score == 0.0
        assert acc.accuracy == 0.0

    def test_to_dict_includes_metrics(self):
        """Test that to_dict includes computed metrics."""
        from src.orchestration_telemetry import HeuristicAccuracy

        acc = HeuristicAccuracy(
            heuristic_name="test",
            true_positives=5,
            false_positives=5,
        )

        d = acc.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1_score" in d
        assert "accuracy" in d


class TestTelemetryDecisionLog:
    """Tests for TelemetryDecisionLog dataclass."""

    def test_creation(self):
        """Test basic creation."""
        from src.orchestration_telemetry import TelemetryDecisionLog

        log = TelemetryDecisionLog(
            query_id="q-123",
            query="What is the meaning of life?",
            query_length=30,
            timestamp="2026-01-13T10:00:00",
            source="heuristic",
            latency_ms=50.0,
            activate_rlm=True,
            activation_reason="discovery_required",
            execution_mode="balanced",
            model_tier="balanced",
            depth_budget=2,
            complexity_score=0.7,
            confidence=0.9,
            heuristics_triggered=["discovery_required"],
            heuristics_checked=["discovery_required", "synthesis_required"],
        )

        assert log.query_id == "q-123"
        assert log.activate_rlm is True
        assert len(log.heuristics_triggered) == 1
        assert len(log.heuristics_checked) == 2

    def test_from_dict(self):
        """Test creation from dictionary."""
        from src.orchestration_telemetry import TelemetryDecisionLog

        data = {
            "query_id": "q-456",
            "query": "Test query",
            "query_length": 10,
            "timestamp": "2026-01-13T11:00:00",
            "source": "api",
            "latency_ms": 100.0,
            "activate_rlm": False,
            "activation_reason": "",
            "execution_mode": "fast",
            "model_tier": "fast",
            "depth_budget": 1,
            "complexity_score": 0.3,
            "confidence": 0.95,
            "heuristics_triggered": [],
            "heuristics_checked": ["discovery_required"],
        }

        log = TelemetryDecisionLog.from_dict(data)
        assert log.query_id == "q-456"
        assert log.source == "api"


# =============================================================================
# OrchestrationTelemetry Tests
# =============================================================================


class TestTelemetryLogging:
    """Tests for decision logging."""

    def test_log_decision_returns_query_id(self, telemetry):
        """Test that logging returns a query ID."""
        query_id = telemetry.log_decision_with_heuristics(
            query="Test query",
            decision={"activate_rlm": True, "activation_reason": "test"},
            heuristics_triggered=["discovery_required"],
            heuristics_checked=["discovery_required", "synthesis_required"],
            source="heuristic",
        )

        assert query_id is not None
        assert len(query_id) > 0

    def test_log_decision_stores_in_memory(self, telemetry):
        """Test that decision is stored in memory."""
        query_id = telemetry.log_decision_with_heuristics(
            query="Test query",
            decision={"activate_rlm": True},
            heuristics_triggered=["discovery_required"],
            heuristics_checked=["discovery_required"],
            source="heuristic",
        )

        assert query_id in telemetry._decisions
        assert telemetry._decisions[query_id].activate_rlm is True

    def test_log_decision_increments_count(self, telemetry):
        """Test that decision count is incremented."""
        assert telemetry._decision_count == 0

        telemetry.log_decision_with_heuristics(
            query="Test 1",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        assert telemetry._decision_count == 1

        telemetry.log_decision_with_heuristics(
            query="Test 2",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        assert telemetry._decision_count == 2

    def test_log_decision_tracks_heuristic_outcomes(self, telemetry):
        """Test that heuristic outcomes are tracked."""
        telemetry.log_decision_with_heuristics(
            query="Test query",
            decision={"activate_rlm": True},
            heuristics_triggered=["discovery_required"],
            heuristics_checked=["discovery_required", "synthesis_required"],
            source="heuristic",
        )

        # Should have 2 outcomes (one per checked heuristic)
        assert len(telemetry._heuristic_outcomes) == 2

        # Check that triggered status is correct
        outcomes_by_name = {o.heuristic_name: o for o in telemetry._heuristic_outcomes}
        assert outcomes_by_name["discovery_required"].triggered is True
        assert outcomes_by_name["synthesis_required"].triggered is False

    def test_log_decision_writes_to_file(self, telemetry, temp_dir):
        """Test that decision is written to file."""
        telemetry.log_decision_with_heuristics(
            query="Test query",
            decision={"activate_rlm": True},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        log_path = Path(temp_dir) / "decisions.jsonl"
        assert log_path.exists()
        assert log_path.stat().st_size > 0


class TestOutcomeRecording:
    """Tests for outcome recording."""

    def test_record_outcome_updates_decision(self, telemetry):
        """Test that outcome updates the decision."""
        query_id = telemetry.log_decision_with_heuristics(
            query="Test",
            decision={"activate_rlm": True},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        telemetry.record_outcome(
            query_id=query_id,
            execution_succeeded=True,
            actual_depth_used=2,
            actual_cost=0.05,
            rlm_was_helpful=True,
        )

        decision = telemetry._decisions[query_id]
        assert decision.execution_succeeded is True
        assert decision.actual_depth_used == 2
        assert decision.actual_cost == 0.05
        assert decision.rlm_was_helpful is True

    def test_record_outcome_updates_heuristic_outcomes(self, telemetry):
        """Test that outcome updates heuristic outcomes."""
        query_id = telemetry.log_decision_with_heuristics(
            query="Test",
            decision={"activate_rlm": True},
            heuristics_triggered=["discovery_required"],
            heuristics_checked=["discovery_required"],
            source="test",
        )

        telemetry.record_outcome(
            query_id=query_id,
            execution_succeeded=True,
            actual_depth_used=2,
            actual_cost=0.05,
            rlm_was_helpful=True,
        )

        outcome = telemetry._heuristic_outcomes[0]
        assert outcome.actual_rlm_needed is True
        assert outcome.execution_succeeded is True

    def test_record_outcome_writes_to_file(self, telemetry, temp_dir):
        """Test that outcome is written to feedback file."""
        query_id = telemetry.log_decision_with_heuristics(
            query="Test",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        telemetry.record_outcome(
            query_id=query_id,
            execution_succeeded=True,
            actual_depth_used=1,
            actual_cost=0.01,
        )

        feedback_path = Path(temp_dir) / "feedback.jsonl"
        assert feedback_path.exists()
        assert feedback_path.stat().st_size > 0


class TestHeuristicAccuracyComputation:
    """Tests for accuracy computation."""

    def test_compute_accuracy_no_feedback(self, telemetry):
        """Test accuracy with no feedback data."""
        telemetry.log_decision_with_heuristics(
            query="Test",
            decision={"activate_rlm": True},
            heuristics_triggered=["discovery_required"],
            heuristics_checked=["discovery_required"],
            source="test",
        )

        acc = telemetry.compute_heuristic_accuracy("discovery_required")

        # No feedback yet, so all zeros
        assert acc.true_positives == 0
        assert acc.false_positives == 0

    def test_compute_accuracy_with_feedback(self, telemetry):
        """Test accuracy with feedback data."""
        # Log multiple decisions with different outcomes
        for i in range(5):
            query_id = telemetry.log_decision_with_heuristics(
                query=f"Test {i}",
                decision={"activate_rlm": True},
                heuristics_triggered=["discovery_required"],
                heuristics_checked=["discovery_required"],
                source="test",
            )
            # 3 true positives, 2 false positives
            rlm_helpful = i < 3
            telemetry.record_outcome(
                query_id=query_id,
                execution_succeeded=True,
                actual_depth_used=1,
                actual_cost=0.01,
                rlm_was_helpful=rlm_helpful,
            )

        acc = telemetry.compute_heuristic_accuracy("discovery_required")

        assert acc.true_positives == 3
        assert acc.false_positives == 2
        assert acc.precision == 0.6  # 3 / 5

    def test_compute_all_accuracies(self, telemetry):
        """Test computing accuracies for all heuristics."""
        query_id = telemetry.log_decision_with_heuristics(
            query="Test",
            decision={"activate_rlm": True},
            heuristics_triggered=["discovery_required"],
            heuristics_checked=["discovery_required", "synthesis_required"],
            source="test",
        )
        telemetry.record_outcome(
            query_id=query_id,
            execution_succeeded=True,
            actual_depth_used=1,
            actual_cost=0.01,
            rlm_was_helpful=True,
        )

        accuracies = telemetry.compute_all_accuracies()

        assert "discovery_required" in accuracies
        assert "synthesis_required" in accuracies


class TestTelemetryReport:
    """Tests for report generation."""

    def test_generate_report_empty(self, telemetry):
        """Test report generation with no data."""
        report = telemetry.generate_report()

        assert report.total_decisions == 0
        assert report.decisions_with_feedback == 0

    def test_generate_report_with_data(self, telemetry):
        """Test report generation with data."""
        # Add some decisions with feedback
        for i in range(10):
            query_id = telemetry.log_decision_with_heuristics(
                query=f"Test {i}",
                decision={"activate_rlm": i < 7},
                heuristics_triggered=["discovery_required"] if i < 7 else [],
                heuristics_checked=["discovery_required"],
                source="test",
            )
            telemetry.record_outcome(
                query_id=query_id,
                execution_succeeded=True,
                actual_depth_used=1,
                actual_cost=0.01,
                rlm_was_helpful=i < 6,  # 6 needed RLM
            )

        report = telemetry.generate_report()

        assert report.total_decisions == 10
        assert report.decisions_with_feedback == 10
        assert "discovery_required" in report.heuristic_accuracies

    def test_report_identifies_underperformers(self, telemetry):
        """Test that report identifies low-accuracy heuristics."""
        # Create a heuristic with many false positives
        for i in range(10):
            query_id = telemetry.log_decision_with_heuristics(
                query=f"Test {i}",
                decision={"activate_rlm": True},
                heuristics_triggered=["bad_heuristic"],
                heuristics_checked=["bad_heuristic"],
                source="test",
            )
            # Only 1/10 was actually helpful (lots of false positives)
            telemetry.record_outcome(
                query_id=query_id,
                execution_succeeded=True,
                actual_depth_used=1,
                actual_cost=0.01,
                rlm_was_helpful=(i == 0),
            )

        report = telemetry.generate_report()

        assert "bad_heuristic" in report.underperformers


class TestTrainingDataExport:
    """Tests for training data export."""

    def test_export_jsonl(self, telemetry, temp_dir):
        """Test JSONL export."""
        # Add decisions with feedback
        for i in range(5):
            query_id = telemetry.log_decision_with_heuristics(
                query=f"Test {i}",
                decision={"activate_rlm": True},
                heuristics_triggered=[],
                heuristics_checked=[],
                source="test",
            )
            telemetry.record_outcome(
                query_id=query_id,
                execution_succeeded=True,
                actual_depth_used=1,
                actual_cost=0.01,
            )

        output_path = os.path.join(temp_dir, "export.jsonl")
        count = telemetry.export_training_data(output_path, format="jsonl")

        assert count == 5
        assert Path(output_path).exists()

    def test_export_csv(self, telemetry, temp_dir):
        """Test CSV export."""
        query_id = telemetry.log_decision_with_heuristics(
            query="Test",
            decision={"activate_rlm": True},
            heuristics_triggered=["discovery_required"],
            heuristics_checked=["discovery_required"],
            source="test",
        )
        telemetry.record_outcome(
            query_id=query_id,
            execution_succeeded=True,
            actual_depth_used=1,
            actual_cost=0.01,
        )

        output_path = os.path.join(temp_dir, "export.csv")
        count = telemetry.export_training_data(output_path, format="csv")

        assert count == 1
        assert Path(output_path).exists()

    def test_export_skips_decisions_without_feedback(self, telemetry, temp_dir):
        """Test that export only includes decisions with feedback."""
        # One with feedback
        query_id = telemetry.log_decision_with_heuristics(
            query="With feedback",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )
        telemetry.record_outcome(
            query_id=query_id,
            execution_succeeded=True,
            actual_depth_used=1,
            actual_cost=0.01,
        )

        # One without feedback
        telemetry.log_decision_with_heuristics(
            query="Without feedback",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        output_path = os.path.join(temp_dir, "export.jsonl")
        count = telemetry.export_training_data(output_path)

        assert count == 1  # Only the one with feedback


class TestHeuristicWeights:
    """Tests for heuristic weight computation."""

    def test_weights_default_for_unused(self, telemetry):
        """Test default weight for unused heuristics."""
        telemetry.log_decision_with_heuristics(
            query="Test",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=["unused_heuristic"],
            source="test",
        )
        # No feedback recorded

        weights = telemetry.get_heuristic_weights()

        assert weights.get("unused_heuristic") == 0.5

    def test_weights_based_on_f1(self, telemetry):
        """Test weights are based on F1 score."""
        # Create heuristic with perfect accuracy
        for i in range(5):
            query_id = telemetry.log_decision_with_heuristics(
                query=f"Test {i}",
                decision={"activate_rlm": True},
                heuristics_triggered=["good_heuristic"],
                heuristics_checked=["good_heuristic"],
                source="test",
            )
            telemetry.record_outcome(
                query_id=query_id,
                execution_succeeded=True,
                actual_depth_used=1,
                actual_cost=0.01,
                rlm_was_helpful=True,  # All true positives
            )

        weights = telemetry.get_heuristic_weights()

        # Should have high weight due to high F1
        assert weights["good_heuristic"] >= 0.8


class TestTelemetryStatistics:
    """Tests for statistics retrieval."""

    def test_get_statistics(self, telemetry):
        """Test statistics retrieval."""
        telemetry.log_decision_with_heuristics(
            query="Test",
            decision={},
            heuristics_triggered=["h1"],
            heuristics_checked=["h1", "h2"],
            source="test",
        )

        stats = telemetry.get_statistics()

        assert stats["total_decisions"] == 1
        assert stats["heuristic_outcomes_tracked"] == 2


class TestTelemetryReset:
    """Tests for reset functionality."""

    def test_reset_clears_memory(self, telemetry):
        """Test that reset clears all in-memory data."""
        telemetry.log_decision_with_heuristics(
            query="Test",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        assert telemetry._decision_count == 1
        assert len(telemetry._decisions) == 1

        telemetry.reset()

        assert telemetry._decision_count == 0
        assert len(telemetry._decisions) == 0
        assert len(telemetry._heuristic_outcomes) == 0


class TestDisabledTelemetry:
    """Tests for disabled telemetry."""

    def test_disabled_returns_query_id(self, temp_dir):
        """Test that disabled telemetry still returns query ID."""
        from src.orchestration_telemetry import OrchestrationTelemetry, TelemetryConfig

        config = TelemetryConfig(enabled=False)
        telemetry = OrchestrationTelemetry(config=config)

        query_id = telemetry.log_decision_with_heuristics(
            query="Test",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        assert query_id is not None

    def test_disabled_does_not_store(self, temp_dir):
        """Test that disabled telemetry doesn't store data."""
        from src.orchestration_telemetry import OrchestrationTelemetry, TelemetryConfig

        config = TelemetryConfig(enabled=False)
        telemetry = OrchestrationTelemetry(config=config)

        telemetry.log_decision_with_heuristics(
            query="Test",
            decision={},
            heuristics_triggered=[],
            heuristics_checked=[],
            source="test",
        )

        assert telemetry._decision_count == 0
        assert len(telemetry._decisions) == 0
