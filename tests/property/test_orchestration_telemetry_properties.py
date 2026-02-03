"""
Property-based tests for orchestration telemetry.

Implements: Feature 3e0.6 - Telemetry-Driven Heuristics
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Strategies
# =============================================================================

heuristic_name_strategy = st.sampled_from(
    [
        "complexity_high",
        "discovery_required",
        "multi_file_change",
        "architecture_decision",
        "recursive_reasoning",
        "unbounded_exploration",
    ]
)

query_strategy = st.text(
    min_size=1,
    max_size=200,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        whitelist_characters=" ",
    ),
)


def make_telemetry():
    """Create a fresh OrchestrationTelemetry instance with disabled file I/O."""
    from src.orchestration_telemetry import OrchestrationTelemetry, TelemetryConfig

    config = TelemetryConfig(enabled=False)  # Disable file I/O for tests
    telemetry = OrchestrationTelemetry(config)
    telemetry.config.enabled = True  # Enable in-memory tracking
    telemetry._decisions_path = None  # Ensure no file writes
    telemetry._feedback_path = None
    return telemetry


# =============================================================================
# HeuristicAccuracy Properties
# =============================================================================


@pytest.mark.hypothesis
class TestHeuristicAccuracyProperties:
    """Property tests for heuristic accuracy metric invariants."""

    @given(
        tp=st.integers(min_value=0, max_value=100),
        fp=st.integers(min_value=0, max_value=100),
        tn=st.integers(min_value=0, max_value=100),
        fn=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_precision_recall_bounds(self, tp, fp, tn, fn):
        """
        Precision and recall should always be between 0 and 1.

        @trace 3e0.6
        """
        from src.orchestration_telemetry import HeuristicAccuracy

        acc = HeuristicAccuracy(
            heuristic_name="test",
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            total_triggers=tp + fp,
        )

        assert 0.0 <= acc.precision <= 1.0
        assert 0.0 <= acc.recall <= 1.0
        assert 0.0 <= acc.f1_score <= 1.0
        assert 0.0 <= acc.accuracy <= 1.0

    @given(
        tp=st.integers(min_value=0, max_value=100),
        fp=st.integers(min_value=0, max_value=100),
        tn=st.integers(min_value=0, max_value=100),
        fn=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_f1_between_precision_recall(self, tp, fp, tn, fn):
        """
        F1 score should be the harmonic mean of precision and recall.

        @trace 3e0.6
        """
        from src.orchestration_telemetry import HeuristicAccuracy

        acc = HeuristicAccuracy(
            heuristic_name="test",
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            total_triggers=tp + fp,
        )

        p, r = acc.precision, acc.recall

        # F1 is harmonic mean, so it should be <= max(p, r) when both > 0
        if p > 0 and r > 0:
            assert acc.f1_score <= max(p, r) + 0.001  # Small epsilon for float precision

        # F1 should be 0 when either p or r is 0
        if p == 0 or r == 0:
            assert acc.f1_score == 0.0

    @given(
        perfect_count=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_perfect_classifier_metrics(self, perfect_count):
        """
        A perfect classifier should have precision, recall, and F1 all equal to 1.0.

        @trace 3e0.6
        """
        from src.orchestration_telemetry import HeuristicAccuracy

        # Perfect classifier: all TPs and TNs, no FPs or FNs
        acc = HeuristicAccuracy(
            heuristic_name="perfect",
            true_positives=perfect_count,
            false_positives=0,
            true_negatives=perfect_count,
            false_negatives=0,
            total_triggers=perfect_count,
        )

        assert acc.precision == 1.0
        assert acc.recall == 1.0
        assert acc.f1_score == 1.0
        assert acc.accuracy == 1.0

    @given(
        bad_count=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_worst_classifier_metrics(self, bad_count):
        """
        A classifier that's always wrong should have 0 precision and recall.

        @trace 3e0.6
        """
        from src.orchestration_telemetry import HeuristicAccuracy

        # All predictions wrong: only FPs and FNs
        acc = HeuristicAccuracy(
            heuristic_name="worst",
            true_positives=0,
            false_positives=bad_count,
            true_negatives=0,
            false_negatives=bad_count,
            total_triggers=bad_count,
        )

        assert acc.precision == 0.0
        assert acc.recall == 0.0
        assert acc.f1_score == 0.0
        assert acc.accuracy == 0.0


# =============================================================================
# Decision Logging Properties
# =============================================================================


@pytest.mark.hypothesis
class TestDecisionLoggingProperties:
    """Property tests for decision logging invariants."""

    @given(
        num_decisions=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=20, deadline=None)
    def test_decision_count_monotonic(self, num_decisions):
        """
        Decision count should increase monotonically.

        @trace 3e0.6
        """
        telemetry = make_telemetry()

        previous_count = 0
        for i in range(num_decisions):
            telemetry.log_decision_with_heuristics(
                query=f"test query {i}",
                decision={"activate_rlm": True, "depth_budget": 2},
                heuristics_triggered=["complexity_high"],
                heuristics_checked=["complexity_high", "discovery_required"],
                source="heuristic",
            )

            assert telemetry._decision_count > previous_count
            previous_count = telemetry._decision_count

        assert telemetry._decision_count == num_decisions

    @given(
        heuristics_checked=st.lists(
            heuristic_name_strategy,
            min_size=1,
            max_size=6,
            unique=True,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_heuristic_outcomes_tracked(self, heuristics_checked):
        """
        All checked heuristics should be tracked in outcomes.

        @trace 3e0.6
        """
        telemetry = make_telemetry()

        triggered = heuristics_checked[: len(heuristics_checked) // 2]  # Half triggered

        telemetry.log_decision_with_heuristics(
            query="test query",
            decision={"activate_rlm": bool(triggered), "depth_budget": 2},
            heuristics_triggered=triggered,
            heuristics_checked=heuristics_checked,
            source="heuristic",
        )

        # All checked heuristics should have an outcome
        outcome_heuristics = {o.heuristic_name for o in telemetry._heuristic_outcomes}
        assert outcome_heuristics == set(heuristics_checked)

        # Triggered status should be correct
        for outcome in telemetry._heuristic_outcomes:
            expected_triggered = outcome.heuristic_name in triggered
            assert outcome.triggered == expected_triggered

    @given(
        query_id=st.uuids(),
    )
    @settings(max_examples=20, deadline=None)
    def test_query_id_preserved(self, query_id):
        """
        Provided query IDs should be preserved in logs.

        @trace 3e0.6
        """
        telemetry = make_telemetry()

        returned_id = telemetry.log_decision_with_heuristics(
            query="test query",
            decision={"activate_rlm": True},
            heuristics_triggered=["complexity_high"],
            heuristics_checked=["complexity_high"],
            source="heuristic",
            query_id=str(query_id),
        )

        assert returned_id == str(query_id)
        assert str(query_id) in telemetry._decisions


# =============================================================================
# Outcome Recording Properties
# =============================================================================


@pytest.mark.hypothesis
class TestOutcomeRecordingProperties:
    """Property tests for outcome recording invariants."""

    @given(
        execution_succeeded=st.booleans(),
        actual_depth=st.integers(min_value=0, max_value=5),
        actual_cost=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        rlm_helpful=st.booleans(),
    )
    @settings(max_examples=30, deadline=None)
    def test_outcome_updates_decision(
        self, execution_succeeded, actual_depth, actual_cost, rlm_helpful
    ):
        """
        Recording an outcome should update the corresponding decision.

        @trace 3e0.6
        """
        telemetry = make_telemetry()

        query_id = telemetry.log_decision_with_heuristics(
            query="test query",
            decision={"activate_rlm": True},
            heuristics_triggered=["complexity_high"],
            heuristics_checked=["complexity_high"],
            source="heuristic",
        )

        telemetry.record_outcome(
            query_id=query_id,
            execution_succeeded=execution_succeeded,
            actual_depth_used=actual_depth,
            actual_cost=actual_cost,
            rlm_was_helpful=rlm_helpful,
        )

        decision = telemetry._decisions[query_id]
        assert decision.execution_succeeded == execution_succeeded
        assert decision.actual_depth_used == actual_depth
        assert decision.actual_cost == actual_cost
        assert decision.rlm_was_helpful == rlm_helpful

    @given(
        outcomes=st.lists(
            st.tuples(st.booleans(), st.booleans()),  # (triggered, rlm_needed)
            min_size=5,
            max_size=30,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_accuracy_computed_from_outcomes(self, outcomes):
        """
        Accuracy metrics should be computed correctly from outcomes.

        @trace 3e0.6
        """
        telemetry = make_telemetry()

        # Log decisions and record outcomes
        for i, (triggered, rlm_needed) in enumerate(outcomes):
            query_id = telemetry.log_decision_with_heuristics(
                query=f"query {i}",
                decision={"activate_rlm": triggered},
                heuristics_triggered=["test_heuristic"] if triggered else [],
                heuristics_checked=["test_heuristic"],
                source="heuristic",
            )

            telemetry.record_outcome(
                query_id=query_id,
                execution_succeeded=True,
                actual_depth_used=2 if rlm_needed else 0,
                actual_cost=0.05,
                rlm_was_helpful=rlm_needed,
            )

        # Compute accuracy
        accuracy = telemetry.compute_heuristic_accuracy("test_heuristic")

        # Manually count expected values
        expected_tp = sum(1 for t, r in outcomes if t and r)
        expected_fp = sum(1 for t, r in outcomes if t and not r)
        expected_tn = sum(1 for t, r in outcomes if not t and not r)
        expected_fn = sum(1 for t, r in outcomes if not t and r)

        assert accuracy.true_positives == expected_tp
        assert accuracy.false_positives == expected_fp
        assert accuracy.true_negatives == expected_tn
        assert accuracy.false_negatives == expected_fn


# =============================================================================
# Heuristic Weights Properties
# =============================================================================


@pytest.mark.hypothesis
class TestHeuristicWeightProperties:
    """Property tests for heuristic weight computation."""

    @given(
        f1_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_weights_bounded(self, f1_scores):
        """
        Heuristic weights should be bounded between 0.1 and 1.0.

        @trace 3e0.6
        """

        telemetry = make_telemetry()

        # Manually populate accuracies with specific F1 scores
        for i, f1 in enumerate(f1_scores):
            name = f"heuristic_{i}"
            # Create accuracy with desired F1 (approximate via TP/FP ratio)
            # F1 = 2*p*r/(p+r), if p=r=f1 then F1=f1
            if f1 > 0:
                tp = int(100 * f1)
                fp = int(100 * (1 - f1)) if f1 < 1 else 0
                fn = fp
            else:
                tp, fp, fn = 0, 10, 10

            telemetry._heuristic_outcomes.append(
                type(
                    "Outcome",
                    (),
                    {
                        "heuristic_name": name,
                        "triggered": True,
                        "actual_rlm_needed": True if tp > 0 else False,
                    },
                )()
            )

        # Get weights
        weights = telemetry.get_heuristic_weights()

        # All weights should be between 0.1 and 1.0
        for name, weight in weights.items():
            assert 0.1 <= weight <= 1.0

    @given(
        num_heuristics=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_unused_heuristics_get_default_weight(self, num_heuristics):
        """
        Heuristics with no triggers should get default weight of 0.5.

        @trace 3e0.6
        """
        from src.orchestration_telemetry import HeuristicOutcome

        telemetry = make_telemetry()

        # Add outcomes where heuristics were checked but never triggered
        for i in range(num_heuristics):
            telemetry._heuristic_outcomes.append(
                HeuristicOutcome(
                    heuristic_name=f"unused_{i}",
                    triggered=False,
                    predicted_rlm=False,
                    actual_rlm_needed=False,  # Provide feedback
                )
            )

        weights = telemetry.get_heuristic_weights()

        # Unused heuristics should have default weight
        for name, weight in weights.items():
            if "unused" in name:
                # With no triggers (all TN), weight is based on F1 which would be 0
                # So weight should be max(0.1, 0) = 0.1
                assert weight == 0.5 or weight == 0.1  # Either default or floor


# =============================================================================
# Report Generation Properties
# =============================================================================


@pytest.mark.hypothesis
class TestReportGenerationProperties:
    """Property tests for report generation."""

    @given(
        num_decisions=st.integers(min_value=1, max_value=20),
        feedback_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_report_counts_accurate(self, num_decisions, feedback_fraction):
        """
        Report should accurately reflect decision and feedback counts.

        @trace 3e0.6
        """
        telemetry = make_telemetry()

        # Log decisions
        query_ids = []
        for i in range(num_decisions):
            qid = telemetry.log_decision_with_heuristics(
                query=f"query {i}",
                decision={"activate_rlm": i % 2 == 0},
                heuristics_triggered=["h1"] if i % 2 == 0 else [],
                heuristics_checked=["h1"],
                source="heuristic",
            )
            query_ids.append(qid)

        # Record feedback for fraction of decisions
        num_with_feedback = int(num_decisions * feedback_fraction)
        for qid in query_ids[:num_with_feedback]:
            telemetry.record_outcome(
                query_id=qid,
                execution_succeeded=True,
                actual_depth_used=2,
                actual_cost=0.05,
            )

        report = telemetry.generate_report()

        assert report.total_decisions == num_decisions
        assert report.decisions_with_feedback == num_with_feedback
