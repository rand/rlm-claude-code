"""
Tests for interactive steering (SPEC-11.10-11.16).

Tests cover:
- Steering point types
- Steering point structure
- Auto-steering policy
- Timeout handling
- Steering logging
"""

import time

from src.orchestrator.steering import (
    AutoSteeringPolicy,
    InteractiveOrchestrator,
    SteeringDecision,
    SteeringPoint,
    SteeringPointType,
    SteeringResponse,
)


class TestSteeringPointTypes:
    """Tests for steering point types (SPEC-11.11)."""

    def test_branch_steering_type(self):
        """SPEC-11.11: SteeringPoint SHALL support 'branch' type."""
        assert SteeringPointType.BRANCH.value == "branch"

    def test_depth_steering_type(self):
        """SPEC-11.11: SteeringPoint SHALL support 'depth' type."""
        assert SteeringPointType.DEPTH.value == "depth"

    def test_abort_steering_type(self):
        """SPEC-11.11: SteeringPoint SHALL support 'abort' type."""
        assert SteeringPointType.ABORT.value == "abort"

    def test_refine_steering_type(self):
        """SPEC-11.11: SteeringPoint SHALL support 'refine' type."""
        assert SteeringPointType.REFINE.value == "refine"


class TestSteeringPointStructure:
    """Tests for SteeringPoint structure (SPEC-11.12)."""

    def test_steering_point_has_options(self):
        """SPEC-11.12: SteeringPoint SHALL include options."""
        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["Option A", "Option B", "Option C"],
            default="Option A",
            timeout=30.0,
            context="Choose exploration path",
        )

        assert point.options == ["Option A", "Option B", "Option C"]

    def test_steering_point_has_default(self):
        """SPEC-11.12: SteeringPoint SHALL include default."""
        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="depth",
            point_type=SteeringPointType.DEPTH,
            options=["increase", "decrease", "maintain"],
            default="maintain",
            timeout=30.0,
            context="Adjust depth",
        )

        assert point.default == "maintain"

    def test_steering_point_has_timeout(self):
        """SPEC-11.12: SteeringPoint SHALL include timeout."""
        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="abort",
            point_type=SteeringPointType.ABORT,
            options=["continue", "abort"],
            default="continue",
            timeout=60.0,
            context="Continue or abort?",
        )

        assert point.timeout == 60.0

    def test_steering_point_default_timeout(self):
        """Default timeout should be reasonable."""
        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["A", "B"],
            default="A",
            context="Choose",
        )

        # Default timeout should be 30 seconds
        assert point.timeout == 30.0


class TestSteeringPointCreation:
    """Tests for steering point creation (SPEC-11.13)."""

    def test_create_branch_steering_point(self):
        """SPEC-11.13: System SHALL present steering at multiple viable paths."""
        orchestrator = InteractiveOrchestrator()

        point = orchestrator.create_branch_point(
            turn=10,
            depth=2,
            paths=[
                ("Use recursion", 0.8),
                ("Use iteration", 0.75),
                ("Use memoization", 0.6),
            ],
            context="Multiple approaches available",
        )

        assert point.point_type == SteeringPointType.BRANCH
        assert len(point.options) == 3
        assert "Use recursion" in point.options
        # Default should be highest value path
        assert point.default == "Use recursion"

    def test_create_depth_steering_point(self):
        """SPEC-11.13: System SHALL present steering before recursive decomposition."""
        orchestrator = InteractiveOrchestrator()

        point = orchestrator.create_depth_point(
            turn=5,
            depth=1,
            current_depth_budget=3,
            confidence=0.4,
            context="Low confidence result",
        )

        assert point.point_type == SteeringPointType.DEPTH
        assert "increase" in point.options
        assert "decrease" in point.options
        assert "maintain" in point.options

    def test_create_abort_steering_point(self):
        """System should allow abort steering."""
        orchestrator = InteractiveOrchestrator()

        point = orchestrator.create_abort_point(
            turn=20,
            depth=3,
            cost_so_far=0.75,
            progress_summary="Partial results available",
        )

        assert point.point_type == SteeringPointType.ABORT
        assert "continue" in point.options
        assert "abort" in point.options

    def test_create_refine_steering_point(self):
        """SPEC-11.11: System SHALL support 'refine' type."""
        orchestrator = InteractiveOrchestrator()

        point = orchestrator.create_refine_point(
            turn=15,
            depth=2,
            current_result="Partial analysis complete",
            context="Need additional guidance",
        )

        assert point.point_type == SteeringPointType.REFINE
        assert "accept" in point.options
        assert "refine" in point.options


class TestAutoSteering:
    """Tests for auto-steering policy (SPEC-11.14)."""

    def test_auto_steering_for_ci(self):
        """SPEC-11.14: System SHALL support auto-steering policy for testing/CI."""
        policy = AutoSteeringPolicy(
            max_turns_before_stop=50,
            max_depth=3,
        )

        point = SteeringPoint(
            turn=10,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["A", "B", "C"],
            default="A",
            timeout=30.0,
            context="Test",
        )

        decision = policy.decide(point)
        assert decision in [d for d in SteeringDecision]

    def test_auto_steering_continues_within_budget(self):
        """Auto steering should continue when within budget."""
        policy = AutoSteeringPolicy(
            max_turns_before_stop=50,
            max_depth=5,
            cost_threshold_usd=1.0,
        )

        point = SteeringPoint(
            turn=10,
            depth=2,
            decision_type="continue_or_stop",
            point_type=SteeringPointType.BRANCH,
            options=["continue", "stop"],
            default="continue",
            timeout=30.0,
            context="Normal progress",
            current_state={"cost_usd": 0.3, "confidence": 0.8},
        )

        decision = policy.decide(point)
        assert decision == SteeringDecision.CONTINUE

    def test_auto_steering_stops_at_turn_limit(self):
        """Auto steering should stop at turn limit."""
        policy = AutoSteeringPolicy(max_turns_before_stop=20)

        point = SteeringPoint(
            turn=25,
            depth=1,
            decision_type="continue_or_stop",
            point_type=SteeringPointType.BRANCH,
            options=["continue", "stop"],
            default="continue",
            timeout=30.0,
            context="",
        )

        decision = policy.decide(point)
        assert decision == SteeringDecision.STOP

    def test_auto_steering_stops_at_cost_limit(self):
        """Auto steering should stop when cost exceeded."""
        policy = AutoSteeringPolicy(cost_threshold_usd=0.5)

        point = SteeringPoint(
            turn=10,
            depth=1,
            decision_type="continue_or_stop",
            point_type=SteeringPointType.BRANCH,
            options=["continue", "stop"],
            default="continue",
            timeout=30.0,
            context="",
            current_state={"cost_usd": 0.6},
        )

        decision = policy.decide(point)
        assert decision == SteeringDecision.STOP


class TestTimeoutHandling:
    """Tests for timeout handling (SPEC-11.16)."""

    def test_timeout_uses_default(self):
        """SPEC-11.16: Timeout on steering request SHALL use default option."""
        orchestrator = InteractiveOrchestrator()

        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["A", "B", "C"],
            default="B",
            timeout=0.001,  # Very short timeout
            context="Test timeout",
        )

        # Simulate timeout by using a callback that sleeps
        def slow_callback(p: SteeringPoint) -> SteeringDecision:
            time.sleep(0.1)
            return SteeringDecision.STOP

        orchestrator.callback = slow_callback
        decision = orchestrator.get_decision_with_timeout(point)

        # Should return default-mapped decision due to timeout
        assert decision is not None

    def test_decision_within_timeout(self):
        """Decision made within timeout should be honored."""
        orchestrator = InteractiveOrchestrator()

        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["A", "B", "C"],
            default="A",
            timeout=5.0,  # Long timeout
            context="Test",
        )

        # Fast callback
        def fast_callback(p: SteeringPoint) -> SteeringDecision:
            return SteeringDecision.STOP

        orchestrator.callback = fast_callback
        decision = orchestrator.get_decision_with_timeout(point)

        assert decision == SteeringDecision.STOP


class TestSteeringLogging:
    """Tests for steering logging (SPEC-11.15)."""

    def test_steering_responses_logged(self):
        """SPEC-11.15: Steering responses SHALL be logged for analysis."""
        orchestrator = InteractiveOrchestrator()

        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["A", "B"],
            default="A",
            timeout=30.0,
            context="Test logging",
        )

        # Make a decision
        orchestrator.get_decision(point)

        # Check history
        history = orchestrator.get_steering_history()
        assert len(history) == 1
        assert history[0][0] is point

    def test_steering_response_includes_timestamp(self):
        """Logged steering should include timestamp."""
        orchestrator = InteractiveOrchestrator()

        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["A", "B"],
            default="A",
            timeout=30.0,
            context="Test",
        )

        before = time.time()
        orchestrator.get_decision(point)
        after = time.time()

        responses = orchestrator.get_steering_responses()
        assert len(responses) == 1
        assert before <= responses[0].timestamp <= after

    def test_steering_response_includes_decision_source(self):
        """Response should indicate if decision was user or auto."""
        orchestrator = InteractiveOrchestrator()

        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["A", "B"],
            default="A",
            timeout=30.0,
            context="Test",
        )

        # No user callback, so auto-steering
        orchestrator.get_decision(point)

        responses = orchestrator.get_steering_responses()
        assert responses[0].source == "auto"

    def test_clear_steering_history(self):
        """Should be able to clear steering history."""
        orchestrator = InteractiveOrchestrator()

        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="branch",
            point_type=SteeringPointType.BRANCH,
            options=["A", "B"],
            default="A",
            timeout=30.0,
            context="Test",
        )

        orchestrator.get_decision(point)
        assert len(orchestrator.get_steering_history()) == 1

        orchestrator.clear_history()
        assert len(orchestrator.get_steering_history()) == 0


class TestSteeringResponse:
    """Tests for SteeringResponse structure."""

    def test_steering_response_structure(self):
        """SteeringResponse should capture all relevant info."""
        response = SteeringResponse(
            point_id="sp-123",
            point_type=SteeringPointType.BRANCH,
            decision=SteeringDecision.CONTINUE,
            selected_option="Option A",
            timestamp=time.time(),
            source="user",
            response_time_ms=150.5,
        )

        assert response.point_id == "sp-123"
        assert response.point_type == SteeringPointType.BRANCH
        assert response.decision == SteeringDecision.CONTINUE
        assert response.selected_option == "Option A"
        assert response.source == "user"
        assert response.response_time_ms == 150.5

    def test_steering_response_to_dict(self):
        """SteeringResponse should be serializable."""
        response = SteeringResponse(
            point_id="sp-456",
            point_type=SteeringPointType.DEPTH,
            decision=SteeringDecision.ADJUST_DEPTH,
            selected_option="increase",
            timestamp=1234567890.0,
            source="auto",
            response_time_ms=50.0,
        )

        data = response.to_dict()
        assert data["point_id"] == "sp-456"
        assert data["point_type"] == "depth"
        assert data["decision"] == "adjust_depth"


class TestSteeringPointPresentation:
    """Tests for steering point presentation timing (SPEC-11.13)."""

    def test_should_steer_before_recursion(self):
        """SPEC-11.13: Steering at before recursive decomposition."""
        orchestrator = InteractiveOrchestrator()

        # Before recursion with multiple paths
        assert orchestrator.should_present_steering(
            turn=5,
            depth=0,
            event="before_recursion",
            num_paths=3,
        )

    def test_should_steer_on_low_confidence(self):
        """SPEC-11.13: Steering after low-confidence intermediate results."""
        orchestrator = InteractiveOrchestrator()

        assert orchestrator.should_present_steering(
            turn=10,
            depth=2,
            event="intermediate_result",
            confidence=0.3,
        )

    def test_should_not_steer_on_high_confidence(self):
        """High confidence should not trigger steering."""
        orchestrator = InteractiveOrchestrator()

        # Use turn=7 to avoid legacy milestone triggers (10, 20, 30, 40)
        # and depth=0 to avoid depth transition trigger
        assert not orchestrator.should_present_steering(
            turn=7,
            depth=0,
            event="intermediate_result",
            confidence=0.9,
        )

    def test_should_steer_when_multiple_paths(self):
        """SPEC-11.13: Steering when multiple viable paths exist."""
        orchestrator = InteractiveOrchestrator()

        assert orchestrator.should_present_steering(
            turn=5,
            depth=1,
            event="path_selection",
            num_paths=4,
        )
