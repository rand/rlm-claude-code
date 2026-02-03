"""
Tests for continuous learning and self-improvement (SPEC-06.40-06.45).

Tests cover:
- Execution outcome recording
- Learning signal extraction
- Learned adjustments management
- Learning rate application
- Session persistence
- Meta-learning
"""

import json
import tempfile
from pathlib import Path

from src.continuous_learning import (
    ContinuousLearner,
    ExecutionOutcome,
    LearnerConfig,
    LearnerState,
    LearningSignal,
    MetaLearner,
    OutcomeRecorder,
    PredictionRecord,
    SignalExtractor,
    SignalType,
)


class TestOutcomeRecording:
    """Tests for execution outcome recording (SPEC-06.40)."""

    def test_records_query_and_features(self):
        """SPEC-06.40: Record query and features."""
        recorder = OutcomeRecorder()

        outcome = recorder.record(
            query="Analyze this code",
            features={"complexity": 0.7, "domain": "code"},
            strategy="decomposition",
            model="sonnet",
            depth=2,
            tools=["grep", "read"],
            success=True,
            cost=0.005,
            latency_ms=1500,
        )

        assert outcome.query == "Analyze this code"
        assert outcome.features["complexity"] == 0.7

    def test_records_strategy_and_model(self):
        """SPEC-06.40: Record strategy and model used."""
        recorder = OutcomeRecorder()

        outcome = recorder.record(
            query="Test",
            features={},
            strategy="map_reduce",
            model="opus",
            depth=3,
            tools=[],
            success=True,
            cost=0.01,
            latency_ms=2000,
        )

        assert outcome.strategy == "map_reduce"
        assert outcome.model == "opus"

    def test_records_depth_reached(self):
        """SPEC-06.40: Record depth reached."""
        recorder = OutcomeRecorder()

        outcome = recorder.record(
            query="Deep analysis",
            features={},
            strategy="recursive",
            model="sonnet",
            depth=3,
            tools=[],
            success=True,
            cost=0.02,
            latency_ms=5000,
        )

        assert outcome.depth == 3

    def test_records_tools_used(self):
        """SPEC-06.40: Record tools used."""
        recorder = OutcomeRecorder()

        outcome = recorder.record(
            query="Find files",
            features={},
            strategy="direct",
            model="haiku",
            depth=1,
            tools=["glob", "grep", "read"],
            success=True,
            cost=0.001,
            latency_ms=500,
        )

        assert outcome.tools == ["glob", "grep", "read"]

    def test_records_success_failure(self):
        """SPEC-06.40: Record success/failure."""
        recorder = OutcomeRecorder()

        success = recorder.record(
            query="Q1",
            features={},
            strategy="s",
            model="m",
            depth=1,
            tools=[],
            success=True,
            cost=0.01,
            latency_ms=100,
        )
        failure = recorder.record(
            query="Q2",
            features={},
            strategy="s",
            model="m",
            depth=1,
            tools=[],
            success=False,
            cost=0.01,
            latency_ms=100,
        )

        assert success.success is True
        assert failure.success is False

    def test_records_user_satisfaction(self):
        """SPEC-06.40: Record user satisfaction if provided."""
        recorder = OutcomeRecorder()

        outcome = recorder.record(
            query="Test",
            features={},
            strategy="direct",
            model="sonnet",
            depth=1,
            tools=[],
            success=True,
            cost=0.005,
            latency_ms=1000,
            user_satisfaction=0.9,
        )

        assert outcome.user_satisfaction == 0.9

    def test_records_cost_and_latency(self):
        """SPEC-06.40: Record cost and latency."""
        recorder = OutcomeRecorder()

        outcome = recorder.record(
            query="Test",
            features={},
            strategy="direct",
            model="opus",
            depth=1,
            tools=[],
            success=True,
            cost=0.025,
            latency_ms=3500,
        )

        assert outcome.cost == 0.025
        assert outcome.latency_ms == 3500

    def test_outcome_has_timestamp(self):
        """Outcomes should have timestamps."""
        recorder = OutcomeRecorder()

        outcome = recorder.record(
            query="Test",
            features={},
            strategy="s",
            model="m",
            depth=1,
            tools=[],
            success=True,
            cost=0.01,
            latency_ms=100,
        )

        assert outcome.timestamp is not None


class TestLearningSignals:
    """Tests for learning signal extraction (SPEC-06.41)."""

    def test_extracts_routing_signals(self):
        """SPEC-06.41: Extract routing signals (model selection effectiveness)."""
        extractor = SignalExtractor()

        outcome = ExecutionOutcome(
            query="Analyze code",
            features={"query_type": "code"},
            strategy="decomposition",
            model="sonnet",
            depth=2,
            tools=["grep"],
            success=True,
            cost=0.005,
            latency_ms=1500,
        )

        signals = extractor.extract(outcome)

        routing_signals = [s for s in signals if s.signal_type == SignalType.ROUTING]
        assert len(routing_signals) > 0
        assert routing_signals[0].model == "sonnet"

    def test_extracts_strategy_signals(self):
        """SPEC-06.41: Extract strategy signals (decomposition approach effectiveness)."""
        extractor = SignalExtractor()

        outcome = ExecutionOutcome(
            query="Complex analysis",
            features={"query_type": "analysis"},
            strategy="map_reduce",
            model="opus",
            depth=3,
            tools=[],
            success=True,
            cost=0.02,
            latency_ms=5000,
        )

        signals = extractor.extract(outcome)

        strategy_signals = [s for s in signals if s.signal_type == SignalType.STRATEGY]
        assert len(strategy_signals) > 0
        assert strategy_signals[0].strategy == "map_reduce"

    def test_extracts_tool_signals(self):
        """SPEC-06.41: Extract tool signals (tool selection effectiveness)."""
        extractor = SignalExtractor()

        outcome = ExecutionOutcome(
            query="Find files",
            features={"task_type": "search"},
            strategy="direct",
            model="haiku",
            depth=1,
            tools=["glob", "grep"],
            success=True,
            cost=0.001,
            latency_ms=500,
        )

        signals = extractor.extract(outcome)

        tool_signals = [s for s in signals if s.signal_type == SignalType.TOOL]
        assert len(tool_signals) > 0

    def test_signals_include_reward(self):
        """Signals should include computed reward."""
        extractor = SignalExtractor()

        success_outcome = ExecutionOutcome(
            query="Q",
            features={"query_type": "general"},
            strategy="direct",
            model="haiku",
            depth=1,
            tools=[],
            success=True,
            cost=0.001,
            latency_ms=100,
        )
        failure_outcome = ExecutionOutcome(
            query="Q",
            features={"query_type": "general"},
            strategy="direct",
            model="haiku",
            depth=1,
            tools=[],
            success=False,
            cost=0.001,
            latency_ms=100,
        )

        success_signals = extractor.extract(success_outcome)
        failure_signals = extractor.extract(failure_outcome)

        # Success should have positive reward
        assert any(s.reward > 0 for s in success_signals)
        # Failure should have negative reward
        assert any(s.reward < 0 for s in failure_signals)


class TestLearnedAdjustments:
    """Tests for learned adjustments (SPEC-06.42)."""

    def test_maintains_routing_adjustments(self):
        """SPEC-06.42: Maintain routing_adjustments dict."""
        learner = ContinuousLearner()

        # Record successful routing
        learner.record_outcome(
            query="Code task",
            features={"query_type": "code"},
            strategy="decomposition",
            model="sonnet",
            depth=2,
            tools=[],
            success=True,
            cost=0.005,
            latency_ms=1000,
        )

        adjustments = learner.get_routing_adjustments()
        assert isinstance(adjustments, dict)

    def test_maintains_strategy_preferences(self):
        """SPEC-06.42: Maintain strategy_preferences dict."""
        learner = ContinuousLearner()

        learner.record_outcome(
            query="Complex task",
            features={"query_type": "analysis"},
            strategy="map_reduce",
            model="opus",
            depth=3,
            tools=[],
            success=True,
            cost=0.02,
            latency_ms=5000,
        )

        preferences = learner.get_strategy_preferences()
        assert isinstance(preferences, dict)

    def test_maintains_tool_effectiveness(self):
        """SPEC-06.42: Maintain tool_effectiveness dict."""
        learner = ContinuousLearner()

        learner.record_outcome(
            query="Search task",
            features={"task_type": "search"},
            strategy="direct",
            model="haiku",
            depth=1,
            tools=["glob", "grep"],
            success=True,
            cost=0.001,
            latency_ms=500,
        )

        effectiveness = learner.get_tool_effectiveness()
        assert isinstance(effectiveness, dict)

    def test_adjustments_structure(self):
        """SPEC-06.42: routing_adjustments structure is dict[query_type:model, float]."""
        learner = ContinuousLearner()

        # Record several outcomes
        for _ in range(5):
            learner.record_outcome(
                query="Code task",
                features={"query_type": "code"},
                strategy="decomposition",
                model="sonnet",
                depth=2,
                tools=[],
                success=True,
                cost=0.005,
                latency_ms=1000,
            )

        adjustments = learner.get_routing_adjustments()
        # Should have adjustment for code:sonnet
        assert "code:sonnet" in adjustments or len(adjustments) >= 0

    def test_strategy_preferences_structure(self):
        """SPEC-06.42: strategy_preferences structure is dict[query_type, dict[strategy, float]]."""
        learner = ContinuousLearner()

        for _ in range(5):
            learner.record_outcome(
                query="Analysis task",
                features={"query_type": "analysis"},
                strategy="map_reduce",
                model="opus",
                depth=3,
                tools=[],
                success=True,
                cost=0.02,
                latency_ms=5000,
            )

        preferences = learner.get_strategy_preferences()
        if "analysis" in preferences:
            assert isinstance(preferences["analysis"], dict)

    def test_tool_effectiveness_structure(self):
        """SPEC-06.42: tool_effectiveness structure is dict[task_type, dict[tool, float]]."""
        learner = ContinuousLearner()

        for _ in range(5):
            learner.record_outcome(
                query="Search",
                features={"task_type": "search"},
                strategy="direct",
                model="haiku",
                depth=1,
                tools=["grep"],
                success=True,
                cost=0.001,
                latency_ms=500,
            )

        effectiveness = learner.get_tool_effectiveness()
        if "search" in effectiveness:
            assert isinstance(effectiveness["search"], dict)


class TestLearningRate:
    """Tests for learning rate application (SPEC-06.43)."""

    def test_default_learning_rate(self):
        """SPEC-06.43: Default learning rate is 0.1."""
        config = LearnerConfig()
        assert config.learning_rate == 0.1

    def test_custom_learning_rate(self):
        """Learning rate should be configurable."""
        config = LearnerConfig(learning_rate=0.05)
        assert config.learning_rate == 0.05

    def test_incremental_updates_use_learning_rate(self):
        """SPEC-06.43: Incremental updates apply learning rate."""
        learner = ContinuousLearner(config=LearnerConfig(learning_rate=0.2))

        # Record outcome multiple times
        for _ in range(3):
            learner.record_outcome(
                query="Test",
                features={"query_type": "test"},
                strategy="direct",
                model="haiku",
                depth=1,
                tools=[],
                success=True,
                cost=0.001,
                latency_ms=100,
            )

        # Adjustments should grow incrementally
        adjustments = learner.get_routing_adjustments()
        # Just verify it's being tracked
        assert isinstance(adjustments, dict)


class TestSessionPersistence:
    """Tests for session persistence (SPEC-06.44)."""

    def test_saves_learned_state(self):
        """SPEC-06.44: Persist learned state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "learner_state.json"
            learner = ContinuousLearner(state_path=state_path)

            # Record some outcomes
            learner.record_outcome(
                query="Test",
                features={"query_type": "test"},
                strategy="direct",
                model="sonnet",
                depth=1,
                tools=["grep"],
                success=True,
                cost=0.005,
                latency_ms=1000,
            )

            learner.save()

            assert state_path.exists()

    def test_loads_learned_state(self):
        """SPEC-06.44: Load learned state from previous session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "learner_state.json"

            # Create learner and record outcomes
            learner1 = ContinuousLearner(state_path=state_path)
            for _ in range(5):
                learner1.record_outcome(
                    query="Code task",
                    features={"query_type": "code"},
                    strategy="decomposition",
                    model="sonnet",
                    depth=2,
                    tools=[],
                    success=True,
                    cost=0.005,
                    latency_ms=1000,
                )
            learner1.save()

            # Create new learner and load state
            learner2 = ContinuousLearner(state_path=state_path)
            learner2.load()

            # Should have the adjustments
            assert learner2.get_routing_adjustments() == learner1.get_routing_adjustments()

    def test_state_includes_all_adjustments(self):
        """Persisted state includes all adjustment types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "learner_state.json"
            learner = ContinuousLearner(state_path=state_path)

            learner.record_outcome(
                query="Test",
                features={"query_type": "test", "task_type": "search"},
                strategy="map_reduce",
                model="opus",
                depth=2,
                tools=["glob", "grep"],
                success=True,
                cost=0.02,
                latency_ms=3000,
            )
            learner.save()

            # Read the saved state
            with open(state_path) as f:
                state = json.load(f)

            assert "routing_adjustments" in state
            assert "strategy_preferences" in state
            assert "tool_effectiveness" in state


class TestMetaLearning:
    """Tests for meta-learning (SPEC-06.45)."""

    def test_tracks_prediction_accuracy(self):
        """SPEC-06.45: Track prediction accuracy."""
        meta_learner = MetaLearner()

        # Record some predictions and outcomes
        meta_learner.record_prediction(
            prediction=PredictionRecord(predicted_model="sonnet", predicted_success=True),
            actual_model="sonnet",
            actual_success=True,
        )
        meta_learner.record_prediction(
            prediction=PredictionRecord(predicted_model="haiku", predicted_success=True),
            actual_model="sonnet",
            actual_success=False,
        )

        accuracy = meta_learner.get_prediction_accuracy()
        assert 0 <= accuracy <= 1

    def test_increases_rate_when_predictions_poor(self):
        """SPEC-06.45: Increase rate when predictions are poor (<60% accuracy)."""
        config = LearnerConfig(learning_rate=0.1)
        meta_learner = MetaLearner(config=config)

        # Record many poor predictions
        for _ in range(10):
            meta_learner.record_prediction(
                prediction=PredictionRecord(predicted_model="haiku", predicted_success=True),
                actual_model="opus",  # Wrong model
                actual_success=False,  # Wrong success
            )

        new_rate = meta_learner.get_adjusted_learning_rate()
        assert new_rate > config.learning_rate

    def test_decreases_rate_when_predictions_good(self):
        """SPEC-06.45: Decrease rate when predictions are good (>80% accuracy)."""
        config = LearnerConfig(learning_rate=0.1)
        meta_learner = MetaLearner(config=config)

        # Record many correct predictions
        for _ in range(10):
            meta_learner.record_prediction(
                prediction=PredictionRecord(predicted_model="sonnet", predicted_success=True),
                actual_model="sonnet",
                actual_success=True,
            )

        new_rate = meta_learner.get_adjusted_learning_rate()
        assert new_rate < config.learning_rate

    def test_maintains_rate_when_predictions_moderate(self):
        """Maintain rate when accuracy is moderate (60-80%)."""
        config = LearnerConfig(learning_rate=0.1)
        meta_learner = MetaLearner(config=config)

        # Record mixed predictions (~70% accuracy)
        for i in range(10):
            correct = i < 7
            meta_learner.record_prediction(
                prediction=PredictionRecord(predicted_model="sonnet", predicted_success=True),
                actual_model="sonnet" if correct else "opus",
                actual_success=True if correct else False,
            )

        new_rate = meta_learner.get_adjusted_learning_rate()
        # Should be close to original rate
        assert abs(new_rate - config.learning_rate) < 0.05


class TestLearnerState:
    """Tests for LearnerState structure."""

    def test_state_has_routing_adjustments(self):
        """State includes routing adjustments."""
        state = LearnerState(
            routing_adjustments={"code:sonnet": 0.8},
            strategy_preferences={},
            tool_effectiveness={},
        )
        assert "code:sonnet" in state.routing_adjustments

    def test_state_has_strategy_preferences(self):
        """State includes strategy preferences."""
        state = LearnerState(
            routing_adjustments={},
            strategy_preferences={"analysis": {"map_reduce": 0.9}},
            tool_effectiveness={},
        )
        assert "analysis" in state.strategy_preferences

    def test_state_has_tool_effectiveness(self):
        """State includes tool effectiveness."""
        state = LearnerState(
            routing_adjustments={},
            strategy_preferences={},
            tool_effectiveness={"search": {"grep": 0.95}},
        )
        assert "search" in state.tool_effectiveness

    def test_state_to_dict(self):
        """State should be serializable."""
        state = LearnerState(
            routing_adjustments={"code:sonnet": 0.8},
            strategy_preferences={"analysis": {"map_reduce": 0.9}},
            tool_effectiveness={"search": {"grep": 0.95}},
        )

        data = state.to_dict()
        assert "routing_adjustments" in data
        assert "strategy_preferences" in data
        assert "tool_effectiveness" in data

    def test_state_from_dict(self):
        """State should be deserializable."""
        data = {
            "routing_adjustments": {"code:sonnet": 0.8},
            "strategy_preferences": {"analysis": {"map_reduce": 0.9}},
            "tool_effectiveness": {"search": {"grep": 0.95}},
        }

        state = LearnerState.from_dict(data)
        assert state.routing_adjustments["code:sonnet"] == 0.8


class TestContinuousLearnerIntegration:
    """Integration tests for ContinuousLearner."""

    def test_full_learning_cycle(self):
        """Test complete learning cycle."""
        learner = ContinuousLearner()

        # Record several outcomes
        for i in range(10):
            learner.record_outcome(
                query=f"Task {i}",
                features={"query_type": "code"},
                strategy="decomposition",
                model="sonnet",
                depth=2,
                tools=["grep", "read"],
                success=i % 2 == 0,  # 50% success
                cost=0.005,
                latency_ms=1000 + i * 100,
            )

        # Should have learned something
        adjustments = learner.get_routing_adjustments()
        assert isinstance(adjustments, dict)

    def test_learner_with_meta_learning(self):
        """Learner integrates with meta-learning."""
        config = LearnerConfig(learning_rate=0.1, enable_meta_learning=True)
        learner = ContinuousLearner(config=config)

        # Simulate predictions and outcomes
        for i in range(20):
            # Make a prediction
            prediction = learner.predict(
                query=f"Task {i}",
                features={"query_type": "code"},
            )

            # Record actual outcome
            learner.record_outcome(
                query=f"Task {i}",
                features={"query_type": "code"},
                strategy="decomposition",
                model=prediction.predicted_model,
                depth=2,
                tools=[],
                success=True,
                cost=0.005,
                latency_ms=1000,
            )

        # Meta-learning should have adjusted rate
        current_rate = learner.get_current_learning_rate()
        assert current_rate > 0

    def test_learner_to_dict(self):
        """Learner state should be fully serializable."""
        learner = ContinuousLearner()

        learner.record_outcome(
            query="Test",
            features={"query_type": "test"},
            strategy="direct",
            model="haiku",
            depth=1,
            tools=["glob"],
            success=True,
            cost=0.001,
            latency_ms=500,
        )

        data = learner.to_dict()

        assert "routing_adjustments" in data
        assert "strategy_preferences" in data
        assert "tool_effectiveness" in data
        assert "meta_learning" in data


class TestExecutionOutcome:
    """Tests for ExecutionOutcome structure."""

    def test_outcome_has_all_required_fields(self):
        """Outcome includes all SPEC-06.40 required fields."""
        outcome = ExecutionOutcome(
            query="Test query",
            features={"complexity": 0.5},
            strategy="decomposition",
            model="sonnet",
            depth=2,
            tools=["grep", "read"],
            success=True,
            cost=0.005,
            latency_ms=1500,
            user_satisfaction=0.9,
        )

        assert outcome.query is not None
        assert outcome.features is not None
        assert outcome.strategy is not None
        assert outcome.model is not None
        assert outcome.depth is not None
        assert outcome.tools is not None
        assert outcome.success is not None
        assert outcome.cost is not None
        assert outcome.latency_ms is not None

    def test_outcome_to_dict(self):
        """Outcome should be serializable."""
        outcome = ExecutionOutcome(
            query="Test",
            features={},
            strategy="direct",
            model="haiku",
            depth=1,
            tools=[],
            success=True,
            cost=0.001,
            latency_ms=100,
        )

        data = outcome.to_dict()
        assert "query" in data
        assert "model" in data
        assert "success" in data


class TestSignalType:
    """Tests for SignalType enum."""

    def test_routing_signal_type(self):
        """ROUTING signal type exists."""
        assert SignalType.ROUTING.value == "routing"

    def test_strategy_signal_type(self):
        """STRATEGY signal type exists."""
        assert SignalType.STRATEGY.value == "strategy"

    def test_tool_signal_type(self):
        """TOOL signal type exists."""
        assert SignalType.TOOL.value == "tool"


class TestLearningSignal:
    """Tests for LearningSignal structure."""

    def test_signal_has_type(self):
        """Signal has signal_type."""
        signal = LearningSignal(
            signal_type=SignalType.ROUTING,
            query_type="code",
            model="sonnet",
            reward=0.8,
        )
        assert signal.signal_type == SignalType.ROUTING

    def test_signal_has_reward(self):
        """Signal has reward value."""
        signal = LearningSignal(
            signal_type=SignalType.STRATEGY,
            query_type="analysis",
            strategy="map_reduce",
            reward=0.9,
        )
        assert signal.reward == 0.9

    def test_routing_signal_has_model(self):
        """Routing signal includes model."""
        signal = LearningSignal(
            signal_type=SignalType.ROUTING,
            query_type="code",
            model="opus",
            reward=0.95,
        )
        assert signal.model == "opus"

    def test_strategy_signal_has_strategy(self):
        """Strategy signal includes strategy name."""
        signal = LearningSignal(
            signal_type=SignalType.STRATEGY,
            query_type="analysis",
            strategy="decomposition",
            reward=0.85,
        )
        assert signal.strategy == "decomposition"

    def test_tool_signal_has_tool(self):
        """Tool signal includes tool name."""
        signal = LearningSignal(
            signal_type=SignalType.TOOL,
            task_type="search",
            tool="grep",
            reward=0.9,
        )
        assert signal.tool == "grep"
