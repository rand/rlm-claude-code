"""
Continuous learning and self-improvement.

Implements: SPEC-06.40-06.45

Records execution outcomes, extracts learning signals, maintains
learned adjustments, and implements meta-learning for adaptive
learning rate control.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SignalType(Enum):
    """
    Type of learning signal.

    Implements: SPEC-06.41
    """

    ROUTING = "routing"
    STRATEGY = "strategy"
    TOOL = "tool"


@dataclass
class ExecutionOutcome:
    """
    Record of an execution outcome.

    Implements: SPEC-06.40
    """

    query: str
    features: dict[str, Any]
    strategy: str
    model: str
    depth: int
    tools: list[str]
    success: bool
    cost: float
    latency_ms: int
    user_satisfaction: float | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "features": self.features,
            "strategy": self.strategy,
            "model": self.model,
            "depth": self.depth,
            "tools": self.tools,
            "success": self.success,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "user_satisfaction": self.user_satisfaction,
            "timestamp": self.timestamp,
        }


@dataclass
class LearningSignal:
    """
    Learning signal extracted from an outcome.

    Implements: SPEC-06.41
    """

    signal_type: SignalType
    reward: float
    query_type: str | None = None
    task_type: str | None = None
    model: str | None = None
    strategy: str | None = None
    tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "reward": self.reward,
            "query_type": self.query_type,
            "task_type": self.task_type,
            "model": self.model,
            "strategy": self.strategy,
            "tool": self.tool,
        }


@dataclass
class LearnerConfig:
    """
    Configuration for continuous learner.

    Implements: SPEC-06.43
    """

    learning_rate: float = 0.1
    enable_meta_learning: bool = False
    min_learning_rate: float = 0.01
    max_learning_rate: float = 0.5
    poor_accuracy_threshold: float = 0.6
    good_accuracy_threshold: float = 0.8
    rate_adjustment_factor: float = 1.5


@dataclass
class LearnerState:
    """
    Persisted state of the learner.

    Implements: SPEC-06.42, SPEC-06.44
    """

    routing_adjustments: dict[str, float]
    strategy_preferences: dict[str, dict[str, float]]
    tool_effectiveness: dict[str, dict[str, float]]
    meta_learning_rate: float = 0.1
    prediction_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "routing_adjustments": self.routing_adjustments,
            "strategy_preferences": self.strategy_preferences,
            "tool_effectiveness": self.tool_effectiveness,
            "meta_learning_rate": self.meta_learning_rate,
            "prediction_history": self.prediction_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnerState:
        """Deserialize from dictionary."""
        return cls(
            routing_adjustments=data.get("routing_adjustments", {}),
            strategy_preferences=data.get("strategy_preferences", {}),
            tool_effectiveness=data.get("tool_effectiveness", {}),
            meta_learning_rate=data.get("meta_learning_rate", 0.1),
            prediction_history=data.get("prediction_history", []),
        )


@dataclass
class PredictionRecord:
    """Record of a prediction for meta-learning."""

    predicted_model: str
    predicted_success: bool
    predicted_strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "predicted_model": self.predicted_model,
            "predicted_success": self.predicted_success,
            "predicted_strategy": self.predicted_strategy,
        }


class OutcomeRecorder:
    """
    Records execution outcomes.

    Implements: SPEC-06.40
    """

    def __init__(self) -> None:
        """Initialize outcome recorder."""
        self.outcomes: list[ExecutionOutcome] = []

    def record(
        self,
        query: str,
        features: dict[str, Any],
        strategy: str,
        model: str,
        depth: int,
        tools: list[str],
        success: bool,
        cost: float,
        latency_ms: int,
        user_satisfaction: float | None = None,
    ) -> ExecutionOutcome:
        """
        Record an execution outcome.

        Implements: SPEC-06.40

        Args:
            query: The executed query
            features: Query features
            strategy: Strategy used
            model: Model used
            depth: Recursion depth reached
            tools: Tools used
            success: Whether execution succeeded
            cost: Cost in dollars
            latency_ms: Latency in milliseconds
            user_satisfaction: Optional user satisfaction score

        Returns:
            The recorded ExecutionOutcome
        """
        outcome = ExecutionOutcome(
            query=query,
            features=features,
            strategy=strategy,
            model=model,
            depth=depth,
            tools=tools,
            success=success,
            cost=cost,
            latency_ms=latency_ms,
            user_satisfaction=user_satisfaction,
        )
        self.outcomes.append(outcome)
        return outcome

    def get_outcomes(self) -> list[ExecutionOutcome]:
        """Get all recorded outcomes."""
        return self.outcomes


class SignalExtractor:
    """
    Extracts learning signals from outcomes.

    Implements: SPEC-06.41
    """

    def extract(self, outcome: ExecutionOutcome) -> list[LearningSignal]:
        """
        Extract learning signals from an outcome.

        Implements: SPEC-06.41

        Args:
            outcome: Execution outcome to extract from

        Returns:
            List of learning signals
        """
        signals: list[LearningSignal] = []

        # Compute base reward
        base_reward = self._compute_reward(outcome)

        # Extract routing signal
        query_type = outcome.features.get("query_type", "general")
        signals.append(
            LearningSignal(
                signal_type=SignalType.ROUTING,
                query_type=query_type,
                model=outcome.model,
                reward=base_reward,
            )
        )

        # Extract strategy signal
        signals.append(
            LearningSignal(
                signal_type=SignalType.STRATEGY,
                query_type=query_type,
                strategy=outcome.strategy,
                reward=base_reward,
            )
        )

        # Extract tool signals
        task_type = outcome.features.get("task_type", "general")
        for tool in outcome.tools:
            signals.append(
                LearningSignal(
                    signal_type=SignalType.TOOL,
                    task_type=task_type,
                    tool=tool,
                    reward=base_reward,
                )
            )

        return signals

    def _compute_reward(self, outcome: ExecutionOutcome) -> float:
        """
        Compute reward from outcome.

        Rewards are:
        - Positive for success
        - Negative for failure
        - Adjusted by user satisfaction and efficiency
        """
        if not outcome.success:
            return -0.5

        # Base reward for success
        reward = 0.7

        # Bonus for user satisfaction
        if outcome.user_satisfaction is not None:
            reward += (outcome.user_satisfaction - 0.5) * 0.4

        # Efficiency bonus (lower cost/latency = higher reward)
        if outcome.cost < 0.01:
            reward += 0.1
        if outcome.latency_ms < 1000:
            reward += 0.1

        return min(1.0, reward)


class MetaLearner:
    """
    Meta-learning for adaptive learning rate control.

    Implements: SPEC-06.45
    """

    def __init__(self, config: LearnerConfig | None = None) -> None:
        """
        Initialize meta-learner.

        Args:
            config: Learner configuration
        """
        self.config = config or LearnerConfig()
        self.predictions: list[dict[str, Any]] = []

    def record_prediction(
        self,
        prediction: PredictionRecord,
        actual_model: str,
        actual_success: bool,
    ) -> None:
        """
        Record a prediction and its actual outcome.

        Implements: SPEC-06.45

        Args:
            prediction: The prediction made
            actual_model: Actual model used
            actual_success: Actual success outcome
        """
        correct_model = prediction.predicted_model == actual_model
        correct_success = prediction.predicted_success == actual_success

        self.predictions.append(
            {
                "prediction": prediction.to_dict(),
                "actual_model": actual_model,
                "actual_success": actual_success,
                "correct_model": correct_model,
                "correct_success": correct_success,
                "timestamp": time.time(),
            }
        )

    def get_prediction_accuracy(self) -> float:
        """
        Get overall prediction accuracy.

        Implements: SPEC-06.45

        Returns:
            Accuracy as float 0-1
        """
        if not self.predictions:
            return 0.5  # Default when no predictions

        correct = sum(1 for p in self.predictions if p["correct_model"] and p["correct_success"])
        return correct / len(self.predictions)

    def get_adjusted_learning_rate(self) -> float:
        """
        Get learning rate adjusted by prediction accuracy.

        Implements: SPEC-06.45
        - Increase rate when predictions are poor (<60% accuracy)
        - Decrease rate when predictions are good (>80% accuracy)

        Returns:
            Adjusted learning rate
        """
        if len(self.predictions) < 5:
            return self.config.learning_rate

        accuracy = self.get_prediction_accuracy()

        if accuracy < self.config.poor_accuracy_threshold:
            # Increase learning rate - we need to learn more
            new_rate = self.config.learning_rate * self.config.rate_adjustment_factor
            return min(new_rate, self.config.max_learning_rate)
        elif accuracy > self.config.good_accuracy_threshold:
            # Decrease learning rate - we're doing well
            new_rate = self.config.learning_rate / self.config.rate_adjustment_factor
            return max(new_rate, self.config.min_learning_rate)
        else:
            # Maintain current rate
            return self.config.learning_rate


class ContinuousLearner:
    """
    Main continuous learning system.

    Implements: SPEC-06.40-06.45
    """

    def __init__(
        self,
        config: LearnerConfig | None = None,
        state_path: Path | None = None,
    ) -> None:
        """
        Initialize continuous learner.

        Args:
            config: Learner configuration
            state_path: Path for state persistence
        """
        self.config = config or LearnerConfig()
        self.state_path = state_path
        self.recorder = OutcomeRecorder()
        self.extractor = SignalExtractor()
        self.meta_learner = MetaLearner(config=self.config)

        # Learned state (SPEC-06.42)
        self.state = LearnerState(
            routing_adjustments={},
            strategy_preferences={},
            tool_effectiveness={},
        )

    def record_outcome(
        self,
        query: str,
        features: dict[str, Any],
        strategy: str,
        model: str,
        depth: int,
        tools: list[str],
        success: bool,
        cost: float,
        latency_ms: int,
        user_satisfaction: float | None = None,
    ) -> ExecutionOutcome:
        """
        Record an execution outcome and update learned adjustments.

        Implements: SPEC-06.40, SPEC-06.42

        Args:
            query: The executed query
            features: Query features
            strategy: Strategy used
            model: Model used
            depth: Recursion depth reached
            tools: Tools used
            success: Whether execution succeeded
            cost: Cost in dollars
            latency_ms: Latency in milliseconds
            user_satisfaction: Optional user satisfaction score

        Returns:
            The recorded ExecutionOutcome
        """
        # Record outcome (SPEC-06.40)
        outcome = self.recorder.record(
            query=query,
            features=features,
            strategy=strategy,
            model=model,
            depth=depth,
            tools=tools,
            success=success,
            cost=cost,
            latency_ms=latency_ms,
            user_satisfaction=user_satisfaction,
        )

        # Extract signals (SPEC-06.41)
        signals = self.extractor.extract(outcome)

        # Update adjustments (SPEC-06.42, SPEC-06.43)
        learning_rate = self.get_current_learning_rate()
        self._update_adjustments(signals, learning_rate)

        return outcome

    def _update_adjustments(
        self,
        signals: list[LearningSignal],
        learning_rate: float,
    ) -> None:
        """
        Update learned adjustments from signals.

        Implements: SPEC-06.42, SPEC-06.43
        """
        for signal in signals:
            if signal.signal_type == SignalType.ROUTING:
                self._update_routing(signal, learning_rate)
            elif signal.signal_type == SignalType.STRATEGY:
                self._update_strategy(signal, learning_rate)
            elif signal.signal_type == SignalType.TOOL:
                self._update_tool(signal, learning_rate)

    def _update_routing(self, signal: LearningSignal, learning_rate: float) -> None:
        """Update routing adjustments."""
        if signal.query_type and signal.model:
            key = f"{signal.query_type}:{signal.model}"
            current = self.state.routing_adjustments.get(key, 0.0)
            # Incremental update with learning rate (SPEC-06.43)
            self.state.routing_adjustments[key] = current + learning_rate * (
                signal.reward - current
            )

    def _update_strategy(self, signal: LearningSignal, learning_rate: float) -> None:
        """Update strategy preferences."""
        if signal.query_type and signal.strategy:
            if signal.query_type not in self.state.strategy_preferences:
                self.state.strategy_preferences[signal.query_type] = {}

            prefs = self.state.strategy_preferences[signal.query_type]
            current = prefs.get(signal.strategy, 0.0)
            prefs[signal.strategy] = current + learning_rate * (signal.reward - current)

    def _update_tool(self, signal: LearningSignal, learning_rate: float) -> None:
        """Update tool effectiveness."""
        if signal.task_type and signal.tool:
            if signal.task_type not in self.state.tool_effectiveness:
                self.state.tool_effectiveness[signal.task_type] = {}

            eff = self.state.tool_effectiveness[signal.task_type]
            current = eff.get(signal.tool, 0.0)
            eff[signal.tool] = current + learning_rate * (signal.reward - current)

    def predict(
        self,
        query: str,
        features: dict[str, Any],
    ) -> PredictionRecord:
        """
        Make a prediction for meta-learning.

        Args:
            query: Query to predict for
            features: Query features

        Returns:
            Prediction record
        """
        query_type = features.get("query_type", "general")

        # Find best model for query type
        best_model = "sonnet"  # Default
        best_score = 0.0
        for key, score in self.state.routing_adjustments.items():
            if key.startswith(f"{query_type}:") and score > best_score:
                best_model = key.split(":")[1]
                best_score = score

        # Predict success based on historical performance
        predicted_success = best_score > 0.5

        prediction = PredictionRecord(
            predicted_model=best_model,
            predicted_success=predicted_success,
        )

        return prediction

    def get_current_learning_rate(self) -> float:
        """
        Get current learning rate, adjusted by meta-learning if enabled.

        Implements: SPEC-06.43, SPEC-06.45
        """
        if self.config.enable_meta_learning:
            return self.meta_learner.get_adjusted_learning_rate()
        return self.config.learning_rate

    def get_routing_adjustments(self) -> dict[str, float]:
        """
        Get routing adjustments.

        Implements: SPEC-06.42
        """
        return self.state.routing_adjustments

    def get_strategy_preferences(self) -> dict[str, dict[str, float]]:
        """
        Get strategy preferences.

        Implements: SPEC-06.42
        """
        return self.state.strategy_preferences

    def get_tool_effectiveness(self) -> dict[str, dict[str, float]]:
        """
        Get tool effectiveness.

        Implements: SPEC-06.42
        """
        return self.state.tool_effectiveness

    def save(self) -> None:
        """
        Save learned state to file.

        Implements: SPEC-06.44
        """
        if self.state_path is None:
            return

        data = self.state.to_dict()
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """
        Load learned state from file.

        Implements: SPEC-06.44
        """
        if self.state_path is None or not self.state_path.exists():
            return

        with open(self.state_path) as f:
            data = json.load(f)

        self.state = LearnerState.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize learner state."""
        return {
            "routing_adjustments": self.state.routing_adjustments,
            "strategy_preferences": self.state.strategy_preferences,
            "tool_effectiveness": self.state.tool_effectiveness,
            "meta_learning": {
                "current_rate": self.get_current_learning_rate(),
                "prediction_count": len(self.meta_learner.predictions),
            },
        }


__all__ = [
    "ContinuousLearner",
    "ExecutionOutcome",
    "LearnerConfig",
    "LearnerState",
    "LearningSignal",
    "MetaLearner",
    "OutcomeRecorder",
    "PredictionRecord",
    "SignalExtractor",
    "SignalType",
]
