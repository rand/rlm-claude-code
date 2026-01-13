"""
Orchestration telemetry for per-heuristic accuracy tracking.

Implements: Feature 3e0.6 - Telemetry-Driven Heuristics

Extends orchestration logging with:
- Per-heuristic accuracy tracking (precision, recall, F1)
- Outcome correlation with decisions
- Training data export with heuristic labels

Usage:
    from src.orchestration_telemetry import OrchestrationTelemetry

    telemetry = OrchestrationTelemetry()
    query_id = telemetry.log_decision_with_heuristics(
        query, decision, heuristics_triggered, heuristics_checked, source
    )
    # ... after execution ...
    telemetry.record_outcome(query_id, success=True, actual_depth=2, cost=0.05)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HeuristicOutcome:
    """Outcome of a single heuristic trigger."""

    heuristic_name: str
    triggered: bool  # Did this heuristic fire?
    predicted_rlm: bool  # Did it predict RLM activation?
    actual_rlm_needed: bool | None = None  # Filled in after execution
    query_id: str = ""
    timestamp: float = 0.0

    # Outcome feedback (filled in after execution)
    execution_succeeded: bool | None = None
    user_satisfaction: str | None = None  # "good", "bad", "neutral"
    actual_depth_used: int | None = None
    actual_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HeuristicAccuracy:
    """Accuracy metrics for a single heuristic."""

    heuristic_name: str
    total_triggers: int = 0
    true_positives: int = 0  # Triggered and RLM was needed
    false_positives: int = 0  # Triggered but RLM wasn't needed
    true_negatives: int = 0  # Didn't trigger and RLM wasn't needed
    false_negatives: int = 0  # Didn't trigger but RLM was needed

    @property
    def precision(self) -> float:
        """Of times triggered, how often was RLM actually needed?"""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Of times RLM was needed, how often did we trigger?"""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        total = (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including computed metrics."""
        return {
            "heuristic_name": self.heuristic_name,
            "total_triggers": self.total_triggers,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
        }


@dataclass
class TelemetryReport:
    """Summary report of heuristic performance."""

    generated_at: str
    total_decisions: int
    decisions_with_feedback: int
    heuristic_accuracies: dict[str, HeuristicAccuracy]
    overall_activation_accuracy: float
    top_performers: list[str]  # Heuristics with highest F1
    underperformers: list[str]  # Heuristics that need tuning
    recommendations: list[str]  # Suggested improvements

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at,
            "total_decisions": self.total_decisions,
            "decisions_with_feedback": self.decisions_with_feedback,
            "heuristic_accuracies": {k: v.to_dict() for k, v in self.heuristic_accuracies.items()},
            "overall_activation_accuracy": self.overall_activation_accuracy,
            "top_performers": self.top_performers,
            "underperformers": self.underperformers,
            "recommendations": self.recommendations,
        }


@dataclass
class TelemetryDecisionLog:
    """Extended decision log with heuristic tracking."""

    # Core decision info
    query_id: str
    query: str
    query_length: int
    timestamp: str
    source: str  # "api", "local", "heuristic"
    latency_ms: float

    # Decision outputs
    activate_rlm: bool
    activation_reason: str
    execution_mode: str
    model_tier: str
    depth_budget: int
    complexity_score: float
    confidence: float

    # Heuristic tracking
    heuristics_triggered: list[str]
    heuristics_checked: list[str]

    # Outcome (filled in later)
    execution_succeeded: bool | None = None
    actual_depth_used: int | None = None
    actual_cost: float | None = None
    rlm_was_helpful: bool | None = None
    user_feedback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelemetryDecisionLog:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry tracking."""

    # Log file paths
    decisions_path: str = "~/.rlm/telemetry_decisions.jsonl"
    feedback_path: str = "~/.rlm/telemetry_feedback.jsonl"

    # Enable/disable
    enabled: bool = True

    # Rotation settings
    max_size_mb: float = 100.0
    max_files: int = 5


# =============================================================================
# OrchestrationTelemetry Class
# =============================================================================


class OrchestrationTelemetry:
    """
    Track and analyze orchestration decision quality.

    Extends OrchestrationLogger with per-heuristic accuracy tracking.
    """

    def __init__(self, config: TelemetryConfig | None = None):
        """
        Initialize telemetry tracker.

        Args:
            config: Optional configuration
        """
        self.config = config or TelemetryConfig()
        self._decisions_path: Path | None = None
        self._feedback_path: Path | None = None

        # In-memory tracking
        self._decisions: dict[str, TelemetryDecisionLog] = {}
        self._heuristic_outcomes: list[HeuristicOutcome] = []
        self._decision_count = 0

        if self.config.enabled:
            self._ensure_paths()

    def _ensure_paths(self) -> None:
        """Ensure log directories exist."""
        self._decisions_path = Path(self.config.decisions_path).expanduser()
        self._decisions_path.parent.mkdir(parents=True, exist_ok=True)

        self._feedback_path = Path(self.config.feedback_path).expanduser()
        self._feedback_path.parent.mkdir(parents=True, exist_ok=True)

    def _check_rotation(self, path: Path) -> None:
        """Check if log file needs rotation."""
        if not path.exists():
            return

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb >= self.config.max_size_mb:
            self._rotate_log(path)

    def _rotate_log(self, path: Path) -> None:
        """Rotate a log file."""
        for i in range(self.config.max_files - 1, 0, -1):
            old_path = path.with_suffix(f".jsonl.{i}")
            new_path = path.with_suffix(f".jsonl.{i + 1}")
            if old_path.exists():
                if i + 1 >= self.config.max_files:
                    old_path.unlink()
                else:
                    old_path.rename(new_path)

        # Rotate current file
        if path.exists():
            path.rename(path.with_suffix(".jsonl.1"))

    def log_decision_with_heuristics(
        self,
        query: str,
        decision: dict[str, Any],
        heuristics_triggered: list[str],
        heuristics_checked: list[str],
        source: str,
        latency_ms: float = 0.0,
        query_id: str | None = None,
    ) -> str:
        """
        Log decision with detailed heuristic breakdown.

        Args:
            query: User query
            decision: Orchestration decision dict
            heuristics_triggered: List of heuristics that fired
            heuristics_checked: All heuristics that were evaluated
            source: Decision source ("api", "local", "heuristic")
            latency_ms: Decision latency
            query_id: Optional query identifier

        Returns:
            query_id for feedback correlation
        """
        if not self.config.enabled:
            return query_id or str(uuid.uuid4())

        qid = query_id or str(uuid.uuid4())

        log_entry = TelemetryDecisionLog(
            query_id=qid,
            query=query[:500],  # Truncate long queries
            query_length=len(query),
            timestamp=datetime.now().isoformat(),
            source=source,
            latency_ms=latency_ms,
            activate_rlm=decision.get("activate_rlm", False),
            activation_reason=decision.get("activation_reason", ""),
            execution_mode=decision.get("execution_mode", "balanced"),
            model_tier=decision.get("model_tier", "balanced"),
            depth_budget=decision.get("depth_budget", 0),
            complexity_score=decision.get("complexity_score", 0.5),
            confidence=decision.get("confidence", 1.0),
            heuristics_triggered=heuristics_triggered,
            heuristics_checked=heuristics_checked,
        )

        # Store in memory
        self._decisions[qid] = log_entry
        self._decision_count += 1

        # Write to file
        if self._decisions_path:
            self._check_rotation(self._decisions_path)
            with open(self._decisions_path, "a") as f:
                f.write(json.dumps(log_entry.to_dict()) + "\n")

        # Track heuristic outcomes
        for heuristic in heuristics_checked:
            triggered = heuristic in heuristics_triggered
            self._heuristic_outcomes.append(
                HeuristicOutcome(
                    heuristic_name=heuristic,
                    triggered=triggered,
                    predicted_rlm=triggered,  # If triggered, it predicted RLM
                    query_id=qid,
                )
            )

        return qid

    def record_outcome(
        self,
        query_id: str,
        execution_succeeded: bool,
        actual_depth_used: int,
        actual_cost: float,
        user_feedback: str | None = None,
        rlm_was_helpful: bool | None = None,
    ) -> None:
        """
        Record outcome for a previous decision.

        Args:
            query_id: ID from log_decision_with_heuristics
            execution_succeeded: Did the task complete successfully?
            actual_depth_used: How deep did recursion go?
            actual_cost: Actual cost incurred
            user_feedback: Explicit user satisfaction signal
            rlm_was_helpful: Was RLM mode actually beneficial?
        """
        if not self.config.enabled:
            return

        # Update in-memory decision
        if query_id in self._decisions:
            decision = self._decisions[query_id]
            decision.execution_succeeded = execution_succeeded
            decision.actual_depth_used = actual_depth_used
            decision.actual_cost = actual_cost
            decision.user_feedback = user_feedback
            decision.rlm_was_helpful = rlm_was_helpful

            # Update heuristic outcomes
            for outcome in self._heuristic_outcomes:
                if outcome.query_id == query_id:
                    outcome.execution_succeeded = execution_succeeded
                    outcome.actual_rlm_needed = rlm_was_helpful
                    outcome.actual_depth_used = actual_depth_used
                    outcome.actual_cost = actual_cost

        # Write feedback to file
        if self._feedback_path:
            self._check_rotation(self._feedback_path)
            feedback_entry = {
                "type": "outcome_feedback",
                "query_id": query_id,
                "execution_succeeded": execution_succeeded,
                "actual_depth_used": actual_depth_used,
                "actual_cost": actual_cost,
                "user_feedback": user_feedback,
                "rlm_was_helpful": rlm_was_helpful,
                "timestamp": datetime.now().isoformat(),
            }
            with open(self._feedback_path, "a") as f:
                f.write(json.dumps(feedback_entry) + "\n")

    def compute_heuristic_accuracy(
        self,
        heuristic_name: str,
    ) -> HeuristicAccuracy:
        """
        Compute accuracy metrics for a specific heuristic.

        Args:
            heuristic_name: Name of heuristic to analyze

        Returns:
            HeuristicAccuracy with precision/recall/F1
        """
        accuracy = HeuristicAccuracy(heuristic_name=heuristic_name)

        for outcome in self._heuristic_outcomes:
            if outcome.heuristic_name != heuristic_name:
                continue

            if outcome.actual_rlm_needed is None:
                continue  # No feedback yet

            if outcome.triggered:
                accuracy.total_triggers += 1
                if outcome.actual_rlm_needed:
                    accuracy.true_positives += 1
                else:
                    accuracy.false_positives += 1
            else:
                if outcome.actual_rlm_needed:
                    accuracy.false_negatives += 1
                else:
                    accuracy.true_negatives += 1

        return accuracy

    def compute_all_accuracies(self) -> dict[str, HeuristicAccuracy]:
        """
        Compute accuracy metrics for all tracked heuristics.

        Returns:
            Dict mapping heuristic name to HeuristicAccuracy
        """
        heuristic_names = {o.heuristic_name for o in self._heuristic_outcomes}
        return {name: self.compute_heuristic_accuracy(name) for name in heuristic_names}

    def generate_report(self) -> TelemetryReport:
        """
        Generate comprehensive telemetry report.

        Returns:
            TelemetryReport with all metrics and recommendations
        """
        accuracies = self.compute_all_accuracies()

        # Count decisions with feedback
        decisions_with_feedback = sum(
            1 for d in self._decisions.values() if d.execution_succeeded is not None
        )

        # Calculate overall activation accuracy
        correct = 0
        total = 0
        for decision in self._decisions.values():
            if decision.rlm_was_helpful is not None:
                total += 1
                if decision.activate_rlm == decision.rlm_was_helpful:
                    correct += 1
        overall_accuracy = correct / total if total > 0 else 0.0

        # Find top performers and underperformers
        sorted_by_f1 = sorted(accuracies.values(), key=lambda a: a.f1_score, reverse=True)

        top_performers = [
            a.heuristic_name for a in sorted_by_f1[:3] if a.f1_score > 0.5 and a.total_triggers > 0
        ]

        underperformers = [
            a.heuristic_name for a in sorted_by_f1 if a.f1_score < 0.3 and a.total_triggers > 5
        ]

        # Generate recommendations
        recommendations = []
        for acc in accuracies.values():
            if acc.total_triggers > 5:
                if acc.precision < 0.3:
                    recommendations.append(
                        f"Tighten '{acc.heuristic_name}' - too many false positives "
                        f"(precision={acc.precision:.2f})"
                    )
                elif acc.recall < 0.3:
                    recommendations.append(
                        f"Loosen '{acc.heuristic_name}' - missing too many cases "
                        f"(recall={acc.recall:.2f})"
                    )

        return TelemetryReport(
            generated_at=datetime.now().isoformat(),
            total_decisions=self._decision_count,
            decisions_with_feedback=decisions_with_feedback,
            heuristic_accuracies=accuracies,
            overall_activation_accuracy=overall_accuracy,
            top_performers=top_performers,
            underperformers=underperformers,
            recommendations=recommendations,
        )

    def export_training_data(
        self,
        output_path: str,
        format: str = "jsonl",
    ) -> int:
        """
        Export logged decisions with outcomes for training.

        Args:
            output_path: Output file path
            format: Export format ("jsonl", "csv")

        Returns:
            Number of records exported
        """
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Filter to decisions with feedback
        exportable = [d for d in self._decisions.values() if d.execution_succeeded is not None]

        if format == "jsonl":
            with open(path, "w") as f:
                for decision in exportable:
                    f.write(json.dumps(decision.to_dict()) + "\n")
        elif format == "csv":
            import csv

            if exportable:
                fieldnames = list(exportable[0].to_dict().keys())
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for decision in exportable:
                        row = decision.to_dict()
                        # Convert lists to strings for CSV
                        row["heuristics_triggered"] = ",".join(row["heuristics_triggered"])
                        row["heuristics_checked"] = ",".join(row["heuristics_checked"])
                        writer.writerow(row)

        return len(exportable)

    def get_heuristic_weights(self) -> dict[str, float]:
        """
        Compute suggested weights for heuristics based on accuracy.

        Heuristics with higher precision get higher weights.

        Returns:
            Dict mapping heuristic name to suggested weight (0.0-1.0)
        """
        accuracies = self.compute_all_accuracies()
        weights = {}

        for name, acc in accuracies.items():
            if acc.total_triggers == 0:
                weights[name] = 0.5  # Default weight for unused heuristics
            else:
                # Weight based on F1 score, with floor at 0.1
                weights[name] = max(0.1, acc.f1_score)

        return weights

    def get_statistics(self) -> dict[str, Any]:
        """Get overall telemetry statistics."""
        accuracies = self.compute_all_accuracies()

        return {
            "total_decisions": self._decision_count,
            "decisions_in_memory": len(self._decisions),
            "heuristic_outcomes_tracked": len(self._heuristic_outcomes),
            "unique_heuristics": len(accuracies),
            "average_f1_score": (
                sum(a.f1_score for a in accuracies.values()) / len(accuracies)
                if accuracies
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all in-memory tracking."""
        self._decisions.clear()
        self._heuristic_outcomes.clear()
        self._decision_count = 0


__all__ = [
    "HeuristicAccuracy",
    "HeuristicOutcome",
    "OrchestrationTelemetry",
    "TelemetryConfig",
    "TelemetryDecisionLog",
    "TelemetryReport",
]
