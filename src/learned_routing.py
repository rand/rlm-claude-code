"""
Learned model routing based on query characteristics.

Implements: SPEC-06.20-06.26

Routes queries to appropriate models based on difficulty estimation,
cost sensitivity, and learned outcomes from previous routing decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelProfile:
    """
    Profile describing a model's capabilities and cost.

    Implements: SPEC-06.21
    """

    name: str
    strengths: list[str]
    cost_per_1k: float
    quality_baseline: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "strengths": self.strengths,
            "cost_per_1k": self.cost_per_1k,
            "quality_baseline": self.quality_baseline,
        }


# Default model profiles
DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "haiku": ModelProfile(
        name="haiku",
        strengths=["simple", "factual", "quick", "classification"],
        cost_per_1k=0.00025,
        quality_baseline=0.7,
    ),
    "sonnet": ModelProfile(
        name="sonnet",
        strengths=["code", "analysis", "reasoning", "balanced"],
        cost_per_1k=0.003,
        quality_baseline=0.85,
    ),
    "opus": ModelProfile(
        name="opus",
        strengths=["complex", "creative", "nuanced", "research", "architecture"],
        cost_per_1k=0.015,
        quality_baseline=0.95,
    ),
}


@dataclass
class QueryFeatures:
    """Features extracted from a query for routing decisions."""

    length: int
    estimated_complexity: float
    domain_hints: list[str]
    has_code: bool = False
    has_math: bool = False
    question_type: str = "unknown"

    @classmethod
    def from_query(cls, query: str) -> QueryFeatures:
        """Extract features from a query string."""
        query_lower = query.lower()

        # Length-based complexity
        length = len(query)
        base_complexity = min(1.0, length / 500)

        # Complexity indicators
        complexity_words = [
            "analyze",
            "compare",
            "evaluate",
            "synthesize",
            "architect",
            "design",
            "implement",
            "optimize",
            "debug",
            "refactor",
            "implications",
            "tradeoffs",
            "considerations",
        ]
        complexity_count = sum(1 for w in complexity_words if w in query_lower)
        complexity = min(1.0, base_complexity + complexity_count * 0.15)

        # Domain hints
        domain_hints = []
        domain_patterns = {
            "code": r"\b(code|function|class|method|debug|implement|bug|error)\b",
            "programming": r"\b(python|javascript|rust|typescript|api|database)\b",
            "math": r"\b(calculate|compute|formula|equation|algorithm)\b",
            "analysis": r"\b(analyze|compare|evaluate|assess)\b",
            "creative": r"\b(write|create|design|generate|story|poem)\b",
        }

        for domain, pattern in domain_patterns.items():
            if re.search(pattern, query_lower):
                domain_hints.append(domain)

        # Code detection
        has_code = bool(re.search(r"```|def |class |function |import ", query))

        # Math detection
        has_math = bool(re.search(r"\d+\s*[\+\-\*\/\^]\s*\d+|equation|formula", query_lower))

        # Question type
        if query_lower.startswith(("what is", "who is", "when")):
            question_type = "factual"
        elif query_lower.startswith(("how", "why", "explain")):
            question_type = "explanatory"
        elif any(w in query_lower for w in ["analyze", "compare", "evaluate"]):
            question_type = "analytical"
        else:
            question_type = "task"

        return cls(
            length=length,
            estimated_complexity=complexity,
            domain_hints=domain_hints,
            has_code=has_code,
            has_math=has_math,
            question_type=question_type,
        )


@dataclass
class DifficultyEstimate:
    """
    Estimated difficulty of a query.

    Implements: SPEC-06.22
    """

    reasoning_depth: float  # 0-1, how much reasoning required
    domain_specificity: float  # 0-1, how specialized the domain
    ambiguity_level: float  # 0-1, how ambiguous the query
    context_size: int  # Approximate tokens needed

    def overall_difficulty(self) -> float:
        """Compute overall difficulty score."""
        return (
            self.reasoning_depth * 0.4
            + self.domain_specificity * 0.3
            + self.ambiguity_level * 0.2
            + min(1.0, self.context_size / 4000) * 0.1
        )


class DifficultyEstimator:
    """
    Estimates query difficulty for routing decisions.

    Implements: SPEC-06.22
    """

    def estimate(self, query: str) -> DifficultyEstimate:
        """
        Estimate difficulty of a query.

        Args:
            query: The query to analyze

        Returns:
            DifficultyEstimate with component scores
        """
        features = QueryFeatures.from_query(query)
        query_lower = query.lower()

        # Reasoning depth
        reasoning_indicators = [
            "if",
            "then",
            "because",
            "therefore",
            "implies",
            "conclude",
            "reason",
            "logic",
            "proof",
            "derive",
        ]
        reasoning_count = sum(1 for w in reasoning_indicators if w in query_lower)
        reasoning_depth = min(1.0, features.estimated_complexity + reasoning_count * 0.1)

        # Domain specificity
        specific_terms = [
            "algorithm",
            "architecture",
            "protocol",
            "theorem",
            "paradigm",
            "methodology",
            "framework",
            "specification",
        ]
        specificity_count = sum(1 for w in specific_terms if w in query_lower)
        domain_specificity = min(1.0, len(features.domain_hints) * 0.2 + specificity_count * 0.15)

        # Ambiguity
        ambiguous_indicators = ["this", "it", "that", "good", "better", "best"]
        clear_indicators = ["specifically", "exactly", "precisely", "calculate"]
        ambiguity = 0.5  # Base ambiguity
        ambiguity += sum(0.1 for w in ambiguous_indicators if w in query_lower)
        ambiguity -= sum(0.15 for w in clear_indicators if w in query_lower)
        ambiguity_level = max(0.0, min(1.0, ambiguity))

        # Context size (rough estimate: 1.3 tokens per word)
        context_size = int(len(query.split()) * 1.3)

        return DifficultyEstimate(
            reasoning_depth=reasoning_depth,
            domain_specificity=domain_specificity,
            ambiguity_level=ambiguity_level,
            context_size=context_size,
        )


@dataclass
class RoutingDecision:
    """Decision about which model to use."""

    model: str
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model": self.model,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class OutcomeRecord:
    """
    Record of a routing outcome for learning.

    Implements: SPEC-06.26
    """

    query: str
    query_embedding: list[float]
    model: str
    success: bool
    quality_score: float
    cost: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "query_embedding": self.query_embedding,
            "model": self.model,
            "success": self.success,
            "quality_score": self.quality_score,
            "cost": self.cost,
        }


@dataclass
class RouterConfig:
    """
    Configuration for routing behavior.

    Implements: SPEC-06.25
    """

    confidence_threshold: float = 0.8
    max_escalations: int = 2
    cascade_order: list[str] = field(default_factory=lambda: ["haiku", "sonnet", "opus"])
    learning_rate: float = 0.1


class LearnedRouter:
    """
    Routes queries to models based on learned patterns.

    Implements: SPEC-06.20-06.26
    """

    def __init__(
        self,
        profiles: dict[str, ModelProfile] | None = None,
        cost_sensitivity: float = 0.5,
        config: RouterConfig | None = None,
    ):
        """
        Initialize learned router.

        Args:
            profiles: Model profiles (defaults to standard profiles)
            cost_sensitivity: 0=quality only, 1=cost only (SPEC-06.23)
            config: Router configuration
        """
        self.profiles = profiles or DEFAULT_PROFILES
        self.cost_sensitivity = cost_sensitivity
        self.config = config or RouterConfig()
        self.estimator = DifficultyEstimator()
        self.outcome_history: list[OutcomeRecord] = []
        self.learned_adjustments: dict[str, dict[str, float]] = {}

    def route(self, query: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """
        Route query to appropriate model.

        Implements: SPEC-06.20

        Args:
            query: Query to route
            context: Optional context information

        Returns:
            RoutingDecision with model and reasoning
        """
        # Estimate difficulty
        difficulty = self.estimator.estimate(query)
        features = QueryFeatures.from_query(query)

        # Score each model
        scores: dict[str, float] = {}
        for name, profile in self.profiles.items():
            score = self._score_model(profile, difficulty, features)
            scores[name] = score

        # Apply cost sensitivity (SPEC-06.23)
        if self.cost_sensitivity == 0.0:
            # Quality only - pick highest quality
            best_model = "opus"
        elif self.cost_sensitivity == 1.0:
            # Cost only - pick cheapest
            best_model = "haiku"
        else:
            # Weighted selection
            best_model = max(scores, key=lambda m: scores[m])

        # Apply learned adjustments
        best_model = self._apply_adjustments(best_model, features)

        confidence = scores.get(best_model, 0.5)

        return RoutingDecision(
            model=best_model,
            confidence=confidence,
            reasoning=self._generate_reasoning(best_model, difficulty, features),
        )

    def _score_model(
        self,
        profile: ModelProfile,
        difficulty: DifficultyEstimate,
        features: QueryFeatures,
    ) -> float:
        """Score a model for the given query."""
        # Base score from quality
        quality_score = profile.quality_baseline

        # Difficulty match
        overall_diff = difficulty.overall_difficulty()
        if (
            overall_diff < 0.3
            and profile.name == "haiku"
            or overall_diff > 0.7
            and profile.name == "opus"
            or 0.3 <= overall_diff <= 0.7
            and profile.name == "sonnet"
        ):
            quality_score += 0.1

        # Domain match
        for hint in features.domain_hints:
            if hint in profile.strengths or any(s in hint for s in profile.strengths):
                quality_score += 0.05

        # Cost factor
        max_cost = max(p.cost_per_1k for p in self.profiles.values())
        cost_score = 1 - (profile.cost_per_1k / max_cost)

        # Combine with cost sensitivity
        final_score = (
            1 - self.cost_sensitivity
        ) * quality_score + self.cost_sensitivity * cost_score

        return min(1.0, final_score)

    def _apply_adjustments(self, model: str, features: QueryFeatures) -> str:
        """Apply learned adjustments to model selection."""
        # Check if we have learned preferences for this domain
        for domain in features.domain_hints:
            if domain in self.learned_adjustments:
                adj = self.learned_adjustments[domain]
                # Find model with highest adjustment
                if adj:
                    best = max(adj, key=lambda m: adj[m])
                    if adj[best] > 0.2:  # Significant preference
                        return best
        return model

    def _generate_reasoning(
        self,
        model: str,
        difficulty: DifficultyEstimate,
        features: QueryFeatures,
    ) -> str:
        """Generate reasoning for the routing decision."""
        parts = []

        if difficulty.overall_difficulty() < 0.3:
            parts.append("Simple query")
        elif difficulty.overall_difficulty() > 0.7:
            parts.append("Complex query requiring deep reasoning")
        else:
            parts.append("Moderate complexity")

        if features.domain_hints:
            parts.append(f"Domain: {', '.join(features.domain_hints)}")

        parts.append(f"Selected {model}")

        return ". ".join(parts)

    def record_outcome(
        self,
        query: str,
        decision: RoutingDecision,
        success: bool,
        quality_score: float,
    ) -> None:
        """
        Record outcome for learning.

        Implements: SPEC-06.26

        Args:
            query: Original query
            decision: Routing decision made
            success: Whether the outcome was successful
            quality_score: Quality score (0-1)
        """
        # Simple embedding (in production, use real embeddings)
        embedding = [float(ord(c) % 100) / 100 for c in query[:50]]

        profile = self.profiles.get(decision.model)
        cost = profile.cost_per_1k * len(query.split()) / 1000 if profile else 0

        record = OutcomeRecord(
            query=query,
            query_embedding=embedding,
            model=decision.model,
            success=success,
            quality_score=quality_score,
            cost=cost,
        )

        self.outcome_history.append(record)

        # Update learned adjustments
        self._update_adjustments(query, decision.model, success, quality_score)

    def _update_adjustments(
        self,
        query: str,
        model: str,
        success: bool,
        quality_score: float,
    ) -> None:
        """Update learned adjustments based on outcome."""
        features = QueryFeatures.from_query(query)

        for domain in features.domain_hints:
            if domain not in self.learned_adjustments:
                self.learned_adjustments[domain] = {}

            current = self.learned_adjustments[domain].get(model, 0.0)
            reward = quality_score if success else -0.5

            # Incremental update with learning rate
            self.learned_adjustments[domain][model] = current + self.config.learning_rate * (
                reward - current
            )

    def get_outcome_history(self) -> list[OutcomeRecord]:
        """Get recorded outcomes."""
        return self.outcome_history

    def get_learned_adjustments(self) -> dict[str, dict[str, float]]:
        """Get learned adjustments."""
        return self.learned_adjustments

    def to_dict(self) -> dict[str, Any]:
        """Serialize router state."""
        return {
            "outcome_history": [r.to_dict() for r in self.outcome_history],
            "learned_adjustments": self.learned_adjustments,
            "cost_sensitivity": self.cost_sensitivity,
        }


@dataclass
class CascadeAttempt:
    """Record of a cascade attempt."""

    model: str
    confidence: float
    escalated: bool


@dataclass
class CascadeResult:
    """Result of cascading routing."""

    final_model: str
    attempts: list[CascadeAttempt]
    total_escalations: int


class CascadingRouter:
    """
    Router that cascades through models on low confidence.

    Implements: SPEC-06.24, SPEC-06.25
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
        profiles: dict[str, ModelProfile] | None = None,
    ):
        """
        Initialize cascading router.

        Args:
            config: Router configuration
            profiles: Model profiles
        """
        self.config = config or RouterConfig()
        self.profiles = profiles or DEFAULT_PROFILES
        self.base_router = LearnedRouter(profiles=profiles)

    def route_with_cascade(
        self,
        query: str,
        mock_confidences: list[float] | None = None,
    ) -> CascadeResult:
        """
        Route with cascading escalation.

        Implements: SPEC-06.24

        Args:
            query: Query to route
            mock_confidences: Optional confidence values for testing

        Returns:
            CascadeResult with attempts and final model
        """
        attempts: list[CascadeAttempt] = []
        escalations = 0

        for i, model in enumerate(self.config.cascade_order):
            # Get confidence (mock or estimated)
            if mock_confidences and i < len(mock_confidences):
                confidence = mock_confidences[i]
            else:
                # Estimate confidence based on model capability
                profile = self.profiles.get(model)
                confidence = profile.quality_baseline if profile else 0.5

            attempt = CascadeAttempt(
                model=model,
                confidence=confidence,
                escalated=i > 0,
            )
            attempts.append(attempt)

            # Check if confidence is sufficient
            if confidence >= self.config.confidence_threshold:
                break

            # Check escalation limit
            if escalations >= self.config.max_escalations:
                break

            escalations += 1

        return CascadeResult(
            final_model=attempts[-1].model,
            attempts=attempts,
            total_escalations=escalations,
        )


__all__ = [
    "CascadeAttempt",
    "CascadeResult",
    "CascadingRouter",
    "DifficultyEstimate",
    "DifficultyEstimator",
    "LearnedRouter",
    "ModelProfile",
    "OutcomeRecord",
    "QueryFeatures",
    "RouterConfig",
    "RoutingDecision",
]
