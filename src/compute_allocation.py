"""
Compute-optimal allocation for RLM operations.

Implements: SPEC-07.10-07.15 Compute-Optimal Allocation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelTier(Enum):
    """Model tier for compute allocation."""

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"

    def __lt__(self, other: ModelTier) -> bool:
        """Compare model tiers by cost/capability."""
        order = {ModelTier.HAIKU: 0, ModelTier.SONNET: 1, ModelTier.OPUS: 2}
        return order[self] < order[other]

    def __le__(self, other: ModelTier) -> bool:
        """Compare model tiers by cost/capability."""
        return self == other or self < other


class TaskType(Enum):
    """Type of task for difficulty estimation."""

    CODE = "code"
    DEBUG = "debug"
    ANALYSIS = "analysis"
    QUESTION = "question"
    REFACTOR = "refactor"


@dataclass
class AllocationReasoning:
    """
    Reasoning for compute allocation decisions.

    Implements: SPEC-07.15
    """

    difficulty_score: float
    difficulty_factors: list[str]
    tradeoff_explanation: str
    budget_constraint_applied: bool = False
    optimization_mode: str = "balanced"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "difficulty_score": self.difficulty_score,
            "factors": self.difficulty_factors,
            "tradeoff": self.tradeoff_explanation,
            "budget_constrained": self.budget_constraint_applied,
            "mode": self.optimization_mode,
        }


@dataclass
class DifficultyEstimate:
    """
    Estimated difficulty of a query.

    Implements: SPEC-07.12
    """

    score: float  # 0.0 to 1.0
    task_type: TaskType
    context_complexity: float
    query_complexity: float
    factors: list[str] = field(default_factory=list)


@dataclass
class ComputeAllocation:
    """
    Compute allocation for an RLM operation.

    Implements: SPEC-07.11
    """

    depth_budget: int
    model_tier: ModelTier
    parallel_calls: int
    timeout_ms: int
    estimated_cost: float
    reasoning: AllocationReasoning | None = None


# Cost estimates per 1K tokens (simplified)
MODEL_COSTS = {
    ModelTier.HAIKU: {"input": 0.00025, "output": 0.00125},
    ModelTier.SONNET: {"input": 0.003, "output": 0.015},
    ModelTier.OPUS: {"input": 0.015, "output": 0.075},
}


class ComputeAllocator:
    """
    Allocator for compute-optimal resource allocation.

    Implements: SPEC-07.10-07.15

    Dynamically allocates compute resources based on query difficulty,
    context complexity, and budget constraints.
    """

    # Task type detection patterns
    TASK_PATTERNS = {
        TaskType.DEBUG: [
            r"\bdebug\b",
            r"\bbug\b",
            r"\berror\b",
            r"\bfix\b",
            r"\bissue\b",
            r"\bsegfault\b",
            r"\bcrash\b",
        ],
        TaskType.REFACTOR: [
            r"\brefactor\b",
            r"\brestructure\b",
            r"\breorganize\b",
            r"\bclean\s*up\b",
        ],
        TaskType.CODE: [
            r"\bwrite\b",
            r"\bimplement\b",
            r"\bcreate\b",
            r"\badd\b.*\bfunction\b",
            r"\badd\b.*\bfeature\b",
        ],
        TaskType.ANALYSIS: [
            r"\banalyze\b",
            r"\bexplain\b",
            r"\bdescribe\b",
            r"\bfind\b.*\bvulnerabilit",
            r"\breview\b",
        ],
        TaskType.QUESTION: [
            r"\bwhat\b",
            r"\bhow\b",
            r"\bwhy\b",
            r"\bwhen\b",
            r"\bwhere\b",
            r"\?$",
        ],
    }

    # Complexity indicators
    COMPLEXITY_INDICATORS = [
        (r"\ball\b", 0.2),
        (r"\bentire\b", 0.2),
        (r"\bevery\b", 0.15),
        (r"\bcomprehensive\b", 0.2),
        (r"\bthorough\b", 0.15),
        (r"\bdetailed\b", 0.1),
        (r"\bmultiple\b", 0.1),
        (r"\bsecurity\b", 0.15),
        (r"\bvulnerabilit", 0.2),
        (r"\bverify\b", 0.1),
        (r"\btest\b", 0.1),
    ]

    def __init__(
        self,
        default_depth: int = 2,
        default_model: ModelTier = ModelTier.SONNET,
        default_timeout_ms: int = 60000,
    ):
        """
        Initialize compute allocator.

        Args:
            default_depth: Default depth budget
            default_model: Default model tier
            default_timeout_ms: Default timeout in milliseconds
        """
        self.default_depth = default_depth
        self.default_model = default_model
        self.default_timeout_ms = default_timeout_ms

    def estimate_difficulty(
        self,
        query: str,
        context: dict[str, str],
    ) -> DifficultyEstimate:
        """
        Estimate difficulty of a query.

        Implements: SPEC-07.12

        Args:
            query: The user query
            context: Context files (filename -> content)

        Returns:
            DifficultyEstimate with score and factors
        """
        factors = []
        query_lower = query.lower()

        # Detect task type
        task_type = self._detect_task_type(query_lower)
        factors.append(f"task_type={task_type.value}")

        # Query complexity
        query_complexity = self._estimate_query_complexity(query_lower)
        factors.append(f"query_complexity={query_complexity:.2f}")

        # Context complexity
        context_complexity = self._estimate_context_complexity(context)
        factors.append(f"context_complexity={context_complexity:.2f}")

        # Task type difficulty modifier
        task_modifier = {
            TaskType.QUESTION: 0.0,
            TaskType.CODE: 0.2,
            TaskType.ANALYSIS: 0.3,
            TaskType.REFACTOR: 0.4,
            TaskType.DEBUG: 0.5,
        }.get(task_type, 0.2)
        factors.append(f"task_modifier={task_modifier:.2f}")

        # Combined score (0.0 to 1.0)
        score = min(
            1.0,
            (query_complexity * 0.3 + context_complexity * 0.3 + task_modifier * 0.4),
        )

        return DifficultyEstimate(
            score=score,
            task_type=task_type,
            context_complexity=context_complexity,
            query_complexity=query_complexity,
            factors=factors,
        )

    def _detect_task_type(self, query_lower: str) -> TaskType:
        """Detect task type from query."""
        scores: dict[TaskType, int] = dict.fromkeys(TaskType, 0)

        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[task_type] += 1

        # Return highest scoring type, default to QUESTION
        max_score = max(scores.values())
        if max_score == 0:
            return TaskType.QUESTION

        for task_type, score in scores.items():
            if score == max_score:
                return task_type

        return TaskType.QUESTION

    def _estimate_query_complexity(self, query_lower: str) -> float:
        """Estimate complexity from query text."""
        complexity = 0.0

        # Check complexity indicators
        for pattern, weight in self.COMPLEXITY_INDICATORS:
            if re.search(pattern, query_lower):
                complexity += weight

        # Multi-step indicators
        step_words = ["first", "then", "after", "finally", "next", "also"]
        step_count = sum(1 for word in step_words if word in query_lower)
        complexity += step_count * 0.1

        # Length factor
        word_count = len(query_lower.split())
        if word_count > 50:
            complexity += 0.2
        elif word_count > 20:
            complexity += 0.1

        return min(1.0, complexity)

    def _estimate_context_complexity(self, context: dict[str, str]) -> float:
        """Estimate complexity from context."""
        if not context:
            return 0.0

        num_files = len(context)
        total_lines = sum(content.count("\n") + 1 for content in context.values())

        # File count factor
        file_factor = min(1.0, num_files / 20)

        # Line count factor
        line_factor = min(1.0, total_lines / 5000)

        return (file_factor + line_factor) / 2

    def allocate(
        self,
        query: str,
        context: dict[str, str],
        total_budget: float | None = None,
        optimize_for: str = "balanced",
    ) -> ComputeAllocation:
        """
        Allocate compute resources for a query.

        Implements: SPEC-07.10-07.15

        Args:
            query: The user query
            context: Context files
            total_budget: Maximum cost budget in USD
            optimize_for: "quality", "cost", or "balanced"

        Returns:
            ComputeAllocation with resource allocation
        """
        # Estimate difficulty
        difficulty = self.estimate_difficulty(query, context)

        # Base allocation from difficulty
        base_depth = self._calculate_depth(difficulty.score)
        base_model = self._calculate_model_tier(difficulty.score, optimize_for)
        base_parallel = self._calculate_parallel_calls(
            difficulty.score, difficulty.context_complexity
        )
        base_timeout = self._calculate_timeout(difficulty.score)

        # Estimate cost
        estimated_cost = self._estimate_cost(base_depth, base_model, base_parallel)

        # Apply budget constraints if needed
        budget_constrained = False
        if total_budget is not None and estimated_cost > total_budget:
            budget_constrained = True
            base_depth, base_model, base_parallel, estimated_cost = self._apply_budget_constraint(
                base_depth,
                base_model,
                base_parallel,
                total_budget,
            )

        # Build reasoning
        tradeoff = self._explain_tradeoff(base_model, base_depth, optimize_for)
        reasoning = AllocationReasoning(
            difficulty_score=difficulty.score,
            difficulty_factors=difficulty.factors,
            tradeoff_explanation=tradeoff,
            budget_constraint_applied=budget_constrained,
            optimization_mode=optimize_for,
        )

        return ComputeAllocation(
            depth_budget=base_depth,
            model_tier=base_model,
            parallel_calls=base_parallel,
            timeout_ms=base_timeout,
            estimated_cost=estimated_cost,
            reasoning=reasoning,
        )

    def _calculate_depth(self, difficulty: float) -> int:
        """Calculate depth budget from difficulty."""
        if difficulty < 0.3:
            return 1
        elif difficulty < 0.6:
            return 2
        else:
            return 3

    def _calculate_model_tier(self, difficulty: float, optimize_for: str) -> ModelTier:
        """Calculate model tier from difficulty and optimization mode."""
        if optimize_for == "cost":
            # Prefer cheaper models
            if difficulty < 0.5:
                return ModelTier.HAIKU
            elif difficulty < 0.8:
                return ModelTier.SONNET
            else:
                return ModelTier.OPUS
        elif optimize_for == "quality":
            # Prefer higher-quality models
            if difficulty < 0.3:
                return ModelTier.SONNET
            else:
                return ModelTier.OPUS
        else:  # balanced
            if difficulty < 0.3:
                return ModelTier.HAIKU
            elif difficulty < 0.7:
                return ModelTier.SONNET
            else:
                return ModelTier.OPUS

    def _calculate_parallel_calls(self, difficulty: float, context_complexity: float) -> int:
        """Calculate parallel call limit."""
        base = 3
        if difficulty > 0.5:
            base += 2
        if context_complexity > 0.5:
            base += 2
        return min(10, base)

    def _calculate_timeout(self, difficulty: float) -> int:
        """Calculate timeout in milliseconds."""
        base = 30000
        if difficulty > 0.5:
            base += 30000
        if difficulty > 0.8:
            base += 60000
        return base

    def _estimate_cost(
        self,
        depth: int,
        model: ModelTier,
        parallel: int,
    ) -> float:
        """Estimate cost for allocation."""
        # Simplified cost model
        costs = MODEL_COSTS[model]
        # Assume ~2K input, ~1K output per call at each depth level
        calls_estimate = depth * max(1, parallel // 2)
        input_cost = calls_estimate * 2 * costs["input"]
        output_cost = calls_estimate * 1 * costs["output"]
        return input_cost + output_cost

    def _apply_budget_constraint(
        self,
        depth: int,
        model: ModelTier,
        parallel: int,
        budget: float,
    ) -> tuple[int, ModelTier, int, float]:
        """Apply budget constraint, reducing resources as needed."""
        # Try progressively cheaper configurations
        configurations = [
            (depth, model, parallel),
            (depth, ModelTier.SONNET if model == ModelTier.OPUS else model, parallel),
            (depth, ModelTier.HAIKU, parallel),
            (max(1, depth - 1), ModelTier.HAIKU, parallel),
            (1, ModelTier.HAIKU, max(1, parallel // 2)),
        ]

        for d, m, p in configurations:
            cost = self._estimate_cost(d, m, p)
            if cost <= budget:
                return d, m, p, cost

        # Minimum allocation
        return 1, ModelTier.HAIKU, 1, self._estimate_cost(1, ModelTier.HAIKU, 1)

    def _explain_tradeoff(self, model: ModelTier, depth: int, optimize_for: str) -> str:
        """Explain the model/depth tradeoff decision."""
        if model == ModelTier.OPUS and depth <= 1:
            return "High-capability model with shallow depth for quality on complex single-step"
        elif model == ModelTier.HAIKU and depth >= 2:
            return "Cost-efficient model with deeper exploration for thorough analysis"
        elif optimize_for == "quality":
            return f"Quality-optimized: {model.value} at depth {depth}"
        elif optimize_for == "cost":
            return f"Cost-optimized: {model.value} at depth {depth}"
        else:
            return f"Balanced allocation: {model.value} at depth {depth}"


__all__ = [
    "AllocationReasoning",
    "ComputeAllocation",
    "ComputeAllocator",
    "DifficultyEstimate",
    "ModelTier",
    "TaskType",
]
