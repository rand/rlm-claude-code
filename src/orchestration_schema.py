"""
Orchestration decision schema for RLM-Claude-Code.

Implements: Spec §8.1 Phase 2 - Orchestration Layer

Defines the OrchestrationPlan dataclass that unifies:
- RLM activation decisions
- Model selection and routing
- Depth/cost budgets
- Execution mode preferences
- Tool access levels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .smart_router import ModelTier, QueryType


class ExecutionMode(Enum):
    """
    User execution preferences.

    Controls the cost/speed/quality tradeoffs.
    """

    FAST = "fast"  # Minimize latency, use cheaper models, shallow depth
    BALANCED = "balanced"  # Default balance of speed and quality
    THOROUGH = "thorough"  # Maximize accuracy, deeper recursion allowed


class ToolAccessLevel(Enum):
    """
    Tool access levels for sub-LLMs.

    Controls what tools recursive calls can invoke.
    """

    NONE = "none"  # Pure reasoning only, no tool access
    REPL_ONLY = "repl_only"  # Only Python REPL execution
    READ_ONLY = "read_only"  # REPL + read file/search tools
    FULL = "full"  # Full Claude Code tool access


# Default configurations per execution mode
MODE_DEFAULTS: dict[ExecutionMode, dict[str, Any]] = {
    ExecutionMode.FAST: {
        "depth_budget": 1,
        "tokens_per_depth": 10_000,
        "model_tier": ModelTier.FAST,
        "tool_access": ToolAccessLevel.REPL_ONLY,
        "max_cost_dollars": 0.50,
    },
    ExecutionMode.BALANCED: {
        "depth_budget": 2,
        "tokens_per_depth": 25_000,
        "model_tier": ModelTier.BALANCED,
        "tool_access": ToolAccessLevel.READ_ONLY,
        "max_cost_dollars": 2.00,
    },
    ExecutionMode.THOROUGH: {
        "depth_budget": 3,
        "tokens_per_depth": 50_000,
        "model_tier": ModelTier.POWERFUL,
        "tool_access": ToolAccessLevel.FULL,
        "max_cost_dollars": 10.00,
    },
}

# Model selection per tier
TIER_MODELS: dict[ModelTier, list[str]] = {
    ModelTier.FAST: ["haiku", "gpt-4o-mini"],
    ModelTier.BALANCED: ["sonnet", "gpt-4o", "o3-mini"],
    ModelTier.POWERFUL: ["opus", "gpt-5.2", "o1"],
    ModelTier.CODE_SPECIALIST: ["gpt-5.2-codex", "sonnet"],
}


@dataclass
class OrchestrationPlan:
    """
    Complete orchestration decision for a query.

    Implements: Spec §8.1 Phase 2 - Orchestration Layer

    This combines activation, routing, and execution preferences
    into a single plan that the orchestrator follows.
    """

    # === RLM Activation ===
    activate_rlm: bool
    activation_reason: str

    # === Model Selection ===
    model_tier: ModelTier
    primary_model: str
    fallback_chain: list[str] = field(default_factory=list)

    # === Depth Management ===
    depth_budget: int = 2  # Maximum recursion depth (0-3)
    tokens_per_depth: int = 25_000  # Token budget per depth level

    # === Execution Mode ===
    execution_mode: ExecutionMode = ExecutionMode.BALANCED

    # === Tool Access ===
    tool_access: ToolAccessLevel = ToolAccessLevel.READ_ONLY

    # === Cost Constraints ===
    max_cost_dollars: float | None = None
    max_tokens: int | None = None

    # === Query Metadata ===
    query_type: QueryType = QueryType.UNKNOWN
    complexity_score: float = 0.5
    confidence: float = 0.7

    # === Additional Metadata ===
    signals: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0

    @property
    def total_token_budget(self) -> int:
        """Total tokens allowed across all depths."""
        return self.depth_budget * self.tokens_per_depth

    @property
    def allows_recursion(self) -> bool:
        """Whether recursive calls are allowed."""
        return self.depth_budget > 0

    @property
    def allows_tools(self) -> bool:
        """Whether any tool access is allowed."""
        return self.tool_access != ToolAccessLevel.NONE

    @property
    def allows_file_read(self) -> bool:
        """Whether file read tools are allowed."""
        return self.tool_access in (ToolAccessLevel.READ_ONLY, ToolAccessLevel.FULL)

    @property
    def allows_file_write(self) -> bool:
        """Whether file write tools are allowed."""
        return self.tool_access == ToolAccessLevel.FULL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "activate_rlm": self.activate_rlm,
            "activation_reason": self.activation_reason,
            "model_tier": self.model_tier.value,
            "primary_model": self.primary_model,
            "fallback_chain": self.fallback_chain,
            "depth_budget": self.depth_budget,
            "tokens_per_depth": self.tokens_per_depth,
            "execution_mode": self.execution_mode.value,
            "tool_access": self.tool_access.value,
            "max_cost_dollars": self.max_cost_dollars,
            "max_tokens": self.max_tokens,
            "query_type": self.query_type.value,
            "complexity_score": self.complexity_score,
            "confidence": self.confidence,
            "signals": self.signals,
        }

    @classmethod
    def bypass(cls, reason: str = "simple_task") -> OrchestrationPlan:
        """Create a bypass plan (no RLM activation)."""
        return cls(
            activate_rlm=False,
            activation_reason=reason,
            model_tier=ModelTier.BALANCED,
            primary_model="sonnet",
            depth_budget=0,
            tokens_per_depth=0,
            execution_mode=ExecutionMode.FAST,
            tool_access=ToolAccessLevel.NONE,
        )

    @classmethod
    def from_mode(
        cls,
        mode: ExecutionMode,
        query_type: QueryType = QueryType.UNKNOWN,
        activation_reason: str = "mode_selected",
        available_models: list[str] | None = None,
    ) -> OrchestrationPlan:
        """
        Create plan from execution mode with sensible defaults.

        Args:
            mode: Execution mode (fast/balanced/thorough)
            query_type: Type of query being processed
            activation_reason: Why RLM was activated
            available_models: Models that are available (have API keys)

        Returns:
            OrchestrationPlan configured for the mode
        """
        defaults = MODE_DEFAULTS[mode]
        tier = defaults["model_tier"]

        # Select primary model from available models
        tier_models = TIER_MODELS.get(tier, TIER_MODELS[ModelTier.BALANCED])
        if available_models:
            candidates = [m for m in tier_models if m in available_models]
        else:
            candidates = tier_models

        primary = candidates[0] if candidates else "sonnet"
        fallbacks = candidates[1:3] if len(candidates) > 1 else []

        return cls(
            activate_rlm=True,
            activation_reason=activation_reason,
            model_tier=tier,
            primary_model=primary,
            fallback_chain=fallbacks,
            depth_budget=defaults["depth_budget"],
            tokens_per_depth=defaults["tokens_per_depth"],
            execution_mode=mode,
            tool_access=defaults["tool_access"],
            max_cost_dollars=defaults["max_cost_dollars"],
            query_type=query_type,
        )


@dataclass
class OrchestrationContext:
    """
    Context for making orchestration decisions.

    Provides all information needed by the orchestrator
    to create an OrchestrationPlan.
    """

    # Query information
    query: str
    context_tokens: int = 0

    # Session state
    current_depth: int = 0
    tokens_used: int = 0
    cost_used: float = 0.0

    # User preferences
    forced_mode: ExecutionMode | None = None
    forced_model: str | None = None
    forced_rlm: bool | None = None  # True=force on, False=force off, None=auto

    # Available resources
    available_models: list[str] = field(default_factory=list)
    budget_remaining_dollars: float = 5.0
    budget_remaining_tokens: int = 100_000

    # Complexity signals (from classifier)
    complexity_signals: dict[str, bool] = field(default_factory=dict)

    @property
    def remaining_depth(self) -> int:
        """Remaining recursion depth from max 3."""
        return max(0, 3 - self.current_depth)

    @property
    def can_recurse(self) -> bool:
        """Whether recursion is still possible."""
        return self.current_depth < 3 and self.budget_remaining_tokens > 5000


@dataclass
class PlanAdjustment:
    """
    Runtime adjustment to an OrchestrationPlan.

    Used when conditions change during execution.
    """

    reason: str
    field_name: str
    old_value: Any
    new_value: Any
    timestamp: float = 0.0


__all__ = [
    "ExecutionMode",
    "MODE_DEFAULTS",
    "OrchestrationContext",
    "OrchestrationPlan",
    "PlanAdjustment",
    "TIER_MODELS",
    "ToolAccessLevel",
]
