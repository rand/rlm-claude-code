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

from .cost_tracker import compute_affordable_tokens
from .smart_router import ModelTier, QueryType


class ExecutionMode(Enum):
    """
    User execution preferences.

    Controls the cost/speed/quality tradeoffs.

    Implements: SPEC-14.01-14.02 for MICRO mode.
    """

    MICRO = "micro"  # Minimal cost, REPL-only, no LLM sub-queries (SPEC-14.01)
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


class ExecutionStrategy(Enum):
    """
    Execution strategy templates mapped to user JTBDs.

    Implements: SPEC-12.06 - Strategy Templates for OODA Decide Phase
    Implements: SPEC-14.01 - MICRO strategy for always-on mode

    Each strategy corresponds to a Job-To-Be-Done from the JTBD analysis:
    - MICRO: Minimal always-on mode (SPEC-14)
    - DIRECT_RESPONSE: Quick answers without RLM (JTBD-7)
    - DISCOVERY: Understand unfamiliar codebase (JTBD-2)
    - EXHAUSTIVE_SEARCH: Find all usages/instances (JTBD-3)
    - RECURSIVE_DEBUG: Multi-layer debugging (JTBD-1)
    - MAP_REDUCE: Security/completeness review (JTBD-5)
    - ARCHITECTURE: Design decisions with tradeoffs (JTBD-4)
    - CONTINUATION: Resume previous work (JTBD-6)
    """

    MICRO = "micro"  # SPEC-14: Always-on minimal mode
    DIRECT_RESPONSE = "direct"  # JTBD-7: Quick answers, bypass RLM
    DISCOVERY = "discovery"  # JTBD-2: Explore codebase structure
    EXHAUSTIVE_SEARCH = "exhaustive_search"  # JTBD-3: Find all instances
    RECURSIVE_DEBUG = "recursive_debug"  # JTBD-1: Multi-layer tracing
    MAP_REDUCE = "map_reduce"  # JTBD-5: Systematic partition analysis
    ARCHITECTURE = "architecture"  # JTBD-4: Design decision exploration
    CONTINUATION = "continuation"  # JTBD-6: Resume with memory context


@dataclass
class DecisionConfidence:
    """
    Per-dimension confidence scores for orchestration decisions.

    Implements: SPEC-12.07 - Confidence Intervals for OODA Decide Phase

    Each dimension indicates how certain the orchestrator is about that
    specific decision. Low confidence can trigger:
    - User confirmation prompts
    - Conservative fallbacks (cheaper model, lower depth)
    - Adaptive execution (start conservative, escalate if needed)
    - Telemetry logging for improving heuristics over time

    Attributes:
        activation: Confidence in activate_rlm decision (0.0-1.0)
        model_tier: Confidence in model tier selection (0.0-1.0)
        depth: Confidence in depth budget choice (0.0-1.0)
        strategy: Confidence in execution strategy selection (0.0-1.0)

    Example:
        >>> conf = DecisionConfidence(activation=0.9, model_tier=0.6)
        >>> conf.min_confidence()  # Most uncertain dimension
        0.6
        >>> conf.average()  # Overall confidence
        0.725
    """

    activation: float = 0.7  # How sure about activate_rlm?
    model_tier: float = 0.7  # How sure about model choice?
    depth: float = 0.7  # How sure about depth budget?
    strategy: float = 0.7  # How sure about execution strategy?

    def __post_init__(self) -> None:
        """Validate confidence values are in [0, 1] range."""
        for field_name in ("activation", "model_tier", "depth", "strategy"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"DecisionConfidence.{field_name} must be in [0, 1], got {value}")

    def average(self) -> float:
        """Average confidence across all dimensions."""
        return (self.activation + self.model_tier + self.depth + self.strategy) / 4

    def min_confidence(self) -> float:
        """Minimum confidence (most uncertain dimension)."""
        return min(self.activation, self.model_tier, self.depth, self.strategy)

    def max_confidence(self) -> float:
        """Maximum confidence (most certain dimension)."""
        return max(self.activation, self.model_tier, self.depth, self.strategy)

    def low_confidence_dimensions(self, threshold: float = 0.5) -> list[str]:
        """Return dimensions with confidence below threshold."""
        dims = []
        if self.activation < threshold:
            dims.append("activation")
        if self.model_tier < threshold:
            dims.append("model_tier")
        if self.depth < threshold:
            dims.append("depth")
        if self.strategy < threshold:
            dims.append("strategy")
        return dims

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "activation": self.activation,
            "model_tier": self.model_tier,
            "depth": self.depth,
            "strategy": self.strategy,
        }

    @classmethod
    def high(cls) -> DecisionConfidence:
        """Create high-confidence instance (clear signals)."""
        return cls(activation=0.9, model_tier=0.9, depth=0.9, strategy=0.9)

    @classmethod
    def medium(cls) -> DecisionConfidence:
        """Create medium-confidence instance (default heuristics)."""
        return cls(activation=0.7, model_tier=0.7, depth=0.7, strategy=0.7)

    @classmethod
    def low(cls) -> DecisionConfidence:
        """Create low-confidence instance (ambiguous signals)."""
        return cls(activation=0.4, model_tier=0.4, depth=0.4, strategy=0.4)


# Default configurations per execution mode
MODE_DEFAULTS: dict[ExecutionMode, dict[str, Any]] = {
    # SPEC-14.02: Micro mode - minimal cost, REPL-only, no LLM sub-queries
    ExecutionMode.MICRO: {
        "depth_budget": 1,
        "tokens_per_depth": 5_000,
        "model_tier": ModelTier.INHERIT,  # Use parent session model
        "tool_access": ToolAccessLevel.REPL_ONLY,
        "max_cost_dollars": 0.02,  # ~$0.01 at current rates
        "max_cost_tokens": 2_000,  # Token-based tracking (SPEC-14.61)
    },
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

# Strategy defaults based on JTBD analysis
# Each strategy has recommended settings and REPL function hints
STRATEGY_DEFAULTS: dict[ExecutionStrategy, dict[str, Any]] = {
    # SPEC-14.01-14.04: Micro mode - always-on with restricted REPL
    ExecutionStrategy.MICRO: {
        "activate_rlm": True,
        "depth_budget": 1,
        "model_tier": ModelTier.INHERIT,
        "tool_access": ToolAccessLevel.REPL_ONLY,
        "execution_mode": ExecutionMode.MICRO,
        "hints": [
            "Use peek() to view context slices",
            "Use search() for pattern matching (no LLM)",
            "Use memory_query() to retrieve facts",
            "Escalate to balanced mode if complexity detected",
        ],
    },
    ExecutionStrategy.DIRECT_RESPONSE: {
        "activate_rlm": False,
        "depth_budget": 0,
        "model_tier": ModelTier.FAST,
        "tool_access": ToolAccessLevel.NONE,
        "execution_mode": ExecutionMode.FAST,
        "hints": [],  # No REPL needed
    },
    ExecutionStrategy.DISCOVERY: {
        "activate_rlm": True,
        "depth_budget": 2,
        "model_tier": ModelTier.BALANCED,
        "tool_access": ToolAccessLevel.READ_ONLY,
        "execution_mode": ExecutionMode.BALANCED,
        "hints": [
            "Use peek() to scan file structure first",
            "Use find_relevant() to identify key files",
            "Use extract_functions() for code structure",
            "Use llm() sub-queries for component explanations",
        ],
    },
    ExecutionStrategy.EXHAUSTIVE_SEARCH: {
        "activate_rlm": True,
        "depth_budget": 2,
        "model_tier": ModelTier.FAST,
        "tool_access": ToolAccessLevel.READ_ONLY,
        "execution_mode": ExecutionMode.THOROUGH,
        "hints": [
            "Use search() with regex for pattern matching",
            "Use llm_batch() for parallel file analysis",
            "Use memory_add_fact() to track findings",
            "Ensure exhaustive coverage before synthesizing",
        ],
    },
    ExecutionStrategy.RECURSIVE_DEBUG: {
        "activate_rlm": True,
        "depth_budget": 3,
        "model_tier": ModelTier.POWERFUL,
        "tool_access": ToolAccessLevel.READ_ONLY,
        "execution_mode": ExecutionMode.THOROUGH,
        "hints": [
            "Use search() to trace error patterns",
            "Use llm() sub-queries for each module layer",
            "Use map_reduce() to synthesize findings",
            "Trace data flow between components",
        ],
    },
    ExecutionStrategy.MAP_REDUCE: {
        "activate_rlm": True,
        "depth_budget": 3,
        "model_tier": ModelTier.POWERFUL,
        "tool_access": ToolAccessLevel.READ_ONLY,
        "execution_mode": ExecutionMode.THOROUGH,
        "hints": [
            "Use map_reduce() with focused analysis prompts",
            "Partition codebase into logical sections",
            "Use memory_add_experience() for findings",
            "Prioritize results by severity/impact",
        ],
    },
    ExecutionStrategy.ARCHITECTURE: {
        "activate_rlm": True,
        "depth_budget": 3,
        "model_tier": ModelTier.POWERFUL,
        "tool_access": ToolAccessLevel.READ_ONLY,
        "execution_mode": ExecutionMode.THOROUGH,
        "hints": [
            "Use llm() sub-queries to analyze each option",
            "Build reasoning traces for decision tree",
            "Enumerate tradeoffs explicitly",
            "Consider codebase context for recommendations",
        ],
    },
    ExecutionStrategy.CONTINUATION: {
        "activate_rlm": True,
        "depth_budget": 1,  # Lower depth since we have memory
        "model_tier": ModelTier.BALANCED,
        "tool_access": ToolAccessLevel.READ_ONLY,
        "execution_mode": ExecutionMode.BALANCED,
        "hints": [
            "Use memory_query() to recall prior context",
            "Skip re-analysis of known facts",
            "Focus on new work since last session",
            "Update memory with new discoveries",
        ],
    },
}


def compute_model_aware_tokens_per_depth(
    model: str,
    budget_dollars: float,
    depth_budget: int,
) -> int:
    """
    Compute tokens per depth level based on model cost and budget.

    This replaces the hardcoded tokens_per_depth values with model-aware
    computation that accounts for the actual cost differences between models.

    Args:
        model: Model name or alias
        budget_dollars: Total budget in dollars
        depth_budget: Number of depth levels

    Returns:
        Tokens per depth level that fits within budget
    """
    if depth_budget <= 0:
        return 0

    # Compute total affordable tokens for the budget
    total_affordable = compute_affordable_tokens(budget_dollars, model)

    # Divide by depth levels, with some buffer for overhead
    tokens_per_depth = int(total_affordable / depth_budget * 0.9)  # 10% buffer

    # Clamp to reasonable bounds
    return max(5_000, min(tokens_per_depth, 200_000))


@dataclass
class OrchestrationPlan:
    """
    Complete orchestration decision for a query.

    Implements: Spec §8.1 Phase 2 - Orchestration Layer

    This combines activation, routing, and execution preferences
    into a single plan that the orchestrator follows.

    Key Attributes:
        activate_rlm: Whether to use RLM (True) or direct response (False)
        activation_reason: Human-readable reason for the decision
        model_tier: FAST/BALANCED/POWERFUL model selection
        primary_model: Specific model ID (e.g., "claude-sonnet-4-20250514")
        depth_budget: Max recursion depth (0=no recursion, 1-3=sub-queries)
        execution_mode: FAST/BALANCED/THOROUGH processing intensity
        tool_access: NONE/REPL_ONLY/READ_ONLY/FULL tool permissions

    Example:
        >>> from src.orchestration_schema import (
        ...     OrchestrationPlan,
        ...     ModelTier,
        ...     ExecutionMode,
        ...     ToolAccessLevel,
        ... )
        >>> plan = OrchestrationPlan(
        ...     activate_rlm=True,
        ...     activation_reason="Complex multi-file debugging",
        ...     model_tier=ModelTier.POWERFUL,
        ...     primary_model="claude-opus-4-5-20251101",
        ...     depth_budget=3,
        ...     execution_mode=ExecutionMode.THOROUGH,
        ...     tool_access=ToolAccessLevel.FULL,
        ... )
        >>> plan.activate_rlm  # Check if RLM will be used
        True
        >>> plan.allows_recursion  # Check if sub-queries allowed
        True
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
    metadata: dict[str, Any] = field(default_factory=dict)  # For telemetry tracking

    # === Memory-Augmented Orientation (SPEC-12.04) ===
    memory_context: list[str] = field(default_factory=list)  # Relevant facts to inject
    prior_strategy: str | None = None  # Strategy from successful past experience

    # === Strategy Templates (SPEC-12.06) ===
    strategy: ExecutionStrategy = ExecutionStrategy.DISCOVERY  # Selected execution strategy
    strategy_hints: list[str] = field(default_factory=list)  # REPL function hints for strategy

    # === Decision Confidence (SPEC-12.07) ===
    decision_confidence: DecisionConfidence = field(default_factory=DecisionConfidence.medium)

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
            "metadata": self.metadata,
            "memory_context": self.memory_context,
            "prior_strategy": self.prior_strategy,
            "strategy": self.strategy.value,
            "strategy_hints": self.strategy_hints,
            "decision_confidence": self.decision_confidence.to_dict(),
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
            strategy=ExecutionStrategy.DIRECT_RESPONSE,
            strategy_hints=[],
            decision_confidence=DecisionConfidence.high(),  # Clear bypass decision
        )

    @classmethod
    def micro(
        cls,
        reason: str = "always_on_default",
        parent_model: str = "sonnet",
    ) -> OrchestrationPlan:
        """
        Create a micro mode plan (SPEC-14.01-14.02).

        Micro mode is the always-on default with minimal cost:
        - depth_budget=1
        - max 5K tokens per depth
        - REPL-only tool access (no LLM sub-queries)
        - Uses parent session model (INHERIT tier)

        Args:
            reason: Why micro mode was activated
            parent_model: Model to inherit from parent session

        Returns:
            OrchestrationPlan configured for micro mode
        """
        defaults = MODE_DEFAULTS[ExecutionMode.MICRO]
        return cls(
            activate_rlm=True,
            activation_reason=reason,
            model_tier=ModelTier.INHERIT,
            primary_model=parent_model,
            fallback_chain=[],
            depth_budget=defaults["depth_budget"],
            tokens_per_depth=defaults["tokens_per_depth"],
            execution_mode=ExecutionMode.MICRO,
            tool_access=defaults["tool_access"],
            max_cost_dollars=defaults["max_cost_dollars"],
            max_tokens=defaults.get("max_cost_tokens", 2000),
            strategy=ExecutionStrategy.MICRO,
            strategy_hints=list(STRATEGY_DEFAULTS[ExecutionStrategy.MICRO]["hints"]),
            decision_confidence=DecisionConfidence.high(),
        )

    @classmethod
    def from_strategy(
        cls,
        strategy: ExecutionStrategy,
        activation_reason: str = "strategy_selected",
        available_models: list[str] | None = None,
        use_model_aware_budget: bool = True,
    ) -> OrchestrationPlan:
        """
        Create plan from execution strategy with JTBD-aligned defaults.

        Implements: SPEC-12.06 - Strategy Templates

        Args:
            strategy: Execution strategy (maps to JTBD)
            activation_reason: Why this strategy was selected
            available_models: Models that are available (have API keys)
            use_model_aware_budget: If True, compute tokens_per_depth based on
                                    model cost and budget (recommended)

        Returns:
            OrchestrationPlan configured for the strategy

        Example:
            >>> plan = OrchestrationPlan.from_strategy(
            ...     ExecutionStrategy.RECURSIVE_DEBUG, activation_reason="multi_layer_error"
            ... )
            >>> plan.depth_budget
            3
            >>> plan.strategy_hints[0]
            'Use search() to trace error patterns'
        """
        defaults = STRATEGY_DEFAULTS[strategy]
        tier = defaults["model_tier"]
        mode = defaults["execution_mode"]

        # Handle direct response (no RLM)
        if not defaults["activate_rlm"]:
            return cls.bypass(reason=activation_reason)

        # Handle MICRO strategy specially (SPEC-14.01)
        if strategy == ExecutionStrategy.MICRO:
            # For micro mode, inherit from available models or default to sonnet
            parent_model = "sonnet"
            if available_models:
                parent_model = available_models[0]
            return cls.micro(reason=activation_reason, parent_model=parent_model)

        # Select primary model from available models
        # Handle INHERIT tier by falling back to BALANCED
        if tier == ModelTier.INHERIT:
            tier_models = TIER_MODELS[ModelTier.BALANCED]
        else:
            tier_models = TIER_MODELS.get(tier, TIER_MODELS[ModelTier.BALANCED])

        if available_models:
            candidates = [m for m in tier_models if m in available_models]
        else:
            candidates = tier_models

        primary = candidates[0] if candidates else "sonnet"
        fallbacks = candidates[1:3] if len(candidates) > 1 else []

        depth_budget = defaults["depth_budget"]
        max_cost = MODE_DEFAULTS[mode]["max_cost_dollars"]

        # Compute model-aware tokens per depth if enabled
        if use_model_aware_budget:
            tokens_per_depth = compute_model_aware_tokens_per_depth(
                model=primary,
                budget_dollars=max_cost,
                depth_budget=depth_budget,
            )
        else:
            tokens_per_depth = MODE_DEFAULTS[mode]["tokens_per_depth"]

        return cls(
            activate_rlm=True,
            activation_reason=activation_reason,
            model_tier=tier,
            primary_model=primary,
            fallback_chain=fallbacks,
            depth_budget=depth_budget,
            tokens_per_depth=tokens_per_depth,
            execution_mode=mode,
            tool_access=defaults["tool_access"],
            max_cost_dollars=max_cost,
            strategy=strategy,
            strategy_hints=list(defaults["hints"]),  # Copy to avoid mutation
            decision_confidence=DecisionConfidence.high(),  # Explicit strategy selection
        )

    @classmethod
    def from_mode(
        cls,
        mode: ExecutionMode,
        query_type: QueryType = QueryType.UNKNOWN,
        activation_reason: str = "mode_selected",
        available_models: list[str] | None = None,
        use_model_aware_budget: bool = True,
    ) -> OrchestrationPlan:
        """
        Create plan from execution mode with sensible defaults.

        Args:
            mode: Execution mode (fast/balanced/thorough)
            query_type: Type of query being processed
            activation_reason: Why RLM was activated
            available_models: Models that are available (have API keys)
            use_model_aware_budget: If True, compute tokens_per_depth based on
                                    model cost and budget (recommended)

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

        depth_budget = defaults["depth_budget"]
        max_cost = defaults["max_cost_dollars"]

        # Compute model-aware tokens per depth if enabled
        if use_model_aware_budget:
            tokens_per_depth = compute_model_aware_tokens_per_depth(
                model=primary,
                budget_dollars=max_cost,
                depth_budget=depth_budget,
            )
        else:
            # Fall back to static defaults
            tokens_per_depth = defaults["tokens_per_depth"]

        return cls(
            activate_rlm=True,
            activation_reason=activation_reason,
            model_tier=tier,
            primary_model=primary,
            fallback_chain=fallbacks,
            depth_budget=depth_budget,
            tokens_per_depth=tokens_per_depth,
            execution_mode=mode,
            tool_access=defaults["tool_access"],
            max_cost_dollars=max_cost,
            query_type=query_type,
            decision_confidence=DecisionConfidence.high(),  # Explicit mode selection
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

    # Memory-augmented context (pre-populated if memory store available)
    memory_facts: list[str] = field(default_factory=list)  # Relevant facts from memory
    memory_experiences: list[dict[str, Any]] = field(default_factory=list)  # Past experiences

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
    "DecisionConfidence",
    "ExecutionMode",
    "ExecutionStrategy",
    "MODE_DEFAULTS",
    "OrchestrationContext",
    "OrchestrationPlan",
    "PlanAdjustment",
    "STRATEGY_DEFAULTS",
    "TIER_MODELS",
    "ToolAccessLevel",
    "compute_model_aware_tokens_per_depth",
]
