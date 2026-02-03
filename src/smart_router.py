"""
Smart routing for query-based model selection across providers.

Implements: Spec §8.1 Phase 4 - Smart Routing

Routes queries to the optimal model based on:
- Query type (code, analysis, planning, etc.)
- Task complexity
- Provider strengths (Codex for code, Opus for reasoning)
- Cost/speed tradeoffs

When rlm_core is available, provides optional delegation to the
rlm_core.SmartRouter for lightweight routing decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import rlm_core

from .api_client import Provider, resolve_model


class QueryType(Enum):
    """Types of queries for routing decisions."""

    FACTUAL = "factual"  # Simple fact lookup
    ANALYTICAL = "analytical"  # Requires reasoning
    CREATIVE = "creative"  # Open-ended generation
    CODE = "code"  # Code generation/analysis
    SEARCH = "search"  # Information retrieval
    SUMMARIZATION = "summarization"  # Condensing information
    PLANNING = "planning"  # Multi-step task planning
    DEBUGGING = "debugging"  # Error analysis
    REFACTORING = "refactoring"  # Code restructuring
    ARCHITECTURE = "architecture"  # System design
    UNKNOWN = "unknown"


class ModelTier(Enum):
    """Model tiers by capability and cost."""

    FAST = "fast"  # Quick, cheap responses
    BALANCED = "balanced"  # Good balance
    POWERFUL = "powerful"  # Highest capability
    CODE_SPECIALIST = "code_specialist"  # Optimized for code
    INHERIT = "inherit"  # Use parent session model (SPEC-14.02)


@dataclass
class ModelOption:
    """A model option with metadata."""

    name: str  # Shorthand name (e.g., "opus", "codex")
    provider: Provider
    tier: ModelTier
    strengths: list[QueryType]  # Query types this model excels at
    cost_factor: float  # Relative cost (1.0 = baseline)
    speed_factor: float  # Relative speed (1.0 = baseline, higher = faster)


# Model catalog with strengths and characteristics
MODEL_CATALOG: dict[str, ModelOption] = {
    # Anthropic models
    "opus": ModelOption(
        name="opus",
        provider=Provider.ANTHROPIC,
        tier=ModelTier.POWERFUL,
        strengths=[
            QueryType.PLANNING,
            QueryType.ANALYTICAL,
            QueryType.ARCHITECTURE,
            QueryType.CREATIVE,
        ],
        cost_factor=3.0,
        speed_factor=0.5,
    ),
    "sonnet": ModelOption(
        name="sonnet",
        provider=Provider.ANTHROPIC,
        tier=ModelTier.BALANCED,
        strengths=[
            QueryType.CODE,
            QueryType.ANALYTICAL,
            QueryType.DEBUGGING,
        ],
        cost_factor=1.0,
        speed_factor=1.0,
    ),
    "haiku": ModelOption(
        name="haiku",
        provider=Provider.ANTHROPIC,
        tier=ModelTier.FAST,
        strengths=[
            QueryType.FACTUAL,
            QueryType.SEARCH,
            QueryType.SUMMARIZATION,
        ],
        cost_factor=0.1,
        speed_factor=3.0,
    ),
    # OpenAI models
    "gpt-5.2-codex": ModelOption(
        name="gpt-5.2-codex",
        provider=Provider.OPENAI,
        tier=ModelTier.CODE_SPECIALIST,
        strengths=[
            QueryType.CODE,
            QueryType.DEBUGGING,
            QueryType.REFACTORING,
            QueryType.ARCHITECTURE,
        ],
        cost_factor=1.5,
        speed_factor=1.2,
    ),
    "gpt-5.2": ModelOption(
        name="gpt-5.2",
        provider=Provider.OPENAI,
        tier=ModelTier.POWERFUL,
        strengths=[
            QueryType.PLANNING,
            QueryType.ANALYTICAL,
            QueryType.CREATIVE,
        ],
        cost_factor=2.5,
        speed_factor=0.6,
    ),
    "gpt-4o": ModelOption(
        name="gpt-4o",
        provider=Provider.OPENAI,
        tier=ModelTier.BALANCED,
        strengths=[
            QueryType.CODE,
            QueryType.ANALYTICAL,
            QueryType.FACTUAL,
        ],
        cost_factor=0.8,
        speed_factor=1.5,
    ),
    "gpt-4o-mini": ModelOption(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        tier=ModelTier.FAST,
        strengths=[
            QueryType.FACTUAL,
            QueryType.SEARCH,
            QueryType.SUMMARIZATION,
        ],
        cost_factor=0.05,
        speed_factor=4.0,
    ),
    "o1": ModelOption(
        name="o1",
        provider=Provider.OPENAI,
        tier=ModelTier.POWERFUL,
        strengths=[
            QueryType.ANALYTICAL,
            QueryType.PLANNING,
            QueryType.ARCHITECTURE,
        ],
        cost_factor=4.0,
        speed_factor=0.3,
    ),
    "o3-mini": ModelOption(
        name="o3-mini",
        provider=Provider.OPENAI,
        tier=ModelTier.BALANCED,
        strengths=[
            QueryType.ANALYTICAL,
            QueryType.CODE,
            QueryType.DEBUGGING,
        ],
        cost_factor=1.2,
        speed_factor=1.0,
    ),
}

# Best model for each query type (considering strengths)
OPTIMAL_MODELS: dict[QueryType, list[str]] = {
    QueryType.CODE: ["gpt-5.2-codex", "sonnet", "gpt-4o"],
    QueryType.DEBUGGING: ["gpt-5.2-codex", "sonnet", "o3-mini"],
    QueryType.REFACTORING: ["gpt-5.2-codex", "sonnet", "gpt-4o"],
    QueryType.PLANNING: ["opus", "gpt-5.2", "o1"],
    QueryType.ANALYTICAL: ["opus", "o1", "gpt-5.2"],
    QueryType.ARCHITECTURE: ["opus", "gpt-5.2-codex", "o1"],
    QueryType.CREATIVE: ["opus", "gpt-5.2", "sonnet"],
    QueryType.FACTUAL: ["haiku", "gpt-4o-mini", "gpt-4o"],
    QueryType.SEARCH: ["haiku", "gpt-4o-mini", "sonnet"],
    QueryType.SUMMARIZATION: ["haiku", "gpt-4o-mini", "sonnet"],
    QueryType.UNKNOWN: ["sonnet", "gpt-4o", "opus"],
}


@dataclass
class QueryClassification:
    """Result of query type classification."""

    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    signals: list[str]  # What triggered this classification
    complexity: float  # 0.0 (simple) to 1.0 (complex)

    @property
    def suggested_models(self) -> list[str]:
        """Get suggested models for this query type."""
        return OPTIMAL_MODELS.get(self.query_type, OPTIMAL_MODELS[QueryType.UNKNOWN])


@dataclass
class RoutingDecision:
    """Final routing decision with reasoning."""

    primary_model: str
    fallback_chain: list[str]
    query_type: QueryType
    confidence: float
    reason: str
    provider: Provider
    estimated_cost: float  # Relative cost estimate
    estimated_speed: float  # Relative speed estimate

    @property
    def all_models(self) -> list[str]:
        """All models in order of preference."""
        return [self.primary_model, *self.fallback_chain]


class QueryClassifier:
    """
    Classify queries by type for routing decisions.

    Implements: Spec §8.1 Query type detection
    """

    # Pattern sets for each query type
    PATTERNS: dict[QueryType, list[str]] = {
        QueryType.FACTUAL: [
            r"\bwhat is\b",
            r"\bwho is\b",
            r"\bwhen did\b",
            r"\bwhere is\b",
            r"\bdefine\b",
            r"\bhow many\b",
        ],
        QueryType.ANALYTICAL: [
            r"\bwhy\b",
            r"\banalyze\b",
            r"\bcompare\b",
            r"\bevaluate\b",
            r"\bassess\b",
            r"\bcritique\b",
            r"\bimplications\b",
            r"\bconsequences\b",
            r"\btrade.?offs?\b",
        ],
        QueryType.CREATIVE: [
            r"\bwrite\b.*\b(story|poem|essay)\b",
            r"\bcreate\b(?!.*\b(file|class|function)\b)",
            r"\bimagine\b",
            r"\binvent\b",
            r"\bbrainstorm\b",
            r"\bgenerate ideas\b",
        ],
        QueryType.CODE: [
            r"\bcode\b",
            r"\bfunction\b",
            r"\bimplement\b",
            r"\bprogram\b",
            r"\bscript\b",
            r"\bclass\b",
            r"\bmethod\b",
            r"\bapi\b",
            r"\bmodule\b",
            r"\bwrite.*\b(code|function|class)\b",
            r"\bcreate.*\b(file|class|function|module)\b",
        ],
        QueryType.SEARCH: [
            r"\bfind\b",
            r"\bsearch\b",
            r"\blook for\b",
            r"\blocate\b",
            r"\bwhere.*\bfile\b",
            r"\bgrep\b",
        ],
        QueryType.SUMMARIZATION: [
            r"\bsummarize\b",
            r"\bsummary\b",
            r"\btl;?dr\b",
            r"\bcondense\b",
            r"\boverview\b",
            r"\bbrief\b",
        ],
        QueryType.PLANNING: [
            r"\bplan\b",
            r"\bstrategy\b",
            r"\bsteps\b",
            r"\bhow (should|would|can) (i|we)\b",
            r"\broadmap\b",
            r"\bapproach\b",
        ],
        QueryType.DEBUGGING: [
            r"\bdebug\b",
            r"\bfix\b.*\b(bug|error|issue)\b",
            r"\bwhy.*\b(fail|error|crash)\b",
            r"\btroubleshoot\b",
            r"\bdiagnose\b",
            r"\bnot working\b",
            r"\bstack.?trace\b",
            r"\bexception\b",
        ],
        QueryType.REFACTORING: [
            r"\brefactor\b",
            r"\brestructure\b",
            r"\bclean.?up\b",
            r"\bimprove.*code\b",
            r"\boptimize\b",
            r"\bsimplify\b",
            r"\bmodernize\b",
        ],
        QueryType.ARCHITECTURE: [
            r"\barchitect\b",
            r"\bdesign\b.*\b(system|api|service)\b",
            r"\bpattern\b",
            r"\bstructure\b",
            r"\bscalability\b",
            r"\bmicroservice\b",
            r"\bdata.?flow\b",
        ],
    }

    # Complexity indicators
    COMPLEXITY_SIGNALS: dict[str, float] = {
        r"\bentire\b": 0.3,
        r"\ball\b.*\bfiles?\b": 0.3,
        r"\bcodebase\b": 0.4,
        r"\bcomprehensive\b": 0.3,
        r"\bdetailed\b": 0.2,
        r"\bthorough\b": 0.3,
        r"\bcomplex\b": 0.3,
        r"\bmultiple\b": 0.2,
        r"\bacross\b": 0.2,
        r"\bintegrat": 0.3,
        r"\brefactor.*entire\b": 0.5,
        r"\banalyze.*architecture\b": 0.4,
    }

    def __init__(self, custom_patterns: dict[QueryType, list[str]] | None = None):
        """Initialize classifier with optional custom patterns."""
        self.patterns = dict(self.PATTERNS)
        if custom_patterns:
            for query_type, patterns in custom_patterns.items():
                if query_type in self.patterns:
                    self.patterns[query_type].extend(patterns)
                else:
                    self.patterns[query_type] = patterns

        # Compile patterns
        self._compiled: dict[QueryType, list[re.Pattern[str]]] = {
            qt: [re.compile(p, re.IGNORECASE) for p in patterns]
            for qt, patterns in self.patterns.items()
        }
        self._complexity_compiled = {
            re.compile(p, re.IGNORECASE): score for p, score in self.COMPLEXITY_SIGNALS.items()
        }

    def classify(self, query: str) -> QueryClassification:
        """Classify a query by type and complexity."""
        scores: dict[QueryType, list[str]] = {qt: [] for qt in QueryType}

        # Match patterns
        for query_type, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[query_type].append(pattern.pattern)

        # Find best match
        best_type = QueryType.UNKNOWN
        best_signals: list[str] = []
        max_matches = 0

        for query_type, signals in scores.items():
            if len(signals) > max_matches:
                max_matches = len(signals)
                best_type = query_type
                best_signals = signals

        # Calculate confidence
        if max_matches == 0:
            confidence = 0.3
        elif max_matches == 1:
            confidence = 0.6
        elif max_matches == 2:
            confidence = 0.8
        else:
            confidence = 0.95

        # Calculate complexity
        complexity = 0.0
        for pattern, score in self._complexity_compiled.items():
            if pattern.search(query):
                complexity = min(1.0, complexity + score)

        # Boost complexity for long queries
        if len(query) > 200:
            complexity = min(1.0, complexity + 0.2)
        if len(query) > 500:
            complexity = min(1.0, complexity + 0.2)

        return QueryClassification(
            query_type=best_type,
            confidence=confidence,
            signals=best_signals,
            complexity=complexity,
        )


class SmartRouter:
    """
    Route queries to optimal models across providers.

    Implements: Spec §8.1 Phase 4 - Smart Routing

    Considers:
    - Query type and complexity
    - Model strengths per task type
    - Available providers
    - Cost/speed preferences
    """

    def __init__(
        self,
        classifier: QueryClassifier | None = None,
        available_providers: list[Provider] | None = None,
        prefer_speed: bool = False,
        prefer_cost: bool = False,
        force_provider: Provider | None = None,
    ):
        """
        Initialize router.

        Args:
            classifier: Query classifier to use
            available_providers: Which providers have API keys configured
            prefer_speed: Prefer faster models
            prefer_cost: Prefer cheaper models
            force_provider: Only use this provider
        """
        self.classifier = classifier or QueryClassifier()
        self.available_providers = available_providers or list(Provider)
        self.prefer_speed = prefer_speed
        self.prefer_cost = prefer_cost
        self.force_provider = force_provider
        self._routing_history: list[dict[str, Any]] = []

        # Initialize rlm_core router
        self._core_router = rlm_core.SmartRouter()

    @property
    def usesrlm_core(self) -> bool:
        """Return True if rlm_core SmartRouter is available."""
        return self._core_router is not None

    def route_core(
        self,
        query: str,
        depth: int = 0,
        max_depth: int = 5,
        remaining_budget: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Route using rlm_core SmartRouter for lightweight decisions.

        This provides a fast, pattern-based routing alternative to the
        full Python router. Returns None if rlm_core is not available.

        Args:
            query: The query to route
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            remaining_budget: Remaining cost budget

        Returns:
            Dict with model, tier, and reason, or None if unavailable
        """
        try:
            # Build RoutingContext
            ctx = rlm_core.RoutingContext()
            ctx = ctx.with_depth(depth)
            ctx = ctx.with_max_depth(max_depth)
            if remaining_budget is not None:
                ctx = ctx.with_budget(remaining_budget)

            # Get routing decision
            decision = self._core_router.route(query, ctx)

            # Convert tier to string
            tier_str = str(decision.tier)
            if "Fast" in tier_str:
                tier = "fast"
            elif "Balanced" in tier_str:
                tier = "balanced"
            elif "Powerful" in tier_str:
                tier = "powerful"
            else:
                tier = "balanced"

            return {
                "model": decision.model.id
                if hasattr(decision.model, "id")
                else str(decision.model),
                "tier": tier,
                "reason": decision.reason,
            }

        except Exception:
            return None

    def route(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        force_model: str | None = None,
    ) -> RoutingDecision:
        """
        Route a query to the optimal model.

        Args:
            query: The query to route
            context: Optional context (depth, remaining_tokens, etc.)
            force_model: Force a specific model

        Returns:
            RoutingDecision with primary model and fallbacks
        """
        # Handle forced model
        if force_model:
            provider, _ = resolve_model(force_model)
            return RoutingDecision(
                primary_model=force_model,
                fallback_chain=[],
                query_type=QueryType.UNKNOWN,
                confidence=1.0,
                reason=f"Forced model: {force_model}",
                provider=provider,
                estimated_cost=MODEL_CATALOG.get(force_model, MODEL_CATALOG["sonnet"]).cost_factor,
                estimated_speed=MODEL_CATALOG.get(
                    force_model, MODEL_CATALOG["sonnet"]
                ).speed_factor,
            )

        # Classify query
        classification = self.classifier.classify(query)

        # Get candidate models
        candidates = self._get_candidates(classification)

        # Filter by available providers
        candidates = [
            m for m in candidates if MODEL_CATALOG[m].provider in self.available_providers
        ]

        # Filter by forced provider
        if self.force_provider:
            candidates = [m for m in candidates if MODEL_CATALOG[m].provider == self.force_provider]

        # Apply context adjustments
        if context:
            candidates = self._adjust_for_context(candidates, context, classification)

        # Apply preferences
        if self.prefer_speed:
            candidates.sort(key=lambda m: -MODEL_CATALOG[m].speed_factor)
        elif self.prefer_cost:
            candidates.sort(key=lambda m: MODEL_CATALOG[m].cost_factor)

        # Ensure we have at least one candidate
        if not candidates:
            candidates = (
                ["sonnet"] if Provider.ANTHROPIC in self.available_providers else ["gpt-4o"]
            )

        primary = candidates[0]
        primary_info = MODEL_CATALOG.get(primary, MODEL_CATALOG["sonnet"])

        # Build fallback chain (different providers for resilience)
        fallbacks = []
        for m in candidates[1:4]:  # Up to 3 fallbacks
            if MODEL_CATALOG[m].provider != primary_info.provider or len(fallbacks) < 2:
                fallbacks.append(m)

        # Build reason
        reason_parts = [
            f"Query type: {classification.query_type.value}",
            f"Complexity: {classification.complexity:.1f}",
        ]
        if classification.signals:
            reason_parts.append(f"Signals: {', '.join(classification.signals[:3])}")
        if self.prefer_speed:
            reason_parts.append("(speed preferred)")
        if self.prefer_cost:
            reason_parts.append("(cost preferred)")

        decision = RoutingDecision(
            primary_model=primary,
            fallback_chain=fallbacks,
            query_type=classification.query_type,
            confidence=classification.confidence,
            reason=" | ".join(reason_parts),
            provider=primary_info.provider,
            estimated_cost=primary_info.cost_factor,
            estimated_speed=primary_info.speed_factor,
        )

        # Track
        self._routing_history.append(
            {
                "query": query[:100],
                "type": classification.query_type.value,
                "model": primary,
                "provider": primary_info.provider.value,
            }
        )

        return decision

    def _get_candidates(self, classification: QueryClassification) -> list[str]:
        """Get candidate models for a classification."""
        # Start with optimal models for this query type
        candidates = list(classification.suggested_models)

        # Add more based on complexity
        if classification.complexity > 0.7:
            # High complexity - add powerful models
            for model in ["opus", "gpt-5.2", "o1"]:
                if model not in candidates:
                    candidates.insert(0, model)
        elif classification.complexity < 0.3:
            # Low complexity - add fast models
            for model in ["haiku", "gpt-4o-mini"]:
                if model not in candidates:
                    candidates.append(model)

        return candidates

    def _adjust_for_context(
        self,
        candidates: list[str],
        context: dict[str, Any],
        classification: QueryClassification,
    ) -> list[str]:
        """Adjust candidates based on context."""
        result = list(candidates)

        # Depth-based adjustment
        depth = context.get("depth", 0)
        if depth > 0:
            # For recursive calls, prefer faster/cheaper models
            fast_models = [
                m for m in result if MODEL_CATALOG[m].tier in (ModelTier.FAST, ModelTier.BALANCED)
            ]
            if fast_models:
                result = fast_models + [m for m in result if m not in fast_models]

        # Token budget adjustment
        remaining_tokens = context.get("remaining_tokens")
        if remaining_tokens is not None and remaining_tokens < 10000:
            # Low budget - prioritize cheap models
            result.sort(key=lambda m: MODEL_CATALOG[m].cost_factor)

        # Force powerful for certain contexts
        if context.get("force_powerful") and classification.complexity > 0.5:
            powerful = [m for m in result if MODEL_CATALOG[m].tier == ModelTier.POWERFUL]
            if powerful:
                result = powerful + [m for m in result if m not in powerful]

        return result

    def record_outcome(self, query: str, model_used: str, success: bool, latency_ms: float) -> None:
        """Record routing outcome for learning."""
        for entry in reversed(self._routing_history[-100:]):
            if entry["query"] == query[:100]:
                entry["outcome"] = {
                    "success": success,
                    "latency_ms": latency_ms,
                    "model_used": model_used,
                }
                break

    def get_statistics(self) -> dict[str, Any]:
        """Get routing statistics."""
        if not self._routing_history:
            return {"total_routes": 0}

        by_type: dict[str, int] = {}
        by_model: dict[str, int] = {}
        by_provider: dict[str, int] = {}

        for entry in self._routing_history:
            qt = entry["type"]
            model = entry["model"]
            provider = entry["provider"]

            by_type[qt] = by_type.get(qt, 0) + 1
            by_model[model] = by_model.get(model, 0) + 1
            by_provider[provider] = by_provider.get(provider, 0) + 1

        return {
            "total_routes": len(self._routing_history),
            "by_query_type": by_type,
            "by_model": by_model,
            "by_provider": by_provider,
        }


class FallbackExecutor:
    """Execute queries with automatic fallback on failure."""

    def __init__(self, router: SmartRouter):
        self.router = router
        self._execution_history: list[dict[str, Any]] = []

    async def execute_with_fallback(
        self,
        query: str,
        execute_fn: Any,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """
        Execute query with automatic fallback.

        Args:
            query: Query to execute
            execute_fn: Async function(query, model) -> result
            context: Optional routing context

        Returns:
            (result, model_used) tuple
        """
        import time

        decision = self.router.route(query, context)
        models = decision.all_models
        last_error: Exception | None = None

        for model in models:
            start = time.time()
            try:
                result = await execute_fn(query, model)
                latency = (time.time() - start) * 1000

                self.router.record_outcome(query, model, success=True, latency_ms=latency)
                self._execution_history.append(
                    {
                        "query": query[:100],
                        "model": model,
                        "success": True,
                        "latency_ms": latency,
                    }
                )

                return result, model

            except Exception as e:
                latency = (time.time() - start) * 1000
                last_error = e

                self.router.record_outcome(query, model, success=False, latency_ms=latency)
                self._execution_history.append(
                    {
                        "query": query[:100],
                        "model": model,
                        "success": False,
                        "error": str(e),
                        "latency_ms": latency,
                    }
                )
                continue

        raise last_error or RuntimeError("All models failed")


# Convenience function
def get_optimal_model(
    query: str,
    available_providers: list[Provider] | None = None,
    prefer_speed: bool = False,
    prefer_cost: bool = False,
) -> str:
    """
    Get the optimal model for a query.

    Args:
        query: The query to route
        available_providers: Which providers are available
        prefer_speed: Prefer faster models
        prefer_cost: Prefer cheaper models

    Returns:
        Model name (shorthand)
    """
    router = SmartRouter(
        available_providers=available_providers,
        prefer_speed=prefer_speed,
        prefer_cost=prefer_cost,
    )
    decision = router.route(query)
    return decision.primary_model


__all__ = [
    "FallbackExecutor",
    "MODEL_CATALOG",
    "ModelOption",
    "ModelTier",
    "OPTIMAL_MODELS",
    "QueryClassification",
    "QueryClassifier",
    "QueryType",
    "RoutingDecision",
    "SmartRouter",
    "get_optimal_model",
]
