"""
Intelligent orchestrator using Claude for decision-making.

Implements: Spec §8.1 Phase 2 - Orchestration Layer

Uses Claude (Haiku for speed) to make orchestration decisions,
with fallback to heuristic-based complexity_classifier.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .api_client import ClaudeClient, Provider, init_client
from .complexity_classifier import extract_complexity_signals, should_activate_rlm
from .cost_tracker import CostComponent
from .orchestration_schema import (
    ExecutionMode,
    OrchestrationContext,
    OrchestrationPlan,
    ToolAccessLevel,
)
from .smart_router import ModelTier, QueryClassifier, QueryType
from .types import SessionContext


# System prompt for orchestration decisions
ORCHESTRATOR_SYSTEM_PROMPT = """You are an orchestration decision engine for RLM (Recursive Language Model) Claude Code.

Your job is to analyze queries and decide how to process them optimally. You must output a JSON object with your decision.

## Decision Factors

1. **RLM Activation**: Should RLM mode be used?
   - YES for: multi-file analysis, debugging complex issues, cross-module reasoning, temporal queries, pattern finding
   - NO for: simple file reads, single commands, factual questions, short responses

2. **Execution Mode**: How thorough should processing be?
   - "fast": Quick response, shallow analysis, cheaper models
   - "balanced": Standard processing, moderate depth
   - "thorough": Deep analysis, multiple passes, expensive models

3. **Model Tier**: Which capability level is needed?
   - "fast": Simple queries (haiku, gpt-4o-mini)
   - "balanced": Standard queries (sonnet, gpt-4o)
   - "powerful": Complex reasoning (opus, o1)
   - "code_specialist": Code-heavy tasks (codex, sonnet)

4. **Depth Budget**: How many recursive sub-calls are allowed? (0-3)
   - 0: No recursion (simple tasks)
   - 1: Light recursion (moderate tasks)
   - 2: Standard recursion (complex tasks)
   - 3: Deep recursion (very complex tasks)

5. **Tool Access**: What tools can sub-calls use?
   - "none": Pure reasoning only
   - "repl_only": Python REPL execution
   - "read_only": REPL + file read/search
   - "full": All Claude Code tools

## Output Format

Output ONLY a JSON object (no markdown, no explanation):

{
  "activate_rlm": true/false,
  "activation_reason": "brief reason",
  "execution_mode": "fast" | "balanced" | "thorough",
  "model_tier": "fast" | "balanced" | "powerful" | "code_specialist",
  "depth_budget": 0-3,
  "tool_access": "none" | "repl_only" | "read_only" | "full",
  "query_type": "code" | "debugging" | "analytical" | "planning" | "factual" | "search" | "summarization" | "refactoring" | "architecture" | "creative" | "unknown",
  "complexity_score": 0.0-1.0,
  "signals": ["signal1", "signal2"]
}"""


@dataclass
class OrchestratorConfig:
    """Configuration for the intelligent orchestrator."""

    # Model to use for orchestration (haiku for speed)
    orchestrator_model: str = "haiku"

    # Timeout for orchestration call (ms)
    timeout_ms: int = 5000

    # Maximum tokens for orchestration response
    max_tokens: int = 500

    # Whether to use heuristic fallback on failure
    use_fallback: bool = True

    # Whether to cache recent decisions
    cache_enabled: bool = True
    cache_size: int = 100


class IntelligentOrchestrator:
    """
    Uses Claude to make intelligent orchestration decisions.

    Implements: Spec §8.1 Phase 2 - Orchestration Layer

    Falls back to heuristic classifier on LLM failure.
    """

    def __init__(
        self,
        client: ClaudeClient | None = None,
        config: OrchestratorConfig | None = None,
        available_models: list[str] | None = None,
    ):
        """
        Initialize the intelligent orchestrator.

        Args:
            client: Claude API client (creates one if None)
            config: Orchestrator configuration
            available_models: List of available model names
        """
        self._client = client
        self.config = config or OrchestratorConfig()
        self.available_models = available_models or ["sonnet", "haiku", "opus"]
        self._query_classifier = QueryClassifier()
        self._decision_cache: dict[str, OrchestrationPlan] = {}
        self._stats = {
            "llm_decisions": 0,
            "fallback_decisions": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    def _ensure_client(self) -> ClaudeClient:
        """Ensure we have an API client."""
        if self._client is None:
            self._client = init_client()
        return self._client

    async def create_plan(
        self,
        query: str,
        context: OrchestrationContext | SessionContext,
    ) -> OrchestrationPlan:
        """
        Create an orchestration plan for a query.

        Uses Claude for intelligent decision-making,
        falls back to heuristics on failure.

        Args:
            query: The user query
            context: Orchestration context or session context

        Returns:
            OrchestrationPlan with the decision
        """
        # Convert SessionContext to OrchestrationContext if needed
        if isinstance(context, SessionContext):
            orch_context = OrchestrationContext(
                query=query,
                context_tokens=context.total_tokens,
                available_models=self.available_models,
            )
        else:
            orch_context = context

        # Check forced overrides
        if orch_context.forced_rlm is False:
            return OrchestrationPlan.bypass("user_forced_off")

        if orch_context.forced_mode is not None:
            return OrchestrationPlan.from_mode(
                orch_context.forced_mode,
                activation_reason="user_forced_mode",
                available_models=self.available_models,
            )

        # Check cache
        cache_key = self._compute_cache_key(query, orch_context)
        if self.config.cache_enabled and cache_key in self._decision_cache:
            self._stats["cache_hits"] += 1
            return self._decision_cache[cache_key]

        # Try LLM-based orchestration
        try:
            plan = await self._llm_orchestrate(query, orch_context)
            self._stats["llm_decisions"] += 1

            # Cache the decision
            if self.config.cache_enabled:
                self._update_cache(cache_key, plan)

            return plan

        except Exception as e:
            self._stats["errors"] += 1

            # Fallback to heuristics
            if self.config.use_fallback:
                self._stats["fallback_decisions"] += 1
                return self._heuristic_orchestrate(query, orch_context)
            else:
                raise RuntimeError(f"Orchestration failed: {e}") from e

    async def _llm_orchestrate(
        self,
        query: str,
        context: OrchestrationContext,
    ) -> OrchestrationPlan:
        """Use Claude to make orchestration decision."""
        client = self._ensure_client()

        # Build context summary for orchestrator
        context_summary = self._summarize_context(context)

        user_message = f"""Analyze this query and decide how to process it:

Query: {query}

Context:
{context_summary}

Output your decision as a JSON object."""

        # Call the orchestrator model
        response = await client.complete(
            messages=[{"role": "user", "content": user_message}],
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            model=self.config.orchestrator_model,
            max_tokens=self.config.max_tokens,
            component=CostComponent.ROOT_PROMPT,
        )

        # Parse the response
        return self._parse_decision(response.content, query, context)

    def _summarize_context(self, context: OrchestrationContext) -> str:
        """Summarize context for orchestrator."""
        lines = [
            f"- Context tokens: {context.context_tokens}",
            f"- Current depth: {context.current_depth}",
            f"- Remaining budget: ${context.budget_remaining_dollars:.2f}",
            f"- Remaining tokens: {context.budget_remaining_tokens:,}",
        ]

        if context.tokens_used > 0:
            lines.append(f"- Tokens used so far: {context.tokens_used:,}")

        if context.complexity_signals:
            signals = [k for k, v in context.complexity_signals.items() if v]
            if signals:
                lines.append(f"- Complexity signals: {', '.join(signals)}")

        if context.forced_model:
            lines.append(f"- User requested model: {context.forced_model}")

        return "\n".join(lines)

    def _parse_decision(
        self,
        response: str,
        query: str,
        context: OrchestrationContext,
    ) -> OrchestrationPlan:
        """Parse LLM response into OrchestrationPlan."""
        # Extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response[:200]}")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        # Parse enums
        execution_mode = ExecutionMode(data.get("execution_mode", "balanced"))
        tool_access = ToolAccessLevel(data.get("tool_access", "read_only"))

        # Map model_tier string to enum
        tier_mapping = {
            "fast": ModelTier.FAST,
            "balanced": ModelTier.BALANCED,
            "powerful": ModelTier.POWERFUL,
            "code_specialist": ModelTier.CODE_SPECIALIST,
        }
        model_tier = tier_mapping.get(
            data.get("model_tier", "balanced"),
            ModelTier.BALANCED,
        )

        # Map query_type string to enum
        query_type_mapping = {
            "code": QueryType.CODE,
            "debugging": QueryType.DEBUGGING,
            "analytical": QueryType.ANALYTICAL,
            "planning": QueryType.PLANNING,
            "factual": QueryType.FACTUAL,
            "search": QueryType.SEARCH,
            "summarization": QueryType.SUMMARIZATION,
            "refactoring": QueryType.REFACTORING,
            "architecture": QueryType.ARCHITECTURE,
            "creative": QueryType.CREATIVE,
            "unknown": QueryType.UNKNOWN,
        }
        query_type = query_type_mapping.get(
            data.get("query_type", "unknown"),
            QueryType.UNKNOWN,
        )

        # Select model based on tier and availability
        primary_model = self._select_model(model_tier)

        return OrchestrationPlan(
            activate_rlm=data.get("activate_rlm", True),
            activation_reason=data.get("activation_reason", "llm_decision"),
            model_tier=model_tier,
            primary_model=primary_model,
            fallback_chain=self._get_fallbacks(model_tier, primary_model),
            depth_budget=min(3, max(0, data.get("depth_budget", 2))),
            tokens_per_depth=25_000,
            execution_mode=execution_mode,
            tool_access=tool_access,
            query_type=query_type,
            complexity_score=data.get("complexity_score", 0.5),
            signals=data.get("signals", []),
        )

    def _select_model(self, tier: ModelTier) -> str:
        """Select best available model for tier."""
        from .orchestration_schema import TIER_MODELS

        tier_models = TIER_MODELS.get(tier, TIER_MODELS[ModelTier.BALANCED])
        for model in tier_models:
            if model in self.available_models:
                return model

        # Fallback
        return self.available_models[0] if self.available_models else "sonnet"

    def _get_fallbacks(self, tier: ModelTier, primary: str) -> list[str]:
        """Get fallback models for the tier."""
        from .orchestration_schema import TIER_MODELS

        tier_models = TIER_MODELS.get(tier, TIER_MODELS[ModelTier.BALANCED])
        fallbacks = [m for m in tier_models if m in self.available_models and m != primary]
        return fallbacks[:2]  # Up to 2 fallbacks

    def _heuristic_orchestrate(
        self,
        query: str,
        context: OrchestrationContext,
    ) -> OrchestrationPlan:
        """Fall back to heuristic-based orchestration."""
        # Create a SessionContext for the classifier
        session_context = SessionContext(
            messages=[],
            files={},
            tool_outputs=[],
            working_memory={},
        )

        # Check activation
        should_activate, reason = should_activate_rlm(query, session_context)

        if not should_activate:
            return OrchestrationPlan.bypass(reason)

        # Get complexity signals
        signals = extract_complexity_signals(query, session_context)

        # Classify query type
        classification = self._query_classifier.classify(query)

        # Determine execution mode from complexity
        if classification.complexity > 0.7:
            mode = ExecutionMode.THOROUGH
        elif classification.complexity < 0.3:
            mode = ExecutionMode.FAST
        else:
            mode = ExecutionMode.BALANCED

        # Build plan from mode
        plan = OrchestrationPlan.from_mode(
            mode,
            query_type=classification.query_type,
            activation_reason=reason,
            available_models=self.available_models,
        )

        # Add signals
        plan.complexity_score = classification.complexity
        plan.signals = classification.signals
        plan.confidence = classification.confidence

        return plan

    def _compute_cache_key(self, query: str, context: OrchestrationContext) -> str:
        """Compute cache key for a query."""
        # Simple cache key based on query prefix and context state
        query_prefix = query[:100].lower().strip()
        context_key = f"{context.context_tokens // 10000}_{context.current_depth}"
        return f"{hash(query_prefix)}_{context_key}"

    def _update_cache(self, key: str, plan: OrchestrationPlan) -> None:
        """Update cache with new decision."""
        if len(self._decision_cache) >= self.config.cache_size:
            # Remove oldest entries
            oldest = list(self._decision_cache.keys())[: self.config.cache_size // 4]
            for k in oldest:
                del self._decision_cache[k]

        self._decision_cache[key] = plan

    def get_statistics(self) -> dict[str, Any]:
        """Get orchestration statistics."""
        total = (
            self._stats["llm_decisions"]
            + self._stats["fallback_decisions"]
            + self._stats["cache_hits"]
        )
        return {
            **self._stats,
            "total_decisions": total,
            "llm_rate": self._stats["llm_decisions"] / total if total > 0 else 0.0,
            "cache_hit_rate": self._stats["cache_hits"] / total if total > 0 else 0.0,
            "error_rate": self._stats["errors"] / (total + self._stats["errors"]) if total > 0 else 0.0,
        }


# Convenience function for one-shot orchestration
async def create_orchestration_plan(
    query: str,
    context: SessionContext | OrchestrationContext,
    client: ClaudeClient | None = None,
    use_llm: bool = True,
) -> OrchestrationPlan:
    """
    Create an orchestration plan for a query.

    Args:
        query: User query
        context: Session or orchestration context
        client: Optional Claude client
        use_llm: Whether to use LLM-based orchestration

    Returns:
        OrchestrationPlan
    """
    config = OrchestratorConfig(use_fallback=True)

    if not use_llm:
        # Use heuristics only
        orchestrator = IntelligentOrchestrator(config=config)
        if isinstance(context, SessionContext):
            orch_context = OrchestrationContext(
                query=query,
                context_tokens=context.total_tokens,
            )
        else:
            orch_context = context
        return orchestrator._heuristic_orchestrate(query, orch_context)

    orchestrator = IntelligentOrchestrator(client=client, config=config)
    return await orchestrator.create_plan(query, context)


__all__ = [
    "IntelligentOrchestrator",
    "OrchestratorConfig",
    "create_orchestration_plan",
]
