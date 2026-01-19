"""
Intelligent orchestrator using Claude for decision-making.

Implements: Spec §8.1 Phase 2 - Orchestration Layer

Uses Claude (Haiku for speed) to make orchestration decisions,
with fallback to heuristic-based complexity_classifier.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from .api_client import ClaudeClient, init_client
from .complexity_classifier import should_activate_rlm
from .cost_tracker import CostComponent
from .memory_store import MemoryStore
from .orchestration_schema import (
    DecisionConfidence,
    ExecutionMode,
    ExecutionStrategy,
    OrchestrationContext,
    OrchestrationPlan,
    STRATEGY_DEFAULTS,
    ToolAccessLevel,
)
from .smart_router import ModelTier, QueryClassifier, QueryType
from .types import SessionContext

# System prompt for orchestration decisions
ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent orchestration engine for RLM (Recursive Language Model) Claude Code.

Your role: Analyze queries to determine whether recursive decomposition adds value, and configure execution optimally.

## Core Philosophy: When Does RLM Add Value?

RLM excels when the problem benefits from **externalized reasoning** — breaking complex tasks into sub-queries that build understanding incrementally. The key question is: "Would thinking out loud in structured sub-steps lead to a better answer than direct response?"

### High-Value RLM Scenarios (activate_rlm: true)

**1. Discovery Required** — The answer requires exploring unknowns
- "Why is the API returning 500 errors?" → Need to trace call paths, check logs, examine state
- "How does authentication work in this codebase?" → Must follow data flow across files
- "What's causing this test to be flaky?" → Need to identify race conditions, timing issues

**2. Multi-Source Synthesis** — Answer requires integrating information from multiple places
- "Update all usages of the deprecated API" → Must find all call sites, understand each context
- "Ensure consistent error handling across services" → Cross-module pattern application
- "What's the impact of changing this schema?" → Trace all consumers

**3. Reasoning Under Uncertainty** — Multiple valid interpretations exist
- "Something feels wrong with this implementation" → Need to enumerate potential issues
- "Best approach for adding caching?" → Compare multiple strategies with tradeoffs
- Ambiguous requirements needing clarification through exploration

**4. Deep Debugging** — Error requires understanding multiple system layers
- Stack traces spanning multiple modules
- Intermittent failures (timing, state, environment-dependent)
- "This used to work" scenarios requiring diff analysis
- Errors where the symptom is far from the cause

**5. Architectural Reasoning** — Decisions with downstream implications
- "Should we use microservices or monolith?" → Need to evaluate constraints, tradeoffs
- Design decisions requiring constraint satisfaction across multiple dimensions
- Migration planning with dependency analysis

**6. Pattern Exhaustion** — Must ensure completeness
- "Find all security vulnerabilities" → Systematic enumeration required
- "Ensure all edge cases are handled" → Need structured coverage analysis
- Code review for a large changeset

**7. Context Recovery** — Previous attempts failed or were confused
- Prior response was incorrect or incomplete
- User says "that's not right" or "try again"
- Same question being asked differently (user unsatisfied)

### Low-Value RLM Scenarios (activate_rlm: false)

**Direct Knowledge** — Answer is immediately retrievable
- "What's the syntax for Python list comprehension?"
- "Show me the contents of config.yaml"
- Single file reads, simple command execution

**Narrow Scope** — Task is well-defined and localized
- "Add a type annotation to this function"
- "Fix the typo in the error message"
- Single-file, single-function changes with clear requirements

**User Explicitly Knows** — User provides complete context
- "Refactor this function using [specific approach]" with code provided
- Requests with complete specifications and no ambiguity
- Copy/transform operations with clear input and output

**Conversational** — Not a task, just interaction
- Acknowledgments, clarifications, follow-up questions
- "Thanks", "OK", "Yes, do that"

## Decision Dimensions

### execution_mode
- **fast**: User wants quick response, low-stakes task, or explicitly said "quick"/"just"
- **balanced**: Default — reasonable effort without excessive cost
- **thorough**: User said "make sure"/"careful"/"thorough", high-stakes, or previous attempt failed

### model_tier
- **fast**: Factual lookups, simple transformations, summarization
- **balanced**: Most coding tasks, standard analysis, debugging
- **powerful**: Architectural decisions, complex planning, nuanced judgment
- **code_specialist**: Heavy code generation, refactoring, debugging in unfamiliar codebases

### depth_budget (0-3)
Think of depth as "how many levels of sub-questions might we need?"
- **0**: Direct answer possible, no decomposition needed
- **1**: May need one round of information gathering
- **2**: Multi-step reasoning: explore → analyze → synthesize
- **3**: Deep investigation: explore → sub-explore → analyze → cross-reference → synthesize

### tool_access
- **none**: Pure reasoning from context already provided
- **repl_only**: Need computation but not file access
- **read_only**: Need to examine files/codebase but not modify
- **full**: Task requires file modifications or system changes

## Signals to Extract

Identify which of these apply (include all relevant ones in "signals"):
- multi_file_scope: Query spans multiple files or modules
- cross_module_reasoning: Need to understand interactions between components
- temporal_reasoning: Involves history, changes over time, "used to work"
- debugging_deep: Error requires tracing through layers
- debugging_surface: Error is localized and clear
- discovery_required: Need to find/explore unknown information
- synthesis_required: Must combine multiple sources
- uncertainty_high: Multiple valid interpretations
- user_frustrated: Prior attempts failed, user expressing dissatisfaction
- user_urgent: Speed signals ("quick", "just", "fast")
- user_careful: Quality signals ("make sure", "careful", "thorough", "verify")
- well_specified: User provided complete, unambiguous requirements
- architectural: Design decisions with tradeoffs
- pattern_exhaustion: Must ensure completeness/coverage
- continuation: Continuing previous work
- knowledge_retrieval: Answer is direct lookup
- narrow_scope: Localized, well-defined change

## Output Format

Output ONLY a JSON object (no markdown, no explanation):

{
  "activate_rlm": true/false,
  "activation_reason": "2-3 word primary reason",
  "execution_mode": "fast" | "balanced" | "thorough",
  "model_tier": "fast" | "balanced" | "powerful" | "code_specialist",
  "depth_budget": 0-3,
  "tool_access": "none" | "repl_only" | "read_only" | "full",
  "query_type": "code" | "debugging" | "analytical" | "planning" | "factual" | "search" | "summarization" | "refactoring" | "architecture" | "creative" | "unknown",
  "complexity_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "signals": ["signal1", "signal2", ...]
}

The confidence field (0.0-1.0) indicates how certain you are about this routing decision. Low confidence (<0.5) suggests the query is ambiguous and might benefit from RLM even if individual signals don't trigger it."""


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

    # === Local Model Options ===
    # Use local model instead of API for orchestration decisions
    use_local_model: bool = False

    # Local model configuration preset: "ultra_fast", "balanced", "quality", "portable"
    local_model_preset: str = "ultra_fast"

    # Whether to fallback to API if local model fails
    fallback_to_api: bool = True

    # === Logging Options ===
    # Enable decision logging for training data collection
    log_decisions: bool = False

    # Path for decision log file
    log_path: str = "~/.rlm/orchestration_decisions.jsonl"

    # Whether to log heuristic decisions (can be noisy)
    log_heuristics: bool = True


class IntelligentOrchestrator:
    """
    Uses Claude to make intelligent orchestration decisions.

    Implements: Spec §8.1 Phase 2 - Orchestration Layer

    Supports three orchestration modes:
    1. Local model (fastest, no API cost) - use_local_model=True
    2. API-based (Claude Haiku) - default
    3. Heuristic fallback - when both fail

    Falls back to heuristic classifier on LLM failure.
    """

    def __init__(
        self,
        client: ClaudeClient | None = None,
        config: OrchestratorConfig | None = None,
        available_models: list[str] | None = None,
        telemetry: Any | None = None,
        memory_store: MemoryStore | None = None,
        session_id: str | None = None,
    ):
        """
        Initialize the intelligent orchestrator.

        Args:
            client: Claude API client (creates one if None)
            config: Orchestrator configuration
            available_models: List of available model names
            telemetry: Optional OrchestrationTelemetry for heuristic tracking
            memory_store: Optional MemoryStore for memory-augmented orientation
            session_id: Session ID for memory isolation (task/session tiers)
        """
        self._client = client
        self.config = config or OrchestratorConfig()
        self.available_models = available_models or ["sonnet", "haiku", "opus"]
        self._query_classifier = QueryClassifier()
        self._decision_cache: dict[str, OrchestrationPlan] = {}
        self._local_orchestrator: Any = None  # Lazy initialized
        self._decision_logger: Any = None  # Lazy initialized
        self._telemetry = telemetry  # OrchestrationTelemetry for per-heuristic tracking
        self._memory_store = memory_store  # For memory-augmented orientation (SPEC-12.04)
        self._session_id = session_id  # For memory isolation
        self._stats = {
            "llm_decisions": 0,
            "local_decisions": 0,
            "fallback_decisions": 0,
            "cache_hits": 0,
            "errors": 0,
            "memory_augmented": 0,
        }

    def _ensure_logger(self) -> Any:
        """Lazily initialize decision logger."""
        if self._decision_logger is None and self.config.log_decisions:
            from .orchestration_logger import LoggerConfig, OrchestrationLogger

            log_config = LoggerConfig(
                log_path=self.config.log_path,
                enabled=True,
                log_heuristics=self.config.log_heuristics,
            )
            self._decision_logger = OrchestrationLogger(config=log_config)
        return self._decision_logger

    def _ensure_local_orchestrator(self) -> Any:
        """Lazily initialize local orchestrator."""
        if self._local_orchestrator is None:
            from .local_orchestrator import RECOMMENDED_CONFIGS, LocalOrchestrator

            preset = self.config.local_model_preset
            if preset in RECOMMENDED_CONFIGS:
                local_config = RECOMMENDED_CONFIGS[preset]
            else:
                from .local_orchestrator import LocalModelConfig

                local_config = LocalModelConfig()

            self._local_orchestrator = LocalOrchestrator(
                config=local_config,
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            )
        return self._local_orchestrator

    def _ensure_client(self) -> ClaudeClient:
        """Ensure we have an API client."""
        if self._client is None:
            self._client = init_client()
        return self._client

    def _query_memory_context(
        self,
        query: str,
        limit: int = 5,
    ) -> tuple[list[str], str | None]:
        """
        Query memory store for relevant context before classification.

        Implements: SPEC-12.04 Memory-Augmented Orientation

        Args:
            query: The user query to find relevant context for
            limit: Maximum number of facts to retrieve

        Returns:
            Tuple of (relevant_facts, prior_strategy)
            - relevant_facts: List of fact content strings
            - prior_strategy: Strategy from successful past experience, or None
        """
        if self._memory_store is None:
            return [], None

        relevant_facts: list[str] = []
        prior_strategy: str | None = None

        try:
            # Search for relevant facts using BM25
            search_results = self._memory_store.search(query, limit=limit)
            for result in search_results:
                if result.bm25_score > 0.5:  # Only high-relevance facts
                    relevant_facts.append(result.content)

            # Query for successful experiences with similar queries
            experiences = self._memory_store.query_nodes(
                node_type="experience",
                tier="session",  # Prioritize recent session experiences
                limit=3,
                session_id=self._session_id,  # Session isolation
            )
            for exp in experiences:
                # Check if experience was successful and has strategy metadata
                if exp.metadata and exp.metadata.get("success"):
                    strategy = exp.metadata.get("strategy")
                    if strategy:
                        prior_strategy = strategy
                        break  # Use most recent successful strategy

            self._stats["memory_augmented"] += 1

        except Exception:
            # Memory query failures should not block orchestration
            pass

        return relevant_facts, prior_strategy

    def _adjust_plan_from_memory(
        self,
        plan: OrchestrationPlan,
        memory_facts: list[str],
        prior_strategy: str | None,
    ) -> OrchestrationPlan:
        """
        Adjust orchestration plan based on memory context.

        Implements: SPEC-12.04 Memory-Augmented Orientation

        If we have high-confidence facts about the topic:
        - Reduce depth budget (don't re-discover known facts)
        - Increase confidence in the approach

        If we have a successful prior strategy:
        - Use that strategy as a starting point
        - Potentially reduce exploration

        Args:
            plan: Original orchestration plan
            memory_facts: Relevant facts from memory
            prior_strategy: Strategy from successful past experience

        Returns:
            Adjusted OrchestrationPlan
        """
        # Store memory context in plan
        plan.memory_context = memory_facts
        plan.prior_strategy = prior_strategy

        # If we have multiple relevant facts, we can reduce exploration depth
        if len(memory_facts) >= 3:
            # Reduce depth by 1 since we already have context
            plan.depth_budget = max(1, plan.depth_budget - 1)
            plan.signals.append("memory_context_available")

        # If we have a prior successful strategy, boost confidence
        if prior_strategy:
            plan.confidence = min(1.0, plan.confidence + 0.15)
            plan.signals.append("prior_strategy_found")
            plan.metadata["prior_strategy"] = prior_strategy

        return plan

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

        # Query memory for context (SPEC-12.04 Memory-Augmented Orientation)
        memory_facts, prior_strategy = self._query_memory_context(query)

        # Check forced overrides (skip memory adjustment for bypass plans)
        if orch_context.forced_rlm is False:
            return OrchestrationPlan.bypass("user_forced_off")

        if orch_context.forced_mode is not None:
            plan = OrchestrationPlan.from_mode(
                orch_context.forced_mode,
                activation_reason="user_forced_mode",
                available_models=self.available_models,
            )
            return self._adjust_plan_from_memory(plan, memory_facts, prior_strategy)

        # Check cache
        cache_key = self._compute_cache_key(query, orch_context)
        if self.config.cache_enabled and cache_key in self._decision_cache:
            self._stats["cache_hits"] += 1
            return self._decision_cache[cache_key]

        # Build context summary for logging
        context_summary = self._summarize_context(orch_context)

        # Try local model first if configured
        if self.config.use_local_model:
            try:
                start_time = time.time()
                plan = await self._local_orchestrate(query, orch_context)
                latency_ms = (time.time() - start_time) * 1000
                self._stats["local_decisions"] += 1

                # Log decision
                self._log_decision(query, context_summary, plan, "local", latency_ms, orch_context)

                # Apply memory adjustments
                plan = self._adjust_plan_from_memory(plan, memory_facts, prior_strategy)

                # Cache the decision
                if self.config.cache_enabled:
                    self._update_cache(cache_key, plan)

                return plan

            except Exception as e:
                self._stats["errors"] += 1
                # Fall through to API if configured
                if not self.config.fallback_to_api:
                    if self.config.use_fallback:
                        self._stats["fallback_decisions"] += 1
                        plan = self._heuristic_orchestrate(query, orch_context)
                        self._log_decision(
                            query,
                            context_summary,
                            plan,
                            "heuristic",
                            0,
                            orch_context,
                            heuristics_triggered=plan.metadata.get("heuristics_triggered"),
                            heuristics_checked=plan.metadata.get("heuristics_checked"),
                        )
                        return self._adjust_plan_from_memory(plan, memory_facts, prior_strategy)
                    raise RuntimeError(f"Local orchestration failed: {e}") from e

        # Try API-based orchestration
        try:
            start_time = time.time()
            plan = await self._llm_orchestrate(query, orch_context)
            latency_ms = (time.time() - start_time) * 1000
            self._stats["llm_decisions"] += 1

            # Log decision
            self._log_decision(query, context_summary, plan, "api", latency_ms, orch_context)

            # Apply memory adjustments
            plan = self._adjust_plan_from_memory(plan, memory_facts, prior_strategy)

            # Cache the decision
            if self.config.cache_enabled:
                self._update_cache(cache_key, plan)

            return plan

        except Exception as e:
            self._stats["errors"] += 1

            # Fallback to heuristics
            if self.config.use_fallback:
                self._stats["fallback_decisions"] += 1
                plan = self._heuristic_orchestrate(query, orch_context)
                self._log_decision(
                    query,
                    context_summary,
                    plan,
                    "heuristic",
                    0,
                    orch_context,
                    heuristics_triggered=plan.metadata.get("heuristics_triggered"),
                    heuristics_checked=plan.metadata.get("heuristics_checked"),
                )
                return self._adjust_plan_from_memory(plan, memory_facts, prior_strategy)
            else:
                raise RuntimeError(f"Orchestration failed: {e}") from e

    def _log_decision(
        self,
        query: str,
        context_summary: str,
        plan: OrchestrationPlan,
        source: str,
        latency_ms: float,
        context: OrchestrationContext,
        heuristics_triggered: list[str] | None = None,
        heuristics_checked: list[str] | None = None,
    ) -> str | None:
        """
        Log an orchestration decision for training data.

        Returns:
            query_id if telemetry is enabled, None otherwise
        """
        query_id = None

        # Log to OrchestrationLogger
        logger = self._ensure_logger()
        if logger is not None:
            decision = plan.to_dict()
            logger.log_decision(
                query=query,
                context_summary=context_summary,
                decision=decision,
                source=source,
                latency_ms=latency_ms,
                context_tokens=context.context_tokens,
                model_used=self.config.orchestrator_model if source == "api" else "",
            )

        # Log to OrchestrationTelemetry (for heuristic tracking)
        if self._telemetry is not None and source == "heuristic":
            decision = plan.to_dict()
            query_id = self._telemetry.log_decision_with_heuristics(
                query=query,
                decision=decision,
                heuristics_triggered=heuristics_triggered or [],
                heuristics_checked=heuristics_checked or [],
                source=source,
                latency_ms=latency_ms,
            )
            # Store query_id in plan metadata for later outcome tracking
            plan.metadata["query_id"] = query_id

        return query_id

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

    async def _local_orchestrate(
        self,
        query: str,
        context: OrchestrationContext,
    ) -> OrchestrationPlan:
        """Use local model to make orchestration decision."""
        local_orch = self._ensure_local_orchestrator()

        # Build context summary for orchestrator
        context_summary = self._summarize_context(context)

        # Get decision from local model
        decision = await local_orch.orchestrate(
            query=query,
            context_summary=context_summary,
        )

        # Convert decision dict to OrchestrationPlan
        return self._parse_decision(json.dumps(decision), query, context)

    def _summarize_context(self, context: OrchestrationContext) -> str:
        """Summarize context for orchestrator with rich signals."""
        lines = [
            f"- Context tokens: {context.context_tokens:,}",
            f"- Current recursion depth: {context.current_depth}/3",
            f"- Remaining budget: ${context.budget_remaining_dollars:.2f}",
            f"- Remaining tokens: {context.budget_remaining_tokens:,}",
        ]

        if context.tokens_used > 0:
            lines.append(f"- Tokens used so far: {context.tokens_used:,}")

        # Add context size interpretation
        if context.context_tokens > 100_000:
            lines.append("- [SIGNAL] Very large context - may benefit from RLM decomposition")
        elif context.context_tokens > 50_000:
            lines.append("- [SIGNAL] Large context - moderate complexity likely")

        # Add depth interpretation
        if context.current_depth > 0:
            lines.append(f"- [SIGNAL] Already in recursive call (depth {context.current_depth})")

        if context.complexity_signals:
            signals = [k for k, v in context.complexity_signals.items() if v]
            if signals:
                lines.append(f"- Detected complexity signals: {', '.join(signals)}")

        if context.forced_model:
            lines.append(f"- User requested model: {context.forced_model}")

        if context.forced_mode:
            lines.append(f"- User requested mode: {context.forced_mode.value}")

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
            raise ValueError(f"Invalid JSON in response: {e}") from e

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

        # Extract confidence - low confidence may trigger RLM even if not explicitly activated
        confidence = data.get("confidence", 0.7)
        activate_rlm = data.get("activate_rlm", True)

        # If confidence is low (<0.5) and RLM was not activated, consider activating anyway
        # This captures edge cases where the orchestrator is uncertain
        if not activate_rlm and confidence < 0.5:
            activate_rlm = True
            activation_reason = "low_confidence_safety"
        else:
            activation_reason = data.get("activation_reason", "llm_decision")

        return OrchestrationPlan(
            activate_rlm=activate_rlm,
            activation_reason=activation_reason,
            model_tier=model_tier,
            primary_model=primary_model,
            fallback_chain=self._get_fallbacks(model_tier, primary_model),
            depth_budget=min(3, max(0, data.get("depth_budget", 2))),
            tokens_per_depth=25_000,
            execution_mode=execution_mode,
            tool_access=tool_access,
            query_type=query_type,
            complexity_score=data.get("complexity_score", 0.5),
            confidence=confidence,
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
        """
        Enhanced heuristic-based orchestration aligned with the new philosophy.

        This fallback captures the key signals from the improved prompt without
        requiring an LLM call. It's designed to be conservative (bias toward RLM
        activation when uncertain) while avoiding unnecessary overhead for
        clearly simple tasks.

        Tracks which heuristics were checked and triggered for telemetry.
        """
        query_lower = query.lower()

        # Track heuristics for telemetry (Feature 3e0.6)
        heuristics_checked: list[str] = []
        heuristics_triggered: list[str] = []

        # === High-Value RLM Signals (activate if any are present) ===
        high_value_signals: list[str] = []

        # 1. Discovery Required
        heuristics_checked.append("discovery_required")
        discovery_patterns = [
            r"\bwhy\s+(is|does|did|are|was|were)\b",
            r"\bhow\s+does\b.*\bwork\b",
            r"\bwhat('s| is)\s+causing\b",
            r"\bwhere\s+(is|are|does)\b.*\b(defined|implemented|used|called)\b",
            r"\btrace\b",
            r"\bfollow\b.*\b(flow|path|data)\b",
        ]
        if any(re.search(p, query_lower) for p in discovery_patterns):
            high_value_signals.append("discovery_required")
            heuristics_triggered.append("discovery_required")

        # 2. Multi-Source Synthesis
        heuristics_checked.append("synthesis_required")
        synthesis_patterns = [
            r"\ball\s+(usages?|instances?|occurrences?|references?)\b",
            r"\bupdate\s+(all|every)\b",
            r"\bensure\s+(consistent|all)\b",
            r"\bimpact\s+of\s+(changing|modifying)\b",
            r"\bacross\s+(modules?|files?|services?|components?)\b",
            # Code changes to modules/components need multi-file impact analysis
            r"\b(add|implement|create|introduce)\s+\w+.*\b(to|for|in)\s+(the\s+)?\w*\s*(module|service|component|class|package)\b",
        ]
        if any(re.search(p, query_lower) for p in synthesis_patterns):
            high_value_signals.append("synthesis_required")
            heuristics_triggered.append("synthesis_required")

        # 3. Uncertainty / Multiple Interpretations
        heuristics_checked.append("uncertainty_high")
        uncertainty_patterns = [
            r"\b(best|better|optimal|right)\s+(way|approach|method)\b",
            r"\bshould\s+(i|we)\b",
            r"\bcompare\b.*\b(options?|approaches?|strategies?)\b",
            r"\btrade.?offs?\b",
            r"\bpros?\s+and\s+cons?\b",
            r"\bsomething\s+(feels?|seems?)\s+(wrong|off|weird)\b",
        ]
        if any(re.search(p, query_lower) for p in uncertainty_patterns):
            high_value_signals.append("uncertainty_high")
            heuristics_triggered.append("uncertainty_high")

        # 4. Deep Debugging
        heuristics_checked.append("debugging_deep")
        deep_debug_patterns = [
            r"\bintermittent\b",
            r"\bflaky\b",
            r"\brandom(ly)?\s+(fails?|errors?)\b",
            r"\brace\s+condition\b",
            r"\bused\s+to\s+work\b",
            r"\bstack\s*trace\b",
            r"\bmultiple\s+(errors?|failures?|exceptions?)\b",
            r"\bcan'?t\s+(figure|understand)\b",
        ]
        if any(re.search(p, query_lower) for p in deep_debug_patterns):
            high_value_signals.append("debugging_deep")
            heuristics_triggered.append("debugging_deep")

        # 5. Architectural Reasoning
        heuristics_checked.append("architectural")
        architecture_patterns = [
            r"\b(design|architect)\b.*\b(system|api|service)\b",
            r"\bmicroservices?\b",
            r"\bmonolith\b",
            r"\bscalability\b",
            r"\bmigrat(e|ion)\b",
            r"\brefactor\b.*\b(entire|whole|major)\b",
        ]
        if any(re.search(p, query_lower) for p in architecture_patterns):
            high_value_signals.append("architectural")
            heuristics_triggered.append("architectural")

        # 6. Pattern Exhaustion / Completeness
        heuristics_checked.append("pattern_exhaustion")
        exhaustion_patterns = [
            r"\bfind\s+all\b",
            r"\b(every|all)\s+(edge\s+)?cases?\b",
            r"\bcomprehensive\b",
            r"\bsecurity\s+(audit|review|vulnerabilities?)\b",
            r"\bmake\s+sure\b.*\b(all|every|nothing)\b",
            r"\bmissing\s+(anything|something)\b",
        ]
        if any(re.search(p, query_lower) for p in exhaustion_patterns):
            high_value_signals.append("pattern_exhaustion")
            heuristics_triggered.append("pattern_exhaustion")

        # 7. Context Recovery (from context signals)
        heuristics_checked.append("user_frustrated")
        if context.complexity_signals.get("previous_turn_was_confused"):
            high_value_signals.append("user_frustrated")
            heuristics_triggered.append("user_frustrated")

        # === Low-Value Signals (skip RLM if ONLY these are present) ===
        low_value_signals: list[str] = []

        # Direct Knowledge
        heuristics_checked.append("knowledge_retrieval")
        knowledge_patterns = [
            r"^(show|cat|read|view|open)\s+[\w./]+$",
            r"^what('s| is)\s+the\s+syntax\b",
            r"^how\s+do\s+(i|you)\s+(write|use)\b.*\bin\s+\w+$",
        ]
        if any(re.match(p, query_lower) for p in knowledge_patterns):
            low_value_signals.append("knowledge_retrieval")
            heuristics_triggered.append("knowledge_retrieval")

        # Narrow Scope
        heuristics_checked.append("narrow_scope")
        narrow_patterns = [
            r"^(add|fix|change|update)\s+(a|the|this)\s+(typo|comment|annotation|import)\b",
            r"^rename\s+\w+\s+to\s+\w+$",
        ]
        if any(re.match(p, query_lower) for p in narrow_patterns):
            low_value_signals.append("narrow_scope")
            heuristics_triggered.append("narrow_scope")

        # Conversational
        heuristics_checked.append("conversational")
        conversational_patterns = [
            r"^(ok(ay)?|yes|no|sure|thanks?|got\s+it|understood)\.?$",
            r"^(do\s+(it|that)|go\s+ahead|proceed|continue)\.?$",
        ]
        if any(re.match(p, query_lower.strip()) for p in conversational_patterns):
            low_value_signals.append("conversational")
            heuristics_triggered.append("conversational")
            plan = OrchestrationPlan.bypass("conversational")
            # Store heuristics for telemetry
            plan.metadata["heuristics_checked"] = heuristics_checked
            plan.metadata["heuristics_triggered"] = heuristics_triggered
            return plan

        # === User Intent Signals ===
        user_signals: list[str] = []

        # Urgency
        heuristics_checked.append("user_urgent")
        if re.search(r"\b(quick(ly)?|just|fast|simple)\b", query_lower):
            user_signals.append("user_urgent")
            heuristics_triggered.append("user_urgent")

        # Carefulness
        heuristics_checked.append("user_careful")
        if re.search(
            r"\b(make\s+sure|careful(ly)?|thorough(ly)?|verify|double.?check)\b", query_lower
        ):
            user_signals.append("user_careful")
            heuristics_triggered.append("user_careful")

        # === Decision Logic ===
        all_signals = high_value_signals + low_value_signals + user_signals

        # If only low-value signals and no high-value signals, skip RLM
        if low_value_signals and not high_value_signals:
            # Unless it's a large context
            if context.context_tokens < 50_000:
                plan = OrchestrationPlan.bypass(f"low_value:{low_value_signals[0]}")
                plan.metadata["heuristics_checked"] = heuristics_checked
                plan.metadata["heuristics_triggered"] = heuristics_triggered
                return plan

        # Get query classification for type
        classification = self._query_classifier.classify(query)

        # Determine if RLM should activate
        should_activate = len(high_value_signals) > 0 or context.context_tokens > 80_000

        # Also check traditional signals
        session_context = SessionContext(
            messages=[],
            files={},
            tool_outputs=[],
            working_memory={},
        )
        traditional_activate, traditional_reason = should_activate_rlm(query, session_context)

        if traditional_activate and not should_activate:
            should_activate = True
            all_signals.append(traditional_reason)

        if not should_activate:
            plan = OrchestrationPlan.bypass("simple_task")
            plan.metadata["heuristics_checked"] = heuristics_checked
            plan.metadata["heuristics_triggered"] = heuristics_triggered
            return plan

        # Determine execution mode
        if "user_careful" in user_signals or "pattern_exhaustion" in high_value_signals:
            mode = ExecutionMode.THOROUGH
        elif "user_urgent" in user_signals and len(high_value_signals) <= 1:
            mode = ExecutionMode.FAST
        elif classification.complexity > 0.7 or len(high_value_signals) >= 2:
            mode = ExecutionMode.THOROUGH
        elif "synthesis_required" in high_value_signals:
            # Code changes needing synthesis (multi-file impact) require BALANCED mode
            mode = ExecutionMode.BALANCED
            heuristics_triggered.append("mode_override:synthesis_balanced")
        elif classification.complexity < 0.3 and len(high_value_signals) <= 1:
            mode = ExecutionMode.FAST
        else:
            mode = ExecutionMode.BALANCED

        # === Signal-Based Model Selection (SPEC-12.05) ===
        # Override model tier based on specific signal types
        model_tier: ModelTier | None = None
        if "architectural" in high_value_signals:
            # Architecture decisions need powerful reasoning
            model_tier = ModelTier.POWERFUL
            heuristics_triggered.append("model_tier_override:powerful")
        elif "debugging_deep" in high_value_signals:
            # Deep debugging benefits from balanced + deeper search
            model_tier = ModelTier.BALANCED
            heuristics_triggered.append("model_tier_override:balanced")
        elif "pattern_exhaustion" in high_value_signals or "synthesis_required" in high_value_signals:
            # Enumeration tasks can use faster models
            model_tier = ModelTier.FAST
            heuristics_triggered.append("model_tier_override:fast")

        # === Context-Aware Depth (SPEC-12.05) ===
        # Adjust depth based on context size and signals
        base_depth = 1
        if "debugging_deep" in high_value_signals or "architectural" in high_value_signals:
            base_depth = 3
        elif (
            "synthesis_required" in high_value_signals
            or "pattern_exhaustion" in high_value_signals
            or "discovery_required" in high_value_signals
        ):
            base_depth = 2

        # Large context (many files/tokens) → increase depth for better decomposition
        if context.context_tokens > 100_000:
            base_depth = min(3, base_depth + 1)
            heuristics_triggered.append("depth_adjust:large_context")
        elif context.context_tokens > 50_000:
            # Moderate context, slight depth increase
            base_depth = min(3, base_depth + 1) if base_depth < 2 else base_depth

        # Prior confusion → increase depth for recovery
        if context.complexity_signals.get("previous_turn_was_confused"):
            base_depth = min(3, base_depth + 1)
            heuristics_triggered.append("depth_adjust:prior_confusion")

        depth_budget = base_depth

        # === Tool Access Inference (SPEC-12.05) ===
        # Infer appropriate tool access from query keywords
        tool_access: ToolAccessLevel | None = None

        # Check for write intent
        write_patterns = [
            r"\b(fix|update|modify|change|add|remove|delete|refactor|rename)\b",
            r"\b(create|write|implement|build)\b",
        ]
        if any(re.search(p, query_lower) for p in write_patterns):
            tool_access = ToolAccessLevel.FULL
            heuristics_triggered.append("tool_access:full")

        # Check for read-only intent (lower priority than write)
        elif re.search(r"\b(read|view|show|explain|understand|analyze|find|search|look)\b", query_lower):
            tool_access = ToolAccessLevel.READ_ONLY
            heuristics_triggered.append("tool_access:read_only")

        # Check for pure reasoning (no file access needed)
        elif re.search(r"\b(what\s+is|how\s+to|best\s+practice|syntax|concept)\b", query_lower):
            tool_access = ToolAccessLevel.REPL_ONLY
            heuristics_triggered.append("tool_access:repl_only")

        # === Strategy Inference (SPEC-12.06) ===
        # Map signals to execution strategies based on JTBD analysis
        strategy: ExecutionStrategy = ExecutionStrategy.DISCOVERY  # Default

        if "debugging_deep" in high_value_signals:
            # JTBD-1: Debug complex, multi-layer issues
            strategy = ExecutionStrategy.RECURSIVE_DEBUG
            heuristics_triggered.append("strategy:recursive_debug")
        elif "architectural" in high_value_signals or "uncertainty_high" in high_value_signals:
            # JTBD-4: Make informed architectural decisions
            strategy = ExecutionStrategy.ARCHITECTURE
            heuristics_triggered.append("strategy:architecture")
        elif "pattern_exhaustion" in high_value_signals:
            # JTBD-5: Ensure security/quality completeness
            strategy = ExecutionStrategy.MAP_REDUCE
            heuristics_triggered.append("strategy:map_reduce")
        elif "synthesis_required" in high_value_signals:
            # JTBD-3: Make comprehensive changes across codebase
            strategy = ExecutionStrategy.EXHAUSTIVE_SEARCH
            heuristics_triggered.append("strategy:exhaustive_search")
        elif "discovery_required" in high_value_signals:
            # JTBD-2: Understand unfamiliar codebase
            strategy = ExecutionStrategy.DISCOVERY
            heuristics_triggered.append("strategy:discovery")
        elif context.complexity_signals.get("task_is_continuation"):
            # JTBD-6: Resume/continue previous work
            strategy = ExecutionStrategy.CONTINUATION
            heuristics_triggered.append("strategy:continuation")

        # Get strategy hints for the selected strategy
        strategy_hints = list(STRATEGY_DEFAULTS[strategy]["hints"])

        # Determine primary reason
        if high_value_signals:
            activation_reason = high_value_signals[0]
        else:
            activation_reason = "complexity_threshold"

        # Build plan from mode
        plan = OrchestrationPlan.from_mode(
            mode,
            query_type=classification.query_type,
            activation_reason=activation_reason,
            available_models=self.available_models,
        )

        # Apply signal-based overrides
        if model_tier is not None:
            plan.model_tier = model_tier
            plan.primary_model = self._select_model(model_tier)
            plan.fallback_chain = self._get_fallbacks(model_tier, plan.primary_model)

        # Apply computed depth
        plan.depth_budget = depth_budget

        # Apply inferred tool access
        if tool_access is not None:
            plan.tool_access = tool_access

        # Apply inferred strategy
        plan.strategy = strategy
        plan.strategy_hints = strategy_hints

        # Add all signals
        plan.complexity_score = classification.complexity
        plan.signals = all_signals
        plan.confidence = classification.confidence

        # === Compute Per-Dimension Confidence (SPEC-12.07) ===
        # Activation confidence: high if multiple signals agree, lower if ambiguous
        if len(high_value_signals) >= 2:
            activation_conf = 0.9  # Multiple confirming signals
        elif len(high_value_signals) == 1:
            activation_conf = 0.75  # Single signal
        elif context.context_tokens > 80_000:
            activation_conf = 0.7  # Large context threshold
        else:
            activation_conf = 0.6  # Edge case activation

        # Model tier confidence: high if explicit signal override
        if model_tier is not None:
            model_conf = 0.85  # Signal-based override
        else:
            model_conf = 0.65  # Default from mode

        # Depth confidence: high if signal-driven, lower for context-based adjustment
        if "debugging_deep" in high_value_signals or "architectural" in high_value_signals:
            depth_conf = 0.85  # Clear depth requirement
        elif "depth_adjust:large_context" in heuristics_triggered:
            depth_conf = 0.6  # Context-based adjustment (less certain)
        elif "depth_adjust:prior_confusion" in heuristics_triggered:
            depth_conf = 0.55  # Recovery mode (uncertain)
        else:
            depth_conf = 0.7  # Default depth from signals

        # Strategy confidence: high if clear signal match
        strategy_signals = [h for h in heuristics_triggered if h.startswith("strategy:")]
        if strategy_signals:
            strategy_conf = 0.8  # Explicit strategy match
        elif context.complexity_signals.get("task_is_continuation"):
            strategy_conf = 0.75  # Continuation from context
        else:
            strategy_conf = 0.6  # Fallback to default strategy

        plan.decision_confidence = DecisionConfidence(
            activation=activation_conf,
            model_tier=model_conf,
            depth=depth_conf,
            strategy=strategy_conf,
        )

        # Store heuristics for telemetry
        plan.metadata["heuristics_checked"] = heuristics_checked
        plan.metadata["heuristics_triggered"] = heuristics_triggered

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
            + self._stats["local_decisions"]
            + self._stats["fallback_decisions"]
            + self._stats["cache_hits"]
        )
        return {
            **self._stats,
            "total_decisions": total,
            "llm_rate": self._stats["llm_decisions"] / total if total > 0 else 0.0,
            "local_rate": self._stats["local_decisions"] / total if total > 0 else 0.0,
            "cache_hit_rate": self._stats["cache_hits"] / total if total > 0 else 0.0,
            "error_rate": self._stats["errors"] / (total + self._stats["errors"])
            if total > 0
            else 0.0,
        }


# Convenience function for one-shot orchestration
async def create_orchestration_plan(
    query: str,
    context: SessionContext | OrchestrationContext,
    client: ClaudeClient | None = None,
    use_llm: bool = True,
    memory_store: MemoryStore | None = None,
) -> OrchestrationPlan:
    """
    Create an orchestration plan for a query.

    Args:
        query: User query
        context: Session or orchestration context
        client: Optional Claude client
        use_llm: Whether to use LLM-based orchestration
        memory_store: Optional MemoryStore for memory-augmented orientation

    Returns:
        OrchestrationPlan
    """
    config = OrchestratorConfig(use_fallback=True)

    if not use_llm:
        # Use heuristics only (still supports memory augmentation)
        orchestrator = IntelligentOrchestrator(config=config, memory_store=memory_store)
        if isinstance(context, SessionContext):
            orch_context = OrchestrationContext(
                query=query,
                context_tokens=context.total_tokens,
            )
        else:
            orch_context = context
        # Query memory and apply adjustments for heuristic mode too
        memory_facts, prior_strategy = orchestrator._query_memory_context(query)
        plan = orchestrator._heuristic_orchestrate(query, orch_context)
        return orchestrator._adjust_plan_from_memory(plan, memory_facts, prior_strategy)

    orchestrator = IntelligentOrchestrator(
        client=client, config=config, memory_store=memory_store
    )
    return await orchestrator.create_plan(query, context)


__all__ = [
    "IntelligentOrchestrator",
    "OrchestratorConfig",
    "create_orchestration_plan",
]
