"""
Prompt optimization for RLM-Claude-Code.

Implements: Spec §8.1 Phase 3 - Prompt Optimization
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PromptType(Enum):
    """Types of prompts in the RLM system."""

    ROOT = "root"  # Initial RLM prompt
    RECURSIVE = "recursive"  # Sub-call prompts
    SUMMARIZATION = "summarization"  # Context summarization
    ANALYSIS = "analysis"  # Code/context analysis
    REPL_INSTRUCTION = "repl_instruction"  # REPL usage guidance


class StrategyType(Enum):
    """Query strategy types."""

    DIRECT = "direct"  # Answer directly
    DECOMPOSE = "decompose"  # Break into sub-tasks
    SEARCH_FIRST = "search_first"  # Search context then answer
    SUMMARIZE_FIRST = "summarize_first"  # Summarize then answer
    ITERATIVE = "iterative"  # Multiple refinement passes


@dataclass
class PromptVariant:
    """A variant of a prompt for A/B testing."""

    id: str
    prompt_type: PromptType
    template: str
    description: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptResult:
    """Result of using a prompt variant."""

    variant_id: str
    success: bool
    tokens_used: int
    execution_time_ms: float
    user_feedback: int | None = None  # -1, 0, 1 for negative/neutral/positive
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class VariantStats:
    """Statistics for a prompt variant."""

    variant_id: str
    uses: int = 0
    successes: int = 0
    failures: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    positive_feedback: int = 0
    negative_feedback: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / self.uses if self.uses > 0 else 0.0

    @property
    def avg_tokens(self) -> float:
        """Average tokens per use."""
        return self.total_tokens / self.uses if self.uses > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average execution time."""
        return self.total_time_ms / self.uses if self.uses > 0 else 0.0

    @property
    def feedback_score(self) -> float:
        """Net feedback score."""
        total = self.positive_feedback + self.negative_feedback
        if total == 0:
            return 0.0
        return (self.positive_feedback - self.negative_feedback) / total


# Default prompt templates

ROOT_PROMPT_V1 = """You are operating in RLM (Recursive Language Model) mode.

## Context
{context_summary}

## Query
{query}

## Available Tools
- Python REPL with `context` variable
- `peek(var, start, end)` - View portion of variable
- `search(var, pattern)` - Search in variable
- `recursive_query(query, context)` - Spawn sub-call

## Instructions
1. Analyze the query requirements
2. Use REPL to examine relevant context
3. Spawn recursive calls for complex sub-tasks
4. Signal completion with FINAL(answer) or FINAL_VAR(var_name)
"""

ROOT_PROMPT_V2 = """You are an RLM agent analyzing a complex query.

CONTEXT: {context_summary}

QUERY: {query}

STRATEGY:
1. First, understand what information you need
2. Use peek() to examine specific context portions
3. Use search() to find relevant content
4. Use recursive_query() for independent sub-problems
5. Synthesize findings into a complete answer

When done: FINAL(your_answer) or FINAL_VAR(variable_name)
"""

ROOT_PROMPT_V3 = """RLM Mode Active | Depth: {depth}

Context Stats:
- Messages: {message_count}
- Files: {file_count}
- Tool outputs: {tool_output_count}

Query: {query}

You have a Python REPL. Variables: context, conversation, files, tool_outputs
Helpers: peek(), search(), summarize(), recursive_query()

Think step-by-step. Use tools to verify assumptions.
End with: FINAL(answer)
"""

RECURSIVE_PROMPT_V1 = """Analyzing sub-query at depth {depth}.

Context provided:
{context}

Query: {query}

Provide a focused, factual response. If you need more information,
you can use the REPL to examine the context variable.
"""

RECURSIVE_PROMPT_V2 = """Sub-task Analysis (Depth {depth})

INPUT: {context}

TASK: {query}

Be precise and factual. Reference specific parts of the context.
If the context is insufficient, state what's missing.
"""

SUMMARIZATION_PROMPT = """Summarize the following content, preserving key facts and relationships.

Content type: {content_type}
Content:
{content}

Provide a concise summary that captures:
1. Main purpose/topic
2. Key entities and relationships
3. Important details that might be needed later
"""

ANALYSIS_PROMPT = """Analyze the following {analysis_type}:

{content}

Focus on:
{focus_areas}

Provide structured analysis with clear findings.
"""


class PromptLibrary:
    """
    Library of prompt variants with A/B testing support.

    Implements: Spec §8.1 Prompt optimization
    """

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize prompt library.

        Args:
            storage_path: Path for persistent storage of results
        """
        if storage_path is None:
            storage_path = Path.home() / ".claude" / "rlm-prompts"
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._variants: dict[str, PromptVariant] = {}
        self._stats: dict[str, VariantStats] = {}
        self._results: list[PromptResult] = []

        # Load default variants
        self._load_defaults()

        # Load persisted stats
        self._load_stats()

    def _load_defaults(self) -> None:
        """Load default prompt variants."""
        defaults = [
            PromptVariant(
                id="root_v1",
                prompt_type=PromptType.ROOT,
                template=ROOT_PROMPT_V1,
                description="Original root prompt with tool list",
            ),
            PromptVariant(
                id="root_v2",
                prompt_type=PromptType.ROOT,
                template=ROOT_PROMPT_V2,
                description="Strategy-focused root prompt",
            ),
            PromptVariant(
                id="root_v3",
                prompt_type=PromptType.ROOT,
                template=ROOT_PROMPT_V3,
                description="Compact root prompt with stats",
            ),
            PromptVariant(
                id="recursive_v1",
                prompt_type=PromptType.RECURSIVE,
                template=RECURSIVE_PROMPT_V1,
                description="Standard recursive prompt",
            ),
            PromptVariant(
                id="recursive_v2",
                prompt_type=PromptType.RECURSIVE,
                template=RECURSIVE_PROMPT_V2,
                description="Compact recursive prompt",
            ),
            PromptVariant(
                id="summarization_v1",
                prompt_type=PromptType.SUMMARIZATION,
                template=SUMMARIZATION_PROMPT,
                description="Standard summarization prompt",
            ),
            PromptVariant(
                id="analysis_v1",
                prompt_type=PromptType.ANALYSIS,
                template=ANALYSIS_PROMPT,
                description="Standard analysis prompt",
            ),
        ]

        for variant in defaults:
            self._variants[variant.id] = variant
            if variant.id not in self._stats:
                self._stats[variant.id] = VariantStats(variant_id=variant.id)

    def _load_stats(self) -> None:
        """Load persisted statistics."""
        stats_file = self.storage_path / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                data = json.load(f)
                for variant_id, stats_data in data.items():
                    self._stats[variant_id] = VariantStats(
                        variant_id=variant_id,
                        uses=stats_data.get("uses", 0),
                        successes=stats_data.get("successes", 0),
                        failures=stats_data.get("failures", 0),
                        total_tokens=stats_data.get("total_tokens", 0),
                        total_time_ms=stats_data.get("total_time_ms", 0.0),
                        positive_feedback=stats_data.get("positive_feedback", 0),
                        negative_feedback=stats_data.get("negative_feedback", 0),
                    )

    def _save_stats(self) -> None:
        """Persist statistics."""
        stats_file = self.storage_path / "stats.json"
        data = {}
        for variant_id, stats in self._stats.items():
            data[variant_id] = {
                "uses": stats.uses,
                "successes": stats.successes,
                "failures": stats.failures,
                "total_tokens": stats.total_tokens,
                "total_time_ms": stats.total_time_ms,
                "positive_feedback": stats.positive_feedback,
                "negative_feedback": stats.negative_feedback,
            }
        with open(stats_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_variant(self, variant: PromptVariant) -> None:
        """Add a new prompt variant."""
        self._variants[variant.id] = variant
        if variant.id not in self._stats:
            self._stats[variant.id] = VariantStats(variant_id=variant.id)

    def get_variant(self, variant_id: str) -> PromptVariant | None:
        """Get a specific variant."""
        return self._variants.get(variant_id)

    def get_variants_by_type(self, prompt_type: PromptType) -> list[PromptVariant]:
        """Get all variants of a specific type."""
        return [v for v in self._variants.values() if v.prompt_type == prompt_type]

    def select_variant(
        self,
        prompt_type: PromptType,
        strategy: str = "epsilon_greedy",
        epsilon: float = 0.1,
    ) -> PromptVariant:
        """
        Select a variant for A/B testing.

        Implements: Spec §8.1 Root prompt variants A/B test

        Args:
            prompt_type: Type of prompt needed
            strategy: Selection strategy ("epsilon_greedy", "ucb", "random")
            epsilon: Exploration rate for epsilon_greedy

        Returns:
            Selected variant
        """
        variants = self.get_variants_by_type(prompt_type)
        if not variants:
            raise ValueError(f"No variants available for {prompt_type}")

        if strategy == "random":
            return random.choice(variants)

        elif strategy == "epsilon_greedy":
            if random.random() < epsilon:
                # Explore: random selection
                return random.choice(variants)
            else:
                # Exploit: best performing
                return self._select_best(variants)

        elif strategy == "ucb":
            return self._select_ucb(variants)

        else:
            return variants[0]

    def _select_best(self, variants: list[PromptVariant]) -> PromptVariant:
        """Select best performing variant."""
        best_variant = variants[0]
        best_score = -1.0

        for variant in variants:
            stats = self._stats.get(variant.id)
            if stats and stats.uses > 0:
                # Combine success rate and feedback
                score = stats.success_rate * 0.7 + (stats.feedback_score + 1) / 2 * 0.3
                if score > best_score:
                    best_score = score
                    best_variant = variant

        return best_variant

    def _select_ucb(self, variants: list[PromptVariant]) -> PromptVariant:
        """Select using Upper Confidence Bound."""
        import math

        total_uses = sum(self._stats.get(v.id, VariantStats(v.id)).uses for v in variants)

        best_variant = variants[0]
        best_ucb = -1.0

        for variant in variants:
            stats = self._stats.get(variant.id, VariantStats(variant.id))
            if stats.uses == 0:
                # Prioritize untested variants
                return variant

            # UCB1 formula
            exploitation = stats.success_rate
            exploration = math.sqrt(2 * math.log(total_uses + 1) / stats.uses)
            ucb = exploitation + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_variant = variant

        return best_variant

    def render_prompt(
        self,
        variant: PromptVariant,
        **kwargs: Any,
    ) -> str:
        """
        Render a prompt template with variables.

        Args:
            variant: Prompt variant to render
            **kwargs: Template variables

        Returns:
            Rendered prompt string
        """
        try:
            return variant.template.format(**kwargs)
        except KeyError as e:
            # Return template with missing vars noted
            return variant.template + f"\n[Missing variable: {e}]"

    def record_result(self, result: PromptResult) -> None:
        """
        Record the result of using a prompt variant.

        Args:
            result: Result to record
        """
        self._results.append(result)

        # Update stats
        stats = self._stats.get(result.variant_id)
        if stats:
            stats.uses += 1
            stats.total_tokens += result.tokens_used
            stats.total_time_ms += result.execution_time_ms

            if result.success:
                stats.successes += 1
            else:
                stats.failures += 1

            if result.user_feedback is not None:
                if result.user_feedback > 0:
                    stats.positive_feedback += 1
                elif result.user_feedback < 0:
                    stats.negative_feedback += 1

        # Persist periodically
        if len(self._results) % 10 == 0:
            self._save_stats()

    def get_stats(self, variant_id: str) -> VariantStats | None:
        """Get statistics for a variant."""
        return self._stats.get(variant_id)

    def get_all_stats(self) -> dict[str, VariantStats]:
        """Get statistics for all variants."""
        return self._stats.copy()

    def get_recommendations(self, prompt_type: PromptType) -> list[dict[str, Any]]:
        """
        Get recommendations for prompt improvements.

        Args:
            prompt_type: Type of prompts to analyze

        Returns:
            List of recommendations
        """
        variants = self.get_variants_by_type(prompt_type)
        recommendations = []

        for variant in variants:
            stats = self._stats.get(variant.id)
            if not stats or stats.uses < 10:
                continue

            # Check for issues
            if stats.success_rate < 0.7:
                recommendations.append({
                    "variant_id": variant.id,
                    "issue": "low_success_rate",
                    "value": stats.success_rate,
                    "suggestion": "Consider revising prompt structure or adding examples",
                })

            if stats.avg_tokens > 5000:
                recommendations.append({
                    "variant_id": variant.id,
                    "issue": "high_token_usage",
                    "value": stats.avg_tokens,
                    "suggestion": "Consider making prompt more concise",
                })

            if stats.feedback_score < -0.3:
                recommendations.append({
                    "variant_id": variant.id,
                    "issue": "negative_feedback",
                    "value": stats.feedback_score,
                    "suggestion": "Review user feedback patterns",
                })

        return recommendations


class StrategySelector:
    """
    Select optimal strategy based on query characteristics.

    Implements: Spec §8.1 Strategy-specific prompts
    """

    def __init__(self):
        self._strategy_patterns: dict[StrategyType, list[str]] = {
            StrategyType.DIRECT: [
                r"what is",
                r"define",
                r"explain",
                r"describe",
            ],
            StrategyType.DECOMPOSE: [
                r"and then",
                r"first .* then",
                r"step by step",
                r"multiple",
            ],
            StrategyType.SEARCH_FIRST: [
                r"find",
                r"locate",
                r"where is",
                r"search for",
            ],
            StrategyType.SUMMARIZE_FIRST: [
                r"summarize",
                r"overview",
                r"brief",
                r"tldr",
            ],
            StrategyType.ITERATIVE: [
                r"refine",
                r"improve",
                r"iterate",
                r"optimize",
            ],
        }

    def select_strategy(
        self,
        query: str,
        context_size: int,
        has_code: bool = False,
    ) -> StrategyType:
        """
        Select best strategy for a query.

        Args:
            query: User query
            context_size: Size of context in tokens
            has_code: Whether context contains code

        Returns:
            Recommended strategy
        """
        import re

        query_lower = query.lower()

        # Check pattern matches
        for strategy, patterns in self._strategy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return strategy

        # Default based on context size
        if context_size > 50000:
            return StrategyType.SUMMARIZE_FIRST
        elif context_size > 20000:
            return StrategyType.SEARCH_FIRST
        elif has_code:
            return StrategyType.DECOMPOSE
        else:
            return StrategyType.DIRECT


# Global instances
_prompt_library: PromptLibrary | None = None
_strategy_selector: StrategySelector | None = None


def get_prompt_library() -> PromptLibrary:
    """Get global prompt library."""
    global _prompt_library
    if _prompt_library is None:
        _prompt_library = PromptLibrary()
    return _prompt_library


def get_strategy_selector() -> StrategySelector:
    """Get global strategy selector."""
    global _strategy_selector
    if _strategy_selector is None:
        _strategy_selector = StrategySelector()
    return _strategy_selector


__all__ = [
    "PromptLibrary",
    "PromptResult",
    "PromptType",
    "PromptVariant",
    "StrategySelector",
    "StrategyType",
    "VariantStats",
    "get_prompt_library",
    "get_strategy_selector",
]
