"""
Trajectory analysis for strategy extraction.

Implements: Spec §8.1 Phase 3 - Strategy Learning

Analyzes successful trajectories to identify effective strategies:
- Peeking: Sample context structure before deep processing
- Grepping: Use regex/patterns to narrow search space
- Partition+Map: Divide context, process chunks via sub-calls
- Programmatic: One-shot computational tasks via code execution
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .trajectory import TrajectoryEvent, TrajectoryEventType


class StrategyType(Enum):
    """Types of strategies detected in trajectories."""

    PEEKING = "peeking"  # Sample context structure first
    GREPPING = "grepping"  # Pattern-based search
    PARTITION_MAP = "partition_map"  # Divide and conquer
    PROGRAMMATIC = "programmatic"  # One-shot code execution
    RECURSIVE = "recursive"  # Recursive sub-calls
    ITERATIVE = "iterative"  # Multiple refinement rounds
    DIRECT = "direct"  # Single-shot answer
    UNKNOWN = "unknown"


@dataclass
class StrategySignal:
    """A signal indicating a particular strategy was used."""

    strategy: StrategyType
    confidence: float  # 0.0 to 1.0
    evidence: str  # What triggered this detection
    event_index: int  # Which event triggered it


@dataclass
class TrajectoryMetrics:
    """Metrics extracted from a trajectory."""

    # Basic counts
    total_events: int = 0
    repl_executions: int = 0
    recursive_calls: int = 0
    error_count: int = 0

    # Depth metrics
    max_depth: int = 0
    avg_depth: float = 0.0

    # Token/cost metrics
    total_tokens: int = 0
    total_cost: float = 0.0

    # Time metrics
    total_time_ms: float = 0.0
    avg_event_time_ms: float = 0.0

    # Success metrics
    completed: bool = False
    final_answer_found: bool = False


@dataclass
class StrategyAnalysis:
    """
    Complete analysis of a trajectory's strategies.

    Provides insights into what approaches were used during an RLM execution
    and how effective they were.

    Key Attributes:
        primary_strategy: The dominant strategy detected (e.g., SEARCH_AND_SYNTHESIZE)
        strategies: All detected strategies with confidence scores
        strategy_confidence: Confidence in the primary strategy identification
        metrics: Quantitative metrics (events, tokens, depth, etc.)
        success: Whether the trajectory completed successfully
        effectiveness_score: Overall effectiveness rating (0.0-1.0)

    Example:
        >>> from src.trajectory_analysis import TrajectoryAnalyzer
        >>> analyzer = TrajectoryAnalyzer()
        >>> analysis = analyzer.analyze(trajectory_events)
        >>> print(f"Strategy: {analysis.primary_strategy.value}")
        Strategy: search_and_synthesize
        >>> print(f"Confidence: {analysis.strategy_confidence:.0%}")
        Confidence: 85%
        >>> print(f"All strategies: {[s.strategy.value for s in analysis.strategies]}")
        All strategies: ['search_and_synthesize', 'recursive_decomposition']
    """

    # Detected strategies
    strategies: list[StrategySignal] = field(default_factory=list)
    primary_strategy: StrategyType = StrategyType.UNKNOWN
    strategy_confidence: float = 0.0

    # Metrics
    metrics: TrajectoryMetrics = field(default_factory=TrajectoryMetrics)

    # Pattern details
    code_patterns: list[str] = field(default_factory=list)
    search_patterns: list[str] = field(default_factory=list)

    # Summary
    success: bool = False
    effectiveness_score: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_strategy": self.primary_strategy.value,
            "strategy_confidence": self.strategy_confidence,
            "strategies": [
                {
                    "type": s.strategy.value,
                    "confidence": s.confidence,
                    "evidence": s.evidence,
                }
                for s in self.strategies
            ],
            "metrics": {
                "total_events": self.metrics.total_events,
                "repl_executions": self.metrics.repl_executions,
                "recursive_calls": self.metrics.recursive_calls,
                "max_depth": self.metrics.max_depth,
                "total_tokens": self.metrics.total_tokens,
                "completed": self.metrics.completed,
            },
            "code_patterns": self.code_patterns,
            "success": self.success,
            "effectiveness_score": self.effectiveness_score,
        }


class TrajectoryAnalyzer:
    """
    Analyze trajectories to extract strategy patterns.

    Implements: Spec §8.1 Strategy extraction from trajectories
    """

    # Pattern detectors for each strategy
    PEEKING_PATTERNS = [
        r"peek\(",
        r"files\[.*\]\[:.*\]",
        r"[:100]|[:50]|[:200]",
        r"head|tail|sample",
        r"first.*lines|last.*lines",
    ]

    GREPPING_PATTERNS = [
        r"grep\(",
        r"re\.search|re\.findall|re\.match",
        r"search\(",
        r"\.find\(|\.index\(",
        r"pattern.*in|matches|contains",
    ]

    PARTITION_MAP_PATTERNS = [
        r"for\s+\w+\s+in\s+(files|chunks|parts)",
        r"split\(|partition\(",
        r"recursive_query\(|llm_batch\(",
        r"summarize\(",
        r"map\(|reduce\(",
    ]

    PROGRAMMATIC_PATTERNS = [
        r"def\s+\w+\(",
        r"class\s+\w+",
        r"import\s+",
        r"json\.|ast\.|re\.",
        r"Counter|defaultdict|OrderedDict",
    ]

    def __init__(self):
        """Initialize analyzer."""
        self._compiled_patterns: dict[StrategyType, list[re.Pattern[str]]] = {
            StrategyType.PEEKING: [re.compile(p, re.IGNORECASE) for p in self.PEEKING_PATTERNS],
            StrategyType.GREPPING: [re.compile(p, re.IGNORECASE) for p in self.GREPPING_PATTERNS],
            StrategyType.PARTITION_MAP: [
                re.compile(p, re.IGNORECASE) for p in self.PARTITION_MAP_PATTERNS
            ],
            StrategyType.PROGRAMMATIC: [
                re.compile(p, re.IGNORECASE) for p in self.PROGRAMMATIC_PATTERNS
            ],
        }

    def analyze(self, events: list[TrajectoryEvent]) -> StrategyAnalysis:
        """
        Analyze a trajectory to extract strategies.

        Args:
            events: List of trajectory events

        Returns:
            StrategyAnalysis with detected strategies and metrics
        """
        analysis = StrategyAnalysis()

        if not events:
            return analysis

        # Extract metrics
        analysis.metrics = self._extract_metrics(events)

        # Detect strategies
        analysis.strategies = self._detect_strategies(events)

        # Determine primary strategy
        if analysis.strategies:
            strategy_counts = Counter(s.strategy for s in analysis.strategies)
            primary, count = strategy_counts.most_common(1)[0]
            analysis.primary_strategy = primary
            analysis.strategy_confidence = count / len(analysis.strategies)
        else:
            analysis.primary_strategy = StrategyType.DIRECT
            analysis.strategy_confidence = 1.0

        # Extract code patterns
        analysis.code_patterns = self._extract_code_patterns(events)
        analysis.search_patterns = self._extract_search_patterns(events)

        # Determine success
        analysis.success = analysis.metrics.final_answer_found and analysis.metrics.error_count == 0

        # Calculate effectiveness score
        analysis.effectiveness_score = self._calculate_effectiveness(analysis)

        return analysis

    def _extract_metrics(self, events: list[TrajectoryEvent]) -> TrajectoryMetrics:
        """Extract metrics from events."""
        metrics = TrajectoryMetrics()
        metrics.total_events = len(events)

        depths = []
        for event in events:
            depths.append(event.depth)

            if event.type == TrajectoryEventType.REPL_EXEC:
                metrics.repl_executions += 1
            elif event.type == TrajectoryEventType.RECURSE_START:
                metrics.recursive_calls += 1
            elif event.type == TrajectoryEventType.ERROR:
                metrics.error_count += 1
            elif event.type == TrajectoryEventType.FINAL:
                metrics.completed = True
                metrics.final_answer_found = True

            # Extract token/cost from metadata
            if event.metadata:
                if "input_tokens" in event.metadata:
                    metrics.total_tokens += event.metadata.get("input_tokens", 0)
                    metrics.total_tokens += event.metadata.get("output_tokens", 0)
                if "cost" in event.metadata and isinstance(event.metadata["cost"], dict):
                    metrics.total_cost += event.metadata["cost"].get("total_cost", 0.0)

        if depths:
            metrics.max_depth = max(depths)
            metrics.avg_depth = sum(depths) / len(depths)

        # Calculate timing
        if len(events) >= 2:
            time_span = events[-1].timestamp - events[0].timestamp
            metrics.total_time_ms = time_span * 1000
            metrics.avg_event_time_ms = metrics.total_time_ms / len(events)

        return metrics

    def _detect_strategies(self, events: list[TrajectoryEvent]) -> list[StrategySignal]:
        """Detect strategy signals from events."""
        signals = []

        for i, event in enumerate(events):
            # Only analyze REPL execution events
            if event.type != TrajectoryEventType.REPL_EXEC:
                continue

            content = event.content

            # Check each strategy pattern
            for strategy, patterns in self._compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(content):
                        signals.append(
                            StrategySignal(
                                strategy=strategy,
                                confidence=0.8,
                                evidence=pattern.pattern,
                                event_index=i,
                            )
                        )
                        break  # One match per strategy per event

        # Detect recursive strategy from event types
        recursive_count = sum(1 for e in events if e.type == TrajectoryEventType.RECURSE_START)
        if recursive_count > 0:
            signals.append(
                StrategySignal(
                    strategy=StrategyType.RECURSIVE,
                    confidence=min(1.0, recursive_count * 0.3),
                    evidence=f"{recursive_count} recursive calls",
                    event_index=0,
                )
            )

        # Detect iterative strategy from multiple rounds
        repl_count = sum(1 for e in events if e.type == TrajectoryEventType.REPL_EXEC)
        if repl_count > 3:
            signals.append(
                StrategySignal(
                    strategy=StrategyType.ITERATIVE,
                    confidence=min(1.0, repl_count * 0.2),
                    evidence=f"{repl_count} REPL executions",
                    event_index=0,
                )
            )

        return signals

    def _extract_code_patterns(self, events: list[TrajectoryEvent]) -> list[str]:
        """Extract notable code patterns from REPL events."""
        patterns = []

        for event in events:
            if event.type != TrajectoryEventType.REPL_EXEC:
                continue

            content = event.content

            # Detect function definitions
            func_match = re.search(r"def\s+(\w+)\s*\(", content)
            if func_match:
                patterns.append(f"function:{func_match.group(1)}")

            # Detect comprehensions
            if re.search(r"\[.*for.*in.*\]", content):
                patterns.append("list_comprehension")

            # Detect recursive calls
            if "recursive_query" in content:
                patterns.append("recursive_query")
            if "llm_batch" in content:
                patterns.append("llm_batch")

            # Detect file operations
            if "files[" in content:
                patterns.append("file_access")

        return list(set(patterns))

    def _extract_search_patterns(self, events: list[TrajectoryEvent]) -> list[str]:
        """Extract search patterns used."""
        patterns = []

        for event in events:
            if event.type != TrajectoryEventType.REPL_EXEC:
                continue

            content = event.content

            # Extract regex patterns
            regex_matches = re.findall(r"r['\"](.+?)['\"]", content)
            patterns.extend(regex_matches)

            # Extract search strings
            search_matches = re.findall(r"search\(['\"](.+?)['\"]", content)
            patterns.extend(search_matches)

        return patterns[:10]  # Limit to 10 patterns

    def _calculate_effectiveness(self, analysis: StrategyAnalysis) -> float:
        """Calculate effectiveness score for the trajectory."""
        score = 0.0

        # Base score for completion
        if analysis.success:
            score = 0.5

        # Bonus for efficiency
        if analysis.metrics.total_events > 0:
            # Fewer events for same result is better
            efficiency = 1.0 / (1.0 + analysis.metrics.total_events / 10.0)
            score += efficiency * 0.2

        # Bonus for low error rate
        if analysis.metrics.total_events > 0:
            error_rate = analysis.metrics.error_count / analysis.metrics.total_events
            score += (1.0 - error_rate) * 0.2

        # Bonus for clear strategy
        if analysis.strategy_confidence > 0.7:
            score += 0.1

        return min(1.0, score)


def analyze_trajectory(events: list[TrajectoryEvent]) -> StrategyAnalysis:
    """
    Convenience function to analyze a trajectory.

    Args:
        events: List of trajectory events

    Returns:
        StrategyAnalysis
    """
    analyzer = TrajectoryAnalyzer()
    return analyzer.analyze(events)


def extract_strategy_summary(events: list[TrajectoryEvent]) -> dict[str, Any]:
    """
    Extract a summary of strategies used.

    Args:
        events: List of trajectory events

    Returns:
        Dictionary with strategy summary
    """
    analysis = analyze_trajectory(events)
    return {
        "primary_strategy": analysis.primary_strategy.value,
        "confidence": analysis.strategy_confidence,
        "success": analysis.success,
        "effectiveness": analysis.effectiveness_score,
        "patterns": analysis.code_patterns,
    }


__all__ = [
    "StrategyAnalysis",
    "StrategySignal",
    "StrategyType",
    "TrajectoryAnalyzer",
    "TrajectoryMetrics",
    "analyze_trajectory",
    "extract_strategy_summary",
]
