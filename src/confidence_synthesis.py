"""
Confidence-weighted synthesis for recursive results.

Implements: SPEC-10.01-10.06 Confidence-Weighted Synthesis
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class RecursiveResult:
    """
    Result from a recursive LLM call with confidence tracking.

    Implements: SPEC-10.01, SPEC-10.02
    """

    content: str
    confidence: float  # 0.0 to 1.0
    reasoning_trace: list[str]
    cost_usd: float


class SynthesisStrategy(Enum):
    """
    Strategy for synthesizing multiple results.

    Implements: SPEC-10.04
    """

    WEIGHTED = "weighted"  # Weight by confidence
    CONSENSUS = "consensus"  # Only include high-confidence agreement
    DIVERSE = "diverse"  # Present alternatives for user decision


@dataclass
class ConfidenceFlags:
    """
    Flags for synthesis results.

    Implements: SPEC-10.06
    """

    has_low_confidence: bool = False
    explanation: str = ""
    low_confidence_sources: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """
    Result of synthesizing multiple recursive results.

    Implements: SPEC-10.04
    """

    content: str
    confidence: float
    primary_source: str = ""
    alternatives: list[str] | None = None
    flags: ConfidenceFlags = field(default_factory=ConfidenceFlags)
    total_cost_usd: float = 0.0


# Threshold for low confidence flagging
LOW_CONFIDENCE_THRESHOLD = 0.3


class ConfidenceEstimator:
    """
    Estimate confidence from various signals.

    Implements: SPEC-10.03
    """

    # Known reliable sources for source reliability estimation
    RELIABLE_SOURCES = {"verified_api", "official_docs", "test_results", "type_checker"}

    def from_self_consistency(self, samples: list[str]) -> float:
        """
        Estimate confidence from self-consistency of multiple samples.

        Higher agreement = higher confidence.

        Args:
            samples: Multiple answer samples

        Returns:
            Confidence score 0.0-1.0
        """
        if not samples:
            return 0.0

        if len(samples) == 1:
            return 0.5  # Single sample, moderate confidence

        # Count occurrences of each unique answer
        counter = Counter(samples)
        most_common_count = counter.most_common(1)[0][1]

        # Agreement ratio
        agreement = most_common_count / len(samples)

        return agreement

    def from_reasoning_coherence(self, trace: list[str]) -> float:
        """
        Estimate confidence from reasoning chain coherence.

        Coherent reasoning chains indicate higher confidence.

        Args:
            trace: List of reasoning steps

        Returns:
            Confidence score 0.0-1.0
        """
        if not trace:
            return 0.0

        # Factors that increase coherence
        coherence_score = 0.0

        # More steps generally indicates more thorough reasoning
        step_factor = min(1.0, len(trace) / 5)
        coherence_score += step_factor * 0.3

        # Check for logical flow indicators
        flow_words = [
            "first",
            "then",
            "next",
            "finally",
            "therefore",
            "because",
            "analyzed",
            "identified",
        ]
        flow_count = sum(1 for step in trace for word in flow_words if word.lower() in step.lower())
        flow_factor = min(1.0, flow_count / 3)
        coherence_score += flow_factor * 0.4

        # Penalize uncertainty indicators
        uncertainty_words = [
            "maybe",
            "perhaps",
            "uncertain",
            "unclear",
            "unsure",
            "actually no",
            "guess",
        ]
        uncertainty_count = sum(
            1 for step in trace for word in uncertainty_words if word.lower() in step.lower()
        )
        uncertainty_penalty = min(0.5, uncertainty_count * 0.15)
        coherence_score -= uncertainty_penalty

        # Length of reasoning steps
        avg_length = sum(len(step) for step in trace) / len(trace)
        length_factor = min(1.0, avg_length / 50) * 0.3
        coherence_score += length_factor

        return max(0.0, min(1.0, coherence_score))

    def from_tool_success(self, results: list[bool]) -> float:
        """
        Estimate confidence from tool execution success rate.

        More successful tool calls = higher confidence.

        Args:
            results: List of success/failure booleans

        Returns:
            Confidence score 0.0-1.0
        """
        if not results:
            return 0.5  # No tools used, neutral confidence

        success_count = sum(1 for r in results if r)
        return success_count / len(results)

    def from_source_reliability(self, sources: list[str]) -> float:
        """
        Estimate confidence from source reliability.

        Verified sources increase confidence.

        Args:
            sources: List of source identifiers

        Returns:
            Confidence score 0.0-1.0
        """
        if not sources:
            return 0.5  # No sources, neutral confidence

        reliable_count = sum(1 for source in sources if source.lower() in self.RELIABLE_SOURCES)

        # Base confidence plus bonus for reliable sources
        base = 0.3
        reliable_bonus = (reliable_count / len(sources)) * 0.7

        return base + reliable_bonus


class WeightedSynthesizer:
    """
    Synthesize multiple recursive results using confidence weighting.

    Implements: SPEC-10.04-10.06
    """

    def __init__(self, strategy: SynthesisStrategy = SynthesisStrategy.WEIGHTED):
        """
        Initialize synthesizer.

        Args:
            strategy: Synthesis strategy to use
        """
        self.strategy = strategy

    def synthesize(self, results: list[RecursiveResult]) -> SynthesisResult:
        """
        Synthesize multiple results into a single result.

        Implements: SPEC-10.04-10.06

        Args:
            results: List of recursive results to synthesize

        Returns:
            SynthesisResult with combined content and metadata
        """
        if not results:
            return SynthesisResult(
                content="",
                confidence=0.0,
                flags=ConfidenceFlags(),
            )

        if len(results) == 1:
            result = results[0]
            flags = self._check_low_confidence([result])
            return SynthesisResult(
                content=result.content,
                confidence=result.confidence,
                primary_source=result.content,
                flags=flags,
                total_cost_usd=result.cost_usd,
            )

        # Apply strategy-specific synthesis
        if self.strategy == SynthesisStrategy.WEIGHTED:
            return self._synthesize_weighted(results)
        elif self.strategy == SynthesisStrategy.CONSENSUS:
            return self._synthesize_consensus(results)
        else:  # DIVERSE
            return self._synthesize_diverse(results)

    def _synthesize_weighted(self, results: list[RecursiveResult]) -> SynthesisResult:
        """Synthesize using confidence weighting."""
        # Sort by confidence descending
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        # Primary result is highest confidence
        primary = sorted_results[0]

        # Calculate weighted confidence
        total_weight = sum(r.confidence for r in results)
        if total_weight > 0:
            weighted_conf = sum(r.confidence * r.confidence for r in results) / total_weight
        else:
            weighted_conf = 0.0

        # Total cost
        total_cost = sum(r.cost_usd for r in results)

        # Check for low confidence
        flags = self._check_low_confidence(results)

        return SynthesisResult(
            content=primary.content,
            confidence=weighted_conf,
            primary_source=primary.content,
            flags=flags,
            total_cost_usd=total_cost,
        )

    def _synthesize_consensus(self, results: list[RecursiveResult]) -> SynthesisResult:
        """Synthesize using consensus strategy."""
        # Filter to high-confidence results
        high_conf = [r for r in results if r.confidence >= 0.5]

        if not high_conf:
            # Fall back to highest available
            high_conf = sorted(results, key=lambda r: r.confidence, reverse=True)[:1]

        # Group by content similarity (exact match for simplicity)
        content_groups: dict[str, list[RecursiveResult]] = {}
        for r in high_conf:
            if r.content not in content_groups:
                content_groups[r.content] = []
            content_groups[r.content].append(r)

        # Find consensus (most common content among high-confidence)
        consensus_content = max(content_groups.keys(), key=lambda k: len(content_groups[k]))

        # Average confidence of consensus group
        consensus_group = content_groups[consensus_content]
        avg_confidence = sum(r.confidence for r in consensus_group) / len(consensus_group)

        total_cost = sum(r.cost_usd for r in results)
        flags = self._check_low_confidence(results)

        return SynthesisResult(
            content=consensus_content,
            confidence=avg_confidence,
            primary_source=consensus_content,
            flags=flags,
            total_cost_usd=total_cost,
        )

    def _synthesize_diverse(self, results: list[RecursiveResult]) -> SynthesisResult:
        """Synthesize using diverse strategy - present alternatives."""
        # Sort by confidence
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        primary = sorted_results[0]
        alternatives = [r.content for r in sorted_results[1:] if r.content != primary.content]

        total_cost = sum(r.cost_usd for r in results)
        flags = self._check_low_confidence(results)

        return SynthesisResult(
            content=primary.content,
            confidence=primary.confidence,
            primary_source=primary.content,
            alternatives=alternatives if alternatives else None,
            flags=flags,
            total_cost_usd=total_cost,
        )

    def _check_low_confidence(self, results: list[RecursiveResult]) -> ConfidenceFlags:
        """
        Check for low-confidence results and create flags.

        Implements: SPEC-10.06
        """
        low_conf_results = [r for r in results if r.confidence < LOW_CONFIDENCE_THRESHOLD]

        if not low_conf_results:
            return ConfidenceFlags(has_low_confidence=False)

        # Build explanation
        explanations = []
        sources = []
        for r in low_conf_results:
            explanations.append(f"Result with confidence {r.confidence:.2f} may be unreliable")
            sources.append(r.content[:50] + "..." if len(r.content) > 50 else r.content)

        return ConfidenceFlags(
            has_low_confidence=True,
            explanation="; ".join(explanations),
            low_confidence_sources=sources,
        )


__all__ = [
    "ConfidenceEstimator",
    "ConfidenceFlags",
    "LOW_CONFIDENCE_THRESHOLD",
    "RecursiveResult",
    "SynthesisResult",
    "SynthesisStrategy",
    "WeightedSynthesizer",
]
