"""
Tests for confidence-weighted synthesis.

@trace SPEC-10.01-10.06
"""

from __future__ import annotations

from src.confidence_synthesis import (
    ConfidenceEstimator,
    RecursiveResult,
    SynthesisStrategy,
    WeightedSynthesizer,
)

# --- Test fixtures ---


def create_high_confidence_result(content: str = "High confidence answer") -> RecursiveResult:
    """Create a high-confidence result."""
    return RecursiveResult(
        content=content,
        confidence=0.9,
        reasoning_trace=["Step 1: Analyzed", "Step 2: Verified", "Step 3: Confirmed"],
        cost_usd=0.05,
    )


def create_medium_confidence_result(content: str = "Medium confidence answer") -> RecursiveResult:
    """Create a medium-confidence result."""
    return RecursiveResult(
        content=content,
        confidence=0.6,
        reasoning_trace=["Step 1: Analyzed", "Step 2: Uncertain"],
        cost_usd=0.03,
    )


def create_low_confidence_result(content: str = "Low confidence answer") -> RecursiveResult:
    """Create a low-confidence result."""
    return RecursiveResult(
        content=content,
        confidence=0.2,
        reasoning_trace=["Step 1: Guessed"],
        cost_usd=0.02,
    )


def create_diverse_results() -> list[RecursiveResult]:
    """Create diverse results with different confidences."""
    return [
        create_high_confidence_result("Answer A"),
        create_medium_confidence_result("Answer B"),
        create_low_confidence_result("Answer C"),
    ]


# --- SPEC-10.01: Confidence tracking ---


class TestConfidenceTracking:
    """Tests for confidence tracking in recursive results."""

    def test_result_has_confidence(self) -> None:
        """
        @trace SPEC-10.01
        RecursiveResult should track confidence.
        """
        result = RecursiveResult(
            content="Test answer",
            confidence=0.85,
            reasoning_trace=["Reasoning step"],
            cost_usd=0.01,
        )

        assert result.confidence == 0.85
        assert 0.0 <= result.confidence <= 1.0


# --- SPEC-10.02: RecursiveResult structure ---


class TestRecursiveResultStructure:
    """Tests for RecursiveResult dataclass structure."""

    def test_result_has_required_fields(self) -> None:
        """
        @trace SPEC-10.02
        RecursiveResult should have all required fields.
        """
        result = RecursiveResult(
            content="Answer text",
            confidence=0.75,
            reasoning_trace=["Step 1", "Step 2"],
            cost_usd=0.05,
        )

        assert result.content == "Answer text"
        assert result.confidence == 0.75
        assert result.reasoning_trace == ["Step 1", "Step 2"]
        assert result.cost_usd == 0.05

    def test_confidence_bounds(self) -> None:
        """
        @trace SPEC-10.02
        Confidence should be between 0.0 and 1.0.
        """
        # Valid confidences
        low = RecursiveResult("a", 0.0, [], 0.0)
        high = RecursiveResult("b", 1.0, [], 0.0)

        assert low.confidence == 0.0
        assert high.confidence == 1.0


# --- SPEC-10.03: Confidence estimation ---


class TestConfidenceEstimation:
    """Tests for confidence estimation methods."""

    def test_estimate_from_self_consistency(self) -> None:
        """
        @trace SPEC-10.03
        Should estimate confidence from self-consistency.
        """
        estimator = ConfidenceEstimator()

        # Multiple samples with agreement
        samples = ["Answer A", "Answer A", "Answer A"]
        confidence = estimator.from_self_consistency(samples)

        assert confidence > 0.8  # High agreement

        # Multiple samples with disagreement
        diverse_samples = ["Answer A", "Answer B", "Answer C"]
        diverse_confidence = estimator.from_self_consistency(diverse_samples)

        assert diverse_confidence < 0.5  # Low agreement

    def test_estimate_from_reasoning_coherence(self) -> None:
        """
        @trace SPEC-10.03
        Should estimate confidence from reasoning chain coherence.
        """
        estimator = ConfidenceEstimator()

        # Coherent reasoning chain
        coherent_trace = [
            "First, I analyzed the input",
            "Then, I identified the pattern",
            "Finally, I applied the transformation",
        ]
        coherent_conf = estimator.from_reasoning_coherence(coherent_trace)

        # Incoherent reasoning chain
        incoherent_trace = ["Maybe this", "Actually no"]
        incoherent_conf = estimator.from_reasoning_coherence(incoherent_trace)

        assert coherent_conf > incoherent_conf

    def test_estimate_from_tool_success(self) -> None:
        """
        @trace SPEC-10.03
        Should estimate confidence from tool execution success.
        """
        estimator = ConfidenceEstimator()

        # All tools succeeded
        all_success = [True, True, True]
        success_conf = estimator.from_tool_success(all_success)

        # Some tools failed
        partial_success = [True, False, True]
        partial_conf = estimator.from_tool_success(partial_success)

        assert success_conf > partial_conf

    def test_estimate_from_source_reliability(self) -> None:
        """
        @trace SPEC-10.03
        Should estimate confidence from source reliability.
        """
        estimator = ConfidenceEstimator()

        # High reliability sources
        reliable = estimator.from_source_reliability(["verified_api", "official_docs"])

        # Low reliability sources
        unreliable = estimator.from_source_reliability(["user_comment", "unverified"])

        assert reliable >= unreliable


# --- SPEC-10.04: Synthesis strategies ---


class TestSynthesisStrategies:
    """Tests for synthesis strategy support."""

    def test_weighted_strategy(self) -> None:
        """
        @trace SPEC-10.04
        Weighted strategy should weight results by confidence.
        """
        synthesizer = WeightedSynthesizer(strategy=SynthesisStrategy.WEIGHTED)
        results = create_diverse_results()

        synthesis = synthesizer.synthesize(results)

        # High confidence result should have more influence
        assert "Answer A" in synthesis.content or synthesis.primary_source == "Answer A"

    def test_consensus_strategy(self) -> None:
        """
        @trace SPEC-10.04
        Consensus strategy should only include high-confidence agreement.
        """
        synthesizer = WeightedSynthesizer(strategy=SynthesisStrategy.CONSENSUS)

        # Results that agree
        agreeing_results = [
            RecursiveResult("Common answer", 0.8, [], 0.01),
            RecursiveResult("Common answer", 0.75, [], 0.01),
            RecursiveResult("Different answer", 0.3, [], 0.01),
        ]

        synthesis = synthesizer.synthesize(agreeing_results)

        # Should focus on the consensus
        assert "Common answer" in synthesis.content

    def test_diverse_strategy(self) -> None:
        """
        @trace SPEC-10.04
        Diverse strategy should include disagreements for user decision.
        """
        synthesizer = WeightedSynthesizer(strategy=SynthesisStrategy.DIVERSE)
        results = create_diverse_results()

        synthesis = synthesizer.synthesize(results)

        # Should present alternatives
        assert synthesis.alternatives is not None
        assert len(synthesis.alternatives) > 0


# --- SPEC-10.05: Default strategy ---


class TestDefaultStrategy:
    """Tests for default synthesis strategy."""

    def test_default_is_weighted(self) -> None:
        """
        @trace SPEC-10.05
        Default synthesis strategy should be weighted.
        """
        synthesizer = WeightedSynthesizer()

        assert synthesizer.strategy == SynthesisStrategy.WEIGHTED


# --- SPEC-10.06: Low confidence flagging ---


class TestLowConfidenceFlagging:
    """Tests for low confidence result flagging."""

    def test_flag_low_confidence_results(self) -> None:
        """
        @trace SPEC-10.06
        Results with confidence < 0.3 should be flagged for review.
        """
        synthesizer = WeightedSynthesizer()
        results = [
            create_high_confidence_result(),
            create_low_confidence_result(),  # confidence = 0.2
        ]

        synthesis = synthesizer.synthesize(results)

        # Should flag the low confidence result
        assert synthesis.flags is not None
        assert synthesis.flags.has_low_confidence

    def test_low_confidence_threshold(self) -> None:
        """
        @trace SPEC-10.06
        Threshold for low confidence should be 0.3.
        """
        synthesizer = WeightedSynthesizer()

        # Just below threshold
        below = RecursiveResult("Below", 0.29, [], 0.01)
        # Just above threshold
        above = RecursiveResult("Above", 0.31, [], 0.01)

        synthesis_below = synthesizer.synthesize([below])
        synthesis_above = synthesizer.synthesize([above])

        assert synthesis_below.flags.has_low_confidence
        assert not synthesis_above.flags.has_low_confidence

    def test_flagged_results_include_explanation(self) -> None:
        """
        @trace SPEC-10.06
        Flagged results should include explanation.
        """
        synthesizer = WeightedSynthesizer()
        results = [create_low_confidence_result()]

        synthesis = synthesizer.synthesize(results)

        assert synthesis.flags.explanation is not None
        assert len(synthesis.flags.explanation) > 0


# --- Integration tests ---


class TestConfidenceSynthesisIntegration:
    """Integration tests for confidence-weighted synthesis."""

    def test_full_synthesis_workflow(self) -> None:
        """
        Test complete synthesis workflow.
        """
        synthesizer = WeightedSynthesizer(strategy=SynthesisStrategy.WEIGHTED)

        # Multiple results from recursive calls
        results = [
            RecursiveResult(
                content="The function calculates factorial recursively",
                confidence=0.9,
                reasoning_trace=["Analyzed code", "Traced recursion"],
                cost_usd=0.05,
            ),
            RecursiveResult(
                content="The function computes factorial using recursion",
                confidence=0.85,
                reasoning_trace=["Read function", "Understood pattern"],
                cost_usd=0.04,
            ),
            RecursiveResult(
                content="Unknown function behavior",
                confidence=0.2,
                reasoning_trace=["Could not analyze"],
                cost_usd=0.02,
            ),
        ]

        synthesis = synthesizer.synthesize(results)

        # Should produce meaningful synthesis
        assert synthesis.content is not None
        assert len(synthesis.content) > 0
        # Should flag low confidence
        assert synthesis.flags.has_low_confidence
        # Total cost tracked
        assert synthesis.total_cost_usd > 0

    def test_empty_results_handling(self) -> None:
        """
        Empty results should return appropriate response.
        """
        synthesizer = WeightedSynthesizer()

        synthesis = synthesizer.synthesize([])

        assert synthesis.content == ""
        assert synthesis.confidence == 0.0

    def test_single_result_passthrough(self) -> None:
        """
        Single result should pass through with its confidence.
        """
        synthesizer = WeightedSynthesizer()
        result = create_high_confidence_result()

        synthesis = synthesizer.synthesize([result])

        assert synthesis.content == result.content
        assert synthesis.confidence == result.confidence
