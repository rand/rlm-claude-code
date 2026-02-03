"""
Property-based tests for epistemic verification logic.

Implements: SPEC-16.38 Property tests for verification logic
"""

import sys
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.epistemic.similarity import (
    SimilarityMethod,
    SimilarityResult,
    cosine_similarity,
    text_overlap_similarity,
)
from src.epistemic.types import (
    ClaimVerification,
    EpistemicGap,
    HallucinationReport,
    VerificationConfig,
)
from src.epistemic.verification_feedback import (
    FeedbackStatistics,
)

# ============================================================================
# Strategies for generating test data
# ============================================================================

# Strategy for valid scores (0.0 to 1.0)
score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for positive floats
positive_float = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Strategy for claim IDs
claim_id_strategy = st.text(
    min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-"
)

# Strategy for claim text
claim_text_strategy = st.text(min_size=1, max_size=200)

# Strategy for vectors (for cosine similarity)
# Use reasonable bounds to avoid floating point underflow issues with subnormal numbers
vector_strategy = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False).filter(
        lambda x: x == 0.0 or abs(x) > 1e-100  # Avoid subnormal floats that underflow
    ),
    min_size=1,
    max_size=100,
)


# ============================================================================
# Similarity Properties
# ============================================================================


class TestSimilarityProperties:
    """Property-based tests for semantic similarity."""

    @given(vector_strategy)
    @settings(max_examples=100)
    def test_cosine_similarity_self_is_one(self, vec: list[float]) -> None:
        """Cosine similarity of a vector with itself is 1.0 (if non-zero)."""
        # Skip zero vectors
        assume(any(v != 0.0 for v in vec))
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-10, f"Self-similarity should be 1.0, got {similarity}"

    @given(vector_strategy, vector_strategy)
    @settings(max_examples=100)
    def test_cosine_similarity_symmetric(self, vec_a: list[float], vec_b: list[float]) -> None:
        """Cosine similarity is symmetric: sim(a, b) == sim(b, a)."""
        # Vectors must be same length
        min_len = min(len(vec_a), len(vec_b))
        assume(min_len > 0)
        vec_a = vec_a[:min_len]
        vec_b = vec_b[:min_len]

        # Skip zero vectors
        assume(any(v != 0.0 for v in vec_a))
        assume(any(v != 0.0 for v in vec_b))

        sim_ab = cosine_similarity(vec_a, vec_b)
        sim_ba = cosine_similarity(vec_b, vec_a)
        assert abs(sim_ab - sim_ba) < 1e-10, f"Similarity not symmetric: {sim_ab} != {sim_ba}"

    @given(vector_strategy, vector_strategy)
    @settings(max_examples=100)
    def test_cosine_similarity_bounded(self, vec_a: list[float], vec_b: list[float]) -> None:
        """Cosine similarity is bounded between -1 and 1."""
        # Vectors must be same length
        min_len = min(len(vec_a), len(vec_b))
        assume(min_len > 0)
        vec_a = vec_a[:min_len]
        vec_b = vec_b[:min_len]

        # Skip zero vectors
        assume(any(v != 0.0 for v in vec_a))
        assume(any(v != 0.0 for v in vec_b))

        similarity = cosine_similarity(vec_a, vec_b)
        assert -1.0 <= similarity <= 1.0, f"Similarity out of bounds: {similarity}"

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_text_overlap_self_is_one(self, text: str) -> None:
        """Text overlap similarity of a text with itself is 1.0."""
        assume(len(text.split()) > 0)  # Need at least one word
        similarity = text_overlap_similarity(text, text)
        assert similarity == 1.0, f"Self-similarity should be 1.0, got {similarity}"

    @given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_text_overlap_symmetric(self, text_a: str, text_b: str) -> None:
        """Text overlap similarity is symmetric."""
        sim_ab = text_overlap_similarity(text_a, text_b)
        sim_ba = text_overlap_similarity(text_b, text_a)
        assert sim_ab == sim_ba, f"Similarity not symmetric: {sim_ab} != {sim_ba}"

    @given(st.text(min_size=0, max_size=100), st.text(min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_text_overlap_bounded(self, text_a: str, text_b: str) -> None:
        """Text overlap similarity is bounded between 0 and 1."""
        similarity = text_overlap_similarity(text_a, text_b)
        assert 0.0 <= similarity <= 1.0, f"Similarity out of bounds: {similarity}"


# ============================================================================
# ClaimVerification Properties
# ============================================================================


class TestClaimVerificationProperties:
    """Property-based tests for ClaimVerification."""

    @given(claim_id_strategy, claim_text_strategy, score_strategy, score_strategy, score_strategy)
    @settings(max_examples=100)
    def test_combined_score_bounded(
        self,
        claim_id: str,
        claim_text: str,
        evidence_support: float,
        evidence_dependence: float,
        consistency_score: float,
    ) -> None:
        """Combined score is bounded between 0 and 1."""
        claim = ClaimVerification(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_support=evidence_support,
            evidence_dependence=evidence_dependence,
            consistency_score=consistency_score,
        )
        assert 0.0 <= claim.combined_score <= 1.0, (
            f"Combined score out of bounds: {claim.combined_score}"
        )

    @given(claim_id_strategy, claim_text_strategy, score_strategy, score_strategy)
    @settings(max_examples=100)
    def test_combined_score_monotonic_support(
        self,
        claim_id: str,
        claim_text: str,
        evidence_support: float,
        evidence_dependence: float,
    ) -> None:
        """Higher evidence support leads to higher or equal combined score."""
        claim_low = ClaimVerification(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_support=evidence_support * 0.5,
            evidence_dependence=evidence_dependence,
        )
        claim_high = ClaimVerification(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_support=evidence_support,
            evidence_dependence=evidence_dependence,
        )
        assert claim_high.combined_score >= claim_low.combined_score

    @given(claim_id_strategy, claim_text_strategy, score_strategy, score_strategy)
    @settings(max_examples=100)
    def test_combined_score_monotonic_dependence(
        self,
        claim_id: str,
        claim_text: str,
        evidence_support: float,
        evidence_dependence: float,
    ) -> None:
        """Higher evidence dependence leads to higher or equal combined score."""
        claim_low = ClaimVerification(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_support=evidence_support,
            evidence_dependence=evidence_dependence * 0.5,
        )
        claim_high = ClaimVerification(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_support=evidence_support,
            evidence_dependence=evidence_dependence,
        )
        assert claim_high.combined_score >= claim_low.combined_score

    @given(
        claim_id_strategy,
        claim_text_strategy,
        st.floats(min_value=0.0, max_value=0.49),  # Low score
        st.floats(min_value=0.0, max_value=0.49),  # Low score
    )
    @settings(max_examples=50)
    def test_needs_attention_for_low_scores(
        self,
        claim_id: str,
        claim_text: str,
        evidence_support: float,
        evidence_dependence: float,
    ) -> None:
        """Claims with low combined scores need attention."""
        claim = ClaimVerification(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_support=evidence_support,
            evidence_dependence=evidence_dependence,
        )
        # Low scores should trigger needs_attention
        if claim.combined_score < 0.5:
            assert claim.needs_attention


# ============================================================================
# EpistemicGap Properties
# ============================================================================


class TestEpistemicGapProperties:
    """Property-based tests for EpistemicGap."""

    @given(claim_id_strategy, claim_text_strategy, positive_float)
    @settings(max_examples=100)
    def test_severity_based_on_gap_bits(
        self,
        claim_id: str,
        claim_text: str,
        gap_bits: float,
    ) -> None:
        """Severity increases with gap_bits for non-critical types."""
        gap = EpistemicGap(
            claim_id=claim_id,
            claim_text=claim_text,
            gap_type="partial_support",
            gap_bits=gap_bits,
        )
        severity = gap.severity
        assert severity in ("low", "medium", "high", "critical")

        # Higher gap_bits should lead to higher severity
        if gap_bits > 3.0:
            assert severity in ("high", "critical")
        elif gap_bits > 1.5:
            assert severity in ("medium", "high", "critical")

    @given(claim_id_strategy, claim_text_strategy)
    @settings(max_examples=50)
    def test_critical_gap_types_are_critical(
        self,
        claim_id: str,
        claim_text: str,
    ) -> None:
        """Phantom citations and contradictions are always critical."""
        for gap_type in ("phantom_citation", "contradicted"):
            gap = EpistemicGap(
                claim_id=claim_id,
                claim_text=claim_text,
                gap_type=gap_type,  # type: ignore
                gap_bits=0.0,  # Even with zero bits, should be critical
            )
            assert gap.severity == "critical"


# ============================================================================
# HallucinationReport Properties
# ============================================================================


class TestHallucinationReportProperties:
    """Property-based tests for HallucinationReport."""

    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_empty_report_verification_rate(self, response_id: str) -> None:
        """Empty report has 100% verification rate."""
        report = HallucinationReport(response_id=response_id)
        assert report.verification_rate == 1.0

    @given(
        st.text(min_size=1, max_size=20),
        st.lists(
            st.tuples(
                claim_id_strategy,
                claim_text_strategy,
                st.booleans(),  # is_flagged
            ),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_verification_rate_bounded(
        self,
        response_id: str,
        claims_data: list[tuple[str, str, bool]],
    ) -> None:
        """Verification rate is bounded between 0 and 1."""
        report = HallucinationReport(response_id=response_id)
        for claim_id, claim_text, is_flagged in claims_data:
            claim = ClaimVerification(
                claim_id=claim_id,
                claim_text=claim_text,
                is_flagged=is_flagged,
            )
            report.add_claim(claim)

        assert 0.0 <= report.verification_rate <= 1.0

    @given(
        st.text(min_size=1, max_size=20),
        st.lists(
            st.tuples(claim_id_strategy, claim_text_strategy),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_claim_counts_consistent(
        self,
        response_id: str,
        claims_data: list[tuple[str, str]],
    ) -> None:
        """Total claims equals verified + flagged."""
        report = HallucinationReport(response_id=response_id)
        for i, (claim_id, claim_text) in enumerate(claims_data):
            claim = ClaimVerification(
                claim_id=claim_id,
                claim_text=claim_text,
                is_flagged=(i % 2 == 0),  # Alternate flagging
            )
            report.add_claim(claim)

        assert report.total_claims == report.verified_claims + report.flagged_claims


# ============================================================================
# VerificationConfig Properties
# ============================================================================


class TestVerificationConfigProperties:
    """Property-based tests for VerificationConfig."""

    @given(
        score_strategy,  # support_threshold
        score_strategy,  # dependence_threshold
        score_strategy,  # sample_rate
        st.integers(min_value=0, max_value=10),  # claim_index
        st.booleans(),  # is_critical
    )
    @settings(max_examples=100)
    def test_should_verify_critical_claims(
        self,
        support_threshold: float,
        dependence_threshold: float,
        sample_rate: float,
        claim_index: int,
        is_critical: bool,
    ) -> None:
        """Critical claims are always verified (except when disabled)."""
        config = VerificationConfig(
            enabled=True,
            support_threshold=support_threshold,
            dependence_threshold=dependence_threshold,
            sample_rate=max(0.01, sample_rate),  # Avoid division by zero
            mode="sample",
        )

        if is_critical:
            assert config.should_verify_claim(claim_index, is_critical=True)

    @given(
        st.integers(min_value=0, max_value=100),
        st.booleans(),
    )
    @settings(max_examples=50)
    def test_disabled_config_never_verifies(
        self,
        claim_index: int,
        is_critical: bool,
    ) -> None:
        """Disabled config never verifies claims."""
        config = VerificationConfig(enabled=False)
        assert not config.should_verify_claim(claim_index, is_critical)

    @given(st.integers(min_value=0, max_value=100), st.booleans())
    @settings(max_examples=50)
    def test_full_mode_verifies_all(
        self,
        claim_index: int,
        is_critical: bool,
    ) -> None:
        """Full mode verifies all claims."""
        config = VerificationConfig(enabled=True, mode="full")
        assert config.should_verify_claim(claim_index, is_critical)

    @given(st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_critical_only_mode(self, claim_index: int) -> None:
        """Critical-only mode only verifies critical claims."""
        config = VerificationConfig(enabled=True, mode="critical_only")
        assert config.should_verify_claim(claim_index, is_critical=True)
        assert not config.should_verify_claim(claim_index, is_critical=False)


# ============================================================================
# FeedbackStatistics Properties
# ============================================================================


class TestFeedbackStatisticsProperties:
    """Property-based tests for FeedbackStatistics."""

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=100)
    def test_rates_bounded(
        self,
        correct: int,
        false_positive: int,
        false_negative: int,
        incorrect: int,
    ) -> None:
        """All rates are bounded between 0 and 1."""
        total = correct + false_positive + false_negative + incorrect
        stats = FeedbackStatistics(
            total_feedback=total,
            correct_count=correct,
            false_positive_count=false_positive,
            false_negative_count=false_negative,
            incorrect_count=incorrect,
        )

        assert 0.0 <= stats.accuracy_rate <= 1.0
        assert 0.0 <= stats.false_positive_rate <= 1.0
        assert 0.0 <= stats.false_negative_rate <= 1.0

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_perfect_accuracy(self, total: int) -> None:
        """100% correct gives accuracy of 1.0."""
        stats = FeedbackStatistics(
            total_feedback=total,
            correct_count=total,
            false_positive_count=0,
            false_negative_count=0,
            incorrect_count=0,
        )
        assert stats.accuracy_rate == 1.0
        assert stats.false_positive_rate == 0.0
        assert stats.false_negative_rate == 0.0


# ============================================================================
# SimilarityResult Properties
# ============================================================================


class TestSimilarityResultProperties:
    """Property-based tests for SimilarityResult."""

    @given(score_strategy)
    @settings(max_examples=100)
    def test_valid_score_accepted(self, score: float) -> None:
        """Valid scores are accepted."""
        result = SimilarityResult(score=score, method=SimilarityMethod.EMBEDDING)
        assert result.score == score

    @given(st.floats(min_value=-100, max_value=-0.001) | st.floats(min_value=1.001, max_value=100))
    @settings(max_examples=50)
    def test_invalid_score_rejected(self, score: float) -> None:
        """Invalid scores (outside 0-1) are rejected."""
        with pytest.raises(ValueError):
            SimilarityResult(score=score, method=SimilarityMethod.EMBEDDING)

    @given(score_strategy, score_strategy)
    @settings(max_examples=50)
    def test_optional_scores_accepted(
        self,
        embedding_score: float,
        llm_score: float,
    ) -> None:
        """Valid optional scores are accepted."""
        result = SimilarityResult(
            score=(embedding_score + llm_score) / 2,
            method=SimilarityMethod.ENSEMBLE,
            embedding_score=embedding_score,
            llm_score=llm_score,
        )
        assert result.embedding_score == embedding_score
        assert result.llm_score == llm_score
