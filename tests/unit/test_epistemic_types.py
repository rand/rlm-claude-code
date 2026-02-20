"""
Unit tests for epistemic verification core types.

Implements: SPEC-16.09 Unit tests for epistemic verification core
"""

import pytest

from src.epistemic import (
    ClaimVerification,
    EpistemicGap,
    HallucinationReport,
    VerificationConfig,
)


class TestClaimVerification:
    """Tests for ClaimVerification dataclass."""

    def test_basic_creation(self) -> None:
        """SPEC-16.01: ClaimVerification core type is defined and constructible."""
        claim = ClaimVerification(
            claim_id="c1",
            claim_text="The file contains 5 functions",
            evidence_ids=["e1", "e2"],
        )
        assert claim.claim_id == "c1"
        assert claim.claim_text == "The file contains 5 functions"
        assert claim.evidence_ids == ["e1", "e2"]
        assert claim.evidence_support == 0.5
        assert claim.evidence_dependence == 0.5
        assert not claim.is_flagged

    def test_combined_score_calculation(self) -> None:
        """Test combined score is geometric mean of support and dependence."""
        claim = ClaimVerification(
            claim_id="c1",
            claim_text="Test claim",
            evidence_support=0.9,
            evidence_dependence=0.8,
        )
        # Geometric mean: sqrt(0.9 * 0.8) ≈ 0.8485
        assert 0.848 < claim.combined_score < 0.849

    def test_needs_attention_when_flagged(self) -> None:
        """Test needs_attention returns True when flagged."""
        claim = ClaimVerification(
            claim_id="c1",
            claim_text="Test claim",
            is_flagged=True,
            flag_reason="unsupported",
        )
        assert claim.needs_attention

    def test_needs_attention_low_score(self) -> None:
        """Test needs_attention returns True when combined score low."""
        claim = ClaimVerification(
            claim_id="c1",
            claim_text="Test claim",
            evidence_support=0.2,
            evidence_dependence=0.2,
        )
        assert claim.combined_score < 0.5
        assert claim.needs_attention

    def test_validation_rejects_invalid_scores(self) -> None:
        """Test that invalid score values are rejected."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ClaimVerification(
                claim_id="c1",
                claim_text="Test",
                evidence_support=1.5,
            )

    def test_validation_rejects_negative_gap_bits(self) -> None:
        """Test that negative gap bits are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ClaimVerification(
                claim_id="c1",
                claim_text="Test",
                evidence_gap_bits=-1.0,
            )


class TestEpistemicGap:
    """Tests for EpistemicGap dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic gap creation."""
        gap = EpistemicGap(
            claim_id="c1",
            claim_text="Unsupported claim",
            gap_type="unsupported",
            gap_bits=2.5,
            suggested_action="Provide evidence or remove claim",
        )
        assert gap.claim_id == "c1"
        assert gap.gap_type == "unsupported"
        assert gap.gap_bits == 2.5

    def test_severity_critical_for_phantom(self) -> None:
        """Test phantom citations are critical severity."""
        gap = EpistemicGap(
            claim_id="c1",
            claim_text="Test",
            gap_type="phantom_citation",
        )
        assert gap.severity == "critical"

    def test_severity_critical_for_contradicted(self) -> None:
        """Test contradicted claims are critical severity."""
        gap = EpistemicGap(
            claim_id="c1",
            claim_text="Test",
            gap_type="contradicted",
        )
        assert gap.severity == "critical"

    def test_severity_high_for_unsupported(self) -> None:
        """Test unsupported claims are high severity."""
        gap = EpistemicGap(
            claim_id="c1",
            claim_text="Test",
            gap_type="unsupported",
        )
        assert gap.severity == "high"

    def test_severity_high_for_large_gap(self) -> None:
        """Test large gap bits result in high severity."""
        gap = EpistemicGap(
            claim_id="c1",
            claim_text="Test",
            gap_type="partial_support",
            gap_bits=4.0,
        )
        assert gap.severity == "high"

    def test_severity_medium_for_partial(self) -> None:
        """Test partial support is medium severity."""
        gap = EpistemicGap(
            claim_id="c1",
            claim_text="Test",
            gap_type="partial_support",
            gap_bits=1.0,
        )
        assert gap.severity == "medium"

    def test_severity_low_for_small_issues(self) -> None:
        """Test small evidence independent issues are low severity."""
        gap = EpistemicGap(
            claim_id="c1",
            claim_text="Test",
            gap_type="evidence_independent",
            gap_bits=0.5,
        )
        assert gap.severity == "low"

    def test_validation_rejects_negative_gap_bits(self) -> None:
        """Test that negative gap bits are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            EpistemicGap(
                claim_id="c1",
                claim_text="Test",
                gap_type="unsupported",
                gap_bits=-1.0,
            )


class TestHallucinationReport:
    """Tests for HallucinationReport dataclass."""

    def test_empty_report(self) -> None:
        """Test empty report has correct defaults."""
        report = HallucinationReport(response_id="r1")
        assert report.total_claims == 0
        assert report.verified_claims == 0
        assert report.flagged_claims == 0
        assert report.verification_rate == 1.0
        assert not report.has_critical_gaps

    def test_add_claim_updates_metrics(self) -> None:
        """Test adding claims updates summary metrics."""
        report = HallucinationReport(response_id="r1")
        claim = ClaimVerification(
            claim_id="c1",
            claim_text="Test",
            evidence_support=0.9,
            evidence_dependence=0.8,
        )
        report.add_claim(claim)

        assert report.total_claims == 1
        assert report.verified_claims == 1
        assert report.flagged_claims == 0

    def test_add_flagged_claim(self) -> None:
        """Test adding flagged claims updates flagged count."""
        report = HallucinationReport(response_id="r1")
        claim = ClaimVerification(
            claim_id="c1",
            claim_text="Test",
            is_flagged=True,
            flag_reason="unsupported",
        )
        report.add_claim(claim)

        assert report.total_claims == 1
        assert report.verified_claims == 0
        assert report.flagged_claims == 1

    def test_add_gap_updates_metrics(self) -> None:
        """Test adding gaps updates gap metrics."""
        report = HallucinationReport(response_id="r1")
        gap = EpistemicGap(
            claim_id="c1",
            claim_text="Test",
            gap_type="phantom_citation",
            gap_bits=3.0,
        )
        report.add_gap(gap)

        assert report.max_gap_bits == 3.0
        assert report.has_critical_gaps

    def test_verification_rate(self) -> None:
        """Test verification rate calculation."""
        report = HallucinationReport(response_id="r1")

        # Add 2 verified, 1 flagged
        report.add_claim(ClaimVerification(claim_id="c1", claim_text="Test1"))
        report.add_claim(ClaimVerification(claim_id="c2", claim_text="Test2"))
        report.add_claim(
            ClaimVerification(
                claim_id="c3",
                claim_text="Test3",
                is_flagged=True,
                flag_reason="unsupported",
            )
        )

        assert report.verification_rate == pytest.approx(2 / 3)

    def test_critical_gaps_filter(self) -> None:
        """Test critical_gaps property filters correctly."""
        report = HallucinationReport(response_id="r1")
        report.add_gap(EpistemicGap(claim_id="c1", claim_text="T1", gap_type="phantom_citation"))
        report.add_gap(EpistemicGap(claim_id="c2", claim_text="T2", gap_type="partial_support"))
        report.add_gap(EpistemicGap(claim_id="c3", claim_text="T3", gap_type="contradicted"))

        critical = report.critical_gaps
        assert len(critical) == 2
        assert all(g.severity == "critical" for g in critical)

    def test_flagged_claim_texts(self) -> None:
        """Test flagged_claim_texts extracts correct texts."""
        report = HallucinationReport(response_id="r1")
        report.add_claim(ClaimVerification(claim_id="c1", claim_text="Good claim"))
        report.add_claim(
            ClaimVerification(
                claim_id="c2",
                claim_text="Bad claim",
                is_flagged=True,
                flag_reason="unsupported",
            )
        )

        assert report.flagged_claim_texts == ["Bad claim"]


class TestVerificationConfig:
    """Tests for VerificationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VerificationConfig()
        assert config.enabled
        assert config.support_threshold == 0.7
        assert config.dependence_threshold == 0.3
        assert config.on_failure == "retry"
        assert config.verification_model == "haiku"
        assert config.mode == "sample"

    def test_should_verify_full_mode(self) -> None:
        """Test full mode verifies all claims."""
        config = VerificationConfig(mode="full")
        assert config.should_verify_claim(0, is_critical=False)
        assert config.should_verify_claim(5, is_critical=False)
        assert config.should_verify_claim(10, is_critical=True)

    def test_should_verify_critical_only_mode(self) -> None:
        """SPEC-16.31: critical_only mode verifies only critical claims."""
        config = VerificationConfig(mode="critical_only")
        assert not config.should_verify_claim(0, is_critical=False)
        assert config.should_verify_claim(0, is_critical=True)

    def test_should_verify_sample_mode(self) -> None:
        """Test sample mode verifies critical + sampled claims."""
        config = VerificationConfig(mode="sample", sample_rate=0.5)
        # Critical always verified
        assert config.should_verify_claim(0, is_critical=True)
        assert config.should_verify_claim(1, is_critical=True)
        # Non-critical sampled (50% = every 2nd)
        assert config.should_verify_claim(0, is_critical=False)
        assert not config.should_verify_claim(1, is_critical=False)
        assert config.should_verify_claim(2, is_critical=False)

    def test_should_verify_disabled(self) -> None:
        """Test disabled config verifies nothing."""
        config = VerificationConfig(enabled=False)
        assert not config.should_verify_claim(0, is_critical=True)
        assert not config.should_verify_claim(0, is_critical=False)

    def test_get_model_for_claim(self) -> None:
        """Test model selection based on criticality."""
        config = VerificationConfig(
            verification_model="haiku",
            critical_model="sonnet",
        )
        assert config.get_model_for_claim(is_critical=False) == "haiku"
        assert config.get_model_for_claim(is_critical=True) == "sonnet"

    def test_validation_rejects_invalid_thresholds(self) -> None:
        """Test that invalid threshold values are rejected."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            VerificationConfig(support_threshold=1.5)

    def test_validation_rejects_negative_gap_threshold(self) -> None:
        """Test that negative gap threshold is rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            VerificationConfig(gap_threshold_bits=-1.0)

    def test_validation_rejects_invalid_max_retries(self) -> None:
        """Test that negative max_retries is rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            VerificationConfig(max_retries=-1)

    def test_validation_rejects_zero_max_claims(self) -> None:
        """Test that zero max_claims_per_response is rejected."""
        with pytest.raises(ValueError, match="must be at least 1"):
            VerificationConfig(max_claims_per_response=0)
