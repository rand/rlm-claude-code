"""
Core types for epistemic verification (hallucination detection).

Implements: SPEC-16.01 Define epistemic verification core types

These types support consistency-based verification for Claude API,
which lacks logprobs. Instead of measuring confidence shifts directly,
we compare answers with/without evidence to detect hallucinations.

References:
- SPEC-16: Epistemic Verification specification
- Pythea/Strawberry: https://github.com/leochlon/pythea
"""

from dataclasses import dataclass, field
from typing import Literal

# Gap types representing different kinds of epistemic failures
GapType = Literal[
    "unsupported",  # No evidence supports claim
    "partial_support",  # Evidence supports part of claim
    "phantom_citation",  # Cited source doesn't exist
    "contradicted",  # Evidence contradicts claim
    "over_extrapolation",  # Claim goes beyond evidence
    "evidence_independent",  # Answer unchanged without evidence
]

# Flag reasons for claim verification
FlagReason = Literal[
    "unsupported",  # Claim lacks supporting evidence
    "phantom_citation",  # Cited evidence doesn't exist
    "low_dependence",  # Answer unchanged without evidence
    "contradiction",  # Evidence contradicts claim
    "over_extrapolation",  # Claim extrapolates beyond evidence
    "confidence_mismatch",  # Stated confidence not justified
]

# Actions on verification failure
OnFailureAction = Literal["flag", "retry", "ask"]

# Verification modes
VerificationMode = Literal["full", "sample", "critical_only"]


@dataclass
class ClaimVerification:
    """
    Result of verifying a single claim against evidence.

    Implements: SPEC-16.01 ClaimVerification type

    The verification process compares the claim against cited evidence
    using both direct verification and consistency checking (evidence
    scrubbing). This Claude-compatible approach doesn't require logprobs.

    Attributes:
        claim_id: Unique identifier for this claim
        claim_text: The textual content of the claim
        evidence_ids: IDs of evidence sources cited by this claim

        evidence_support: How well evidence supports claim (0.0-1.0)
            - 1.0 = evidence explicitly states what claim asserts
            - 0.5 = partial support
            - 0.0 = no support or contradiction

        evidence_dependence: How much answer changed without evidence (0.0-1.0)
            - 1.0 = answer completely different without evidence (good)
            - 0.0 = answer identical without evidence (potential hallucination)

        consistency_score: Semantic consistency across variations (0.0-1.0)
            - Higher values indicate stable, consistent reasoning

        confidence_justified: Whether stated confidence is supported by evidence
        evidence_gap_bits: Information gap in bits (Strawberry-compatible metric)
            - Higher values indicate larger gap between claim and evidence

        is_flagged: Whether this claim requires attention
        flag_reason: Reason for flagging (if flagged)
    """

    claim_id: str
    claim_text: str
    evidence_ids: list[str] = field(default_factory=list)

    # Verification scores
    evidence_support: float = 0.5
    evidence_dependence: float = 0.5
    consistency_score: float = 1.0

    # Computed metrics
    confidence_justified: bool = True
    evidence_gap_bits: float = 0.0

    # Flags
    is_flagged: bool = False
    flag_reason: FlagReason | None = None

    def __post_init__(self) -> None:
        """Validate score ranges."""
        for attr in ("evidence_support", "evidence_dependence", "consistency_score"):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {value}")
        if self.evidence_gap_bits < 0:
            raise ValueError(
                f"evidence_gap_bits must be non-negative, got {self.evidence_gap_bits}"
            )

    @property
    def combined_score(self) -> float:
        """
        Compute combined verification score.

        Combines evidence support and dependence into a single metric.
        Both are important: good evidence support with low dependence
        suggests the answer was derived from prior knowledge, not evidence.
        """
        # Geometric mean gives lower weight to imbalanced scores
        return (self.evidence_support * self.evidence_dependence) ** 0.5

    @property
    def needs_attention(self) -> bool:
        """Check if this claim requires human review."""
        return self.is_flagged or self.combined_score < 0.5


@dataclass
class EpistemicGap:
    """
    An identified gap between claim and evidence.

    Implements: SPEC-16.01 EpistemicGap type

    Represents a specific failure mode in the reasoning chain where
    a claim is not adequately supported by its cited evidence.

    Attributes:
        claim_id: ID of the claim with the gap
        claim_text: Text of the problematic claim
        gap_type: Category of the epistemic gap
        gap_bits: Information gap measured in bits (KL-divergence equivalent)
        suggested_action: Recommended action to resolve the gap
    """

    claim_id: str
    claim_text: str
    gap_type: GapType
    gap_bits: float = 0.0
    suggested_action: str = ""

    def __post_init__(self) -> None:
        """Validate gap_bits is non-negative."""
        if self.gap_bits < 0:
            raise ValueError(f"gap_bits must be non-negative, got {self.gap_bits}")

    @property
    def severity(self) -> Literal["low", "medium", "high", "critical"]:
        """
        Compute severity level based on gap type and size.

        Returns:
            Severity level for prioritizing remediation
        """
        # Critical gaps that completely invalidate the claim
        if self.gap_type in ("phantom_citation", "contradicted"):
            return "critical"

        # High severity for unsupported claims with large gaps
        if self.gap_type == "unsupported" or self.gap_bits > 3.0:
            return "high"

        # Medium severity for partial issues
        if self.gap_type in ("partial_support", "over_extrapolation") or self.gap_bits > 1.5:
            return "medium"

        # Low severity for minor issues
        return "low"


@dataclass
class HallucinationReport:
    """
    Full hallucination detection report for a response.

    Implements: SPEC-16.01 HallucinationReport type

    Aggregates verification results for all claims in a response,
    providing summary metrics and actionable recommendations.

    Attributes:
        response_id: Unique identifier for the verified response
        total_claims: Number of claims extracted from response
        verified_claims: Number of claims that passed verification
        flagged_claims: Number of claims that failed verification

        claims: Detailed verification results for each claim
        gaps: Identified epistemic gaps

        overall_confidence: Weighted average claim confidence
        max_gap_bits: Largest evidence gap in bits
        has_critical_gaps: Whether any gaps exceed threshold

        should_retry: Whether the response should be regenerated
        retry_guidance: Specific guidance for retry attempt
    """

    response_id: str
    total_claims: int = 0
    verified_claims: int = 0
    flagged_claims: int = 0

    claims: list[ClaimVerification] = field(default_factory=list)
    gaps: list[EpistemicGap] = field(default_factory=list)

    # Summary metrics
    overall_confidence: float = 1.0
    max_gap_bits: float = 0.0
    has_critical_gaps: bool = False

    # Recommendations
    should_retry: bool = False
    retry_guidance: str | None = None

    def __post_init__(self) -> None:
        """Compute derived metrics from claims and gaps."""
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Recompute summary metrics from claims and gaps."""
        if self.claims:
            self.total_claims = len(self.claims)
            self.verified_claims = sum(1 for c in self.claims if not c.is_flagged)
            self.flagged_claims = sum(1 for c in self.claims if c.is_flagged)

            # Compute overall confidence as weighted average
            scores = [c.combined_score for c in self.claims]
            self.overall_confidence = sum(scores) / len(scores) if scores else 1.0

        if self.gaps:
            self.max_gap_bits = max(g.gap_bits for g in self.gaps)
            self.has_critical_gaps = any(g.severity == "critical" for g in self.gaps)

    def add_claim(self, claim: ClaimVerification) -> None:
        """Add a verified claim to the report."""
        self.claims.append(claim)
        self._update_metrics()

    def add_gap(self, gap: EpistemicGap) -> None:
        """Add an identified gap to the report."""
        self.gaps.append(gap)
        self._update_metrics()

    @property
    def verification_rate(self) -> float:
        """Percentage of claims that passed verification."""
        if self.total_claims == 0:
            return 1.0
        return self.verified_claims / self.total_claims

    @property
    def critical_gaps(self) -> list[EpistemicGap]:
        """Get only critical-severity gaps."""
        return [g for g in self.gaps if g.severity == "critical"]

    @property
    def flagged_claim_texts(self) -> list[str]:
        """Get text of all flagged claims for retry guidance."""
        return [c.claim_text for c in self.claims if c.is_flagged]


@dataclass
class VerificationConfig:
    """
    Configuration for epistemic verification.

    Implements: SPEC-16.01 VerificationConfig type

    Controls how verification is performed, including thresholds,
    model selection, and behavior on failure.

    Attributes:
        enabled: Whether verification is active
        support_threshold: Minimum evidence support score (0.0-1.0)
        dependence_threshold: Minimum evidence dependence (0.0-1.0)
        gap_threshold_bits: Maximum acceptable gap in bits

        on_failure: Action when verification fails
        max_retries: Maximum retry attempts before escalating

        verification_model: Model for standard verification
        critical_model: Model for critical path verification
        max_claims_per_response: Limit claims verified for cost control
        parallel_verification: Whether to verify claims in parallel

        mode: Verification mode (full/sample/critical_only)
        sample_rate: Fraction of claims to verify in sample mode
    """

    # Enable/disable
    enabled: bool = True

    # Verification thresholds
    support_threshold: float = 0.7
    dependence_threshold: float = 0.3
    gap_threshold_bits: float = 2.0

    # Behavior on failure
    on_failure: OnFailureAction = "retry"
    max_retries: int = 2

    # Model selection
    verification_model: str = "haiku"
    critical_model: str = "sonnet"

    # Performance limits
    max_claims_per_response: int = 10
    parallel_verification: bool = True

    # Modes
    mode: VerificationMode = "sample"
    sample_rate: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        for attr in ("support_threshold", "dependence_threshold", "sample_rate"):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {value}")

        if self.gap_threshold_bits < 0:
            raise ValueError(
                f"gap_threshold_bits must be non-negative, got {self.gap_threshold_bits}"
            )

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")

        if self.max_claims_per_response < 1:
            raise ValueError(
                f"max_claims_per_response must be at least 1, got {self.max_claims_per_response}"
            )

    def should_verify_claim(
        self,
        claim_index: int,
        is_critical: bool,
        claim_text: str | None = None,
    ) -> bool:
        """
        Determine if a claim should be verified based on mode.

        Implements: SPEC-16.28 Sample mode for cost control

        Prioritizes:
        1. Critical claims (always verified)
        2. Claims with uncertainty markers (hedging language)
        3. Random sample of remaining claims

        Args:
            claim_index: Index of the claim (for sampling)
            is_critical: Whether the claim is marked as critical
            claim_text: Optional claim text to check for uncertainty markers

        Returns:
            True if the claim should be verified
        """
        if not self.enabled:
            return False

        if self.mode == "full":
            return True

        if self.mode == "critical_only":
            return is_critical

        # Sample mode: verify critical claims + sample of others
        if is_critical:
            return True

        # SPEC-16.28: Prioritize claims with uncertainty markers
        if claim_text and self._has_uncertainty_markers(claim_text):
            return True

        # Use deterministic sampling based on index for reproducibility
        return (claim_index % int(1 / self.sample_rate)) == 0 if self.sample_rate > 0 else False

    def _has_uncertainty_markers(self, claim_text: str) -> bool:
        """
        Check if claim text contains uncertainty markers.

        Implements: SPEC-16.28

        Looks for hedging language that indicates the claim
        should be prioritized for verification.
        """
        uncertainty_markers = (
            "might",
            "probably",
            "possibly",
            "i think",
            "i believe",
            "likely",
            "unlikely",
            "may ",  # Space to avoid matching "maybe" as prefix
            "could",
            "should",
            "seems",
            "appears",
            "suggests",
            "indicates",
            "approximately",
            "roughly",
            "about ",  # Space to avoid false positives
            "around ",
            "estimate",
        )
        claim_lower = claim_text.lower()
        return any(marker in claim_lower for marker in uncertainty_markers)

    def get_model_for_claim(self, is_critical: bool) -> str:
        """Get the appropriate model for verifying a claim."""
        return self.critical_model if is_critical else self.verification_model
