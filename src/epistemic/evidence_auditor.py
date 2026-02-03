"""
Evidence auditor for epistemic verification.

Implements: SPEC-16.07 Evidence auditor

Verifies claims against their cited evidence using direct LLM verification.
Detects phantom citations, unsupported claims, and over-extrapolations.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from src.epistemic.claim_extractor import ExtractedClaim
from src.epistemic.types import ClaimVerification, EpistemicGap, FlagReason, GapType

if TYPE_CHECKING:
    from src.api_client import APIResponse


class LLMClient(Protocol):
    """Protocol for LLM client to enable dependency injection."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> APIResponse: ...


@dataclass
class AuditResult:
    """
    Result of auditing a claim against evidence.

    Attributes:
        verification: The ClaimVerification result
        gaps: Any epistemic gaps identified
        audit_model: Model used for auditing
    """

    verification: ClaimVerification
    gaps: list[EpistemicGap] = field(default_factory=list)
    audit_model: str = "haiku"


@dataclass
class BatchAuditResult:
    """
    Result of auditing multiple claims.

    Attributes:
        results: Individual audit results
        total_claims: Number of claims audited
        flagged_count: Number of claims flagged
        phantom_count: Number of phantom citations found
    """

    results: list[AuditResult]
    total_claims: int = 0
    flagged_count: int = 0
    phantom_count: int = 0


# Prompts for evidence auditing
DIRECT_VERIFICATION_SYSTEM = """You are an expert at evaluating whether claims are supported by evidence.

Your task is to determine how well a given claim is supported by the provided evidence.
Be precise and conservative - only rate claims as well-supported if the evidence explicitly
states or strongly implies what the claim asserts."""

DIRECT_VERIFICATION_PROMPT = """Evaluate whether this claim is supported by the evidence.

CLAIM: {claim}

EVIDENCE:
{evidence}

Rate the support level and identify any issues:

1. support_score: 0.0 (completely unsupported) to 1.0 (fully supported)
2. issues: List any problems found:
   - "unsupported" - evidence doesn't support this claim
   - "partial" - evidence only partially supports the claim
   - "contradiction" - evidence contradicts the claim
   - "extrapolation" - claim goes beyond what evidence states
3. reasoning: Brief explanation of your assessment

Respond in JSON format:
{{
  "support_score": 0.85,
  "issues": [],
  "reasoning": "The evidence explicitly states..."
}}"""

PHANTOM_CHECK_SYSTEM = (
    """You are verifying whether cited references actually exist in the provided context."""
)

PHANTOM_CHECK_PROMPT = """Check if these cited references exist in the available evidence.

CITED REFERENCES:
{citations}

AVAILABLE EVIDENCE SOURCES:
{evidence_sources}

For each citation, determine if it exists in the available sources.

Respond in JSON format:
{{
  "results": [
    {{"citation": "src/foo.py:42", "exists": true, "matched_source": "e1"}},
    {{"citation": "nonexistent.py", "exists": false, "matched_source": null}}
  ]
}}"""


class EvidenceAuditor:
    """
    Audits claims against their cited evidence.

    Uses LLM-based verification to check if claims are supported by
    their cited evidence. Detects phantom citations, unsupported claims,
    contradictions, and over-extrapolations.

    Example:
        >>> auditor = EvidenceAuditor(client)
        >>> result = await auditor.audit_claim(
        ...     claim=ExtractedClaim(claim_id="c1", claim_text="X is Y"),
        ...     evidence={"e1": "X is definitely Y"},
        ... )
        >>> result.verification.evidence_support
        0.95
    """

    def __init__(
        self,
        client: LLMClient,
        default_model: str = "haiku",
        critical_model: str = "sonnet",
        support_threshold: float = 0.7,
    ):
        """
        Initialize the evidence auditor.

        Args:
            client: LLM client for API calls
            default_model: Model for standard verification
            critical_model: Model for critical path verification
            support_threshold: Threshold below which claims are flagged
        """
        self.client = client
        self.default_model = default_model
        self.critical_model = critical_model
        self.support_threshold = support_threshold

    async def audit_claim(
        self,
        claim: ExtractedClaim,
        evidence: dict[str, str],
        is_critical: bool = False,
    ) -> AuditResult:
        """
        Audit a single claim against available evidence.

        Args:
            claim: The claim to audit
            evidence: Dict mapping evidence IDs to content
            is_critical: Whether to use critical model

        Returns:
            AuditResult with verification and any gaps
        """
        model = self.critical_model if is_critical else self.default_model
        gaps: list[EpistemicGap] = []

        # Check for phantom citations first
        phantom_ids = self._check_phantom_citations(claim.evidence_ids, evidence)
        if phantom_ids:
            for phantom_id in phantom_ids:
                gaps.append(
                    EpistemicGap(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        gap_type="phantom_citation",
                        gap_bits=float("inf"),  # Phantom citations are critical
                        suggested_action=f"Remove or correct citation: {phantom_id}",
                    )
                )

        # Get relevant evidence for this claim
        relevant_evidence = self._get_relevant_evidence(claim.evidence_ids, evidence)

        # If no valid evidence, flag as unsupported
        if not relevant_evidence:
            verification = ClaimVerification(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                evidence_ids=claim.evidence_ids,
                evidence_support=0.0,
                evidence_dependence=0.0,
                is_flagged=True,
                flag_reason="unsupported",
            )
            gaps.append(
                EpistemicGap(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    gap_type="unsupported",
                    gap_bits=3.0,  # High gap for unsupported claims
                    suggested_action="Provide supporting evidence or remove claim",
                )
            )
            return AuditResult(verification=verification, gaps=gaps, audit_model=model)

        # Perform direct verification
        support_score, issues, reasoning = await self._verify_direct(
            claim.claim_text, relevant_evidence, model
        )

        # Process issues into gaps
        for issue in issues:
            gap_type = self._issue_to_gap_type(issue)
            if gap_type:
                gaps.append(
                    EpistemicGap(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        gap_type=gap_type,
                        gap_bits=self._compute_gap_bits(support_score),
                        suggested_action=self._get_suggested_action(gap_type),
                    )
                )

        # Determine if claim should be flagged
        is_flagged = support_score < self.support_threshold or len(gaps) > 0
        flag_reason = self._determine_flag_reason(support_score, issues, gaps)

        verification = ClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            evidence_ids=claim.evidence_ids,
            evidence_support=support_score,
            evidence_dependence=0.5,  # Will be set by consistency checker
            consistency_score=1.0,  # Will be set by consistency checker
            confidence_justified=support_score >= self.support_threshold,
            evidence_gap_bits=self._compute_gap_bits(support_score),
            is_flagged=is_flagged,
            flag_reason=flag_reason,
        )

        return AuditResult(verification=verification, gaps=gaps, audit_model=model)

    async def audit_claims(
        self,
        claims: list[ExtractedClaim],
        evidence: dict[str, str],
        critical_claim_ids: set[str] | None = None,
        parallel: bool = True,
    ) -> BatchAuditResult:
        """
        Audit multiple claims against available evidence.

        Implements: SPEC-16.26 Parallel claim verification

        Args:
            claims: Claims to audit
            evidence: Dict mapping evidence IDs to content
            critical_claim_ids: IDs of claims to use critical model for
            parallel: Whether to audit claims in parallel (default True)

        Returns:
            BatchAuditResult with all verification results
        """
        import asyncio

        critical_ids = critical_claim_ids or set()

        if parallel and len(claims) > 1:
            # SPEC-16.26: Parallel verification using asyncio.gather
            tasks = [
                self.audit_claim(
                    claim,
                    evidence,
                    is_critical=claim.claim_id in critical_ids or claim.is_critical,
                )
                for claim in claims
            ]
            results = await asyncio.gather(*tasks)
        else:
            # Sequential verification
            results = []
            for claim in claims:
                is_critical = claim.claim_id in critical_ids or claim.is_critical
                result = await self.audit_claim(claim, evidence, is_critical)
                results.append(result)

        # Aggregate results
        flagged_count = sum(1 for r in results if r.verification.is_flagged)
        phantom_count = sum(1 for r in results for g in r.gaps if g.gap_type == "phantom_citation")

        return BatchAuditResult(
            results=list(results),
            total_claims=len(claims),
            flagged_count=flagged_count,
            phantom_count=phantom_count,
        )

    async def _verify_direct(
        self,
        claim_text: str,
        evidence: str,
        model: str,
    ) -> tuple[float, list[str], str]:
        """
        Perform direct verification via LLM.

        Returns:
            Tuple of (support_score, issues, reasoning)
        """
        prompt = DIRECT_VERIFICATION_PROMPT.format(
            claim=claim_text,
            evidence=evidence,
        )

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=DIRECT_VERIFICATION_SYSTEM,
            model=model,
            max_tokens=512,
            temperature=0.0,
        )

        return self._parse_verification_response(response.content)

    def _parse_verification_response(self, content: str) -> tuple[float, list[str], str]:
        """Parse the JSON response from verification."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            # Default to moderate support if parsing fails
            return 0.5, [], "Unable to parse verification response"

        try:
            data = json.loads(json_match.group())

            support_score = float(data.get("support_score", 0.5))
            # Clamp to valid range
            support_score = max(0.0, min(1.0, support_score))

            issues = data.get("issues", [])
            if isinstance(issues, str):
                issues = [issues] if issues else []

            reasoning = str(data.get("reasoning", ""))

            return support_score, issues, reasoning

        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.5, [], "Unable to parse verification response"

    def _check_phantom_citations(
        self,
        citation_ids: list[str],
        evidence: dict[str, str],
    ) -> list[str]:
        """
        Check for phantom citations (citations that don't exist).

        Returns list of citation IDs that don't exist in evidence.
        """
        phantoms: list[str] = []
        evidence_keys = set(evidence.keys())

        for cid in citation_ids:
            # Direct match
            if cid in evidence_keys:
                continue

            # Check if citation refers to content within evidence
            # (e.g., "src/foo.py:42" might be within evidence["e1"])
            found = False
            for content in evidence.values():
                if cid in content:
                    found = True
                    break

            if not found:
                phantoms.append(cid)

        return phantoms

    def _get_relevant_evidence(
        self,
        evidence_ids: list[str],
        evidence: dict[str, str],
    ) -> str:
        """Get concatenated evidence for the given IDs."""
        parts: list[str] = []

        for eid in evidence_ids:
            if eid in evidence:
                parts.append(f"[{eid}]:\n{evidence[eid]}")

        # If no direct matches, use all evidence
        if not parts and evidence:
            for eid, content in evidence.items():
                parts.append(f"[{eid}]:\n{content}")

        return "\n\n".join(parts)

    def _issue_to_gap_type(self, issue: str) -> GapType | None:
        """Convert issue string to GapType."""
        issue_lower = issue.lower()

        if "unsupported" in issue_lower:
            return "unsupported"
        if "partial" in issue_lower:
            return "partial_support"
        if "contradict" in issue_lower:
            return "contradicted"
        if "extrapolat" in issue_lower:
            return "over_extrapolation"

        return None

    def _compute_gap_bits(self, support_score: float) -> float:
        """
        Compute information gap in bits.

        Uses negative log2 of support to get bits of "missing information".
        """
        if support_score <= 0:
            return 10.0  # Cap at 10 bits for zero support
        if support_score >= 1:
            return 0.0

        # -log2(support) gives bits of uncertainty
        return -math.log2(support_score)

    def _determine_flag_reason(
        self,
        support_score: float,
        issues: list[str],
        gaps: list[EpistemicGap],
    ) -> FlagReason | None:
        """Determine the primary reason for flagging a claim."""
        # Check for phantom citations first (most critical)
        if any(g.gap_type == "phantom_citation" for g in gaps):
            return "phantom_citation"

        # Check for contradictions
        if any(g.gap_type == "contradicted" for g in gaps):
            return "contradiction"

        # Check for over-extrapolation
        if any(g.gap_type == "over_extrapolation" for g in gaps):
            return "over_extrapolation"

        # Check for unsupported
        if support_score < self.support_threshold:
            return "unsupported"

        return None

    def _get_suggested_action(self, gap_type: GapType) -> str:
        """Get suggested action for a gap type."""
        actions = {
            "unsupported": "Provide supporting evidence or remove claim",
            "partial_support": "Qualify claim to match available evidence",
            "phantom_citation": "Remove or correct the citation",
            "contradicted": "Reconcile claim with contradicting evidence",
            "over_extrapolation": "Reduce claim scope to match evidence",
            "evidence_independent": "Verify claim uses provided evidence",
        }
        return actions.get(gap_type, "Review claim for accuracy")


def compute_evidence_support(
    _claim: str,
    _evidence: str,
    verification_result: dict,
) -> float:
    """
    Compute evidence support score from verification result.

    This is a helper function for manual verification result processing.

    Args:
        claim: The claim text
        evidence: The evidence text
        verification_result: Dict with support_score and issues

    Returns:
        Support score between 0.0 and 1.0
    """
    base_score = float(verification_result.get("support_score", 0.5))
    issues = verification_result.get("issues", [])

    # Penalize for issues
    penalty = len(issues) * 0.1
    adjusted_score = max(0.0, base_score - penalty)

    return adjusted_score
