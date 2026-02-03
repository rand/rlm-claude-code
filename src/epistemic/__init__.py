"""
Epistemic verification module for hallucination detection.

Implements: SPEC-16 Epistemic Verification

This module provides consistency-based verification for Claude API,
detecting procedural hallucinations where the model "knows but doesn't use"
information correctly.

Example:
    >>> from src.epistemic import ClaimVerification, HallucinationReport
    >>> claim = ClaimVerification(
    ...     claim_id="c1",
    ...     claim_text="The file contains 5 functions",
    ...     evidence_support=0.9,
    ...     evidence_dependence=0.8,
    ... )
    >>> claim.combined_score
    0.8485281374238571
"""

from src.epistemic.claim_extractor import (
    ClaimExtractor,
    ExtractedClaim,
    ExtractionResult,
    extract_evidence_references,
    quick_extract_claims,
    quick_hallucination_score,
)
from src.epistemic.consistency_checker import (
    ConsistencyChecker,
    ConsistencyConfig,
    ConsistencyResult,
    compute_consistency_score,
    evidence_dependence_from_similarity,
)
from src.epistemic.evidence_auditor import (
    AuditResult,
    BatchAuditResult,
    EvidenceAuditor,
    compute_evidence_support,
)
from src.epistemic.prompts import (
    PROMPTS,
    Prompt,
    PromptTemplate,
    estimate_prompt_tokens,
    format_claims_compact,
    format_evidence_compact,
    format_prompt,
    get_prompt,
    truncate_evidence,
)
from src.epistemic.similarity import (
    EmbeddingSimilarity,
    LLMJudgeSimilarity,
    SemanticSimilarity,
    SimilarityConfig,
    SimilarityMethod,
    SimilarityResult,
    cosine_similarity,
    text_overlap_similarity,
)
from src.epistemic.types import (
    ClaimVerification,
    EpistemicGap,
    FlagReason,
    GapType,
    HallucinationReport,
    OnFailureAction,
    VerificationConfig,
    VerificationMode,
)
from src.epistemic.verification_cache import (
    CachedVerification,
    VerificationCache,
    create_cached_auditor,
)
from src.epistemic.verification_feedback import (
    FeedbackStatistics,
    FeedbackStore,
    FeedbackType,
    VerificationFeedback,
    record_feedback,
)

__all__ = [
    # Core types
    "ClaimVerification",
    "EpistemicGap",
    "HallucinationReport",
    "VerificationConfig",
    # Claim extraction
    "ClaimExtractor",
    "ExtractedClaim",
    "ExtractionResult",
    "extract_evidence_references",
    "quick_extract_claims",
    "quick_hallucination_score",
    # Evidence auditing
    "EvidenceAuditor",
    "AuditResult",
    "BatchAuditResult",
    "compute_evidence_support",
    # Semantic similarity
    "SemanticSimilarity",
    "EmbeddingSimilarity",
    "LLMJudgeSimilarity",
    "SimilarityResult",
    "SimilarityConfig",
    "SimilarityMethod",
    "cosine_similarity",
    "text_overlap_similarity",
    # Consistency checking
    "ConsistencyChecker",
    "ConsistencyConfig",
    "ConsistencyResult",
    "compute_consistency_score",
    "evidence_dependence_from_similarity",
    # Type aliases
    "GapType",
    "FlagReason",
    "OnFailureAction",
    "VerificationMode",
    # User feedback (SPEC-16.37)
    "FeedbackType",
    "VerificationFeedback",
    "FeedbackStatistics",
    "FeedbackStore",
    "record_feedback",
    # Verification cache (SPEC-16.32)
    "CachedVerification",
    "VerificationCache",
    "create_cached_auditor",
    # Prompt templates (SPEC-16.33)
    "Prompt",
    "PromptTemplate",
    "PROMPTS",
    "get_prompt",
    "format_prompt",
    "estimate_prompt_tokens",
    "truncate_evidence",
    "format_claims_compact",
    "format_evidence_compact",
]
