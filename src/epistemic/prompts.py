"""
Optimized prompt templates for epistemic verification.

Implements: SPEC-16.33 Optimized prompt templates

Centralizes all prompts for claim extraction, evidence auditing,
and verification. Optimized for:
- Token efficiency
- Clear instruction structure
- Consistent JSON output format
- Few-shot examples where helpful
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PromptTemplate(str, Enum):
    """Identifiers for prompt templates."""

    CLAIM_EXTRACTION = "claim_extraction"
    DIRECT_VERIFICATION = "direct_verification"
    PHANTOM_CHECK = "phantom_check"
    EVIDENCE_MAPPING = "evidence_mapping"
    CONSISTENCY_CHECK = "consistency_check"
    LLM_JUDGE_SIMILARITY = "llm_judge_similarity"


@dataclass
class Prompt:
    """A prompt with system and user components."""

    system: str
    user_template: str
    max_tokens: int = 1024
    temperature: float = 0.0


# ============================================================================
# Claim Extraction Prompts (SPEC-16.06)
# ============================================================================

CLAIM_EXTRACTION = Prompt(
    system="""Extract atomic, verifiable claims from text.

Rules:
- ATOMIC: One assertion per claim
- COMPLETE: Self-contained
- OBJECTIVE: Facts only

Output JSON array of claims.""",
    user_template="""Extract claims from:

<text>
{text}
</text>

Format:
[{{"claim": "...", "original_span": "...", "cites_evidence": ["file:line"], "is_critical": bool, "confidence": 0.0-1.0}}]

Empty array if no claims: []""",
    max_tokens=2048,
)

# ============================================================================
# Direct Verification Prompts (SPEC-16.07)
# ============================================================================

DIRECT_VERIFICATION = Prompt(
    system="""Evaluate if claims are supported by evidence. Be precise and conservative.""",
    user_template="""Claim: {claim}

Evidence:
{evidence}

Rate support and identify issues:

{{"support_score": 0.0-1.0, "issues": ["unsupported"|"partial"|"contradiction"|"extrapolation"], "reasoning": "..."}}""",
    max_tokens=512,
)

# ============================================================================
# Phantom Citation Check Prompts (SPEC-16.07)
# ============================================================================

PHANTOM_CHECK = Prompt(
    system="""Verify cited references exist in available sources.""",
    user_template="""Citations: {citations}

Sources: {evidence_sources}

{{"results": [{{"citation": "...", "exists": bool, "matched_source": "id"|null}}]}}""",
    max_tokens=512,
)

# ============================================================================
# Evidence Mapping Prompts (SPEC-16.06)
# ============================================================================

EVIDENCE_MAPPING = Prompt(
    system="""Map claims to supporting evidence sources.""",
    user_template="""Claims:
{claims_json}

Evidence:
{evidence_json}

Map claim indices to evidence IDs:
{{"0": ["e1"], "1": ["e2", "e3"], "2": []}}""",
    max_tokens=1024,
)

# ============================================================================
# Consistency Check Prompts (SPEC-16.08)
# ============================================================================

CONSISTENCY_CHECK = Prompt(
    system="""Check if rephrased claim is semantically equivalent to original.""",
    user_template="""Original: {original}
Rephrased: {rephrased}

{{"equivalent": bool, "similarity": 0.0-1.0, "differences": ["..."]}}""",
    max_tokens=256,
)

# ============================================================================
# LLM Judge Similarity Prompts (SPEC-16.10)
# ============================================================================

LLM_JUDGE_SIMILARITY = Prompt(
    system="""Rate semantic similarity between two texts. Focus on meaning, not wording.""",
    user_template="""Text A: {text_a}
Text B: {text_b}

{{"similarity": 0.0-1.0, "reasoning": "..."}}""",
    max_tokens=256,
)


# ============================================================================
# Prompt Registry
# ============================================================================

PROMPTS: dict[PromptTemplate, Prompt] = {
    PromptTemplate.CLAIM_EXTRACTION: CLAIM_EXTRACTION,
    PromptTemplate.DIRECT_VERIFICATION: DIRECT_VERIFICATION,
    PromptTemplate.PHANTOM_CHECK: PHANTOM_CHECK,
    PromptTemplate.EVIDENCE_MAPPING: EVIDENCE_MAPPING,
    PromptTemplate.CONSISTENCY_CHECK: CONSISTENCY_CHECK,
    PromptTemplate.LLM_JUDGE_SIMILARITY: LLM_JUDGE_SIMILARITY,
}


def get_prompt(template: PromptTemplate) -> Prompt:
    """
    Get a prompt template by identifier.

    Args:
        template: The prompt template identifier

    Returns:
        The Prompt object with system and user template
    """
    return PROMPTS[template]


def format_prompt(
    template: PromptTemplate,
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Format a prompt template with variables.

    Args:
        template: The prompt template identifier
        **kwargs: Variables to substitute in the template

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    prompt = get_prompt(template)
    user_prompt = prompt.user_template.format(**kwargs)
    return prompt.system, user_prompt


def estimate_prompt_tokens(
    template: PromptTemplate,
    **kwargs: Any,
) -> int:
    """
    Estimate token count for a formatted prompt.

    Uses simple heuristic: ~4 chars per token.

    Args:
        template: The prompt template identifier
        **kwargs: Variables to substitute

    Returns:
        Estimated token count
    """
    system, user = format_prompt(template, **kwargs)
    total_chars = len(system) + len(user)
    return total_chars // 4


# ============================================================================
# Prompt Optimization Utilities
# ============================================================================


def truncate_evidence(
    evidence: str,
    max_chars: int = 2000,
    preserve_start: int = 500,
    preserve_end: int = 500,
) -> str:
    """
    Truncate evidence while preserving important parts.

    Keeps the start and end of evidence, which often contain
    the most relevant information (function signatures, conclusions).

    Args:
        evidence: Evidence text to truncate
        max_chars: Maximum characters to return
        preserve_start: Characters to preserve from start
        preserve_end: Characters to preserve from end

    Returns:
        Truncated evidence with ellipsis marker if truncated
    """
    if len(evidence) <= max_chars:
        return evidence

    if preserve_start + preserve_end >= max_chars:
        # Just truncate from the end
        return evidence[: max_chars - 3] + "..."

    middle_marker = "\n[...truncated...]\n"
    available = max_chars - len(middle_marker)
    start_chars = min(preserve_start, available // 2)
    end_chars = min(preserve_end, available - start_chars)

    return evidence[:start_chars] + middle_marker + evidence[-end_chars:]


def format_claims_compact(claims: list[dict[str, Any]]) -> str:
    """
    Format claims list in compact form for prompts.

    Args:
        claims: List of claim dictionaries

    Returns:
        Compact string representation
    """
    lines = []
    for i, claim in enumerate(claims):
        text = claim.get("claim", claim.get("claim_text", ""))
        lines.append(f"{i}: {text}")
    return "\n".join(lines)


def format_evidence_compact(evidence: dict[str, str], max_per_item: int = 300) -> str:
    """
    Format evidence dict in compact form for prompts.

    Args:
        evidence: Dict mapping IDs to content
        max_per_item: Max characters per evidence item

    Returns:
        Compact string representation
    """
    lines = []
    for eid, content in evidence.items():
        truncated = content[:max_per_item] + "..." if len(content) > max_per_item else content
        # Remove newlines for compactness
        truncated = " ".join(truncated.split())
        lines.append(f"[{eid}]: {truncated}")
    return "\n".join(lines)


__all__ = [
    "Prompt",
    "PromptTemplate",
    "PROMPTS",
    "get_prompt",
    "format_prompt",
    "estimate_prompt_tokens",
    "truncate_evidence",
    "format_claims_compact",
    "format_evidence_compact",
    # Individual prompts for direct access
    "CLAIM_EXTRACTION",
    "DIRECT_VERIFICATION",
    "PHANTOM_CHECK",
    "EVIDENCE_MAPPING",
    "CONSISTENCY_CHECK",
    "LLM_JUDGE_SIMILARITY",
]
