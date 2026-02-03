"""
Claim extraction for epistemic verification.

Implements: SPEC-16.06 Claim extraction via Claude

Extracts atomic, verifiable claims from LLM responses and maps them
to cited evidence spans. Uses Claude to decompose complex statements
into individually verifiable units.

When rlm_core is available, provides fast pattern-based extraction
via quick_extract_claims() as a complement to LLM-based extraction.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import rlm_core

if TYPE_CHECKING:
    from src.api_client import APIResponse


def quick_hallucination_score(text: str) -> float:
    """
    Quick hallucination risk score using rlm_core.

    Returns a float between 0.0 and 1.0 indicating hallucination risk.

    Args:
        text: Text to check for hallucination risk

    Returns:
        Risk score (0.0 = low risk, 1.0 = high risk)
    """
    return rlm_core.quick_hallucination_check(text)


def _map_core_category_to_metadata(category: Any) -> dict[str, str]:
    """Map rlm_core.ClaimCategory to metadata dict."""
    # Convert category to string for comparison (enum may not be hashable)
    category_str = str(category)

    # Map by string representation
    if "CodeBehavior" in category_str:
        return {"category": "code_behavior"}
    elif "Factual" in category_str:
        return {"category": "factual"}
    elif "Numerical" in category_str:
        return {"category": "numerical"}
    elif "Temporal" in category_str:
        return {"category": "temporal"}
    elif "Relational" in category_str:
        return {"category": "relational"}
    elif "UserIntent" in category_str:
        return {"category": "user_intent"}
    elif "MetaReasoning" in category_str:
        return {"category": "meta_reasoning"}
    else:
        return {"category": "unknown"}


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
class ExtractedClaim:
    """
    An atomic claim extracted from a response.

    Represents a single verifiable statement with its cited evidence.

    Attributes:
        claim_id: Unique identifier for this claim
        claim_text: The atomic claim text
        original_span: The original text span this claim was extracted from
        evidence_ids: IDs of evidence sources this claim cites
        confidence: Extraction confidence (how certain we are this is a real claim)
        is_critical: Whether this claim is critical to the response
        metadata: Additional extraction metadata
    """

    claim_id: str
    claim_text: str
    original_span: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    is_critical: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """
    Result of claim extraction from a response.

    Attributes:
        claims: List of extracted claims
        response_id: ID of the original response
        total_spans: Number of text spans analyzed
        extraction_model: Model used for extraction
    """

    claims: list[ExtractedClaim]
    response_id: str
    total_spans: int = 0
    extraction_model: str = "haiku"


# ============================================================================
# Quick Extraction via rlm_core
# ============================================================================


def quick_extract_claims(
    text: str,
    response_id: str | None = None,
) -> ExtractionResult:
    """
    Fast pattern-based claim extraction using rlm_core.

    This is a quick, non-LLM extraction that uses pattern matching
    and heuristics. Use for pre-filtering or when LLM extraction
    is too slow/expensive.

    Args:
        text: Text to extract claims from
        response_id: Optional ID for the response

    Returns:
        ExtractionResult with extracted claims
    """
    response_id = response_id or str(uuid.uuid4())[:8]

    extractor = rlm_core.ClaimExtractor()
    core_claims = extractor.extract(text)

    claims = []
    for i, cc in enumerate(core_claims):
        claims.append(
            ExtractedClaim(
                claim_id=cc.id if hasattr(cc, "id") else f"{response_id}-c{i}",
                claim_text=cc.text,
                original_span=cc.text,
                evidence_ids=[],
                confidence=cc.specificity if hasattr(cc, "specificity") else 0.8,
                is_critical=False,
                metadata=_map_core_category_to_metadata(cc.category),
            )
        )

    return ExtractionResult(
        claims=claims,
        response_id=response_id,
        total_spans=len(claims),
        extraction_model="rlm_core",
    )


# Prompts for claim extraction
CLAIM_EXTRACTION_SYSTEM = """You are an expert at decomposing text into atomic, verifiable claims.

Your task is to extract individual factual claims from a given text. Each claim should be:
1. ATOMIC: Contains a single assertion that can be independently verified
2. COMPLETE: Self-contained and understandable without additional context
3. OBJECTIVE: States facts, not opinions (unless quoting someone's opinion as a fact)

For each claim, also identify:
- Whether it cites specific evidence (file names, line numbers, quotes, etc.)
- Whether it's critical to the main point of the response
- Your confidence that this is a real, extractable claim (0.0-1.0)

Respond in JSON format with an array of claims."""

CLAIM_EXTRACTION_PROMPT = """Extract atomic claims from this text:

<text>
{text}
</text>

For each claim, provide:
- "claim": The atomic claim text
- "original_span": The original text this came from
- "cites_evidence": List of evidence references (file names, quotes, etc.)
- "is_critical": Whether this is critical to the main point
- "confidence": Your confidence this is a valid claim (0.0-1.0)

Respond with a JSON array. Example:
[
  {{
    "claim": "The function calculate_total returns the sum of all items",
    "original_span": "The calculate_total function...",
    "cites_evidence": ["src/utils.py:42"],
    "is_critical": true,
    "confidence": 0.95
  }}
]

If the text contains no verifiable claims, respond with an empty array: []"""

EVIDENCE_MAPPING_SYSTEM = """You are an expert at identifying which pieces of evidence support which claims.

Given a list of claims and available evidence sources, determine which evidence sources
each claim relies on or cites."""

EVIDENCE_MAPPING_PROMPT = """Map these claims to their supporting evidence:

<claims>
{claims_json}
</claims>

<available_evidence>
{evidence_json}
</available_evidence>

For each claim, identify which evidence_ids it relies on. A claim relies on evidence if:
1. It directly quotes or paraphrases the evidence
2. It makes assertions that can only be verified by that evidence
3. It references specific details found in that evidence

Respond with a JSON object mapping claim indices to evidence IDs:
{{
  "0": ["e1", "e3"],
  "1": ["e2"],
  "2": []
}}

Use empty arrays for claims that don't rely on any provided evidence."""


class ClaimExtractor:
    """
    Extracts atomic claims from LLM responses.

    Uses Claude to decompose complex responses into individually
    verifiable claims that can be checked against evidence.

    Example:
        >>> extractor = ClaimExtractor(client)
        >>> result = await extractor.extract_claims(
        ...     "The file has 5 functions. The main() function starts at line 10."
        ... )
        >>> len(result.claims)
        2
    """

    def __init__(
        self,
        client: LLMClient,
        default_model: str = "haiku",
        max_claims: int = 20,
    ):
        """
        Initialize the claim extractor.

        Args:
            client: LLM client for API calls
            default_model: Model to use for extraction
            max_claims: Maximum claims to extract per response
        """
        self.client = client
        self.default_model = default_model
        self.max_claims = max_claims

    @property
    def usesrlm_core(self) -> bool:
        """Return True if rlm_core is available for quick extraction."""
        return True

    def quick_extract(
        self,
        text: str,
        response_id: str | None = None,
    ) -> ExtractionResult:
        """
        Fast pattern-based extraction using rlm_core.

        This is a convenience method that wraps quick_extract_claims().
        Use for pre-filtering before expensive LLM extraction.

        Args:
            text: Text to extract claims from
            response_id: Optional ID for the response

        Returns:
            ExtractionResult with extracted claims
        """
        return quick_extract_claims(text, response_id)

    async def extract_claims(
        self,
        text: str,
        model: str | None = None,
        response_id: str | None = None,
    ) -> ExtractionResult:
        """
        Extract atomic claims from text.

        Args:
            text: The text to extract claims from
            model: Model to use (defaults to self.default_model)
            response_id: ID to assign to the response

        Returns:
            ExtractionResult with extracted claims
        """
        model = model or self.default_model
        response_id = response_id or str(uuid.uuid4())[:8]

        # Handle empty or trivial text
        if not text or len(text.strip()) < 10:
            return ExtractionResult(
                claims=[],
                response_id=response_id,
                total_spans=0,
                extraction_model=model,
            )

        # Call LLM to extract claims
        prompt = CLAIM_EXTRACTION_PROMPT.format(text=text)
        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=CLAIM_EXTRACTION_SYSTEM,
            model=model,
            max_tokens=2048,
            temperature=0.0,
        )

        # Parse response
        claims = self._parse_extraction_response(response.content, response_id)

        # Limit claims
        if len(claims) > self.max_claims:
            # Keep critical claims and highest confidence others
            critical = [c for c in claims if c.is_critical]
            non_critical = [c for c in claims if not c.is_critical]
            non_critical.sort(key=lambda c: c.confidence, reverse=True)

            remaining_slots = self.max_claims - len(critical)
            claims = critical + non_critical[:remaining_slots]

        return ExtractionResult(
            claims=claims,
            response_id=response_id,
            total_spans=len(claims),
            extraction_model=model,
        )

    async def map_claims_to_evidence(
        self,
        claims: list[ExtractedClaim],
        evidence: dict[str, str],
        model: str | None = None,
    ) -> list[ExtractedClaim]:
        """
        Map extracted claims to evidence sources.

        Args:
            claims: Claims to map
            evidence: Dict mapping evidence IDs to content
            model: Model to use for mapping

        Returns:
            Claims with updated evidence_ids
        """
        if not claims or not evidence:
            return claims

        model = model or self.default_model

        # Format claims for prompt
        claims_data = [{"index": i, "claim": c.claim_text} for i, c in enumerate(claims)]

        # Format evidence for prompt (truncate long evidence)
        evidence_data = {
            eid: content[:500] + "..." if len(content) > 500 else content
            for eid, content in evidence.items()
        }

        prompt = EVIDENCE_MAPPING_PROMPT.format(
            claims_json=json.dumps(claims_data, indent=2),
            evidence_json=json.dumps(evidence_data, indent=2),
        )

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=EVIDENCE_MAPPING_SYSTEM,
            model=model,
            max_tokens=1024,
            temperature=0.0,
        )

        # Parse mapping response
        mapping = self._parse_mapping_response(response.content)

        # Update claims with evidence IDs
        for idx_str, evidence_ids in mapping.items():
            try:
                idx = int(idx_str)
                if 0 <= idx < len(claims):
                    # Merge with existing evidence IDs
                    existing = set(claims[idx].evidence_ids)
                    existing.update(evidence_ids)
                    claims[idx].evidence_ids = list(existing)
            except (ValueError, IndexError):
                continue

        return claims

    def _parse_extraction_response(
        self,
        content: str,
        response_id: str,
    ) -> list[ExtractedClaim]:
        """Parse the JSON response from claim extraction."""
        claims: list[ExtractedClaim] = []

        # Try to extract JSON from response
        json_match = re.search(r"\[[\s\S]*\]", content)
        if not json_match:
            return claims

        try:
            data = json.loads(json_match.group())
            if not isinstance(data, list):
                return claims

            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue

                claim_text = item.get("claim", "").strip()
                if not claim_text:
                    continue

                # Extract evidence references from cites_evidence
                evidence_refs = item.get("cites_evidence", [])
                if isinstance(evidence_refs, str):
                    evidence_refs = [evidence_refs]

                claims.append(
                    ExtractedClaim(
                        claim_id=f"{response_id}-c{i}",
                        claim_text=claim_text,
                        original_span=item.get("original_span", ""),
                        evidence_ids=evidence_refs,
                        confidence=float(item.get("confidence", 0.8)),
                        is_critical=bool(item.get("is_critical", False)),
                    )
                )

        except (json.JSONDecodeError, TypeError, ValueError):
            # If parsing fails, try to extract claims heuristically
            pass

        return claims

    def _parse_mapping_response(self, content: str) -> dict[str, list[str]]:
        """Parse the JSON response from evidence mapping."""
        # Try to extract JSON object from response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return {}

        try:
            data = json.loads(json_match.group())
            if not isinstance(data, dict):
                return {}

            # Validate and normalize the mapping
            result: dict[str, list[str]] = {}
            for key, value in data.items():
                if isinstance(value, list):
                    result[str(key)] = [str(v) for v in value]
                elif isinstance(value, str):
                    result[str(key)] = [value]

            return result

        except (json.JSONDecodeError, TypeError, ValueError):
            return {}


def extract_evidence_references(text: str) -> list[str]:
    """
    Extract potential evidence references from text.

    Looks for patterns like:
    - File paths: src/foo.py, ./bar.js
    - Line references: line 42, L42, :42
    - Quote markers: "...", '...'
    - Code references: `foo()`, function_name

    Args:
        text: Text to extract references from

    Returns:
        List of potential evidence reference strings
    """
    references: list[str] = []

    # File paths
    file_pattern = r"[a-zA-Z0-9_\-./]+\.[a-zA-Z]{1,4}"
    references.extend(re.findall(file_pattern, text))

    # Line references
    line_pattern = r"(?:line\s*|L|:)(\d+)"
    for match in re.finditer(line_pattern, text, re.IGNORECASE):
        references.append(f"line:{match.group(1)}")

    # Inline code references
    code_pattern = r"`([^`]+)`"
    references.extend(re.findall(code_pattern, text))

    return list(set(references))
