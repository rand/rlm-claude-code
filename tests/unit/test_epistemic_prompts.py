"""
Tests for epistemic prompts module.

Implements: SPEC-16.33 Optimized prompt templates tests
"""

import pytest

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


class TestPromptTemplate:
    """Tests for PromptTemplate enum."""

    def test_all_templates_exist(self) -> None:
        """All required templates should exist."""
        required = [
            PromptTemplate.CLAIM_EXTRACTION,
            PromptTemplate.DIRECT_VERIFICATION,
            PromptTemplate.PHANTOM_CHECK,
            PromptTemplate.EVIDENCE_MAPPING,
            PromptTemplate.CONSISTENCY_CHECK,
            PromptTemplate.LLM_JUDGE_SIMILARITY,
        ]
        for template in required:
            assert template in PROMPTS

    def test_templates_have_string_values(self) -> None:
        """Templates should have string values."""
        for template in PromptTemplate:
            assert isinstance(template.value, str)


class TestPrompt:
    """Tests for Prompt dataclass."""

    def test_prompt_has_required_fields(self) -> None:
        """Prompt should have system and user_template fields."""
        prompt = Prompt(
            system="Test system",
            user_template="Test template {var}",
        )
        assert prompt.system == "Test system"
        assert prompt.user_template == "Test template {var}"

    def test_prompt_defaults(self) -> None:
        """Prompt should have sensible defaults."""
        prompt = Prompt(system="sys", user_template="user")
        assert prompt.max_tokens == 1024
        assert prompt.temperature == 0.0


class TestGetPrompt:
    """Tests for get_prompt function."""

    def test_get_claim_extraction_prompt(self) -> None:
        """Should return claim extraction prompt."""
        prompt = get_prompt(PromptTemplate.CLAIM_EXTRACTION)
        assert isinstance(prompt, Prompt)
        assert "claim" in prompt.system.lower()
        assert "{text}" in prompt.user_template

    def test_get_direct_verification_prompt(self) -> None:
        """Should return direct verification prompt."""
        prompt = get_prompt(PromptTemplate.DIRECT_VERIFICATION)
        assert isinstance(prompt, Prompt)
        assert "support" in prompt.system.lower() or "evidence" in prompt.system.lower()
        assert "{claim}" in prompt.user_template
        assert "{evidence}" in prompt.user_template

    def test_get_all_prompts(self) -> None:
        """Should be able to get all registered prompts."""
        for template in PromptTemplate:
            prompt = get_prompt(template)
            assert isinstance(prompt, Prompt)
            assert len(prompt.system) > 0
            assert len(prompt.user_template) > 0


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_format_claim_extraction(self) -> None:
        """Should format claim extraction prompt."""
        system, user = format_prompt(
            PromptTemplate.CLAIM_EXTRACTION,
            text="The function returns 42.",
        )
        assert len(system) > 0
        assert "The function returns 42." in user

    def test_format_direct_verification(self) -> None:
        """Should format direct verification prompt."""
        system, user = format_prompt(
            PromptTemplate.DIRECT_VERIFICATION,
            claim="The API returns JSON",
            evidence="response.json() is called",
        )
        assert len(system) > 0
        assert "The API returns JSON" in user
        assert "response.json() is called" in user

    def test_format_returns_tuple(self) -> None:
        """Should return tuple of (system, user)."""
        result = format_prompt(
            PromptTemplate.CLAIM_EXTRACTION,
            text="test",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_format_missing_variable_raises(self) -> None:
        """Should raise KeyError for missing variables."""
        with pytest.raises(KeyError):
            format_prompt(PromptTemplate.CLAIM_EXTRACTION)  # Missing 'text'


class TestEstimatePromptTokens:
    """Tests for estimate_prompt_tokens function."""

    def test_estimate_simple_prompt(self) -> None:
        """Should estimate tokens for simple prompt."""
        tokens = estimate_prompt_tokens(
            PromptTemplate.CLAIM_EXTRACTION,
            text="Short text",
        )
        assert tokens > 0
        assert tokens < 1000  # Should be relatively small

    def test_estimate_longer_prompt(self) -> None:
        """Longer prompts should have more tokens."""
        short_tokens = estimate_prompt_tokens(
            PromptTemplate.CLAIM_EXTRACTION,
            text="Short",
        )
        long_tokens = estimate_prompt_tokens(
            PromptTemplate.CLAIM_EXTRACTION,
            text="x" * 1000,
        )
        assert long_tokens > short_tokens

    def test_estimate_returns_integer(self) -> None:
        """Should return integer token count."""
        tokens = estimate_prompt_tokens(
            PromptTemplate.CLAIM_EXTRACTION,
            text="test",
        )
        assert isinstance(tokens, int)


class TestTruncateEvidence:
    """Tests for truncate_evidence function."""

    def test_short_evidence_unchanged(self) -> None:
        """Short evidence should not be truncated."""
        evidence = "Short evidence text"
        result = truncate_evidence(evidence, max_chars=1000)
        assert result == evidence

    def test_long_evidence_truncated(self) -> None:
        """Long evidence should be truncated."""
        evidence = "x" * 5000
        result = truncate_evidence(evidence, max_chars=1000)
        assert len(result) <= 1000
        assert "truncated" in result.lower() or "..." in result

    def test_preserves_start_and_end(self) -> None:
        """Should preserve start and end of evidence."""
        evidence = "START" + "x" * 5000 + "END"
        result = truncate_evidence(
            evidence,
            max_chars=100,
            preserve_start=10,
            preserve_end=10,
        )
        assert result.startswith("START")
        assert result.endswith("END")

    def test_exact_max_chars_not_truncated(self) -> None:
        """Evidence exactly at max_chars should not be truncated."""
        evidence = "x" * 100
        result = truncate_evidence(evidence, max_chars=100)
        assert result == evidence
        assert "truncated" not in result.lower()

    def test_custom_preserve_sizes(self) -> None:
        """Should respect custom preserve sizes."""
        evidence = "A" * 100 + "B" * 100 + "C" * 100
        result = truncate_evidence(
            evidence,
            max_chars=150,
            preserve_start=50,
            preserve_end=50,
        )
        assert len(result) <= 150


class TestFormatClaimsCompact:
    """Tests for format_claims_compact function."""

    def test_format_empty_claims(self) -> None:
        """Empty claims list should return empty string."""
        result = format_claims_compact([])
        assert result == ""

    def test_format_single_claim(self) -> None:
        """Single claim should be formatted correctly."""
        claims = [{"claim": "The function returns 42"}]
        result = format_claims_compact(claims)
        assert "0:" in result
        assert "The function returns 42" in result

    def test_format_multiple_claims(self) -> None:
        """Multiple claims should be numbered."""
        claims = [
            {"claim": "First claim"},
            {"claim": "Second claim"},
            {"claim": "Third claim"},
        ]
        result = format_claims_compact(claims)
        assert "0: First claim" in result
        assert "1: Second claim" in result
        assert "2: Third claim" in result

    def test_format_handles_claim_text_key(self) -> None:
        """Should handle 'claim_text' key as fallback."""
        claims = [{"claim_text": "The API returns JSON"}]
        result = format_claims_compact(claims)
        assert "The API returns JSON" in result


class TestFormatEvidenceCompact:
    """Tests for format_evidence_compact function."""

    def test_format_empty_evidence(self) -> None:
        """Empty evidence should return empty string."""
        result = format_evidence_compact({})
        assert result == ""

    def test_format_single_evidence(self) -> None:
        """Single evidence should be formatted with ID."""
        evidence = {"e1": "Evidence content here"}
        result = format_evidence_compact(evidence)
        assert "[e1]:" in result
        assert "Evidence content here" in result

    def test_format_multiple_evidence(self) -> None:
        """Multiple evidence items should all appear."""
        evidence = {
            "e1": "First evidence",
            "e2": "Second evidence",
        }
        result = format_evidence_compact(evidence)
        assert "[e1]:" in result
        assert "[e2]:" in result

    def test_truncates_long_evidence(self) -> None:
        """Long evidence should be truncated."""
        evidence = {"e1": "x" * 1000}
        result = format_evidence_compact(evidence, max_per_item=100)
        assert len(result) < 1000
        assert "..." in result

    def test_removes_newlines(self) -> None:
        """Newlines should be removed for compactness."""
        evidence = {"e1": "Line 1\nLine 2\nLine 3"}
        result = format_evidence_compact(evidence)
        assert "\n" not in result.replace("[e1]: ", "").strip()


class TestPromptQuality:
    """Tests for prompt quality and completeness."""

    def test_all_prompts_request_json_output(self) -> None:
        """All prompts should request JSON output format."""
        for template in PromptTemplate:
            prompt = get_prompt(template)
            # Check that JSON formatting is requested
            combined = prompt.system + prompt.user_template
            assert "{" in combined, f"{template.value} should have JSON format example"

    def test_prompts_are_concise(self) -> None:
        """Prompts should be reasonably concise."""
        for template in PromptTemplate:
            prompt = get_prompt(template)
            # System prompts should be under 500 chars for efficiency
            assert len(prompt.system) < 500, f"{template.value} system prompt too long"

    def test_prompts_have_reasonable_max_tokens(self) -> None:
        """Prompts should have reasonable max_tokens settings."""
        for template in PromptTemplate:
            prompt = get_prompt(template)
            assert prompt.max_tokens > 0
            assert prompt.max_tokens <= 4096  # Reasonable upper bound
