"""
End-to-end integration tests for epistemic verification.

Implements: SPEC-16.30 End-to-end integration tests

Tests the full verification pipeline:
- Claim extraction from responses
- Evidence auditing and verification
- Caching and feedback
- Rich output generation
- Trajectory integration
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest

from src.epistemic import (
    AuditResult,
    ClaimExtractor,
    EvidenceAuditor,
    ExtractedClaim,
    FeedbackStore,
    FeedbackType,
    PromptTemplate,
    VerificationCache,
    VerificationConfig,
    VerificationMode,
    format_prompt,
)
from src.epistemic.types import OnFailureAction
from src.rich_output import OutputConfig, RLMConsole
from src.trajectory import TrajectoryEvent, TrajectoryEventType, VerificationPayload
from src.trajectory_analysis import TrajectoryAnalyzer

# ============================================================================
# Mock API Client for Testing
# ============================================================================


@dataclass
class MockAPIResponse:
    """Mock API response for testing."""

    content: str
    input_tokens: int = 100
    output_tokens: int = 50


class MockLLMClient:
    """Mock LLM client for integration testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        """
        Initialize with predefined responses.

        Args:
            responses: Dict mapping prompts (substrings) to responses
        """
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> MockAPIResponse:
        """Mock complete method."""
        user_content = messages[0]["content"] if messages else ""

        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "model": model,
                "max_tokens": max_tokens,
            }
        )

        # Find matching response
        for key, response in self.responses.items():
            if key in user_content or (system and key in system):
                return MockAPIResponse(content=response)

        # Default response based on prompt type
        if "Extract" in user_content or "claim" in user_content.lower():
            return MockAPIResponse(
                content="""[
                    {"claim": "The function returns 42", "original_span": "returns 42", "cites_evidence": ["src/main.py:10"], "is_critical": true, "confidence": 0.9}
                ]"""
            )
        elif "Evaluate" in user_content or "support" in user_content.lower():
            return MockAPIResponse(
                content="""{"support_score": 0.85, "issues": [], "reasoning": "Evidence supports the claim"}"""
            )
        elif "Map" in user_content:
            return MockAPIResponse(content="""{"0": ["e1"]}""")
        else:
            return MockAPIResponse(content="""{"result": "ok"}""")


# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================


class TestFullVerificationPipeline:
    """Tests for the complete verification pipeline."""

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def extractor(self, mock_client: MockLLMClient) -> ClaimExtractor:
        """Create claim extractor with mock client."""
        return ClaimExtractor(client=mock_client)

    @pytest.fixture
    def auditor(self, mock_client: MockLLMClient) -> EvidenceAuditor:
        """Create evidence auditor with mock client."""
        return EvidenceAuditor(client=mock_client)

    @pytest.mark.asyncio
    async def test_extract_and_audit_pipeline(
        self,
        extractor: ClaimExtractor,
        auditor: EvidenceAuditor,
    ) -> None:
        """Test full extract -> audit pipeline."""
        # Step 1: Extract claims from response
        response_text = """
        The function calculate_total() in src/main.py:10 returns 42.
        The API endpoint /users returns JSON data.
        """

        extraction_result = await extractor.extract_claims(response_text)

        assert len(extraction_result.claims) > 0
        assert extraction_result.response_id is not None

        # Step 2: Audit claims against evidence
        evidence = {
            "e1": "def calculate_total(): return 42",
            "e2": "@app.get('/users') def get_users(): return jsonify(users)",
        }

        audit_result = await auditor.audit_claims(
            claims=extraction_result.claims,
            evidence=evidence,
        )

        assert audit_result.total_claims > 0
        assert isinstance(audit_result.results[0], AuditResult)

    @pytest.mark.asyncio
    async def test_pipeline_with_flagged_claims(
        self,
        extractor: ClaimExtractor,
        mock_client: MockLLMClient,
    ) -> None:
        """Test pipeline identifies flagged claims."""
        # Configure mock to return low support
        mock_client.responses["Evaluate"] = (
            """{"support_score": 0.3, "issues": ["unsupported"], "reasoning": "No evidence"}"""
        )

        auditor = EvidenceAuditor(client=mock_client, support_threshold=0.7)

        claims = [
            ExtractedClaim(
                claim_id="c1",
                claim_text="The database uses PostgreSQL",
                evidence_ids=["e1"],
            )
        ]

        evidence = {"e1": "config = SQLiteDatabase()"}

        result = await auditor.audit_claims(claims, evidence)

        assert result.flagged_count > 0
        assert result.results[0].verification.is_flagged

    @pytest.mark.asyncio
    async def test_pipeline_with_phantom_citations(
        self,
        auditor: EvidenceAuditor,
    ) -> None:
        """Test pipeline detects phantom citations."""
        claims = [
            ExtractedClaim(
                claim_id="c1",
                claim_text="The config is in settings.py",
                evidence_ids=["settings.py"],  # Doesn't exist in evidence
            )
        ]

        evidence = {"e1": "config.json content here"}

        result = await auditor.audit_claims(claims, evidence)

        # Should have phantom citation gap
        has_phantom = any(g.gap_type == "phantom_citation" for r in result.results for g in r.gaps)
        assert has_phantom or result.results[0].verification.is_flagged


class TestCachingIntegration:
    """Tests for verification caching integration."""

    @pytest.fixture
    def cache(self) -> VerificationCache:
        """Create temporary verification cache."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cache = VerificationCache(path, ttl_hours=1)
        yield cache
        os.unlink(path)

    @pytest.fixture
    def feedback_store(self) -> FeedbackStore:
        """Create temporary feedback store."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = FeedbackStore(path)
        yield store
        os.unlink(path)

    def test_cache_integration_round_trip(self, cache: VerificationCache) -> None:
        """Test full cache round trip."""
        claim = "The function returns an integer"
        evidence = "def foo() -> int: return 42"

        # Store verification result
        cache_key = cache.put(
            claim_text=claim,
            evidence=evidence,
            support_score=0.95,
            issues=[],
            reasoning="Evidence explicitly shows int return type",
        )

        # Retrieve and verify
        result = cache.get(claim, evidence)

        assert result is not None
        assert result.support_score == 0.95
        assert result.reasoning == "Evidence explicitly shows int return type"

    def test_feedback_integration_round_trip(
        self,
        feedback_store: FeedbackStore,
    ) -> None:
        """Test full feedback round trip."""
        # Record feedback
        feedback_id = feedback_store.add_feedback(
            claim_id="c1",
            claim_text="The API returns JSON",
            feedback_type=FeedbackType.CORRECT,
            original_flag_reason=None,
            user_note="Verified manually",
        )

        # Get statistics
        stats = feedback_store.get_statistics()

        assert stats.total_feedback == 1
        assert stats.correct_count == 1

    def test_cache_and_feedback_combined(
        self,
        cache: VerificationCache,
        feedback_store: FeedbackStore,
    ) -> None:
        """Test cache and feedback work together."""
        claim = "The function is pure"
        evidence = "def pure_fn(x): return x * 2"

        # First verification - goes to cache
        cache_key = cache.put(
            claim_text=claim,
            evidence=evidence,
            support_score=0.9,
            issues=[],
            reasoning="No side effects observed",
        )

        # User provides feedback
        feedback_store.add_feedback(
            claim_id="c1",
            claim_text=claim,
            feedback_type=FeedbackType.CORRECT,
            original_flag_reason=None,
            user_note="Confirmed correct",
        )

        # Cache hit on same claim
        cached = cache.get(claim, evidence)
        assert cached is not None
        assert cached.hit_count >= 0

        # Feedback stats updated
        stats = feedback_store.get_statistics()
        assert stats.total_feedback == 1


class TestTrajectoryIntegration:
    """Tests for trajectory analysis integration."""

    def test_verification_event_in_trajectory(self) -> None:
        """Test verification events appear in trajectory analysis."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.RLM_START,
                depth=0,
                content="Starting RLM",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.VERIFICATION,
                depth=0,
                content="Verified 8/10 claims",
                typed_payload=VerificationPayload(
                    claims_total=10,
                    claims_verified=8,
                    claims_flagged=2,
                    confidence=0.85,
                ),
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                depth=0,
                content="Done",
            ),
        ]

        analyzer = TrajectoryAnalyzer()
        analysis = analyzer.analyze(events)

        assert analysis.metrics.verification_count == 1
        assert analysis.metrics.verified_claims == 8
        assert analysis.metrics.flagged_claims == 2
        assert analysis.metrics.avg_verification_confidence == 0.85

    def test_trajectory_to_dict_includes_verification(self) -> None:
        """Test trajectory dict includes verification metrics."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.VERIFICATION,
                depth=0,
                content="Verified",
                typed_payload=VerificationPayload(
                    claims_total=5,
                    claims_verified=5,
                    claims_flagged=0,
                    confidence=0.95,
                ),
            ),
        ]

        analyzer = TrajectoryAnalyzer()
        analysis = analyzer.analyze(events)
        d = analysis.to_dict()

        assert "verification_count" in d["metrics"]
        assert d["metrics"]["verification_count"] == 1
        assert d["metrics"]["verified_claims"] == 5


class TestRichOutputIntegration:
    """Tests for rich output integration."""

    def test_verification_report_renders(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test verification report renders correctly."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)

        console.emit_verification_report(
            claims_total=10,
            claims_verified=8,
            claims_flagged=2,
            confidence=0.80,
            flagged_claims=[
                ("c1", "The API returns XML", "unsupported"),
                ("c2", "The database is MySQL", "contradicted"),
            ],
        )

        captured = capsys.readouterr()
        assert "Verification Report" in captured.out
        assert "8" in captured.out  # verified count
        assert "unsupported" in captured.out

    def test_verification_inline_renders(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test inline verification renders correctly."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)

        console.emit_verification(
            claims_total=5,
            claims_verified=5,
            claims_flagged=0,
            confidence=0.95,
        )

        captured = capsys.readouterr()
        assert "Verified" in captured.out
        assert "5/5" in captured.out
        assert "95%" in captured.out


class TestPromptIntegration:
    """Tests for prompt template integration."""

    def test_format_claim_extraction_prompt(self) -> None:
        """Test claim extraction prompt formats correctly."""
        system, user = format_prompt(
            PromptTemplate.CLAIM_EXTRACTION,
            text="The function calculate_total returns 42.",
        )

        assert len(system) > 0
        assert "calculate_total" in user
        assert "42" in user

    def test_format_verification_prompt(self) -> None:
        """Test verification prompt formats correctly."""
        system, user = format_prompt(
            PromptTemplate.DIRECT_VERIFICATION,
            claim="The API returns JSON",
            evidence="return jsonify(data)",
        )

        assert len(system) > 0
        assert "API returns JSON" in user
        assert "jsonify" in user


class TestConfigurationIntegration:
    """Tests for verification configuration integration."""

    def test_verification_modes(self) -> None:
        """Test all verification modes can be configured."""
        for mode in VerificationMode:
            config = VerificationConfig(mode=mode)
            assert config.mode == mode

    def test_on_failure_actions(self) -> None:
        """Test all on_failure actions can be configured."""
        for action in OnFailureAction:
            config = VerificationConfig(on_failure=action)
            assert config.on_failure == action

    def test_threshold_configuration(self) -> None:
        """Test threshold configuration."""
        config = VerificationConfig(
            support_threshold=0.8,
            sample_rate=0.5,
        )
        assert config.support_threshold == 0.8
        assert config.sample_rate == 0.5


class TestEndToEndScenarios:
    """Full end-to-end scenario tests."""

    @pytest.mark.asyncio
    async def test_verify_simple_response(self) -> None:
        """Test verification of a simple response."""
        mock_client = MockLLMClient(
            {
                "Extract": """[{"claim": "The sum is 10", "original_span": "sum is 10", "cites_evidence": [], "is_critical": true, "confidence": 0.9}]""",
                "Evaluate": """{"support_score": 1.0, "issues": [], "reasoning": "Exact match"}""",
            }
        )

        extractor = ClaimExtractor(client=mock_client)
        auditor = EvidenceAuditor(client=mock_client)

        # Simulate response verification
        response = "The sum is 10"
        evidence = {"e1": "total = 5 + 5  # sum is 10"}

        # Extract
        extraction = await extractor.extract_claims(response)
        assert len(extraction.claims) >= 1

        # Audit
        audit = await auditor.audit_claims(extraction.claims, evidence)
        assert audit.total_claims >= 1
        assert audit.flagged_count == 0

    @pytest.mark.asyncio
    async def test_verify_response_with_errors(self) -> None:
        """Test verification catches errors in response."""
        mock_client = MockLLMClient(
            {
                "Extract": """[{"claim": "The file has 100 lines", "original_span": "100 lines", "cites_evidence": ["main.py"], "is_critical": false, "confidence": 0.8}]""",
                "Evaluate": """{"support_score": 0.2, "issues": ["unsupported"], "reasoning": "File only has 50 lines"}""",
            }
        )

        extractor = ClaimExtractor(client=mock_client)
        auditor = EvidenceAuditor(client=mock_client, support_threshold=0.7)

        response = "The file has 100 lines"
        evidence = {"main.py": "# 50 lines of code\n" * 50}

        extraction = await extractor.extract_claims(response)
        audit = await auditor.audit_claims(extraction.claims, evidence)

        assert audit.flagged_count >= 1
        assert audit.results[0].verification.is_flagged
