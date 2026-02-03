"""
Benchmark tests for epistemic verification module.

Implements: SPEC-16.39 Performance benchmarks for verification

Run with:
    pytest tests/benchmarks/test_verification_benchmarks.py --benchmark-only
    pytest tests/benchmarks/test_verification_benchmarks.py --benchmark-json=verification_results.json
"""

import os
import tempfile
from typing import Any

import pytest

from src.epistemic import (
    ClaimVerification,
    EpistemicGap,
    ExtractedClaim,
    HallucinationReport,
    VerificationCache,
    VerificationConfig,
)
from src.epistemic.prompts import (
    PromptTemplate,
    estimate_prompt_tokens,
    format_claims_compact,
    format_evidence_compact,
    format_prompt,
    truncate_evidence,
)
from src.epistemic.similarity import (
    cosine_similarity,
    text_overlap_similarity,
)


class TestClaimVerificationBenchmarks:
    """Benchmark tests for ClaimVerification operations."""

    def test_claim_verification_creation(self, benchmark: Any) -> None:
        """Benchmark ClaimVerification object creation."""

        def create_claims() -> list[ClaimVerification]:
            claims = []
            for i in range(100):
                claims.append(
                    ClaimVerification(
                        claim_id=f"c{i}",
                        claim_text=f"The function returns {i}",
                        evidence_ids=[f"e{i}", f"e{i + 1}"],
                        evidence_support=0.8 + (i % 20) / 100,
                        evidence_dependence=0.7 + (i % 30) / 100,
                        consistency_score=0.9,
                    )
                )
            return claims

        result = benchmark(create_claims)
        assert len(result) == 100

    def test_claim_verification_combined_score(self, benchmark: Any) -> None:
        """Benchmark combined_score computation."""
        claims = [
            ClaimVerification(
                claim_id=f"c{i}",
                claim_text=f"Claim {i}",
                evidence_support=0.8,
                evidence_dependence=0.7,
            )
            for i in range(100)
        ]

        def compute_scores() -> list[float]:
            return [c.combined_score for c in claims]

        result = benchmark(compute_scores)
        assert len(result) == 100


class TestHallucinationReportBenchmarks:
    """Benchmark tests for HallucinationReport operations."""

    def test_report_creation_with_claims(self, benchmark: Any) -> None:
        """Benchmark HallucinationReport creation with many claims."""

        def create_report() -> HallucinationReport:
            claims = [
                ClaimVerification(
                    claim_id=f"c{i}",
                    claim_text=f"Claim number {i} about the system",
                    evidence_ids=[f"e{i}"],
                    evidence_support=0.7 + (i % 30) / 100,
                    evidence_dependence=0.6 + (i % 40) / 100,
                    is_flagged=i % 5 == 0,
                    flag_reason="unsupported" if i % 5 == 0 else None,
                )
                for i in range(50)
            ]
            gaps = [
                EpistemicGap(
                    claim_id=f"c{i}",
                    claim_text=f"Claim {i}",
                    gap_type="unsupported",
                    gap_bits=1.5 + i * 0.1,
                )
                for i in range(10)
            ]
            return HallucinationReport(
                response_id="resp-bench",
                claims=claims,
                gaps=gaps,
            )

        result = benchmark(create_report)
        assert result.total_claims == 50
        assert result.flagged_claims == 10

    def test_report_add_claims_incrementally(self, benchmark: Any) -> None:
        """Benchmark adding claims incrementally to report."""

        def add_claims() -> HallucinationReport:
            report = HallucinationReport(response_id="resp-bench")
            for i in range(50):
                report.add_claim(
                    ClaimVerification(
                        claim_id=f"c{i}",
                        claim_text=f"Claim {i}",
                        evidence_support=0.8,
                        evidence_dependence=0.7,
                    )
                )
            return report

        result = benchmark(add_claims)
        assert result.total_claims == 50


class TestVerificationConfigBenchmarks:
    """Benchmark tests for VerificationConfig operations."""

    def test_config_should_verify_sampling(self, benchmark: Any) -> None:
        """Benchmark should_verify_claim in sample mode."""
        config = VerificationConfig(mode="sample", sample_rate=0.3)

        def check_many_claims() -> list[bool]:
            results = []
            for i in range(1000):
                results.append(
                    config.should_verify_claim(
                        claim_index=i,
                        is_critical=i % 10 == 0,
                        claim_text=f"This claim might be {i}" if i % 7 == 0 else f"Claim {i}",
                    )
                )
            return results

        result = benchmark(check_many_claims)
        assert len(result) == 1000

    def test_config_uncertainty_marker_detection(self, benchmark: Any) -> None:
        """Benchmark uncertainty marker detection."""
        config = VerificationConfig()
        texts = [
            "The function definitely returns 42",
            "The function probably returns 42",
            "The function might return 42",
            "I think the function returns 42",
            "The function seems to return 42",
            "The function returns approximately 42",
        ] * 100

        def check_uncertainty() -> list[bool]:
            return [config._has_uncertainty_markers(text) for text in texts]

        result = benchmark(check_uncertainty)
        assert len(result) == 600


class TestVerificationCacheBenchmarks:
    """Benchmark tests for VerificationCache operations."""

    @pytest.fixture
    def cache(self) -> VerificationCache:
        """Create temporary verification cache."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cache = VerificationCache(path, ttl_hours=1)
        yield cache
        os.unlink(path)

    def test_cache_put_performance(self, benchmark: Any, cache: VerificationCache) -> None:
        """Benchmark cache put operations."""

        def do_puts() -> list[str]:
            keys = []
            for i in range(100):
                key = cache.put(
                    claim_text=f"The function returns {i}",
                    evidence=f"def func(): return {i}",
                    support_score=0.8 + (i % 20) / 100,
                    issues=[],
                    reasoning=f"Evidence supports claim {i}",
                )
                keys.append(key)
            return keys

        result = benchmark(do_puts)
        assert len(result) == 100

    def test_cache_get_performance(self, benchmark: Any, cache: VerificationCache) -> None:
        """Benchmark cache get operations (hits)."""
        # Pre-populate cache
        claims_evidence = []
        for i in range(100):
            claim = f"The function returns {i}"
            evidence = f"def func(): return {i}"
            cache.put(
                claim_text=claim,
                evidence=evidence,
                support_score=0.9,
                issues=[],
                reasoning="Evidence supports claim",
            )
            claims_evidence.append((claim, evidence))

        def do_gets() -> int:
            hits = 0
            for claim, evidence in claims_evidence:
                result = cache.get(claim, evidence)
                if result:
                    hits += 1
            return hits

        result = benchmark(do_gets)
        assert result == 100

    def test_cache_mixed_operations(self, benchmark: Any, cache: VerificationCache) -> None:
        """Benchmark mixed cache operations."""

        def do_mixed() -> int:
            hits = 0
            for i in range(50):
                claim = f"Claim number {i}"
                evidence = f"Evidence for {i}"
                # Put
                cache.put(
                    claim_text=claim,
                    evidence=evidence,
                    support_score=0.85,
                    issues=[],
                    reasoning="Reasoning",
                )
                # Get (should hit)
                result = cache.get(claim, evidence)
                if result:
                    hits += 1
                # Get miss
                cache.get(f"Nonexistent claim {i}", "No evidence")
            return hits

        result = benchmark(do_mixed)
        assert result == 50


class TestSimilarityBenchmarks:
    """Benchmark tests for similarity computation."""

    def test_cosine_similarity_performance(self, benchmark: Any) -> None:
        """Benchmark cosine similarity computation."""
        # Create random-ish vectors
        vec_a = [float(i % 100) / 100 for i in range(384)]
        vec_b = [float((i + 50) % 100) / 100 for i in range(384)]

        def compute_similarity() -> float:
            total = 0.0
            for _ in range(100):
                total += cosine_similarity(vec_a, vec_b)
            return total

        result = benchmark(compute_similarity)
        assert result > 0

    def test_text_overlap_similarity_performance(self, benchmark: Any) -> None:
        """Benchmark text overlap similarity."""
        text_a = "The quick brown fox jumps over the lazy dog " * 10
        text_b = "The fast brown fox leaps over the sleepy dog " * 10

        def compute_overlap() -> float:
            total = 0.0
            for _ in range(100):
                total += text_overlap_similarity(text_a, text_b)
            return total

        result = benchmark(compute_overlap)
        assert result > 0


class TestPromptBenchmarks:
    """Benchmark tests for prompt operations."""

    def test_format_prompt_performance(self, benchmark: Any) -> None:
        """Benchmark prompt formatting."""

        def format_prompts() -> list[tuple[str, str]]:
            results = []
            for i in range(100):
                system, user = format_prompt(
                    PromptTemplate.DIRECT_VERIFICATION,
                    claim=f"The function returns {i}",
                    evidence=f"def func(): return {i}\n" * 10,
                )
                results.append((system, user))
            return results

        result = benchmark(format_prompts)
        assert len(result) == 100

    def test_estimate_tokens_performance(self, benchmark: Any) -> None:
        """Benchmark token estimation."""

        def estimate_many() -> list[int]:
            results = []
            for i in range(100):
                tokens = estimate_prompt_tokens(
                    PromptTemplate.CLAIM_EXTRACTION,
                    text=f"This is a long text about function {i}. " * 50,
                )
                results.append(tokens)
            return results

        result = benchmark(estimate_many)
        assert len(result) == 100

    def test_truncate_evidence_performance(self, benchmark: Any) -> None:
        """Benchmark evidence truncation."""
        long_evidence = "x" * 10000

        def truncate_many() -> list[str]:
            results = []
            for max_chars in [500, 1000, 2000, 3000, 5000] * 20:
                results.append(truncate_evidence(long_evidence, max_chars=max_chars))
            return results

        result = benchmark(truncate_many)
        assert len(result) == 100

    def test_format_claims_compact_performance(self, benchmark: Any) -> None:
        """Benchmark compact claims formatting."""
        claims = [{"claim": f"This is claim number {i} about the system"} for i in range(50)]

        def format_many() -> list[str]:
            results = []
            for _ in range(100):
                results.append(format_claims_compact(claims))
            return results

        result = benchmark(format_many)
        assert len(result) == 100

    def test_format_evidence_compact_performance(self, benchmark: Any) -> None:
        """Benchmark compact evidence formatting."""
        evidence = {f"e{i}": f"Evidence content number {i} " * 20 for i in range(20)}

        def format_many() -> list[str]:
            results = []
            for _ in range(100):
                results.append(format_evidence_compact(evidence, max_per_item=200))
            return results

        result = benchmark(format_many)
        assert len(result) == 100


class TestExtractedClaimBenchmarks:
    """Benchmark tests for ExtractedClaim operations."""

    def test_extracted_claim_creation(self, benchmark: Any) -> None:
        """Benchmark ExtractedClaim creation."""

        def create_claims() -> list[ExtractedClaim]:
            claims = []
            for i in range(100):
                claims.append(
                    ExtractedClaim(
                        claim_id=f"c{i}",
                        claim_text=f"The function at line {i} returns an integer value",
                        original_span=f"returns {i}",
                        evidence_ids=[f"src/main.py:{i}"],
                        is_critical=i % 5 == 0,
                        confidence=0.8 + (i % 20) / 100,
                    )
                )
            return claims

        result = benchmark(create_claims)
        assert len(result) == 100


class TestIntegrationBenchmarks:
    """Integration benchmarks for complete verification workflows."""

    @pytest.fixture
    def cache(self) -> VerificationCache:
        """Create temporary verification cache."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cache = VerificationCache(path, ttl_hours=1)
        yield cache
        os.unlink(path)

    def test_full_verification_data_flow(self, benchmark: Any, cache: VerificationCache) -> None:
        """Benchmark complete verification data flow (without LLM calls)."""
        config = VerificationConfig(mode="sample", sample_rate=0.3)

        def do_workflow() -> HallucinationReport:
            # Simulate extracted claims
            claims = [
                ExtractedClaim(
                    claim_id=f"c{i}",
                    claim_text=f"The function returns {i}",
                    evidence_ids=[f"e{i}"],
                    is_critical=i % 5 == 0,
                    confidence=0.85,
                )
                for i in range(20)
            ]

            # Filter claims based on config
            claims_to_verify = [
                claim
                for i, claim in enumerate(claims)
                if config.should_verify_claim(i, claim.is_critical, claim.claim_text)
            ]

            # Check cache for each claim
            cache_hits = 0
            for claim in claims_to_verify:
                cached = cache.get(claim.claim_text, f"evidence for {claim.claim_id}")
                if cached:
                    cache_hits += 1

            # Create verification results
            verifications = [
                ClaimVerification(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    evidence_ids=claim.evidence_ids,
                    evidence_support=0.85,
                    evidence_dependence=0.75,
                    is_flagged=False,
                )
                for claim in claims_to_verify
            ]

            # Cache results
            for v in verifications:
                cache.put(
                    claim_text=v.claim_text,
                    evidence=f"evidence for {v.claim_id}",
                    support_score=v.evidence_support,
                    issues=[],
                    reasoning="Verified",
                )

            # Build report
            report = HallucinationReport(response_id="resp-bench", claims=verifications, gaps=[])

            return report

        result = benchmark(do_workflow)
        assert result.total_claims > 0

    def test_batch_claim_processing(self, benchmark: Any) -> None:
        """Benchmark batch claim processing."""

        def process_batch() -> list[dict[str, Any]]:
            # Simulate batch of responses with claims
            batch_results = []
            for resp_idx in range(10):
                claims = []
                for claim_idx in range(10):
                    # Format prompt
                    system, user = format_prompt(
                        PromptTemplate.DIRECT_VERIFICATION,
                        claim=f"Claim {claim_idx} from response {resp_idx}",
                        evidence=f"Evidence {claim_idx}",
                    )
                    # Estimate tokens
                    tokens = estimate_prompt_tokens(
                        PromptTemplate.DIRECT_VERIFICATION,
                        claim=f"Claim {claim_idx}",
                        evidence=f"Evidence {claim_idx}",
                    )
                    claims.append(
                        {
                            "claim_idx": claim_idx,
                            "system_len": len(system),
                            "user_len": len(user),
                            "tokens": tokens,
                        }
                    )
                batch_results.append({"resp_idx": resp_idx, "claims": claims})
            return batch_results

        result = benchmark(process_batch)
        assert len(result) == 10
        assert all(len(r["claims"]) == 10 for r in result)
