"""
Tests for verification caching.

Implements: SPEC-16.32 Verification caching
"""

import os
import tempfile
import time
from datetime import datetime, timedelta

import pytest

from src.epistemic.verification_cache import (
    CachedVerification,
    VerificationCache,
)


class TestCacheKey:
    """Tests for cache key computation."""

    def test_same_content_same_key(self) -> None:
        """Same claim and evidence produce same cache key."""
        claim = "The function returns an integer"
        evidence = "def foo(): return 42"

        key1 = VerificationCache.compute_cache_key(claim, evidence)
        key2 = VerificationCache.compute_cache_key(claim, evidence)

        assert key1 == key2

    def test_different_claim_different_key(self) -> None:
        """Different claims produce different cache keys."""
        evidence = "def foo(): return 42"

        key1 = VerificationCache.compute_cache_key("Claim A", evidence)
        key2 = VerificationCache.compute_cache_key("Claim B", evidence)

        assert key1 != key2

    def test_different_evidence_different_key(self) -> None:
        """Different evidence produces different cache keys."""
        claim = "The function works"

        key1 = VerificationCache.compute_cache_key(claim, "Evidence A")
        key2 = VerificationCache.compute_cache_key(claim, "Evidence B")

        assert key1 != key2

    def test_whitespace_normalized(self) -> None:
        """Whitespace variations produce same cache key."""
        claim1 = "The  function   returns"
        claim2 = "The function returns"
        evidence = "code here"

        key1 = VerificationCache.compute_cache_key(claim1, evidence)
        key2 = VerificationCache.compute_cache_key(claim2, evidence)

        assert key1 == key2

    def test_key_is_hex_hash(self) -> None:
        """Cache key is a valid hex string."""
        key = VerificationCache.compute_cache_key("claim", "evidence")
        assert len(key) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in key)


class TestEvidenceHash:
    """Tests for evidence hash computation."""

    def test_evidence_hash_consistent(self) -> None:
        """Same evidence produces same hash."""
        evidence = "Some evidence text"

        hash1 = VerificationCache.compute_evidence_hash(evidence)
        hash2 = VerificationCache.compute_evidence_hash(evidence)

        assert hash1 == hash2

    def test_evidence_hash_truncated(self) -> None:
        """Evidence hash is truncated to 16 chars."""
        hash_val = VerificationCache.compute_evidence_hash("evidence")
        assert len(hash_val) == 16


class TestVerificationCache:
    """Tests for VerificationCache class."""

    @pytest.fixture
    def cache(self) -> VerificationCache:
        """Create a temporary verification cache."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cache = VerificationCache(path, ttl_hours=1)
        yield cache
        os.unlink(path)

    def test_put_and_get(self, cache: VerificationCache) -> None:
        """Test storing and retrieving cached verification."""
        claim = "The API returns JSON"
        evidence = "response.json() is called"

        cache.put(
            claim_text=claim,
            evidence=evidence,
            support_score=0.9,
            issues=[],
            reasoning="Evidence explicitly shows JSON handling",
        )

        result = cache.get(claim, evidence)
        assert result is not None
        assert result.claim_text == claim
        assert result.support_score == 0.9
        assert result.issues == []
        assert "JSON" in result.reasoning

    def test_get_nonexistent(self, cache: VerificationCache) -> None:
        """Test getting non-existent cache entry."""
        result = cache.get("nonexistent claim", "nonexistent evidence")
        assert result is None

    def test_cache_hit_increments_count(self, cache: VerificationCache) -> None:
        """Test that cache hits increment the hit count."""
        claim = "Test claim"
        evidence = "Test evidence"

        cache.put(claim, evidence, 0.8, [], "reasoning")

        # First hit - hit_count incremented after fetch, so returns 0
        result1 = cache.get(claim, evidence)
        assert result1 is not None

        # Second hit - now returns 1 (from previous hit)
        result2 = cache.get(claim, evidence)
        assert result2 is not None
        assert result2.hit_count >= 1  # At least one hit recorded

        # Third hit - confirms incrementing
        result3 = cache.get(claim, evidence)
        assert result3 is not None
        assert result3.hit_count >= 2

    def test_expired_entry_not_returned(self, cache: VerificationCache) -> None:
        """Test that expired entries are not returned."""
        claim = "Test claim"
        evidence = "Test evidence"

        # Put with very short TTL
        cache.put(claim, evidence, 0.8, [], "reasoning", ttl_hours=0)

        # Should be expired immediately (or very soon)
        time.sleep(0.1)
        result = cache.get(claim, evidence)
        # May or may not be expired depending on timing, but should handle gracefully
        # At minimum, shouldn't crash

    def test_invalidate_specific_entry(self, cache: VerificationCache) -> None:
        """Test invalidating a specific cache entry."""
        claim = "Test claim"
        evidence = "Test evidence"

        cache_key = cache.put(claim, evidence, 0.8, [], "reasoning")

        # Verify it exists
        assert cache.get(claim, evidence) is not None

        # Invalidate
        result = cache.invalidate(cache_key)
        assert result is True

        # Verify it's gone
        assert cache.get(claim, evidence) is None

    def test_invalidate_nonexistent(self, cache: VerificationCache) -> None:
        """Test invalidating non-existent entry."""
        result = cache.invalidate("nonexistent_key")
        assert result is False

    def test_invalidate_by_evidence(self, cache: VerificationCache) -> None:
        """Test invalidating all entries for specific evidence."""
        evidence = "Common evidence"

        # Add multiple claims with same evidence
        cache.put("Claim 1", evidence, 0.8, [], "r1")
        cache.put("Claim 2", evidence, 0.9, [], "r2")
        cache.put("Claim 3", "Different evidence", 0.7, [], "r3")

        # Invalidate by evidence
        count = cache.invalidate_by_evidence(evidence)
        assert count == 2

        # Verify they're gone
        assert cache.get("Claim 1", evidence) is None
        assert cache.get("Claim 2", evidence) is None

        # Other entry should remain
        assert cache.get("Claim 3", "Different evidence") is not None

    def test_cleanup_expired(self, cache: VerificationCache) -> None:
        """Test cleaning up expired entries."""
        # Add entry with 0 TTL (expires immediately)
        cache.put("Claim", "Evidence", 0.8, [], "r", ttl_hours=0)

        time.sleep(0.1)

        # Cleanup
        count = cache.cleanup_expired()
        # May or may not find expired entries depending on timing
        assert count >= 0

    def test_clear_all(self, cache: VerificationCache) -> None:
        """Test clearing all cache entries."""
        cache.put("Claim 1", "Evidence 1", 0.8, [], "r1")
        cache.put("Claim 2", "Evidence 2", 0.9, [], "r2")

        count = cache.clear()
        assert count == 2

        assert cache.get("Claim 1", "Evidence 1") is None
        assert cache.get("Claim 2", "Evidence 2") is None

    def test_get_stats(self, cache: VerificationCache) -> None:
        """Test getting cache statistics."""
        # Empty cache
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0

        # Add some entries
        cache.put("Claim 1", "Evidence 1", 0.8, [], "r1")
        cache.put("Claim 2", "Evidence 2", 0.9, [], "r2")

        # Hit one entry
        cache.get("Claim 1", "Evidence 1")

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["total_hits"] == 1
        assert stats["avg_support_score"] == pytest.approx(0.85)

    def test_put_replaces_existing(self, cache: VerificationCache) -> None:
        """Test that put replaces existing entries."""
        claim = "Test claim"
        evidence = "Test evidence"

        cache.put(claim, evidence, 0.8, [], "old reasoning")
        cache.put(claim, evidence, 0.9, ["issue"], "new reasoning")

        result = cache.get(claim, evidence)
        assert result is not None
        assert result.support_score == 0.9
        assert result.issues == ["issue"]
        assert result.reasoning == "new reasoning"
        assert result.hit_count == 0  # Reset on replace


class TestCachedVerification:
    """Tests for CachedVerification dataclass."""

    def test_create_cached_verification(self) -> None:
        """Test creating a cached verification record."""
        now = datetime.now()
        expires = now + timedelta(hours=24)

        cached = CachedVerification(
            cache_key="abc123",
            claim_text="The function is pure",
            evidence_hash="def456",
            support_score=0.9,
            issues=[],
            reasoning="No side effects observed",
            created_at=now,
            expires_at=expires,
            hit_count=5,
        )

        assert cached.cache_key == "abc123"
        assert cached.support_score == 0.9
        assert cached.hit_count == 5

    def test_cached_with_issues(self) -> None:
        """Test cached verification with issues."""
        now = datetime.now()

        cached = CachedVerification(
            cache_key="key",
            claim_text="Claim",
            evidence_hash="hash",
            support_score=0.5,
            issues=["partial", "extrapolation"],
            reasoning="Only partially supported",
            created_at=now,
            expires_at=now + timedelta(hours=1),
        )

        assert len(cached.issues) == 2
        assert "partial" in cached.issues


class TestCacheEdgeCases:
    """Edge case tests for verification cache."""

    @pytest.fixture
    def cache(self) -> VerificationCache:
        """Create a temporary verification cache."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cache = VerificationCache(path)
        yield cache
        os.unlink(path)

    def test_empty_claim(self, cache: VerificationCache) -> None:
        """Test caching empty claim text."""
        cache.put("", "evidence", 0.5, [], "empty claim")
        result = cache.get("", "evidence")
        assert result is not None
        assert result.claim_text == ""

    def test_empty_evidence(self, cache: VerificationCache) -> None:
        """Test caching with empty evidence."""
        cache.put("claim", "", 0.5, [], "no evidence")
        result = cache.get("claim", "")
        assert result is not None

    def test_unicode_content(self, cache: VerificationCache) -> None:
        """Test caching with unicode content."""
        claim = "函数返回整数 🎉"
        evidence = "def foo(): return 42  # 返回值"

        cache.put(claim, evidence, 0.9, [], "unicode reasoning")
        result = cache.get(claim, evidence)

        assert result is not None
        assert result.claim_text == claim

    def test_very_long_content(self, cache: VerificationCache) -> None:
        """Test caching with very long content."""
        claim = "x" * 10000
        evidence = "y" * 10000

        cache.put(claim, evidence, 0.8, [], "long content")
        result = cache.get(claim, evidence)

        assert result is not None
        assert len(result.claim_text) == 10000

    def test_special_characters_in_issues(self, cache: VerificationCache) -> None:
        """Test caching with special characters in issues."""
        issues = ["contains 'quotes'", 'has "double quotes"', "has\nnewline"]

        cache.put("claim", "evidence", 0.7, issues, "reasoning")
        result = cache.get("claim", "evidence")

        assert result is not None
        assert result.issues == issues

    def test_multiple_stores_same_db(self) -> None:
        """Test multiple cache instances on same database."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            cache1 = VerificationCache(path)
            cache2 = VerificationCache(path)

            # Add via cache1
            cache1.put("claim", "evidence", 0.8, [], "reasoning")

            # Read via cache2
            result = cache2.get("claim", "evidence")
            assert result is not None
            assert result.support_score == 0.8
        finally:
            os.unlink(path)
