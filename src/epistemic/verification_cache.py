"""
Verification caching for epistemic verification.

Implements: SPEC-16.32 Verification caching

Caches verification results to avoid re-verifying identical claims.
Uses content hash as cache key with TTL-based expiration.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class CachedVerification:
    """
    A cached verification result.

    Attributes:
        cache_key: Hash of claim + evidence
        claim_text: The claim text that was verified
        evidence_hash: Hash of the evidence used
        support_score: Cached evidence support score
        issues: List of issues found during verification
        reasoning: Verification reasoning
        created_at: When the cache entry was created
        expires_at: When the cache entry expires
        hit_count: Number of times this cache entry was used
    """

    cache_key: str
    claim_text: str
    evidence_hash: str
    support_score: float
    issues: list[str]
    reasoning: str
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0


# Schema for verification cache table
CACHE_SCHEMA_SQL = """
-- Verification cache table (SPEC-16.32)
CREATE TABLE IF NOT EXISTS verification_cache (
    cache_key TEXT PRIMARY KEY,
    claim_text TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    support_score REAL NOT NULL,
    issues TEXT NOT NULL,  -- JSON array
    reasoning TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    hit_count INTEGER DEFAULT 0
);

-- Index for expiration cleanup
CREATE INDEX IF NOT EXISTS idx_cache_expires ON verification_cache(expires_at);

-- Index for evidence hash (for batch invalidation)
CREATE INDEX IF NOT EXISTS idx_cache_evidence ON verification_cache(evidence_hash);
"""


class VerificationCache:
    """
    Cache for verification results.

    Implements: SPEC-16.32

    Uses SQLite for persistence with TTL-based expiration.
    Cache keys are hashes of claim + evidence content.
    """

    DEFAULT_TTL_HOURS = 24

    def __init__(
        self,
        db_path: str | None = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ):
        """
        Initialize verification cache.

        Args:
            db_path: Path to SQLite database. If None, uses default.
            ttl_hours: Time-to-live for cache entries in hours.
        """
        if db_path is None:
            db_path = str(Path.home() / ".claude" / "rlm-verification-cache.db")

        self.db_path = db_path
        self.ttl_hours = ttl_hours
        self._ensure_directory()
        self._init_database()

    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self) -> None:
        """Initialize database with schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(CACHE_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def compute_cache_key(claim_text: str, evidence: str) -> str:
        """
        Compute cache key from claim and evidence.

        Uses SHA-256 hash of normalized content.

        Args:
            claim_text: The claim text
            evidence: The evidence text

        Returns:
            Hex digest of the hash
        """
        # Normalize whitespace for consistent hashing
        normalized_claim = " ".join(claim_text.split())
        normalized_evidence = " ".join(evidence.split())

        content = f"CLAIM:{normalized_claim}\nEVIDENCE:{normalized_evidence}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_evidence_hash(evidence: str) -> str:
        """
        Compute hash of evidence for batch invalidation.

        Args:
            evidence: The evidence text

        Returns:
            Hex digest of the hash
        """
        normalized = " ".join(evidence.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get(
        self,
        claim_text: str,
        evidence: str,
    ) -> CachedVerification | None:
        """
        Get cached verification result.

        Args:
            claim_text: The claim to look up
            evidence: The evidence used for verification

        Returns:
            CachedVerification if found and not expired, None otherwise
        """
        cache_key = self.compute_cache_key(claim_text, evidence)
        now_ms = int(time.time() * 1000)

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM verification_cache
                WHERE cache_key = ? AND expires_at > ?
                """,
                (cache_key, now_ms),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Increment hit count
            conn.execute(
                "UPDATE verification_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (cache_key,),
            )
            conn.commit()

            return self._row_to_cached(row)
        finally:
            conn.close()

    def put(
        self,
        claim_text: str,
        evidence: str,
        support_score: float,
        issues: list[str],
        reasoning: str,
        ttl_hours: int | None = None,
    ) -> str:
        """
        Store verification result in cache.

        Args:
            claim_text: The claim that was verified
            evidence: The evidence used
            support_score: The verification support score
            issues: List of issues found
            reasoning: Verification reasoning
            ttl_hours: Optional custom TTL (uses default if None)

        Returns:
            Cache key
        """
        cache_key = self.compute_cache_key(claim_text, evidence)
        evidence_hash = self.compute_evidence_hash(evidence)

        ttl = ttl_hours if ttl_hours is not None else self.ttl_hours
        now = datetime.now()
        expires = now + timedelta(hours=ttl)

        now_ms = int(now.timestamp() * 1000)
        expires_ms = int(expires.timestamp() * 1000)

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO verification_cache
                (cache_key, claim_text, evidence_hash, support_score, issues,
                 reasoning, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    cache_key,
                    claim_text,
                    evidence_hash,
                    support_score,
                    json.dumps(issues),
                    reasoning,
                    now_ms,
                    expires_ms,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return cache_key

    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            cache_key: The cache key to invalidate

        Returns:
            True if entry was found and deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM verification_cache WHERE cache_key = ?",
                (cache_key,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def invalidate_by_evidence(self, evidence: str) -> int:
        """
        Invalidate all cache entries for specific evidence.

        Useful when evidence content changes.

        Args:
            evidence: The evidence text

        Returns:
            Number of entries invalidated
        """
        evidence_hash = self.compute_evidence_hash(evidence)

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM verification_cache WHERE evidence_hash = ?",
                (evidence_hash,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        now_ms = int(time.time() * 1000)

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM verification_cache WHERE expires_at <= ?",
                (now_ms,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM verification_cache")
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        now_ms = int(time.time() * 1000)

        conn = self._get_connection()
        try:
            # Total entries
            cursor = conn.execute("SELECT COUNT(*) as count FROM verification_cache")
            total = cursor.fetchone()["count"]

            # Valid (non-expired) entries
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM verification_cache WHERE expires_at > ?",
                (now_ms,),
            )
            valid = cursor.fetchone()["count"]

            # Total hits
            cursor = conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) as hits FROM verification_cache"
            )
            total_hits = cursor.fetchone()["hits"]

            # Average support score
            cursor = conn.execute("SELECT AVG(support_score) as avg FROM verification_cache")
            row = cursor.fetchone()
            avg_score = row["avg"] if row["avg"] is not None else 0.0

            return {
                "total_entries": total,
                "valid_entries": valid,
                "expired_entries": total - valid,
                "total_hits": total_hits,
                "avg_support_score": avg_score,
                "hit_rate": total_hits / total if total > 0 else 0.0,
            }
        finally:
            conn.close()

    def _row_to_cached(self, row: sqlite3.Row) -> CachedVerification:
        """Convert database row to CachedVerification."""
        return CachedVerification(
            cache_key=row["cache_key"],
            claim_text=row["claim_text"],
            evidence_hash=row["evidence_hash"],
            support_score=row["support_score"],
            issues=json.loads(row["issues"]),
            reasoning=row["reasoning"],
            created_at=datetime.fromtimestamp(row["created_at"] / 1000),
            expires_at=datetime.fromtimestamp(row["expires_at"] / 1000),
            hit_count=row["hit_count"],
        )


def create_cached_auditor(
    cache: VerificationCache,
    auditor: Any,  # EvidenceAuditor
) -> Any:
    """
    Create a caching wrapper around an EvidenceAuditor.

    This is a factory function that wraps an auditor to use caching.

    Args:
        cache: VerificationCache instance
        auditor: EvidenceAuditor instance

    Returns:
        Wrapped auditor with caching
    """
    from functools import wraps

    original_verify = auditor._verify_direct

    @wraps(original_verify)
    async def cached_verify(
        claim_text: str,
        evidence: str,
        model: str,
    ) -> tuple[float, list[str], str]:
        # Check cache first
        cached = cache.get(claim_text, evidence)
        if cached is not None:
            return cached.support_score, cached.issues, cached.reasoning

        # Call original
        support_score, issues, reasoning = await original_verify(claim_text, evidence, model)

        # Cache result
        cache.put(claim_text, evidence, support_score, issues, reasoning)

        return support_score, issues, reasoning

    auditor._verify_direct = cached_verify
    return auditor


__all__ = [
    "CachedVerification",
    "VerificationCache",
    "create_cached_auditor",
]
