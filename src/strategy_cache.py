"""
Strategy cache with similarity matching.

Implements: Spec §8.1 Phase 3 - Strategy Learning

Caches successful strategies and suggests them for similar queries.
Uses feature-based similarity matching for query comparison.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .trajectory_analysis import StrategyAnalysis, StrategyType


@dataclass
class QueryFeatures:
    """Features extracted from a query for similarity matching."""

    # Structural features
    word_count: int = 0
    has_file_reference: bool = False
    has_code_reference: bool = False
    has_error_mention: bool = False

    # Intent features
    is_question: bool = False
    is_command: bool = False
    is_analysis: bool = False
    is_debugging: bool = False

    # Complexity features
    references_multiple_files: bool = False
    mentions_architecture: bool = False
    mentions_refactoring: bool = False

    # Keywords (top 5)
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def to_vector(self) -> list[float]:
        """Convert to numeric vector for similarity."""
        return [
            self.word_count / 100.0,  # Normalized
            float(self.has_file_reference),
            float(self.has_code_reference),
            float(self.has_error_mention),
            float(self.is_question),
            float(self.is_command),
            float(self.is_analysis),
            float(self.is_debugging),
            float(self.references_multiple_files),
            float(self.mentions_architecture),
            float(self.mentions_refactoring),
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "word_count": self.word_count,
            "has_file_reference": self.has_file_reference,
            "has_code_reference": self.has_code_reference,
            "has_error_mention": self.has_error_mention,
            "is_question": self.is_question,
            "is_command": self.is_command,
            "is_analysis": self.is_analysis,
            "is_debugging": self.is_debugging,
            "references_multiple_files": self.references_multiple_files,
            "mentions_architecture": self.mentions_architecture,
            "mentions_refactoring": self.mentions_refactoring,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryFeatures:
        """Create from dictionary."""
        return cls(
            word_count=data.get("word_count", 0),
            has_file_reference=data.get("has_file_reference", False),
            has_code_reference=data.get("has_code_reference", False),
            has_error_mention=data.get("has_error_mention", False),
            is_question=data.get("is_question", False),
            is_command=data.get("is_command", False),
            is_analysis=data.get("is_analysis", False),
            is_debugging=data.get("is_debugging", False),
            references_multiple_files=data.get("references_multiple_files", False),
            mentions_architecture=data.get("mentions_architecture", False),
            mentions_refactoring=data.get("mentions_refactoring", False),
            keywords=tuple(data.get("keywords", [])),
        )


@dataclass
class CachedStrategy:
    """A cached strategy entry."""

    strategy: StrategyType
    query_features: QueryFeatures
    effectiveness: float
    code_patterns: list[str]
    use_count: int = 1
    last_used: float = field(default_factory=time.time)
    query_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "query_features": self.query_features.to_dict(),
            "effectiveness": self.effectiveness,
            "code_patterns": self.code_patterns,
            "use_count": self.use_count,
            "last_used": self.last_used,
            "query_hash": self.query_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedStrategy:
        """Create from dictionary."""
        return cls(
            strategy=StrategyType(data["strategy"]),
            query_features=QueryFeatures.from_dict(data["query_features"]),
            effectiveness=data["effectiveness"],
            code_patterns=data.get("code_patterns", []),
            use_count=data.get("use_count", 1),
            last_used=data.get("last_used", time.time()),
            query_hash=data.get("query_hash", ""),
        )


@dataclass
class StrategySuggestion:
    """A suggested strategy based on cache match."""

    strategy: StrategyType
    confidence: float
    reason: str
    code_patterns: list[str]
    similar_query_count: int


class FeatureExtractor:
    """Extract features from queries for similarity matching."""

    FILE_PATTERNS = [
        r"\b\w+\.(py|ts|js|go|rs|tsx|jsx|java|cpp|c|h)\b",
        r"\bfile\b",
        r"\bmodule\b",
    ]

    CODE_PATTERNS = [
        r"\bfunction\b",
        r"\bclass\b",
        r"\bmethod\b",
        r"\bvariable\b",
        r"\bimport\b",
    ]

    ERROR_PATTERNS = [
        r"\berror\b",
        r"\bbug\b",
        r"\bfail\b",
        r"\bcrash\b",
        r"\bexception\b",
        r"\bbroken\b",
    ]

    QUESTION_PATTERNS = [
        r"\bwhy\b",
        r"\bhow\b",
        r"\bwhat\b",
        r"\bwhere\b",
        r"\bwhen\b",
        r"\?$",
    ]

    COMMAND_PATTERNS = [
        r"^(run|execute|show|find|list|create|delete|update)\b",
        r"\bplease\b.*\b(run|do|make|create)\b",
    ]

    ANALYSIS_PATTERNS = [
        r"\banalyze\b",
        r"\bexplain\b",
        r"\bunderstand\b",
        r"\breview\b",
        r"\bexamine\b",
    ]

    DEBUG_PATTERNS = [
        r"\bdebug\b",
        r"\bfix\b",
        r"\btroubleshoot\b",
        r"\bdiagnose\b",
    ]

    MULTI_FILE_PATTERNS = [
        r"\band\b.*\.(py|ts|js)",
        r"(files|modules)\b",
        r"across\b",
    ]

    ARCHITECTURE_PATTERNS = [
        r"\barchitecture\b",
        r"\bdesign\b",
        r"\bstructure\b",
        r"\bpattern\b",
    ]

    REFACTORING_PATTERNS = [
        r"\brefactor\b",
        r"\brestructure\b",
        r"\bclean\s*up\b",
        r"\bimprove\b",
    ]

    # Stop words for keyword extraction
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "also", "now", "and", "but", "or", "if", "because",
        "until", "while", "although", "though", "after", "before",
        "this", "that", "these", "those", "it", "its", "i", "me", "my",
        "you", "your", "he", "him", "his", "she", "her", "we", "us",
        "our", "they", "them", "their", "what", "which", "who", "whom",
    }

    def extract(self, query: str) -> QueryFeatures:
        """Extract features from a query."""
        query_lower = query.lower()
        words = re.findall(r"\w+", query_lower)

        features = QueryFeatures(
            word_count=len(words),
            has_file_reference=self._match_any(query_lower, self.FILE_PATTERNS),
            has_code_reference=self._match_any(query_lower, self.CODE_PATTERNS),
            has_error_mention=self._match_any(query_lower, self.ERROR_PATTERNS),
            is_question=self._match_any(query_lower, self.QUESTION_PATTERNS),
            is_command=self._match_any(query_lower, self.COMMAND_PATTERNS),
            is_analysis=self._match_any(query_lower, self.ANALYSIS_PATTERNS),
            is_debugging=self._match_any(query_lower, self.DEBUG_PATTERNS),
            references_multiple_files=self._match_any(query_lower, self.MULTI_FILE_PATTERNS),
            mentions_architecture=self._match_any(query_lower, self.ARCHITECTURE_PATTERNS),
            mentions_refactoring=self._match_any(query_lower, self.REFACTORING_PATTERNS),
            keywords=self._extract_keywords(words),
        )

        return features

    def _match_any(self, text: str, patterns: list[str]) -> bool:
        """Check if text matches any pattern."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_keywords(self, words: list[str]) -> tuple[str, ...]:
        """Extract top keywords from words."""
        # Filter stop words
        keywords = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # Count frequencies
        freq: dict[str, int] = defaultdict(int)
        for w in keywords:
            freq[w] += 1

        # Sort by frequency and return top 5
        sorted_keywords = sorted(freq.items(), key=lambda x: -x[1])
        return tuple(k for k, _ in sorted_keywords[:5])


class StrategyCache:
    """
    Cache for storing and retrieving successful strategies.

    Implements: Spec §8.1 Strategy cache with similarity matching
    """

    def __init__(
        self,
        max_entries: int = 1000,
        min_effectiveness: float = 0.5,
        persistence_path: Path | None = None,
    ):
        """
        Initialize strategy cache.

        Args:
            max_entries: Maximum number of cached entries
            min_effectiveness: Minimum effectiveness to cache
            persistence_path: Path for persistence (optional)
        """
        self.max_entries = max_entries
        self.min_effectiveness = min_effectiveness
        self.persistence_path = persistence_path
        self._entries: list[CachedStrategy] = []
        self._extractor = FeatureExtractor()
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "entries_added": 0,
            "entries_evicted": 0,
        }

        # Load from persistence if available
        if persistence_path and persistence_path.exists():
            self.load()

    def add(
        self,
        query: str,
        analysis: StrategyAnalysis,
    ) -> bool:
        """
        Add a successful strategy to cache.

        Args:
            query: The original query
            analysis: Strategy analysis from trajectory

        Returns:
            True if added, False if rejected
        """
        # Only cache effective strategies
        if analysis.effectiveness_score < self.min_effectiveness:
            return False

        features = self._extractor.extract(query)
        query_hash = self._compute_hash(query)

        # Check for duplicates
        for entry in self._entries:
            if entry.query_hash == query_hash:
                # Update existing entry
                entry.use_count += 1
                entry.last_used = time.time()
                entry.effectiveness = max(entry.effectiveness, analysis.effectiveness_score)
                return True

        # Create new entry
        entry = CachedStrategy(
            strategy=analysis.primary_strategy,
            query_features=features,
            effectiveness=analysis.effectiveness_score,
            code_patterns=analysis.code_patterns,
            query_hash=query_hash,
        )

        self._entries.append(entry)
        self._stats["entries_added"] += 1

        # Evict if over capacity
        if len(self._entries) > self.max_entries:
            self._evict()

        return True

    def suggest(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.6,
    ) -> list[StrategySuggestion]:
        """
        Suggest strategies for a query based on cache.

        Args:
            query: Query to find suggestions for
            top_k: Maximum suggestions to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of strategy suggestions
        """
        if not self._entries:
            self._stats["cache_misses"] += 1
            return []

        features = self._extractor.extract(query)
        query_vector = features.to_vector()

        # Calculate similarities
        scored: list[tuple[float, CachedStrategy]] = []
        for entry in self._entries:
            similarity = self._cosine_similarity(query_vector, entry.query_features.to_vector())

            # Boost for keyword overlap
            keyword_overlap = len(set(features.keywords) & set(entry.query_features.keywords))
            if keyword_overlap > 0:
                similarity += keyword_overlap * 0.1

            if similarity >= min_similarity:
                scored.append((similarity, entry))

        if not scored:
            self._stats["cache_misses"] += 1
            return []

        self._stats["cache_hits"] += 1

        # Sort by similarity
        scored.sort(key=lambda x: -x[0])

        # Group by strategy type
        strategy_scores: dict[StrategyType, list[tuple[float, CachedStrategy]]] = defaultdict(list)
        for sim, entry in scored:
            strategy_scores[entry.strategy].append((sim, entry))

        # Create suggestions
        suggestions = []
        for strategy, matches in strategy_scores.items():
            if len(suggestions) >= top_k:
                break

            avg_sim = sum(s for s, _ in matches) / len(matches)
            best_match = max(matches, key=lambda x: x[0])

            suggestions.append(
                StrategySuggestion(
                    strategy=strategy,
                    confidence=avg_sim,
                    reason=f"Similar to {len(matches)} cached queries",
                    code_patterns=best_match[1].code_patterns,
                    similar_query_count=len(matches),
                )
            )

        return suggestions

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _compute_hash(self, query: str) -> str:
        """Compute hash for query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _evict(self) -> None:
        """Evict least valuable entries."""
        # Score entries by value (effectiveness * recency * use_count)
        now = time.time()
        scored = []
        for entry in self._entries:
            age_hours = (now - entry.last_used) / 3600
            recency = 1.0 / (1.0 + age_hours)
            value = entry.effectiveness * recency * (1 + entry.use_count * 0.1)
            scored.append((value, entry))

        # Sort by value and keep top entries
        scored.sort(key=lambda x: -x[0])
        keep_count = int(self.max_entries * 0.9)  # Keep 90%
        evicted = len(self._entries) - keep_count

        self._entries = [e for _, e in scored[:keep_count]]
        self._stats["entries_evicted"] += evicted

    def save(self) -> None:
        """Save cache to persistence path."""
        if not self.persistence_path:
            return

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [e.to_dict() for e in self._entries],
            "stats": self._stats,
        }
        with open(self.persistence_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load cache from persistence path."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        with open(self.persistence_path) as f:
            data = json.load(f)

        self._entries = [CachedStrategy.from_dict(e) for e in data.get("entries", [])]
        self._stats.update(data.get("stats", {}))

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_lookups = self._stats["cache_hits"] + self._stats["cache_misses"]
        return {
            **self._stats,
            "total_entries": len(self._entries),
            "hit_rate": self._stats["cache_hits"] / total_lookups if total_lookups > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self._entries.clear()


# Global cache instance
_global_cache: StrategyCache | None = None


def get_strategy_cache() -> StrategyCache:
    """Get global strategy cache."""
    global _global_cache
    if _global_cache is None:
        cache_path = Path.home() / ".config" / "rlm-claude-code" / "strategy_cache.json"
        _global_cache = StrategyCache(persistence_path=cache_path)
    return _global_cache


__all__ = [
    "CachedStrategy",
    "FeatureExtractor",
    "QueryFeatures",
    "StrategyCache",
    "StrategySuggestion",
    "get_strategy_cache",
]
