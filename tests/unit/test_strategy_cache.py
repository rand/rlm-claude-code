"""
Unit tests for strategy_cache module.

Implements: Spec §8.1 Phase 3 - Strategy Learning tests
"""

import tempfile
from pathlib import Path

import pytest

from src.strategy_cache import (
    CachedStrategy,
    FeatureExtractor,
    QueryFeatures,
    StrategyCache,
    StrategySuggestion,
    get_strategy_cache,
)
from src.trajectory_analysis import StrategyAnalysis, StrategyType


class TestQueryFeatures:
    """Tests for QueryFeatures dataclass."""

    def test_default_features(self):
        """Default features are zero/false."""
        features = QueryFeatures()

        assert features.word_count == 0
        assert features.has_file_reference is False
        assert features.is_question is False
        assert features.keywords == ()

    def test_to_vector(self):
        """Converts to numeric vector."""
        features = QueryFeatures(
            word_count=50,
            has_file_reference=True,
            is_question=True,
        )

        vector = features.to_vector()

        assert len(vector) == 11
        assert vector[0] == 0.5  # word_count / 100
        assert vector[1] == 1.0  # has_file_reference
        assert vector[4] == 1.0  # is_question

    def test_to_dict(self):
        """Converts to dictionary."""
        features = QueryFeatures(
            word_count=10,
            has_file_reference=True,
            keywords=("test", "code"),
        )

        d = features.to_dict()

        assert d["word_count"] == 10
        assert d["has_file_reference"] is True
        assert d["keywords"] == ("test", "code")

    def test_from_dict(self):
        """Creates from dictionary."""
        data = {
            "word_count": 20,
            "has_code_reference": True,
            "keywords": ["python", "test"],
        }

        features = QueryFeatures.from_dict(data)

        assert features.word_count == 20
        assert features.has_code_reference is True
        assert features.keywords == ("python", "test")

    def test_round_trip(self):
        """Dict round-trip preserves data."""
        original = QueryFeatures(
            word_count=30,
            has_file_reference=True,
            is_analysis=True,
            mentions_refactoring=True,
            keywords=("refactor", "module"),
        )

        restored = QueryFeatures.from_dict(original.to_dict())

        assert restored.word_count == original.word_count
        assert restored.has_file_reference == original.has_file_reference
        assert restored.is_analysis == original.is_analysis
        assert restored.mentions_refactoring == original.mentions_refactoring
        assert restored.keywords == original.keywords


class TestCachedStrategy:
    """Tests for CachedStrategy dataclass."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return QueryFeatures(
            word_count=10,
            has_file_reference=True,
        )

    def test_create_cached_strategy(self, sample_features):
        """Can create cached strategy."""
        entry = CachedStrategy(
            strategy=StrategyType.PEEKING,
            query_features=sample_features,
            effectiveness=0.8,
            code_patterns=["peek("],
        )

        assert entry.strategy == StrategyType.PEEKING
        assert entry.effectiveness == 0.8
        assert entry.use_count == 1

    def test_to_dict(self, sample_features):
        """Converts to dictionary."""
        entry = CachedStrategy(
            strategy=StrategyType.GREPPING,
            query_features=sample_features,
            effectiveness=0.9,
            code_patterns=["re.search"],
            query_hash="abc123",
        )

        d = entry.to_dict()

        assert d["strategy"] == "grepping"
        assert d["effectiveness"] == 0.9
        assert d["query_hash"] == "abc123"

    def test_from_dict(self):
        """Creates from dictionary."""
        data = {
            "strategy": "peeking",
            "query_features": {"word_count": 5},
            "effectiveness": 0.7,
            "code_patterns": ["peek"],
            "use_count": 3,
            "query_hash": "xyz789",
        }

        entry = CachedStrategy.from_dict(data)

        assert entry.strategy == StrategyType.PEEKING
        assert entry.query_features.word_count == 5
        assert entry.effectiveness == 0.7
        assert entry.use_count == 3


class TestStrategySuggestion:
    """Tests for StrategySuggestion dataclass."""

    def test_create_suggestion(self):
        """Can create strategy suggestion."""
        suggestion = StrategySuggestion(
            strategy=StrategyType.PARTITION_MAP,
            confidence=0.85,
            reason="Similar to 5 cached queries",
            code_patterns=["for file in files"],
            similar_query_count=5,
        )

        assert suggestion.strategy == StrategyType.PARTITION_MAP
        assert suggestion.confidence == 0.85
        assert suggestion.similar_query_count == 5


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create extractor."""
        return FeatureExtractor()

    def test_extract_word_count(self, extractor):
        """Extracts word count."""
        features = extractor.extract("This is a test query with seven words")

        assert features.word_count == 8

    def test_detect_file_reference(self, extractor):
        """Detects file references."""
        features = extractor.extract("Look at main.py for the implementation")

        assert features.has_file_reference is True

    def test_detect_code_reference(self, extractor):
        """Detects code references."""
        features = extractor.extract("The function should return a list")

        assert features.has_code_reference is True

    def test_detect_error_mention(self, extractor):
        """Detects error mentions."""
        features = extractor.extract("There's a bug in the login flow")

        assert features.has_error_mention is True

    def test_detect_question(self, extractor):
        """Detects questions."""
        features = extractor.extract("How does the auth system work?")

        assert features.is_question is True

    def test_detect_command(self, extractor):
        """Detects commands."""
        features = extractor.extract("Run the test suite")

        assert features.is_command is True

    def test_detect_analysis(self, extractor):
        """Detects analysis requests."""
        features = extractor.extract("Analyze the performance of this module")

        assert features.is_analysis is True

    def test_detect_debugging(self, extractor):
        """Detects debugging requests."""
        features = extractor.extract("Debug this failing test")

        assert features.is_debugging is True

    def test_detect_multiple_files(self, extractor):
        """Detects multi-file references."""
        features = extractor.extract("Check auth.py and user.py files")

        assert features.references_multiple_files is True

    def test_detect_architecture(self, extractor):
        """Detects architecture mentions."""
        features = extractor.extract("Explain the architecture of this system")

        assert features.mentions_architecture is True

    def test_detect_refactoring(self, extractor):
        """Detects refactoring requests."""
        features = extractor.extract("Refactor the database layer")

        assert features.mentions_refactoring is True

    def test_extract_keywords(self, extractor):
        """Extracts top keywords."""
        features = extractor.extract("Analyze the authentication module for security issues")

        assert len(features.keywords) > 0
        assert "authentication" in features.keywords or "module" in features.keywords

    def test_keywords_exclude_stop_words(self, extractor):
        """Keywords exclude stop words."""
        features = extractor.extract("The quick brown fox jumps under the lazy dog")

        # "the" and "under" are stop words
        assert "the" not in features.keywords
        assert "under" not in features.keywords

    def test_empty_query(self, extractor):
        """Handles empty query."""
        features = extractor.extract("")

        assert features.word_count == 0
        assert features.keywords == ()


class TestStrategyCache:
    """Tests for StrategyCache class."""

    @pytest.fixture
    def cache(self):
        """Create empty cache."""
        return StrategyCache(max_entries=100)

    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis."""
        return StrategyAnalysis(
            primary_strategy=StrategyType.PEEKING,
            strategy_confidence=0.9,
            effectiveness_score=0.8,
            success=True,
            code_patterns=["peek("],
        )

    def test_add_successful_strategy(self, cache, sample_analysis):
        """Adds successful strategy to cache."""
        result = cache.add("How do I read file.py?", sample_analysis)

        assert result is True
        assert len(cache._entries) == 1

    def test_reject_low_effectiveness(self, cache):
        """Rejects low effectiveness strategies."""
        analysis = StrategyAnalysis(effectiveness_score=0.3)

        result = cache.add("Some query", analysis)

        assert result is False
        assert len(cache._entries) == 0

    def test_update_duplicate(self, cache, sample_analysis):
        """Updates duplicate entries."""
        cache.add("Same query", sample_analysis)
        cache.add("Same query", sample_analysis)

        assert len(cache._entries) == 1
        assert cache._entries[0].use_count == 2

    def test_suggest_similar(self, cache, sample_analysis):
        """Suggests for similar queries."""
        cache.add("How do I read main.py?", sample_analysis)

        suggestions = cache.suggest("How do I read test.py?")

        assert len(suggestions) >= 1
        assert suggestions[0].strategy == StrategyType.PEEKING

    def test_no_suggestions_for_dissimilar(self, cache, sample_analysis):
        """No suggestions for dissimilar queries."""
        cache.add("Read the config file", sample_analysis)

        suggestions = cache.suggest(
            "Run the deployment script in production",
            min_similarity=0.9,
        )

        assert len(suggestions) == 0

    def test_suggest_empty_cache(self, cache):
        """Returns empty for empty cache."""
        suggestions = cache.suggest("Any query")

        assert suggestions == []

    def test_suggest_respects_min_similarity(self, cache, sample_analysis):
        """Respects minimum similarity threshold."""
        cache.add("Analyze auth.py", sample_analysis)

        high_threshold = cache.suggest("Analyze auth.py", min_similarity=0.99)
        low_threshold = cache.suggest("Analyze auth.py", min_similarity=0.1)

        assert len(high_threshold) <= len(low_threshold)

    def test_suggest_respects_top_k(self, cache, sample_analysis):
        """Respects top_k limit."""
        for i in range(5):
            analysis = StrategyAnalysis(
                primary_strategy=StrategyType.PEEKING,
                effectiveness_score=0.8,
            )
            cache.add(f"Read file{i}.py", analysis)

        suggestions = cache.suggest("Read file.py", top_k=2)

        assert len(suggestions) <= 2

    def test_evict_when_over_capacity(self):
        """Evicts entries when over capacity."""
        cache = StrategyCache(max_entries=5)

        for i in range(10):
            analysis = StrategyAnalysis(
                primary_strategy=StrategyType.PEEKING,
                effectiveness_score=0.8,
            )
            cache.add(f"Query {i} about file{i}.py", analysis)

        # Should have evicted some entries
        assert len(cache._entries) <= 5

    def test_evict_keeps_high_value(self):
        """Eviction keeps high-value entries."""
        cache = StrategyCache(max_entries=3)

        # Add low effectiveness
        low = StrategyAnalysis(effectiveness_score=0.51)
        cache.add("Low value query", low)

        # Add high effectiveness
        high = StrategyAnalysis(
            primary_strategy=StrategyType.GREPPING,
            effectiveness_score=0.99,
        )
        cache.add("High value query about grep", high)

        # Add more to trigger eviction
        for i in range(5):
            med = StrategyAnalysis(effectiveness_score=0.6)
            cache.add(f"Medium query {i}", med)

        # High value should survive
        strategies = [e.strategy for e in cache._entries]
        assert StrategyType.GREPPING in strategies

    def test_cosine_similarity(self, cache):
        """Calculates cosine similarity correctly."""
        v1 = [1.0, 0.0, 1.0]
        v2 = [1.0, 0.0, 1.0]

        assert cache._cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self, cache):
        """Orthogonal vectors have zero similarity."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]

        assert cache._cosine_similarity(v1, v2) == 0.0

    def test_cosine_similarity_zero_vector(self, cache):
        """Zero vectors return zero similarity."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]

        assert cache._cosine_similarity(v1, v2) == 0.0

    def test_compute_hash_consistent(self, cache):
        """Hash is consistent for same query."""
        h1 = cache._compute_hash("Test query")
        h2 = cache._compute_hash("Test query")

        assert h1 == h2

    def test_compute_hash_case_insensitive(self, cache):
        """Hash is case insensitive."""
        h1 = cache._compute_hash("Test Query")
        h2 = cache._compute_hash("test query")

        assert h1 == h2

    def test_statistics(self, cache, sample_analysis):
        """Tracks statistics."""
        cache.add("Read file.py content", sample_analysis)
        cache.suggest("Read file.py content")  # Hit

        stats = cache.get_statistics()

        assert stats["entries_added"] >= 1
        assert stats["cache_hits"] >= 1
        assert "hit_rate" in stats

    def test_statistics_miss(self, cache):
        """Tracks cache misses."""
        # Empty cache always misses
        cache.suggest("Any query")

        stats = cache.get_statistics()

        assert stats["cache_misses"] >= 1

    def test_clear(self, cache, sample_analysis):
        """Clears all entries."""
        cache.add("Query", sample_analysis)
        cache.clear()

        assert len(cache._entries) == 0

    def test_persistence_save_load(self, sample_analysis):
        """Saves and loads from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"

            # Save
            cache1 = StrategyCache(persistence_path=path)
            cache1.add("Test query for file.py", sample_analysis)
            cache1.save()

            # Load
            cache2 = StrategyCache(persistence_path=path)

            assert len(cache2._entries) == 1
            assert cache2._entries[0].strategy == StrategyType.PEEKING

    def test_persistence_auto_load(self, sample_analysis):
        """Auto-loads on init if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"

            # Create and save
            cache1 = StrategyCache(persistence_path=path)
            cache1.add("Persisted query about file.py", sample_analysis)
            cache1.save()

            # New cache should auto-load
            cache2 = StrategyCache(persistence_path=path)

            assert len(cache2._entries) == 1


class TestGetStrategyCache:
    """Tests for get_strategy_cache function."""

    def test_returns_cache(self):
        """Returns a StrategyCache instance."""
        cache = get_strategy_cache()

        assert isinstance(cache, StrategyCache)

    def test_returns_same_instance(self):
        """Returns same global instance."""
        cache1 = get_strategy_cache()
        cache2 = get_strategy_cache()

        assert cache1 is cache2


class TestIntegration:
    """Integration tests for strategy cache workflow."""

    def test_full_workflow(self):
        """Full add-suggest-evict workflow."""
        cache = StrategyCache(max_entries=5)

        # Add strategies for similar queries
        for i in range(3):
            analysis = StrategyAnalysis(
                primary_strategy=StrategyType.PEEKING,
                effectiveness_score=0.8,
                code_patterns=["peek(content[:100])"],
            )
            cache.add(f"Read file{i}.py content", analysis)

        # Add a different strategy
        grep_analysis = StrategyAnalysis(
            primary_strategy=StrategyType.GREPPING,
            effectiveness_score=0.85,
            code_patterns=["re.search"],
        )
        cache.add("Search for pattern in code.py", grep_analysis)

        # Suggest for similar file reading query
        suggestions = cache.suggest("Read data.py content")

        assert len(suggestions) >= 1
        # Should suggest peeking since we have 3 similar file reading entries
        strategies = [s.strategy for s in suggestions]
        assert StrategyType.PEEKING in strategies

    def test_keyword_boost(self):
        """Keywords boost similarity."""
        cache = StrategyCache()

        analysis = StrategyAnalysis(
            primary_strategy=StrategyType.PROGRAMMATIC,
            effectiveness_score=0.9,
            code_patterns=["json.loads"],
        )
        # Use a query with file reference for consistent feature matching
        cache.add("Parse the JSON configuration file config.json", analysis)

        # Query with same keywords and similar structure should match
        suggestions = cache.suggest("Parse JSON data from config.json file", min_similarity=0.4)

        assert len(suggestions) >= 1
        assert suggestions[0].strategy == StrategyType.PROGRAMMATIC
