"""
Property-based tests for strategy learning modules.

Tests invariants and properties of:
- StrategyCache
- TrajectoryAnalysis
- ToolBridge
"""

import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.orchestration_schema import ToolAccessLevel
from src.strategy_cache import (
    FeatureExtractor,
    QueryFeatures,
    StrategyCache,
)
from src.tool_bridge import ToolBridge, ToolPermissions
from src.trajectory import TrajectoryEvent, TrajectoryEventType
from src.trajectory_analysis import (
    StrategyAnalysis,
    StrategySignal,
    StrategyType,
    TrajectoryAnalyzer,
    TrajectoryMetrics,
)

# Strategies for generating test data
strategy_types = st.sampled_from(list(StrategyType))
event_types = st.sampled_from(list(TrajectoryEventType))
confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
word_counts = st.integers(min_value=0, max_value=1000)

# Generate valid query text
query_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
)


class TestQueryFeaturesProperties:
    """Property tests for QueryFeatures."""

    @given(word_count=word_counts)
    def test_vector_length_constant(self, word_count: int):
        """Feature vector always has consistent length."""
        features = QueryFeatures(word_count=word_count)
        vector = features.to_vector()

        assert len(vector) == 11  # Fixed feature count

    @given(
        word_count=word_counts,
        has_file=st.booleans(),
        has_code=st.booleans(),
        is_question=st.booleans(),
    )
    def test_to_dict_round_trip(
        self, word_count: int, has_file: bool, has_code: bool, is_question: bool
    ):
        """Features round-trip through dict correctly."""
        original = QueryFeatures(
            word_count=word_count,
            has_file_reference=has_file,
            has_code_reference=has_code,
            is_question=is_question,
        )

        d = original.to_dict()
        restored = QueryFeatures.from_dict(d)

        assert restored.word_count == original.word_count
        assert restored.has_file_reference == original.has_file_reference
        assert restored.has_code_reference == original.has_code_reference
        assert restored.is_question == original.is_question

    @given(
        wc1=word_counts,
        wc2=word_counts,
    )
    def test_vector_normalization(self, wc1: int, wc2: int):
        """Word count is normalized in vector representation."""
        f1 = QueryFeatures(word_count=wc1)
        f2 = QueryFeatures(word_count=wc2)

        v1 = f1.to_vector()
        v2 = f2.to_vector()

        # First element is normalized word count
        assert v1[0] == wc1 / 100.0
        assert v2[0] == wc2 / 100.0


class TestFeatureExtractorProperties:
    """Property tests for FeatureExtractor."""

    @given(query=query_text)
    @settings(max_examples=100)
    def test_extract_produces_valid_features(self, query: str):
        """Extraction always produces valid features."""
        extractor = FeatureExtractor()
        features = extractor.extract(query)

        assert isinstance(features, QueryFeatures)
        assert features.word_count >= 0
        assert len(features.keywords) <= 5

    @given(query=query_text)
    def test_extract_deterministic(self, query: str):
        """Same query produces same features."""
        extractor = FeatureExtractor()

        f1 = extractor.extract(query)
        f2 = extractor.extract(query)

        assert f1.word_count == f2.word_count
        assert f1.keywords == f2.keywords
        assert f1.to_vector() == f2.to_vector()


class TestStrategyCacheProperties:
    """Property tests for StrategyCache."""

    @given(
        effectiveness=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        strategy=strategy_types,
    )
    def test_add_respects_min_effectiveness(self, effectiveness: float, strategy: StrategyType):
        """Cache only accepts entries above min effectiveness."""
        cache = StrategyCache(min_effectiveness=0.5)

        analysis = StrategyAnalysis(
            primary_strategy=strategy,
            effectiveness_score=effectiveness,
        )

        result = cache.add("test query", analysis)

        if effectiveness >= 0.5:
            assert result is True
            assert len(cache._entries) == 1
        else:
            assert result is False
            assert len(cache._entries) == 0

    @given(max_entries=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_eviction_respects_max_entries(self, max_entries: int):
        """Cache never exceeds max entries after eviction."""
        cache = StrategyCache(max_entries=max_entries)

        # Add more entries than max
        for i in range(max_entries + 5):
            analysis = StrategyAnalysis(
                primary_strategy=StrategyType.PEEKING,
                effectiveness_score=0.8,
            )
            cache.add(f"unique query {i} about file{i}.py", analysis)

        assert len(cache._entries) <= max_entries

    @given(
        v1=st.lists(
            st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=3,
        ),
        v2=st.lists(
            st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=3,
        ),
    )
    def test_cosine_similarity_bounds(self, v1: list[float], v2: list[float]):
        """Cosine similarity is always approximately in [-1, 1]."""
        cache = StrategyCache()
        similarity = cache._cosine_similarity(v1, v2)

        # Allow small floating point epsilon
        epsilon = 1e-10
        assert -1.0 - epsilon <= similarity <= 1.0 + epsilon or similarity == 0.0

    @given(query=query_text)
    def test_hash_deterministic(self, query: str):
        """Query hash is deterministic."""
        cache = StrategyCache()

        h1 = cache._compute_hash(query)
        h2 = cache._compute_hash(query)

        assert h1 == h2
        assert len(h1) == 16  # SHA256 truncated to 16 chars


class TestTrajectoryAnalysisProperties:
    """Property tests for TrajectoryAnalysis."""

    @given(strategy=strategy_types, conf=confidence)
    def test_strategy_signal_valid(self, strategy: StrategyType, conf: float):
        """Strategy signals can be created with any valid strategy/confidence."""
        signal = StrategySignal(
            strategy=strategy,
            confidence=conf,
            evidence="test",
            event_index=0,
        )

        assert signal.strategy == strategy
        assert signal.confidence == conf

    @given(
        total=st.integers(min_value=0, max_value=100),
        errors=st.integers(min_value=0, max_value=50),
    )
    def test_metrics_consistency(self, total: int, errors: int):
        """Metrics maintain consistency."""
        assume(errors <= total)

        metrics = TrajectoryMetrics(
            total_events=total,
            error_count=errors,
        )

        assert metrics.error_count <= metrics.total_events

    @given(strategy=strategy_types)
    def test_analysis_to_dict_contains_strategy(self, strategy: StrategyType):
        """Analysis dict always contains primary strategy."""
        analysis = StrategyAnalysis(
            primary_strategy=strategy,
            strategy_confidence=0.9,
        )

        d = analysis.to_dict()

        assert "primary_strategy" in d
        assert d["primary_strategy"] == strategy.value


class TestTrajectoryAnalyzerProperties:
    """Property tests for TrajectoryAnalyzer."""

    @given(num_events=st.integers(min_value=0, max_value=20))
    @settings(max_examples=50)
    def test_analyze_handles_any_event_count(self, num_events: int):
        """Analyzer handles any number of events."""
        analyzer = TrajectoryAnalyzer()

        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content=f"code {i}",
            )
            for i in range(num_events)
        ]

        analysis = analyzer.analyze(events)

        assert analysis.metrics.total_events == num_events

    @given(depth=st.integers(min_value=0, max_value=10))
    def test_max_depth_tracked(self, depth: int):
        """Max depth is correctly tracked."""
        analyzer = TrajectoryAnalyzer()

        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=d,
                content="code",
            )
            for d in range(depth + 1)
        ]

        analysis = analyzer.analyze(events)

        assert analysis.metrics.max_depth == depth


class TestToolBridgeProperties:
    """Property tests for ToolBridge."""

    @given(level=st.sampled_from(list(ToolAccessLevel)))
    def test_permissions_from_level(self, level: ToolAccessLevel):
        """Permissions can be created from any access level."""
        perms = ToolPermissions.from_access_level(level)

        assert perms.access_level == level

        # Verify permission hierarchy
        if level == ToolAccessLevel.NONE:
            assert perms.allow_bash is False
            assert perms.allow_file_read is False
        elif level == ToolAccessLevel.FULL:
            assert perms.allow_bash is True
            assert perms.allow_file_read is True
            assert perms.allow_file_write is True

    @given(level=st.sampled_from([ToolAccessLevel.NONE, ToolAccessLevel.REPL_ONLY]))
    def test_restricted_levels_deny_file_read(self, level: ToolAccessLevel):
        """Restricted levels deny file reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            perms = ToolPermissions.from_access_level(level)
            bridge = ToolBridge(permissions=perms, working_dir=tmpdir)

            result = bridge.tool_call("read", "any_file.txt")

            assert result.success is False
            assert "not permitted" in result.error.lower()

    @given(
        level=st.sampled_from(
            [ToolAccessLevel.NONE, ToolAccessLevel.REPL_ONLY, ToolAccessLevel.READ_ONLY]
        )
    )
    def test_non_full_levels_deny_bash(self, level: ToolAccessLevel):
        """Non-full levels deny bash execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            perms = ToolPermissions.from_access_level(level)
            bridge = ToolBridge(permissions=perms, working_dir=tmpdir)

            result = bridge.tool_call("bash", "echo hello")

            assert result.success is False
            assert "not permitted" in result.error.lower()

    def test_history_accumulates(self):
        """Tool invocation history accumulates correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "test.txt").write_text("content")

            perms = ToolPermissions.from_access_level(ToolAccessLevel.READ_ONLY)
            bridge = ToolBridge(permissions=perms, working_dir=tmpdir)

            bridge.tool_call("read", "test.txt")
            bridge.tool_call("glob", "*.txt")
            bridge.tool_call("ls")

            history = bridge.get_history()
            assert len(history) == 3

            bridge.clear_history()
            assert len(bridge.get_history()) == 0
