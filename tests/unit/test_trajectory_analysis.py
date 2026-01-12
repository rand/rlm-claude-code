"""
Unit tests for trajectory_analysis module.

Implements: Spec §8.1 Phase 3 - Strategy Learning tests
"""

import time

import pytest

from src.trajectory import TrajectoryEvent, TrajectoryEventType
from src.trajectory_analysis import (
    StrategyAnalysis,
    StrategySignal,
    StrategyType,
    TrajectoryAnalyzer,
    TrajectoryMetrics,
    analyze_trajectory,
    extract_strategy_summary,
)


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_all_types_exist(self):
        """All strategy types exist."""
        assert StrategyType.PEEKING
        assert StrategyType.GREPPING
        assert StrategyType.PARTITION_MAP
        assert StrategyType.PROGRAMMATIC
        assert StrategyType.RECURSIVE
        assert StrategyType.ITERATIVE
        assert StrategyType.DIRECT
        assert StrategyType.UNKNOWN


class TestTrajectoryMetrics:
    """Tests for TrajectoryMetrics dataclass."""

    def test_default_metrics(self):
        """Default metrics are zero."""
        metrics = TrajectoryMetrics()

        assert metrics.total_events == 0
        assert metrics.repl_executions == 0
        assert metrics.recursive_calls == 0
        assert metrics.error_count == 0
        assert metrics.completed is False


class TestStrategySignal:
    """Tests for StrategySignal dataclass."""

    def test_create_signal(self):
        """Can create strategy signal."""
        signal = StrategySignal(
            strategy=StrategyType.PEEKING,
            confidence=0.8,
            evidence="peek(",
            event_index=5,
        )

        assert signal.strategy == StrategyType.PEEKING
        assert signal.confidence == 0.8
        assert signal.evidence == "peek("
        assert signal.event_index == 5


class TestTrajectoryAnalyzer:
    """Tests for TrajectoryAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return TrajectoryAnalyzer()

    @pytest.fixture
    def basic_events(self):
        """Create basic trajectory events."""
        now = time.time()
        return [
            TrajectoryEvent(
                type=TrajectoryEventType.RLM_START,
                depth=0,
                content="Starting RLM",
                timestamp=now,
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="peek(files['main.py'][:100])",
                timestamp=now + 0.1,
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_RESULT,
                depth=0,
                content="First 100 chars...",
                timestamp=now + 0.2,
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                depth=0,
                content="The answer is 42",
                timestamp=now + 0.3,
            ),
        ]

    def test_analyze_empty_events(self, analyzer):
        """Handles empty event list."""
        analysis = analyzer.analyze([])

        assert analysis.metrics.total_events == 0
        assert analysis.primary_strategy == StrategyType.UNKNOWN

    def test_analyze_basic_trajectory(self, analyzer, basic_events):
        """Analyzes basic trajectory."""
        analysis = analyzer.analyze(basic_events)

        assert analysis.metrics.total_events == 4
        assert analysis.metrics.completed is True
        assert analysis.metrics.final_answer_found is True

    def test_detect_peeking_strategy(self, analyzer):
        """Detects peeking strategy."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="result = peek(context[:50])",
            ),
        ]

        analysis = analyzer.analyze(events)

        strategies = [s.strategy for s in analysis.strategies]
        assert StrategyType.PEEKING in strategies

    def test_detect_grepping_strategy(self, analyzer):
        """Detects grepping strategy."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="matches = re.search(r'pattern', content)",
            ),
        ]

        analysis = analyzer.analyze(events)

        strategies = [s.strategy for s in analysis.strategies]
        assert StrategyType.GREPPING in strategies

    def test_detect_partition_map_strategy(self, analyzer):
        """Detects partition+map strategy."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="for f in files: recursive_query(f)",
            ),
        ]

        analysis = analyzer.analyze(events)

        strategies = [s.strategy for s in analysis.strategies]
        assert StrategyType.PARTITION_MAP in strategies

    def test_detect_programmatic_strategy(self, analyzer):
        """Detects programmatic strategy."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="def analyze_file(path):\n    import json\n    return json.loads(content)",
            ),
        ]

        analysis = analyzer.analyze(events)

        strategies = [s.strategy for s in analysis.strategies]
        assert StrategyType.PROGRAMMATIC in strategies

    def test_detect_recursive_strategy(self, analyzer):
        """Detects recursive strategy from event types."""
        events = [
            TrajectoryEvent(type=TrajectoryEventType.RECURSE_START, depth=0, content="Spawning"),
            TrajectoryEvent(type=TrajectoryEventType.RECURSE_END, depth=0, content="Returned"),
            TrajectoryEvent(type=TrajectoryEventType.RECURSE_START, depth=0, content="Spawning"),
            TrajectoryEvent(type=TrajectoryEventType.RECURSE_END, depth=0, content="Returned"),
        ]

        analysis = analyzer.analyze(events)

        assert analysis.metrics.recursive_calls == 2
        strategies = [s.strategy for s in analysis.strategies]
        assert StrategyType.RECURSIVE in strategies

    def test_detect_iterative_strategy(self, analyzer):
        """Detects iterative strategy from many REPL executions."""
        events = [
            TrajectoryEvent(type=TrajectoryEventType.REPL_EXEC, depth=0, content=f"step {i}")
            for i in range(5)
        ]

        analysis = analyzer.analyze(events)

        strategies = [s.strategy for s in analysis.strategies]
        assert StrategyType.ITERATIVE in strategies

    def test_extract_code_patterns(self, analyzer):
        """Extracts code patterns."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="def process_file(path): pass",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="results = [x for x in files]",
            ),
        ]

        analysis = analyzer.analyze(events)

        assert "function:process_file" in analysis.code_patterns
        assert "list_comprehension" in analysis.code_patterns

    def test_extract_metrics(self, analyzer):
        """Extracts metrics correctly."""
        now = time.time()
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="code",
                timestamp=now,
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=1,
                content="code",
                timestamp=now + 0.5,
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.ERROR,
                depth=1,
                content="Error occurred",
                timestamp=now + 1.0,
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                depth=0,
                content="Answer",
                timestamp=now + 1.5,
            ),
        ]

        analysis = analyzer.analyze(events)

        assert analysis.metrics.total_events == 4
        assert analysis.metrics.repl_executions == 2
        assert analysis.metrics.error_count == 1
        assert analysis.metrics.max_depth == 1
        assert analysis.metrics.completed is True

    def test_primary_strategy_determination(self, analyzer):
        """Determines primary strategy correctly."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="peek(content[:50])",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="peek(files[:100])",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="re.search(pattern, text)",
            ),
        ]

        analysis = analyzer.analyze(events)

        # Peeking should be primary since it appears twice
        assert analysis.primary_strategy == StrategyType.PEEKING

    def test_effectiveness_calculation(self, analyzer):
        """Calculates effectiveness score."""
        events = [
            TrajectoryEvent(type=TrajectoryEventType.REPL_EXEC, depth=0, content="code"),
            TrajectoryEvent(type=TrajectoryEventType.FINAL, depth=0, content="Answer"),
        ]

        analysis = analyzer.analyze(events)

        # Successful, efficient trajectory should have high score
        assert analysis.success is True
        assert analysis.effectiveness_score > 0.5

    def test_to_dict(self, analyzer, basic_events):
        """Converts analysis to dict."""
        analysis = analyzer.analyze(basic_events)
        d = analysis.to_dict()

        assert "primary_strategy" in d
        assert "strategies" in d
        assert "metrics" in d
        assert "success" in d
        assert "effectiveness_score" in d


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_trajectory(self):
        """analyze_trajectory convenience function works."""
        events = [
            TrajectoryEvent(type=TrajectoryEventType.REPL_EXEC, depth=0, content="code"),
            TrajectoryEvent(type=TrajectoryEventType.FINAL, depth=0, content="Answer"),
        ]

        analysis = analyze_trajectory(events)

        assert isinstance(analysis, StrategyAnalysis)
        assert analysis.metrics.total_events == 2

    def test_extract_strategy_summary(self):
        """extract_strategy_summary works."""
        events = [
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                depth=0,
                content="peek(content[:50])",
            ),
            TrajectoryEvent(type=TrajectoryEventType.FINAL, depth=0, content="Answer"),
        ]

        summary = extract_strategy_summary(events)

        assert "primary_strategy" in summary
        assert "confidence" in summary
        assert "success" in summary
        assert "effectiveness" in summary
        assert "patterns" in summary


class TestStrategyAnalysis:
    """Tests for StrategyAnalysis dataclass."""

    def test_default_analysis(self):
        """Default analysis has sensible values."""
        analysis = StrategyAnalysis()

        assert analysis.primary_strategy == StrategyType.UNKNOWN
        assert analysis.strategy_confidence == 0.0
        assert analysis.success is False
        assert analysis.effectiveness_score == 0.0

    def test_to_dict_structure(self):
        """to_dict returns expected structure."""
        analysis = StrategyAnalysis(
            primary_strategy=StrategyType.GREPPING,
            strategy_confidence=0.9,
            success=True,
            effectiveness_score=0.75,
        )

        d = analysis.to_dict()

        assert d["primary_strategy"] == "grepping"
        assert d["strategy_confidence"] == 0.9
        assert d["success"] is True
        assert d["effectiveness_score"] == 0.75
