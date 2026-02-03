"""
Unit tests for visualization module.

Implements: Spec §8.1 Phase 4 - Visualization tests
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from src.trajectory import TrajectoryEvent, TrajectoryEventType
from src.visualization import (
    ExportFormat,
    TimelineEntry,
    TrajectoryReplayer,
    TrajectoryStats,
    TrajectoryVisualizer,
    visualize_trajectory,
)


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_all_formats_exist(self):
        """All expected formats exist."""
        expected = ["json", "html", "markdown"]
        actual = [ef.value for ef in ExportFormat]
        assert set(expected) == set(actual)


class TestTimelineEntry:
    """Tests for TimelineEntry dataclass."""

    def test_create_entry(self):
        """Can create timeline entry."""
        entry = TimelineEntry(
            timestamp=time.time(),
            event_type=TrajectoryEventType.REPL_EXEC,
            label="REPL Execute",
            content="print('hello')",
            depth=0,
        )

        assert entry.event_type == TrajectoryEventType.REPL_EXEC
        assert entry.depth == 0

    def test_formatted_time(self):
        """Formats time correctly."""
        entry = TimelineEntry(
            timestamp=1704067200.123,  # 2024-01-01 00:00:00.123
            event_type=TrajectoryEventType.RLM_START,
            label="Start",
            content="",
            depth=0,
        )

        formatted = entry.formatted_time
        assert ":" in formatted
        assert "." in formatted  # Includes milliseconds

    def test_indent(self):
        """Calculates indent correctly."""
        entry0 = TimelineEntry(
            timestamp=time.time(),
            event_type=TrajectoryEventType.RLM_START,
            label="Start",
            content="",
            depth=0,
        )
        entry2 = TimelineEntry(
            timestamp=time.time(),
            event_type=TrajectoryEventType.RECURSE_START,
            label="Recursive",
            content="",
            depth=2,
        )

        assert entry0.indent == ""
        assert entry2.indent == "    "  # 2 * 2 spaces


class TestTrajectoryStats:
    """Tests for TrajectoryStats dataclass."""

    def test_create_stats(self):
        """Can create stats."""
        stats = TrajectoryStats(
            total_events=10,
            total_duration_ms=5000,
            max_depth=2,
            events_by_type={"rlm_start": 1, "repl_execute": 5},
            tokens_used=1000,
            repl_executions=5,
            recursive_calls=2,
        )

        assert stats.total_events == 10
        assert stats.max_depth == 2


class TestTrajectoryVisualizer:
    """Tests for TrajectoryVisualizer class."""

    @pytest.fixture
    def sample_events(self):
        """Create sample events."""
        base_time = time.time()
        return [
            TrajectoryEvent(
                type=TrajectoryEventType.RLM_START,
                timestamp=base_time,
                depth=0,
                content="Starting RLM",
                metadata={"query": "test query"},
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.ANALYZE,
                timestamp=base_time + 0.1,
                depth=0,
                content="Analyzing query",
                metadata={},
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                timestamp=base_time + 0.2,
                depth=0,
                content="peek('file.py')",
                metadata={"tokens": 100},
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.RECURSE_START,
                timestamp=base_time + 0.3,
                depth=1,
                content="Sub-query",
                metadata={"tokens": 200},
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.RECURSE_END,
                timestamp=base_time + 0.5,
                depth=1,
                content="Response text",
                metadata={"tokens": 150},
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                timestamp=base_time + 0.6,
                depth=0,
                content="Complete",
                metadata={},
            ),
        ]

    @pytest.fixture
    def visualizer(self, sample_events):
        """Create visualizer with sample events."""
        return TrajectoryVisualizer(sample_events)

    def test_create_visualizer(self):
        """Can create visualizer."""
        viz = TrajectoryVisualizer()
        assert viz.events == []

    def test_create_with_events(self, visualizer, sample_events):
        """Can create with events."""
        assert len(visualizer.events) == len(sample_events)

    def test_build_timeline(self, visualizer):
        """Builds timeline correctly."""
        timeline = visualizer.build_timeline()

        assert len(timeline) == 6
        assert timeline[0].event_type == TrajectoryEventType.RLM_START
        assert timeline[-1].event_type == TrajectoryEventType.FINAL

    def test_timeline_caching(self, visualizer):
        """Timeline is cached."""
        timeline1 = visualizer.build_timeline()
        timeline2 = visualizer.build_timeline()

        assert timeline1 is timeline2

    def test_get_stats(self, visualizer):
        """Calculates stats correctly."""
        stats = visualizer.get_stats()

        assert stats.total_events == 6
        assert stats.max_depth == 1
        assert stats.repl_executions == 1
        assert stats.recursive_calls == 1
        assert stats.tokens_used == 450  # 100 + 200 + 150

    def test_get_stats_empty(self):
        """Stats for empty visualizer."""
        viz = TrajectoryVisualizer()
        stats = viz.get_stats()

        assert stats.total_events == 0
        assert stats.max_depth == 0

    def test_export_json(self, visualizer):
        """Exports to JSON."""
        content = visualizer.export(ExportFormat.JSON)

        data = json.loads(content)
        assert "stats" in data
        assert "events" in data
        assert len(data["events"]) == 6

    def test_export_html(self, visualizer):
        """Exports to HTML."""
        content = visualizer.export(ExportFormat.HTML)

        assert "<!DOCTYPE html>" in content
        assert "RLM Trajectory Viewer" in content
        assert "timeline" in content

    def test_export_markdown(self, visualizer):
        """Exports to Markdown."""
        content = visualizer.export(ExportFormat.MARKDOWN)

        assert "# RLM Trajectory" in content
        assert "## Statistics" in content
        assert "## Timeline" in content

    def test_export_to_file(self, visualizer):
        """Exports to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectory.html"
            visualizer.export(ExportFormat.HTML, path)

            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_load_from_json_file(self, sample_events):
        """Loads from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectory.json"

            # Write events
            data = [
                {
                    "event_type": e.type.value,
                    "timestamp": e.timestamp,
                    "depth": e.depth,
                    "content": e.content,
                    "metadata": e.metadata,
                }
                for e in sample_events
            ]
            path.write_text(json.dumps(data))

            # Load
            viz = TrajectoryVisualizer()
            viz.load_from_file(path)

            assert len(viz.events) == len(sample_events)

    def test_load_from_jsonl_file(self, sample_events):
        """Loads from JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectory.jsonl"

            # Write events as JSONL
            lines = [
                json.dumps(
                    {
                        "event_type": e.type.value,
                        "timestamp": e.timestamp,
                        "depth": e.depth,
                        "content": e.content,
                        "metadata": e.metadata,
                    }
                )
                for e in sample_events
            ]
            path.write_text("\n".join(lines))

            # Load
            viz = TrajectoryVisualizer()
            viz.load_from_file(path)

            assert len(viz.events) == len(sample_events)


class TestTrajectoryReplayer:
    """Tests for TrajectoryReplayer class."""

    @pytest.fixture
    def sample_events(self):
        """Create sample events."""
        base_time = time.time()
        return [
            TrajectoryEvent(
                type=TrajectoryEventType.RLM_START,
                timestamp=base_time,
                depth=0,
                content="Start",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                timestamp=base_time + 0.1,
                depth=0,
                content="Execute 1",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.REPL_EXEC,
                timestamp=base_time + 0.2,
                depth=0,
                content="Execute 2",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                timestamp=base_time + 0.3,
                depth=0,
                content="End",
            ),
        ]

    @pytest.fixture
    def replayer(self, sample_events):
        """Create replayer with sample events."""
        viz = TrajectoryVisualizer(sample_events)
        return TrajectoryReplayer(viz)

    def test_total_steps(self, replayer):
        """Returns correct total steps."""
        assert replayer.total_steps == 4

    def test_current_position(self, replayer):
        """Starts at position 0."""
        assert replayer.current_position == 0

    def test_current_event(self, replayer):
        """Returns current event."""
        event = replayer.current_event
        assert event is not None
        assert event.type == TrajectoryEventType.RLM_START

    def test_step_forward(self, replayer):
        """Steps forward."""
        event = replayer.step_forward()

        assert event is not None
        assert replayer.current_position == 1
        assert event.type == TrajectoryEventType.REPL_EXEC

    def test_step_forward_at_end(self, replayer):
        """Returns None at end."""
        # Move to end
        for _ in range(4):
            replayer.step_forward()

        event = replayer.step_forward()
        assert event is None

    def test_step_backward(self, replayer):
        """Steps backward."""
        replayer.step_forward()
        replayer.step_forward()

        event = replayer.step_backward()

        assert event is not None
        assert replayer.current_position == 1

    def test_step_backward_at_start(self, replayer):
        """Returns None at start."""
        event = replayer.step_backward()
        assert event is None

    def test_seek(self, replayer):
        """Seeks to position."""
        event = replayer.seek(2)

        assert event is not None
        assert replayer.current_position == 2

    def test_seek_invalid(self, replayer):
        """Returns None for invalid position."""
        event = replayer.seek(100)
        assert event is None

    def test_seek_to_event_type_forward(self, replayer):
        """Seeks to event type forward."""
        event = replayer.seek_to_event_type(TrajectoryEventType.FINAL, forward=True)

        assert event is not None
        assert event.type == TrajectoryEventType.FINAL
        assert replayer.current_position == 3

    def test_seek_to_event_type_backward(self, replayer):
        """Seeks to event type backward."""
        replayer.seek(3)  # Go to end

        event = replayer.seek_to_event_type(TrajectoryEventType.REPL_EXEC, forward=False)

        assert event is not None
        assert event.type == TrajectoryEventType.REPL_EXEC

    def test_seek_to_event_type_not_found(self, replayer):
        """Returns None if type not found."""
        event = replayer.seek_to_event_type(TrajectoryEventType.ERROR, forward=True)
        assert event is None

    def test_progress(self, replayer):
        """Calculates progress correctly."""
        assert replayer.progress == 0.0

        replayer.seek(2)
        assert replayer.progress == 0.5

    def test_reset(self, replayer):
        """Resets to beginning."""
        replayer.seek(3)
        replayer.reset()

        assert replayer.current_position == 0

    def test_get_events_until_now(self, replayer):
        """Gets events up to current position."""
        replayer.seek(2)
        events = replayer.get_events_until_now()

        assert len(events) == 3


class TestVisualizeTrajectory:
    """Tests for visualize_trajectory convenience function."""

    @pytest.fixture
    def sample_events(self):
        """Create sample events."""
        return [
            TrajectoryEvent(
                type=TrajectoryEventType.RLM_START,
                timestamp=time.time(),
                depth=0,
                content="Start",
            ),
            TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                timestamp=time.time() + 0.1,
                depth=0,
                content="End",
            ),
        ]

    def test_visualize_with_events(self, sample_events):
        """Visualizes from events."""
        content = visualize_trajectory(events=sample_events, output_format=ExportFormat.JSON)

        data = json.loads(content)
        assert len(data["events"]) == 2

    def test_visualize_from_file(self, sample_events):
        """Visualizes from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            output_path = Path(tmpdir) / "output.html"

            # Write input
            data = [
                {
                    "event_type": e.type.value,
                    "timestamp": e.timestamp,
                    "depth": e.depth,
                    "content": e.content,
                    "metadata": e.metadata,
                }
                for e in sample_events
            ]
            input_path.write_text(json.dumps(data))

            # Visualize
            content = visualize_trajectory(
                path=input_path,
                output_format=ExportFormat.HTML,
                output_path=output_path,
            )

            assert "<!DOCTYPE html>" in content
            assert output_path.exists()

    def test_visualize_markdown(self, sample_events):
        """Visualizes as markdown."""
        content = visualize_trajectory(events=sample_events, output_format=ExportFormat.MARKDOWN)

        assert "# RLM Trajectory" in content
