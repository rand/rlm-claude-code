"""
Unit tests for trajectory module.

Implements: Spec §6.6 tests
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trajectory import (
    StreamingTrajectory,
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryRenderer,
    TrajectoryStream,
)


class TestTrajectoryEventType:
    """Tests for TrajectoryEventType enum."""

    def test_all_event_types_exist(self):
        """All expected event types are defined."""
        expected = [
            "RLM_START",
            "ANALYZE",
            "REPL_EXEC",
            "REPL_RESULT",
            "REASON",
            "RECURSE_START",
            "RECURSE_END",
            "FINAL",
            "ERROR",
            "TOOL_USE",
        ]
        for name in expected:
            assert hasattr(TrajectoryEventType, name)

    def test_event_values(self):
        """Event types have expected string values."""
        assert TrajectoryEventType.RLM_START.value == "rlm_start"
        assert TrajectoryEventType.RECURSE_START.value == "recurse_start"
        assert TrajectoryEventType.ERROR.value == "error"


class TestTrajectoryEvent:
    """Tests for TrajectoryEvent dataclass."""

    def test_create_basic_event(self):
        """Can create basic event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=0,
            content="Analyzing context",
        )

        assert event.type == TrajectoryEventType.ANALYZE
        assert event.depth == 0
        assert event.content == "Analyzing context"
        assert event.metadata is None
        assert event.timestamp > 0

    def test_create_event_with_metadata(self):
        """Can create event with metadata."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            depth=1,
            content="Spawning sub-call",
            metadata={"query": "test query", "spawn_repl": True},
        )

        assert event.metadata == {"query": "test query", "spawn_repl": True}

    def test_to_dict(self):
        """Can convert to dictionary."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=0,
            content="Final answer",
            metadata={"tokens": 100},
        )

        d = event.to_dict()

        assert d["type"] == "final"
        assert d["depth"] == 0
        assert d["content"] == "Final answer"
        assert d["metadata"] == {"tokens": 100}
        assert "timestamp" in d


class TestTrajectoryRenderer:
    """Tests for TrajectoryRenderer."""

    def test_init_with_defaults(self):
        """Can initialize with defaults."""
        renderer = TrajectoryRenderer()

        assert renderer.verbosity == "normal"
        assert renderer.colors is True
        assert renderer.reset == "\033[0m"

    def test_init_without_colors(self):
        """Can initialize without colors."""
        renderer = TrajectoryRenderer(colors=False)

        assert renderer.colors is False
        assert renderer.reset == ""

    def test_render_rlm_start_event(self):
        """Renders RLM start event."""
        renderer = TrajectoryRenderer(colors=False)
        event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content="Starting RLM",
        )

        output = renderer.render_event(event)

        assert "RLM" in output
        assert "Starting RLM" in output
        assert "◆" in output

    def test_render_recursive_event_with_depth(self):
        """Renders recursive event with depth indicator."""
        renderer = TrajectoryRenderer(colors=False)
        event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            depth=1,
            content="Sub-call",
        )

        output = renderer.render_event(event)

        assert "RECURSE" in output
        assert "depth=2" in output  # depth+1 shown

    def test_render_with_indentation(self):
        """Events are indented based on depth."""
        renderer = TrajectoryRenderer(colors=False)
        event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=2,
            content="Deep analysis",
        )

        output = renderer.render_event(event)

        assert "│   │   " in output  # Two levels of indentation

    def test_render_error_event(self):
        """Renders error events."""
        renderer = TrajectoryRenderer(colors=False)
        event = TrajectoryEvent(
            type=TrajectoryEventType.ERROR,
            depth=0,
            content="Something went wrong",
        )

        output = renderer.render_event(event)

        assert "ERROR" in output
        assert "✗" in output

    def test_render_repl_result_event(self):
        """Renders REPL result events."""
        renderer = TrajectoryRenderer(colors=False)
        event = TrajectoryEvent(
            type=TrajectoryEventType.REPL_RESULT,
            depth=0,
            content="42",
        )

        output = renderer.render_event(event)

        assert "└─" in output
        assert "[42]" in output

    def test_truncate_content_minimal(self):
        """Minimal verbosity truncates heavily."""
        renderer = TrajectoryRenderer(verbosity="minimal", colors=False)
        long_content = "x" * 100

        truncated = renderer._truncate_content(long_content, TrajectoryEventType.ANALYZE)

        assert len(truncated) <= 60
        assert truncated.endswith("...")

    def test_truncate_content_verbose(self):
        """Verbose mode allows longer content."""
        renderer = TrajectoryRenderer(verbosity="verbose", colors=False)
        medium_content = "x" * 200

        truncated = renderer._truncate_content(medium_content, TrajectoryEventType.ANALYZE)

        # Should not be truncated at 200 chars in verbose mode
        assert len(truncated) == 200

    def test_truncate_content_debug(self):
        """Debug mode allows very long content."""
        renderer = TrajectoryRenderer(verbosity="debug", colors=False)
        long_content = "x" * 500

        truncated = renderer._truncate_content(long_content, TrajectoryEventType.ANALYZE)

        assert len(truncated) == 500

    def test_all_event_types_have_icons(self):
        """All event types have icons defined."""
        for event_type in TrajectoryEventType:
            assert event_type in TrajectoryRenderer.ICONS

    def test_all_event_types_have_colors(self):
        """All event types have colors defined."""
        for event_type in TrajectoryEventType:
            assert event_type in TrajectoryRenderer.COLORS


class TestStreamingTrajectory:
    """Tests for StreamingTrajectory."""

    def test_init(self):
        """Can initialize streaming trajectory."""
        renderer = TrajectoryRenderer()
        trajectory = StreamingTrajectory(renderer)

        assert trajectory.events == []
        assert trajectory.subscribers == []

    @pytest.mark.asyncio
    async def test_emit_adds_to_events(self):
        """Emit adds event to list."""
        renderer = TrajectoryRenderer()
        trajectory = StreamingTrajectory(renderer)
        event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=0,
            content="Test",
        )

        await trajectory.emit(event)

        assert len(trajectory.events) == 1
        assert trajectory.events[0] is event

    @pytest.mark.asyncio
    async def test_emit_notifies_subscribers(self):
        """Emit notifies all subscribers."""
        renderer = TrajectoryRenderer(colors=False)
        trajectory = StreamingTrajectory(renderer)
        queue = trajectory.subscribe()

        event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=0,
            content="Test event",
        )
        await trajectory.emit(event)

        # Check subscriber received rendered output
        rendered = await queue.get()
        assert "Test event" in rendered

    def test_get_full_trajectory(self):
        """Can get copy of full trajectory."""
        renderer = TrajectoryRenderer()
        trajectory = StreamingTrajectory(renderer)
        event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=0,
            content="Done",
        )
        trajectory.events.append(event)

        result = trajectory.get_full_trajectory()

        assert len(result) == 1
        # Should be a copy
        assert result is not trajectory.events

    def test_export_json(self):
        """Can export trajectory as JSON."""
        renderer = TrajectoryRenderer()
        trajectory = StreamingTrajectory(renderer)
        event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content="Start",
            metadata={"query": "test"},
        )
        trajectory.events.append(event)

        json_str = trajectory.export_json()
        data = json.loads(json_str)

        assert len(data) == 1
        assert data[0]["type"] == "rlm_start"
        assert data[0]["content"] == "Start"
        assert data[0]["metadata"]["query"] == "test"

    def test_export_json_to_file(self):
        """Can export trajectory to file."""
        renderer = TrajectoryRenderer()
        trajectory = StreamingTrajectory(renderer)
        event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=0,
            content="Complete",
        )
        trajectory.events.append(event)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            trajectory.export_json(path)

            with open(path) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["content"] == "Complete"
        finally:
            Path(path).unlink()


class TestTrajectoryStream:
    """Tests for TrajectoryStream interface."""

    def test_init_with_defaults(self):
        """Can initialize with defaults."""
        stream = TrajectoryStream()

        assert stream.streaming is True
        assert stream.events == []

    def test_init_with_options(self):
        """Can initialize with custom options."""
        stream = TrajectoryStream(
            verbosity="debug",
            colors=False,
            streaming=False,
        )

        assert stream.renderer.verbosity == "debug"
        assert stream.renderer.colors is False
        assert stream.streaming is False

    def test_emit_recursive_start(self):
        """Can emit recursive start event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_recursive_start(
            depth=1,
            query="Analyze this code",
            spawn_repl=True,
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.RECURSE_START
        assert event.depth == 0  # depth-1
        assert "Analyze this code" in event.content
        assert event.metadata["spawn_repl"] is True

    def test_emit_recursive_complete(self):
        """Can emit recursive complete event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_recursive_complete(
            depth=1,
            tokens_used=500,
            execution_time_ms=150.5,
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.RECURSE_END
        assert "500 tokens" in event.content
        assert "150ms" in event.content

    def test_emit_recursive_error(self):
        """Can emit recursive error event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_recursive_error(
            depth=2,
            error="Max depth exceeded",
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.ERROR
        assert "Max depth exceeded" in event.content

    def test_emit_rlm_loop_start(self):
        """Can emit RLM loop start event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_rlm_loop_start(
            depth=0,
            model="claude-opus-4",
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.ANALYZE
        assert "claude-opus-4" in event.content

    def test_emit_rlm_loop_complete(self):
        """Can emit RLM loop complete event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_rlm_loop_complete(
            depth=0,
            tokens_used=1000,
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.FINAL
        assert "1000 tokens" in event.content

    def test_emit_repl_execution(self):
        """Can emit REPL execution event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_repl_execution(
            depth=1,
            code="len(files)",
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.REPL_EXEC
        assert event.content == "len(files)"

    def test_emit_repl_result(self):
        """Can emit REPL result event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_repl_result(
            depth=1,
            result="42",
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.REPL_RESULT
        assert event.content == "42"

    def test_emit_reasoning(self):
        """Can emit reasoning event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_reasoning(
            depth=0,
            reasoning="The error is in line 42",
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.REASON
        assert "line 42" in event.content

    def test_emit_tool_use(self):
        """Can emit tool use event."""
        stream = TrajectoryStream(streaming=False)

        stream.emit_tool_use(
            depth=0,
            tool="bash",
            args="npm test",
        )

        assert len(stream.events) == 1
        event = stream.events[0]
        assert event.type == TrajectoryEventType.TOOL_USE
        assert "bash" in event.content
        assert "npm test" in event.content

    def test_get_full_trajectory(self):
        """Can get full trajectory."""
        stream = TrajectoryStream(streaming=False)
        stream.emit_recursive_start(depth=1, query="test", spawn_repl=False)
        stream.emit_recursive_complete(depth=1, tokens_used=100, execution_time_ms=50)

        trajectory = stream.get_full_trajectory()

        assert len(trajectory) == 2

    def test_export_json(self):
        """Can export as JSON."""
        stream = TrajectoryStream(streaming=False)
        stream.emit_reasoning(depth=0, reasoning="Test reasoning")

        json_str = stream.export_json()
        data = json.loads(json_str)

        assert len(data) == 1
        assert data[0]["type"] == "reason"
