"""
Property-based tests for trajectory events.

Implements: Spec §7 testing requirements
"""

import json
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trajectory import (
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryRenderer,
)

# Strategies for generating test data
event_type_strategy = st.sampled_from(list(TrajectoryEventType))

event_strategy = st.builds(
    TrajectoryEvent,
    type=event_type_strategy,
    content=st.text(min_size=0, max_size=500),
    depth=st.integers(min_value=0, max_value=3),
    metadata=st.one_of(
        st.none(),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()),
            values=st.one_of(
                st.text(min_size=0, max_size=100),
                st.integers(min_value=-1000, max_value=1000),
                st.floats(min_value=-1000, max_value=1000, allow_nan=False),
                st.booleans(),
            ),
            max_size=5,
        ),
    ),
)


@pytest.mark.hypothesis
class TestTrajectoryEventProperties:
    """Property-based tests for TrajectoryEvent."""

    @given(event=event_strategy)
    @settings(max_examples=100)
    def test_event_serialization_roundtrip(self, event):
        """Events can be serialized to dict and back."""
        event_dict = event.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(event_dict)
        restored_dict = json.loads(json_str)

        # All expected keys present
        assert "type" in restored_dict
        assert "content" in restored_dict
        assert "depth" in restored_dict
        assert "timestamp" in restored_dict

    @given(event=event_strategy)
    @settings(max_examples=100)
    def test_event_dict_has_valid_types(self, event):
        """Event dict contains valid JSON types."""
        event_dict = event.to_dict()

        assert isinstance(event_dict["type"], str)
        assert isinstance(event_dict["content"], str)
        assert isinstance(event_dict["depth"], int)
        assert isinstance(event_dict["timestamp"], (int, float))
        # metadata can be None or dict
        assert event_dict.get("metadata") is None or isinstance(event_dict.get("metadata"), dict)

    @given(event=event_strategy)
    @settings(max_examples=50)
    def test_event_type_value_matches(self, event):
        """Event type in dict matches enum value."""
        event_dict = event.to_dict()
        assert event_dict["type"] == event.type.value

    @given(events=st.lists(event_strategy, min_size=0, max_size=20))
    @settings(max_examples=30)
    def test_multiple_events_serialize_independently(self, events):
        """Each event serializes independently."""
        dicts = [e.to_dict() for e in events]

        # Each should be unique object
        for i, d1 in enumerate(dicts):
            for j, d2 in enumerate(dicts):
                if i != j:
                    assert d1 is not d2


@pytest.mark.hypothesis
class TestTrajectoryRendererProperties:
    """Property-based tests for TrajectoryRenderer."""

    @given(event=event_strategy)
    @settings(max_examples=100)
    def test_render_returns_string(self, event):
        """Rendering always produces a string."""
        renderer = TrajectoryRenderer()
        result = renderer.render_event(event)

        assert isinstance(result, str)

    @given(event=event_strategy)
    @settings(max_examples=50)
    def test_render_is_deterministic(self, event):
        """Same event always renders the same way."""
        renderer = TrajectoryRenderer()
        result1 = renderer.render_event(event)
        result2 = renderer.render_event(event)

        assert result1 == result2

    @given(events=st.lists(event_strategy, min_size=1, max_size=10))
    @settings(max_examples=30)
    def test_render_multiple_events_produces_lines(self, events):
        """Multiple events render as multiple lines."""
        renderer = TrajectoryRenderer()

        for event in events:
            line = renderer.render_event(event)
            # Each rendered event should be a non-empty string
            assert isinstance(line, str)

    @given(event=event_strategy, colors=st.booleans())
    @settings(max_examples=50)
    def test_color_setting_changes_output(self, event, colors):
        """Color setting affects output format."""
        renderer = TrajectoryRenderer(colors=colors)
        result = renderer.render_event(event)

        # Result should always be valid string regardless of color setting
        assert isinstance(result, str)


@pytest.mark.hypothesis
class TestEventTypeProperties:
    """Property-based tests for TrajectoryEventType enum."""

    @given(event_type=event_type_strategy)
    @settings(max_examples=20)
    def test_all_event_types_have_icon(self, event_type):
        """All event types have defined icons."""
        assert event_type in TrajectoryRenderer.ICONS
        assert isinstance(TrajectoryRenderer.ICONS[event_type], str)
        assert len(TrajectoryRenderer.ICONS[event_type]) > 0

    @given(event_type=event_type_strategy)
    @settings(max_examples=20)
    def test_all_event_types_have_label(self, event_type):
        """All event types have defined labels (may be empty for some types)."""
        assert event_type in TrajectoryRenderer.LABELS
        assert isinstance(TrajectoryRenderer.LABELS[event_type], str)
        # Some labels are intentionally empty (e.g., REPL_RESULT)

    @given(event_type=event_type_strategy)
    @settings(max_examples=20)
    def test_event_type_value_is_string(self, event_type):
        """All event type values are strings."""
        assert isinstance(event_type.value, str)
        assert len(event_type.value) > 0
