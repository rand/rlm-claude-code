"""
Tests for progressive trajectory disclosure.

@trace SPEC-11.01-11.06
"""

from __future__ import annotations

from typing import Any

import pytest

from src.progressive_trajectory import (
    CostAttribution,
    CostBreakdown,
    ProgressiveTrajectory,
    render_cost_breakdown,
    render_detail,
    render_overview,
    render_summary,
)
from src.trajectory import TrajectoryEvent, TrajectoryEventType

# --- Test fixtures ---


def create_sample_events() -> list[TrajectoryEvent]:
    """Create sample trajectory events for testing."""
    return [
        TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content="Activating RLM mode for: analyze codebase",
            metadata={"query": "analyze codebase", "context_tokens": 5000},
        ),
        TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=0,
            content="Starting analysis with claude-opus-4-5",
            metadata={"model": "claude-opus-4-5"},
        ),
        TrajectoryEvent(
            type=TrajectoryEventType.REPL_EXEC,
            depth=0,
            content="peek(context, 0, 100)",
            metadata={},
        ),
        TrajectoryEvent(
            type=TrajectoryEventType.REPL_RESULT,
            depth=0,
            content="[100 lines of code]",
            metadata={},
        ),
        TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            depth=0,
            content="Spawning sub-call: summarize file1.py",
            metadata={"spawn_repl": False},
        ),
        TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_END,
            depth=0,
            content="Returned (500 tokens, 234ms)",
            metadata={"tokens_used": 500, "execution_time_ms": 234},
        ),
        TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            depth=0,
            content="Spawning sub-call: summarize file2.py",
            metadata={"spawn_repl": False},
        ),
        TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_END,
            depth=0,
            content="Returned (600 tokens, 312ms)",
            metadata={"tokens_used": 600, "execution_time_ms": 312},
        ),
        TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=0,
            content="Analysis complete: found 3 potential issues",
            metadata={"tokens_used": 2500},
        ),
    ]


def create_events_with_error() -> list[TrajectoryEvent]:
    """Create events including an error."""
    events = create_sample_events()[:5]
    events.append(
        TrajectoryEvent(
            type=TrajectoryEventType.ERROR,
            depth=0,
            content="Recursive call failed: timeout",
            metadata={"error": "timeout"},
        )
    )
    return events


def create_cost_metadata() -> dict[str, Any]:
    """Create sample cost metadata."""
    return {
        "total_cost_usd": 0.25,
        "costs_by_model": {
            "claude-opus-4-5": 0.15,
            "claude-haiku-4-5": 0.10,
        },
        "costs_by_operation": {
            "recursive_call": 0.12,
            "tool_execution": 0.08,
            "synthesis": 0.05,
        },
        "costs_by_component": {
            "orchestrator": 0.10,
            "repl": 0.08,
            "memory": 0.07,
        },
    }


# --- SPEC-11.01: Progressive trajectory rendering ---


class TestProgressiveRendering:
    """Tests for progressive trajectory rendering."""

    def test_progressive_trajectory_creation(self) -> None:
        """
        @trace SPEC-11.01
        Should create ProgressiveTrajectory from events.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        assert trajectory is not None
        assert len(trajectory.events) == len(events)

    def test_supports_multiple_render_modes(self) -> None:
        """
        @trace SPEC-11.01
        Should support summary, overview, and detail rendering.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        summary = trajectory.render_summary()
        overview = trajectory.render_overview()
        detail = trajectory.render_detail(0)  # First event

        assert summary is not None
        assert overview is not None
        assert detail is not None


# --- SPEC-11.02: ProgressiveTrajectory methods ---


class TestProgressiveTrajectoryMethods:
    """Tests for ProgressiveTrajectory method requirements."""

    def test_render_summary_returns_string(self) -> None:
        """
        @trace SPEC-11.02
        render_summary() should return a string.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        summary = trajectory.render_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_render_overview_returns_string(self) -> None:
        """
        @trace SPEC-11.02
        render_overview() should return a string.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        overview = trajectory.render_overview()

        assert isinstance(overview, str)

    def test_render_detail_for_event(self) -> None:
        """
        @trace SPEC-11.02
        render_detail(event_id) should return full details for specific event.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        detail = trajectory.render_detail(4)  # RECURSE_START event

        assert isinstance(detail, str)
        assert "sub-call" in detail.lower() or "recurse" in detail.lower()

    def test_render_cost_breakdown_returns_string(self) -> None:
        """
        @trace SPEC-11.02
        render_cost_breakdown() should return detailed cost attribution.
        """
        events = create_sample_events()
        cost_metadata = create_cost_metadata()
        trajectory = ProgressiveTrajectory(events, cost_metadata=cost_metadata)

        breakdown = trajectory.render_cost_breakdown()

        assert isinstance(breakdown, str)
        assert "cost" in breakdown.lower() or "$" in breakdown


# --- SPEC-11.03: Summary format ---


class TestSummaryFormat:
    """Tests for summary format requirements."""

    def test_summary_includes_recursive_call_count(self) -> None:
        """
        @trace SPEC-11.03
        Summary should include number of recursive calls.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        summary = trajectory.render_summary()

        # Should mention recursive calls (we have 2 in sample)
        assert "2" in summary or "recursive" in summary.lower()

    def test_summary_format_matches_spec(self) -> None:
        """
        @trace SPEC-11.03
        Summary format should be: "RLM: {N} recursive calls, {key_finding}"
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        summary = trajectory.render_summary()

        # Should start with RLM:
        assert summary.startswith("RLM:")

    def test_summary_includes_key_finding(self) -> None:
        """
        @trace SPEC-11.03
        Summary should include key finding from final event.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        summary = trajectory.render_summary()

        # Should include some info from final event
        # "Analysis complete: found 3 potential issues"
        assert (
            "complete" in summary.lower()
            or "issues" in summary.lower()
            or "found" in summary.lower()
        )


# --- SPEC-11.04: Overview filtering ---


class TestOverviewFiltering:
    """Tests for overview event filtering."""

    def test_overview_shows_recurse_boundaries(self) -> None:
        """
        @trace SPEC-11.04
        Overview should show RECURSE_START and RECURSE_END events.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        overview = trajectory.render_overview()

        # Should include recursive call info
        assert "recurse" in overview.lower() or "sub-call" in overview.lower()

    def test_overview_shows_final_event(self) -> None:
        """
        @trace SPEC-11.04
        Overview should show FINAL events.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        overview = trajectory.render_overview()

        # Should include final event content
        assert "complete" in overview.lower() or "final" in overview.lower()

    def test_overview_shows_error_events(self) -> None:
        """
        @trace SPEC-11.04
        Overview should show ERROR events.
        """
        events = create_events_with_error()
        trajectory = ProgressiveTrajectory(events)

        overview = trajectory.render_overview()

        # Should include error info
        assert "error" in overview.lower() or "failed" in overview.lower()

    def test_overview_excludes_repl_details(self) -> None:
        """
        @trace SPEC-11.04
        Overview should NOT include REPL_EXEC and REPL_RESULT details.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        overview = trajectory.render_overview()

        # Should not include REPL code
        assert "peek(" not in overview


# --- SPEC-11.05: Expandable rendering ---


class TestExpandableRendering:
    """Tests for expandable rendering support."""

    def test_detail_includes_metadata(self) -> None:
        """
        @trace SPEC-11.05
        Detail rendering should include event metadata.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        # Get detail for RECURSE_END event (has tokens_used metadata)
        detail = trajectory.render_detail(5)

        # Should include metadata like tokens_used
        assert "500" in detail or "tokens" in detail.lower()

    def test_can_expand_to_full_event_list(self) -> None:
        """
        @trace SPEC-11.05
        Should support expanding to full event list.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)

        full = trajectory.render_full()

        # Full rendering should be longer than overview
        overview = trajectory.render_overview()
        assert len(full) >= len(overview)


# --- SPEC-11.06: Cost breakdown ---


class TestCostBreakdown:
    """Tests for cost breakdown by category."""

    def test_cost_by_model_tier(self) -> None:
        """
        @trace SPEC-11.06
        Cost breakdown should attribute costs to model tier.
        """
        events = create_sample_events()
        cost_metadata = create_cost_metadata()
        trajectory = ProgressiveTrajectory(events, cost_metadata=cost_metadata)

        breakdown = trajectory.render_cost_breakdown()

        # Should mention models
        assert "opus" in breakdown.lower() or "haiku" in breakdown.lower()

    def test_cost_by_operation_type(self) -> None:
        """
        @trace SPEC-11.06
        Cost breakdown should attribute costs to operation type.
        """
        events = create_sample_events()
        cost_metadata = create_cost_metadata()
        trajectory = ProgressiveTrajectory(events, cost_metadata=cost_metadata)

        breakdown = trajectory.render_cost_breakdown()

        # Should mention operation types
        assert (
            "recursive" in breakdown.lower()
            or "tool" in breakdown.lower()
            or "synthesis" in breakdown.lower()
        )

    def test_cost_by_component(self) -> None:
        """
        @trace SPEC-11.06
        Cost breakdown should attribute costs to component.
        """
        events = create_sample_events()
        cost_metadata = create_cost_metadata()
        trajectory = ProgressiveTrajectory(events, cost_metadata=cost_metadata)

        breakdown = trajectory.render_cost_breakdown()

        # Should mention components
        assert (
            "orchestrator" in breakdown.lower()
            or "repl" in breakdown.lower()
            or "memory" in breakdown.lower()
        )

    def test_cost_breakdown_without_metadata(self) -> None:
        """
        Cost breakdown should handle missing metadata gracefully.
        """
        events = create_sample_events()
        trajectory = ProgressiveTrajectory(events)  # No cost metadata

        breakdown = trajectory.render_cost_breakdown()

        assert "no cost" in breakdown.lower() or "unavailable" in breakdown.lower()


# --- Convenience function tests ---


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_render_summary_function(self) -> None:
        """
        Convenience function should work correctly.
        """
        events = create_sample_events()

        summary = render_summary(events)

        assert isinstance(summary, str)
        assert summary.startswith("RLM:")

    def test_render_overview_function(self) -> None:
        """
        Convenience function should work correctly.
        """
        events = create_sample_events()

        overview = render_overview(events)

        assert isinstance(overview, str)

    def test_render_detail_function(self) -> None:
        """
        Convenience function should work correctly.
        """
        events = create_sample_events()

        detail = render_detail(events, 0)

        assert isinstance(detail, str)

    def test_render_cost_breakdown_function(self) -> None:
        """
        Convenience function should work correctly.
        """
        events = create_sample_events()
        cost_metadata = create_cost_metadata()

        breakdown = render_cost_breakdown(events, cost_metadata)

        assert isinstance(breakdown, str)


# --- CostBreakdown dataclass tests ---


class TestCostBreakdownDataclass:
    """Tests for CostBreakdown dataclass."""

    def test_cost_breakdown_creation(self) -> None:
        """
        CostBreakdown should be created from metadata.
        """
        cost_metadata = create_cost_metadata()

        breakdown = CostBreakdown.from_metadata(cost_metadata)

        assert breakdown.total_cost_usd == 0.25
        assert len(breakdown.by_model) > 0
        assert len(breakdown.by_operation) > 0
        assert len(breakdown.by_component) > 0

    def test_cost_attribution_percentage(self) -> None:
        """
        CostAttribution should calculate percentages.
        """
        attribution = CostAttribution(
            name="opus",
            cost_usd=0.15,
            total_cost_usd=0.25,
        )

        assert attribution.percentage == pytest.approx(60.0)
