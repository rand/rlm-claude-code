"""
Progressive trajectory disclosure for RLM operations.

Implements: SPEC-11.01-11.06 Progressive Disclosure in Trajectory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trajectory import TrajectoryEvent, TrajectoryEventType, TrajectoryRenderer


@dataclass
class CostAttribution:
    """Attribution of cost to a specific category."""

    name: str
    cost_usd: float
    total_cost_usd: float

    @property
    def percentage(self) -> float:
        """Calculate percentage of total cost."""
        if self.total_cost_usd == 0:
            return 0.0
        return (self.cost_usd / self.total_cost_usd) * 100


@dataclass
class CostBreakdown:
    """
    Breakdown of costs by category.

    Implements: SPEC-11.06
    """

    total_cost_usd: float
    by_model: list[CostAttribution] = field(default_factory=list)
    by_operation: list[CostAttribution] = field(default_factory=list)
    by_component: list[CostAttribution] = field(default_factory=list)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> CostBreakdown:
        """Create CostBreakdown from cost metadata dict."""
        total_cost = metadata.get("total_cost_usd", 0.0)

        by_model = [
            CostAttribution(name=name, cost_usd=cost, total_cost_usd=total_cost)
            for name, cost in metadata.get("costs_by_model", {}).items()
        ]

        by_operation = [
            CostAttribution(name=name, cost_usd=cost, total_cost_usd=total_cost)
            for name, cost in metadata.get("costs_by_operation", {}).items()
        ]

        by_component = [
            CostAttribution(name=name, cost_usd=cost, total_cost_usd=total_cost)
            for name, cost in metadata.get("costs_by_component", {}).items()
        ]

        return cls(
            total_cost_usd=total_cost,
            by_model=by_model,
            by_operation=by_operation,
            by_component=by_component,
        )


class ProgressiveTrajectory:
    """
    Progressive disclosure for trajectory events.

    Implements: SPEC-11.01-11.06

    Supports multiple rendering levels:
    - Summary: One-line progress summary
    - Overview: Key events without details
    - Detail: Full details for specific event
    - Cost breakdown: Detailed cost attribution
    """

    # Event types to include in overview (SPEC-11.04)
    OVERVIEW_EVENT_TYPES = {
        TrajectoryEventType.RLM_START,
        TrajectoryEventType.RECURSE_START,
        TrajectoryEventType.RECURSE_END,
        TrajectoryEventType.FINAL,
        TrajectoryEventType.ERROR,
    }

    def __init__(
        self,
        events: list[TrajectoryEvent],
        cost_metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize progressive trajectory.

        Args:
            events: List of trajectory events
            cost_metadata: Optional cost metadata for breakdown
        """
        self.events = events
        self.cost_metadata = cost_metadata
        self._renderer = TrajectoryRenderer(verbosity="normal", colors=False)

    def render_summary(self) -> str:
        """
        Render one-line progress summary.

        Implements: SPEC-11.02, SPEC-11.03

        Format: "RLM: {N} recursive calls, {key_finding}"

        Returns:
            One-line summary string
        """
        # Count recursive calls
        recursive_count = sum(1 for e in self.events if e.type == TrajectoryEventType.RECURSE_START)

        # Find key finding from final event
        key_finding = "completed"
        for event in reversed(self.events):
            if event.type == TrajectoryEventType.FINAL:
                # Extract key info from final content
                content = event.content
                # Try to extract meaningful part
                if ":" in content:
                    key_finding = content.split(":", 1)[1].strip()[:50]
                else:
                    key_finding = content[:50]
                break
            elif event.type == TrajectoryEventType.ERROR:
                key_finding = f"error: {event.content[:30]}"
                break

        return f"RLM: {recursive_count} recursive calls, {key_finding}"

    def render_overview(self) -> str:
        """
        Render key events without details.

        Implements: SPEC-11.02, SPEC-11.04

        Shows only: RECURSE boundaries, FINAL, ERROR events.

        Returns:
            Overview string with key events
        """
        lines = []

        for event in self.events:
            if event.type in self.OVERVIEW_EVENT_TYPES:
                rendered = self._render_overview_event(event)
                lines.append(rendered)

        return "\n".join(lines)

    def _render_overview_event(self, event: TrajectoryEvent) -> str:
        """Render a single event for overview."""
        indent = "  " * event.depth
        type_label = event.type.value.upper().replace("_", " ")

        # Truncate content for overview
        content = event.content
        if len(content) > 60:
            content = content[:57] + "..."

        return f"{indent}[{type_label}] {content}"

    def render_detail(self, event_index: int) -> str:
        """
        Render full details for specific event.

        Implements: SPEC-11.02

        Args:
            event_index: Index of event to render

        Returns:
            Detailed event string including metadata
        """
        if event_index < 0 or event_index >= len(self.events):
            return f"Event index {event_index} out of range"

        event = self.events[event_index]

        lines = [
            f"Event #{event_index}: {event.type.value}",
            f"  Depth: {event.depth}",
            f"  Content: {event.content}",
            f"  Timestamp: {event.timestamp}",
        ]

        if event.metadata:
            lines.append("  Metadata:")
            for key, value in event.metadata.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def render_cost_breakdown(self) -> str:
        """
        Render detailed cost attribution.

        Implements: SPEC-11.02, SPEC-11.06

        Returns:
            Cost breakdown string
        """
        if not self.cost_metadata:
            return "Cost breakdown unavailable (no cost metadata provided)"

        breakdown = CostBreakdown.from_metadata(self.cost_metadata)

        lines = [
            f"Total Cost: ${breakdown.total_cost_usd:.4f}",
            "",
            "By Model Tier:",
        ]

        for attr in sorted(breakdown.by_model, key=lambda x: -x.cost_usd):
            lines.append(f"  {attr.name}: ${attr.cost_usd:.4f} ({attr.percentage:.1f}%)")

        lines.append("")
        lines.append("By Operation Type:")

        for attr in sorted(breakdown.by_operation, key=lambda x: -x.cost_usd):
            lines.append(f"  {attr.name}: ${attr.cost_usd:.4f} ({attr.percentage:.1f}%)")

        lines.append("")
        lines.append("By Component:")

        for attr in sorted(breakdown.by_component, key=lambda x: -x.cost_usd):
            lines.append(f"  {attr.name}: ${attr.cost_usd:.4f} ({attr.percentage:.1f}%)")

        return "\n".join(lines)

    def render_full(self) -> str:
        """
        Render full trajectory with all events.

        Implements: SPEC-11.05

        Returns:
            Full trajectory string
        """
        lines = []

        for i, event in enumerate(self.events):
            lines.append(f"[{i}] {self._renderer.render_event(event)}")

        return "\n".join(lines)


# --- Convenience functions ---


def render_summary(events: list[TrajectoryEvent]) -> str:
    """
    Render one-line summary of trajectory.

    Args:
        events: List of trajectory events

    Returns:
        Summary string
    """
    trajectory = ProgressiveTrajectory(events)
    return trajectory.render_summary()


def render_overview(events: list[TrajectoryEvent]) -> str:
    """
    Render overview of key events.

    Args:
        events: List of trajectory events

    Returns:
        Overview string
    """
    trajectory = ProgressiveTrajectory(events)
    return trajectory.render_overview()


def render_detail(events: list[TrajectoryEvent], event_index: int) -> str:
    """
    Render detail for specific event.

    Args:
        events: List of trajectory events
        event_index: Index of event to render

    Returns:
        Detail string
    """
    trajectory = ProgressiveTrajectory(events)
    return trajectory.render_detail(event_index)


def render_cost_breakdown(
    events: list[TrajectoryEvent],
    cost_metadata: dict[str, Any],
) -> str:
    """
    Render cost breakdown.

    Args:
        events: List of trajectory events
        cost_metadata: Cost metadata dict

    Returns:
        Cost breakdown string
    """
    trajectory = ProgressiveTrajectory(events, cost_metadata=cost_metadata)
    return trajectory.render_cost_breakdown()


__all__ = [
    "CostAttribution",
    "CostBreakdown",
    "ProgressiveTrajectory",
    "render_cost_breakdown",
    "render_detail",
    "render_overview",
    "render_summary",
]
