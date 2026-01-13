"""
Trajectory events and rendering for RLM-Claude-Code.

Implements: Spec §6.6 Streaming Trajectory Visibility
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrajectoryEventType(Enum):
    """Types of trajectory events."""

    RLM_START = "rlm_start"
    ANALYZE = "analyze"
    REPL_EXEC = "repl_exec"
    REPL_RESULT = "repl_result"
    REASON = "reason"
    RECURSE_START = "recurse_start"
    RECURSE_END = "recurse_end"
    FINAL = "final"
    ERROR = "error"
    TOOL_USE = "tool_use"
    COST_REPORT = "cost_report"
    BUDGET_ALERT = "budget_alert"


@dataclass
class TrajectoryEvent:
    """
    A single event in the RLM trajectory.

    Implements: Spec §6.6 Streaming Trajectory Visibility

    Attributes:
        type: Event type from TrajectoryEventType enum
        depth: Recursion depth (0 = root, 1+ = sub-queries)
        content: Human-readable event description
        metadata: Optional structured data for the event
        timestamp: Unix timestamp (auto-generated if not provided)

    Example:
        >>> from src.trajectory import TrajectoryEvent, TrajectoryEventType
        >>> event = TrajectoryEvent(
        ...     type=TrajectoryEventType.RLM_START,
        ...     depth=0,
        ...     content="Starting RLM analysis"
        ... )
        >>> # Or with metadata:
        >>> event = TrajectoryEvent(
        ...     type=TrajectoryEventType.REPL_EXEC,
        ...     depth=1,
        ...     content="x = search(files, 'error')",
        ...     metadata={"code_length": 25}
        ... )
    """

    type: TrajectoryEventType
    depth: int
    content: str
    metadata: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "depth": self.depth,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class TrajectoryRenderer:
    """
    Renders trajectory events to terminal with streaming support.

    Implements: Spec §6.6 Streaming Trajectory Visibility
    """

    ICONS = {
        TrajectoryEventType.RLM_START: "◆",
        TrajectoryEventType.ANALYZE: "▶",
        TrajectoryEventType.REPL_EXEC: "▶",
        TrajectoryEventType.REPL_RESULT: "└─",
        TrajectoryEventType.REASON: "▶",
        TrajectoryEventType.RECURSE_START: "▶",
        TrajectoryEventType.RECURSE_END: "◀",
        TrajectoryEventType.FINAL: "▶",
        TrajectoryEventType.ERROR: "✗",
        TrajectoryEventType.TOOL_USE: "⚙",
        TrajectoryEventType.COST_REPORT: "💰",
        TrajectoryEventType.BUDGET_ALERT: "⚠",
    }

    LABELS = {
        TrajectoryEventType.RLM_START: "RLM",
        TrajectoryEventType.ANALYZE: "ANALYZE",
        TrajectoryEventType.REPL_EXEC: "REPL",
        TrajectoryEventType.REPL_RESULT: "",
        TrajectoryEventType.REASON: "REASON",
        TrajectoryEventType.RECURSE_START: "RECURSE",
        TrajectoryEventType.RECURSE_END: "RETURN",
        TrajectoryEventType.FINAL: "FINAL",
        TrajectoryEventType.ERROR: "ERROR",
        TrajectoryEventType.TOOL_USE: "TOOL",
        TrajectoryEventType.COST_REPORT: "COST",
        TrajectoryEventType.BUDGET_ALERT: "BUDGET",
    }

    COLORS = {
        TrajectoryEventType.RLM_START: "\033[1;36m",  # Bold cyan
        TrajectoryEventType.ANALYZE: "\033[34m",  # Blue
        TrajectoryEventType.REPL_EXEC: "\033[33m",  # Yellow
        TrajectoryEventType.REPL_RESULT: "\033[2m",  # Dim
        TrajectoryEventType.REASON: "\033[32m",  # Green
        TrajectoryEventType.RECURSE_START: "\033[35m",  # Magenta
        TrajectoryEventType.RECURSE_END: "\033[35m",  # Magenta
        TrajectoryEventType.FINAL: "\033[1;32m",  # Bold green
        TrajectoryEventType.ERROR: "\033[1;31m",  # Bold red
        TrajectoryEventType.TOOL_USE: "\033[36m",  # Cyan
        TrajectoryEventType.COST_REPORT: "\033[1;33m",  # Bold yellow
        TrajectoryEventType.BUDGET_ALERT: "\033[1;33m",  # Bold yellow
    }

    def __init__(self, verbosity: str = "normal", colors: bool = True):
        """
        Initialize renderer.

        Args:
            verbosity: "minimal" | "normal" | "verbose" | "debug"
            colors: Whether to use ANSI colors
        """
        self.verbosity = verbosity
        self.colors = colors
        self.reset = "\033[0m" if colors else ""

    def render_event(self, event: TrajectoryEvent) -> str:
        """Render a single event to terminal string."""
        indent = "│   " * event.depth
        icon = self.ICONS[event.type]
        label = self.LABELS[event.type]

        # Depth indicator for recursive calls
        depth_indicator = ""
        if event.type == TrajectoryEventType.RECURSE_START:
            depth_indicator = f" │ depth={event.depth + 1} │"

        # Truncate content based on verbosity
        content = self._truncate_content(event.content, event.type)

        # Color coding
        color = self.COLORS.get(event.type, "") if self.colors else ""

        if event.type == TrajectoryEventType.REPL_RESULT:
            return f"{indent}  {color}{icon} [{content}]{self.reset}"
        elif label:
            return f"{indent}{color}{icon} {label:7}{self.reset}{depth_indicator} │ {content}"
        else:
            return f"{indent}{content}"

    def _truncate_content(self, content: str, event_type: TrajectoryEventType) -> str:
        """Truncate content based on verbosity."""
        limits = {
            "minimal": {"default": 60, "repl_result": 40, "reason": 80},
            "normal": {"default": 120, "repl_result": 80, "reason": 200},
            "verbose": {"default": 300, "repl_result": 200, "reason": 500},
            "debug": {"default": 1000, "repl_result": 1000, "reason": 1000},
        }

        key = (
            "repl_result"
            if event_type == TrajectoryEventType.REPL_RESULT
            else "reason"
            if event_type == TrajectoryEventType.REASON
            else "default"
        )
        limit = limits[self.verbosity][key]

        if len(content) <= limit:
            return content
        return content[: limit - 3] + "..."


class StreamingTrajectory:
    """
    Manages streaming trajectory output during RLM execution.

    Implements: Spec §6.6 Streaming Trajectory Visibility
    """

    def __init__(self, renderer: TrajectoryRenderer):
        self.renderer = renderer
        self.events: list[TrajectoryEvent] = []
        self.subscribers: list[asyncio.Queue] = []

    async def emit(self, event: TrajectoryEvent) -> None:
        """Emit event to all subscribers."""
        self.events.append(event)
        rendered = self.renderer.render_event(event)

        for queue in self.subscribers:
            await queue.put(rendered)

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to trajectory stream."""
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    async def stream(self) -> AsyncIterator[str]:
        """Async iterator for streaming output."""
        queue = self.subscribe()
        try:
            while True:
                line = await queue.get()
                yield line
        except asyncio.CancelledError:
            self.subscribers.remove(queue)
            raise

    def get_full_trajectory(self) -> list[TrajectoryEvent]:
        """Get complete trajectory for logging/replay."""
        return self.events.copy()

    def export_json(self, path: str | None = None) -> str:
        """Export trajectory as JSON."""
        data = json.dumps([e.to_dict() for e in self.events], indent=2)
        if path:
            with open(path, "w") as f:
                f.write(data)
        return data


class TrajectoryStream:
    """
    Interface for trajectory event emission used by RecursiveREPL.

    Implements: Spec §6.6 Streaming Trajectory Visibility

    This provides the methods called by recursive_handler for trajectory events.
    """

    def __init__(
        self,
        verbosity: str = "normal",
        colors: bool = True,
        streaming: bool = True,
    ):
        """
        Initialize trajectory stream.

        Args:
            verbosity: Output verbosity level
            colors: Whether to use ANSI colors
            streaming: Whether to stream output (vs batch)
        """
        self.renderer = TrajectoryRenderer(verbosity=verbosity, colors=colors)
        self.streaming_trajectory = StreamingTrajectory(self.renderer)
        self.streaming = streaming

    @property
    def events(self) -> list[TrajectoryEvent]:
        """Get all emitted events."""
        return self.streaming_trajectory.events

    def emit_rlm_start(self, query: str, context_tokens: int) -> None:
        """Emit RLM start event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content=f"Activating RLM mode for: {query[:100]}",
            metadata={"query": query, "context_tokens": context_tokens},
        )
        asyncio.get_event_loop().run_until_complete(
            self.streaming_trajectory.emit(event)
        ) if self.streaming else self.streaming_trajectory.events.append(event)

    def emit_recursive_start(
        self,
        depth: int,
        query: str,
        spawn_repl: bool,
    ) -> None:
        """Emit recursive call start event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            depth=depth - 1,  # Parent depth
            content=f"Spawning sub-call: {query}",
            metadata={"spawn_repl": spawn_repl},
        )
        self._emit_sync(event)

    def emit_recursive_complete(
        self,
        depth: int,
        tokens_used: int,
        execution_time_ms: float,
    ) -> None:
        """Emit recursive call completion event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_END,
            depth=depth - 1,
            content=f"Returned ({tokens_used} tokens, {execution_time_ms:.0f}ms)",
            metadata={"tokens_used": tokens_used, "execution_time_ms": execution_time_ms},
        )
        self._emit_sync(event)

    def emit_recursive_error(self, depth: int, error: str) -> None:
        """Emit recursive call error event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.ERROR,
            depth=depth - 1,
            content=f"Recursive call failed: {error}",
            metadata={"error": error},
        )
        self._emit_sync(event)

    def emit_rlm_loop_start(self, depth: int, model: str) -> None:
        """Emit RLM loop start event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=depth,
            content=f"Starting RLM loop with {model}",
            metadata={"model": model},
        )
        self._emit_sync(event)

    def emit_rlm_loop_complete(self, depth: int, tokens_used: int) -> None:
        """Emit RLM loop completion event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=depth,
            content=f"RLM loop complete ({tokens_used} tokens)",
            metadata={"tokens_used": tokens_used},
        )
        self._emit_sync(event)

    def emit_repl_execution(self, depth: int, code: str) -> None:
        """Emit REPL code execution event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.REPL_EXEC,
            depth=depth,
            content=code,
        )
        self._emit_sync(event)

    def emit_repl_result(self, depth: int, result: str) -> None:
        """Emit REPL result event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.REPL_RESULT,
            depth=depth,
            content=result,
        )
        self._emit_sync(event)

    def emit_reasoning(self, depth: int, reasoning: str) -> None:
        """Emit reasoning step event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.REASON,
            depth=depth,
            content=reasoning,
        )
        self._emit_sync(event)

    def emit_tool_use(self, depth: int, tool: str, args: str) -> None:
        """Emit tool use event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.TOOL_USE,
            depth=depth,
            content=f"{tool}: {args}",
            metadata={"tool": tool, "args": args},
        )
        self._emit_sync(event)

    def emit_model_downgrade(
        self,
        original_model: str,
        new_model: str,
        reason: str,
        budget_utilization: float,
    ) -> None:
        """
        Emit model downgrade event from adaptive depth budgeting.

        Feature 3e0.4 - Adaptive Depth Budgeting
        """
        event = TrajectoryEvent(
            type=TrajectoryEventType.BUDGET_ALERT,
            depth=0,
            content=f"Model downgrade: {original_model} → {new_model} ({reason})",
            metadata={
                "original_model": original_model,
                "new_model": new_model,
                "reason": reason,
                "budget_utilization": budget_utilization,
            },
        )
        self._emit_sync(event)

    def _emit_sync(self, event: TrajectoryEvent) -> None:
        """Emit event synchronously (for non-async contexts)."""
        # Just append to events list directly for sync contexts
        self.streaming_trajectory.events.append(event)

    def get_full_trajectory(self) -> list[TrajectoryEvent]:
        """Get complete trajectory."""
        return self.streaming_trajectory.get_full_trajectory()

    def export_json(self, path: str | None = None) -> str:
        """Export trajectory as JSON."""
        return self.streaming_trajectory.export_json(path)


__all__ = [
    "TrajectoryEventType",
    "TrajectoryEvent",
    "TrajectoryRenderer",
    "StreamingTrajectory",
    "TrajectoryStream",
]
