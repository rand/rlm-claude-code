"""
Trajectory events and rendering for RLM-Claude-Code.

Implements: Spec §6.6 Streaming Trajectory Visibility
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import rlm_core

# ============================================================================
# Typed Payload Schemas (SPEC-12.08)
# ============================================================================


@dataclass
class RLMStartPayload:
    """Payload for RLM_START events."""

    query: str
    context_tokens: int
    model: str | None = None
    depth_budget: int | None = None


@dataclass
class RecursePayload:
    """Payload for RECURSE_START/RECURSE_END events."""

    query: str
    depth: int
    parent_id: str | None = None
    spawn_repl: bool = True
    tokens_used: int | None = None
    execution_time_ms: float | None = None


@dataclass
class REPLExecPayload:
    """Payload for REPL_EXEC events."""

    code: str
    function_calls: list[str] = field(default_factory=list)
    code_length: int = 0

    def __post_init__(self) -> None:
        if self.code_length == 0:
            self.code_length = len(self.code)


@dataclass
class REPLResultPayload:
    """Payload for REPL_RESULT events."""

    result: str
    execution_time_ms: float = 0.0
    memory_used_bytes: int = 0
    functions_called: list[str] = field(default_factory=list)
    truncated: bool = False


@dataclass
class ReasoningPayload:
    """Payload for REASON events."""

    reasoning: str
    step_number: int | None = None
    total_steps: int | None = None


@dataclass
class ErrorPayload:
    """Payload for ERROR events."""

    error_type: str
    error_message: str
    recoverable: bool = True
    depth: int = 0
    context: str | None = None


@dataclass
class FinalPayload:
    """Payload for FINAL events."""

    answer: str
    confidence: float | None = None
    sources: list[str] = field(default_factory=list)
    tokens_used: int = 0
    total_time_ms: float = 0.0


@dataclass
class ToolUsePayload:
    """Payload for TOOL_USE events."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    execution_time_ms: float | None = None


@dataclass
class CostPayload:
    """Payload for COST_REPORT events."""

    total_cost: float
    input_tokens: int
    output_tokens: int
    model: str
    budget_remaining: float | None = None


@dataclass
class BudgetAlertPayload:
    """Payload for BUDGET_ALERT events."""

    alert_type: str  # "warning", "exceeded", "model_downgrade"
    budget_utilization: float
    original_model: str | None = None
    new_model: str | None = None
    reason: str | None = None


@dataclass
class VerificationPayload:
    """Payload for VERIFICATION events (SPEC-16.36)."""

    claims_total: int = 0
    claims_verified: int = 0
    claims_flagged: int = 0
    confidence: float = 0.0
    flagged_claim_ids: list[str] = field(default_factory=list)
    retry_count: int = 0


# Union type for all payloads
TrajectoryPayload = (
    RLMStartPayload
    | RecursePayload
    | REPLExecPayload
    | REPLResultPayload
    | ReasoningPayload
    | ErrorPayload
    | FinalPayload
    | ToolUsePayload
    | CostPayload
    | BudgetAlertPayload
    | VerificationPayload
)


# ============================================================================
# Core Event Types
# ============================================================================


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
    VERIFICATION = "verification"  # SPEC-16.22: Epistemic verification checkpoint

    # Convenience aliases for common usage patterns
    REPL = "repl_exec"  # Alias for REPL_EXEC (start of REPL operation)
    RECURSE = "recurse_start"  # Alias for RECURSE_START (start of recursive call)


@dataclass
class TrajectoryEvent:
    """
    A single event in the RLM trajectory.

    Implements: Spec §6.6 Streaming Trajectory Visibility

    Attributes:
        type: Event type from TrajectoryEventType enum
        depth: Recursion depth (0 = root, 1+ = sub-queries)
        content: Human-readable event description
        metadata: Optional unstructured data (legacy, prefer typed_payload)
        typed_payload: Strongly-typed payload for the event type (SPEC-12.08)
        timestamp: Unix timestamp (auto-generated if not provided)

    Example:
        >>> from src.trajectory import TrajectoryEvent, TrajectoryEventType
        >>> event = TrajectoryEvent(
        ...     type=TrajectoryEventType.RLM_START, depth=0, content="Starting RLM analysis"
        ... )
        >>> # With typed payload:
        >>> from src.trajectory import REPLExecPayload
        >>> event = TrajectoryEvent(
        ...     type=TrajectoryEventType.REPL_EXEC,
        ...     depth=1,
        ...     content="x = search(files, 'error')",
        ...     typed_payload=REPLExecPayload(
        ...         code="x = search(files, 'error')", function_calls=["search"]
        ...     ),
        ... )
    """

    type: TrajectoryEventType
    depth: int
    content: str
    metadata: dict[str, Any] | None = None
    typed_payload: TrajectoryPayload | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type.value,
            "depth": self.depth,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
        if self.typed_payload is not None:
            result["typed_payload"] = {
                "payload_type": type(self.typed_payload).__name__,
                **asdict(self.typed_payload),
            }
        return result


# ============================================================================
# rlm_core Delegation Helpers
# ============================================================================


def _map_event_type_to_core(event_type: TrajectoryEventType) -> Any:
    """Map Python TrajectoryEventType to rlm_core.TrajectoryEventType."""
    mapping = {
        TrajectoryEventType.RLM_START: rlm_core.TrajectoryEventType.RlmStart,
        TrajectoryEventType.ANALYZE: rlm_core.TrajectoryEventType.Analyze,
        TrajectoryEventType.REPL_EXEC: rlm_core.TrajectoryEventType.ReplExec,
        TrajectoryEventType.REPL_RESULT: rlm_core.TrajectoryEventType.ReplResult,
        TrajectoryEventType.REASON: rlm_core.TrajectoryEventType.Reason,
        TrajectoryEventType.RECURSE_START: rlm_core.TrajectoryEventType.RecurseStart,
        TrajectoryEventType.RECURSE_END: rlm_core.TrajectoryEventType.RecurseEnd,
        TrajectoryEventType.FINAL: rlm_core.TrajectoryEventType.Final,
        TrajectoryEventType.ERROR: rlm_core.TrajectoryEventType.Error,
        TrajectoryEventType.TOOL_USE: rlm_core.TrajectoryEventType.ToolUse,
        TrajectoryEventType.COST_REPORT: rlm_core.TrajectoryEventType.CostReport,
        TrajectoryEventType.BUDGET_ALERT: rlm_core.TrajectoryEventType.BudgetComputed,
        TrajectoryEventType.VERIFICATION: rlm_core.TrajectoryEventType.VerifyComplete,
    }
    return mapping.get(event_type, rlm_core.TrajectoryEventType.Analyze)


def _map_core_event_type_to_python(core_type: Any) -> TrajectoryEventType:
    """Map rlm_core.TrajectoryEventType to Python TrajectoryEventType."""
    # Build reverse mapping
    mapping = {
        rlm_core.TrajectoryEventType.RlmStart: TrajectoryEventType.RLM_START,
        rlm_core.TrajectoryEventType.Analyze: TrajectoryEventType.ANALYZE,
        rlm_core.TrajectoryEventType.ReplExec: TrajectoryEventType.REPL_EXEC,
        rlm_core.TrajectoryEventType.ReplResult: TrajectoryEventType.REPL_RESULT,
        rlm_core.TrajectoryEventType.Reason: TrajectoryEventType.REASON,
        rlm_core.TrajectoryEventType.RecurseStart: TrajectoryEventType.RECURSE_START,
        rlm_core.TrajectoryEventType.RecurseEnd: TrajectoryEventType.RECURSE_END,
        rlm_core.TrajectoryEventType.Final: TrajectoryEventType.FINAL,
        rlm_core.TrajectoryEventType.Error: TrajectoryEventType.ERROR,
        rlm_core.TrajectoryEventType.ToolUse: TrajectoryEventType.TOOL_USE,
        rlm_core.TrajectoryEventType.CostReport: TrajectoryEventType.COST_REPORT,
        rlm_core.TrajectoryEventType.BudgetComputed: TrajectoryEventType.BUDGET_ALERT,
        rlm_core.TrajectoryEventType.VerifyComplete: TrajectoryEventType.VERIFICATION,
    }
    return mapping.get(core_type, TrajectoryEventType.ANALYZE)


def _create_core_event(
    event_type: TrajectoryEventType,
    depth: int,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Create an rlm_core.TrajectoryEvent using factory methods.

    Returns the core event or None if creation fails.
    """
    try:
        # Use factory methods based on event type
        core_event: Any = None

        if event_type == TrajectoryEventType.RLM_START:
            core_event = rlm_core.TrajectoryEvent.rlm_start(content)
        elif event_type == TrajectoryEventType.RECURSE_START:
            core_event = rlm_core.TrajectoryEvent.recurse_start(depth, content)
        elif event_type == TrajectoryEventType.RECURSE_END:
            core_event = rlm_core.TrajectoryEvent.recurse_end(depth, content)
        elif event_type == TrajectoryEventType.REPL_EXEC:
            core_event = rlm_core.TrajectoryEvent.repl_exec(depth, content)
        elif event_type == TrajectoryEventType.REPL_RESULT:
            core_event = rlm_core.TrajectoryEvent.repl_result(depth, content)
        elif event_type == TrajectoryEventType.ANALYZE:
            core_event = rlm_core.TrajectoryEvent.analyze(content, depth=depth)
        elif event_type == TrajectoryEventType.REASON:
            core_event = rlm_core.TrajectoryEvent.reason(depth, content)
        elif event_type == TrajectoryEventType.ERROR:
            core_event = rlm_core.TrajectoryEvent.error(depth, content)
        elif event_type == TrajectoryEventType.FINAL:
            core_event = rlm_core.TrajectoryEvent.final_answer(content, depth=depth)
        else:
            # For event types without factory methods, use analyze as fallback
            core_event = rlm_core.TrajectoryEvent.analyze(content, depth=depth)

        # Add metadata if provided
        if core_event is not None and metadata:
            for key, value in metadata.items():
                core_event = core_event.with_metadata(key, str(value))

        return core_event

    except Exception:
        # Silently fall back to Python-only on any error
        return None


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
        TrajectoryEventType.VERIFICATION: "✓",
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
        TrajectoryEventType.VERIFICATION: "VERIFY",
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
        TrajectoryEventType.VERIFICATION: "\033[1;34m",  # Bold blue
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


class RichTrajectoryRenderer:
    """
    Rich-based trajectory renderer using RLMConsole.

    Implements: SPEC-13 Rich Output Formatting
    """

    def __init__(self, verbosity: str = "normal", colors: bool = True):
        """
        Initialize Rich renderer.

        Args:
            verbosity: "minimal" | "normal" | "verbose" | "debug"
            colors: Whether to use colors
        """
        from src.rich_output import OutputConfig, RLMConsole

        # Map old verbosity to new
        verbosity_map = {
            "minimal": "quiet",
            "normal": "normal",
            "verbose": "verbose",
            "debug": "debug",
        }
        mapped_verbosity = verbosity_map.get(verbosity, "normal")

        config = OutputConfig(verbosity=mapped_verbosity, colors=colors)  # type: ignore[arg-type]
        self.console = RLMConsole(config)
        self.verbosity = verbosity
        self.colors = colors

    def render_event(self, event: TrajectoryEvent) -> str:
        """
        Render event using Rich console.

        Returns empty string as Rich prints directly.
        """
        event_type = event.type
        depth = event.depth
        content = self._truncate_content(event.content, event_type)

        if event_type == TrajectoryEventType.RLM_START:
            self.console.emit_start(content, depth_budget=3)
        elif event_type == TrajectoryEventType.REPL_EXEC:
            # Parse function name from content
            func = "repl"
            args = content[:50]
            if "(" in content:
                func = content.split("(")[0].strip()
                args = content
            self.console.emit_repl(func, args, depth=depth)
        elif event_type == TrajectoryEventType.REPL_RESULT:
            self.console.emit_result(content, depth=depth)
        elif event_type == TrajectoryEventType.RECURSE_START:
            self.console.emit_recurse(content, depth=depth + 1)
        elif event_type == TrajectoryEventType.RECURSE_END:
            # Extract token count from content like "Returned (500 tokens, 100ms)"
            tokens = 0
            if "tokens" in content:
                try:
                    tokens = int(content.split("(")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            self.console.emit_complete(tokens_used=tokens, depth=depth)
        elif event_type == TrajectoryEventType.ERROR:
            self.console.emit_error(content, depth=depth)
        elif event_type == TrajectoryEventType.BUDGET_ALERT:
            self.console.emit_warning(content, depth=depth)
        elif event_type == TrajectoryEventType.FINAL:
            self.console.emit_result(content, depth=depth, is_last=True)
        elif event_type == TrajectoryEventType.COST_REPORT:
            # Parse tokens from content like "Cost: $0.01 (1000+500 tokens)"
            if "tokens" in content.lower():
                try:
                    parts = content.split("(")[1].split("+")
                    input_tokens = int(parts[0])
                    output_tokens = int(parts[1].split()[0])
                    self.console.emit_budget(input_tokens + output_tokens, 100000)
                except (IndexError, ValueError):
                    pass
        # Other event types use default rendering
        else:
            # Fall back to basic output for unsupported types
            self.console.console.print(f"  {content}")

        return ""  # Rich prints directly

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

    def __init__(self, renderer: TrajectoryRenderer | RichTrajectoryRenderer):
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
    When rlm_core is available, events are also tracked in the core trajectory
    system for unified observability.
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

        # Track core events when rlm_core is available
        self._core_events: list[Any] = []

    @property
    def usesrlm_core(self) -> bool:
        """Return True if rlm_core is being used for trajectory tracking."""
        return True

    @property
    def events(self) -> list[TrajectoryEvent]:
        """Get all emitted events."""
        return self.streaming_trajectory.events

    def emit_rlm_start(
        self,
        query: str,
        context_tokens: int,
        model: str | None = None,
        depth_budget: int | None = None,
    ) -> None:
        """Emit RLM start event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content=f"Activating RLM mode for: {query[:100]}",
            metadata={"query": query, "context_tokens": context_tokens},
            typed_payload=RLMStartPayload(
                query=query,
                context_tokens=context_tokens,
                model=model,
                depth_budget=depth_budget,
            ),
        )
        # Use _emit_sync for consistency and core event tracking
        self._emit_sync(event)

    def emit_recursive_start(
        self,
        depth: int,
        query: str,
        spawn_repl: bool,
        parent_id: str | None = None,
    ) -> None:
        """Emit recursive call start event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            depth=depth - 1,  # Parent depth
            content=f"Spawning sub-call: {query}",
            metadata={"spawn_repl": spawn_repl},
            typed_payload=RecursePayload(
                query=query,
                depth=depth,
                parent_id=parent_id,
                spawn_repl=spawn_repl,
            ),
        )
        self._emit_sync(event)

    def emit_recursive_complete(
        self,
        depth: int,
        tokens_used: int,
        execution_time_ms: float,
        query: str = "",
    ) -> None:
        """Emit recursive call completion event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_END,
            depth=depth - 1,
            content=f"Returned ({tokens_used} tokens, {execution_time_ms:.0f}ms)",
            metadata={"tokens_used": tokens_used, "execution_time_ms": execution_time_ms},
            typed_payload=RecursePayload(
                query=query,
                depth=depth,
                tokens_used=tokens_used,
                execution_time_ms=execution_time_ms,
            ),
        )
        self._emit_sync(event)

    def emit_recursive_error(self, depth: int, error: str, recoverable: bool = True) -> None:
        """Emit recursive call error event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.ERROR,
            depth=depth - 1,
            content=f"Recursive call failed: {error}",
            metadata={"error": error},
            typed_payload=ErrorPayload(
                error_type="RecursionError",
                error_message=error,
                recoverable=recoverable,
                depth=depth,
            ),
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

    def emit_rlm_loop_complete(
        self,
        depth: int,
        tokens_used: int,
        answer: str = "",
        confidence: float | None = None,
        total_time_ms: float = 0.0,
    ) -> None:
        """Emit RLM loop completion event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=depth,
            content=f"RLM loop complete ({tokens_used} tokens)",
            metadata={"tokens_used": tokens_used},
            typed_payload=FinalPayload(
                answer=answer[:500] if answer else "",  # Truncate for storage
                confidence=confidence,
                tokens_used=tokens_used,
                total_time_ms=total_time_ms,
            ),
        )
        self._emit_sync(event)

    def emit_repl_execution(
        self,
        depth: int,
        code: str,
        function_calls: list[str] | None = None,
    ) -> None:
        """Emit REPL code execution event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.REPL_EXEC,
            depth=depth,
            content=code,
            typed_payload=REPLExecPayload(
                code=code,
                function_calls=function_calls or [],
            ),
        )
        self._emit_sync(event)

    def emit_repl_result(
        self,
        depth: int,
        result: str,
        execution_time_ms: float = 0.0,
        memory_used_bytes: int = 0,
        functions_called: list[str] | None = None,
        truncated: bool = False,
    ) -> None:
        """Emit REPL result event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.REPL_RESULT,
            depth=depth,
            content=result,
            typed_payload=REPLResultPayload(
                result=result,
                execution_time_ms=execution_time_ms,
                memory_used_bytes=memory_used_bytes,
                functions_called=functions_called or [],
                truncated=truncated,
            ),
        )
        self._emit_sync(event)

    def emit_reasoning(
        self,
        depth: int,
        reasoning: str,
        step_number: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        """Emit reasoning step event."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.REASON,
            depth=depth,
            content=reasoning,
            typed_payload=ReasoningPayload(
                reasoning=reasoning,
                step_number=step_number,
                total_steps=total_steps,
            ),
        )
        self._emit_sync(event)

    def emit_tool_use(
        self,
        depth: int,
        tool: str,
        args: str | dict[str, Any],
        result: str | None = None,
        execution_time_ms: float | None = None,
    ) -> None:
        """Emit tool use event."""
        args_dict = args if isinstance(args, dict) else {"raw": args}
        args_str = args if isinstance(args, str) else str(args)
        event = TrajectoryEvent(
            type=TrajectoryEventType.TOOL_USE,
            depth=depth,
            content=f"{tool}: {args_str}",
            metadata={"tool": tool, "args": args_str},
            typed_payload=ToolUsePayload(
                tool=tool,
                args=args_dict,
                result=result,
                execution_time_ms=execution_time_ms,
            ),
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
            typed_payload=BudgetAlertPayload(
                alert_type="model_downgrade",
                budget_utilization=budget_utilization,
                original_model=original_model,
                new_model=new_model,
                reason=reason,
            ),
        )
        self._emit_sync(event)

    def emit_error(
        self,
        depth: int,
        error_type: str,
        error_message: str,
        recoverable: bool = True,
        context: str | None = None,
    ) -> None:
        """Emit a general error event with typed payload."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.ERROR,
            depth=depth,
            content=f"{error_type}: {error_message}",
            metadata={"error_type": error_type, "error_message": error_message},
            typed_payload=ErrorPayload(
                error_type=error_type,
                error_message=error_message,
                recoverable=recoverable,
                depth=depth,
                context=context,
            ),
        )
        self._emit_sync(event)

    def emit_cost_report(
        self,
        total_cost: float,
        input_tokens: int,
        output_tokens: int,
        model: str,
        budget_remaining: float | None = None,
    ) -> None:
        """Emit cost report event with typed payload."""
        event = TrajectoryEvent(
            type=TrajectoryEventType.COST_REPORT,
            depth=0,
            content=f"Cost: ${total_cost:.4f} ({input_tokens}+{output_tokens} tokens)",
            metadata={
                "total_cost": total_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            typed_payload=CostPayload(
                total_cost=total_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                budget_remaining=budget_remaining,
            ),
        )
        self._emit_sync(event)

    def emit_verification(
        self,
        depth: int,
        claims_total: int,
        claims_verified: int,
        claims_flagged: int,
        confidence: float,
        flagged_claim_ids: list[str] | None = None,
        retry_count: int = 0,
    ) -> None:
        """
        Emit verification checkpoint event.

        SPEC-16.36: Verification events in trajectory analysis.

        Args:
            depth: Current recursion depth
            claims_total: Total claims verified
            claims_verified: Claims that passed verification
            claims_flagged: Claims that failed verification
            confidence: Overall verification confidence
            flagged_claim_ids: IDs of flagged claims
            retry_count: Number of verification retries performed
        """
        event = TrajectoryEvent(
            type=TrajectoryEventType.VERIFICATION,
            depth=depth,
            content=f"Verified {claims_verified}/{claims_total} claims (confidence: {confidence:.0%})",
            metadata={
                "claims_total": claims_total,
                "claims_verified": claims_verified,
                "claims_flagged": claims_flagged,
                "confidence": confidence,
            },
            typed_payload=VerificationPayload(
                claims_total=claims_total,
                claims_verified=claims_verified,
                claims_flagged=claims_flagged,
                confidence=confidence,
                flagged_claim_ids=flagged_claim_ids or [],
                retry_count=retry_count,
            ),
        )
        self._emit_sync(event)

    def _emit_sync(self, event: TrajectoryEvent) -> None:
        """Emit event synchronously (for non-async contexts)."""
        # Just append to events list directly for sync contexts
        self.streaming_trajectory.events.append(event)

        # Also create and track core event
        core_event = _create_core_event(
            event.type,
            event.depth,
            event.content,
            event.metadata,
        )
        if core_event is not None:
            self._core_events.append(core_event)

    def get_full_trajectory(self) -> list[TrajectoryEvent]:
        """Get complete trajectory."""
        return self.streaming_trajectory.get_full_trajectory()

    def get_core_events(self) -> list[Any]:
        """
        Get rlm_core trajectory events.

        Returns empty list if rlm_core is not available.
        """
        return self._core_events.copy()

    def export_json(self, path: str | None = None) -> str:
        """Export trajectory as JSON."""
        return self.streaming_trajectory.export_json(path)

    def export_core_json(self, path: str | None = None) -> str | None:
        """
        Export trajectory using rlm_core serialization.

        Returns None if rlm_core is not available.
        Uses the core TrajectoryEvent.to_json() for each event.
        """
        if not self._core_events:
            return None

        events_json = [event.to_json() for event in self._core_events]
        data = "[\n" + ",\n".join(events_json) + "\n]"

        if path:
            with open(path, "w") as f:
                f.write(data)
        return data


__all__ = [
    # Core types
    "TrajectoryEventType",
    "TrajectoryEvent",
    "TrajectoryRenderer",
    "RichTrajectoryRenderer",
    "StreamingTrajectory",
    "TrajectoryStream",
    # Typed payloads (SPEC-12.08)
    "TrajectoryPayload",
    "RLMStartPayload",
    "RecursePayload",
    "REPLExecPayload",
    "REPLResultPayload",
    "ReasoningPayload",
    "ErrorPayload",
    "FinalPayload",
    "ToolUsePayload",
    "CostPayload",
    "BudgetAlertPayload",
    "VerificationPayload",  # SPEC-16.36
]
