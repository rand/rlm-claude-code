"""
Shared type definitions for RLM-Claude-Code.

Implements: Spec §3.1 Context Variable Schema
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


class MessageRole(str, Enum):
    """Role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """A single message in conversation history."""

    role: MessageRole
    content: str
    timestamp: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ToolOutput:
    """Result of a tool execution."""

    tool_name: str
    content: str
    exit_code: int | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SessionContext:
    """
    Complete session context for RLM processing.

    Implements: Spec §3.1 Context Variable Schema
    """

    messages: list[Message] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    tool_outputs: list[ToolOutput] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Estimate total tokens in context."""
        # Rough estimate: 4 chars per token
        chars = sum(len(m.content) for m in self.messages)
        chars += sum(len(c) for c in self.files.values())
        chars += sum(len(o.content) for o in self.tool_outputs)
        return chars // 4

    @property
    def active_modules(self) -> set:
        """Extract module names from cached files."""
        from pathlib import Path

        modules = set()
        for path in self.files:
            parts = Path(path).parts
            if parts:
                modules.add(parts[0])
        return modules


class TaskComplexitySignals(BaseModel):
    """
    Signals extracted from user prompt and context for complexity classification.

    Implements: Spec §6.3 Task Complexity-Based Activation
    """

    # Prompt analysis
    references_multiple_files: bool = False
    requires_cross_context_reasoning: bool = False
    involves_temporal_reasoning: bool = False
    asks_about_patterns: bool = False
    debugging_task: bool = False

    # Context analysis
    context_has_multiple_domains: bool = False
    recent_tool_outputs_large: bool = False
    conversation_has_state_changes: bool = False
    files_span_multiple_modules: bool = False

    # Historical signals
    previous_turn_was_confused: bool = False
    task_is_continuation: bool = False


class ActivationDecision(BaseModel):
    """Result of complexity classification."""

    should_activate: bool
    reason: str
    score: int = 0
    signals: TaskComplexitySignals | None = None


@dataclass
class ExecutionResult:
    """Result of REPL code execution."""

    success: bool
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


class VerificationResult(BaseModel):
    """Result of constraint verification."""

    safe: bool
    reason: str | None = None
    witness: dict[str, Any] | None = None
    conflicts: list[str] | None = None


@dataclass
class RecursiveCallResult:
    """
    Result of a recursive sub-call.

    Implements: Spec §4.2 Recursive Call Implementation
    """

    content: str
    depth: int
    model_used: str
    tokens_used: int = 0
    execution_time_ms: float = 0.0
    had_repl: bool = False
    child_results: list["RecursiveCallResult"] = field(default_factory=list)


# RLM Error Classes
# Implements: Spec §Error Handling


@dataclass
class DeferredOperation:
    """
    Represents an async operation that needs to be processed after sync REPL execution.

    Implements: Spec §4.2 Recursive Call Implementation (async/sync bridge)
    """

    operation_id: str
    operation_type: str  # "recursive_query", "summarize", "llm_batch"
    query: str
    context: Any = None
    spawn_repl: bool = False
    resolved: bool = False
    result: Any = None

    def __repr__(self) -> str:
        """Return placeholder string that can be detected in output."""
        return f"<<DEFERRED:{self.operation_id}>>"


@dataclass
class DeferredBatch:
    """
    A batch of parallel LLM calls to be processed together.

    Implements: Spec §4.2 Parallel Sub-Calls (llm_batch)
    """

    batch_id: str
    operations: list[DeferredOperation] = field(default_factory=list)
    resolved: bool = False
    results: list[Any] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<<BATCH:{self.batch_id}>>"


class RLMError(Exception):
    """Base class for RLM errors."""

    pass


class ContextTooLargeError(RLMError):
    """Context exceeds maximum size for RLM processing."""

    def __init__(self, size: int, max_size: int):
        self.size = size
        self.max_size = max_size
        super().__init__(f"Context size {size} exceeds maximum {max_size}")


class RecursionDepthError(RLMError):
    """Maximum recursion depth exceeded."""

    def __init__(self, depth: int, max_depth: int):
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(f"Recursion depth {depth} exceeds maximum {max_depth}")


class CostLimitError(RLMError):
    """Cost limit exceeded during RLM processing."""

    def __init__(self, tokens_used: int, limit: int):
        self.tokens_used = tokens_used
        self.limit = limit
        super().__init__(f"Token usage {tokens_used} exceeds limit {limit}")
