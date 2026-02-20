"""
Unit tests for deferred-operation bridge behavior in orchestrator core.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.orchestrator.core import RLMOrchestrator
from src.repl_environment import RLMEnvironment
from src.trajectory import TrajectoryEvent, TrajectoryEventType
from src.types import Message, MessageRole, SessionContext


class _StubClient:
    """Deterministic recursive-query stub with configurable transient failures."""

    def __init__(self, transient_failures: dict[str, int] | None = None):
        self.transient_failures = transient_failures or {}
        self.calls: dict[str, int] = {}

    async def recursive_query(self, query: str, context: Any) -> str:
        key = f"{query}|{context}"
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.calls[key] <= self.transient_failures.get(key, 0):
            raise RuntimeError(f"transient-{self.calls[key]}")
        return f"resolved:{query}:{context}"


class _TrajectoryRecorder:
    """Minimal trajectory sink used for testing event emission."""

    def __init__(self):
        self.events: list[TrajectoryEvent] = []

    async def emit(self, event: TrajectoryEvent) -> None:
        self.events.append(event)


@pytest.fixture
def basic_context() -> SessionContext:
    return SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="hello"),
            Message(role=MessageRole.ASSISTANT, content="hi"),
        ],
        files={"main.py": "print('hello')"},
        working_memory={},
    )


@pytest.mark.asyncio
async def test_deferred_bridge_resolves_individual_and_batch_ops(basic_context: SessionContext) -> None:
    orchestrator = RLMOrchestrator()
    repl = RLMEnvironment(basic_context, use_restricted=False)
    client = _StubClient()
    trajectory = _TrajectoryRecorder()

    op = repl._recursive_query("q1", "c1")
    batch = repl._llm_batch([("q2", "c2"), ("q3", "c3")])

    events = [
        event
        async for event in orchestrator._process_deferred_operations(
            repl, client, depth=0, trajectory=trajectory
        )
    ]

    assert [event.type for event in events] == [
        TrajectoryEventType.RECURSE_START,
        TrajectoryEventType.RECURSE_END,
    ]
    assert repl.globals["working_memory"][op.operation_id] == "resolved:q1:c1"
    assert repl.globals["working_memory"][batch.batch_id] == [
        "resolved:q2:c2",
        "resolved:q3:c3",
    ]

    end_metadata = events[-1].metadata or {}
    assert end_metadata["total_operations"] == 3
    assert end_metadata["succeeded"] == 3
    assert end_metadata["failed"] == 0
    assert end_metadata["retried"] == 0


@pytest.mark.asyncio
async def test_deferred_bridge_retries_transient_failures_deterministically(
    basic_context: SessionContext,
) -> None:
    orchestrator = RLMOrchestrator()
    repl = RLMEnvironment(basic_context, use_restricted=False)
    client = _StubClient(transient_failures={"retry|ctx": 1})
    trajectory = _TrajectoryRecorder()

    op = repl._recursive_query("retry", "ctx")
    events = [
        event
        async for event in orchestrator._process_deferred_operations(
            repl, client, depth=0, trajectory=trajectory
        )
    ]

    assert client.calls["retry|ctx"] == 2
    assert repl.globals["working_memory"][op.operation_id] == "resolved:retry:ctx"

    end_metadata = events[-1].metadata or {}
    assert end_metadata["failed"] == 0
    assert end_metadata["retried"] == 1
    op_telemetry = end_metadata["operations"][0]
    assert op_telemetry["status"] == "success"
    assert op_telemetry["attempts"] == 2


@pytest.mark.asyncio
async def test_deferred_bridge_surfaces_terminal_errors_with_telemetry(
    basic_context: SessionContext,
) -> None:
    orchestrator = RLMOrchestrator()
    repl = RLMEnvironment(basic_context, use_restricted=False)
    client = _StubClient(transient_failures={"boom|ctx": 4})
    trajectory = _TrajectoryRecorder()

    op = repl._recursive_query("boom", "ctx")
    events = [
        event
        async for event in orchestrator._process_deferred_operations(
            repl, client, depth=0, trajectory=trajectory
        )
    ]

    assert client.calls["boom|ctx"] == 2  # max_attempts in bridge
    assert str(repl.globals["working_memory"][op.operation_id]).startswith("[Error:")

    end_metadata = events[-1].metadata or {}
    assert end_metadata["failed"] == 1
    assert end_metadata["retried"] == 1
    op_telemetry = end_metadata["operations"][0]
    assert op_telemetry["status"] == "error"
    assert op_telemetry["attempts"] == 2


def test_submit_final_answer_prefers_answer_field() -> None:
    result = RLMOrchestrator._final_answer_from_submit({"outputs": {"answer": "done", "x": 1}})
    assert result == "done"


def test_submit_final_answer_serializes_object_without_answer() -> None:
    result = RLMOrchestrator._final_answer_from_submit({"outputs": {"x": 1}})
    assert "\"x\": 1" in result
