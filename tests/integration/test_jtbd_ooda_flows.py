"""Empirical JTBD/OODA flow validation across the seven primary user jobs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.auto_activation import AutoActivator
from src.complexity_classifier import extract_complexity_signals
from src.orchestration_schema import ExecutionStrategy, OrchestrationContext
from src.orchestrator import create_orchestration_plan
from src.types import Message, MessageRole, SessionContext, ToolOutput


@dataclass(frozen=True)
class OODAScenario:
    name: str
    query: str
    context: SessionContext
    expected_activate: bool
    expected_strategy: ExecutionStrategy | None
    expected_signals: tuple[str, ...]
    expected_hint_token: str | None


def _scenario_context(
    *,
    files: dict[str, str] | None = None,
    tool_outputs: list[ToolOutput] | None = None,
    messages: list[Message] | None = None,
) -> SessionContext:
    return SessionContext(
        messages=messages or [],
        files=files or {},
        tool_outputs=tool_outputs or [],
        working_memory={},
    )


SCENARIOS: tuple[OODAScenario, ...] = (
    OODAScenario(
        name="jtbd-1-debug-multi-layer",
        query=(
            "Intermittent stack trace: why does auth.py fail when api.py calls it after retries?"
        ),
        context=_scenario_context(
            files={
                "src/auth.py": "def authenticate(): raise RuntimeError('boom')",
                "src/api.py": "from auth import authenticate",
                "src/retry.py": "def retry(callable_obj): return callable_obj()",
            },
            tool_outputs=[ToolOutput(tool_name="Bash", content="Traceback: auth failed")],
        ),
        expected_activate=True,
        expected_strategy=ExecutionStrategy.RECURSIVE_DEBUG,
        expected_signals=("debugging_task", "requires_cross_context_reasoning"),
        expected_hint_token="map_reduce()",
    ),
    OODAScenario(
        name="jtbd-2-codebase-discovery",
        query="How does the authentication system work across API and DB modules?",
        context=_scenario_context(
            files={
                "src/api/routes.py": "def route(): pass",
                "src/auth/service.py": "def validate(): pass",
                "src/db/models.py": "class User: pass",
            }
        ),
        expected_activate=True,
        expected_strategy=ExecutionStrategy.DISCOVERY,
        expected_signals=("architecture_analysis", "references_multiple_files"),
        expected_hint_token="find_relevant()",
    ),
    OODAScenario(
        name="jtbd-3-comprehensive-change",
        query=(
            "Update all usages of LegacyAuth in auth.py and api.py and ensure consistent handling."
        ),
        context=_scenario_context(
            files={
                "src/auth/legacy.py": "class LegacyAuth: pass",
                "src/api/handler.py": "LegacyAuth()",
                "src/tests/test_auth.py": "assert LegacyAuth is not None",
            }
        ),
        expected_activate=True,
        expected_strategy=ExecutionStrategy.EXHAUSTIVE_SEARCH,
        expected_signals=("requires_exhaustive_search", "references_multiple_files"),
        expected_hint_token="llm_batch()",
    ),
    OODAScenario(
        name="jtbd-4-architectural-decision",
        query=(
            "Explain the architecture trade-offs: should we migrate auth from monolith to microservices?"
        ),
        context=_scenario_context(
            files={
                "src/auth/service.py": "class AuthService: pass",
                "docs/adr/auth.md": "Current auth design notes",
            }
        ),
        expected_activate=True,
        expected_strategy=ExecutionStrategy.ARCHITECTURE,
        expected_signals=("architecture_analysis",),
        expected_hint_token="tradeoffs",
    ),
    OODAScenario(
        name="jtbd-5-security-completeness",
        query="Find all security vulnerabilities and make sure every edge case is reviewed.",
        context=_scenario_context(
            files={
                "src/api/auth.py": "def login(user, pwd): pass",
                "src/db/query.py": "def run(sql): return sql",
                "src/web/forms.py": "def render(): pass",
            }
        ),
        expected_activate=True,
        expected_strategy=ExecutionStrategy.MAP_REDUCE,
        expected_signals=("requires_exhaustive_search", "security_review_task"),
        expected_hint_token="map_reduce()",
    ),
    OODAScenario(
        name="jtbd-6-session-continuation",
        query="continue with the same task and pick up where we left off",
        context=_scenario_context(
            messages=[
                Message(role=MessageRole.USER, content="Fix the issue"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Actually I was wrong about the root cause, let me retry.",
                ),
            ]
        ),
        expected_activate=True,
        expected_strategy=ExecutionStrategy.CONTINUATION,
        expected_signals=("task_is_continuation", "previous_turn_was_confused"),
        expected_hint_token="memory_query()",
    ),
    OODAScenario(
        name="jtbd-7-simple-fast-path",
        query="ok",
        context=_scenario_context(),
        expected_activate=False,
        expected_strategy=None,
        expected_signals=(),
        expected_hint_token=None,
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
async def test_jtbd_ooda_decision_flow(scenario: OODAScenario) -> None:
    """Validate Observe→Orient→Decide→Act for each JTBD scenario."""
    # Observe
    signals = extract_complexity_signals(scenario.query, scenario.context)
    signal_dict = signals.model_dump()
    for signal_name in scenario.expected_signals:
        assert signal_dict.get(signal_name), f"expected signal {signal_name} not set"

    # Orient + Decide (activation gate)
    activator = AutoActivator()
    activation = activator.should_activate(scenario.query, scenario.context)
    assert activation.should_activate is scenario.expected_activate

    orch_context = OrchestrationContext(
        query=scenario.query,
        context_tokens=scenario.context.total_tokens,
        complexity_signals=signal_dict,
    )
    plan = await create_orchestration_plan(scenario.query, orch_context, use_llm=False)

    if not scenario.expected_activate:
        assert plan.activate_rlm is False
        assert plan.activation_reason in {"conversational", "simple_task", "low_value:narrow_scope"}
        return

    # Decide (strategy selection)
    assert plan.activate_rlm is True
    assert plan.strategy == scenario.expected_strategy

    # Act (strategy-specific executable guidance)
    assert scenario.expected_hint_token is not None
    hints_joined = " ".join(plan.strategy_hints).lower()
    assert scenario.expected_hint_token.lower() in hints_joined
