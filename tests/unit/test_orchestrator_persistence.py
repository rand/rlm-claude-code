"""
Unit tests for orchestrator persistence and bridge wiring.

Verifies that StatePersistence, ReasoningTraces, and StrategyCache
are properly wired into the RLM orchestration loop (#4, #15 fixes).
"""

import copy
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import RLMConfig, default_config
from src.orchestrator import core as orchestrator_core_module
from src.orchestrator.core import RLMOrchestrator
from src.state_persistence import StatePersistence
from src.trajectory import TrajectoryEvent, TrajectoryEventType


def _test_config() -> RLMConfig:
    """Config with trajectory export disabled for tests."""
    cfg = copy.deepcopy(default_config)
    cfg.trajectory.export_enabled = False
    return cfg


def _mock_client_with_final(answer: str = "FINAL: done") -> AsyncMock:
    """Create a mock client that returns a FINAL answer."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = answer
    mock_response.input_tokens = 10
    mock_response.output_tokens = 20
    mock_response.model = "test"
    mock_response.provider = MagicMock(value="anthropic")
    mock_client.complete = AsyncMock(return_value=mock_response)
    return mock_client


class TestOrchestratorPersistenceParam:
    """Tests that persistence parameter is accepted and stored."""

    def test_accepts_persistence_param(self):
        """Orchestrator accepts persistence parameter."""
        mock_persistence = MagicMock(spec=StatePersistence)
        orch = RLMOrchestrator(persistence=mock_persistence)
        assert orch._persistence is mock_persistence

    def test_persistence_defaults_to_none(self):
        """Persistence defaults to None when not provided."""
        orch = RLMOrchestrator()
        assert orch._persistence is None


class TestPersistenceSessionInit:
    """Tests that persistence.init_session is called at RLM_START."""

    @pytest.mark.asyncio
    async def test_init_session_called_on_run(self):
        """init_session is called with session ID from env."""
        mock_persistence = MagicMock(spec=StatePersistence)
        mock_persistence.current_state = MagicMock()

        orch = RLMOrchestrator(
            config=_test_config(),
            persistence=mock_persistence,
            client=_mock_client_with_final(),
            smart_routing=False,
        )

        with (
            patch.object(
                orchestrator_core_module, "should_activate_rlm", return_value=(True, "test")
            ),
            patch.dict("os.environ", {"CLAUDE_SESSION_ID": "test-session-123"}),
        ):
            from src.types import SessionContext

            ctx = SessionContext()
            async for _ in orch.run("test query", ctx):
                pass

        mock_persistence.init_session.assert_called_once_with("test-session-123")
        mock_persistence.update_rlm_active.assert_called_once_with(True)


class TestPostExecutionBridges:
    """Tests for post-execution persistence and bridge wiring."""

    @pytest.mark.asyncio
    async def test_persistence_save_called_after_run(self):
        """save_state is called after run completes."""
        mock_persistence = MagicMock(spec=StatePersistence)
        mock_persistence.current_state = MagicMock()

        orch = RLMOrchestrator(
            config=_test_config(),
            persistence=mock_persistence,
            client=_mock_client_with_final(),
            smart_routing=False,
        )

        with patch.object(
            orchestrator_core_module, "should_activate_rlm", return_value=(True, "test")
        ):
            from src.types import SessionContext

            ctx = SessionContext()
            async for _ in orch.run("test query", ctx):
                pass

        mock_persistence.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_trajectory_events_counted(self):
        """increment_trajectory_events is called with event count."""
        mock_persistence = MagicMock(spec=StatePersistence)
        mock_persistence.current_state = MagicMock()

        orch = RLMOrchestrator(
            config=_test_config(),
            persistence=mock_persistence,
            client=_mock_client_with_final(),
            smart_routing=False,
        )

        with patch.object(
            orchestrator_core_module, "should_activate_rlm", return_value=(True, "test")
        ):
            from src.types import SessionContext

            ctx = SessionContext()
            async for _ in orch.run("test query", ctx):
                pass

        mock_persistence.increment_trajectory_events.assert_called_once()
        call_args = mock_persistence.increment_trajectory_events.call_args
        assert call_args[0][0] > 0

    @pytest.mark.asyncio
    async def test_reasoning_traces_bridge(self):
        """ReasoningTraces.from_trajectory_event called for each event."""
        mock_memory = MagicMock()
        mock_traces = MagicMock()

        orch = RLMOrchestrator(
            config=_test_config(),
            memory_store=mock_memory,
            client=_mock_client_with_final(),
            smart_routing=False,
        )

        # Patch where ReasoningTraces is imported (lazy import in core.py)
        import src.reasoning_traces as rt_module

        with (
            patch.object(
                orchestrator_core_module, "should_activate_rlm", return_value=(True, "test")
            ),
            patch.object(rt_module, "ReasoningTraces", return_value=mock_traces),
        ):
            from src.types import SessionContext

            ctx = SessionContext()
            async for _ in orch.run("test query", ctx):
                pass

        assert mock_traces.from_trajectory_event.call_count > 0

    @pytest.mark.asyncio
    async def test_strategy_cache_bridge(self):
        """StrategyCache.add called with query and analysis."""
        mock_cache = MagicMock()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = MagicMock()

        orch = RLMOrchestrator(
            config=_test_config(),
            client=_mock_client_with_final(),
            smart_routing=False,
        )

        import src.strategy_cache as sc_module
        import src.trajectory_analysis as ta_module

        with (
            patch.object(
                orchestrator_core_module, "should_activate_rlm", return_value=(True, "test")
            ),
            patch.object(ta_module, "TrajectoryAnalyzer", return_value=mock_analyzer),
            patch.object(sc_module, "get_strategy_cache", return_value=mock_cache),
        ):
            from src.types import SessionContext

            ctx = SessionContext()
            async for _ in orch.run("test query", ctx):
                pass

        mock_cache.add.assert_called_once()
        assert mock_cache.add.call_args[0][0] == "test query"
        mock_cache.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_bridges_fail_silently(self):
        """Bridge errors don't crash the orchestrator."""
        mock_memory = MagicMock()

        orch = RLMOrchestrator(
            config=_test_config(),
            memory_store=mock_memory,
            client=_mock_client_with_final(),
            smart_routing=False,
        )

        import src.reasoning_traces as rt_module

        with (
            patch.object(
                orchestrator_core_module, "should_activate_rlm", return_value=(True, "test")
            ),
            patch.object(rt_module, "ReasoningTraces", side_effect=RuntimeError("broken")),
        ):
            from src.types import SessionContext

            ctx = SessionContext()
            events = []
            async for event in orch.run("test query", ctx):
                events.append(event)

        final_events = [
            e
            for e in events
            if isinstance(e, TrajectoryEvent) and e.type == TrajectoryEventType.FINAL
        ]
        assert len(final_events) > 0
