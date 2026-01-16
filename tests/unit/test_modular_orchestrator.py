"""
Tests for modular orchestrator architecture.

@trace SPEC-12.01-12.07
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --- SPEC-12.01: Module structure ---


class TestModuleStructure:
    """Tests for orchestrator module structure."""

    def test_orchestrator_package_exists(self) -> None:
        """
        @trace SPEC-12.01
        The orchestrator/ package should exist.
        """
        from src import orchestrator

        assert orchestrator is not None

    def test_core_module_exists(self) -> None:
        """
        @trace SPEC-12.02
        core.py module should exist with base RLMOrchestrator.
        """
        from src.orchestrator import core

        assert hasattr(core, "RLMOrchestrator")
        assert hasattr(core, "OrchestrationState")

    def test_intelligent_module_exists(self) -> None:
        """
        @trace SPEC-12.03
        intelligent.py module should exist with IntelligentOrchestrator.
        """
        from src.orchestrator import intelligent

        assert hasattr(intelligent, "IntelligentOrchestrator")
        assert hasattr(intelligent, "OrchestratorConfig")

    def test_async_executor_module_exists(self) -> None:
        """
        @trace SPEC-12.04
        async_executor.py module should exist with AsyncRLMOrchestrator.
        """
        from src.orchestrator import async_executor

        assert hasattr(async_executor, "AsyncExecutor")
        assert hasattr(async_executor, "AsyncRLMOrchestrator")

    def test_checkpointing_module_exists(self) -> None:
        """
        @trace SPEC-12.05
        checkpointing.py module should exist with checkpoint support.
        """
        from src.orchestrator import checkpointing

        assert hasattr(checkpointing, "RLMCheckpoint")
        assert hasattr(checkpointing, "CheckpointingOrchestrator")

    def test_steering_module_exists(self) -> None:
        """
        @trace SPEC-12.06
        steering.py module should exist with user interaction support.
        """
        from src.orchestrator import steering

        assert hasattr(steering, "SteeringPoint")
        assert hasattr(steering, "InteractiveOrchestrator")


# --- SPEC-12.07: Backward compatibility ---


class TestBackwardCompatibility:
    """Tests for backward compatibility via __init__.py exports."""

    def test_rlm_orchestrator_importable_from_package(self) -> None:
        """
        @trace SPEC-12.07
        RLMOrchestrator should be importable from orchestrator package.
        """
        from src.orchestrator import RLMOrchestrator

        assert RLMOrchestrator is not None

    def test_intelligent_orchestrator_importable_from_package(self) -> None:
        """
        @trace SPEC-12.07
        IntelligentOrchestrator should be importable from orchestrator package.
        """
        from src.orchestrator import IntelligentOrchestrator

        assert IntelligentOrchestrator is not None

    def test_async_classes_importable_from_package(self) -> None:
        """
        @trace SPEC-12.07
        Async classes should be importable from orchestrator package.
        """
        from src.orchestrator import AsyncExecutor, AsyncRLMOrchestrator

        assert AsyncExecutor is not None
        assert AsyncRLMOrchestrator is not None

    def test_old_import_paths_still_work(self) -> None:
        """
        @trace SPEC-12.07
        Old import paths from src module should still work.
        """
        # These should still work via src/__init__.py re-exports
        from src import (
            AsyncExecutor,
            AsyncRLMOrchestrator,
            IntelligentOrchestrator,
            RLMOrchestrator,
        )

        assert RLMOrchestrator is not None
        assert IntelligentOrchestrator is not None
        assert AsyncExecutor is not None
        assert AsyncRLMOrchestrator is not None

    def test_orchestration_state_importable(self) -> None:
        """
        @trace SPEC-12.07
        OrchestrationState should be importable from package.
        """
        from src.orchestrator import OrchestrationState

        assert OrchestrationState is not None

    def test_orchestrator_config_importable(self) -> None:
        """
        @trace SPEC-12.07
        OrchestratorConfig should be importable from package.
        """
        from src.orchestrator import OrchestratorConfig

        assert OrchestratorConfig is not None


# --- No circular dependencies ---


class TestNoCircularDependencies:
    """Tests for no circular imports between modules."""

    def test_core_imports_independently(self) -> None:
        """Core module should import without other orchestrator modules."""
        # Clear any cached imports
        modules_to_clear = [k for k in sys.modules.keys() if "orchestrator" in k.lower()]
        for mod in modules_to_clear:
            if "test" not in mod:
                sys.modules.pop(mod, None)

        # Import core directly
        from src.orchestrator import core

        assert core is not None

    def test_intelligent_imports_independently(self) -> None:
        """Intelligent module should import without circular issues."""
        from src.orchestrator import intelligent

        assert intelligent is not None

    def test_async_executor_imports_independently(self) -> None:
        """Async executor module should import without circular issues."""
        from src.orchestrator import async_executor

        assert async_executor is not None


# --- Module testability ---


class TestModuleTestability:
    """Tests for independent module testability."""

    def test_orchestration_state_creatable(self) -> None:
        """OrchestrationState should be independently testable."""
        from src.orchestrator.core import OrchestrationState

        state = OrchestrationState()
        assert state.depth == 0
        assert state.turn == 0
        assert state.max_turns == 20
        assert state.final_answer is None

    def test_orchestrator_config_creatable(self) -> None:
        """OrchestratorConfig should be independently testable."""
        from src.orchestrator.intelligent import OrchestratorConfig

        config = OrchestratorConfig()
        assert config.orchestrator_model == "haiku"
        assert config.use_fallback is True

    def test_async_executor_creatable(self) -> None:
        """AsyncExecutor should be independently testable."""
        from src.orchestrator.async_executor import AsyncExecutor

        executor = AsyncExecutor(max_concurrency=5)
        assert executor.max_concurrency == 5


# --- SPEC-12.05: Checkpointing ---


class TestCheckpointing:
    """Tests for checkpointing module functionality."""

    def test_checkpoint_dataclass_structure(self) -> None:
        """
        @trace SPEC-12.05
        RLMCheckpoint should capture required state.
        """
        from src.orchestrator.checkpointing import RLMCheckpoint

        checkpoint = RLMCheckpoint(
            session_id="test-session",
            depth=1,
            turn=5,
            messages=[{"role": "user", "content": "test"}],
            repl_state={"x": 10},
            trajectory_events=[],
        )

        assert checkpoint.session_id == "test-session"
        assert checkpoint.depth == 1
        assert checkpoint.turn == 5
        assert checkpoint.messages == [{"role": "user", "content": "test"}]
        assert checkpoint.repl_state == {"x": 10}

    def test_checkpoint_serialization(self) -> None:
        """
        @trace SPEC-12.05
        Checkpoints should be serializable to JSON.
        """
        from src.orchestrator.checkpointing import RLMCheckpoint

        checkpoint = RLMCheckpoint(
            session_id="test",
            depth=0,
            turn=1,
            messages=[],
            repl_state={},
            trajectory_events=[],
        )

        # Should have to_dict method
        data = checkpoint.to_dict()
        assert isinstance(data, dict)
        assert data["session_id"] == "test"

    def test_checkpoint_deserialization(self) -> None:
        """
        @trace SPEC-12.05
        Checkpoints should be deserializable from JSON.
        """
        from src.orchestrator.checkpointing import RLMCheckpoint

        data = {
            "session_id": "test",
            "depth": 0,
            "turn": 1,
            "messages": [],
            "repl_state": {},
            "trajectory_events": [],
        }

        checkpoint = RLMCheckpoint.from_dict(data)
        assert checkpoint.session_id == "test"


# --- SPEC-12.06: Steering ---


class TestSteering:
    """Tests for steering module functionality."""

    def test_steering_point_dataclass(self) -> None:
        """
        @trace SPEC-12.06
        SteeringPoint should capture steering decision points.
        """
        from src.orchestrator.steering import SteeringPoint

        point = SteeringPoint(
            turn=5,
            depth=1,
            decision_type="continue_or_stop",
            options=["continue", "stop", "adjust_depth"],
            context="Current analysis is 60% complete",
        )

        assert point.turn == 5
        assert point.depth == 1
        assert "continue" in point.options

    def test_auto_steering_policy(self) -> None:
        """
        @trace SPEC-12.06
        Auto-steering should provide default decisions.
        """
        from src.orchestrator.steering import AutoSteeringPolicy

        policy = AutoSteeringPolicy()
        # Policy should be configurable
        assert policy is not None

    def test_interactive_orchestrator_extends_base(self) -> None:
        """
        @trace SPEC-12.06
        InteractiveOrchestrator should extend base functionality.
        """
        from src.orchestrator.steering import InteractiveOrchestrator

        # Should be importable without errors
        assert InteractiveOrchestrator is not None


# --- Integration tests ---


class TestModularOrchestratorIntegration:
    """Integration tests for modular orchestrator."""

    def test_all_modules_work_together(self) -> None:
        """All orchestrator modules should work together."""
        from src.orchestrator import (
            AsyncExecutor,
            AsyncRLMOrchestrator,
            IntelligentOrchestrator,
            OrchestrationState,
            OrchestratorConfig,
            RLMOrchestrator,
        )

        # Create instances without errors
        state = OrchestrationState()
        config = OrchestratorConfig()
        executor = AsyncExecutor()

        assert state is not None
        assert config is not None
        assert executor is not None

    def test_checkpointing_with_orchestrator(self) -> None:
        """Checkpointing should integrate with orchestrator state."""
        from src.orchestrator.checkpointing import CheckpointingOrchestrator, RLMCheckpoint
        from src.orchestrator.core import OrchestrationState

        # Create state
        state = OrchestrationState(depth=1, turn=3)

        # Should be able to create checkpoint from state
        checkpoint = RLMCheckpoint(
            session_id="integration-test",
            depth=state.depth,
            turn=state.turn,
            messages=state.messages,
            repl_state={},
            trajectory_events=[],
        )

        assert checkpoint.depth == 1
        assert checkpoint.turn == 3

    def test_steering_with_orchestrator(self) -> None:
        """Steering should integrate with orchestrator."""
        from src.orchestrator.steering import SteeringPoint

        # Create steering point
        point = SteeringPoint(
            turn=10,
            depth=2,
            decision_type="continue_or_stop",
            options=["continue", "stop"],
            context="Test",
        )

        assert point.turn == 10


# --- SPEC-16.22: Verification Checkpoint ---


class TestVerificationCheckpoint:
    """Tests for epistemic verification checkpoint in orchestrator."""

    def test_orchestrator_has_verification_config(self) -> None:
        """
        @trace SPEC-16.22
        RLMOrchestrator should have verification_config.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        assert hasattr(orchestrator, "verification_config")
        assert orchestrator.verification_config is not None

    def test_verification_config_enabled_by_default(self) -> None:
        """
        @trace SPEC-16.22
        Verification should be enabled by default (always-on).
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        assert orchestrator.verification_config.enabled is True

    def test_verification_config_can_be_disabled(self) -> None:
        """
        @trace SPEC-16.22
        Verification should be disableable via config.
        """
        from src.epistemic import VerificationConfig
        from src.orchestrator.core import RLMOrchestrator

        config = VerificationConfig(enabled=False)
        orchestrator = RLMOrchestrator(verification_config=config)
        assert orchestrator.verification_config.enabled is False

    def test_verify_response_method_exists(self) -> None:
        """
        @trace SPEC-16.22
        _verify_response method should exist on orchestrator.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        assert hasattr(orchestrator, "_verify_response")
        assert callable(orchestrator._verify_response)

    @pytest.mark.asyncio
    async def test_verify_response_returns_report_when_disabled(self) -> None:
        """
        @trace SPEC-16.22
        _verify_response should return empty report when disabled.
        """
        from src.epistemic import VerificationConfig
        from src.orchestrator.core import RLMOrchestrator
        from src.trajectory import StreamingTrajectory, TrajectoryRenderer
        from src.types import SessionContext

        config = VerificationConfig(enabled=False)
        orchestrator = RLMOrchestrator(verification_config=config)

        context = SessionContext()
        trajectory = StreamingTrajectory(TrajectoryRenderer())

        report, should_retry = await orchestrator._verify_response(
            "Test response", context, MagicMock(), trajectory
        )

        assert report.response_id == "disabled"
        assert should_retry is False

    @pytest.mark.asyncio
    async def test_verify_response_returns_report_when_no_evidence(self) -> None:
        """
        @trace SPEC-16.22
        _verify_response should return empty report when no evidence.
        """
        from src.orchestrator.core import RLMOrchestrator
        from src.trajectory import StreamingTrajectory, TrajectoryRenderer
        from src.types import SessionContext

        orchestrator = RLMOrchestrator()
        context = SessionContext()  # Empty context, no files or tool outputs
        trajectory = StreamingTrajectory(TrajectoryRenderer())

        report, should_retry = await orchestrator._verify_response(
            "Test response", context, MagicMock(), trajectory
        )

        assert report.response_id == "no_evidence"
        assert should_retry is False

    def test_trajectory_event_type_verification_exists(self) -> None:
        """
        @trace SPEC-16.22
        TrajectoryEventType should have VERIFICATION type.
        """
        from src.trajectory import TrajectoryEventType

        assert hasattr(TrajectoryEventType, "VERIFICATION")
        assert TrajectoryEventType.VERIFICATION.value == "verification"

    def test_verification_config_default_values(self) -> None:
        """
        @trace SPEC-16.22
        VerificationConfig should have sensible defaults.
        """
        from src.epistemic import VerificationConfig

        config = VerificationConfig()
        assert config.enabled is True
        assert config.mode == "sample"  # Sample mode by default for cost control
        assert config.support_threshold == 0.7
        assert config.max_retries == 2
        assert config.on_failure == "retry"  # Default to retry on failure


# --- SPEC-16.24: Retry with Evidence Focus ---


class TestRetryWithEvidenceFocus:
    """Tests for evidence-focused retry functionality."""

    def test_build_evidence_focused_retry_prompt_exists(self) -> None:
        """
        @trace SPEC-16.24
        _build_evidence_focused_retry_prompt method should exist.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        assert hasattr(orchestrator, "_build_evidence_focused_retry_prompt")
        assert callable(orchestrator._build_evidence_focused_retry_prompt)

    def test_build_evidence_focused_retry_prompt_includes_flagged_claims(self) -> None:
        """
        @trace SPEC-16.24
        Retry prompt should include flagged claims.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        flagged = ["The function returns None", "The API supports JSON"]
        prompt = orchestrator._build_evidence_focused_retry_prompt(
            flagged_texts=flagged,
            critical_gaps=[],
            evidence_sources=[],
        )

        assert "Unverified Claims" in prompt
        assert "The function returns None" in prompt
        assert "The API supports JSON" in prompt

    def test_build_evidence_focused_retry_prompt_includes_evidence_sources(self) -> None:
        """
        @trace SPEC-16.24
        Retry prompt should list available evidence sources.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        sources = ["src/api.py", "src/models.py"]
        prompt = orchestrator._build_evidence_focused_retry_prompt(
            flagged_texts=["Some claim"],
            critical_gaps=[],
            evidence_sources=sources,
        )

        assert "Available Evidence Sources" in prompt
        assert "src/api.py" in prompt
        assert "src/models.py" in prompt

    def test_build_evidence_focused_retry_prompt_includes_critical_gaps(self) -> None:
        """
        @trace SPEC-16.24
        Retry prompt should explain critical issues found.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()

        # Mock a gap with gap_type attribute
        class MockGap:
            gap_type = "phantom_citation"

        prompt = orchestrator._build_evidence_focused_retry_prompt(
            flagged_texts=["Some claim"],
            critical_gaps=[MockGap()],
            evidence_sources=[],
        )

        assert "Critical Issues Found" in prompt
        assert "Phantom citation" in prompt

    def test_build_evidence_focused_retry_prompt_includes_requirements(self) -> None:
        """
        @trace SPEC-16.24
        Retry prompt should include grounding requirements.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        prompt = orchestrator._build_evidence_focused_retry_prompt(
            flagged_texts=["Some claim"],
            critical_gaps=[],
            evidence_sources=[],
        )

        assert "Requirements for Revised Response" in prompt
        assert "Only make claims you can support" in prompt
        assert "Cite specific sources" in prompt
        assert "Acknowledge uncertainty" in prompt

    def test_verification_config_has_critical_model(self) -> None:
        """
        @trace SPEC-16.24
        VerificationConfig should have critical_model for retry.
        """
        from src.epistemic import VerificationConfig

        config = VerificationConfig()
        assert hasattr(config, "critical_model")
        # Critical model should be Sonnet by default
        assert config.critical_model == "sonnet"

    def test_verification_config_max_retries_default(self) -> None:
        """
        @trace SPEC-16.24
        VerificationConfig should have max_retries for escalation control.
        """
        from src.epistemic import VerificationConfig

        config = VerificationConfig()
        assert hasattr(config, "max_retries")
        assert config.max_retries == 2  # Default to 2 retries before escalation

    def test_hallucination_report_has_flagged_claim_texts(self) -> None:
        """
        @trace SPEC-16.24
        HallucinationReport should expose flagged claim texts.
        """
        from src.epistemic import HallucinationReport

        report = HallucinationReport(response_id="test")
        assert hasattr(report, "flagged_claim_texts")
        # Should be list
        assert isinstance(report.flagged_claim_texts, list)

    def test_hallucination_report_has_critical_gaps(self) -> None:
        """
        @trace SPEC-16.24
        HallucinationReport should expose critical gaps.
        """
        from src.epistemic import HallucinationReport

        report = HallucinationReport(response_id="test")
        assert hasattr(report, "critical_gaps")
        # Should be list
        assert isinstance(report.critical_gaps, list)
