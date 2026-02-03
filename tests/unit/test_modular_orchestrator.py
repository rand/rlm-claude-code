"""
Tests for modular orchestrator architecture.

@trace SPEC-12.01-12.07
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

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
            OrchestrationState,
            OrchestratorConfig,
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
        from src.orchestrator.checkpointing import RLMCheckpoint
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


class TestClaimFlagging:
    """Tests for claim flagging on verification failure (SPEC-16.23)."""

    def test_annotate_flagged_claims_method_exists(self) -> None:
        """
        @trace SPEC-16.23
        _annotate_flagged_claims method should exist.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        assert hasattr(orchestrator, "_annotate_flagged_claims")
        assert callable(orchestrator._annotate_flagged_claims)

    def test_annotate_flagged_claims_returns_original_when_no_flags(self) -> None:
        """
        @trace SPEC-16.23
        Response unchanged when no claims are flagged.
        """
        from src.epistemic import HallucinationReport
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        report = HallucinationReport(response_id="test")
        original = "This is the response text."

        result = orchestrator._annotate_flagged_claims(original, report)
        assert result == original

    def test_annotate_flagged_claims_adds_markers(self) -> None:
        """
        @trace SPEC-16.23
        Flagged claims should be annotated with uncertainty markers.
        """
        from src.epistemic import ClaimVerification, HallucinationReport
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        report = HallucinationReport(response_id="test")

        # Add a flagged claim
        claim = ClaimVerification(
            claim_id="c1",
            claim_text="The API returns JSON",
            evidence_ids=["e1"],
            evidence_support=0.3,
            is_flagged=True,
            flag_reason="unsupported",
        )
        report.add_claim(claim)

        original = "The API returns JSON and handles errors."
        result = orchestrator._annotate_flagged_claims(original, report)

        assert "[UNVERIFIED:" in result
        assert "unsupported" in result.lower() or "No supporting evidence" in result


class TestAskUserEscalation:
    """Tests for ask user escalation (SPEC-16.24)."""

    def test_build_ask_user_prompt_method_exists(self) -> None:
        """
        @trace SPEC-16.24
        _build_ask_user_prompt method should exist.
        """
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        assert hasattr(orchestrator, "_build_ask_user_prompt")
        assert callable(orchestrator._build_ask_user_prompt)

    def test_build_ask_user_prompt_returns_dict(self) -> None:
        """
        @trace SPEC-16.24
        Ask prompt should return dict with question structure.
        """
        from src.epistemic import ClaimVerification, HallucinationReport
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        report = HallucinationReport(response_id="test")

        claim = ClaimVerification(
            claim_id="c1",
            claim_text="Unverified claim",
            evidence_ids=[],
            evidence_support=0.2,
            is_flagged=True,
            flag_reason="unsupported",
        )
        report.add_claim(claim)

        result = orchestrator._build_ask_user_prompt(report)

        assert isinstance(result, dict)
        assert "question" in result
        assert "header" in result
        assert "options" in result

    def test_build_ask_user_prompt_includes_options(self) -> None:
        """
        @trace SPEC-16.24
        Ask prompt should include accept/revise/context options.
        """
        from src.epistemic import ClaimVerification, HallucinationReport
        from src.orchestrator.core import RLMOrchestrator

        orchestrator = RLMOrchestrator()
        report = HallucinationReport(response_id="test")

        claim = ClaimVerification(
            claim_id="c1",
            claim_text="Unverified claim",
            evidence_ids=[],
            evidence_support=0.2,
            is_flagged=True,
            flag_reason="unsupported",
        )
        report.add_claim(claim)

        result = orchestrator._build_ask_user_prompt(report)

        options = result["options"]
        assert len(options) == 3
        labels = [o["label"] for o in options]
        assert "Accept response" in labels
        assert "Request revision" in labels
        assert "Add context" in labels


class TestParallelVerification:
    """Tests for parallel claim verification (SPEC-16.26)."""

    def test_audit_claims_has_parallel_parameter(self) -> None:
        """
        @trace SPEC-16.26
        audit_claims should accept parallel parameter.
        """
        # Check signature
        import inspect

        from src.epistemic.evidence_auditor import EvidenceAuditor

        sig = inspect.signature(EvidenceAuditor.audit_claims)
        params = list(sig.parameters.keys())
        assert "parallel" in params

    def test_verification_config_has_parallel_flag(self) -> None:
        """
        @trace SPEC-16.26
        VerificationConfig should have parallel_verification flag.
        """
        from src.epistemic import VerificationConfig

        config = VerificationConfig()
        assert hasattr(config, "parallel_verification")
        # Should be enabled by default for performance
        assert config.parallel_verification is True


class TestSampleMode:
    """Tests for sample mode cost control (SPEC-16.28)."""

    def test_should_verify_claim_accepts_claim_text(self) -> None:
        """
        @trace SPEC-16.28
        should_verify_claim should accept optional claim_text.
        """
        from src.epistemic import VerificationConfig

        config = VerificationConfig(mode="sample")

        # Should accept claim_text parameter
        result = config.should_verify_claim(0, False, claim_text="Some claim")
        assert isinstance(result, bool)

    def test_should_verify_claim_prioritizes_uncertainty_markers(self) -> None:
        """
        @trace SPEC-16.28
        Claims with uncertainty markers should be prioritized.
        """
        from src.epistemic import VerificationConfig

        config = VerificationConfig(mode="sample", sample_rate=0.1)

        # Claim with uncertainty marker should be verified
        uncertain = config.should_verify_claim(99, False, claim_text="This might be the case")
        assert uncertain is True

        # Claim without uncertainty marker at non-sampled index
        certain = config.should_verify_claim(99, False, claim_text="This is definitely true")
        # Index 99 with rate 0.1 -> 99 % 10 != 0, so not sampled
        assert certain is False

    def test_has_uncertainty_markers_method_exists(self) -> None:
        """
        @trace SPEC-16.28
        _has_uncertainty_markers helper should exist.
        """
        from src.epistemic import VerificationConfig

        config = VerificationConfig()
        assert hasattr(config, "_has_uncertainty_markers")

    def test_uncertainty_markers_detection(self) -> None:
        """
        @trace SPEC-16.28
        Should detect common hedging language.
        """
        from src.epistemic import VerificationConfig

        config = VerificationConfig()

        # Should detect various uncertainty markers
        assert config._has_uncertainty_markers("This might be true") is True
        assert config._has_uncertainty_markers("I think this works") is True
        assert config._has_uncertainty_markers("It probably returns JSON") is True
        assert config._has_uncertainty_markers("The function seems correct") is True
        assert config._has_uncertainty_markers("approximately 100 items") is True

        # Should not flag certain statements
        assert config._has_uncertainty_markers("The function returns 42") is False
        assert config._has_uncertainty_markers("This is a fact") is False
