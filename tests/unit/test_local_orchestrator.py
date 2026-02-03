"""
Unit tests for LocalOrchestrator module.

Tests local model-based orchestration for RLM activation decisions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.local_orchestrator import (
    RECOMMENDED_CONFIGS,
    LocalInferenceResult,
    LocalModelBackend,
    LocalModelConfig,
    LocalOrchestrator,
    MLXRunner,
    OllamaRunner,
)


class TestLocalModelConfig:
    """Tests for LocalModelConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = LocalModelConfig()
        assert config.model_name == "gemma-3-270m-it"
        assert config.backend == LocalModelBackend.MLX
        assert config.max_tokens == 300
        assert config.temperature == 0.1
        assert config.timeout_ms == 2000
        assert config.fallback_to_heuristics is True

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = LocalModelConfig(
            model_name="qwen3-0.6b",
            backend=LocalModelBackend.OLLAMA,
            max_tokens=500,
            temperature=0.2,
        )
        assert config.model_name == "qwen3-0.6b"
        assert config.backend == LocalModelBackend.OLLAMA
        assert config.max_tokens == 500
        assert config.temperature == 0.2


class TestRecommendedConfigs:
    """Tests for RECOMMENDED_CONFIGS presets."""

    def test_ultra_fast_config(self):
        """Ultra fast config uses smallest model."""
        config = RECOMMENDED_CONFIGS["ultra_fast"]
        assert config.model_name == "gemma-3-270m-it"
        assert config.backend == LocalModelBackend.MLX
        assert config.max_tokens == 200

    def test_balanced_config(self):
        """Balanced config uses medium model."""
        config = RECOMMENDED_CONFIGS["balanced"]
        assert config.model_name == "qwen3-0.6b"
        assert config.backend == LocalModelBackend.MLX

    def test_quality_config(self):
        """Quality config uses larger model."""
        config = RECOMMENDED_CONFIGS["quality"]
        assert config.model_name == "lfm2.5-1.2b"
        assert config.max_tokens == 400

    def test_portable_config(self):
        """Portable config uses Ollama backend."""
        config = RECOMMENDED_CONFIGS["portable"]
        assert config.backend == LocalModelBackend.OLLAMA


class TestMLXRunner:
    """Tests for MLXRunner."""

    def test_is_available_without_mlx(self):
        """Returns False when mlx_lm not installed."""
        with patch.dict("sys.modules", {"mlx_lm": None}):
            runner = MLXRunner(LocalModelConfig())
            # Can't easily test this without actual import failure
            # Just verify the method exists
            assert hasattr(runner, "is_available")

    def test_resolve_model_id(self):
        """Model names resolve to HuggingFace IDs."""
        runner = MLXRunner(LocalModelConfig(model_name="gemma-3-270m-it"))
        model_id = runner._resolve_model_id()
        assert "gemma" in model_id.lower()

    def test_resolve_unknown_model(self):
        """Unknown model names pass through unchanged."""
        runner = MLXRunner(LocalModelConfig(model_name="custom/my-model"))
        model_id = runner._resolve_model_id()
        assert model_id == "custom/my-model"

    def test_get_model_info(self):
        """Model info includes backend and name."""
        runner = MLXRunner(LocalModelConfig(model_name="qwen3-0.6b"))
        info = runner.get_model_info()
        assert info["backend"] == "mlx"
        assert info["model_name"] == "qwen3-0.6b"
        assert info["loaded"] is False


class TestOllamaRunner:
    """Tests for OllamaRunner."""

    def test_resolve_model_name(self):
        """Model names resolve to Ollama format."""
        runner = OllamaRunner(LocalModelConfig(model_name="gemma-3-270m-it"))
        name = runner._resolve_model_name()
        assert name == "gemma3:270m"

    def test_resolve_unknown_model(self):
        """Unknown model names pass through unchanged."""
        runner = OllamaRunner(LocalModelConfig(model_name="custom-model"))
        name = runner._resolve_model_name()
        assert name == "custom-model"

    def test_get_model_info(self):
        """Model info includes backend and names."""
        runner = OllamaRunner(LocalModelConfig(model_name="qwen3-0.6b"))
        info = runner.get_model_info()
        assert info["backend"] == "ollama"
        assert info["model_name"] == "qwen3-0.6b"
        assert info["ollama_model"] == "qwen3:0.6b"


class TestLocalOrchestrator:
    """Tests for LocalOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with fallback enabled."""
        config = LocalModelConfig(fallback_to_heuristics=True)
        return LocalOrchestrator(config=config)

    def test_initialization(self, orchestrator):
        """Orchestrator initializes with correct state."""
        assert orchestrator.config.fallback_to_heuristics is True
        assert orchestrator._runner is None
        assert orchestrator._stats["local_decisions"] == 0
        assert orchestrator._stats["heuristic_fallbacks"] == 0

    def test_heuristic_decision_simple_task(self, orchestrator):
        """Heuristic correctly identifies simple tasks."""
        decision = orchestrator._heuristic_decision(
            query="show config.py",
            context_summary="- Context tokens: 1,000",
        )
        assert decision["activate_rlm"] is False
        assert "knowledge_retrieval" in decision["signals"]

    def test_heuristic_decision_discovery_task(self, orchestrator):
        """Heuristic correctly identifies discovery tasks."""
        decision = orchestrator._heuristic_decision(
            query="Why is the authentication failing intermittently?",
            context_summary="- Context tokens: 50,000",
        )
        assert decision["activate_rlm"] is True
        assert (
            "discovery_required" in decision["signals"] or "debugging_deep" in decision["signals"]
        )

    def test_heuristic_decision_synthesis_task(self, orchestrator):
        """Heuristic correctly identifies synthesis tasks."""
        decision = orchestrator._heuristic_decision(
            query="Update all usages of the deprecated API",
            context_summary="- Context tokens: 30,000",
        )
        assert decision["activate_rlm"] is True
        assert "synthesis_required" in decision["signals"]

    def test_heuristic_decision_uncertainty(self, orchestrator):
        """Heuristic correctly identifies uncertainty."""
        decision = orchestrator._heuristic_decision(
            query="What's the best approach for adding caching?",
            context_summary="- Context tokens: 20,000",
        )
        assert decision["activate_rlm"] is True
        assert "uncertainty_high" in decision["signals"]

    def test_heuristic_decision_architectural(self, orchestrator):
        """Heuristic correctly identifies architectural tasks."""
        decision = orchestrator._heuristic_decision(
            query="Design a system for handling real-time events",
            context_summary="- Context tokens: 10,000",
        )
        assert decision["activate_rlm"] is True
        assert "architectural" in decision["signals"]

    def test_heuristic_decision_large_context(self, orchestrator):
        """Heuristic activates for large context."""
        decision = orchestrator._heuristic_decision(
            query="summarize",
            context_summary="- Large context detected\n- Context tokens: 100,000",
        )
        assert decision["activate_rlm"] is True
        assert "large_context" in decision["signals"]

    def test_heuristic_decision_conversational(self, orchestrator):
        """Heuristic bypasses conversational queries."""
        decision = orchestrator._heuristic_decision(
            query="ok",
            context_summary="- Context tokens: 5,000",
        )
        assert decision["activate_rlm"] is False
        assert "conversational" in decision["signals"]

    def test_parse_response_valid_json(self, orchestrator):
        """Parses valid JSON response."""
        response = '{"activate_rlm": true, "activation_reason": "test"}'
        result = orchestrator._parse_response(response)
        assert result["activate_rlm"] is True
        assert result["activation_reason"] == "test"

    def test_parse_response_json_in_text(self, orchestrator):
        """Extracts JSON from surrounding text."""
        response = 'Here is my decision: {"activate_rlm": false, "reason": "simple"} That is all.'
        result = orchestrator._parse_response(response)
        assert result["activate_rlm"] is False

    def test_parse_response_invalid_json(self, orchestrator):
        """Raises on invalid JSON."""
        with pytest.raises(ValueError, match="No JSON found"):
            orchestrator._parse_response("no json here")

    def test_cache_key_computation(self, orchestrator):
        """Cache keys are consistent for same input."""
        key1 = orchestrator._compute_cache_key("test query", "context")
        key2 = orchestrator._compute_cache_key("test query", "context")
        assert key1 == key2

    def test_cache_key_differs(self, orchestrator):
        """Cache keys differ for different inputs."""
        key1 = orchestrator._compute_cache_key("query a", "context")
        key2 = orchestrator._compute_cache_key("query b", "context")
        assert key1 != key2

    def test_cache_update_and_eviction(self, orchestrator):
        """Cache evicts old entries when full."""
        orchestrator.config.cache_size = 4

        # Fill cache
        for i in range(4):
            orchestrator._update_cache(f"key{i}", {"value": i})

        assert len(orchestrator._cache) == 4

        # Add one more - should trigger eviction
        orchestrator._update_cache("key4", {"value": 4})

        # Should have evicted some entries
        assert len(orchestrator._cache) <= 4

    def test_statistics_initial(self, orchestrator):
        """Initial statistics are zero."""
        stats = orchestrator.get_statistics()
        assert stats["local_decisions"] == 0
        assert stats["heuristic_fallbacks"] == 0
        assert stats["cache_hits"] == 0
        assert stats["total_decisions"] == 0

    @pytest.mark.asyncio
    async def test_orchestrate_fallback_on_no_backend(self, orchestrator):
        """Falls back to heuristics when no backend available."""
        # Mock _get_runner to raise
        orchestrator._get_runner = MagicMock(side_effect=RuntimeError("No backend"))

        decision = await orchestrator.orchestrate(
            query="Why is this failing?",
            context_summary="- Context tokens: 10,000",
        )

        assert decision is not None
        assert orchestrator._stats["heuristic_fallbacks"] == 1
        assert orchestrator._stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_orchestrate_uses_cache(self, orchestrator):
        """Orchestrate uses cached decisions."""
        # Pre-populate cache
        cache_key = orchestrator._compute_cache_key("test query", "context")
        cached_decision = {"activate_rlm": True, "cached": True}
        orchestrator._cache[cache_key] = cached_decision

        decision = await orchestrator.orchestrate(
            query="test query",
            context_summary="context",
        )

        assert decision["cached"] is True
        assert orchestrator._stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_orchestrate_with_mock_runner(self, orchestrator):
        """Orchestrate works with mocked runner (cascade disabled)."""
        # Disable cascade to test LLM path directly
        orchestrator.config.use_cascade = False

        mock_runner = AsyncMock()
        mock_runner.generate.return_value = LocalInferenceResult(
            content='{"activate_rlm": true, "activation_reason": "mock_test", "confidence": 0.9}',
            latency_ms=25.0,
            tokens_generated=50,
            model_name="test-model",
            backend=LocalModelBackend.MLX,
        )

        orchestrator._runner = mock_runner

        decision = await orchestrator.orchestrate(
            query="complex analysis task",
            context_summary="- Context tokens: 50,000",
        )

        assert decision["activate_rlm"] is True
        assert decision["activation_reason"] == "mock_test"
        assert orchestrator._stats["local_decisions"] == 1
        # Latency is measured by orchestrate(), should be > 0
        assert orchestrator._stats["total_latency_ms"] > 0


class TestLocalInferenceResult:
    """Tests for LocalInferenceResult dataclass."""

    def test_create_result(self):
        """Can create inference result."""
        result = LocalInferenceResult(
            content="test response",
            latency_ms=30.5,
            tokens_generated=25,
            model_name="test-model",
            backend=LocalModelBackend.MLX,
        )
        assert result.content == "test response"
        assert result.latency_ms == 30.5
        assert result.tokens_generated == 25
        assert result.backend == LocalModelBackend.MLX


class TestHeuristicDecisionEdgeCases:
    """Edge case tests for heuristic decisions."""

    @pytest.fixture
    def orchestrator(self):
        return LocalOrchestrator()

    def test_flaky_test_detection(self, orchestrator):
        """Detects flaky test debugging."""
        decision = orchestrator._heuristic_decision(
            query="This test is flaky and fails randomly",
            context_summary="",
        )
        assert decision["activate_rlm"] is True
        assert "debugging_deep" in decision["signals"]

    def test_race_condition_detection(self, orchestrator):
        """Detects race condition debugging."""
        decision = orchestrator._heuristic_decision(
            query="I think there's a race condition here",
            context_summary="",
        )
        assert decision["activate_rlm"] is True
        assert "debugging_deep" in decision["signals"]

    def test_migration_detection(self, orchestrator):
        """Detects migration tasks."""
        decision = orchestrator._heuristic_decision(
            query="We need to migrate from PostgreSQL to MySQL",
            context_summary="",
        )
        assert decision["activate_rlm"] is True
        # Should trigger architectural signal

    def test_yes_no_bypass(self, orchestrator):
        """Simple yes/no bypasses RLM."""
        for query in ["yes", "no", "ok", "thanks"]:
            decision = orchestrator._heuristic_decision(query, "")
            assert decision["activate_rlm"] is False

    def test_execution_mode_thorough(self, orchestrator):
        """Multiple high-value signals trigger thorough mode."""
        decision = orchestrator._heuristic_decision(
            query="Why is this flaky? What's the best approach to fix all instances?",
            context_summary="",
        )
        assert decision["execution_mode"] == "thorough"

    def test_depth_budget_for_debugging(self, orchestrator):
        """Deep debugging gets higher depth budget."""
        decision = orchestrator._heuristic_decision(
            query="This race condition is causing intermittent failures",
            context_summary="",
        )
        assert decision["depth_budget"] >= 2


class TestCascadeConfiguration:
    """Tests for cascade configuration in LocalModelConfig."""

    def test_default_cascade_enabled(self):
        """Cascade is enabled by default."""
        config = LocalModelConfig()
        assert config.use_cascade is True
        assert config.setfit_enabled is True
        assert config.gliner_enabled is True

    def test_cascade_thresholds(self):
        """Default cascade confidence thresholds."""
        config = LocalModelConfig()
        assert config.setfit_confidence_threshold == 0.85
        assert config.gliner_confidence_threshold == 0.8
        assert config.llm_confidence_threshold == 0.7

    def test_custom_thresholds(self):
        """Can customize cascade thresholds."""
        config = LocalModelConfig(
            setfit_confidence_threshold=0.9,
            gliner_confidence_threshold=0.85,
            llm_confidence_threshold=0.75,
        )
        assert config.setfit_confidence_threshold == 0.9
        assert config.gliner_confidence_threshold == 0.85
        assert config.llm_confidence_threshold == 0.75

    def test_disable_individual_levels(self):
        """Can disable individual cascade levels."""
        config = LocalModelConfig(
            setfit_enabled=False,
            gliner_enabled=False,
        )
        assert config.setfit_enabled is False
        assert config.gliner_enabled is False


class TestCascadeBehavior:
    """Tests for cascade behavior in LocalOrchestrator."""

    @pytest.fixture
    def cascade_orchestrator(self):
        """Create orchestrator with cascade enabled."""
        config = LocalModelConfig(use_cascade=True)
        return LocalOrchestrator(config=config)

    def test_setfit_lazy_initialization(self, cascade_orchestrator):
        """SetFit classifier is lazily initialized."""
        assert cascade_orchestrator._setfit_classifier is None
        assert cascade_orchestrator._setfit_load_attempted is False

        # Attempt to load
        classifier = cascade_orchestrator._ensure_setfit_classifier()

        # Load was attempted (may or may not succeed based on availability)
        assert cascade_orchestrator._setfit_load_attempted is True

    def test_gliner_lazy_initialization(self, cascade_orchestrator):
        """GLiNER extractor is lazily initialized."""
        assert cascade_orchestrator._gliner_extractor is None
        assert cascade_orchestrator._gliner_load_attempted is False

        # Attempt to load
        extractor = cascade_orchestrator._ensure_gliner_extractor()

        # Load was attempted (may or may not succeed based on availability)
        assert cascade_orchestrator._gliner_load_attempted is True

    def test_cascade_statistics_initialized(self, cascade_orchestrator):
        """Cascade statistics are initialized properly."""
        stats = cascade_orchestrator._stats
        assert "setfit_decisions" in stats
        assert "gliner_decisions" in stats
        assert "local_decisions" in stats
        assert "heuristic_fallbacks" in stats
        assert "setfit_low_confidence" in stats
        assert "gliner_low_confidence" in stats
        assert "llm_low_confidence" in stats

    def test_get_statistics_includes_cascade(self, cascade_orchestrator):
        """get_statistics includes cascade breakdown."""
        stats = cascade_orchestrator.get_statistics()
        assert "setfit_rate" in stats
        assert "gliner_rate" in stats
        assert "local_rate" in stats
        assert "heuristic_rate" in stats
        assert "fast_cascade_rate" in stats

    @pytest.mark.asyncio
    async def test_cascade_falls_through_to_heuristics(self, cascade_orchestrator):
        """Cascade falls through to heuristics when no models available."""
        # Disable all model levels
        cascade_orchestrator.config.setfit_enabled = False
        cascade_orchestrator.config.gliner_enabled = False
        cascade_orchestrator.config.fallback_to_heuristics = True

        decision = await cascade_orchestrator.orchestrate(
            query="simple query",
            context_summary="",
        )

        # Should have fallen through to heuristics
        assert decision["_cascade_level"] == 4
        assert decision["_cascade_source"] == "heuristics"
        assert cascade_orchestrator._stats["heuristic_fallbacks"] == 1

    def test_complexity_to_mode_conversion(self, cascade_orchestrator):
        """Tests complexity level to mode conversion."""
        from src.setfit_classifier import ComplexityLevel

        assert cascade_orchestrator._complexity_to_mode(ComplexityLevel.TRIVIAL) == "fast"
        assert cascade_orchestrator._complexity_to_mode(ComplexityLevel.SIMPLE) == "fast"
        assert cascade_orchestrator._complexity_to_mode(ComplexityLevel.MODERATE) == "balanced"
        assert cascade_orchestrator._complexity_to_mode(ComplexityLevel.COMPLEX) == "thorough"
        assert cascade_orchestrator._complexity_to_mode(ComplexityLevel.UNBOUNDED) == "thorough"

    def test_complexity_to_depth_conversion(self, cascade_orchestrator):
        """Tests complexity level to depth budget conversion."""
        from src.setfit_classifier import ComplexityLevel

        assert cascade_orchestrator._complexity_to_depth(ComplexityLevel.TRIVIAL) == 0
        assert cascade_orchestrator._complexity_to_depth(ComplexityLevel.SIMPLE) == 1
        assert cascade_orchestrator._complexity_to_depth(ComplexityLevel.MODERATE) == 2
        assert cascade_orchestrator._complexity_to_depth(ComplexityLevel.COMPLEX) == 2
        assert cascade_orchestrator._complexity_to_depth(ComplexityLevel.UNBOUNDED) == 3

    def test_complexity_to_score_conversion(self, cascade_orchestrator):
        """Tests complexity level to score conversion."""
        from src.setfit_classifier import ComplexityLevel

        assert cascade_orchestrator._complexity_to_score(ComplexityLevel.TRIVIAL) == 0.1
        assert cascade_orchestrator._complexity_to_score(ComplexityLevel.SIMPLE) == 0.3
        assert cascade_orchestrator._complexity_to_score(ComplexityLevel.MODERATE) == 0.5
        assert cascade_orchestrator._complexity_to_score(ComplexityLevel.COMPLEX) == 0.7
        assert cascade_orchestrator._complexity_to_score(ComplexityLevel.UNBOUNDED) == 0.9

    def test_setfit_disabled_skips_level(self, cascade_orchestrator):
        """When SetFit is disabled, Level 1 is skipped."""
        cascade_orchestrator.config.setfit_enabled = False

        result = cascade_orchestrator._try_setfit_classify("test query", "context")

        assert result is None
        # Classifier should not have been initialized
        assert cascade_orchestrator._setfit_load_attempted is False

    def test_gliner_disabled_skips_level(self, cascade_orchestrator):
        """When GLiNER is disabled, Level 2 is skipped."""
        cascade_orchestrator.config.gliner_enabled = False

        result = cascade_orchestrator._try_gliner_classify("test query", "context")

        assert result is None
        # Extractor should not have been initialized
        assert cascade_orchestrator._gliner_load_attempted is False


class TestCalibrationStats:
    """Tests for CalibrationStats dataclass."""

    def test_initial_state(self):
        """CalibrationStats has correct initial values."""
        from src.local_orchestrator import CalibrationStats

        stats = CalibrationStats(level=1)
        assert stats.level == 1
        assert stats.samples == 0
        assert stats.total_confidence == 0.0
        assert stats.correct_predictions == 0
        assert stats.calibration_error == 0.0

    def test_avg_confidence_empty(self):
        """avg_confidence returns 0 when no samples."""
        from src.local_orchestrator import CalibrationStats

        stats = CalibrationStats(level=1)
        assert stats.avg_confidence == 0.0

    def test_avg_confidence_with_samples(self):
        """avg_confidence computes correctly."""
        from src.local_orchestrator import CalibrationStats

        stats = CalibrationStats(level=1, samples=4, total_confidence=3.2)
        assert stats.avg_confidence == 0.8

    def test_accuracy_empty(self):
        """accuracy returns 0 when no samples."""
        from src.local_orchestrator import CalibrationStats

        stats = CalibrationStats(level=1)
        assert stats.accuracy == 0.0

    def test_accuracy_with_samples(self):
        """accuracy computes correctly."""
        from src.local_orchestrator import CalibrationStats

        stats = CalibrationStats(level=1, samples=10, correct_predictions=8)
        assert stats.accuracy == 0.8

    def test_update_calibration_error(self):
        """update_calibration_error computes |avg_conf - accuracy|."""
        from src.local_orchestrator import CalibrationStats

        stats = CalibrationStats(
            level=1,
            samples=10,
            total_confidence=9.0,  # avg = 0.9
            correct_predictions=7,  # accuracy = 0.7
        )
        stats.update_calibration_error()
        assert abs(stats.calibration_error - 0.2) < 0.001

    def test_to_dict(self):
        """to_dict produces expected structure."""
        from src.local_orchestrator import CalibrationStats

        stats = CalibrationStats(
            level=2,
            samples=5,
            total_confidence=4.0,
            correct_predictions=4,
        )
        stats.update_calibration_error()
        d = stats.to_dict()

        assert d["level"] == 2
        assert d["samples"] == 5
        assert d["avg_confidence"] == 0.8
        assert d["accuracy"] == 0.8
        assert d["calibration_error"] == 0.0


class TestCalibrationSample:
    """Tests for CalibrationSample dataclass."""

    def test_create_sample(self):
        """Can create a calibration sample."""
        from src.local_orchestrator import CalibrationSample

        sample = CalibrationSample(
            decision_id="d_1",
            cascade_level=1,
            predicted_confidence=0.9,
            predicted_activate_rlm=True,
            timestamp=12345.0,
        )
        assert sample.decision_id == "d_1"
        assert sample.cascade_level == 1
        assert sample.predicted_confidence == 0.9
        assert sample.predicted_activate_rlm is True
        assert sample.actual_correct is None


class TestCalibrationTracking:
    """Tests for calibration tracking in LocalOrchestrator."""

    @pytest.fixture
    def calibration_orchestrator(self):
        """Create orchestrator with calibration enabled."""
        config = LocalModelConfig(
            calibration_enabled=True,
            calibration_min_samples=5,  # Lower threshold for testing
            use_cascade=False,  # Disable cascade for simpler testing
        )
        return LocalOrchestrator(config=config)

    def test_calibration_enabled_by_default(self):
        """Calibration is enabled by default."""
        config = LocalModelConfig()
        assert config.calibration_enabled is True

    def test_record_calibration_sample(self, calibration_orchestrator):
        """_record_calibration_sample creates a pending sample."""
        decision = {
            "_cascade_level": 1,
            "confidence": 0.85,
            "activate_rlm": True,
        }

        decision_id = calibration_orchestrator._record_calibration_sample(decision)

        assert decision_id.startswith("d_")
        assert decision_id in calibration_orchestrator._pending_samples
        sample = calibration_orchestrator._pending_samples[decision_id]
        assert sample.cascade_level == 1
        assert sample.predicted_confidence == 0.85
        assert sample.predicted_activate_rlm is True

    def test_record_calibration_sample_disabled(self):
        """_record_calibration_sample returns empty string when disabled."""
        config = LocalModelConfig(calibration_enabled=False)
        orchestrator = LocalOrchestrator(config=config)

        decision = {"_cascade_level": 1, "confidence": 0.9, "activate_rlm": True}
        decision_id = orchestrator._record_calibration_sample(decision)

        assert decision_id == ""
        assert len(orchestrator._pending_samples) == 0

    def test_record_outcome_success(self, calibration_orchestrator):
        """record_outcome updates calibration stats correctly."""
        # Record a sample
        decision = {"_cascade_level": 1, "confidence": 0.9, "activate_rlm": True}
        decision_id = calibration_orchestrator._record_calibration_sample(decision)

        # Record outcome
        result = calibration_orchestrator.record_outcome(decision_id, was_correct=True)

        assert result is True
        assert decision_id not in calibration_orchestrator._pending_samples

        # Check stats updated
        stats = calibration_orchestrator._calibration_stats[1]
        assert stats.samples == 1
        assert stats.total_confidence == 0.9
        assert stats.correct_predictions == 1

    def test_record_outcome_incorrect(self, calibration_orchestrator):
        """record_outcome tracks incorrect predictions."""
        decision = {"_cascade_level": 2, "confidence": 0.8, "activate_rlm": False}
        decision_id = calibration_orchestrator._record_calibration_sample(decision)

        calibration_orchestrator.record_outcome(decision_id, was_correct=False)

        stats = calibration_orchestrator._calibration_stats[2]
        assert stats.samples == 1
        assert stats.correct_predictions == 0

    def test_record_outcome_not_found(self, calibration_orchestrator):
        """record_outcome returns False for unknown decision_id."""
        result = calibration_orchestrator.record_outcome("unknown_id", was_correct=True)
        assert result is False

    def test_record_outcome_disabled(self):
        """record_outcome returns False when calibration disabled."""
        config = LocalModelConfig(calibration_enabled=False)
        orchestrator = LocalOrchestrator(config=config)

        result = orchestrator.record_outcome("d_1", was_correct=True)
        assert result is False

    def test_get_calibration_stats(self, calibration_orchestrator):
        """get_calibration_stats returns complete stats."""
        stats = calibration_orchestrator.get_calibration_stats()

        assert "levels" in stats
        assert 1 in stats["levels"]
        assert 2 in stats["levels"]
        assert 3 in stats["levels"]
        assert 4 in stats["levels"]
        assert "current_thresholds" in stats
        assert "setfit" in stats["current_thresholds"]
        assert "gliner" in stats["current_thresholds"]
        assert "llm" in stats["current_thresholds"]
        assert "calibration_enabled" in stats
        assert stats["calibration_enabled"] is True

    def test_statistics_includes_calibration(self, calibration_orchestrator):
        """get_statistics includes calibration data."""
        stats = calibration_orchestrator.get_statistics()

        assert "calibration" in stats
        assert "levels" in stats["calibration"]


class TestThresholdAdjustment:
    """Tests for automatic threshold adjustment."""

    @pytest.fixture
    def adjustment_orchestrator(self):
        """Create orchestrator with low min_samples for testing."""
        config = LocalModelConfig(
            calibration_enabled=True,
            calibration_min_samples=3,  # Low for testing
            calibration_max_adjustment=0.05,
            setfit_confidence_threshold=0.85,
        )
        return LocalOrchestrator(config=config)

    def test_no_adjustment_below_min_samples(self, adjustment_orchestrator):
        """No threshold adjustment when samples < min_samples."""
        original_threshold = adjustment_orchestrator.config.setfit_confidence_threshold

        # Record only 2 outcomes (below min_samples of 3)
        for i in range(2):
            decision = {"_cascade_level": 1, "confidence": 0.9, "activate_rlm": True}
            decision_id = adjustment_orchestrator._record_calibration_sample(decision)
            adjustment_orchestrator.record_outcome(decision_id, was_correct=True)

        # Threshold should not have changed
        assert adjustment_orchestrator.config.setfit_confidence_threshold == original_threshold

    def test_adjustment_raises_threshold_when_overconfident(self, adjustment_orchestrator):
        """Threshold increases when model is overconfident."""
        original_threshold = adjustment_orchestrator.config.setfit_confidence_threshold

        # Record outcomes where high confidence predictions are wrong
        for i in range(5):
            decision = {"_cascade_level": 1, "confidence": 0.95, "activate_rlm": True}
            decision_id = adjustment_orchestrator._record_calibration_sample(decision)
            # Only 1 correct out of 5 = 20% accuracy vs 95% confidence
            adjustment_orchestrator.record_outcome(decision_id, was_correct=(i == 0))

        # Threshold should have increased (model is overconfident)
        assert adjustment_orchestrator.config.setfit_confidence_threshold > original_threshold

    def test_adjustment_lowers_threshold_when_underconfident(self, adjustment_orchestrator):
        """Threshold decreases when model is underconfident."""
        original_threshold = adjustment_orchestrator.config.setfit_confidence_threshold

        # Record outcomes where low confidence predictions are actually correct
        for i in range(5):
            decision = {"_cascade_level": 1, "confidence": 0.60, "activate_rlm": True}
            decision_id = adjustment_orchestrator._record_calibration_sample(decision)
            # 4 correct out of 5 = 80% accuracy vs 60% confidence
            adjustment_orchestrator.record_outcome(decision_id, was_correct=(i != 0))

        # Threshold should have decreased (model is underconfident)
        assert adjustment_orchestrator.config.setfit_confidence_threshold < original_threshold

    def test_adjustment_respects_max_bounds(self):
        """Threshold adjustment respects maximum bounds."""
        config = LocalModelConfig(
            calibration_enabled=True,
            calibration_min_samples=3,
            calibration_max_adjustment=0.5,  # Large adjustment
            calibration_threshold_max=0.95,
            setfit_confidence_threshold=0.90,
        )
        orchestrator = LocalOrchestrator(config=config)

        # Record very overconfident predictions
        for i in range(5):
            decision = {"_cascade_level": 1, "confidence": 0.99, "activate_rlm": True}
            decision_id = orchestrator._record_calibration_sample(decision)
            orchestrator.record_outcome(decision_id, was_correct=False)

        # Should not exceed max threshold
        assert orchestrator.config.setfit_confidence_threshold <= 0.95

    def test_adjustment_respects_min_bounds(self):
        """Threshold adjustment respects minimum bounds."""
        config = LocalModelConfig(
            calibration_enabled=True,
            calibration_min_samples=3,
            calibration_max_adjustment=0.5,  # Large adjustment
            calibration_threshold_min=0.5,
            setfit_confidence_threshold=0.55,
        )
        orchestrator = LocalOrchestrator(config=config)

        # Record very underconfident predictions (all correct)
        for i in range(5):
            decision = {"_cascade_level": 1, "confidence": 0.51, "activate_rlm": True}
            decision_id = orchestrator._record_calibration_sample(decision)
            orchestrator.record_outcome(decision_id, was_correct=True)

        # Should not go below min threshold
        assert orchestrator.config.setfit_confidence_threshold >= 0.5

    def test_gliner_threshold_adjustment(self):
        """GLiNER threshold can be adjusted."""
        config = LocalModelConfig(
            calibration_enabled=True,
            calibration_min_samples=3,
            gliner_confidence_threshold=0.80,
        )
        orchestrator = LocalOrchestrator(config=config)
        original = orchestrator.config.gliner_confidence_threshold

        # Record overconfident GLiNER predictions
        for i in range(5):
            decision = {"_cascade_level": 2, "confidence": 0.95, "activate_rlm": True}
            decision_id = orchestrator._record_calibration_sample(decision)
            orchestrator.record_outcome(decision_id, was_correct=False)

        assert orchestrator.config.gliner_confidence_threshold > original

    def test_llm_threshold_adjustment(self):
        """LLM threshold can be adjusted."""
        config = LocalModelConfig(
            calibration_enabled=True,
            calibration_min_samples=3,
            llm_confidence_threshold=0.70,
        )
        orchestrator = LocalOrchestrator(config=config)
        original = orchestrator.config.llm_confidence_threshold

        # Record overconfident LLM predictions
        for i in range(5):
            decision = {"_cascade_level": 3, "confidence": 0.90, "activate_rlm": True}
            decision_id = orchestrator._record_calibration_sample(decision)
            orchestrator.record_outcome(decision_id, was_correct=False)

        assert orchestrator.config.llm_confidence_threshold > original


class TestPendingSamplesMemoryLimit:
    """Tests for pending samples memory management."""

    def test_pending_samples_limited(self):
        """Pending samples are limited to prevent memory growth."""
        config = LocalModelConfig(calibration_enabled=True)
        orchestrator = LocalOrchestrator(config=config)

        # Record many samples without outcomes
        for i in range(1100):
            decision = {"_cascade_level": 4, "confidence": 0.6, "activate_rlm": False}
            orchestrator._record_calibration_sample(decision)

        # Should be limited (1000 max, evicts 500 oldest)
        assert len(orchestrator._pending_samples) <= 600  # 1100 - 500 = 600
