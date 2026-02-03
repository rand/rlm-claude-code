"""
Unit tests for SetFit classifier module.

Tests cover both SetFit-enabled and fallback modes.
"""

from src.setfit_classifier import (
    LEVEL_ACTIVATION_MAP,
    SIGNAL_COMPLEXITY_HINTS,
    ComplexityClassification,
    ComplexityLevel,
    SetFitClassifier,
    SetFitConfig,
    is_setfit_available,
)


class TestComplexityLevel:
    """Tests for ComplexityLevel enum."""

    def test_all_levels_exist(self):
        """All expected complexity levels exist."""
        levels = [level.value for level in ComplexityLevel]
        assert "trivial" in levels
        assert "simple" in levels
        assert "moderate" in levels
        assert "complex" in levels
        assert "unbounded" in levels

    def test_level_count(self):
        """Exactly 5 complexity levels."""
        assert len(ComplexityLevel) == 5


class TestComplexityClassification:
    """Tests for ComplexityClassification dataclass."""

    def test_create_basic_classification(self):
        """Can create a basic classification."""
        classification = ComplexityClassification(
            activate_rlm=True,
            complexity_level=ComplexityLevel.COMPLEX,
        )
        assert classification.activate_rlm is True
        assert classification.complexity_level == ComplexityLevel.COMPLEX
        assert classification.signals == []
        assert classification.confidence == 0.0

    def test_create_with_all_fields(self):
        """Can create classification with all fields."""
        classification = ComplexityClassification(
            activate_rlm=True,
            complexity_level=ComplexityLevel.MODERATE,
            signals=["discovery_required", "debugging_deep"],
            confidence=0.85,
            latency_ms=5.2,
            model_used="setfit",
        )
        assert classification.activate_rlm is True
        assert classification.confidence == 0.85
        assert "discovery_required" in classification.signals
        assert classification.model_used == "setfit"

    def test_to_dict(self):
        """to_dict produces expected structure."""
        classification = ComplexityClassification(
            activate_rlm=False,
            complexity_level=ComplexityLevel.SIMPLE,
            signals=["narrow_scope"],
            confidence=0.9,
        )
        result = classification.to_dict()
        assert result["activate_rlm"] is False
        assert result["complexity_level"] == "simple"
        assert result["signals"] == ["narrow_scope"]
        assert result["confidence"] == 0.9


class TestLevelActivationMap:
    """Tests for complexity level to activation mapping."""

    def test_trivial_no_activation(self):
        """Trivial level doesn't activate RLM."""
        assert LEVEL_ACTIVATION_MAP[ComplexityLevel.TRIVIAL] is False

    def test_simple_no_activation(self):
        """Simple level doesn't activate RLM."""
        assert LEVEL_ACTIVATION_MAP[ComplexityLevel.SIMPLE] is False

    def test_moderate_activates(self):
        """Moderate level activates RLM."""
        assert LEVEL_ACTIVATION_MAP[ComplexityLevel.MODERATE] is True

    def test_complex_activates(self):
        """Complex level activates RLM."""
        assert LEVEL_ACTIVATION_MAP[ComplexityLevel.COMPLEX] is True

    def test_unbounded_activates(self):
        """Unbounded level activates RLM."""
        assert LEVEL_ACTIVATION_MAP[ComplexityLevel.UNBOUNDED] is True


class TestSetFitConfig:
    """Tests for SetFitConfig dataclass."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = SetFitConfig()
        assert config.confidence_threshold == 0.85
        assert config.max_length == 512
        assert config.fallback_on_error is True

    def test_custom_config(self):
        """Can create custom config."""
        config = SetFitConfig(
            model_path="/custom/path",
            confidence_threshold=0.9,
        )
        assert config.model_path == "/custom/path"
        assert config.confidence_threshold == 0.9


class TestSetFitClassifier:
    """Tests for SetFitClassifier class."""

    def test_create_classifier(self):
        """Can create classifier instance."""
        classifier = SetFitClassifier()
        assert classifier is not None
        assert classifier.config is not None

    def test_create_with_config(self):
        """Can create classifier with custom config."""
        config = SetFitConfig(confidence_threshold=0.7)
        classifier = SetFitClassifier(config)
        assert classifier.config.confidence_threshold == 0.7

    def test_is_available_property(self):
        """is_available property works."""
        classifier = SetFitClassifier()
        # Should return bool regardless of actual availability
        assert isinstance(classifier.is_available, bool)

    def test_classify_returns_classification(self):
        """classify() returns ComplexityClassification."""
        classifier = SetFitClassifier()
        result = classifier.classify("How do I fix this bug?")
        assert isinstance(result, ComplexityClassification)
        assert isinstance(result.complexity_level, ComplexityLevel)
        assert 0 <= result.confidence <= 1

    def test_classify_with_context(self):
        """classify() accepts context summary."""
        classifier = SetFitClassifier()
        result = classifier.classify(
            "Refactor the authentication module",
            context_summary="Large codebase with 500 files",
        )
        assert isinstance(result, ComplexityClassification)


class TestFallbackClassification:
    """Tests for heuristic fallback classification."""

    def test_debug_query_detected(self):
        """Debug-related queries are detected."""
        classifier = SetFitClassifier()
        result = classifier._fallback_classify("Fix the broken test", "", 0)
        assert "debugging_deep" in result.signals
        assert result.complexity_level in (ComplexityLevel.COMPLEX, ComplexityLevel.MODERATE)

    def test_architectural_query_detected(self):
        """Architecture-related queries are detected."""
        classifier = SetFitClassifier()
        result = classifier._fallback_classify("Refactor the entire module structure", "", 0)
        assert "architectural" in result.signals
        assert result.activate_rlm is True

    def test_discovery_query_detected(self):
        """Discovery-related queries are detected."""
        classifier = SetFitClassifier()
        result = classifier._fallback_classify("Find all usages across the codebase", "", 0)
        assert "discovery_required" in result.signals

    def test_simple_query_detected(self):
        """Simple queries are detected."""
        classifier = SetFitClassifier()
        result = classifier._fallback_classify("What is x?", "", 0)
        assert result.complexity_level == ComplexityLevel.TRIVIAL
        assert result.activate_rlm is False

    def test_fallback_uses_heuristic_model(self):
        """Fallback uses 'heuristic_fallback' as model name."""
        classifier = SetFitClassifier()
        result = classifier._fallback_classify("test query", "", 0)
        assert result.model_used == "heuristic_fallback"

    def test_fallback_low_confidence(self):
        """Fallback has low confidence."""
        classifier = SetFitClassifier()
        result = classifier._fallback_classify("test query", "", 0)
        assert result.confidence == 0.5


class TestSignalInference:
    """Tests for signal inference from queries."""

    def test_infer_discovery_signal(self):
        """Discovery signals are inferred from keywords."""
        classifier = SetFitClassifier()
        signals = classifier._infer_signals(
            ComplexityLevel.COMPLEX, "Find where this function is used"
        )
        assert "discovery_required" in signals

    def test_infer_synthesis_signal(self):
        """Synthesis signals are inferred from keywords."""
        classifier = SetFitClassifier()
        signals = classifier._infer_signals(
            ComplexityLevel.COMPLEX, "Combine these modules together"
        )
        assert "synthesis_required" in signals

    def test_infer_debugging_signal(self):
        """Debugging signals are inferred from keywords."""
        classifier = SetFitClassifier()
        signals = classifier._infer_signals(ComplexityLevel.COMPLEX, "Debug this error message")
        assert "debugging_deep" in signals

    def test_infer_conversational_signal(self):
        """Conversational signals are inferred from short questions."""
        classifier = SetFitClassifier()
        signals = classifier._infer_signals(ComplexityLevel.TRIVIAL, "What is this?")
        assert "conversational" in signals

    def test_infer_narrow_scope_for_trivial(self):
        """Trivial complexity implies narrow scope."""
        classifier = SetFitClassifier()
        signals = classifier._infer_signals(ComplexityLevel.TRIVIAL, "simple task")
        assert "narrow_scope" in signals


class TestIsSetFitAvailable:
    """Tests for is_setfit_available function."""

    def test_returns_bool(self):
        """is_setfit_available returns boolean."""
        result = is_setfit_available()
        assert isinstance(result, bool)


class TestSignalComplexityHints:
    """Tests for SIGNAL_COMPLEXITY_HINTS mapping."""

    def test_all_signals_have_hints(self):
        """All signal types have complexity hints."""
        expected_signals = [
            "discovery_required",
            "synthesis_required",
            "uncertainty_high",
            "debugging_deep",
            "architectural",
            "pattern_exhaustion",
            "knowledge_retrieval",
            "narrow_scope",
            "conversational",
        ]
        for signal in expected_signals:
            assert signal in SIGNAL_COMPLEXITY_HINTS

    def test_high_complexity_signals(self):
        """High-complexity signals map to complex/unbounded."""
        assert SIGNAL_COMPLEXITY_HINTS["architectural"] == ComplexityLevel.UNBOUNDED
        assert SIGNAL_COMPLEXITY_HINTS["pattern_exhaustion"] == ComplexityLevel.UNBOUNDED
        assert SIGNAL_COMPLEXITY_HINTS["discovery_required"] == ComplexityLevel.COMPLEX

    def test_low_complexity_signals(self):
        """Low-complexity signals map to trivial/simple."""
        assert SIGNAL_COMPLEXITY_HINTS["knowledge_retrieval"] == ComplexityLevel.TRIVIAL
        assert SIGNAL_COMPLEXITY_HINTS["conversational"] == ComplexityLevel.TRIVIAL
        assert SIGNAL_COMPLEXITY_HINTS["narrow_scope"] == ComplexityLevel.SIMPLE


class TestHotSwapping:
    """Tests for model hot-swapping functionality."""

    def test_model_mtime_none_when_no_model(self):
        """_get_model_mtime returns None when model doesn't exist."""
        config = SetFitConfig(model_path="/nonexistent/path")
        classifier = SetFitClassifier(config)
        assert classifier._get_model_mtime() is None

    def test_check_model_changed_false_when_not_loaded(self):
        """_check_model_changed returns False when model not loaded."""
        classifier = SetFitClassifier()
        assert classifier._check_model_changed() is False

    def test_check_model_changed_false_when_no_mtime(self):
        """_check_model_changed returns False when mtime is None."""
        classifier = SetFitClassifier()
        classifier._loaded = True  # Simulate loaded state
        classifier._model_mtime = None
        assert classifier._check_model_changed() is False

    def test_reload_model_resets_state(self):
        """reload_model resets internal state."""
        classifier = SetFitClassifier()
        # Simulate a loaded state
        classifier._model = "mock_model"
        classifier._loaded = True
        classifier._load_attempted = True
        classifier._model_mtime = 12345.0

        # Reload (will fail since no model exists)
        result = classifier.reload_model()

        # State should be reset
        assert classifier._model is None
        assert classifier._loaded is False
        # load_attempted should be True since _try_load was called
        assert classifier._load_attempted is True
        assert classifier._model_mtime is None
        # Result should be False since no model exists
        assert result is False

    def test_initial_model_mtime_is_none(self):
        """Model mtime is None on initialization."""
        classifier = SetFitClassifier()
        assert classifier._model_mtime is None
