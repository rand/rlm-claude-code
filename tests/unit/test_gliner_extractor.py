"""
Unit tests for GLiNER extractor module.

Tests cover both GLiNER-enabled and fallback modes.
"""

from src.gliner_extractor import (
    SIGNAL_SCHEMA,
    SIGNAL_WEIGHTS,
    ExtractedSignal,
    GLiNERConfig,
    GLiNERExtractor,
    SignalExtractionResult,
    is_gliner_available,
)


class TestSignalSchema:
    """Tests for SIGNAL_SCHEMA constant."""

    def test_expected_signal_types(self):
        """All expected signal types exist."""
        expected = [
            "discovery",
            "synthesis",
            "debugging",
            "architectural",
            "uncertainty",
            "multi_file",
        ]
        for signal_type in expected:
            assert signal_type in SIGNAL_SCHEMA

    def test_signal_keywords_not_empty(self):
        """Each signal type has keywords."""
        for signal_type, keywords in SIGNAL_SCHEMA.items():
            assert len(keywords) > 0, f"{signal_type} has no keywords"

    def test_discovery_keywords(self):
        """Discovery signal has expected keywords."""
        keywords = SIGNAL_SCHEMA["discovery"]
        assert "exploration" in keywords
        assert "finding" in keywords
        assert "searching" in keywords

    def test_debugging_keywords(self):
        """Debugging signal has expected keywords."""
        keywords = SIGNAL_SCHEMA["debugging"]
        assert "error" in keywords
        assert "bug" in keywords
        assert "fix" in keywords


class TestSignalWeights:
    """Tests for SIGNAL_WEIGHTS constant."""

    def test_all_signal_types_have_weights(self):
        """All signal types in schema have weights."""
        for signal_type in SIGNAL_SCHEMA:
            assert signal_type in SIGNAL_WEIGHTS

    def test_weights_in_valid_range(self):
        """All weights are between 0 and 1."""
        for signal_type, weight in SIGNAL_WEIGHTS.items():
            assert 0 <= weight <= 1, f"{signal_type} weight {weight} out of range"

    def test_architectural_high_weight(self):
        """Architectural signals have high weight."""
        assert SIGNAL_WEIGHTS["architectural"] >= 0.8

    def test_debugging_moderate_weight(self):
        """Debugging signals have moderate weight."""
        assert 0.4 <= SIGNAL_WEIGHTS["debugging"] <= 0.6


class TestExtractedSignal:
    """Tests for ExtractedSignal dataclass."""

    def test_create_signal(self):
        """Can create an extracted signal."""
        signal = ExtractedSignal(
            signal_type="debugging",
            text="bug",
            score=0.9,
            start=10,
            end=13,
        )
        assert signal.signal_type == "debugging"
        assert signal.text == "bug"
        assert signal.score == 0.9
        assert signal.start == 10
        assert signal.end == 13


class TestSignalExtractionResult:
    """Tests for SignalExtractionResult dataclass."""

    def test_create_empty_result(self):
        """Can create empty extraction result."""
        result = SignalExtractionResult()
        assert result.signals == []
        assert result.signal_types == []
        assert result.complexity_score == 0.0

    def test_create_with_signals(self):
        """Can create result with signals."""
        signals = [
            ExtractedSignal("debugging", "error", 0.8, 0, 5),
            ExtractedSignal("discovery", "find", 0.7, 10, 14),
        ]
        result = SignalExtractionResult(
            signals=signals,
            signal_types=["debugging", "discovery"],
            complexity_score=0.6,
            latency_ms=8.5,
            model_used="gliner",
        )
        assert len(result.signals) == 2
        assert "debugging" in result.signal_types
        assert result.complexity_score == 0.6

    def test_to_dict(self):
        """to_dict produces expected structure."""
        signals = [ExtractedSignal("debugging", "bug", 0.9, 0, 3)]
        result = SignalExtractionResult(
            signals=signals,
            signal_types=["debugging"],
            complexity_score=0.5,
            latency_ms=5.0,
            model_used="gliner",
        )
        d = result.to_dict()
        assert d["signal_types"] == ["debugging"]
        assert d["complexity_score"] == 0.5
        assert len(d["signals"]) == 1
        assert d["signals"][0]["type"] == "debugging"


class TestGLiNERConfig:
    """Tests for GLiNERConfig dataclass."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = GLiNERConfig()
        assert config.confidence_threshold == 0.5
        assert config.flat_ner is True
        assert config.max_length == 512
        assert config.fallback_on_error is True

    def test_custom_config(self):
        """Can create custom config."""
        config = GLiNERConfig(
            model_name="custom/model",
            confidence_threshold=0.7,
        )
        assert config.model_name == "custom/model"
        assert config.confidence_threshold == 0.7


class TestGLiNERExtractor:
    """Tests for GLiNERExtractor class."""

    def test_create_extractor(self):
        """Can create extractor instance."""
        extractor = GLiNERExtractor()
        assert extractor is not None
        assert extractor.config is not None

    def test_create_with_config(self):
        """Can create extractor with custom config."""
        config = GLiNERConfig(confidence_threshold=0.6)
        extractor = GLiNERExtractor(config)
        assert extractor.config.confidence_threshold == 0.6

    def test_is_available_property(self):
        """is_available property works."""
        extractor = GLiNERExtractor()
        # Should return bool regardless of actual availability
        assert isinstance(extractor.is_available, bool)

    def test_extract_signals_returns_result(self):
        """extract_signals() returns SignalExtractionResult."""
        extractor = GLiNERExtractor()
        result = extractor.extract_signals("Fix the bug in the authentication module")
        assert isinstance(result, SignalExtractionResult)
        assert isinstance(result.complexity_score, float)

    def test_extract_signals_measures_latency(self):
        """extract_signals() measures latency."""
        extractor = GLiNERExtractor()
        result = extractor.extract_signals("test query")
        assert result.latency_ms > 0


class TestFallbackExtraction:
    """Tests for keyword-based fallback extraction."""

    def test_detects_debugging_keywords(self):
        """Fallback detects debugging keywords."""
        extractor = GLiNERExtractor()
        result = extractor._fallback_extract("Fix this error in the code", 0)
        signal_types = [s.signal_type for s in result.signals]
        assert "debugging" in signal_types

    def test_detects_discovery_keywords(self):
        """Fallback detects discovery keywords."""
        extractor = GLiNERExtractor()
        result = extractor._fallback_extract("I'm searching for the function", 0)
        signal_types = [s.signal_type for s in result.signals]
        assert "discovery" in signal_types

    def test_detects_architectural_keywords(self):
        """Fallback detects architectural keywords."""
        extractor = GLiNERExtractor()
        result = extractor._fallback_extract("Refactor the design pattern", 0)
        signal_types = [s.signal_type for s in result.signals]
        assert "architectural" in signal_types

    def test_detects_synthesis_keywords(self):
        """Fallback detects synthesis keywords."""
        extractor = GLiNERExtractor()
        result = extractor._fallback_extract("Combine these modules together", 0)
        signal_types = [s.signal_type for s in result.signals]
        assert "synthesis" in signal_types

    def test_detects_multi_file_keywords(self):
        """Fallback detects multi-file keywords."""
        extractor = GLiNERExtractor()
        result = extractor._fallback_extract("Update this across multiple files", 0)
        signal_types = [s.signal_type for s in result.signals]
        assert "multi_file" in signal_types

    def test_fallback_uses_keyword_model(self):
        """Fallback uses 'keyword_fallback' as model name."""
        extractor = GLiNERExtractor()
        result = extractor._fallback_extract("test query", 0)
        assert result.model_used == "keyword_fallback"

    def test_fallback_moderate_confidence(self):
        """Fallback signals have moderate confidence."""
        extractor = GLiNERExtractor()
        result = extractor._fallback_extract("fix this bug", 0)
        for signal in result.signals:
            assert signal.score == 0.6

    def test_no_signals_for_generic_query(self):
        """Generic queries may have no signals."""
        extractor = GLiNERExtractor()
        result = extractor._fallback_extract("hello world", 0)
        # Should not crash, may or may not have signals
        assert isinstance(result.signals, list)


class TestComplexityScoring:
    """Tests for complexity score computation."""

    def test_empty_signals_zero_score(self):
        """Empty signals produce zero complexity score."""
        extractor = GLiNERExtractor()
        score = extractor._compute_complexity_score([])
        assert score == 0.0

    def test_single_signal_produces_score(self):
        """Single signal produces non-zero score."""
        extractor = GLiNERExtractor()
        signals = [ExtractedSignal("debugging", "bug", 0.8, 0, 3)]
        score = extractor._compute_complexity_score(signals)
        assert score > 0

    def test_architectural_signal_high_score(self):
        """Architectural signal produces high score."""
        extractor = GLiNERExtractor()
        signals = [ExtractedSignal("architectural", "design", 0.9, 0, 6)]
        score = extractor._compute_complexity_score(signals)
        # architectural has weight 0.9, so score should be high
        assert score >= 0.7

    def test_multiple_signals_combined(self):
        """Multiple signals are combined in score."""
        extractor = GLiNERExtractor()
        signals = [
            ExtractedSignal("debugging", "bug", 0.8, 0, 3),
            ExtractedSignal("discovery", "find", 0.7, 5, 9),
        ]
        score = extractor._compute_complexity_score(signals)
        assert score > 0

    def test_score_bounded_to_one(self):
        """Score is bounded at 1.0."""
        extractor = GLiNERExtractor()
        # Many high-confidence signals
        signals = [
            ExtractedSignal("debugging", "bug", 1.0, 0, 3),
            ExtractedSignal("discovery", "find", 1.0, 5, 9),
            ExtractedSignal("architectural", "design", 1.0, 10, 16),
            ExtractedSignal("synthesis", "combine", 1.0, 20, 27),
        ]
        score = extractor._compute_complexity_score(signals)
        assert score <= 1.0


class TestWeightedRulesResult:
    """Tests for weighted rules application."""

    def test_high_weight_signals_activate(self):
        """High-weight signals trigger activation."""
        extractor = GLiNERExtractor()
        signals = SignalExtractionResult(
            signals=[ExtractedSignal("architectural", "design", 0.8, 0, 6)],
            signal_types=["architectural"],
            complexity_score=0.5,
        )
        result = extractor.get_weighted_rules_result(signals)
        assert result["activate_rlm"] is True
        assert "high-weight signals" in result["reason"]

    def test_high_complexity_activates(self):
        """High complexity score triggers activation."""
        extractor = GLiNERExtractor()
        signals = SignalExtractionResult(
            signals=[ExtractedSignal("uncertainty", "maybe", 0.8, 0, 5)],
            signal_types=["uncertainty"],
            complexity_score=0.6,  # Above 0.5 threshold
        )
        result = extractor.get_weighted_rules_result(signals)
        assert result["activate_rlm"] is True
        assert "complexity score" in result["reason"]

    def test_low_complexity_no_activation(self):
        """Low complexity without high-weight signals doesn't activate."""
        extractor = GLiNERExtractor()
        signals = SignalExtractionResult(
            signals=[],
            signal_types=[],
            complexity_score=0.2,
        )
        result = extractor.get_weighted_rules_result(signals)
        assert result["activate_rlm"] is False
        assert "no complexity signals" in result["reason"]

    def test_result_includes_confidence(self):
        """Result includes confidence score."""
        extractor = GLiNERExtractor()
        signals = SignalExtractionResult(
            signals=[],
            signal_types=[],
            complexity_score=0.3,
        )
        result = extractor.get_weighted_rules_result(signals)
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1


class TestIsGLiNERAvailable:
    """Tests for is_gliner_available function."""

    def test_returns_bool(self):
        """is_gliner_available returns boolean."""
        result = is_gliner_available()
        assert isinstance(result, bool)


class TestHotSwapping:
    """Tests for model hot-swapping functionality."""

    def test_model_mtime_none_when_no_local_path(self):
        """_get_model_mtime returns None when no local model_path set."""
        # Default config uses HuggingFace model name, not local path
        extractor = GLiNERExtractor()
        assert extractor._get_model_mtime() is None

    def test_model_mtime_none_when_path_not_exists(self):
        """_get_model_mtime returns None when local path doesn't exist."""
        config = GLiNERConfig(model_path="/nonexistent/path")
        extractor = GLiNERExtractor(config)
        assert extractor._get_model_mtime() is None

    def test_check_model_changed_false_when_not_loaded(self):
        """_check_model_changed returns False when model not loaded."""
        extractor = GLiNERExtractor()
        assert extractor._check_model_changed() is False

    def test_check_model_changed_false_when_no_mtime(self):
        """_check_model_changed returns False when mtime is None."""
        extractor = GLiNERExtractor()
        extractor._loaded = True  # Simulate loaded state
        extractor._model_mtime = None
        assert extractor._check_model_changed() is False

    def test_reload_model_resets_state(self):
        """reload_model resets internal state."""
        # Use a non-existent local path to prevent HuggingFace download
        config = GLiNERConfig(model_path="/nonexistent/path/to/model")
        extractor = GLiNERExtractor(config=config)
        # Simulate a loaded state
        extractor._model = "mock_model"
        extractor._loaded = True
        extractor._load_attempted = True
        extractor._model_mtime = 12345.0

        # Reload (will fail since model path doesn't exist)
        result = extractor.reload_model()

        # State should be reset
        assert extractor._model is None
        assert extractor._loaded is False
        # load_attempted should be True since _try_load was called
        assert extractor._load_attempted is True
        assert extractor._model_mtime is None
        # Result should be False since model path doesn't exist
        assert result is False

    def test_initial_model_mtime_is_none(self):
        """Model mtime is None on initialization."""
        extractor = GLiNERExtractor()
        assert extractor._model_mtime is None
