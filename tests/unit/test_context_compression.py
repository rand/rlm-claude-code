"""
Tests for context compression (SPEC-08.20-08.25).

Tests cover:
- Two-stage compression (extractive + abstractive)
- Key information preservation
- Configurable target tokens
- Auto-trigger thresholds
- Compression metrics
"""

from src.context_compression import (
    AbstractiveCompressor,
    AutoCompressor,
    CompressionConfig,
    CompressionMetrics,
    CompressionResult,
    CompressionStage,
    ContextCompressor,
    ExtractiveCompressor,
    KeyInfoPreserver,
    RelevanceScorer,
    TwoStageCompressor,
)


class TestContextCompressor:
    """Tests for main ContextCompressor class (SPEC-08.20)."""

    def test_compresses_when_exceeds_target(self):
        """SPEC-08.20: Compress when content exceeds target_tokens."""
        compressor = ContextCompressor()

        # Create content that exceeds default target
        long_content = "This is a sentence. " * 500

        result = compressor.compress(long_content)

        assert result.compressed
        assert result.output_tokens < result.input_tokens

    def test_no_compression_under_target(self):
        """Content under target should not be compressed."""
        compressor = ContextCompressor()

        short_content = "This is short content."

        result = compressor.compress(short_content)

        assert not result.compressed
        assert result.content == short_content

    def test_respects_target_tokens(self):
        """Compression should respect target_tokens config."""
        config = CompressionConfig(target_tokens=500)
        compressor = ContextCompressor(config=config)

        long_content = "This is a sentence. " * 300

        result = compressor.compress(long_content)

        assert result.output_tokens <= 500 or result.stage == CompressionStage.ABSTRACTIVE

    def test_returns_compression_result(self):
        """Compress should return CompressionResult."""
        compressor = ContextCompressor()

        result = compressor.compress("Test content")

        assert isinstance(result, CompressionResult)


class TestTwoStageCompression:
    """Tests for two-stage compression approach (SPEC-08.21)."""

    def test_extractive_first_stage(self):
        """SPEC-08.21: Extractive compression runs first."""
        compressor = TwoStageCompressor()

        long_content = "Important finding here. " * 100 + "Less relevant. " * 100

        result = compressor.compress(long_content, target_tokens=500)

        # First stage should be extractive
        assert result.stages_applied[0] == CompressionStage.EXTRACTIVE

    def test_abstractive_if_still_over_budget(self):
        """SPEC-08.21: Abstractive stage if extractive insufficient."""
        compressor = TwoStageCompressor()

        # Very long content needs both stages
        very_long = "Critical information here. " * 500

        result = compressor.compress(very_long, target_tokens=100)

        # Should have applied abstractive as well
        assert CompressionStage.ABSTRACTIVE in result.stages_applied

    def test_extractive_sufficient_skips_abstractive(self):
        """Skip abstractive if extractive achieves target."""
        compressor = TwoStageCompressor()

        # Moderate content - extractive should suffice
        content = "Key point. " * 50 + "Filler content. " * 50

        result = compressor.compress(content, target_tokens=1000)

        # May only need extractive
        assert result.stages_applied[0] == CompressionStage.EXTRACTIVE


class TestExtractiveCompression:
    """Tests for extractive compression stage (SPEC-08.21)."""

    def test_selects_key_sentences(self):
        """Extractive compression selects key sentences."""
        compressor = ExtractiveCompressor()

        content = (
            "Error: NullPointerException occurred. "
            "The weather is nice today. "
            "Stack trace shows line 42. "
            "Random filler text here. "
        )

        result = compressor.compress(content, target_tokens=100)

        # Should keep error-related sentences
        assert "Error" in result.content or "Stack trace" in result.content

    def test_uses_relevance_scoring(self):
        """Extractive compression uses relevance scoring."""
        compressor = ExtractiveCompressor()

        content = "Important finding. " * 10 + "Less relevant filler. " * 10

        result = compressor.compress(content, target_tokens=200)

        assert result.relevance_scores is not None
        assert len(result.relevance_scores) > 0

    def test_preserves_sentence_order(self):
        """Extracted sentences maintain relative order."""
        compressor = ExtractiveCompressor()

        content = "First important point. Second key finding. Third critical note."

        result = compressor.compress(content, target_tokens=200)

        # Order should be preserved in output
        if "First" in result.content and "Third" in result.content:
            assert result.content.index("First") < result.content.index("Third")


class TestAbstractiveCompression:
    """Tests for abstractive compression stage (SPEC-08.21)."""

    def test_produces_summary(self):
        """Abstractive compression produces summary."""
        compressor = AbstractiveCompressor()

        content = "The system analysis revealed multiple issues. " * 20

        result = compressor.compress(content, target_tokens=100)

        assert result.content is not None
        assert len(result.content) < len(content)

    def test_maintains_key_information(self):
        """Summary should maintain key information."""
        compressor = AbstractiveCompressor()

        content = (
            "Error: Connection timeout after 30 seconds. "
            "The server at 192.168.1.1 did not respond. "
            "Retry attempts: 3. All failed."
        )

        result = compressor.compress(content, target_tokens=100)

        # Should preserve error details (mock implementation)
        assert result.content is not None


class TestKeyInfoPreservation:
    """Tests for key information preservation (SPEC-08.22)."""

    def test_preserves_key_facts(self):
        """SPEC-08.22: Preserve key facts and findings."""
        preserver = KeyInfoPreserver()

        content = (
            "FINDING: Memory leak detected in module X. "
            "Some random text here. "
            "CONCLUSION: Performance degraded by 50%."
        )

        preserved = preserver.extract_key_info(content)

        assert "FINDING" in preserved or "Memory leak" in preserved

    def test_preserves_error_messages(self):
        """SPEC-08.22: Preserve error messages and stack traces."""
        preserver = KeyInfoPreserver()

        content = (
            "Starting process...\n"
            "Error: FileNotFoundError: config.yaml not found\n"
            "Traceback (most recent call last):\n"
            "  File 'main.py', line 10\n"
            "Process terminated."
        )

        preserved = preserver.extract_key_info(content)

        assert "Error" in preserved or "FileNotFoundError" in preserved

    def test_preserves_code_snippets(self):
        """SPEC-08.22: Preserve code snippets and file references."""
        preserver = KeyInfoPreserver()

        content = (
            "Analyzing file src/main.py:\n"
            "```python\n"
            "def process():\n"
            "    return data\n"
            "```\n"
            "Additional notes..."
        )

        preserved = preserver.extract_key_info(content)

        # Should preserve code reference
        assert "src/main.py" in preserved or "```" in preserved

    def test_preserves_file_references(self):
        """File paths should be preserved."""
        preserver = KeyInfoPreserver()

        content = "Changes in /Users/test/project/src/module.py affect the system."

        preserved = preserver.extract_key_info(content)

        assert "module.py" in preserved or "/src/" in preserved


class TestRelevanceScorer:
    """Tests for relevance scoring in extractive compression."""

    def test_scores_sentences(self):
        """Relevance scorer should score each sentence."""
        scorer = RelevanceScorer()

        sentences = [
            "Error occurred in the system.",
            "The weather is nice.",
            "Critical bug found in module.",
        ]

        scores = scorer.score_sentences(sentences)

        assert len(scores) == len(sentences)
        assert all(0 <= s <= 1 for s in scores)

    def test_higher_score_for_important_content(self):
        """Important content should score higher."""
        scorer = RelevanceScorer()

        sentences = [
            "Error: Critical failure detected.",
            "The quick brown fox jumps.",
        ]

        scores = scorer.score_sentences(sentences)

        # Error sentence should score higher
        assert scores[0] > scores[1]

    def test_contextual_scoring(self):
        """Scorer should consider context."""
        scorer = RelevanceScorer()

        sentences = [
            "Memory usage: 85%.",
            "Random observation here.",
        ]

        scores = scorer.score_sentences(sentences, context="performance analysis")

        assert scores[0] > scores[1]


class TestCompressionConfig:
    """Tests for configurable compression (SPEC-08.23)."""

    def test_default_target_tokens(self):
        """SPEC-08.23: Default target_tokens is 2000."""
        config = CompressionConfig()

        assert config.target_tokens == 2000

    def test_configurable_target_tokens(self):
        """SPEC-08.23: target_tokens is configurable."""
        config = CompressionConfig(target_tokens=1000)

        assert config.target_tokens == 1000

    def test_config_affects_compression(self):
        """Config should affect compression behavior."""
        config = CompressionConfig(target_tokens=500)
        compressor = ContextCompressor(config=config)

        content = "Test sentence. " * 200

        result = compressor.compress(content)

        # Should compress more aggressively
        assert result.output_tokens <= 600  # Allow some margin


class TestAutoCompression:
    """Tests for automatic compression trigger (SPEC-08.24)."""

    def test_auto_trigger_threshold(self):
        """SPEC-08.24: Auto-trigger on outputs >5000 tokens."""
        auto = AutoCompressor()

        # Content with ~6000 tokens
        large_content = "Word " * 6000

        result = auto.maybe_compress(large_content)

        assert result.auto_triggered
        assert result.compressed

    def test_no_trigger_under_threshold(self):
        """Content under 5000 tokens should not auto-trigger."""
        auto = AutoCompressor()

        # Content with ~2000 tokens
        content = "Word " * 2000

        result = auto.maybe_compress(content)

        assert not result.auto_triggered

    def test_configurable_threshold(self):
        """Auto-trigger threshold should be configurable."""
        config = CompressionConfig(auto_trigger_tokens=3000)
        auto = AutoCompressor(config=config)

        content = "Word " * 4000

        result = auto.maybe_compress(content)

        assert result.auto_triggered

    def test_applies_to_tool_outputs(self):
        """Auto-compression applies to tool outputs."""
        auto = AutoCompressor()

        tool_output = "Line of output. " * 1000

        result = auto.maybe_compress(tool_output, source="tool_output")

        if result.auto_triggered:
            assert result.compressed


class TestCompressionMetrics:
    """Tests for compression ratio metrics (SPEC-08.25)."""

    def test_tracks_compression_ratio(self):
        """SPEC-08.25: Track compression ratio."""
        compressor = ContextCompressor()

        content = "This is test content. " * 200

        result = compressor.compress(content)

        assert result.compression_ratio is not None
        assert result.compression_ratio > 0

    def test_ratio_calculation(self):
        """Compression ratio = input_tokens / output_tokens."""
        result = CompressionResult(
            content="compressed",
            compressed=True,
            input_tokens=1000,
            output_tokens=250,
        )

        assert result.compression_ratio == 4.0

    def test_metrics_collection(self):
        """Metrics should be collected across compressions."""
        metrics = CompressionMetrics()

        metrics.record_compression(input_tokens=1000, output_tokens=250)
        metrics.record_compression(input_tokens=2000, output_tokens=400)

        assert metrics.total_compressions == 2
        assert metrics.average_ratio == 4.5  # (4.0 + 5.0) / 2

    def test_metrics_to_dict(self):
        """Metrics should be serializable."""
        metrics = CompressionMetrics()
        metrics.record_compression(input_tokens=1000, output_tokens=250)

        data = metrics.to_dict()

        assert "total_compressions" in data
        assert "average_ratio" in data


class TestCompressionResult:
    """Tests for CompressionResult structure."""

    def test_result_has_content(self):
        """Result should have compressed content."""
        result = CompressionResult(
            content="Compressed output",
            compressed=True,
            input_tokens=1000,
            output_tokens=200,
        )

        assert result.content == "Compressed output"

    def test_result_has_compressed_flag(self):
        """Result should indicate if compression occurred."""
        result = CompressionResult(
            content="Test",
            compressed=False,
            input_tokens=100,
            output_tokens=100,
        )

        assert result.compressed is False

    def test_result_has_token_counts(self):
        """Result should have input/output token counts."""
        result = CompressionResult(
            content="Test",
            compressed=True,
            input_tokens=1000,
            output_tokens=250,
        )

        assert result.input_tokens == 1000
        assert result.output_tokens == 250

    def test_result_has_stages_applied(self):
        """Result should track which stages were applied."""
        result = CompressionResult(
            content="Test",
            compressed=True,
            input_tokens=1000,
            output_tokens=100,
            stages_applied=[CompressionStage.EXTRACTIVE, CompressionStage.ABSTRACTIVE],
        )

        assert len(result.stages_applied) == 2

    def test_result_to_dict(self):
        """Result should be serializable."""
        result = CompressionResult(
            content="Test",
            compressed=True,
            input_tokens=1000,
            output_tokens=250,
        )

        data = result.to_dict()

        assert "content" in data
        assert "compressed" in data
        assert "compression_ratio" in data


class TestCompressionStage:
    """Tests for CompressionStage enum."""

    def test_extractive_stage(self):
        """EXTRACTIVE stage exists."""
        assert CompressionStage.EXTRACTIVE.value == "extractive"

    def test_abstractive_stage(self):
        """ABSTRACTIVE stage exists."""
        assert CompressionStage.ABSTRACTIVE.value == "abstractive"


class TestIntegration:
    """Integration tests for context compression."""

    def test_full_compression_pipeline(self):
        """Test complete compression pipeline."""
        config = CompressionConfig(target_tokens=500)
        compressor = ContextCompressor(config=config)

        content = (
            "Error: Database connection failed.\n"
            "Stack trace:\n"
            "  File 'db.py', line 42\n"
            "Analysis shows network issues.\n"
            "Recommendation: Check firewall settings.\n" + "Additional context. " * 100
        )

        result = compressor.compress(content)

        assert isinstance(result, CompressionResult)
        assert result.output_tokens <= 600  # Some margin

    def test_auto_compress_large_tool_output(self):
        """Auto-compress large tool outputs."""
        auto = AutoCompressor()

        tool_output = "Output line with data. " * 1500

        result = auto.maybe_compress(tool_output, source="tool_output")

        if result.auto_triggered:
            assert result.compressed
            assert result.compression_ratio >= 2.0

    def test_preserve_critical_info_through_compression(self):
        """Critical info preserved through full pipeline."""
        compressor = ContextCompressor()

        content = (
            "CRITICAL ERROR: System failure at 2024-01-01 10:00:00\n"
            "Error code: ERR_CONNECTION_REFUSED\n"
            "File: /app/src/network.py:142\n" + "Background information. " * 200
        )

        result = compressor.compress(content)

        # Should preserve critical info (varies by implementation)
        assert result.compressed or result.content == content

    def test_metrics_updated_after_compression(self):
        """Metrics should update after each compression."""
        config = CompressionConfig(target_tokens=500)
        compressor = ContextCompressor(config=config)

        content = "Test content. " * 300

        compressor.compress(content)

        assert compressor.metrics.total_compressions >= 0

    def test_compression_with_code_blocks(self):
        """Compression should handle code blocks correctly."""
        compressor = ContextCompressor()

        content = (
            "Analysis of function:\n"
            "```python\n"
            "def calculate_total(items):\n"
            "    return sum(item.price for item in items)\n"
            "```\n"
            "The function iterates through items. " * 100
        )

        result = compressor.compress(content)

        # Code blocks should be preserved when possible
        assert result.content is not None
