"""
Unit tests for rich_output.py.

Tests SPEC-13: Rich Output Formatting
"""

import os
from unittest.mock import patch

import pytest

from src.rich_output import (
    SYMBOL_COLORS,
    Color,
    DepthTracker,
    OutputConfig,
    RLMConsole,
    Symbol,
    build_tree_prefix,
    create_budget_gauge,
    format_token_count,
    get_console,
)

# ============================================================================
# Symbol Tests (SPEC-13.01-13.02)
# ============================================================================


class TestSymbols:
    """Test symbol vocabulary."""

    def test_symbols_are_single_characters(self) -> None:
        """SPEC-13.02: Each symbol should be a single character (or short string)."""
        single_char_symbols = [
            Symbol.RLM,
            Symbol.ACTION,
            Symbol.PEEK,
            Symbol.SEARCH,
            Symbol.LLM,
            Symbol.SUCCESS,
            Symbol.ERROR,
            Symbol.WARNING,
            Symbol.BUDGET,
            Symbol.MEMORY,
            Symbol.LEAN,
            Symbol.CONTINUE,
            Symbol.BRANCH,
            Symbol.LAST,
            Symbol.LINE,
        ]
        for symbol in single_char_symbols:
            assert len(symbol.value) == 1, f"{symbol.name} should be single char"

    def test_no_emoji_in_symbols(self) -> None:
        """SPEC-13.03: No emoji characters in output."""
        # Emoji typically have ord() > 0x1F000
        for symbol in Symbol:
            if symbol == Symbol.SPINNER:
                # Spinner is a sequence of braille chars
                continue
            for char in symbol.value:
                assert ord(char) < 0x1F000, f"{symbol.name} contains emoji-like char"

    def test_symbol_colors_complete(self) -> None:
        """Each semantic symbol should have a color mapping."""
        semantic_symbols = [
            Symbol.RLM,
            Symbol.ACTION,
            Symbol.PEEK,
            Symbol.SEARCH,
            Symbol.LLM,
            Symbol.SUCCESS,
            Symbol.ERROR,
            Symbol.WARNING,
            Symbol.BUDGET,
            Symbol.MEMORY,
            Symbol.LEAN,
        ]
        for symbol in semantic_symbols:
            assert symbol in SYMBOL_COLORS, f"{symbol.name} missing color"

    def test_spinner_is_braille(self) -> None:
        """SPEC-13.24: Spinner uses braille pattern."""
        # Braille patterns are in Unicode range U+2800-U+28FF
        for char in Symbol.SPINNER.value:
            assert 0x2800 <= ord(char) <= 0x28FF, f"Spinner char {char} not braille"


# ============================================================================
# OutputConfig Tests (SPEC-13.50-13.52)
# ============================================================================


class TestOutputConfig:
    """Test configuration handling."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OutputConfig()
        assert config.verbosity == "normal"
        assert config.colors is True
        assert config.max_depth_display == 5
        assert config.progress_throttle_hz == 10

    def test_from_env_verbosity(self) -> None:
        """SPEC-13.52: Environment variables override config."""
        with patch.dict(os.environ, {"RLM_VERBOSITY": "verbose"}):
            config = OutputConfig.from_env()
            assert config.verbosity == "verbose"

    def test_from_env_invalid_verbosity(self) -> None:
        """Invalid verbosity falls back to normal."""
        with patch.dict(os.environ, {"RLM_VERBOSITY": "invalid"}):
            config = OutputConfig.from_env()
            assert config.verbosity == "normal"

    def test_from_env_no_color(self) -> None:
        """SPEC-13.05: Respect NO_COLOR environment variable."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            config = OutputConfig.from_env()
            assert config.colors is False

    def test_from_env_rlm_colors_false(self) -> None:
        """RLM_COLORS=false disables colors."""
        with patch.dict(os.environ, {"RLM_COLORS": "false"}, clear=True):
            config = OutputConfig.from_env()
            assert config.colors is False

    def test_from_env_colors_enabled_by_default(self) -> None:
        """Colors enabled by default when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove NO_COLOR and RLM_COLORS if present
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("RLM_COLORS", None)
            config = OutputConfig.from_env()
            assert config.colors is True


# ============================================================================
# Budget Gauge Tests (SPEC-13.20-13.22)
# ============================================================================


class TestBudgetGauge:
    """Test token budget gauge display."""

    def test_format_token_count_small(self) -> None:
        """Small token counts displayed as-is."""
        assert format_token_count(500) == "500"
        assert format_token_count(999) == "999"

    def test_format_token_count_thousands(self) -> None:
        """Token counts in thousands use K suffix."""
        assert format_token_count(1000) == "1.0K"
        assert format_token_count(45000) == "45.0K"
        assert format_token_count(100000) == "100.0K"

    def test_format_token_count_millions(self) -> None:
        """Token counts in millions use M suffix."""
        assert format_token_count(1000000) == "1.0M"
        assert format_token_count(2500000) == "2.5M"

    def test_create_budget_gauge_basic(self) -> None:
        """SPEC-13.21: Budget gauge format."""
        gauge = create_budget_gauge(45000, 100000)
        text = gauge.plain
        assert "45.0K" in text
        assert "100.0K" in text
        assert Symbol.BUDGET.value in text

    def test_create_budget_gauge_with_depth(self) -> None:
        """Budget gauge includes depth info when provided."""
        gauge = create_budget_gauge(45000, 100000, depth=1, max_depth=3)
        text = gauge.plain
        assert "depth 1/3" in text

    def test_create_budget_gauge_no_dollars(self) -> None:
        """SPEC-13.22: No dollar costs in output."""
        gauge = create_budget_gauge(45000, 100000)
        text = gauge.plain
        assert "$" not in text

    def test_create_budget_gauge_zero_budget(self) -> None:
        """Handle zero budget without division error."""
        gauge = create_budget_gauge(0, 0)
        # Should not raise, percentage becomes 0
        assert gauge is not None


# ============================================================================
# Depth Visualization Tests (SPEC-13.30-13.33)
# ============================================================================


class TestDepthVisualization:
    """Test depth tree visualization."""

    def test_build_tree_prefix_depth_0(self) -> None:
        """Depth 0 has no prefix."""
        prefix = build_tree_prefix(0)
        assert prefix == ""

    def test_build_tree_prefix_depth_1(self) -> None:
        """Depth 1 has branch connector."""
        prefix = build_tree_prefix(1)
        assert Symbol.BRANCH.value in prefix
        assert Symbol.LINE.value in prefix

    def test_build_tree_prefix_depth_1_last(self) -> None:
        """Last item at depth 1 uses terminal connector."""
        prefix = build_tree_prefix(1, is_last=True)
        assert Symbol.LAST.value in prefix
        assert Symbol.BRANCH.value not in prefix

    def test_build_tree_prefix_depth_2(self) -> None:
        """Depth 2 has continuation line plus connector."""
        prefix = build_tree_prefix(2)
        assert Symbol.CONTINUE.value in prefix
        assert Symbol.BRANCH.value in prefix

    def test_depth_tracker_basic(self) -> None:
        """SPEC-13.32: Track depth levels."""
        tracker = DepthTracker(max_depth=3)
        assert tracker.enter()  # depth 1
        assert tracker.enter()  # depth 2
        assert tracker.enter()  # depth 3
        assert not tracker.enter()  # depth 4 - hidden

    def test_depth_tracker_hidden_summary(self) -> None:
        """SPEC-13.33: Hidden depth summary on exit."""
        tracker = DepthTracker(max_depth=2)
        tracker.enter()  # 1
        tracker.enter()  # 2
        tracker.enter()  # 3 - hidden
        tracker.enter()  # 4 - hidden

        summary = tracker.exit()
        assert summary is not None
        assert "+2 deeper" in summary


# ============================================================================
# RLMConsole Tests (SPEC-13.10-13.14)
# ============================================================================


class TestRLMConsole:
    """Test RLM console output."""

    def test_console_creation(self) -> None:
        """Console creates without error."""
        console = RLMConsole()
        assert console is not None
        assert console.console is not None

    def test_console_with_config(self) -> None:
        """Console respects configuration."""
        config = OutputConfig(verbosity="quiet", colors=False)
        console = RLMConsole(config)
        assert console.config.verbosity == "quiet"
        assert console.config.colors is False

    def test_should_emit_quiet(self) -> None:
        """Quiet mode suppresses non-error output."""
        config = OutputConfig(verbosity="quiet")
        console = RLMConsole(config)
        assert not console._should_emit("normal")
        assert not console._should_emit("verbose")

    def test_should_emit_verbose(self) -> None:
        """Verbose mode includes verbose events."""
        config = OutputConfig(verbosity="verbose")
        console = RLMConsole(config)
        assert console._should_emit("normal")
        assert console._should_emit("verbose")
        assert not console._should_emit("debug")

    def test_get_console_factory(self) -> None:
        """get_console returns configured instance."""
        config = OutputConfig(verbosity="debug")
        console = get_console(config)
        assert console.config.verbosity == "debug"

    def test_emit_start(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_start outputs RLM activation."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_start("test query", depth_budget=3)

        captured = capsys.readouterr()
        assert Symbol.RLM.value in captured.out
        assert "RLM" in captured.out

    def test_emit_repl_peek(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_repl uses peek symbol for read operations."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_repl("peek", "context, 0:1000")

        captured = capsys.readouterr()
        assert Symbol.PEEK.value in captured.out
        assert "peek" in captured.out

    def test_emit_repl_search(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_repl uses search symbol for search operations."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_repl("search", 'code, "pattern"')

        captured = capsys.readouterr()
        assert Symbol.SEARCH.value in captured.out

    def test_emit_repl_llm(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_repl uses llm symbol for llm operations."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_repl("llm", '"summarize this"')

        captured = capsys.readouterr()
        assert Symbol.LLM.value in captured.out

    def test_emit_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_error outputs error with symbol."""
        config = OutputConfig(colors=False, verbosity="quiet")
        console = RLMConsole(config)
        console.emit_error("Test error message")

        captured = capsys.readouterr()
        assert Symbol.ERROR.value in captured.out
        assert "Test error message" in captured.out

    def test_emit_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_warning outputs warning with symbol."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_warning("Test warning")

        captured = capsys.readouterr()
        assert Symbol.WARNING.value in captured.out

    def test_emit_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_result outputs success with symbol."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_result("Operation completed successfully")

        captured = capsys.readouterr()
        assert Symbol.SUCCESS.value in captured.out

    def test_emit_budget(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_budget outputs token gauge."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_budget(45000, 100000)

        captured = capsys.readouterr()
        assert Symbol.BUDGET.value in captured.out
        assert "45.0K" in captured.out

    def test_emit_complete(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_complete outputs completion info."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_complete(tokens_used=5000, execution_time_ms=150.5)

        captured = capsys.readouterr()
        assert Symbol.SUCCESS.value in captured.out
        assert "Complete" in captured.out
        assert "5.0K" in captured.out
        assert "150ms" in captured.out

    def test_tree_rendering(self, capsys: pytest.CaptureFixture[str]) -> None:
        """SPEC-13.14: Tree rendering works."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)

        console.start_tree("Test Query")
        console.add_tree_node("peek(context)", depth=1, symbol=Symbol.PEEK)
        console.add_tree_node("search(code)", depth=1, symbol=Symbol.SEARCH)
        console.add_tree_node("found 3", depth=2, symbol=Symbol.SUCCESS)
        console.render_tree()

        captured = capsys.readouterr()
        assert Symbol.RLM.value in captured.out
        assert "Test Query" in captured.out

    def test_quiet_mode_suppresses_normal(self, capsys: pytest.CaptureFixture[str]) -> None:
        """SPEC-13.12: Quiet mode only shows errors."""
        config = OutputConfig(verbosity="quiet", colors=False)
        console = RLMConsole(config)

        console.emit_start("query", 3)
        console.emit_repl("peek", "args")
        console.emit_result("done")

        captured = capsys.readouterr()
        # All above should be suppressed in quiet mode
        assert captured.out.strip() == ""


# ============================================================================
# Panel Rendering Tests (SPEC-13.13)
# ============================================================================


class TestPanelRendering:
    """Test panel rendering functionality."""

    def test_print_panel(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Basic panel rendering."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.print_panel("Panel content", title="Test")

        captured = capsys.readouterr()
        assert "Panel content" in captured.out
        assert "Test" in captured.out

    def test_print_error_panel(self, capsys: pytest.CaptureFixture[str]) -> None:
        """SPEC-13.42: Error panel with red border."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.print_error_panel(
            error_type="ValueError",
            message="Invalid input",
            suggestion="Check your input",
        )

        captured = capsys.readouterr()
        assert "ValueError" in captured.out
        assert "Invalid input" in captured.out


# ============================================================================
# Verification Output Tests (SPEC-16.34)
# ============================================================================


class TestVerificationOutput:
    """Test verification output functionality."""

    def test_verify_symbol_exists(self) -> None:
        """VERIFY symbol should exist."""
        assert Symbol.VERIFY.value == "⊙"

    def test_verify_symbol_has_color(self) -> None:
        """VERIFY symbol should have color mapping."""
        assert Symbol.VERIFY in SYMBOL_COLORS
        assert SYMBOL_COLORS[Symbol.VERIFY] == Color.VERIFY

    def test_emit_verification_all_passed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_verification shows all claims passed."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_verification(
            claims_total=10,
            claims_verified=10,
            claims_flagged=0,
            confidence=0.95,
        )

        captured = capsys.readouterr()
        assert Symbol.VERIFY.value in captured.out
        assert "Verified" in captured.out
        assert "10/10" in captured.out
        assert "95%" in captured.out

    def test_emit_verification_with_flagged(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_verification shows flagged claims count."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_verification(
            claims_total=10,
            claims_verified=8,
            claims_flagged=2,
            confidence=0.80,
        )

        captured = capsys.readouterr()
        assert "8/10" in captured.out
        assert "2 flagged" in captured.out

    def test_emit_verification_quiet_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_verification suppressed in quiet mode."""
        config = OutputConfig(colors=False, verbosity="quiet")
        console = RLMConsole(config)
        console.emit_verification(
            claims_total=10,
            claims_verified=10,
            claims_flagged=0,
            confidence=0.95,
        )

        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_emit_verification_report_basic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_verification_report shows panel output."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_verification_report(
            claims_total=5,
            claims_verified=5,
            claims_flagged=0,
            confidence=0.90,
        )

        captured = capsys.readouterr()
        assert "Verification Report" in captured.out
        assert "Claims:" in captured.out
        assert "5 total" in captured.out
        assert "5 verified" in captured.out
        assert "Confidence:" in captured.out

    def test_emit_verification_report_with_flagged_claims(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """emit_verification_report shows flagged claims details."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_verification_report(
            claims_total=5,
            claims_verified=3,
            claims_flagged=2,
            confidence=0.70,
            flagged_claims=[
                ("c1", "The API returns XML", "unsupported"),
                ("c2", "The function is pure", "contradicted"),
            ],
        )

        captured = capsys.readouterr()
        assert "Flagged Claims:" in captured.out
        assert "[c1]" in captured.out
        assert "unsupported" in captured.out

    def test_emit_verification_report_with_gaps(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_verification_report shows evidence gaps."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_verification_report(
            claims_total=5,
            claims_verified=4,
            claims_flagged=1,
            confidence=0.75,
            gaps=[
                ("partial_support", "Claim goes beyond evidence"),
                ("phantom_citation", "Referenced section not found"),
            ],
        )

        captured = capsys.readouterr()
        assert "Evidence Gaps:" in captured.out
        assert "partial_support" in captured.out

    def test_emit_verification_report_truncates_long_lists(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """emit_verification_report truncates lists beyond limit."""
        config = OutputConfig(colors=False)
        console = RLMConsole(config)
        console.emit_verification_report(
            claims_total=10,
            claims_verified=2,
            claims_flagged=8,
            confidence=0.30,
            flagged_claims=[(f"c{i}", f"Claim {i} text", "reason") for i in range(10)],
        )

        captured = capsys.readouterr()
        assert "... and 5 more" in captured.out

    def test_emit_flagged_claim(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_flagged_claim shows individual claim details."""
        config = OutputConfig(colors=False, verbosity="verbose")
        console = RLMConsole(config)
        console.emit_flagged_claim(
            claim_id="c1",
            claim_text="The function returns JSON",
            reason="unsupported",
            suggestion="Provide supporting evidence",
        )

        captured = capsys.readouterr()
        assert "[c1]" in captured.out
        assert "function returns JSON" in captured.out
        assert "unsupported" in captured.out
        assert "Suggestion:" in captured.out

    def test_emit_flagged_claim_hidden_in_normal_mode(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """emit_flagged_claim suppressed in normal verbosity."""
        config = OutputConfig(colors=False, verbosity="normal")
        console = RLMConsole(config)
        console.emit_flagged_claim(
            claim_id="c1",
            claim_text="The function returns JSON",
            reason="unsupported",
        )

        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_emit_flagged_claim_truncates_long_text(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """emit_flagged_claim truncates very long claim text."""
        config = OutputConfig(colors=False, verbosity="verbose")
        console = RLMConsole(config)
        long_claim = "x" * 100
        console.emit_flagged_claim(
            claim_id="c1",
            claim_text=long_claim,
            reason="unsupported",
        )

        captured = capsys.readouterr()
        assert "..." in captured.out  # Truncated
        assert "x" * 100 not in captured.out
