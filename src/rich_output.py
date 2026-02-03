"""
Rich terminal output for RLM-Claude-Code.

Implements: SPEC-13 Rich Output Formatting

Provides enhanced terminal output using the Rich library with:
- Consistent symbol vocabulary (no emoji)
- Token budget gauges
- Depth-based tree visualization
- Error display with context
- Configurable verbosity levels
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

# ============================================================================
# Visual Language System (SPEC-13.01-13.05)
# ============================================================================


class Symbol(str, Enum):
    """
    Semantic symbols for RLM output.

    SPEC-13.02: Symbol vocabulary with consistent semantic meaning.
    SPEC-13.03: No emoji characters.
    """

    # Core operations
    RLM = "◆"  # RLM activated
    ACTION = "▶"  # Execution/action
    PEEK = "◇"  # Read/peek operation
    SEARCH = "⊕"  # Search operation
    LLM = "∴"  # LLM sub-query (therefore)

    # Status indicators
    SUCCESS = "✓"  # Success/complete
    ERROR = "✗"  # Error/failure
    WARNING = "⚠"  # Warning/caution
    VERIFY = "⊙"  # Verification checkpoint (SPEC-16.34)

    # Auxiliary
    BUDGET = "≡"  # Cost/budget report
    MEMORY = "∿"  # Memory operation
    LEAN = "⊢"  # Lean verification (turnstile)

    # Tree connectors
    CONTINUE = "│"  # Depth continuation
    BRANCH = "├"  # Branch point
    LAST = "└"  # Terminal node
    LINE = "─"  # Horizontal line

    # Progress spinner (braille pattern)
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Color(str, Enum):
    """
    Semantic colors for RLM output.

    SPEC-13.02: ANSI colors mapped to symbol meanings.
    """

    RLM = "cyan"
    ACTION = "yellow"
    PEEK = "blue"
    SEARCH = "blue"
    LLM = "magenta"
    SUCCESS = "green"
    ERROR = "red"
    WARNING = "yellow"
    BUDGET = "white"
    MEMORY = "cyan"
    LEAN = "dark_cyan"
    VERIFY = "bright_blue"  # SPEC-16.34
    DIM = "dim"


# Symbol to color mapping
SYMBOL_COLORS: dict[Symbol, Color] = {
    Symbol.RLM: Color.RLM,
    Symbol.ACTION: Color.ACTION,
    Symbol.PEEK: Color.PEEK,
    Symbol.SEARCH: Color.SEARCH,
    Symbol.LLM: Color.LLM,
    Symbol.SUCCESS: Color.SUCCESS,
    Symbol.ERROR: Color.ERROR,
    Symbol.WARNING: Color.WARNING,
    Symbol.BUDGET: Color.BUDGET,
    Symbol.MEMORY: Color.MEMORY,
    Symbol.LEAN: Color.LEAN,
    Symbol.VERIFY: Color.VERIFY,  # SPEC-16.34
    Symbol.CONTINUE: Color.DIM,
    Symbol.BRANCH: Color.DIM,
    Symbol.LAST: Color.DIM,
}


# ============================================================================
# Configuration (SPEC-13.50-13.52)
# ============================================================================


@dataclass
class OutputConfig:
    """
    Configuration for Rich output.

    SPEC-13.51: Configuration options for output formatting.
    """

    verbosity: Literal["quiet", "normal", "verbose", "debug"] = "normal"
    colors: bool = True
    max_depth_display: int = 5
    progress_throttle_hz: int = 10
    panel_width: int | None = None  # Auto-detect if None

    @classmethod
    def from_env(cls) -> OutputConfig:
        """
        Load configuration from environment variables.

        SPEC-13.52: Environment variables override config file.
        SPEC-13.05: Respect NO_COLOR environment variable.
        """
        # Check NO_COLOR first (standard convention)
        no_color = os.environ.get("NO_COLOR") is not None

        verbosity = os.environ.get("RLM_VERBOSITY", "normal")
        if verbosity not in ("quiet", "normal", "verbose", "debug"):
            verbosity = "normal"

        colors_env = os.environ.get("RLM_COLORS", "").lower()
        colors = not no_color and colors_env != "false"

        return cls(
            verbosity=verbosity,  # type: ignore[arg-type]
            colors=colors,
        )


# ============================================================================
# Budget Display (SPEC-13.20-13.25)
# ============================================================================


def format_token_count(tokens: int) -> str:
    """Format token count with K/M suffix."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def create_budget_gauge(
    tokens_used: int,
    tokens_budget: int,
    depth: int | None = None,
    max_depth: int | None = None,
    width: int = 10,
) -> Text:
    """
    Create a token budget gauge.

    SPEC-13.20: Display token budget as a visual gauge.
    SPEC-13.21: Budget gauge format.
    SPEC-13.22: No dollar costs in output.

    Example output:
        ≡ Tokens: ████████░░ 45K/100K (depth 1/3)
    """
    # Calculate fill percentage
    percentage = min(tokens_used / tokens_budget, 1.0) if tokens_budget > 0 else 0
    filled = int(width * percentage)
    empty = width - filled

    # Build gauge
    text = Text()
    text.append(f"{Symbol.BUDGET.value} ", style=Color.BUDGET.value)
    text.append("Tokens: ", style="bold")

    # Color based on utilization
    if percentage >= 0.9:
        bar_color = Color.ERROR.value
    elif percentage >= 0.75:
        bar_color = Color.WARNING.value
    else:
        bar_color = Color.SUCCESS.value

    text.append("█" * filled, style=bar_color)
    text.append("░" * empty, style=Color.DIM.value)

    # Token counts
    text.append(f" {format_token_count(tokens_used)}/{format_token_count(tokens_budget)}")

    # Depth indicator
    if depth is not None and max_depth is not None:
        text.append(f" (depth {depth}/{max_depth})", style=Color.DIM.value)

    return text


# ============================================================================
# Depth Visualization (SPEC-13.30-13.33)
# ============================================================================


def build_tree_prefix(depth: int, is_last: bool = False) -> str:
    """
    Build tree connector prefix for a given depth.

    SPEC-13.30: Visualize recursive depth using tree connectors.
    """
    if depth == 0:
        return ""

    prefix = f"{Symbol.CONTINUE.value}   " * (depth - 1)
    connector = Symbol.LAST.value if is_last else Symbol.BRANCH.value
    return f"{prefix}{connector}{Symbol.LINE.value} "


class DepthTracker:
    """
    Track depth state for tree rendering.

    SPEC-13.32: Maximum rendered depth is configurable.
    SPEC-13.33: Deep recursion beyond max shows ... (+N deeper).
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.current_depth = 0
        self.hidden_depth = 0
        self._at_depth: dict[int, int] = {}  # Items at each depth level

    def enter(self) -> bool:
        """Enter a deeper level. Returns True if should render."""
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            self.hidden_depth += 1
            return False
        return True

    def exit(self) -> str | None:
        """Exit current level. Returns summary if hidden levels existed."""
        if self.hidden_depth > 0:
            summary = f"... (+{self.hidden_depth} deeper)"
            self.hidden_depth = 0
            self.current_depth -= 1
            return summary
        self.current_depth -= 1
        return None


# ============================================================================
# RLM Console (SPEC-13.10-13.14)
# ============================================================================


class RLMConsole:
    """
    Rich-formatted console output for RLM execution.

    SPEC-13.10: Use Rich Console for all trajectory output.
    SPEC-13.11: Provides emit_* methods for all event types.
    SPEC-13.12: Configurable via verbosity levels.
    SPEC-13.13: Supports Panel rendering.
    SPEC-13.14: Supports Tree rendering.
    """

    def __init__(self, config: OutputConfig | None = None):
        """Initialize RLM console with configuration."""
        self.config = config or OutputConfig.from_env()
        self.console = Console(
            force_terminal=True,
            no_color=not self.config.colors,
            width=self.config.panel_width,
        )
        self.depth_tracker = DepthTracker(max_depth=self.config.max_depth_display)
        self._tree: Tree | None = None
        self._tree_nodes: dict[int, Any] = {}  # depth -> current tree node

    def _should_emit(self, level: str) -> bool:
        """Check if event should be emitted based on verbosity."""
        levels = ["quiet", "normal", "verbose", "debug"]
        event_level = levels.index(level) if level in levels else 1
        config_level = levels.index(self.config.verbosity)
        return event_level <= config_level

    def _styled_symbol(self, symbol: Symbol) -> Text:
        """Create a styled symbol."""
        color = SYMBOL_COLORS.get(symbol, Color.DIM)
        return Text(symbol.value, style=color.value)

    def _format_prefix(self, depth: int, is_last: bool = False) -> Text:
        """Format tree prefix with styling."""
        text = Text()
        if depth > 0:
            for _ in range(depth - 1):
                text.append(f"{Symbol.CONTINUE.value}   ", style=Color.DIM.value)
            connector = Symbol.LAST.value if is_last else Symbol.BRANCH.value
            text.append(f"{connector}{Symbol.LINE.value} ", style=Color.DIM.value)
        return text

    # -------------------------------------------------------------------------
    # Event Emitters (SPEC-13.11)
    # -------------------------------------------------------------------------

    def emit_start(
        self,
        query: str,
        depth_budget: int,
        tokens_budget: int | None = None,
    ) -> None:
        """
        Emit RLM activation event.

        SPEC-13.11: emit_start(query, depth_budget)
        """
        if not self._should_emit("normal"):
            return

        text = Text()
        text.append(f"{Symbol.RLM.value} ", style=f"bold {Color.RLM.value}")
        text.append("RLM: ", style="bold")

        # Truncate long queries
        display_query = query[:60] + "..." if len(query) > 60 else query
        text.append(f'"{display_query}"', style=Color.DIM.value)

        self.console.print(text)

    def emit_repl(
        self,
        func: str,
        args: str,
        depth: int = 0,
        preview: str | None = None,
    ) -> None:
        """
        Emit REPL operation event.

        SPEC-13.11: emit_repl(func, args, preview)
        """
        if not self._should_emit("normal"):
            return

        # Select symbol based on function type
        if func in ("peek", "read"):
            symbol = Symbol.PEEK
        elif func in ("search", "grep", "find"):
            symbol = Symbol.SEARCH
        elif func in ("llm", "llm_batch", "summarize"):
            symbol = Symbol.LLM
        elif func.startswith("memory_"):
            symbol = Symbol.MEMORY
        elif func.startswith("lean_"):
            symbol = Symbol.LEAN
        else:
            symbol = Symbol.ACTION

        text = self._format_prefix(depth)
        text.append(f"{symbol.value} ", style=SYMBOL_COLORS[symbol].value)
        text.append(func, style="bold")
        text.append(f"({args})", style=Color.DIM.value)

        if preview and self._should_emit("verbose"):
            text.append(f" -> {preview[:50]}...", style=Color.DIM.value)

        self.console.print(text)

    def emit_recurse(self, query: str, depth: int) -> None:
        """
        Emit recursive sub-query event.

        SPEC-13.11: emit_recurse(query, depth)
        """
        if not self._should_emit("normal"):
            return

        text = self._format_prefix(depth - 1)
        text.append(f"{Symbol.LLM.value} ", style=f"bold {Color.LLM.value}")
        text.append("llm(", style="bold")

        display_query = query[:40] + "..." if len(query) > 40 else query
        text.append(f'"{display_query}"', style=Color.DIM.value)
        text.append(")", style="bold")

        self.console.print(text)

    def emit_result(
        self,
        summary: str,
        depth: int = 0,
        confidence: float | None = None,
        is_last: bool = False,
    ) -> None:
        """
        Emit operation result event.

        SPEC-13.11: emit_result(summary, confidence)
        """
        if not self._should_emit("normal"):
            return

        text = self._format_prefix(depth, is_last=is_last)
        text.append(f"{Symbol.SUCCESS.value} ", style=Color.SUCCESS.value)

        display_summary = summary[:80] + "..." if len(summary) > 80 else summary
        text.append(f'"{display_summary}"')

        if confidence is not None and self._should_emit("verbose"):
            text.append(f" [{confidence:.0%}]", style=Color.DIM.value)

        self.console.print(text)

    def emit_error(
        self,
        error: str,
        depth: int = 0,
        context: str | None = None,
        error_type: str | None = None,
    ) -> None:
        """
        Emit error event with context.

        SPEC-13.11: emit_error(error, context)
        SPEC-13.40-13.44: Error display with context and syntax highlighting.
        """
        # Errors always shown (even in quiet mode for visibility)
        text = self._format_prefix(depth)
        text.append(f"{Symbol.ERROR.value} ", style=f"bold {Color.ERROR.value}")

        if error_type:
            text.append(f"{error_type}: ", style=f"bold {Color.ERROR.value}")

        text.append(error)
        self.console.print(text)

        # Show context in verbose mode
        if context and self._should_emit("verbose"):
            # Try to syntax highlight if it looks like code
            if "\n" in context or context.strip().startswith(("def ", "class ", "import ")):
                syntax = Syntax(
                    context,
                    "python",
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                )
                self.console.print(Panel(syntax, border_style=Color.ERROR.value, padding=(0, 1)))
            else:
                self.console.print(Panel(context, border_style=Color.ERROR.value, padding=(0, 1)))

    def emit_warning(self, message: str, depth: int = 0) -> None:
        """
        Emit warning event.

        SPEC-13.43: Warnings with yellow border and warning prefix.
        """
        if not self._should_emit("normal"):
            return

        text = self._format_prefix(depth)
        text.append(f"{Symbol.WARNING.value} ", style=f"bold {Color.WARNING.value}")
        text.append(message)
        self.console.print(text)

    def emit_budget(
        self,
        tokens_used: int,
        tokens_budget: int,
        depth: int | None = None,
        max_depth: int | None = None,
    ) -> None:
        """
        Emit budget status event.

        SPEC-13.11: emit_budget(tokens_used, tokens_budget)
        SPEC-13.20-13.22: Token budget gauge display.
        """
        if not self._should_emit("normal"):
            return

        gauge = create_budget_gauge(
            tokens_used=tokens_used,
            tokens_budget=tokens_budget,
            depth=depth,
            max_depth=max_depth,
        )
        self.console.print(gauge)

    def emit_complete(
        self,
        tokens_used: int,
        depth: int = 0,
        execution_time_ms: float | None = None,
    ) -> None:
        """Emit completion event."""
        if not self._should_emit("normal"):
            return

        text = self._format_prefix(depth, is_last=True)
        text.append(f"{Symbol.SUCCESS.value} ", style=Color.SUCCESS.value)
        text.append("Complete ", style="bold green")
        text.append(f"({format_token_count(tokens_used)} tokens", style=Color.DIM.value)

        if execution_time_ms is not None:
            text.append(f", {execution_time_ms:.0f}ms", style=Color.DIM.value)

        text.append(")", style=Color.DIM.value)
        self.console.print(text)

    # -------------------------------------------------------------------------
    # Verification Output (SPEC-16.34)
    # -------------------------------------------------------------------------

    def emit_verification(
        self,
        claims_total: int,
        claims_verified: int,
        claims_flagged: int,
        confidence: float,
        depth: int = 0,
    ) -> None:
        """
        Emit verification checkpoint event.

        SPEC-16.34: Rich output for verification results.

        Args:
            claims_total: Total claims checked
            claims_verified: Claims that passed verification
            claims_flagged: Claims that failed verification
            confidence: Overall confidence score (0.0-1.0)
            depth: Current recursion depth
        """
        if not self._should_emit("normal"):
            return

        text = self._format_prefix(depth)
        text.append(f"{Symbol.VERIFY.value} ", style=f"bold {Color.VERIFY.value}")
        text.append("Verified: ", style="bold")

        # Color based on results
        if claims_flagged == 0:
            result_style = Color.SUCCESS.value
        elif claims_flagged < claims_total / 2:
            result_style = Color.WARNING.value
        else:
            result_style = Color.ERROR.value

        text.append(f"{claims_verified}/{claims_total}", style=result_style)
        text.append(f" ({confidence:.0%} confidence)", style=Color.DIM.value)

        if claims_flagged > 0:
            text.append(f" [{claims_flagged} flagged]", style=Color.WARNING.value)

        self.console.print(text)

    def emit_verification_report(
        self,
        claims_total: int,
        claims_verified: int,
        claims_flagged: int,
        confidence: float,
        flagged_claims: list[tuple[str, str, str]] | None = None,
        gaps: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Emit a full verification report panel.

        SPEC-16.34: Rich panel output for verification reports.

        Args:
            claims_total: Total claims checked
            claims_verified: Claims that passed verification
            claims_flagged: Claims that failed verification
            confidence: Overall confidence score
            flagged_claims: List of (claim_id, claim_text, reason) tuples
            gaps: List of (gap_type, description) tuples
        """
        text = Text()

        # Header
        text.append(f"{Symbol.VERIFY.value} ", style=f"bold {Color.VERIFY.value}")
        text.append("Verification Report\n", style="bold")
        text.append("─" * 40 + "\n", style=Color.DIM.value)

        # Summary line
        text.append("Claims: ", style="bold")
        text.append(f"{claims_total} total, ")

        if claims_verified == claims_total:
            text.append(f"{claims_verified} verified ", style=Color.SUCCESS.value)
            text.append(f"{Symbol.SUCCESS.value}\n", style=Color.SUCCESS.value)
        else:
            text.append(f"{claims_verified} verified, ", style=Color.SUCCESS.value)
            text.append(f"{claims_flagged} flagged\n", style=Color.WARNING.value)

        # Confidence bar
        text.append("Confidence: ", style="bold")
        bar_width = 10
        filled = int(bar_width * confidence)
        empty = bar_width - filled

        if confidence >= 0.8:
            bar_color = Color.SUCCESS.value
        elif confidence >= 0.6:
            bar_color = Color.WARNING.value
        else:
            bar_color = Color.ERROR.value

        text.append("█" * filled, style=bar_color)
        text.append("░" * empty, style=Color.DIM.value)
        text.append(f" {confidence:.0%}\n", style=bar_color)

        # Flagged claims section
        if flagged_claims:
            text.append("\nFlagged Claims:\n", style=f"bold {Color.WARNING.value}")
            for claim_id, claim_text, reason in flagged_claims[:5]:  # Limit to 5
                text.append(f"  [{claim_id}] ", style=Color.DIM.value)
                # Truncate long claims
                display_text = claim_text[:50] + "..." if len(claim_text) > 50 else claim_text
                text.append(f'"{display_text}"\n')
                text.append(f"       Reason: {reason}\n", style=Color.WARNING.value)

            if len(flagged_claims) > 5:
                text.append(f"  ... and {len(flagged_claims) - 5} more\n", style=Color.DIM.value)

        # Evidence gaps section
        if gaps:
            text.append("\nEvidence Gaps:\n", style=f"bold {Color.ERROR.value}")
            for gap_type, description in gaps[:3]:  # Limit to 3
                text.append(f"  • {gap_type}: ", style=Color.WARNING.value)
                text.append(f"{description}\n")

            if len(gaps) > 3:
                text.append(f"  ... and {len(gaps) - 3} more\n", style=Color.DIM.value)

        # Determine border color
        if claims_flagged == 0 and confidence >= 0.8:
            border_color = Color.SUCCESS.value
        elif claims_flagged > 0 or confidence < 0.6:
            border_color = Color.WARNING.value
        else:
            border_color = Color.VERIFY.value

        self.console.print(Panel(text, border_style=border_color, padding=(0, 1)))

    def emit_flagged_claim(
        self,
        claim_id: str,
        claim_text: str,
        reason: str,
        suggestion: str | None = None,
        depth: int = 0,
    ) -> None:
        """
        Emit a flagged claim with details.

        SPEC-16.34: Individual flagged claim output.
        """
        if not self._should_emit("verbose"):
            return

        text = self._format_prefix(depth)
        text.append(f"{Symbol.WARNING.value} ", style=f"bold {Color.WARNING.value}")
        text.append(f"[{claim_id}] ", style=Color.DIM.value)

        # Truncate long claims
        display_text = claim_text[:60] + "..." if len(claim_text) > 60 else claim_text
        text.append(f'"{display_text}"\n')

        # Reason
        text.append(f"         Reason: {reason}", style=Color.WARNING.value)

        if suggestion:
            text.append(f"\n         Suggestion: {suggestion}", style=Color.SUCCESS.value)

        self.console.print(text)

    # -------------------------------------------------------------------------
    # Tree Rendering (SPEC-13.14, SPEC-13.30-13.33)
    # -------------------------------------------------------------------------

    def start_tree(self, label: str) -> None:
        """Start a new tree visualization."""
        text = Text()
        text.append(f"{Symbol.RLM.value} ", style=f"bold {Color.RLM.value}")
        text.append(label, style="bold")
        self._tree = Tree(text)
        self._tree_nodes = {0: self._tree}

    def add_tree_node(
        self,
        label: str,
        depth: int,
        symbol: Symbol = Symbol.ACTION,
    ) -> None:
        """Add a node to the current tree."""
        if self._tree is None:
            return

        # Don't render beyond max depth
        if depth > self.config.max_depth_display:
            return

        text = Text()
        text.append(f"{symbol.value} ", style=SYMBOL_COLORS[symbol].value)
        text.append(label)

        parent_depth = max(0, depth - 1)
        parent = self._tree_nodes.get(parent_depth, self._tree)
        node = parent.add(text)
        self._tree_nodes[depth] = node

    def render_tree(self) -> None:
        """Render the accumulated tree."""
        if self._tree is not None:
            self.console.print(self._tree)
            self._tree = None
            self._tree_nodes = {}

    # -------------------------------------------------------------------------
    # Panel Rendering (SPEC-13.13)
    # -------------------------------------------------------------------------

    def print_panel(
        self,
        content: str,
        title: str | None = None,
        border_style: str = "cyan",
        syntax: str | None = None,
    ) -> None:
        """
        Print content in a panel.

        SPEC-13.13: Support Panel rendering for structured output.
        """
        if syntax:
            renderable = Syntax(content, syntax, theme="monokai", word_wrap=True)
        else:
            renderable = content

        panel = Panel(
            renderable,
            title=title,
            border_style=border_style,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_error_panel(
        self,
        error_type: str,
        message: str,
        location: str | None = None,
        context: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """
        Print an error panel with full context.

        SPEC-13.40-13.44: Error display with contextual information.
        """
        text = Text()
        text.append(f"{Symbol.ERROR.value} ", style=f"bold {Color.ERROR.value}")
        text.append(f"{error_type}\n", style=f"bold {Color.ERROR.value}")
        text.append(f"{message}\n")

        if location:
            text.append(f"\nLocation: {location}\n", style=Color.DIM.value)

        if context:
            text.append("\nContext:\n", style="bold")
            # Add syntax highlighting for code context
            syntax = Syntax(context, "python", theme="monokai", line_numbers=True)
            self.console.print(
                Panel(
                    syntax,
                    title=f"{Symbol.ERROR.value} {error_type}",
                    border_style=Color.ERROR.value,
                    padding=(0, 1),
                )
            )
            return  # Panel already printed

        if suggestion:
            text.append(f"\nSuggestion: {suggestion}", style=Color.SUCCESS.value)

        self.console.print(
            Panel(
                text,
                border_style=Color.ERROR.value,
                padding=(0, 1),
            )
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def get_console(config: OutputConfig | None = None) -> RLMConsole:
    """Get a configured RLM console instance."""
    return RLMConsole(config)


# Global console instance (lazy-initialized)
_global_console: RLMConsole | None = None


def console() -> RLMConsole:
    """Get the global RLM console instance."""
    global _global_console
    if _global_console is None:
        _global_console = RLMConsole()
    return _global_console


__all__ = [
    # Symbols and colors
    "Symbol",
    "Color",
    "SYMBOL_COLORS",
    # Configuration
    "OutputConfig",
    # Budget display
    "format_token_count",
    "create_budget_gauge",
    # Depth visualization
    "build_tree_prefix",
    "DepthTracker",
    # Console
    "RLMConsole",
    "get_console",
    "console",
]
