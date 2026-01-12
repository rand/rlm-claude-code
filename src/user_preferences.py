"""
User preferences for RLM-Claude-Code.

Implements: Spec §8.1 Phase 2 - User Preferences

Allows users to configure:
- Execution mode (fast/balanced/thorough)
- Cost budgets
- Model preferences
- Tool access levels
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .orchestration_schema import ExecutionMode, ToolAccessLevel


# Default preferences file location
DEFAULT_PREFS_PATH = Path.home() / ".config" / "rlm-claude-code" / "preferences.json"


@dataclass
class UserPreferences:
    """
    User preferences for RLM execution.

    Implements: Spec §8.1 User preference system

    Supports:
    - Execution mode (fast/balanced/thorough)
    - Cost/token budgets
    - Model preferences
    - Persistence across sessions
    """

    # Execution preferences
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    auto_activate: bool = True  # Automatically activate RLM for complex tasks

    # Budget constraints
    budget_dollars: float = 5.0  # Maximum cost per session
    budget_tokens: int = 100_000  # Maximum tokens per session
    budget_per_query_dollars: float | None = None  # Per-query limit

    # Model preferences
    preferred_model: str | None = None  # Force specific model
    fallback_enabled: bool = True  # Use fallback models on failure
    force_provider: str | None = None  # Force specific provider

    # Depth and tool settings
    max_depth: int = 2  # Maximum recursion depth
    tool_access: ToolAccessLevel = ToolAccessLevel.READ_ONLY

    # UI preferences
    trajectory_verbosity: str = "normal"  # minimal/normal/verbose/debug
    show_cost_report: bool = True
    stream_output: bool = True

    # Advanced
    cache_decisions: bool = True
    use_llm_orchestrator: bool = True  # Use Claude for orchestration vs heuristics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "execution_mode": self.execution_mode.value,
            "auto_activate": self.auto_activate,
            "budget_dollars": self.budget_dollars,
            "budget_tokens": self.budget_tokens,
            "budget_per_query_dollars": self.budget_per_query_dollars,
            "preferred_model": self.preferred_model,
            "fallback_enabled": self.fallback_enabled,
            "force_provider": self.force_provider,
            "max_depth": self.max_depth,
            "tool_access": self.tool_access.value,
            "trajectory_verbosity": self.trajectory_verbosity,
            "show_cost_report": self.show_cost_report,
            "stream_output": self.stream_output,
            "cache_decisions": self.cache_decisions,
            "use_llm_orchestrator": self.use_llm_orchestrator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserPreferences:
        """Create from dictionary."""
        prefs = cls()

        if "execution_mode" in data:
            prefs.execution_mode = ExecutionMode(data["execution_mode"])
        if "auto_activate" in data:
            prefs.auto_activate = data["auto_activate"]
        if "budget_dollars" in data:
            prefs.budget_dollars = float(data["budget_dollars"])
        if "budget_tokens" in data:
            prefs.budget_tokens = int(data["budget_tokens"])
        if "budget_per_query_dollars" in data:
            prefs.budget_per_query_dollars = data["budget_per_query_dollars"]
        if "preferred_model" in data:
            prefs.preferred_model = data["preferred_model"]
        if "fallback_enabled" in data:
            prefs.fallback_enabled = data["fallback_enabled"]
        if "force_provider" in data:
            prefs.force_provider = data["force_provider"]
        if "max_depth" in data:
            prefs.max_depth = int(data["max_depth"])
        if "tool_access" in data:
            prefs.tool_access = ToolAccessLevel(data["tool_access"])
        if "trajectory_verbosity" in data:
            prefs.trajectory_verbosity = data["trajectory_verbosity"]
        if "show_cost_report" in data:
            prefs.show_cost_report = data["show_cost_report"]
        if "stream_output" in data:
            prefs.stream_output = data["stream_output"]
        if "cache_decisions" in data:
            prefs.cache_decisions = data["cache_decisions"]
        if "use_llm_orchestrator" in data:
            prefs.use_llm_orchestrator = data["use_llm_orchestrator"]

        return prefs

    def save(self, path: Path | None = None) -> None:
        """Save preferences to file."""
        path = path or DEFAULT_PREFS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | None = None) -> UserPreferences:
        """Load preferences from file."""
        path = path or DEFAULT_PREFS_PATH
        if path.exists():
            with open(path) as f:
                return cls.from_dict(json.load(f))
        return cls()


@dataclass
class PreferenceUpdate:
    """A single preference update."""

    field: str
    old_value: Any
    new_value: Any
    reason: str = ""


class PreferencesManager:
    """
    Manages user preferences with command parsing.

    Supports commands like:
    - /rlm mode fast
    - /rlm mode balanced
    - /rlm mode thorough
    - /rlm budget $5
    - /rlm depth 3
    - /rlm model opus
    - /rlm off
    - /rlm on
    - /rlm status
    - /rlm reset
    """

    def __init__(self, prefs: UserPreferences | None = None):
        """Initialize with preferences."""
        self.prefs = prefs or UserPreferences.load()
        self._update_history: list[PreferenceUpdate] = []

    def parse_command(self, command: str) -> tuple[str, dict[str, Any]]:
        """
        Parse an RLM command and apply changes.

        Args:
            command: Command string (e.g., "mode fast", "budget $5")

        Returns:
            (response_message, updated_values) tuple
        """
        parts = command.strip().lower().split()

        if not parts:
            return self._status_message(), {}

        action = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        # Mode commands
        if action == "mode":
            return self._handle_mode(args)

        # Budget commands
        if action == "budget":
            return self._handle_budget(args)

        # Depth commands
        if action == "depth":
            return self._handle_depth(args)

        # Model commands
        if action == "model":
            return self._handle_model(args)

        # Tool access commands
        if action == "tools":
            return self._handle_tools(args)

        # Verbosity commands
        if action == "verbosity":
            return self._handle_verbosity(args)

        # Toggle commands
        if action in ("on", "enable"):
            return self._toggle_rlm(True)

        if action in ("off", "disable"):
            return self._toggle_rlm(False)

        # Status and reset
        if action == "status":
            return self._status_message(), {}

        if action == "reset":
            return self._reset()

        # Save/load
        if action == "save":
            self.prefs.save()
            return "Preferences saved.", {}

        return f"Unknown command: {action}", {}

    def _handle_mode(self, args: list[str]) -> tuple[str, dict[str, Any]]:
        """Handle mode command."""
        if not args:
            return f"Current mode: {self.prefs.execution_mode.value}", {}

        mode_str = args[0]
        mode_map = {
            "fast": ExecutionMode.FAST,
            "balanced": ExecutionMode.BALANCED,
            "thorough": ExecutionMode.THOROUGH,
        }

        if mode_str not in mode_map:
            return f"Unknown mode: {mode_str}. Use: fast, balanced, or thorough", {}

        old_mode = self.prefs.execution_mode
        self.prefs.execution_mode = mode_map[mode_str]

        self._record_update("execution_mode", old_mode.value, mode_str)
        return f"Mode set to: {mode_str}", {"execution_mode": mode_str}

    def _handle_budget(self, args: list[str]) -> tuple[str, dict[str, Any]]:
        """Handle budget command."""
        if not args:
            return f"Current budget: ${self.prefs.budget_dollars:.2f}", {}

        budget_str = args[0].replace("$", "")
        try:
            budget = float(budget_str)
            if budget <= 0:
                return "Budget must be positive.", {}

            old_budget = self.prefs.budget_dollars
            self.prefs.budget_dollars = budget

            self._record_update("budget_dollars", old_budget, budget)
            return f"Budget set to: ${budget:.2f}", {"budget_dollars": budget}

        except ValueError:
            return f"Invalid budget value: {args[0]}", {}

    def _handle_depth(self, args: list[str]) -> tuple[str, dict[str, Any]]:
        """Handle depth command."""
        if not args:
            return f"Current max depth: {self.prefs.max_depth}", {}

        try:
            depth = int(args[0])
            if depth < 0 or depth > 5:
                return "Depth must be between 0 and 5.", {}

            old_depth = self.prefs.max_depth
            self.prefs.max_depth = depth

            self._record_update("max_depth", old_depth, depth)
            return f"Max depth set to: {depth}", {"max_depth": depth}

        except ValueError:
            return f"Invalid depth value: {args[0]}", {}

    def _handle_model(self, args: list[str]) -> tuple[str, dict[str, Any]]:
        """Handle model command."""
        if not args:
            model = self.prefs.preferred_model or "auto"
            return f"Current model preference: {model}", {}

        model = args[0]

        if model in ("auto", "none", "clear"):
            self.prefs.preferred_model = None
            return "Model preference cleared (auto-select).", {"preferred_model": None}

        valid_models = ["opus", "sonnet", "haiku", "gpt-4o", "gpt-5.2", "codex", "o1"]
        if model not in valid_models:
            return f"Unknown model: {model}. Valid: {', '.join(valid_models)}", {}

        old_model = self.prefs.preferred_model
        self.prefs.preferred_model = model

        self._record_update("preferred_model", old_model, model)
        return f"Preferred model set to: {model}", {"preferred_model": model}

    def _handle_tools(self, args: list[str]) -> tuple[str, dict[str, Any]]:
        """Handle tools command."""
        if not args:
            return f"Current tool access: {self.prefs.tool_access.value}", {}

        level_str = args[0]
        level_map = {
            "none": ToolAccessLevel.NONE,
            "repl": ToolAccessLevel.REPL_ONLY,
            "repl_only": ToolAccessLevel.REPL_ONLY,
            "read": ToolAccessLevel.READ_ONLY,
            "read_only": ToolAccessLevel.READ_ONLY,
            "full": ToolAccessLevel.FULL,
        }

        if level_str not in level_map:
            return f"Unknown tool level: {level_str}. Use: none, repl, read, or full", {}

        old_level = self.prefs.tool_access
        self.prefs.tool_access = level_map[level_str]

        self._record_update("tool_access", old_level.value, level_str)
        return f"Tool access set to: {level_str}", {"tool_access": level_str}

    def _handle_verbosity(self, args: list[str]) -> tuple[str, dict[str, Any]]:
        """Handle verbosity command."""
        if not args:
            return f"Current verbosity: {self.prefs.trajectory_verbosity}", {}

        level = args[0]
        valid = ["minimal", "normal", "verbose", "debug"]

        if level not in valid:
            return f"Unknown verbosity: {level}. Use: {', '.join(valid)}", {}

        old_verbosity = self.prefs.trajectory_verbosity
        self.prefs.trajectory_verbosity = level

        self._record_update("trajectory_verbosity", old_verbosity, level)
        return f"Verbosity set to: {level}", {"trajectory_verbosity": level}

    def _toggle_rlm(self, enable: bool) -> tuple[str, dict[str, Any]]:
        """Toggle RLM auto-activation."""
        old_value = self.prefs.auto_activate
        self.prefs.auto_activate = enable

        if old_value != enable:
            self._record_update("auto_activate", old_value, enable)

        status = "enabled" if enable else "disabled"
        return f"RLM auto-activation {status}.", {"auto_activate": enable}

    def _status_message(self) -> str:
        """Generate status message."""
        lines = [
            "RLM Preferences:",
            f"  Mode: {self.prefs.execution_mode.value}",
            f"  Auto-activate: {'on' if self.prefs.auto_activate else 'off'}",
            f"  Budget: ${self.prefs.budget_dollars:.2f}",
            f"  Max depth: {self.prefs.max_depth}",
            f"  Tool access: {self.prefs.tool_access.value}",
            f"  Model: {self.prefs.preferred_model or 'auto'}",
            f"  Verbosity: {self.prefs.trajectory_verbosity}",
        ]
        return "\n".join(lines)

    def _reset(self) -> tuple[str, dict[str, Any]]:
        """Reset to defaults."""
        self.prefs = UserPreferences()
        return "Preferences reset to defaults.", {"reset": True}

    def _record_update(self, field: str, old: Any, new: Any) -> None:
        """Record a preference update."""
        self._update_history.append(
            PreferenceUpdate(field=field, old_value=old, new_value=new)
        )

    def get_update_history(self) -> list[PreferenceUpdate]:
        """Get history of preference updates."""
        return self._update_history.copy()


# Global preferences instance
_global_prefs: UserPreferences | None = None
_global_manager: PreferencesManager | None = None


def get_preferences() -> UserPreferences:
    """Get global user preferences."""
    global _global_prefs
    if _global_prefs is None:
        _global_prefs = UserPreferences.load()
    return _global_prefs


def get_preferences_manager() -> PreferencesManager:
    """Get global preferences manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = PreferencesManager(get_preferences())
    return _global_manager


def handle_rlm_command(command: str) -> str:
    """
    Handle an /rlm command.

    Args:
        command: Command after "/rlm " (e.g., "mode fast")

    Returns:
        Response message
    """
    manager = get_preferences_manager()
    message, _ = manager.parse_command(command)
    return message


__all__ = [
    "DEFAULT_PREFS_PATH",
    "PreferenceUpdate",
    "PreferencesManager",
    "UserPreferences",
    "get_preferences",
    "get_preferences_manager",
    "handle_rlm_command",
]
