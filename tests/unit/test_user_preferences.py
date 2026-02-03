"""
Unit tests for user_preferences module.

Implements: Spec §8.1 Phase 2 - User Preferences tests
"""

import tempfile
from pathlib import Path

import pytest

from src.orchestration_schema import ExecutionMode, ToolAccessLevel
from src.user_preferences import (
    PreferencesManager,
    PreferenceUpdate,
    UserPreferences,
    handle_rlm_command,
)


class TestUserPreferences:
    """Tests for UserPreferences dataclass."""

    def test_default_preferences(self):
        """Default preferences are sensible."""
        prefs = UserPreferences()

        assert prefs.execution_mode == ExecutionMode.BALANCED
        assert prefs.auto_activate is True
        assert prefs.budget_dollars == 5.0
        assert prefs.budget_tokens == 100_000
        assert prefs.max_depth == 2
        assert prefs.tool_access == ToolAccessLevel.READ_ONLY
        assert prefs.trajectory_verbosity == "normal"

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        prefs = UserPreferences(
            execution_mode=ExecutionMode.FAST,
            budget_dollars=2.0,
            max_depth=1,
        )

        d = prefs.to_dict()

        assert d["execution_mode"] == "fast"
        assert d["budget_dollars"] == 2.0
        assert d["max_depth"] == 1

    def test_from_dict(self):
        """Creates from dictionary correctly."""
        data = {
            "execution_mode": "thorough",
            "budget_dollars": 10.0,
            "max_depth": 3,
            "tool_access": "full",
        }

        prefs = UserPreferences.from_dict(data)

        assert prefs.execution_mode == ExecutionMode.THOROUGH
        assert prefs.budget_dollars == 10.0
        assert prefs.max_depth == 3
        assert prefs.tool_access == ToolAccessLevel.FULL

    def test_from_dict_partial(self):
        """Handles partial dictionary with defaults."""
        data = {"execution_mode": "fast"}

        prefs = UserPreferences.from_dict(data)

        assert prefs.execution_mode == ExecutionMode.FAST
        assert prefs.budget_dollars == 5.0  # Default

    def test_save_and_load(self):
        """Saves and loads from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"

            prefs = UserPreferences(
                execution_mode=ExecutionMode.THOROUGH,
                budget_dollars=7.50,
                preferred_model="opus",
            )
            prefs.save(path)

            loaded = UserPreferences.load(path)

            assert loaded.execution_mode == ExecutionMode.THOROUGH
            assert loaded.budget_dollars == 7.50
            assert loaded.preferred_model == "opus"

    def test_load_nonexistent_returns_default(self):
        """Loading nonexistent file returns defaults."""
        path = Path("/nonexistent/path/prefs.json")

        prefs = UserPreferences.load(path)

        assert prefs.execution_mode == ExecutionMode.BALANCED


class TestPreferencesManager:
    """Tests for PreferencesManager class."""

    @pytest.fixture
    def manager(self):
        """Create manager with default preferences."""
        return PreferencesManager(UserPreferences())

    def test_parse_status_command(self, manager):
        """Status command shows current preferences."""
        message, _ = manager.parse_command("status")

        assert "Mode:" in message
        assert "Budget:" in message
        assert "balanced" in message.lower()

    def test_parse_empty_command(self, manager):
        """Empty command shows status."""
        message, _ = manager.parse_command("")

        assert "Mode:" in message

    def test_mode_fast(self, manager):
        """Sets fast mode."""
        message, updates = manager.parse_command("mode fast")

        assert "fast" in message.lower()
        assert updates.get("execution_mode") == "fast"
        assert manager.prefs.execution_mode == ExecutionMode.FAST

    def test_mode_balanced(self, manager):
        """Sets balanced mode."""
        manager.prefs.execution_mode = ExecutionMode.FAST
        message, updates = manager.parse_command("mode balanced")

        assert "balanced" in message.lower()
        assert manager.prefs.execution_mode == ExecutionMode.BALANCED

    def test_mode_thorough(self, manager):
        """Sets thorough mode."""
        message, updates = manager.parse_command("mode thorough")

        assert "thorough" in message.lower()
        assert manager.prefs.execution_mode == ExecutionMode.THOROUGH

    def test_mode_invalid(self, manager):
        """Handles invalid mode."""
        original = manager.prefs.execution_mode
        message, _ = manager.parse_command("mode invalid")

        assert "unknown" in message.lower()
        assert manager.prefs.execution_mode == original

    def test_mode_query(self, manager):
        """Queries current mode."""
        message, _ = manager.parse_command("mode")

        assert "balanced" in message.lower()

    def test_budget_set(self, manager):
        """Sets budget."""
        message, updates = manager.parse_command("budget $10")

        assert "$10" in message or "10.00" in message
        assert manager.prefs.budget_dollars == 10.0

    def test_budget_without_dollar_sign(self, manager):
        """Sets budget without dollar sign."""
        message, _ = manager.parse_command("budget 7.50")

        assert manager.prefs.budget_dollars == 7.50

    def test_budget_invalid(self, manager):
        """Handles invalid budget."""
        original = manager.prefs.budget_dollars
        message, _ = manager.parse_command("budget abc")

        assert "invalid" in message.lower()
        assert manager.prefs.budget_dollars == original

    def test_budget_negative(self, manager):
        """Rejects negative budget."""
        original = manager.prefs.budget_dollars
        message, _ = manager.parse_command("budget -5")

        assert "positive" in message.lower()
        assert manager.prefs.budget_dollars == original

    def test_budget_query(self, manager):
        """Queries current budget."""
        message, _ = manager.parse_command("budget")

        assert "$5.00" in message

    def test_depth_set(self, manager):
        """Sets max depth."""
        message, _ = manager.parse_command("depth 3")

        assert "3" in message
        assert manager.prefs.max_depth == 3

    def test_depth_invalid(self, manager):
        """Handles invalid depth."""
        message, _ = manager.parse_command("depth abc")

        assert "invalid" in message.lower()

    def test_depth_out_of_range(self, manager):
        """Rejects out of range depth."""
        message, _ = manager.parse_command("depth 10")

        assert "between" in message.lower()

    def test_model_set(self, manager):
        """Sets preferred model."""
        message, _ = manager.parse_command("model opus")

        assert "opus" in message.lower()
        assert manager.prefs.preferred_model == "opus"

    def test_model_clear(self, manager):
        """Clears model preference."""
        manager.prefs.preferred_model = "opus"
        message, _ = manager.parse_command("model auto")

        assert "auto" in message.lower() or "cleared" in message.lower()
        assert manager.prefs.preferred_model is None

    def test_model_invalid(self, manager):
        """Handles invalid model."""
        message, _ = manager.parse_command("model invalid_model")

        assert "unknown" in message.lower()

    def test_tools_set(self, manager):
        """Sets tool access level."""
        message, _ = manager.parse_command("tools full")

        assert "full" in message.lower()
        assert manager.prefs.tool_access == ToolAccessLevel.FULL

    def test_tools_repl(self, manager):
        """Sets REPL-only tool access."""
        message, _ = manager.parse_command("tools repl")

        assert manager.prefs.tool_access == ToolAccessLevel.REPL_ONLY

    def test_tools_none(self, manager):
        """Sets no tool access."""
        message, _ = manager.parse_command("tools none")

        assert manager.prefs.tool_access == ToolAccessLevel.NONE

    def test_verbosity_set(self, manager):
        """Sets verbosity level."""
        message, _ = manager.parse_command("verbosity debug")

        assert "debug" in message.lower()
        assert manager.prefs.trajectory_verbosity == "debug"

    def test_on_command(self, manager):
        """Enables RLM."""
        manager.prefs.auto_activate = False
        message, _ = manager.parse_command("on")

        assert "enabled" in message.lower()
        assert manager.prefs.auto_activate is True

    def test_off_command(self, manager):
        """Disables RLM."""
        message, _ = manager.parse_command("off")

        assert "disabled" in message.lower()
        assert manager.prefs.auto_activate is False

    def test_reset_command(self, manager):
        """Resets to defaults."""
        manager.prefs.execution_mode = ExecutionMode.THOROUGH
        manager.prefs.budget_dollars = 100.0

        message, _ = manager.parse_command("reset")

        assert "reset" in message.lower()
        assert manager.prefs.execution_mode == ExecutionMode.BALANCED
        assert manager.prefs.budget_dollars == 5.0

    def test_unknown_command(self, manager):
        """Handles unknown command."""
        message, _ = manager.parse_command("foobar")

        assert "unknown" in message.lower()

    def test_update_history(self, manager):
        """Tracks update history."""
        manager.parse_command("mode fast")
        manager.parse_command("budget 10")

        history = manager.get_update_history()

        assert len(history) == 2
        assert history[0].field == "execution_mode"
        assert history[1].field == "budget_dollars"


class TestPreferenceUpdate:
    """Tests for PreferenceUpdate dataclass."""

    def test_create_update(self):
        """Can create preference update."""
        update = PreferenceUpdate(
            field="execution_mode",
            old_value="balanced",
            new_value="fast",
            reason="user request",
        )

        assert update.field == "execution_mode"
        assert update.old_value == "balanced"
        assert update.new_value == "fast"


class TestHandleRlmCommand:
    """Tests for handle_rlm_command function."""

    def test_handle_mode_command(self):
        """Handles mode command."""
        # Reset to default first
        handle_rlm_command("reset")

        result = handle_rlm_command("mode fast")
        assert "fast" in result.lower()

    def test_handle_status_command(self):
        """Handles status command."""
        result = handle_rlm_command("status")
        assert "Mode:" in result


class TestPreferencesIntegration:
    """Integration tests for preferences system."""

    def test_full_workflow(self):
        """Tests full preferences workflow."""
        manager = PreferencesManager(UserPreferences())

        # Set mode
        manager.parse_command("mode thorough")
        assert manager.prefs.execution_mode == ExecutionMode.THOROUGH

        # Set budget
        manager.parse_command("budget $15")
        assert manager.prefs.budget_dollars == 15.0

        # Set depth
        manager.parse_command("depth 3")
        assert manager.prefs.max_depth == 3

        # Check status
        status, _ = manager.parse_command("status")
        assert "thorough" in status.lower()
        assert "15" in status
        assert "3" in status

        # Reset
        manager.parse_command("reset")
        assert manager.prefs.execution_mode == ExecutionMode.BALANCED

    def test_persistence_workflow(self):
        """Tests save/load workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_prefs.json"

            # Create and save
            prefs1 = UserPreferences(
                execution_mode=ExecutionMode.FAST,
                budget_dollars=3.0,
            )
            prefs1.save(path)

            # Load
            prefs2 = UserPreferences.load(path)

            assert prefs2.execution_mode == ExecutionMode.FAST
            assert prefs2.budget_dollars == 3.0
