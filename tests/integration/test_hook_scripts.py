"""
Integration tests for hook scripts.

Tests the full flow of hook script execution.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_env_with_pythonpath(tmp_path=None):
    """Get environment with correct PYTHONPATH for imports."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    if tmp_path:
        env["HOME"] = str(tmp_path)
    return env


class TestInitRLMScript:
    """Tests for init_rlm.py script."""

    def test_creates_config_file(self, tmp_path, monkeypatch):
        """Script creates default config if not exists."""
        # Use temp home directory
        monkeypatch.setenv("HOME", str(tmp_path))

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "legacy" / "init_rlm.py")],
            capture_output=True,
            text=True,
            env=get_env_with_pythonpath(tmp_path),
        )

        # Check config was created
        config_file = tmp_path / ".claude" / "rlm-config.json"
        assert config_file.exists()

        # Verify config content
        with open(config_file) as f:
            config = json.load(f)

        assert "activation" in config
        assert "depth" in config
        assert "models" in config
        assert config["activation"]["mode"] == "complexity"

    def test_creates_trajectories_dir(self, tmp_path, monkeypatch):
        """Script creates trajectories directory."""
        monkeypatch.setenv("HOME", str(tmp_path))

        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "legacy" / "init_rlm.py")],
            capture_output=True,
            text=True,
            env=get_env_with_pythonpath(tmp_path),
        )

        trajectories_dir = tmp_path / ".claude" / "rlm-trajectories"
        assert trajectories_dir.exists()


class TestCheckComplexityScript:
    """Tests for check_complexity.py script.

    The script uses Claude Code compliant hook output:
    - When RLM activates: outputs hookSpecificOutput with additionalContext
    - When RLM does not activate: outputs nothing (empty stdout)

    The actual activation decision is stored in ~/.claude/rlm-state/activation.json
    """

    def test_simple_query_not_activated(self, tmp_path, monkeypatch):
        """Simple query should not activate RLM and produce no stdout."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "legacy" / "check_complexity.py"),
                "git status",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=get_env_with_pythonpath(tmp_path),
        )

        # When not activated, stdout should be empty (preserves prompt suggestions)
        assert result.stdout.strip() == ""

        # Activation state is saved to file
        state_file = tmp_path / ".claude" / "rlm-state" / "activation.json"
        assert state_file.exists()
        with open(state_file) as f:
            state = json.load(f)
        assert state["activate_rlm"] is False

    def test_complex_query_activated(self, tmp_path, monkeypatch):
        """Complex query should activate RLM with proper hook output."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "legacy" / "check_complexity.py"),
                "Find the bug in auth.py and fix it, then update the tests in test_auth.py",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=get_env_with_pythonpath(tmp_path),
        )

        # When activated, should output Claude Code compliant format
        output = json.loads(result.stdout)
        assert "hookSpecificOutput" in output
        assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "additionalContext" in output["hookSpecificOutput"]
        assert "RLM auto-activated" in output["hookSpecificOutput"]["additionalContext"]

        # Activation state is also saved to file
        state_file = tmp_path / ".claude" / "rlm-state" / "activation.json"
        assert state_file.exists()
        with open(state_file) as f:
            state = json.load(f)
        assert state["activate_rlm"] is True


class TestSyncContextScript:
    """Tests for sync_context.py script."""

    def test_runs_without_error(self, tmp_path, monkeypatch):
        """Script runs without error."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_SESSION_ID", "test-session")

        env = get_env_with_pythonpath(tmp_path)
        env["CLAUDE_SESSION_ID"] = "test-session"

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "legacy" / "sync_context.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        # Should output JSON
        output = json.loads(result.stdout)
        assert "status" in output


class TestCaptureOutputScript:
    """Tests for capture_output.py script."""

    def test_captures_tool_output(self, tmp_path, monkeypatch):
        """Script captures tool output."""
        monkeypatch.setenv("HOME", str(tmp_path))

        env = get_env_with_pythonpath(tmp_path)
        env["CLAUDE_SESSION_ID"] = "capture-test"
        env["CLAUDE_TOOL_NAME"] = "bash"
        env["CLAUDE_TOOL_OUTPUT"] = "test output"
        env["CLAUDE_TOOL_EXIT_CODE"] = "0"

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "legacy" / "capture_output.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        output = json.loads(result.stdout)
        assert output["status"] == "captured"
        assert output["tool"] == "bash"


class TestExternalizeContextScript:
    """Tests for externalize_context.py script."""

    def test_creates_externalized_files(self, tmp_path, monkeypatch):
        """Script creates externalized context files."""
        monkeypatch.setenv("HOME", str(tmp_path))

        env = get_env_with_pythonpath(tmp_path)
        env["CLAUDE_SESSION_ID"] = "extern-test"

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "legacy" / "externalize_context.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        output = json.loads(result.stdout)
        assert output["status"] == "externalized"
        assert "manifest" in output

        # Check manifest exists
        manifest_path = Path(output["manifest"])
        assert manifest_path.exists()


class TestSaveTrajectoryScript:
    """Tests for save_trajectory.py script."""

    def test_saves_on_session_end(self, tmp_path, monkeypatch):
        """Script saves trajectory on session end."""
        # First init a session
        monkeypatch.setenv("HOME", str(tmp_path))

        env = get_env_with_pythonpath(tmp_path)

        # Initialize session first
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "legacy" / "init_rlm.py")],
            capture_output=True,
            text=True,
            env=env,
        )

        # Sync context to create session
        env["CLAUDE_SESSION_ID"] = "save-test"
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "legacy" / "sync_context.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        # Save trajectory
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "legacy" / "save_trajectory.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        output = json.loads(result.stdout)
        assert output["status"] in ["saved", "skipped"]
