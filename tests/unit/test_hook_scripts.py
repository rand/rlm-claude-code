"""
Unit tests for Python fallback hook scripts.

Tests that session-init.py and trajectory-save.py produce valid JSON
and handle errors gracefully (fail-open).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


class TestSessionInitHook:
    """Tests for scripts/session-init.py."""

    def test_produces_valid_json(self):
        """session-init.py produces valid JSON output."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "session-init.py")],
            capture_output=True,
            text=True,
            timeout=10,
        )

        output = json.loads(result.stdout.strip())
        assert "status" in output

    def test_initializes_with_session_id(self):
        """session-init.py uses CLAUDE_SESSION_ID from env."""
        env = {"CLAUDE_SESSION_ID": "test-hook-session", "PATH": ""}
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "session-init.py")],
            capture_output=True,
            text=True,
            timeout=10,
            env={**dict(__import__("os").environ), **env},
        )

        output = json.loads(result.stdout.strip())
        # Either initialized successfully or skipped gracefully
        assert output["status"] in ("initialized", "skipped")

    def test_fail_open_on_error(self):
        """session-init.py outputs skipped status on import failure."""
        script = SCRIPTS_DIR / "session-init.py"
        # Poison state_persistence module to force ImportError,
        # while keeping stdlib intact and __file__ defined for the script
        code = (
            f"__file__ = {str(script)!r}; "
            "import sys; sys.modules['state_persistence'] = None; "
            f"exec(compile(open({str(script)!r}).read(), {str(script)!r}, 'exec'))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should not crash — must produce valid JSON with skipped status
        assert result.returncode == 0, f"Hook crashed: {result.stderr}"
        assert result.stdout.strip(), "Hook produced no output"
        output = json.loads(result.stdout.strip())
        assert output["status"] == "skipped"


class TestTrajectorySaveHook:
    """Tests for scripts/trajectory-save.py."""

    def test_produces_valid_json(self):
        """trajectory-save.py produces valid JSON output."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "trajectory-save.py")],
            capture_output=True,
            text=True,
            timeout=10,
        )

        output = json.loads(result.stdout.strip())
        assert "status" in output

    def test_handles_no_state_gracefully(self):
        """trajectory-save.py handles missing state file gracefully."""
        env = {"CLAUDE_SESSION_ID": "nonexistent-session-12345"}
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "trajectory-save.py")],
            capture_output=True,
            text=True,
            timeout=10,
            env={**dict(__import__("os").environ), **env},
        )

        output = json.loads(result.stdout.strip())
        assert output["status"] == "skipped"
