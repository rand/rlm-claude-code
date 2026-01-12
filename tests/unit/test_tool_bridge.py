"""
Unit tests for tool_bridge module.

Implements: Spec §8.1 Phase 2 - Tool Access tests
"""

import tempfile
from pathlib import Path

import pytest

from src.orchestration_schema import ToolAccessLevel
from src.tool_bridge import (
    ToolBridge,
    ToolPermissions,
    ToolResult,
    create_tool_bridge,
)


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_create_success_result(self):
        """Can create successful result."""
        result = ToolResult(
            success=True,
            output="file contents here",
            tool_name="read",
        )

        assert result.success is True
        assert result.output == "file contents here"
        assert result.error is None

    def test_create_error_result(self):
        """Can create error result."""
        result = ToolResult(
            success=False,
            output="",
            error="Permission denied",
            tool_name="bash",
        )

        assert result.success is False
        assert result.error == "Permission denied"


class TestToolPermissions:
    """Tests for ToolPermissions dataclass."""

    def test_default_permissions(self):
        """Default permissions are read-only."""
        perms = ToolPermissions()

        assert perms.access_level == ToolAccessLevel.READ_ONLY
        assert perms.allow_file_read is True
        assert perms.allow_file_write is False
        assert perms.allow_bash is False

    def test_from_none_level(self):
        """None level disables all access."""
        perms = ToolPermissions.from_access_level(ToolAccessLevel.NONE)

        assert perms.allow_bash is False
        assert perms.allow_file_read is False
        assert perms.allow_search is False

    def test_from_repl_only_level(self):
        """REPL only level has minimal access."""
        perms = ToolPermissions.from_access_level(ToolAccessLevel.REPL_ONLY)

        assert perms.allow_bash is False
        assert perms.allow_file_read is False
        assert perms.allow_search is False

    def test_from_read_only_level(self):
        """Read only level allows reading and search."""
        perms = ToolPermissions.from_access_level(ToolAccessLevel.READ_ONLY)

        assert perms.allow_bash is False
        assert perms.allow_file_read is True
        assert perms.allow_file_write is False
        assert perms.allow_search is True

    def test_from_full_level(self):
        """Full level allows everything."""
        perms = ToolPermissions.from_access_level(ToolAccessLevel.FULL)

        assert perms.allow_bash is True
        assert perms.allow_file_read is True
        assert perms.allow_file_write is True
        assert perms.allow_search is True

    def test_blocked_commands(self):
        """Has sensible blocked commands."""
        perms = ToolPermissions()

        assert "rm" in perms.blocked_commands
        assert "sudo" in perms.blocked_commands


class TestToolBridge:
    """Tests for ToolBridge class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create test files
            (path / "test.txt").write_text("Hello World\nLine 2\nLine 3")
            (path / "code.py").write_text("def main():\n    print('hi')")
            (path / "subdir").mkdir()
            (path / "subdir" / "nested.txt").write_text("Nested content")

            yield path

    @pytest.fixture
    def read_only_bridge(self, temp_dir):
        """Create read-only tool bridge."""
        perms = ToolPermissions.from_access_level(ToolAccessLevel.READ_ONLY)
        return ToolBridge(permissions=perms, working_dir=temp_dir)

    @pytest.fixture
    def full_bridge(self, temp_dir):
        """Create full-access tool bridge."""
        perms = ToolPermissions.from_access_level(ToolAccessLevel.FULL)
        return ToolBridge(permissions=perms, working_dir=temp_dir)

    def test_read_file(self, read_only_bridge):
        """Can read file with read permissions."""
        result = read_only_bridge.tool_call("read", "test.txt")

        assert result.success is True
        assert "Hello World" in result.output
        assert result.tool_name == "read"

    def test_read_file_not_found(self, read_only_bridge):
        """Returns error for missing file."""
        result = read_only_bridge.tool_call("read", "nonexistent.txt")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_with_offset_limit(self, read_only_bridge):
        """Respects offset and limit."""
        result = read_only_bridge.tool_call("read", "test.txt", offset=1, limit=1)

        assert result.success is True
        assert "Line 2" in result.output
        assert "Hello World" not in result.output

    def test_read_denied_without_permission(self, temp_dir):
        """Denies read without permission."""
        perms = ToolPermissions.from_access_level(ToolAccessLevel.NONE)
        bridge = ToolBridge(permissions=perms, working_dir=temp_dir)

        result = bridge.tool_call("read", "test.txt")

        assert result.success is False
        assert "not permitted" in result.error.lower()

    def test_bash_with_full_permission(self, full_bridge):
        """Can run bash with full permissions."""
        result = full_bridge.tool_call("bash", "echo hello")

        assert result.success is True
        assert "hello" in result.output

    def test_bash_denied_without_permission(self, read_only_bridge):
        """Denies bash without permission."""
        result = read_only_bridge.tool_call("bash", "echo hello")

        assert result.success is False
        assert "not permitted" in result.error.lower()

    def test_bash_blocks_dangerous_commands(self, full_bridge):
        """Blocks dangerous commands."""
        result = full_bridge.tool_call("bash", "rm -rf /")

        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_grep(self, read_only_bridge):
        """Can search with grep."""
        result = read_only_bridge.tool_call("grep", "Hello", "test.txt")

        assert result.success is True
        assert "Hello" in result.output

    def test_grep_case_insensitive(self, read_only_bridge):
        """Grep is case insensitive by default."""
        result = read_only_bridge.tool_call("grep", "hello", "test.txt")

        assert result.success is True
        assert "Hello" in result.output

    def test_glob(self, read_only_bridge):
        """Can find files with glob."""
        result = read_only_bridge.tool_call("glob", "*.txt")

        assert result.success is True
        assert "test.txt" in result.output

    def test_glob_recursive(self, read_only_bridge):
        """Can find files recursively."""
        result = read_only_bridge.tool_call("glob", "**/*.txt")

        assert result.success is True
        assert "test.txt" in result.output
        assert "nested.txt" in result.output

    def test_list_dir(self, read_only_bridge):
        """Can list directory."""
        result = read_only_bridge.tool_call("ls")

        assert result.success is True
        assert "test.txt" in result.output
        assert "subdir" in result.output

    def test_list_subdir(self, read_only_bridge):
        """Can list subdirectory."""
        result = read_only_bridge.tool_call("ls", "subdir")

        assert result.success is True
        assert "nested.txt" in result.output

    def test_unknown_tool(self, read_only_bridge):
        """Returns error for unknown tool."""
        result = read_only_bridge.tool_call("unknown_tool")

        assert result.success is False
        assert "Unknown tool" in result.error

    def test_history_tracking(self, read_only_bridge):
        """Tracks tool invocation history."""
        read_only_bridge.tool_call("read", "test.txt")
        read_only_bridge.tool_call("glob", "*.py")

        history = read_only_bridge.get_history()

        assert len(history) == 2
        assert history[0].tool_name == "read"
        assert history[1].tool_name == "glob"

    def test_clear_history(self, read_only_bridge):
        """Can clear history."""
        read_only_bridge.tool_call("read", "test.txt")
        read_only_bridge.clear_history()

        assert len(read_only_bridge.get_history()) == 0

    def test_execution_time_tracked(self, read_only_bridge):
        """Tracks execution time."""
        result = read_only_bridge.tool_call("read", "test.txt")

        assert result.execution_time_ms > 0


class TestPathRestrictions:
    """Tests for path restriction functionality."""

    @pytest.fixture
    def temp_dirs(self):
        """Create allowed and blocked directories."""
        with tempfile.TemporaryDirectory() as allowed:
            with tempfile.TemporaryDirectory() as blocked:
                allowed_path = Path(allowed)
                blocked_path = Path(blocked)

                (allowed_path / "allowed.txt").write_text("Allowed content")
                (blocked_path / "blocked.txt").write_text("Blocked content")

                yield allowed_path, blocked_path

    def test_allowed_path_accessible(self, temp_dirs):
        """Can access allowed paths."""
        allowed_path, blocked_path = temp_dirs

        perms = ToolPermissions(
            allow_file_read=True,
            allowed_paths=[str(allowed_path)],
        )
        bridge = ToolBridge(permissions=perms, working_dir=allowed_path)

        result = bridge.tool_call("read", "allowed.txt")

        assert result.success is True

    def test_blocked_path_denied(self, temp_dirs):
        """Denies access to paths outside allowed list."""
        allowed_path, blocked_path = temp_dirs

        perms = ToolPermissions(
            allow_file_read=True,
            allowed_paths=[str(allowed_path)],
        )
        bridge = ToolBridge(permissions=perms, working_dir=allowed_path)

        result = bridge.tool_call("read", str(blocked_path / "blocked.txt"))

        assert result.success is False
        assert "not allowed" in result.error.lower()


class TestCreateToolBridge:
    """Tests for create_tool_bridge factory function."""

    def test_creates_read_only(self):
        """Creates read-only bridge."""
        bridge = create_tool_bridge(ToolAccessLevel.READ_ONLY)

        assert bridge.permissions.allow_file_read is True
        assert bridge.permissions.allow_bash is False

    def test_creates_full_access(self):
        """Creates full-access bridge."""
        bridge = create_tool_bridge(ToolAccessLevel.FULL)

        assert bridge.permissions.allow_bash is True
        assert bridge.permissions.allow_file_write is True
