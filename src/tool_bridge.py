"""
Tool bridge for sub-LLM tool access.

Implements: Spec §8.1 Phase 2 - Tool Access for Sub-LLMs

Allows REPL code to invoke Claude Code tools with proper
isolation and permission controls.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .orchestration_schema import ToolAccessLevel


@dataclass
class ToolResult:
    """Result from a tool invocation."""

    success: bool
    output: str
    error: str | None = None
    tool_name: str = ""
    truncated: bool = False
    execution_time_ms: float = 0.0


@dataclass
class ToolPermissions:
    """Permissions for tool access."""

    access_level: ToolAccessLevel = ToolAccessLevel.READ_ONLY

    # Fine-grained permissions
    allow_bash: bool = False
    allow_file_read: bool = True
    allow_file_write: bool = False
    allow_search: bool = True
    allow_web: bool = False

    # Restrictions
    allowed_paths: list[str] = field(default_factory=list)  # Empty = all paths
    blocked_commands: list[str] = field(
        default_factory=lambda: ["rm", "mv", "chmod", "chown", "sudo", "su"]
    )
    max_output_chars: int = 50_000
    timeout_seconds: float = 30.0

    @classmethod
    def from_access_level(cls, level: ToolAccessLevel) -> ToolPermissions:
        """Create permissions from access level."""
        if level == ToolAccessLevel.NONE:
            return cls(
                access_level=level,
                allow_bash=False,
                allow_file_read=False,
                allow_file_write=False,
                allow_search=False,
            )
        elif level == ToolAccessLevel.REPL_ONLY:
            return cls(
                access_level=level,
                allow_bash=False,
                allow_file_read=False,
                allow_file_write=False,
                allow_search=False,
            )
        elif level == ToolAccessLevel.READ_ONLY:
            return cls(
                access_level=level,
                allow_bash=False,
                allow_file_read=True,
                allow_file_write=False,
                allow_search=True,
            )
        else:  # FULL
            return cls(
                access_level=level,
                allow_bash=True,
                allow_file_read=True,
                allow_file_write=True,
                allow_search=True,
                allow_web=True,
            )


class ToolBridge:
    """
    Bridge for sub-LLMs to invoke Claude Code tools.

    Implements: Spec §8.1 Tool access with output isolation

    Provides controlled access to:
    - File reading (read, glob, grep)
    - Bash execution (with restrictions)
    - Search operations
    """

    def __init__(
        self,
        permissions: ToolPermissions | None = None,
        working_dir: str | Path | None = None,
    ):
        """
        Initialize tool bridge.

        Args:
            permissions: Tool permissions (defaults to READ_ONLY)
            working_dir: Working directory for tool operations
        """
        self.permissions = permissions or ToolPermissions()
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self._history: list[ToolResult] = []

    def tool_call(self, tool_name: str, *args: Any, **kwargs: Any) -> ToolResult:
        """
        Invoke a tool by name.

        Args:
            tool_name: Name of the tool (read, bash, grep, glob)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            ToolResult with output or error
        """
        import time

        start = time.time()

        # Route to appropriate handler
        handlers: dict[str, Callable[..., ToolResult]] = {
            "read": self._read_file,
            "bash": self._run_bash,
            "grep": self._grep,
            "glob": self._glob,
            "search": self._grep,  # Alias
            "ls": self._list_dir,
        }

        if tool_name not in handlers:
            result = ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {list(handlers.keys())}",
                tool_name=tool_name,
            )
        else:
            try:
                result = handlers[tool_name](*args, **kwargs)
            except Exception as e:
                result = ToolResult(
                    success=False,
                    output="",
                    error=str(e),
                    tool_name=tool_name,
                )

        result.execution_time_ms = (time.time() - start) * 1000
        result.tool_name = tool_name
        self._history.append(result)
        return result

    def _read_file(self, path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
        """Read a file."""
        if not self.permissions.allow_file_read:
            return ToolResult(
                success=False,
                output="",
                error="File reading not permitted",
            )

        file_path = self._resolve_path(path)
        if not self._is_path_allowed(file_path):
            return ToolResult(
                success=False,
                output="",
                error=f"Path not allowed: {path}",
            )

        try:
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}",
                )

            with open(file_path) as f:
                lines = f.readlines()

            # Apply offset and limit
            selected = lines[offset : offset + limit]
            content = "".join(selected)

            # Truncate if needed
            truncated = False
            if len(content) > self.permissions.max_output_chars:
                content = content[: self.permissions.max_output_chars]
                truncated = True

            return ToolResult(
                success=True,
                output=content,
                truncated=truncated,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Read error: {e}",
            )

    def _run_bash(self, command: str, timeout: float | None = None) -> ToolResult:
        """Run a bash command."""
        if not self.permissions.allow_bash:
            return ToolResult(
                success=False,
                output="",
                error="Bash execution not permitted",
            )

        # Check for blocked commands
        cmd_parts = command.split()
        if cmd_parts:
            base_cmd = cmd_parts[0]
            if base_cmd in self.permissions.blocked_commands:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command not allowed: {base_cmd}",
                )

        timeout = timeout or self.permissions.timeout_seconds

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            # Truncate if needed
            truncated = False
            if len(output) > self.permissions.max_output_chars:
                output = output[: self.permissions.max_output_chars]
                truncated = True

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Bash error: {e}",
            )

    def _grep(
        self,
        pattern: str,
        path: str | None = None,
        ignore_case: bool = True,
    ) -> ToolResult:
        """Search for pattern in files."""
        if not self.permissions.allow_search:
            return ToolResult(
                success=False,
                output="",
                error="Search not permitted",
            )

        search_path = self._resolve_path(path) if path else self.working_dir

        # Build grep command
        flags = "-rn"
        if ignore_case:
            flags += "i"

        # Use grep for searching
        cmd = f"grep {flags} '{pattern}' {search_path}"

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.permissions.timeout_seconds,
                cwd=self.working_dir,
            )

            output = result.stdout

            # Truncate if needed
            truncated = False
            if len(output) > self.permissions.max_output_chars:
                output = output[: self.permissions.max_output_chars]
                truncated = True

            return ToolResult(
                success=True,
                output=output,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Search timed out",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Search error: {e}",
            )

    def _glob(self, pattern: str, path: str | None = None) -> ToolResult:
        """Find files matching pattern."""
        if not self.permissions.allow_search:
            return ToolResult(
                success=False,
                output="",
                error="Search not permitted",
            )

        search_path = self._resolve_path(path) if path else self.working_dir

        try:
            matches = list(search_path.glob(pattern))

            # Sort by modification time (most recent first)
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Limit results
            max_results = 100
            truncated = len(matches) > max_results
            matches = matches[:max_results]

            output = "\n".join(str(m.relative_to(self.working_dir)) for m in matches)

            return ToolResult(
                success=True,
                output=output,
                truncated=truncated,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Glob error: {e}",
            )

    def _list_dir(self, path: str | None = None) -> ToolResult:
        """List directory contents."""
        if not self.permissions.allow_file_read:
            return ToolResult(
                success=False,
                output="",
                error="Directory listing not permitted",
            )

        dir_path = self._resolve_path(path) if path else self.working_dir

        try:
            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a directory: {path}",
                )

            entries = list(dir_path.iterdir())
            entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

            lines = []
            for entry in entries[:100]:  # Limit to 100 entries
                prefix = "d" if entry.is_dir() else "-"
                lines.append(f"{prefix} {entry.name}")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                truncated=len(entries) > 100,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"List error: {e}",
            )

    def _resolve_path(self, path: str | None) -> Path:
        """Resolve a path relative to working directory."""
        if path is None:
            return self.working_dir

        p = Path(path)
        if p.is_absolute():
            return p
        return self.working_dir / p

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is allowed by permissions."""
        if not self.permissions.allowed_paths:
            return True  # No restrictions

        resolved = path.resolve()
        for allowed in self.permissions.allowed_paths:
            allowed_path = Path(allowed).resolve()
            try:
                resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue

        return False

    def get_history(self) -> list[ToolResult]:
        """Get tool invocation history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear tool invocation history."""
        self._history.clear()


def create_tool_bridge(access_level: ToolAccessLevel) -> ToolBridge:
    """
    Create a tool bridge with the specified access level.

    Args:
        access_level: Tool access level

    Returns:
        Configured ToolBridge
    """
    permissions = ToolPermissions.from_access_level(access_level)
    return ToolBridge(permissions=permissions)


__all__ = [
    "ToolBridge",
    "ToolPermissions",
    "ToolResult",
    "create_tool_bridge",
]
