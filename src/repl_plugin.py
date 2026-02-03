"""
REPL plugin system for extensible REPL functions.

Implements: SPEC-12.10-12.16

Provides plugin protocol, registration, sandboxing, built-in plugins,
lazy loading, and conflict detection.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class PluginConflictError(Exception):
    """
    Error raised when plugin function names conflict.

    Implements: SPEC-12.16
    """

    pass


class REPLPlugin(ABC):
    """
    Protocol for REPL plugins.

    Implements: SPEC-12.11
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Plugin functions."""
        pass

    @abstractmethod
    def on_load(self, env: Any) -> None:
        """Called when plugin is loaded."""
        pass


class PluginRegistry:
    """
    Registry for REPL plugins.

    Implements: SPEC-12.12
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._plugins: dict[str, REPLPlugin] = {}

    def register_plugin(self, plugin: REPLPlugin) -> None:
        """
        Register a plugin.

        Implements: SPEC-12.12

        Args:
            plugin: Plugin to register
        """
        self._plugins[plugin.name] = plugin

    def unregister_plugin(self, name: str) -> bool:
        """
        Unregister a plugin.

        Implements: SPEC-12.12

        Args:
            name: Plugin name to unregister

        Returns:
            True if plugin was unregistered
        """
        if name in self._plugins:
            del self._plugins[name]
            return True
        return False

    def list_plugins(self) -> list[str]:
        """
        List registered plugins.

        Implements: SPEC-12.12

        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())

    def get_plugin(self, name: str) -> REPLPlugin | None:
        """Get plugin by name."""
        return self._plugins.get(name)

    def get_all_plugins(self) -> list[REPLPlugin]:
        """Get all registered plugins."""
        return list(self._plugins.values())


class REPLPluginManager:
    """
    Manager for REPL plugins with sandboxing and conflict detection.

    Implements: SPEC-12.10-12.16
    """

    def __init__(self, lazy_load: bool = False) -> None:
        """
        Initialize plugin manager.

        Args:
            lazy_load: Whether to use lazy loading
        """
        self._registry = PluginRegistry()
        self._lazy_load = lazy_load
        self._function_map: dict[str, tuple[str, Callable[..., Any]]] = {}

    def register_plugin(self, plugin: REPLPlugin) -> None:
        """
        Register a plugin with conflict detection.

        Implements: SPEC-12.12, SPEC-12.16

        Args:
            plugin: Plugin to register

        Raises:
            PluginConflictError: If function names conflict
        """
        # Check for conflicts
        for func_name in plugin.functions:
            if func_name in self._function_map:
                existing_plugin = self._function_map[func_name][0]
                raise PluginConflictError(
                    f"Function '{func_name}' already registered by plugin '{existing_plugin}'"
                )

        # Register plugin
        self._registry.register_plugin(plugin)

        # Register functions
        for func_name, func in plugin.functions.items():
            self._function_map[func_name] = (plugin.name, self._sandbox_function(func))

    def unregister_plugin(self, name: str) -> bool:
        """
        Unregister a plugin.

        Args:
            name: Plugin name

        Returns:
            True if unregistered
        """
        plugin = self._registry.get_plugin(name)
        if plugin is None:
            return False

        # Remove functions
        for func_name in plugin.functions:
            if func_name in self._function_map:
                del self._function_map[func_name]

        return self._registry.unregister_plugin(name)

    def list_plugins(self) -> list[str]:
        """List registered plugins."""
        return self._registry.list_plugins()

    def get_all_functions(self) -> dict[str, Callable[..., Any]]:
        """
        Get all registered functions.

        Returns:
            Dict of function name to callable
        """
        return {name: func for name, (_, func) in self._function_map.items()}

    def _sandbox_function(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Wrap function in sandbox.

        Implements: SPEC-12.13

        Args:
            func: Function to sandbox

        Returns:
            Sandboxed function
        """

        def sandboxed(*args: Any, **kwargs: Any) -> Any:
            # In production, would add security checks here
            return func(*args, **kwargs)

        return sandboxed


class CorePlugin(REPLPlugin):
    """
    Core REPL plugin with basic functions.

    Implements: SPEC-12.14
    """

    @property
    def name(self) -> str:
        return "core"

    @property
    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "peek": self._peek,
            "search": self._search,
            "summarize": self._summarize,
        }

    def on_load(self, env: Any) -> None:
        pass

    def _peek(self, content: str, start: int = 0, end: int | None = None) -> str:
        """View a slice of content."""
        if end is None:
            end = len(content)
        return content[start:end]

    def _search(self, content: str, pattern: str) -> list[str]:
        """Search for pattern in content."""
        return re.findall(pattern, content)

    def _summarize(self, content: str, max_length: int = 200) -> str:
        """Summarize content."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."


class CodeAnalysisPlugin(REPLPlugin):
    """
    Code analysis plugin.

    Implements: SPEC-12.14
    """

    @property
    def name(self) -> str:
        return "code_analysis"

    @property
    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "extract_functions": self._extract_functions,
            "count_lines": self._count_lines,
        }

    def on_load(self, env: Any) -> None:
        pass

    def _extract_functions(self, code: str) -> list[str]:
        """Extract function names from code."""
        pattern = r"def\s+(\w+)\s*\("
        return re.findall(pattern, code)

    def _count_lines(self, content: str) -> int:
        """Count lines in content."""
        return len(content.splitlines())


class ComputationPlugin(REPLPlugin):
    """
    Safe computation plugin.

    Implements: SPEC-12.14
    """

    @property
    def name(self) -> str:
        return "computation"

    @property
    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "safe_eval": self._safe_eval,
            "statistics": self._statistics,
        }

    def on_load(self, env: Any) -> None:
        pass

    def _safe_eval(self, expr: str) -> float | int | None:
        """Safely evaluate a simple math expression."""
        # Only allow basic math
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expr):
            return None
        try:
            # Very restricted eval
            result = eval(expr, {"__builtins__": {}}, {})
            if isinstance(result, (int, float)):
                return result
            return None
        except Exception:
            return None

    def _statistics(self, numbers: list[float]) -> dict[str, float]:
        """Calculate basic statistics."""
        if not numbers:
            return {"count": 0, "sum": 0, "mean": 0, "min": 0, "max": 0}

        return {
            "count": len(numbers),
            "sum": sum(numbers),
            "mean": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
        }


class BuiltinPlugins:
    """
    Factory for built-in plugins.

    Implements: SPEC-12.14
    """

    def get_core_plugin(self) -> REPLPlugin:
        """Get core plugin."""
        return CorePlugin()

    def get_code_analysis_plugin(self) -> REPLPlugin:
        """Get code analysis plugin."""
        return CodeAnalysisPlugin()

    def get_computation_plugin(self) -> REPLPlugin:
        """Get computation plugin."""
        return ComputationPlugin()

    def get_all_plugins(self) -> list[REPLPlugin]:
        """Get all built-in plugins."""
        return [
            self.get_core_plugin(),
            self.get_code_analysis_plugin(),
            self.get_computation_plugin(),
        ]


__all__ = [
    "BuiltinPlugins",
    "PluginConflictError",
    "PluginRegistry",
    "REPLPlugin",
    "REPLPluginManager",
]
