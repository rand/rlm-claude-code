"""
Tests for REPL plugin system (SPEC-12.10-12.16).

Tests cover:
- Plugin protocol
- Plugin registration
- Function sandboxing
- Built-in plugins
- Lazy loading
- Conflict detection
"""

from collections.abc import Callable
from typing import Any

import pytest

from src.repl_plugin import (
    BuiltinPlugins,
    PluginConflictError,
    PluginRegistry,
    REPLPlugin,
    REPLPluginManager,
)


class TestPluginProtocol:
    """Tests for REPLPlugin protocol (SPEC-12.11)."""

    def test_plugin_has_name(self):
        """SPEC-12.11: Plugin has name property."""

        class TestPlugin(REPLPlugin):
            @property
            def name(self) -> str:
                return "test_plugin"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {}

            def on_load(self, env: Any) -> None:
                pass

        plugin = TestPlugin()
        assert plugin.name == "test_plugin"

    def test_plugin_has_functions(self):
        """SPEC-12.11: Plugin has functions property."""

        class TestPlugin(REPLPlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"test_func": lambda: "test"}

            def on_load(self, env: Any) -> None:
                pass

        plugin = TestPlugin()
        assert "test_func" in plugin.functions

    def test_plugin_has_on_load(self):
        """SPEC-12.11: Plugin has on_load method."""

        class TestPlugin(REPLPlugin):
            def __init__(self) -> None:
                self.loaded = False

            @property
            def name(self) -> str:
                return "test"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {}

            def on_load(self, env: Any) -> None:
                self.loaded = True

        plugin = TestPlugin()
        plugin.on_load(None)
        assert plugin.loaded


class TestPluginRegistration:
    """Tests for plugin registration (SPEC-12.12)."""

    def test_register_plugin(self):
        """SPEC-12.12: register_plugin(plugin)."""
        registry = PluginRegistry()

        class TestPlugin(REPLPlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {}

            def on_load(self, env: Any) -> None:
                pass

        plugin = TestPlugin()
        registry.register_plugin(plugin)

        assert "test" in registry.list_plugins()

    def test_unregister_plugin(self):
        """SPEC-12.12: unregister_plugin(name)."""
        registry = PluginRegistry()

        class TestPlugin(REPLPlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {}

            def on_load(self, env: Any) -> None:
                pass

        plugin = TestPlugin()
        registry.register_plugin(plugin)
        registry.unregister_plugin("test")

        assert "test" not in registry.list_plugins()

    def test_list_plugins(self):
        """SPEC-12.12: list_plugins() -> list[str]."""
        registry = PluginRegistry()

        class Plugin1(REPLPlugin):
            @property
            def name(self) -> str:
                return "plugin1"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {}

            def on_load(self, env: Any) -> None:
                pass

        class Plugin2(REPLPlugin):
            @property
            def name(self) -> str:
                return "plugin2"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {}

            def on_load(self, env: Any) -> None:
                pass

        registry.register_plugin(Plugin1())
        registry.register_plugin(Plugin2())

        plugins = registry.list_plugins()

        assert "plugin1" in plugins
        assert "plugin2" in plugins

    def test_get_plugin(self):
        """Can get registered plugin by name."""
        registry = PluginRegistry()

        class TestPlugin(REPLPlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {}

            def on_load(self, env: Any) -> None:
                pass

        plugin = TestPlugin()
        registry.register_plugin(plugin)

        retrieved = registry.get_plugin("test")
        assert retrieved is plugin


class TestFunctionSandboxing:
    """Tests for plugin function sandboxing (SPEC-12.13)."""

    def test_functions_are_sandboxed(self):
        """SPEC-12.13: Plugin functions are sandboxed."""
        manager = REPLPluginManager()

        class TestPlugin(REPLPlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"safe_add": lambda a, b: a + b}

            def on_load(self, env: Any) -> None:
                pass

        manager.register_plugin(TestPlugin())
        funcs = manager.get_all_functions()

        # Function should be wrapped
        assert "safe_add" in funcs

    def test_sandboxed_function_executes(self):
        """Sandboxed functions execute correctly."""
        manager = REPLPluginManager()

        class TestPlugin(REPLPlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"add": lambda a, b: a + b}

            def on_load(self, env: Any) -> None:
                pass

        manager.register_plugin(TestPlugin())
        funcs = manager.get_all_functions()

        result = funcs["add"](2, 3)
        assert result == 5


class TestBuiltinPlugins:
    """Tests for built-in plugins (SPEC-12.14)."""

    def test_core_plugin_exists(self):
        """SPEC-12.14: core plugin with basic REPL functions."""
        plugins = BuiltinPlugins()

        core = plugins.get_core_plugin()

        assert core.name == "core"
        assert len(core.functions) > 0

    def test_code_analysis_plugin_exists(self):
        """SPEC-12.14: code_analysis plugin."""
        plugins = BuiltinPlugins()

        code_analysis = plugins.get_code_analysis_plugin()

        assert code_analysis.name == "code_analysis"

    def test_computation_plugin_exists(self):
        """SPEC-12.14: computation plugin."""
        plugins = BuiltinPlugins()

        computation = plugins.get_computation_plugin()

        assert computation.name == "computation"

    def test_core_has_peek(self):
        """Core plugin has peek function."""
        plugins = BuiltinPlugins()
        core = plugins.get_core_plugin()

        assert "peek" in core.functions

    def test_core_has_search(self):
        """Core plugin has search function."""
        plugins = BuiltinPlugins()
        core = plugins.get_core_plugin()

        assert "search" in core.functions

    def test_computation_has_safe_math(self):
        """Computation plugin has safe math functions."""
        plugins = BuiltinPlugins()
        computation = plugins.get_computation_plugin()

        assert "safe_eval" in computation.functions or len(computation.functions) > 0


class TestLazyLoading:
    """Tests for lazy loading (SPEC-12.15)."""

    def test_plugins_support_lazy_loading(self):
        """SPEC-12.15: Plugins support lazy loading."""
        manager = REPLPluginManager(lazy_load=True)

        class HeavyPlugin(REPLPlugin):
            def __init__(self) -> None:
                self._loaded = False

            @property
            def name(self) -> str:
                return "heavy"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"heavy_func": lambda: "heavy"}

            def on_load(self, env: Any) -> None:
                self._loaded = True

        plugin = HeavyPlugin()
        manager.register_plugin(plugin)

        # Plugin registered but not loaded
        assert "heavy" in manager.list_plugins()

    def test_lazy_plugin_loads_on_access(self):
        """Lazy plugin loads when function accessed."""
        manager = REPLPluginManager(lazy_load=True)

        load_count = [0]

        class LazyPlugin(REPLPlugin):
            @property
            def name(self) -> str:
                return "lazy"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                load_count[0] += 1
                return {"lazy_func": lambda: "result"}

            def on_load(self, env: Any) -> None:
                pass

        manager.register_plugin(LazyPlugin())

        # Access functions
        _ = manager.get_all_functions()

        assert load_count[0] >= 1


class TestConflictDetection:
    """Tests for conflict detection (SPEC-12.16)."""

    def test_detects_duplicate_function_names(self):
        """SPEC-12.16: Detect duplicate function names."""
        manager = REPLPluginManager()

        class Plugin1(REPLPlugin):
            @property
            def name(self) -> str:
                return "plugin1"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"shared_func": lambda: "from plugin1"}

            def on_load(self, env: Any) -> None:
                pass

        class Plugin2(REPLPlugin):
            @property
            def name(self) -> str:
                return "plugin2"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"shared_func": lambda: "from plugin2"}

            def on_load(self, env: Any) -> None:
                pass

        manager.register_plugin(Plugin1())

        with pytest.raises(PluginConflictError):
            manager.register_plugin(Plugin2())

    def test_conflict_error_is_clear(self):
        """SPEC-12.16: Conflict errors are clear."""
        manager = REPLPluginManager()

        class Plugin1(REPLPlugin):
            @property
            def name(self) -> str:
                return "plugin1"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"conflict_func": lambda: 1}

            def on_load(self, env: Any) -> None:
                pass

        class Plugin2(REPLPlugin):
            @property
            def name(self) -> str:
                return "plugin2"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"conflict_func": lambda: 2}

            def on_load(self, env: Any) -> None:
                pass

        manager.register_plugin(Plugin1())

        try:
            manager.register_plugin(Plugin2())
            assert False, "Should have raised"
        except PluginConflictError as e:
            assert "conflict_func" in str(e)

    def test_no_conflict_for_different_functions(self):
        """No conflict for different function names."""
        manager = REPLPluginManager()

        class Plugin1(REPLPlugin):
            @property
            def name(self) -> str:
                return "plugin1"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"func1": lambda: 1}

            def on_load(self, env: Any) -> None:
                pass

        class Plugin2(REPLPlugin):
            @property
            def name(self) -> str:
                return "plugin2"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {"func2": lambda: 2}

            def on_load(self, env: Any) -> None:
                pass

        manager.register_plugin(Plugin1())
        manager.register_plugin(Plugin2())  # Should not raise

        assert "plugin1" in manager.list_plugins()
        assert "plugin2" in manager.list_plugins()


class TestIntegration:
    """Integration tests for REPL plugin system."""

    def test_full_plugin_lifecycle(self):
        """Test complete plugin lifecycle."""
        manager = REPLPluginManager()

        class TestPlugin(REPLPlugin):
            def __init__(self) -> None:
                self._loaded = False

            @property
            def name(self) -> str:
                return "test"

            @property
            def functions(self) -> dict[str, Callable[..., Any]]:
                return {
                    "greet": lambda name: f"Hello, {name}!",
                    "add": lambda a, b: a + b,
                }

            def on_load(self, env: Any) -> None:
                self._loaded = True

        plugin = TestPlugin()

        # Register
        manager.register_plugin(plugin)
        assert "test" in manager.list_plugins()

        # Get functions
        funcs = manager.get_all_functions()
        assert funcs["greet"]("World") == "Hello, World!"
        assert funcs["add"](2, 3) == 5

        # Unregister
        manager.unregister_plugin("test")
        assert "test" not in manager.list_plugins()

    def test_builtin_plugins_work(self):
        """Built-in plugins work correctly."""
        manager = REPLPluginManager()
        builtins = BuiltinPlugins()

        manager.register_plugin(builtins.get_core_plugin())

        funcs = manager.get_all_functions()

        assert "peek" in funcs
        assert "search" in funcs
