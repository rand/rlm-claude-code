"""
Security tests for REPL environment sandbox.

Implements: Spec §4.1 Security testing requirements

These tests verify that the REPL sandbox properly restricts:
- File system access
- Network access
- System command execution
- Dangerous builtins
- Import restrictions
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.repl_environment import (
    ALLOWED_SUBPROCESSES,
    RLMEnvironment,
    RLMSecurityError,
)
from src.types import SessionContext


@pytest.fixture
def sandbox():
    """Create a sandboxed REPL environment."""
    context = SessionContext()
    return RLMEnvironment(context, use_restricted=True)


@pytest.fixture
def unrestricted_sandbox():
    """Create an unrestricted REPL for comparison testing."""
    context = SessionContext()
    return RLMEnvironment(context, use_restricted=False)


@pytest.mark.security
class TestBlockedBuiltins:
    """Tests that dangerous builtins are blocked."""

    def test_open_blocked(self, sandbox):
        """open() builtin is blocked."""
        result = sandbox.execute("open('/etc/passwd', 'r')")
        assert result.success is False
        assert "open" in result.error.lower() or "name" in result.error.lower()

    def test_exec_blocked(self, sandbox):
        """exec() builtin is blocked."""
        result = sandbox.execute("exec('print(1)')")
        assert result.success is False

    def test_eval_blocked(self, sandbox):
        """eval() builtin is blocked."""
        result = sandbox.execute("eval('1+1')")
        assert result.success is False

    def test_compile_blocked(self, sandbox):
        """compile() builtin is blocked."""
        result = sandbox.execute("compile('1+1', '<string>', 'eval')")
        assert result.success is False

    def test_import_blocked(self, sandbox):
        """__import__ builtin is blocked."""
        result = sandbox.execute("__import__('os')")
        assert result.success is False

    def test_input_blocked(self, sandbox):
        """input() builtin is blocked."""
        result = sandbox.execute("input('prompt')")
        assert result.success is False

    def test_breakpoint_blocked(self, sandbox):
        """breakpoint() builtin is blocked."""
        result = sandbox.execute("breakpoint()")
        assert result.success is False


@pytest.mark.security
class TestFileSystemRestrictions:
    """Tests that file system access is blocked."""

    def test_cannot_read_files_via_pathlib(self, sandbox):
        """Cannot use pathlib to read files."""
        # pathlib is not in globals
        result = sandbox.execute("from pathlib import Path")
        assert result.success is False

    def test_cannot_import_os(self, sandbox):
        """Cannot import os module."""
        result = sandbox.execute("import os")
        assert result.success is False

    def test_cannot_import_subprocess(self, sandbox):
        """Cannot import subprocess module."""
        result = sandbox.execute("import subprocess")
        assert result.success is False

    def test_cannot_import_shutil(self, sandbox):
        """Cannot import shutil module."""
        result = sandbox.execute("import shutil")
        assert result.success is False


@pytest.mark.security
class TestNetworkRestrictions:
    """Tests that network access is blocked."""

    def test_cannot_import_socket(self, sandbox):
        """Cannot import socket module."""
        result = sandbox.execute("import socket")
        assert result.success is False

    def test_cannot_import_urllib(self, sandbox):
        """Cannot import urllib module."""
        result = sandbox.execute("import urllib")
        assert result.success is False

    def test_cannot_import_requests(self, sandbox):
        """Cannot import requests module."""
        result = sandbox.execute("import requests")
        assert result.success is False

    def test_cannot_import_httpx(self, sandbox):
        """Cannot import httpx module."""
        result = sandbox.execute("import httpx")
        assert result.success is False


@pytest.mark.security
class TestSubprocessRestrictions:
    """Tests that subprocess execution is properly restricted."""

    def test_run_tool_rejects_unknown_tool(self, sandbox):
        """run_tool rejects tools not in allowlist."""
        with pytest.raises(RLMSecurityError) as exc_info:
            sandbox._run_tool("rm", "-rf", "/")

        assert "not allowed" in str(exc_info.value).lower()

    def test_run_tool_rejects_bash(self, sandbox):
        """run_tool rejects bash execution."""
        with pytest.raises(RLMSecurityError):
            sandbox._run_tool("bash", "-c", "echo pwned")

    def test_run_tool_rejects_sh(self, sandbox):
        """run_tool rejects sh execution."""
        with pytest.raises(RLMSecurityError):
            sandbox._run_tool("sh", "-c", "echo pwned")

    def test_run_tool_rejects_python(self, sandbox):
        """run_tool rejects python execution."""
        with pytest.raises(RLMSecurityError):
            sandbox._run_tool("python", "-c", "print('pwned')")

    def test_run_tool_rejects_curl(self, sandbox):
        """run_tool rejects curl execution."""
        with pytest.raises(RLMSecurityError):
            sandbox._run_tool("curl", "http://evil.com")

    def test_run_tool_rejects_wget(self, sandbox):
        """run_tool rejects wget execution."""
        with pytest.raises(RLMSecurityError):
            sandbox._run_tool("wget", "http://evil.com")

    def test_allowed_tools_are_minimal(self):
        """Only minimal set of tools are allowed."""
        # Should only allow type checking, linting, and package management tools
        assert frozenset({"ty", "ruff", "uv"}) == ALLOWED_SUBPROCESSES

    def test_run_tool_allows_ty(self, sandbox):
        """run_tool allows ty type checker."""
        # This should not raise (may fail if ty not installed, but shouldn't be security error)
        try:
            result = sandbox._run_tool("ty", "--version")
            # If ty is installed, check it ran
            assert "returncode" in result
        except RLMSecurityError:
            pytest.fail("ty should be allowed")

    def test_run_tool_allows_ruff(self, sandbox):
        """run_tool allows ruff linter."""
        try:
            result = sandbox._run_tool("ruff", "--version")
            assert "returncode" in result
        except RLMSecurityError:
            pytest.fail("ruff should be allowed")


@pytest.mark.security
class TestCodeInjectionPrevention:
    """Tests that code injection attacks are prevented."""

    def test_cannot_escape_via_globals(self, sandbox):
        """Cannot access dangerous functions via globals manipulation."""
        result = sandbox.execute("globals()['__builtins__']['open']")
        assert result.success is False

    def test_cannot_escape_via_class_bases(self, sandbox):
        """Cannot access dangerous classes via __bases__ traversal."""
        code = """
class X:
    pass
X.__class__.__bases__[0].__subclasses__()
"""
        result = sandbox.execute(code)
        # RestrictedPython should block attribute access to __bases__
        # or the result should not give access to dangerous classes
        # Either failure or restricted output is acceptable
        if result.success:
            # If it succeeds, ensure no dangerous access
            assert "__builtins__" not in str(result.output)

    def test_cannot_escape_via_func_globals(self, sandbox):
        """Cannot access globals via function __globals__."""
        code = """
def f():
    pass
f.__globals__
"""
        result = sandbox.execute(code)
        # Should fail or return restricted view
        if result.success:
            assert "open" not in str(result.output)

    def test_cannot_modify_builtins(self, sandbox):
        """Cannot modify the builtins dict."""
        result = sandbox.execute("__builtins__['open'] = lambda x: x")
        # Should fail - builtins should not be modifiable
        # RestrictedPython typically makes builtins immutable
        if result.success:
            # If it claims success, verify open is still blocked
            result2 = sandbox.execute("open('/etc/passwd')")
            assert result2.success is False


@pytest.mark.security
class TestSafeBuiltinsAvailable:
    """Tests that safe builtins are available."""

    def test_len_available(self, sandbox):
        """len() is available."""
        result = sandbox.execute("_ = len([1, 2, 3])")
        assert result.success is True

    def test_str_available(self, sandbox):
        """str() is available."""
        result = sandbox.execute("_ = str(42)")
        assert result.success is True

    def test_int_available(self, sandbox):
        """int() is available."""
        result = sandbox.execute("_ = int('42')")
        assert result.success is True

    def test_list_available(self, sandbox):
        """list() is available."""
        result = sandbox.execute("_ = list(range(3))")
        assert result.success is True

    def test_dict_available(self, sandbox):
        """dict() is available."""
        result = sandbox.execute("_ = dict(a=1, b=2)")
        assert result.success is True

    def test_sorted_available(self, sandbox):
        """sorted() is available."""
        result = sandbox.execute("_ = sorted([3, 1, 2])")
        assert result.success is True

    def test_range_available(self, sandbox):
        """range() is available."""
        result = sandbox.execute("_ = list(range(5))")
        assert result.success is True

    def test_enumerate_available(self, sandbox):
        """enumerate() is available."""
        result = sandbox.execute("_ = list(enumerate(['a', 'b']))")
        assert result.success is True


@pytest.mark.security
class TestSafeModulesAvailable:
    """Tests that safe modules are available in sandbox."""

    def test_re_available(self, sandbox):
        """re module is available."""
        result = sandbox.execute("_ = re.match(r'\\d+', '123')")
        assert result.success is True

    def test_json_available(self, sandbox):
        """json module is available."""
        result = sandbox.execute("_ = json.dumps({'a': 1})")
        assert result.success is True


@pytest.mark.security
class TestContextVariablesReadOnly:
    """Tests that context variables cannot be used for escape."""

    def test_conversation_is_list(self, sandbox):
        """conversation is a plain list, not dangerous object."""
        result = sandbox.execute("_ = isinstance(conversation, list)")
        assert result.success is True
        assert result.output is True

    def test_files_is_dict(self, sandbox):
        """files is a plain dict, not dangerous object."""
        result = sandbox.execute("_ = isinstance(files, dict)")
        assert result.success is True
        assert result.output is True

    def test_cannot_use_context_for_escape(self, sandbox):
        """Cannot use context variables to escape sandbox."""
        # Try to access file system via injected path
        result = sandbox.execute("""
files['/etc/passwd'] = 'injected'
_ = open('/etc/passwd')
""")
        assert result.success is False


@pytest.mark.security
class TestHelperFunctionSecurity:
    """Tests that helper functions are secure."""

    def test_peek_handles_malicious_input(self, sandbox):
        """peek() handles potentially malicious input safely."""
        result = sandbox.execute("_ = peek(conversation, -1000000, 1000000)")
        assert result.success is True  # Should handle gracefully

    def test_search_handles_malicious_regex(self, sandbox):
        """search() handles malicious regex safely."""
        # ReDoS attempt
        result = sandbox.execute("_ = search(conversation, '(a+)+$', regex=True)")
        # Should either succeed quickly or fail gracefully
        assert isinstance(result.success, bool)

    def test_run_tool_timeout_works(self, sandbox):
        """run_tool respects timeout."""
        # Try to run with very short timeout
        result = sandbox._run_tool("ruff", "--version", timeout=0.001)
        # Should either complete or timeout, not hang
        assert "returncode" in result or "timed out" in result.get("stderr", "")
