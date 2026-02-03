"""
Unit tests for repl_environment module.

Implements: Spec §4 tests
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.repl_environment import (
    ALLOWED_SUBPROCESSES,
    RLMEnvironment,
    RLMSecurityError,
    create_micro_environment,
)
from src.types import Message, MessageRole, SessionContext, ToolOutput


@pytest.fixture
def basic_context():
    """Create a basic context for testing."""
    return SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ],
        files={"main.py": "print('hello')", "utils.py": "def helper(): pass"},
        tool_outputs=[ToolOutput(tool_name="bash", content="test output")],
        working_memory={"counter": 0},
    )


@pytest.fixture
def basic_env(basic_context):
    """Create a basic RLM environment for testing."""
    return RLMEnvironment(basic_context, use_restricted=False)


@pytest.fixture
def restricted_env(basic_context):
    """Create a restricted RLM environment for testing."""
    return RLMEnvironment(basic_context, use_restricted=True)


class TestRLMEnvironmentInit:
    """Tests for RLMEnvironment initialization."""

    def test_creates_globals(self, basic_env):
        """Environment has expected global variables."""
        assert "conversation" in basic_env.globals
        assert "files" in basic_env.globals
        assert "tool_outputs" in basic_env.globals
        assert "working_memory" in basic_env.globals

    def test_conversation_externalized(self, basic_env):
        """Conversation is properly externalized."""
        conv = basic_env.globals["conversation"]

        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == "Hello"

    def test_files_externalized(self, basic_env):
        """Files are properly externalized."""
        files = basic_env.globals["files"]

        assert "main.py" in files
        assert files["main.py"] == "print('hello')"

    def test_helper_functions_available(self, basic_env):
        """Helper functions are available in globals."""
        assert "peek" in basic_env.globals
        assert "search" in basic_env.globals
        assert "run_tool" in basic_env.globals
        assert callable(basic_env.globals["peek"])

    def test_restricted_python_guards_available(self, restricted_env):
        """RestrictedPython guards are set up."""
        assert "_getiter_" in restricted_env.globals
        assert "_iter_unpack_sequence_" in restricted_env.globals
        assert "_getattr_" in restricted_env.globals

    def test_working_memory_copied(self, basic_env):
        """Working memory is copied, not referenced."""
        basic_env.globals["working_memory"]["new_key"] = "new_value"
        # Original context should not be modified
        assert "new_key" not in basic_env.context.working_memory


class TestRLMEnvironmentExecute:
    """Tests for RLMEnvironment.execute method."""

    def test_execute_simple_expression(self, basic_env):
        """Can execute simple Python expressions."""
        result = basic_env.execute("x = 1 + 1")

        assert result.success is True
        assert result.error is None
        assert basic_env.locals["x"] == 2

    def test_execute_with_syntax_error(self, basic_env):
        """Syntax errors are caught and reported."""
        result = basic_env.execute("def broken(")

        assert result.success is False
        assert result.error is not None
        # Python 3.10+ uses more descriptive messages like "was never closed"
        assert "(" in result.error or "syntax" in result.error.lower()

    def test_execute_with_runtime_error(self, basic_env):
        """Runtime errors are caught and reported."""
        result = basic_env.execute("x = 1 / 0")

        assert result.success is False
        assert "division" in result.error.lower()

    def test_execute_can_access_context(self, basic_env):
        """Code can access context variables."""
        result = basic_env.execute("file_count = len(files)")

        assert result.success is True
        assert basic_env.locals["file_count"] == 2

    def test_execute_can_use_stdlib(self, basic_env):
        """Code can use standard library functions."""
        result = basic_env.execute("import_match = re.search(r'print', files['main.py'])")

        assert result.success is True
        assert basic_env.locals["import_match"] is not None

    def test_execute_tracks_time(self, basic_env):
        """Execution time is tracked."""
        result = basic_env.execute("x = sum(range(1000))")

        assert result.success is True
        assert result.execution_time_ms >= 0

    def test_execute_records_history(self, basic_env):
        """Execution is recorded in history."""
        basic_env.execute("x = 1")
        basic_env.execute("y = 2")

        history = basic_env.get_execution_history()
        assert len(history) == 2
        assert history[0]["code"] == "x = 1"
        assert history[1]["code"] == "y = 2"

    def test_print_capture(self, basic_env):
        """Print output is captured."""
        result = basic_env.execute("print('hello world')")

        assert result.success is True
        assert result.output == "hello world"

    def test_expression_value_capture_numeric(self, basic_env):
        """Numeric expressions return their value."""
        result = basic_env.execute("1 + 1")

        assert result.success is True
        assert result.output == 2

    def test_expression_value_capture_string(self, basic_env):
        """String expressions return their value."""
        result = basic_env.execute('"hello" + " world"')

        assert result.success is True
        assert result.output == "hello world"

    def test_expression_value_capture_list(self, basic_env):
        """List expressions return their value."""
        result = basic_env.execute("[1, 2, 3]")

        assert result.success is True
        assert result.output == [1, 2, 3]

    def test_expression_value_capture_variable(self, basic_env):
        """Variable access returns the value."""
        basic_env.execute("x = 42")
        result = basic_env.execute("x")

        assert result.success is True
        assert result.output == 42

    def test_statement_returns_none(self, basic_env):
        """Statements (non-expressions) return None."""
        result = basic_env.execute("x = 123")

        assert result.success is True
        assert result.output is None


class TestRLMEnvironmentRestricted:
    """Tests for RestrictedPython sandbox."""

    def test_restricted_basic_execution(self, restricted_env):
        """Basic code works in restricted mode."""
        result = restricted_env.execute("x = [1, 2, 3]\ny = sum(x)")

        assert result.success is True
        assert restricted_env.locals["y"] == 6

    def test_restricted_list_comprehension(self, restricted_env):
        """List comprehensions work in restricted mode."""
        result = restricted_env.execute("squares = [x**2 for x in range(5)]")

        assert result.success is True
        assert restricted_env.locals["squares"] == [0, 1, 4, 9, 16]

    def test_restricted_dict_operations(self, restricted_env):
        """Dict operations work in restricted mode."""
        result = restricted_env.execute("d = {'a': 1, 'b': 2}\nkeys = list(d.keys())")

        assert result.success is True
        assert set(restricted_env.locals["keys"]) == {"a", "b"}

    def test_restricted_can_access_context(self, restricted_env):
        """Can access context in restricted mode."""
        result = restricted_env.execute("file_list = list(files.keys())")

        assert result.success is True
        assert "main.py" in restricted_env.locals["file_list"]

    def test_restricted_expression_value_capture(self, restricted_env):
        """Expression values are captured in restricted mode."""
        result = restricted_env.execute("1 + 1")

        assert result.success is True
        assert result.output == 2

    def test_restricted_expression_string(self, restricted_env):
        """String expressions return their value in restricted mode."""
        result = restricted_env.execute('"hello"')

        assert result.success is True
        assert result.output == "hello"

    def test_restricted_expression_list(self, restricted_env):
        """List expressions return their value in restricted mode."""
        result = restricted_env.execute("[1, 2, 3]")

        assert result.success is True
        assert result.output == [1, 2, 3]

    def test_restricted_print_capture(self, restricted_env):
        """Print output is captured in restricted mode."""
        result = restricted_env.execute('print("test")')

        assert result.success is True
        assert result.output == "test"


class TestRLMEnvironmentPeek:
    """Tests for RLMEnvironment._peek helper."""

    def test_peek_string(self, basic_env):
        """Peek on string returns substring."""
        result = basic_env._peek("hello world", 0, 5)

        assert result == "hello"

    def test_peek_list(self, basic_env):
        """Peek on list returns slice."""
        result = basic_env._peek([1, 2, 3, 4, 5], 1, 3)

        assert result == [2, 3]

    def test_peek_dict(self, basic_env):
        """Peek on dict returns subset of items."""
        result = basic_env._peek({"a": 1, "b": 2, "c": 3}, 0, 2)

        assert len(result) == 2

    def test_peek_other_type(self, basic_env):
        """Peek on other types converts to string."""
        result = basic_env._peek(12345, 0, 3)

        assert result == "123"


class TestRLMEnvironmentSearch:
    """Tests for RLMEnvironment._search helper."""

    def test_search_string_found(self, basic_env):
        """Search finds pattern in string."""
        result = basic_env._search("hello world hello", "hello")

        assert len(result) == 1
        assert result[0]["type"] == "string"

    def test_search_string_not_found(self, basic_env):
        """Search returns empty when not found."""
        result = basic_env._search("hello world", "xyz")

        assert result == []

    def test_search_list(self, basic_env):
        """Search finds pattern in list items."""
        result = basic_env._search(["apple", "banana", "apricot"], "ap")

        assert len(result) == 2
        assert result[0]["index"] == 0
        assert result[1]["index"] == 2

    def test_search_dict(self, basic_env):
        """Search finds pattern in dict values."""
        result = basic_env._search({"a": "hello", "b": "world"}, "hello")

        assert len(result) == 1
        assert result[0]["key"] == "a"

    def test_search_regex(self, basic_env):
        """Search supports regex patterns."""
        result = basic_env._search("test123abc456", r"\d+", regex=True)

        assert len(result) == 1

    def test_search_case_insensitive(self, basic_env):
        """Search is case-insensitive by default."""
        result = basic_env._search("Hello World", "hello")

        assert len(result) == 1


class TestRLMEnvironmentExtendedTooling:
    """Tests for extended tooling (pydantic, hypothesis, cpmpy)."""

    def test_pydantic_available(self, basic_env):
        """Pydantic is available in environment."""
        assert "pydantic" in basic_env.globals
        assert "BaseModel" in basic_env.globals

    def test_hypothesis_available(self, basic_env):
        """Hypothesis is available in environment."""
        assert "hypothesis" in basic_env.globals
        assert "given" in basic_env.globals
        assert "st" in basic_env.globals

    def test_cpmpy_available(self, basic_env):
        """CPMpy is available in environment."""
        assert "cp" in basic_env.globals or "cpmpy" in basic_env.globals

    def test_can_use_pydantic(self, basic_env):
        """Can use pydantic in executed code."""
        # Use pre-loaded BaseModel from globals (import is blocked in sandbox)
        code = """
class Test(BaseModel):
    name: str
t = Test(name="test")
result = t.name
"""
        result = basic_env.execute(code)

        assert result.success is True
        assert basic_env.locals["result"] == "test"


class TestRLMEnvironmentSecurity:
    """Security tests for the sandbox."""

    def test_blocked_subprocess_raises_error(self, basic_env):
        """Blocked subprocess raises RLMSecurityError."""
        with pytest.raises(RLMSecurityError) as exc:
            basic_env._run_tool("rm", "-rf", "/")

        assert "not allowed" in str(exc.value).lower()
        assert "rm" in str(exc.value)

    def test_allowed_subprocess_list(self):
        """Only ty and ruff are allowed."""
        assert "ty" in ALLOWED_SUBPROCESSES
        assert "ruff" in ALLOWED_SUBPROCESSES
        assert "rm" not in ALLOWED_SUBPROCESSES
        assert "curl" not in ALLOWED_SUBPROCESSES

    def test_run_tool_timeout(self, basic_env):
        """Tool execution respects timeout."""
        # This should fail quickly since ty isn't available in test env
        result = basic_env._run_tool("ty", "--help", timeout=1.0)

        # Should return a result (success or not found), not hang
        assert "returncode" in result

    def test_run_tool_not_found(self, basic_env):
        """Tool not found is handled gracefully."""
        # Add a fake tool to allowed list for testing
        from src import repl_environment

        original = repl_environment.ALLOWED_SUBPROCESSES
        repl_environment.ALLOWED_SUBPROCESSES = frozenset({"nonexistent_tool"})

        try:
            result = basic_env._run_tool("nonexistent_tool")
            assert result["success"] is False
            assert "not found" in result["stderr"].lower()
        finally:
            repl_environment.ALLOWED_SUBPROCESSES = original


class TestRLMEnvironmentContextStats:
    """Tests for context statistics."""

    def test_get_context_stats(self, basic_env):
        """Can get context statistics."""
        stats = basic_env.get_context_stats()

        assert "conversation_count" in stats
        assert "file_count" in stats
        assert "total_tokens" in stats
        assert stats["conversation_count"] == 2
        assert stats["file_count"] == 2

    def test_inject_result(self, basic_env):
        """Can inject results into REPL namespace."""
        basic_env.inject_result("injected", 42)

        assert basic_env.locals["injected"] == 42

        # Can access in subsequent execution
        result = basic_env.execute("doubled = injected * 2")
        assert result.success is True
        assert basic_env.locals["doubled"] == 84


class TestDeferredOperations:
    """Tests for deferred async operation handling."""

    def test_recursive_query_returns_deferred_operation(self, basic_env):
        """recursive_query returns a DeferredOperation, not a coroutine."""
        from src.types import DeferredOperation

        op = basic_env._recursive_query("What is this?", "some context")

        assert isinstance(op, DeferredOperation)
        assert op.operation_type == "recursive_query"
        assert op.query == "What is this?"
        assert "some context" in op.context

    def test_summarize_returns_deferred_operation(self, basic_env):
        """summarize returns a DeferredOperation, not a coroutine."""
        from src.types import DeferredOperation

        op = basic_env._summarize("Some long text to summarize")

        assert isinstance(op, DeferredOperation)
        assert op.operation_type == "summarize"
        assert "Summarize" in op.query

    def test_llm_batch_returns_deferred_batch(self, basic_env):
        """llm_batch returns a DeferredBatch for parallel processing."""
        from src.types import DeferredBatch

        batch = basic_env._llm_batch(
            [
                ("Query 1", "context 1"),
                ("Query 2", "context 2"),
                ("Query 3", "context 3"),
            ]
        )

        assert isinstance(batch, DeferredBatch)
        assert len(batch.operations) == 3
        assert batch.operations[0].query == "Query 1"
        assert batch.operations[1].query == "Query 2"
        assert batch.operations[2].query == "Query 3"

    def test_has_pending_operations(self, basic_env):
        """has_pending_operations correctly tracks pending ops."""
        assert basic_env.has_pending_operations() is False

        basic_env._recursive_query("test", "ctx")

        assert basic_env.has_pending_operations() is True

    def test_get_pending_operations(self, basic_env):
        """get_pending_operations returns both individual ops and batches."""
        basic_env._recursive_query("q1", "c1")
        basic_env._summarize("text")
        basic_env._llm_batch([("q2", "c2"), ("q3", "c3")])

        ops, batches = basic_env.get_pending_operations()

        assert len(ops) == 2  # recursive_query + summarize
        assert len(batches) == 1
        assert len(batches[0].operations) == 2

    def test_resolve_operation(self, basic_env):
        """resolve_operation injects result into working_memory."""
        op = basic_env._recursive_query("test query", "context")

        basic_env.resolve_operation(op.operation_id, "The answer is 42")

        assert op.resolved is True
        assert op.result == "The answer is 42"
        assert basic_env.globals["working_memory"][op.operation_id] == "The answer is 42"

    def test_resolve_batch(self, basic_env):
        """resolve_batch resolves all operations in a batch."""
        batch = basic_env._llm_batch(
            [
                ("Query 1", "ctx1"),
                ("Query 2", "ctx2"),
            ]
        )

        basic_env.resolve_batch(batch.batch_id, ["Answer 1", "Answer 2"])

        assert batch.resolved is True
        assert batch.results == ["Answer 1", "Answer 2"]
        assert batch.operations[0].resolved is True
        assert batch.operations[1].resolved is True
        assert basic_env.globals["working_memory"][batch.batch_id] == ["Answer 1", "Answer 2"]

    def test_clear_pending_operations(self, basic_env):
        """clear_pending_operations removes all pending ops."""
        basic_env._recursive_query("q1", "c1")
        basic_env._llm_batch([("q2", "c2")])

        assert basic_env.has_pending_operations() is True

        basic_env.clear_pending_operations()

        assert basic_env.has_pending_operations() is False

    def test_deferred_operation_repr(self, basic_env):
        """DeferredOperation has identifiable repr for output detection."""
        op = basic_env._recursive_query("test", "ctx")

        repr_str = repr(op)

        assert "<<DEFERRED:" in repr_str
        assert op.operation_id in repr_str

    def test_operation_ids_are_unique(self, basic_env):
        """Each operation gets a unique ID."""
        op1 = basic_env._recursive_query("q1", "c1")
        op2 = basic_env._recursive_query("q2", "c2")
        op3 = basic_env._summarize("text")

        ids = {op1.operation_id, op2.operation_id, op3.operation_id}

        assert len(ids) == 3  # All unique

    def test_llm_batch_available_in_repl(self, basic_env):
        """llm_batch is accessible as a helper in REPL execution."""
        assert "llm_batch" in basic_env.globals
        assert callable(basic_env.globals["llm_batch"])

    def test_recursive_query_via_repl_execution(self, basic_env):
        """recursive_query can be called from REPL code."""
        result = basic_env.execute("op = recursive_query('What is 2+2?', 'math context')")

        assert result.success is True
        assert basic_env.has_pending_operations() is True

        ops, _ = basic_env.get_pending_operations()
        assert len(ops) == 1
        assert ops[0].query == "What is 2+2?"


# ============================================================================
# Micro Mode Tests (SPEC-14.03-14.04)
# ============================================================================


class TestMicroMode:
    """Tests for micro mode restricted REPL (SPEC-14.03-14.04)."""

    @pytest.fixture
    def micro_env(self, basic_context):
        """Create a micro mode REPL environment."""
        from src.repl_environment import create_micro_environment

        return create_micro_environment(basic_context, use_restricted=False)

    def test_micro_mode_has_peek(self, micro_env):
        """SPEC-14.03: Micro mode has peek function."""
        assert "peek" in micro_env.globals
        result = micro_env.execute("peek('hello world', 0, 5)")
        assert result.success is True
        assert result.output == "hello"

    def test_micro_mode_has_search(self, micro_env):
        """SPEC-14.03: Micro mode has search function."""
        assert "search" in micro_env.globals
        result = micro_env.execute("search(['apple', 'banana'], 'apple')")
        assert result.success is True
        assert len(result.output) == 1

    def test_micro_mode_has_summarize_local(self, micro_env):
        """SPEC-14.03: Micro mode has summarize_local function."""
        assert "summarize_local" in micro_env.globals
        result = micro_env.execute("summarize_local('hello world')")
        assert result.success is True
        assert "hello world" in result.output

    def test_micro_mode_no_llm(self, micro_env):
        """SPEC-14.04: Micro mode does NOT have llm function."""
        assert "llm" not in micro_env.globals
        result = micro_env.execute("llm('test')")
        assert result.success is False
        assert "not defined" in result.error

    def test_micro_mode_no_llm_batch(self, micro_env):
        """SPEC-14.04: Micro mode does NOT have llm_batch function."""
        assert "llm_batch" not in micro_env.globals

    def test_micro_mode_no_map_reduce(self, micro_env):
        """SPEC-14.04: Micro mode does NOT have map_reduce function."""
        assert "map_reduce" not in micro_env.globals

    def test_micro_mode_no_summarize(self, micro_env):
        """SPEC-14.04: Micro mode does NOT have LLM-based summarize."""
        assert "summarize" not in micro_env.globals

    def test_micro_mode_no_recursive_query(self, micro_env):
        """SPEC-14.04: Micro mode does NOT have recursive_query."""
        assert "recursive_query" not in micro_env.globals

    def test_micro_mode_no_run_tool(self, micro_env):
        """Micro mode does NOT have run_tool for subprocess execution."""
        assert "run_tool" not in micro_env.globals

    def test_micro_mode_no_extended_tooling(self, micro_env):
        """Micro mode does NOT have pydantic/hypothesis/cpmpy."""
        assert "pydantic" not in micro_env.globals
        assert "hypothesis" not in micro_env.globals
        assert "cpmpy" not in micro_env.globals

    def test_micro_mode_has_context_variables(self, micro_env):
        """Micro mode has standard context variables."""
        assert "conversation" in micro_env.globals
        assert "files" in micro_env.globals
        assert "tool_outputs" in micro_env.globals
        assert "working_memory" in micro_env.globals

    def test_micro_mode_has_stdlib(self, micro_env):
        """Micro mode has re and json modules."""
        assert "re" in micro_env.globals
        assert "json" in micro_env.globals


class TestSummarizeLocal:
    """Tests for summarize_local function (SPEC-14.03)."""

    @pytest.fixture
    def env(self, basic_context):
        """Create a REPL environment."""
        return RLMEnvironment(basic_context, use_restricted=False)

    def test_summarize_local_string_short(self, env):
        """Short strings returned as-is."""
        result = env._summarize_local("hello world")
        assert result == "hello world"

    def test_summarize_local_string_truncated(self, env):
        """Long strings are truncated with count."""
        long_str = "x" * 1000
        result = env._summarize_local(long_str, max_chars=100)
        assert len(result) < 200
        assert "1000 chars total" in result
        assert "..." in result

    def test_summarize_local_empty_list(self, env):
        """Empty list returns empty message."""
        result = env._summarize_local([])
        assert "empty list" in result

    def test_summarize_local_list(self, env):
        """Lists show count and preview."""
        result = env._summarize_local([1, 2, 3, 4, 5])
        assert "5" in result or "items" in result

    def test_summarize_local_list_many_items(self, env):
        """Large lists show count and truncated preview."""
        result = env._summarize_local(list(range(100)), max_chars=100)
        assert "100 items total" in result
        assert "..." in result

    def test_summarize_local_empty_dict(self, env):
        """Empty dict returns empty message."""
        result = env._summarize_local({})
        assert "empty dict" in result

    def test_summarize_local_dict(self, env):
        """Dicts show count and preview."""
        result = env._summarize_local({"a": 1, "b": 2})
        assert "'a'" in result
        assert "'b'" in result

    def test_summarize_local_dict_many_keys(self, env):
        """Large dicts show count and truncated preview."""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = env._summarize_local(large_dict, max_chars=100)
        assert "100 keys total" in result
        assert "..." in result

    def test_summarize_local_other_type(self, env):
        """Other types show type name and truncated string."""
        result = env._summarize_local(12345)
        assert "(int)" in result
        assert "12345" in result


class TestAccessLevels:
    """Tests for REPL access levels."""

    @pytest.fixture
    def standard_env(self, basic_context):
        """Create a standard access level environment."""
        return RLMEnvironment(basic_context, use_restricted=False, access_level="standard")

    @pytest.fixture
    def full_env(self, basic_context):
        """Create a full access level environment."""
        return RLMEnvironment(basic_context, use_restricted=False, access_level="full")

    def test_standard_has_all_repl_functions(self, standard_env):
        """Standard mode has all REPL functions."""
        assert "peek" in standard_env.globals
        assert "search" in standard_env.globals
        assert "summarize_local" in standard_env.globals
        assert "summarize" in standard_env.globals
        assert "llm" in standard_env.globals
        assert "llm_batch" in standard_env.globals
        assert "map_reduce" in standard_env.globals
        assert "run_tool" in standard_env.globals

    def test_full_has_all_functions(self, full_env):
        """Full mode has all functions including tool access."""
        assert "peek" in full_env.globals
        assert "search" in full_env.globals
        assert "llm" in full_env.globals
        assert "run_tool" in full_env.globals

    def test_default_is_standard(self, basic_context):
        """Default access level is standard."""
        env = RLMEnvironment(basic_context, use_restricted=False)
        assert env.access_level == "standard"
        assert "llm" in env.globals
        assert "run_tool" in env.globals


# ============================================================================
# Epistemic Verification REPL Functions (SPEC-16)
# ============================================================================


class TestVerifyClaimREPL:
    """Tests for verify_claim REPL function (SPEC-16.02)."""

    @pytest.fixture
    def env(self, basic_context):
        """Create a REPL environment."""
        return RLMEnvironment(basic_context, use_restricted=False)

    @pytest.fixture
    def micro_env(self, basic_context):
        """Create a micro mode environment."""
        return create_micro_environment(basic_context, use_restricted=False)

    def test_verify_claim_available_in_standard_mode(self, env):
        """SPEC-16.02: verify_claim is available in standard mode."""
        assert "verify_claim" in env.globals
        assert callable(env.globals["verify_claim"])

    def test_verify_claim_not_available_in_micro_mode(self, micro_env):
        """verify_claim is NOT available in micro mode."""
        assert "verify_claim" not in micro_env.globals

    def test_verify_claim_returns_deferred_operation(self, env):
        """verify_claim returns a DeferredOperation."""
        from src.types import DeferredOperation

        op = env._verify_claim(
            claim="The function returns 42",
            evidence="def func(): return 42",
        )

        assert isinstance(op, DeferredOperation)
        assert op.operation_type == "verify_claim"
        assert "The function returns 42" in op.query

    def test_verify_claim_with_string_evidence(self, env):
        """verify_claim accepts string evidence."""
        op = env._verify_claim(
            claim="X is Y",
            evidence="X equals Y in all cases",
        )

        assert "X equals Y" in op.context

    def test_verify_claim_with_dict_evidence(self, env):
        """verify_claim accepts dict evidence and formats it."""
        op = env._verify_claim(
            claim="The module has tests",
            evidence={
                "test_file.py": "def test_something(): pass",
                "main.py": "def main(): pass",
            },
        )

        assert "[test_file.py]" in op.context
        assert "[main.py]" in op.context
        assert "def test_something" in op.context

    def test_verify_claim_stores_threshold_in_metadata(self, env):
        """verify_claim stores threshold in operation metadata."""
        op = env._verify_claim(
            claim="Test claim",
            evidence="Test evidence",
            threshold=0.8,
        )

        assert op.metadata["threshold"] == 0.8

    def test_verify_claim_default_threshold(self, env):
        """verify_claim uses default threshold of 0.7."""
        op = env._verify_claim(
            claim="Test claim",
            evidence="Test evidence",
        )

        assert op.metadata["threshold"] == 0.7

    def test_verify_claim_creates_pending_operation(self, env):
        """verify_claim adds operation to pending operations."""
        assert env.has_pending_operations() is False

        env._verify_claim("claim", "evidence")

        assert env.has_pending_operations() is True
        ops, _ = env.get_pending_operations()
        assert len(ops) == 1
        assert ops[0].operation_type == "verify_claim"

    def test_verify_claim_via_repl_execution(self, env):
        """verify_claim can be called from REPL code."""
        result = env.execute("op = verify_claim('The function returns 42', 'def f(): return 42')")

        assert result.success is True
        assert env.has_pending_operations() is True

    def test_verify_claim_unique_operation_ids(self, env):
        """Each verify_claim gets a unique operation ID."""
        op1 = env._verify_claim("claim 1", "evidence 1")
        op2 = env._verify_claim("claim 2", "evidence 2")

        assert op1.operation_id != op2.operation_id
        assert "verify_" in op1.operation_id
        assert "verify_" in op2.operation_id


class TestEvidenceDependenceREPL:
    """Tests for evidence_dependence REPL function (SPEC-16.04)."""

    @pytest.fixture
    def env(self, basic_context):
        """Create a REPL environment."""
        return RLMEnvironment(basic_context, use_restricted=False)

    @pytest.fixture
    def micro_env(self, basic_context):
        """Create a micro mode environment."""
        return create_micro_environment(basic_context, use_restricted=False)

    def test_evidence_dependence_available_in_standard_mode(self, env):
        """SPEC-16.04: evidence_dependence is available in standard mode."""
        assert "evidence_dependence" in env.globals
        assert callable(env.globals["evidence_dependence"])

    def test_evidence_dependence_not_available_in_micro_mode(self, micro_env):
        """evidence_dependence is NOT available in micro mode."""
        assert "evidence_dependence" not in micro_env.globals

    def test_evidence_dependence_returns_deferred_operation(self, env):
        """evidence_dependence returns a DeferredOperation."""
        from src.types import DeferredOperation

        op = env._evidence_dependence(
            question="What color is the widget?",
            answer="The widget is blue.",
            evidence="According to the spec, widgets are blue.",
        )

        assert isinstance(op, DeferredOperation)
        assert op.operation_type == "evidence_dependence"

    def test_evidence_dependence_stores_components_in_metadata(self, env):
        """evidence_dependence stores question, answer, evidence in metadata."""
        op = env._evidence_dependence(
            question="What is X?",
            answer="X is Y",
            evidence="The document says X equals Y",
        )

        assert op.metadata["question"] == "What is X?"
        assert op.metadata["answer"] == "X is Y"
        assert op.metadata["evidence"] == "The document says X equals Y"

    def test_evidence_dependence_creates_pending_operation(self, env):
        """evidence_dependence adds operation to pending operations."""
        assert env.has_pending_operations() is False

        env._evidence_dependence("Q", "A", "E")

        assert env.has_pending_operations() is True
        ops, _ = env.get_pending_operations()
        assert len(ops) == 1
        assert ops[0].operation_type == "evidence_dependence"

    def test_evidence_dependence_via_repl_execution(self, env):
        """evidence_dependence can be called from REPL code."""
        result = env.execute("op = evidence_dependence('What is X?', 'X is Y', 'Evidence about X')")

        assert result.success is True
        assert env.has_pending_operations() is True

    def test_evidence_dependence_unique_operation_ids(self, env):
        """Each evidence_dependence gets a unique operation ID."""
        op1 = env._evidence_dependence("Q1", "A1", "E1")
        op2 = env._evidence_dependence("Q2", "A2", "E2")

        assert op1.operation_id != op2.operation_id
        assert "dep_" in op1.operation_id
        assert "dep_" in op2.operation_id

    def test_evidence_dependence_context_format(self, env):
        """evidence_dependence formats context with all components."""
        op = env._evidence_dependence(
            question="Test question?",
            answer="Test answer",
            evidence="Test evidence",
        )

        assert "Question: Test question?" in op.context
        assert "Answer: Test answer" in op.context
        assert "Evidence: Test evidence" in op.context


class TestAuditReasoningREPL:
    """Tests for audit_reasoning REPL function (SPEC-16.03)."""

    @pytest.fixture
    def env(self, basic_context):
        """Create a REPL environment."""
        return RLMEnvironment(basic_context, use_restricted=False)

    @pytest.fixture
    def micro_env(self, basic_context):
        """Create a micro mode environment."""
        return create_micro_environment(basic_context, use_restricted=False)

    def test_audit_reasoning_available_in_standard_mode(self, env):
        """SPEC-16.03: audit_reasoning is available in standard mode."""
        assert "audit_reasoning" in env.globals
        assert callable(env.globals["audit_reasoning"])

    def test_audit_reasoning_not_available_in_micro_mode(self, micro_env):
        """audit_reasoning is NOT available in micro mode."""
        assert "audit_reasoning" not in micro_env.globals

    def test_audit_reasoning_returns_deferred_operation(self, env):
        """audit_reasoning returns a DeferredOperation."""
        from src.types import DeferredOperation

        steps = [
            {"claim": "The function returns 42", "cites": ["src1"]},
        ]
        sources = {"src1": "def func(): return 42"}

        op = env._audit_reasoning(steps, sources)

        assert isinstance(op, DeferredOperation)
        assert op.operation_type == "audit_reasoning"

    def test_audit_reasoning_with_multiple_steps(self, env):
        """audit_reasoning handles multiple reasoning steps."""
        steps = [
            {"claim": "Step 1 claim", "cites": ["src1"]},
            {"claim": "Step 2 claim", "cites": ["src2"]},
            {"claim": "Step 3 claim", "cites": ["src1", "src2"]},
        ]
        sources = {
            "src1": "Source 1 content",
            "src2": "Source 2 content",
        }

        op = env._audit_reasoning(steps, sources)

        assert op.metadata["step_count"] == 3
        assert len(op.metadata["steps"]) == 3

    def test_audit_reasoning_stores_steps_in_metadata(self, env):
        """audit_reasoning stores validated steps in metadata."""
        steps = [
            {"claim": "Test claim", "cites": ["s1", "s2"]},
        ]
        sources = {"s1": "Source 1", "s2": "Source 2"}

        op = env._audit_reasoning(steps, sources)

        assert op.metadata["steps"][0]["claim"] == "Test claim"
        assert op.metadata["steps"][0]["cites"] == ["s1", "s2"]

    def test_audit_reasoning_stores_sources_in_metadata(self, env):
        """audit_reasoning stores sources in metadata."""
        steps = [{"claim": "claim", "cites": ["src"]}]
        sources = {"src": "The source content"}

        op = env._audit_reasoning(steps, sources)

        assert op.metadata["sources"] == sources

    def test_audit_reasoning_formats_sources_in_context(self, env):
        """audit_reasoning formats all sources in context string."""
        steps = [{"claim": "claim", "cites": ["s1"]}]
        sources = {
            "s1": "Content of source 1",
            "s2": "Content of source 2",
        }

        op = env._audit_reasoning(steps, sources)

        assert "[s1]" in op.context
        assert "[s2]" in op.context
        assert "Content of source 1" in op.context
        assert "Content of source 2" in op.context

    def test_audit_reasoning_creates_pending_operation(self, env):
        """audit_reasoning adds operation to pending operations."""
        assert env.has_pending_operations() is False

        env._audit_reasoning(
            [{"claim": "c", "cites": ["s"]}],
            {"s": "source"},
        )

        assert env.has_pending_operations() is True
        ops, _ = env.get_pending_operations()
        assert len(ops) == 1
        assert ops[0].operation_type == "audit_reasoning"

    def test_audit_reasoning_via_repl_execution(self, env):
        """audit_reasoning can be called from REPL code."""
        result = env.execute("""
steps = [{"claim": "Test claim", "cites": ["src"]}]
sources = {"src": "Test source"}
op = audit_reasoning(steps, sources)
""")

        assert result.success is True
        assert env.has_pending_operations() is True

    def test_audit_reasoning_unique_operation_ids(self, env):
        """Each audit_reasoning gets a unique operation ID."""
        op1 = env._audit_reasoning([{"claim": "c1", "cites": []}], {})
        op2 = env._audit_reasoning([{"claim": "c2", "cites": []}], {})

        assert op1.operation_id != op2.operation_id
        assert "audit_" in op1.operation_id
        assert "audit_" in op2.operation_id

    def test_audit_reasoning_validates_step_structure(self, env):
        """audit_reasoning validates that steps have required fields."""
        # Missing 'claim' key should raise ValueError
        with pytest.raises(ValueError, match="missing required 'claim' key"):
            env._audit_reasoning([{"cites": ["s"]}], {"s": "source"})

    def test_audit_reasoning_validates_step_type(self, env):
        """audit_reasoning validates that steps are dicts."""
        with pytest.raises(TypeError, match="must be a dict"):
            env._audit_reasoning(["not a dict"], {})

    def test_audit_reasoning_handles_empty_cites(self, env):
        """audit_reasoning handles steps with no citations."""
        steps = [{"claim": "An uncited claim"}]
        sources = {}

        op = env._audit_reasoning(steps, sources)

        assert op.metadata["steps"][0]["cites"] == []

    def test_audit_reasoning_normalizes_single_cite_to_list(self, env):
        """audit_reasoning converts single cite value to list."""
        steps = [{"claim": "claim", "cites": "single_source"}]
        sources = {"single_source": "content"}

        op = env._audit_reasoning(steps, sources)

        # Should be normalized to a list
        assert op.metadata["steps"][0]["cites"] == ["single_source"]

    def test_audit_reasoning_query_includes_step_count(self, env):
        """audit_reasoning query mentions the number of steps."""
        steps = [
            {"claim": "claim1", "cites": []},
            {"claim": "claim2", "cites": []},
        ]

        op = env._audit_reasoning(steps, {})

        assert "2 reasoning steps" in op.query

    def test_audit_reasoning_empty_steps(self, env):
        """audit_reasoning handles empty steps list."""
        op = env._audit_reasoning([], {})

        assert op.metadata["step_count"] == 0
        assert op.metadata["steps"] == []
        assert "0 reasoning steps" in op.query


class TestDetectHallucinationsREPL:
    """Tests for detect_hallucinations REPL function (SPEC-16.05)."""

    @pytest.fixture
    def env(self, basic_context):
        """Create a REPL environment."""
        return RLMEnvironment(basic_context, use_restricted=False)

    @pytest.fixture
    def micro_env(self, basic_context):
        """Create a micro mode environment."""
        return create_micro_environment(basic_context, use_restricted=False)

    def test_detect_hallucinations_available_in_standard_mode(self, env):
        """SPEC-16.05: detect_hallucinations is available in standard mode."""
        assert "detect_hallucinations" in env.globals
        assert callable(env.globals["detect_hallucinations"])

    def test_detect_hallucinations_not_available_in_micro_mode(self, micro_env):
        """detect_hallucinations is NOT available in micro mode."""
        assert "detect_hallucinations" not in micro_env.globals

    def test_detect_hallucinations_returns_deferred_operation(self, env):
        """detect_hallucinations returns a DeferredOperation."""
        from src.types import DeferredOperation

        op = env._detect_hallucinations(
            response="The function returns 42.",
            context="def func(): return 42",
        )

        assert isinstance(op, DeferredOperation)
        assert op.operation_type == "detect_hallucinations"

    def test_detect_hallucinations_with_string_context(self, env):
        """detect_hallucinations accepts string context."""
        op = env._detect_hallucinations(
            response="X is Y.",
            context="X equals Y in all cases.",
        )

        assert "X equals Y" in op.context
        assert op.metadata["context_sources"] == {"main": "X equals Y in all cases."}

    def test_detect_hallucinations_with_dict_context(self, env):
        """detect_hallucinations accepts dict context and formats it."""
        op = env._detect_hallucinations(
            response="The module has functions.",
            context={
                "main.py": "def main(): pass",
                "utils.py": "def helper(): pass",
            },
        )

        assert "[main.py]" in op.context
        assert "[utils.py]" in op.context
        assert "def main" in op.context
        assert op.metadata["context_sources"]["main.py"] == "def main(): pass"

    def test_detect_hallucinations_stores_response_in_metadata(self, env):
        """detect_hallucinations stores the response in metadata."""
        response = "This is the response to check."
        op = env._detect_hallucinations(
            response=response,
            context="Some context",
        )

        assert op.metadata["response"] == response

    def test_detect_hallucinations_stores_thresholds_in_metadata(self, env):
        """detect_hallucinations stores thresholds in metadata."""
        op = env._detect_hallucinations(
            response="Response",
            context="Context",
            support_threshold=0.8,
            dependence_threshold=0.4,
        )

        assert op.metadata["support_threshold"] == 0.8
        assert op.metadata["dependence_threshold"] == 0.4

    def test_detect_hallucinations_default_thresholds(self, env):
        """detect_hallucinations uses default thresholds."""
        op = env._detect_hallucinations(
            response="Response",
            context="Context",
        )

        assert op.metadata["support_threshold"] == 0.7
        assert op.metadata["dependence_threshold"] == 0.3

    def test_detect_hallucinations_validates_support_threshold(self, env):
        """detect_hallucinations validates support_threshold range."""
        with pytest.raises(ValueError, match="support_threshold must be between"):
            env._detect_hallucinations("R", "C", support_threshold=1.5)

        with pytest.raises(ValueError, match="support_threshold must be between"):
            env._detect_hallucinations("R", "C", support_threshold=-0.1)

    def test_detect_hallucinations_validates_dependence_threshold(self, env):
        """detect_hallucinations validates dependence_threshold range."""
        with pytest.raises(ValueError, match="dependence_threshold must be between"):
            env._detect_hallucinations("R", "C", dependence_threshold=2.0)

        with pytest.raises(ValueError, match="dependence_threshold must be between"):
            env._detect_hallucinations("R", "C", dependence_threshold=-0.5)

    def test_detect_hallucinations_creates_pending_operation(self, env):
        """detect_hallucinations adds operation to pending operations."""
        assert env.has_pending_operations() is False

        env._detect_hallucinations("Response", "Context")

        assert env.has_pending_operations() is True
        ops, _ = env.get_pending_operations()
        assert len(ops) == 1
        assert ops[0].operation_type == "detect_hallucinations"

    def test_detect_hallucinations_via_repl_execution(self, env):
        """detect_hallucinations can be called from REPL code."""
        result = env.execute("""
report = detect_hallucinations(
    response="The function returns 42.",
    context="def func(): return 42"
)
""")

        assert result.success is True
        assert env.has_pending_operations() is True

    def test_detect_hallucinations_unique_operation_ids(self, env):
        """Each detect_hallucinations gets a unique operation ID."""
        op1 = env._detect_hallucinations("Response 1", "Context 1")
        op2 = env._detect_hallucinations("Response 2", "Context 2")

        assert op1.operation_id != op2.operation_id
        assert "halluc_" in op1.operation_id
        assert "halluc_" in op2.operation_id

    def test_detect_hallucinations_query_describes_task(self, env):
        """detect_hallucinations query describes the detection task."""
        op = env._detect_hallucinations("Response", "Context")

        assert "hallucinations" in op.query.lower()
        assert "claims" in op.query.lower()

    def test_detect_hallucinations_boundary_thresholds(self, env):
        """detect_hallucinations accepts boundary threshold values."""
        # Valid boundary values
        op1 = env._detect_hallucinations("R", "C", support_threshold=0.0)
        assert op1.metadata["support_threshold"] == 0.0

        op2 = env._detect_hallucinations("R", "C", support_threshold=1.0)
        assert op2.metadata["support_threshold"] == 1.0

        op3 = env._detect_hallucinations("R", "C", dependence_threshold=0.0)
        assert op3.metadata["dependence_threshold"] == 0.0

        op4 = env._detect_hallucinations("R", "C", dependence_threshold=1.0)
        assert op4.metadata["dependence_threshold"] == 1.0


# ============================================================================
# RestrictedPython Integration Tests (Critical Path Validation)
# ============================================================================


class TestRestrictedPythonIntegration:
    """
    Integration tests that validate critical operations work WITH RestrictedPython.

    These tests specifically exercise the _getitem_ and _write_ guards that were
    added to fix production issues. All tests MUST use use_restricted=True.

    See issue: REPL tests bypass RestrictedPython with use_restricted=False
    """

    @pytest.fixture
    def context(self):
        """Create a context for testing."""
        return SessionContext(
            messages=[
                Message(role=MessageRole.USER, content="Test message"),
                Message(role=MessageRole.ASSISTANT, content="Test response"),
            ],
            files={
                "main.py": "def main(): return 42",
                "utils.py": "def helper(): pass",
                "config.json": '{"key": "value"}',
            },
            tool_outputs=[ToolOutput(tool_name="Read", content="file content")],
            working_memory={"existing_key": "existing_value"},
        )

    @pytest.fixture
    def restricted_env(self, context):
        """Create a RestrictedPython-enabled environment."""
        return RLMEnvironment(context, use_restricted=True)

    # -------------------------------------------------------------------------
    # Dictionary Subscript Access (_getitem_ guard)
    # -------------------------------------------------------------------------

    def test_files_dict_subscript_access(self, restricted_env):
        """CRITICAL: Dictionary subscript access on files dict works."""
        result = restricted_env.execute("content = files['main.py']")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["content"] == "def main(): return 42"

    def test_files_dict_get_method(self, restricted_env):
        """Dictionary .get() method works in restricted mode."""
        result = restricted_env.execute("content = files.get('nonexistent', 'default')")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["content"] == "default"

    def test_files_dict_keys_iteration(self, restricted_env):
        """Can iterate over dictionary keys in restricted mode."""
        result = restricted_env.execute("file_list = list(files.keys())")

        assert result.success is True, f"Failed: {result.error}"
        assert "main.py" in restricted_env.locals["file_list"]

    def test_files_dict_values_iteration(self, restricted_env):
        """Can iterate over dictionary values in restricted mode."""
        result = restricted_env.execute("contents = list(files.values())")

        assert result.success is True, f"Failed: {result.error}"
        assert len(restricted_env.locals["contents"]) == 3

    def test_files_dict_items_iteration(self, restricted_env):
        """Can iterate over dictionary items in restricted mode."""
        result = restricted_env.execute("items = list(files.items())")

        assert result.success is True, f"Failed: {result.error}"
        assert len(restricted_env.locals["items"]) == 3

    def test_working_memory_subscript_read(self, restricted_env):
        """Can read from working_memory via subscript."""
        result = restricted_env.execute("val = working_memory['existing_key']")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["val"] == "existing_value"

    def test_list_subscript_access(self, restricted_env):
        """List subscript access works in restricted mode."""
        result = restricted_env.execute("first = conversation[0]")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["first"]["role"] == "user"

    def test_string_subscript_access(self, restricted_env):
        """String subscript access works in restricted mode."""
        result = restricted_env.execute("char = 'hello'[0]")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["char"] == "h"

    def test_nested_subscript_access(self, restricted_env):
        """Nested subscript access works in restricted mode."""
        result = restricted_env.execute("role = conversation[0]['role']")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["role"] == "user"

    # -------------------------------------------------------------------------
    # Dictionary Write Access (_write_ guard)
    # -------------------------------------------------------------------------

    def test_working_memory_subscript_write(self, restricted_env):
        """CRITICAL: Can write to working_memory via subscript."""
        result = restricted_env.execute("working_memory['new_key'] = 'new_value'")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.globals["working_memory"]["new_key"] == "new_value"

    def test_working_memory_multiple_writes(self, restricted_env):
        """Can perform multiple writes to working_memory."""
        result = restricted_env.execute("""
working_memory['key1'] = 'value1'
working_memory['key2'] = 'value2'
working_memory['key3'] = [1, 2, 3]
""")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.globals["working_memory"]["key1"] == "value1"
        assert restricted_env.globals["working_memory"]["key2"] == "value2"
        assert restricted_env.globals["working_memory"]["key3"] == [1, 2, 3]

    def test_new_dict_creation_and_write(self, restricted_env):
        """Can create new dict and write to it."""
        result = restricted_env.execute("""
d = {}
d['key'] = 'value'
result = d['key']
""")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["result"] == "value"

    def test_list_item_assignment(self, restricted_env):
        """Can assign to list items in restricted mode."""
        result = restricted_env.execute("""
lst = [1, 2, 3]
lst[0] = 99
result = lst[0]
""")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["result"] == 99

    # -------------------------------------------------------------------------
    # Helper Functions in Restricted Mode
    # -------------------------------------------------------------------------

    def test_peek_with_restricted_python(self, restricted_env):
        """peek() helper works in restricted mode."""
        result = restricted_env.execute("segment = peek(files['main.py'], 0, 10)")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["segment"] == "def main()"

    def test_search_with_restricted_python(self, restricted_env):
        """search() helper works in restricted mode."""
        result = restricted_env.execute("matches = search(files, 'def', regex=False)")

        assert result.success is True, f"Failed: {result.error}"
        assert len(restricted_env.locals["matches"]) >= 2

    def test_search_on_dict_subscript_result(self, restricted_env):
        """search() on result of dict subscript access."""
        result = restricted_env.execute("matches = search(files['main.py'], 'return')")

        assert result.success is True, f"Failed: {result.error}"
        assert len(restricted_env.locals["matches"]) >= 1

    def test_summarize_local_with_restricted_python(self, restricted_env):
        """summarize_local() helper works in restricted mode."""
        result = restricted_env.execute("summary = summarize_local(files)")

        assert result.success is True, f"Failed: {result.error}"
        # summarize_local returns dict repr for small dicts, or "N keys" for large ones
        summary = restricted_env.locals["summary"]
        assert "main.py" in summary or "3 keys" in summary

    # -------------------------------------------------------------------------
    # Complex Operations in Restricted Mode
    # -------------------------------------------------------------------------

    def test_list_comprehension_with_dict_access(self, restricted_env):
        """List comprehension with dict subscript access."""
        result = restricted_env.execute("sizes = [len(files[f]) for f in files.keys()]")

        assert result.success is True, f"Failed: {result.error}"
        assert len(restricted_env.locals["sizes"]) == 3

    def test_dict_comprehension(self, restricted_env):
        """Dict comprehension works in restricted mode."""
        result = restricted_env.execute("lengths = {k: len(v) for k, v in files.items()}")

        assert result.success is True, f"Failed: {result.error}"
        assert "main.py" in restricted_env.locals["lengths"]

    def test_filter_and_map_on_dict(self, restricted_env):
        """filter/map operations on dict in restricted mode."""
        result = restricted_env.execute("""
py_files = [k for k in files.keys() if k.endswith('.py')]
""")

        assert result.success is True, f"Failed: {result.error}"
        assert "main.py" in restricted_env.locals["py_files"]
        assert "config.json" not in restricted_env.locals["py_files"]

    def test_enumerate_over_dict_keys(self, restricted_env):
        """enumerate() over dict keys works."""
        result = restricted_env.execute("""
indexed = [(i, k) for i, k in enumerate(files.keys())]
""")

        assert result.success is True, f"Failed: {result.error}"
        assert len(restricted_env.locals["indexed"]) == 3

    def test_combined_read_transform_write(self, restricted_env):
        """Combined read, transform, and write operation."""
        result = restricted_env.execute("""
content = files['main.py']
lines = content.split('\\n')
working_memory['line_count'] = len(lines)
""")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.globals["working_memory"]["line_count"] == 1

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_subscript_with_variable_key(self, restricted_env):
        """Subscript access with variable as key."""
        result = restricted_env.execute("""
key = 'main.py'
content = files[key]
""")

        assert result.success is True, f"Failed: {result.error}"
        assert "def main" in restricted_env.locals["content"]

    def test_subscript_with_expression_key(self, restricted_env):
        """Subscript access with expression as key."""
        result = restricted_env.execute("""
prefix = 'main'
content = files[prefix + '.py']
""")

        assert result.success is True, f"Failed: {result.error}"
        assert "def main" in restricted_env.locals["content"]

    def test_negative_list_index(self, restricted_env):
        """Negative list index works in restricted mode."""
        result = restricted_env.execute("last = conversation[-1]")

        assert result.success is True, f"Failed: {result.error}"
        assert restricted_env.locals["last"]["role"] == "assistant"

    def test_slice_access(self, restricted_env):
        """Slice access on lists works in restricted mode."""
        result = restricted_env.execute("all_msgs = conversation[:]")

        assert result.success is True, f"Failed: {result.error}"
        assert len(restricted_env.locals["all_msgs"]) == 2


@pytest.mark.parametrize("use_restricted", [True, False])
class TestREPLHelpersBothModes:
    """
    Parametrized tests that verify helper functions work in BOTH modes.

    This ensures we don't have regressions where something works in
    unrestricted mode but fails in restricted mode.
    """

    @pytest.fixture
    def context(self):
        """Create a context for testing."""
        return SessionContext(
            messages=[Message(role=MessageRole.USER, content="Test")],
            files={"test.py": "print('hello')"},
            tool_outputs=[],
            working_memory={},
        )

    @pytest.fixture
    def env(self, context, use_restricted):
        """Create environment with parametrized restriction mode."""
        return RLMEnvironment(context, use_restricted=use_restricted)

    def test_dict_subscript_both_modes(self, env, use_restricted):
        """Dict subscript works in both modes."""
        result = env.execute("content = files['test.py']")
        assert result.success is True, f"Failed in restricted={use_restricted}: {result.error}"

    def test_working_memory_write_both_modes(self, env, use_restricted):
        """Working memory write works in both modes."""
        result = env.execute("working_memory['key'] = 'value'")
        assert result.success is True, f"Failed in restricted={use_restricted}: {result.error}"

    def test_peek_both_modes(self, env, use_restricted):
        """peek() works in both modes."""
        result = env.execute("segment = peek(files['test.py'], 0, 5)")
        assert result.success is True, f"Failed in restricted={use_restricted}: {result.error}"

    def test_search_both_modes(self, env, use_restricted):
        """search() works in both modes."""
        result = env.execute("matches = search(files, 'print')")
        assert result.success is True, f"Failed in restricted={use_restricted}: {result.error}"

    def test_list_comprehension_both_modes(self, env, use_restricted):
        """List comprehension with dict access works in both modes."""
        result = env.execute("keys = [k for k in files.keys()]")
        assert result.success is True, f"Failed in restricted={use_restricted}: {result.error}"
