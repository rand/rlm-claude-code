"""
Unit tests for repl_environment module.

Implements: Spec §4 tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.types import Message, MessageRole, SessionContext, ToolOutput
from src.repl_environment import (
    RLMEnvironment,
    RLMSecurityError,
    ALLOWED_SUBPROCESSES,
)


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

        batch = basic_env._llm_batch([
            ("Query 1", "context 1"),
            ("Query 2", "context 2"),
            ("Query 3", "context 3"),
        ])

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
        batch = basic_env._llm_batch([
            ("Query 1", "ctx1"),
            ("Query 2", "ctx2"),
        ])

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
