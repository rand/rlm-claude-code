"""
Unit tests for recursive_handler module.

Implements: Spec §4.2, §6.4 tests
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CostConfig, DepthConfig, ModelConfig, RLMConfig
from src.recursive_handler import RecursiveREPL
from src.router_integration import CompletionResult, ModelRouter
from src.types import (
    CostLimitError,
    Message,
    MessageRole,
    RecursionDepthError,
    RecursiveCallResult,
    SessionContext,
    ToolOutput,
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
def mock_router():
    """Create a mock model router."""
    router = MagicMock(spec=ModelRouter)
    router.get_model.return_value = "claude-sonnet-4"

    async def mock_complete(*args, **kwargs):
        return CompletionResult(
            content="Mock completion response",
            model=kwargs.get("model", "claude-sonnet-4"),
            tokens_used=100,
        )

    router.complete = AsyncMock(side_effect=mock_complete)
    return router


@pytest.fixture
def basic_config():
    """Create a basic config for testing."""
    return RLMConfig(
        depth=DepthConfig(default=2, max=3),
        cost_controls=CostConfig(
            max_recursive_calls_per_turn=10,
            max_tokens_per_recursive_call=8000,
            abort_on_cost_threshold=50000,
        ),
        models=ModelConfig(
            root_model="opus",
            recursive_depth_1="sonnet",
            recursive_depth_2="haiku",
        ),
    )


class TestRecursiveREPLInit:
    """Tests for RecursiveREPL initialization."""

    def test_creates_with_session_context(self, basic_context, mock_router, basic_config):
        """Can create REPL with SessionContext."""
        repl = RecursiveREPL(
            context=basic_context,
            depth=0,
            config=basic_config,
            router=mock_router,
        )

        assert repl.depth == 0
        assert repl.context is basic_context
        assert repl.repl is not None  # Should create RLMEnvironment

    def test_creates_with_string_context(self, mock_router, basic_config):
        """Can create REPL with string context."""
        repl = RecursiveREPL(
            context="Some string context",
            depth=1,
            config=basic_config,
            router=mock_router,
        )

        assert repl.depth == 1
        assert repl.context == "Some string context"
        assert repl.repl is None  # No RLMEnvironment for string context

    def test_uses_config_max_depth(self, basic_context, mock_router, basic_config):
        """Uses max depth from config when not specified."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        assert repl.max_depth == 3  # From basic_config

    def test_can_override_max_depth(self, basic_context, mock_router, basic_config):
        """Can override max depth from config."""
        repl = RecursiveREPL(
            context=basic_context,
            max_depth=5,
            config=basic_config,
            router=mock_router,
        )

        assert repl.max_depth == 5

    def test_initializes_cost_tracking(self, basic_context, mock_router, basic_config):
        """Initializes cost tracking to zero."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        assert repl._tokens_used == 0
        assert repl._recursive_calls == 0
        assert repl._results == []


class TestRecursiveREPLModelSelection:
    """Tests for model selection by depth."""

    def test_depth_0_selects_root_model(self, basic_context, basic_config):
        """Depth 0 selects root model (Opus)."""
        router = ModelRouter(basic_config)
        repl = RecursiveREPL(
            context=basic_context,
            depth=0,
            config=basic_config,
            router=router,
        )

        model = repl.get_model_for_depth()
        assert model == "opus"

    def test_depth_1_selects_recursive_model(self, basic_context, basic_config):
        """Depth 1 selects recursive model (Sonnet)."""
        router = ModelRouter(basic_config)
        repl = RecursiveREPL(
            context=basic_context,
            depth=1,
            config=basic_config,
            router=router,
        )

        model = repl.get_model_for_depth()
        assert model == "sonnet"

    def test_depth_2_selects_deep_model(self, basic_context, basic_config):
        """Depth 2+ selects deep recursive model (Haiku)."""
        router = ModelRouter(basic_config)
        repl = RecursiveREPL(
            context=basic_context,
            depth=2,
            config=basic_config,
            router=router,
        )

        model = repl.get_model_for_depth()
        assert model == "haiku"


class TestRecursiveREPLRecursiveQuery:
    """Tests for recursive_query method."""

    @pytest.mark.asyncio
    async def test_simple_recursive_call(self, basic_context, mock_router, basic_config):
        """Can make simple recursive call."""
        repl = RecursiveREPL(
            context=basic_context,
            depth=0,
            max_depth=2,
            config=basic_config,
            router=mock_router,
        )

        result = await repl.recursive_query(
            query="Analyze this",
            context="Test context",
            spawn_repl=False,
        )

        assert result == "Mock completion response"
        assert repl._recursive_calls == 1
        assert len(repl._results) == 1

    @pytest.mark.asyncio
    async def test_respects_max_depth(self, basic_context, mock_router, basic_config):
        """Raises error when max depth exceeded."""
        repl = RecursiveREPL(
            context=basic_context,
            depth=2,
            max_depth=2,
            config=basic_config,
            router=mock_router,
        )

        with pytest.raises(RecursionDepthError) as exc:
            await repl.recursive_query(
                query="Analyze this",
                context="Test context",
            )

        assert exc.value.depth == 3
        assert exc.value.max_depth == 2

    @pytest.mark.asyncio
    async def test_spawn_repl_creates_child(self, basic_context, mock_router, basic_config):
        """spawn_repl=True creates child REPL."""
        repl = RecursiveREPL(
            context=basic_context,
            depth=0,
            max_depth=3,
            config=basic_config,
            router=mock_router,
        )

        await repl.recursive_query(
            query="Analyze this",
            context=basic_context,
            spawn_repl=True,
        )

        assert len(repl.child_repls) == 1
        assert repl.child_repls[0].depth == 1

    @pytest.mark.asyncio
    async def test_cost_limit_enforced(self, basic_context, mock_router, basic_config):
        """Raises error when cost limit exceeded."""
        # Set very low cost limit
        basic_config.cost_controls.abort_on_cost_threshold = 50

        repl = RecursiveREPL(
            context=basic_context,
            depth=0,
            config=basic_config,
            router=mock_router,
        )

        # First call should work
        await repl.recursive_query(
            query="First call",
            context="Test",
        )

        # Second call should fail (100 tokens from first call > 50 limit)
        with pytest.raises(CostLimitError):
            await repl.recursive_query(
                query="Second call",
                context="Test",
            )

    @pytest.mark.asyncio
    async def test_call_limit_enforced(self, basic_context, mock_router, basic_config):
        """Raises error when call limit exceeded."""
        basic_config.cost_controls.max_recursive_calls_per_turn = 2
        basic_config.cost_controls.abort_on_cost_threshold = 999999

        repl = RecursiveREPL(
            context=basic_context,
            depth=0,
            config=basic_config,
            router=mock_router,
        )

        # First two calls should work
        await repl.recursive_query(query="Call 1", context="Test")
        await repl.recursive_query(query="Call 2", context="Test")

        # Third call should fail
        with pytest.raises(CostLimitError):
            await repl.recursive_query(query="Call 3", context="Test")


class TestRecursiveREPLContextSerialization:
    """Tests for context serialization."""

    def test_serialize_string_context(self, basic_context, mock_router, basic_config):
        """String context passes through unchanged."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        result = repl._serialize_context("Hello world")
        assert result == "Hello world"

    def test_serialize_session_context(self, basic_context, mock_router, basic_config):
        """SessionContext is summarized."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        result = repl._serialize_context(basic_context)
        assert "Conversation: 2 messages" in result
        assert "main.py" in result
        assert "Tool outputs: 1 results" in result

    def test_serialize_dict_context(self, basic_context, mock_router, basic_config):
        """Dict context is JSON serialized."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        result = repl._serialize_context({"key": "value", "num": 42})
        assert '"key": "value"' in result
        assert '"num": 42' in result

    def test_serialize_list_context(self, basic_context, mock_router, basic_config):
        """List context is joined by newlines."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        result = repl._serialize_context(["item1", "item2", "item3"])
        assert result == "item1\nitem2\nitem3"


class TestRecursiveREPLPromptBuilding:
    """Tests for prompt building."""

    def test_build_sub_prompt_includes_query(self, basic_context, mock_router, basic_config):
        """Sub-prompt includes the query."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        prompt = repl._build_sub_prompt("Find the bug", "code here")
        assert "Find the bug" in prompt

    def test_build_sub_prompt_includes_context(self, basic_context, mock_router, basic_config):
        """Sub-prompt includes context."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        prompt = repl._build_sub_prompt("Query", "my context data")
        assert "my context data" in prompt

    def test_build_sub_prompt_truncates_large_context(
        self, basic_context, mock_router, basic_config
    ):
        """Sub-prompt truncates very large context."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        large_context = "x" * 20000
        prompt = repl._build_sub_prompt("Query", large_context)

        assert "[truncated" in prompt
        assert "20000 chars total" in prompt

    def test_build_rlm_prompt_includes_instructions(self, basic_context, mock_router, basic_config):
        """RLM prompt includes REPL instructions."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        prompt = repl._build_rlm_prompt("Test query")

        assert "RLM" in prompt
        assert "depth=" in prompt
        assert "peek" in prompt
        assert "search" in prompt
        assert "recursive_query" in prompt
        assert "FINAL" in prompt


class TestRecursiveREPLCostTracking:
    """Tests for cost tracking and aggregation."""

    def test_total_tokens_includes_children(self, basic_context, mock_router, basic_config):
        """Total tokens includes all children."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )
        repl._tokens_used = 100

        child1 = RecursiveREPL(
            context="child1",
            depth=1,
            config=basic_config,
            router=mock_router,
            parent=repl,
        )
        child1._tokens_used = 50
        repl.child_repls.append(child1)

        child2 = RecursiveREPL(
            context="child2",
            depth=1,
            config=basic_config,
            router=mock_router,
            parent=repl,
        )
        child2._tokens_used = 75
        repl.child_repls.append(child2)

        assert repl.total_tokens_used == 225

    def test_total_calls_includes_children(self, basic_context, mock_router, basic_config):
        """Total calls includes all children."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )
        repl._recursive_calls = 2

        child = RecursiveREPL(
            context="child",
            depth=1,
            config=basic_config,
            router=mock_router,
            parent=repl,
        )
        child._recursive_calls = 3
        repl.child_repls.append(child)

        assert repl.total_recursive_calls == 5

    def test_get_cost_summary(self, basic_context, mock_router, basic_config):
        """Can get cost summary."""
        repl = RecursiveREPL(
            context=basic_context,
            depth=0,
            config=basic_config,
            router=mock_router,
        )
        repl._tokens_used = 100
        repl._recursive_calls = 2

        summary = repl.get_cost_summary()

        assert summary["total_tokens"] == 100
        assert summary["total_calls"] == 2
        assert summary["max_depth_reached"] == 0
        assert "model_usage" in summary


class TestRecursiveREPLResultAggregation:
    """Tests for result aggregation."""

    def test_get_aggregated_results(self, basic_context, mock_router, basic_config):
        """Can aggregate results from children."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )
        repl._results = [
            RecursiveCallResult(
                content="Result 1",
                depth=0,
                model_used="opus",
                tokens_used=100,
            )
        ]

        child = RecursiveREPL(
            context="child",
            depth=1,
            config=basic_config,
            router=mock_router,
            parent=repl,
        )
        child._results = [
            RecursiveCallResult(
                content="Result 2",
                depth=1,
                model_used="sonnet",
                tokens_used=50,
            )
        ]
        repl.child_repls.append(child)

        all_results = repl.get_aggregated_results()

        assert len(all_results) == 2
        assert all_results[0].content == "Result 1"
        assert all_results[1].content == "Result 2"


class TestRecursiveREPLToolInjection:
    """Tests for tool result injection."""

    def test_inject_tool_result(self, basic_context, mock_router, basic_config):
        """Can inject tool results into REPL."""
        repl = RecursiveREPL(
            context=basic_context,
            config=basic_config,
            router=mock_router,
        )

        repl.inject_tool_result({"status": "success", "data": [1, 2, 3]})

        assert repl.repl.locals["_tool_result"] == {"status": "success", "data": [1, 2, 3]}
        assert len(repl.repl.globals["tool_outputs"]) > 1  # Original + injected

    def test_inject_tool_result_no_repl(self, mock_router, basic_config):
        """Inject does nothing when no REPL (string context)."""
        repl = RecursiveREPL(
            context="String context",
            config=basic_config,
            router=mock_router,
        )

        # Should not raise
        repl.inject_tool_result({"status": "success"})
        assert repl.repl is None


class TestRecursiveREPLDepthTracking:
    """Tests for depth tracking."""

    def test_get_max_depth_reached_single(self, basic_context, mock_router, basic_config):
        """Max depth is own depth when no children."""
        repl = RecursiveREPL(
            context=basic_context,
            depth=1,
            config=basic_config,
            router=mock_router,
        )

        assert repl._get_max_depth_reached() == 1

    def test_get_max_depth_reached_with_children(self, basic_context, mock_router, basic_config):
        """Max depth includes children."""
        repl = RecursiveREPL(
            context=basic_context,
            depth=0,
            config=basic_config,
            router=mock_router,
        )

        child = RecursiveREPL(
            context="child",
            depth=1,
            config=basic_config,
            router=mock_router,
            parent=repl,
        )
        repl.child_repls.append(child)

        grandchild = RecursiveREPL(
            context="grandchild",
            depth=2,
            config=basic_config,
            router=mock_router,
            parent=child,
        )
        child.child_repls.append(grandchild)

        assert repl._get_max_depth_reached() == 2


class TestRecursiveCallResult:
    """Tests for RecursiveCallResult dataclass."""

    def test_create_basic_result(self):
        """Can create basic result."""
        result = RecursiveCallResult(
            content="Test response",
            depth=1,
            model_used="claude-sonnet-4",
        )

        assert result.content == "Test response"
        assert result.depth == 1
        assert result.model_used == "claude-sonnet-4"
        assert result.tokens_used == 0
        assert result.had_repl is False

    def test_create_with_children(self):
        """Can create result with child results."""
        child_result = RecursiveCallResult(
            content="Child response",
            depth=2,
            model_used="haiku",
        )

        result = RecursiveCallResult(
            content="Parent response",
            depth=1,
            model_used="sonnet",
            child_results=[child_result],
            had_repl=True,
        )

        assert len(result.child_results) == 1
        assert result.had_repl is True
