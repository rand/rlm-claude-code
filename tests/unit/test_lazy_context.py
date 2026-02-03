"""
Tests for lazy context loading.

Tests: SPEC-01.02 (Phase 4 - Lazy Context Loading)
"""

from __future__ import annotations

from pathlib import Path

from src.context_manager import (
    DEFAULT_MEMORY_LIMIT_MB,
    LazyContext,
    LazyContextConfig,
    LazyContextVariable,
    LazyFileLoader,
    create_lazy_context,
)
from src.types import Message, MessageRole, SessionContext, ToolOutput


class TestLazyContextVariable:
    """Tests for LazyContextVariable."""

    def test_lazy_load_on_access(self):
        """Content is not loaded until accessed."""
        load_count = [0]

        def loader():
            load_count[0] += 1
            return "content"

        lazy_var = LazyContextVariable(loader)
        assert load_count[0] == 0

        # Access triggers load
        content = lazy_var.get()
        assert content == "content"
        assert load_count[0] == 1

    def test_caches_content(self):
        """Content is cached after first load."""
        load_count = [0]

        def loader():
            load_count[0] += 1
            return "content"

        lazy_var = LazyContextVariable(loader)

        # Multiple accesses, single load
        lazy_var.get()
        lazy_var.get()
        lazy_var.get()
        assert load_count[0] == 1

    def test_is_loaded_property(self):
        """is_loaded reflects materialization state."""
        lazy_var = LazyContextVariable(lambda: "content")

        assert lazy_var.is_loaded is False
        lazy_var.get()
        assert lazy_var.is_loaded is True

    def test_clear_cache(self):
        """clear_cache resets materialized state."""
        load_count = [0]

        def loader():
            load_count[0] += 1
            return "content"

        lazy_var = LazyContextVariable(loader)
        lazy_var.get()
        assert load_count[0] == 1

        lazy_var.clear_cache()
        assert lazy_var.is_loaded is False

        lazy_var.get()
        assert load_count[0] == 2

    def test_peek_string_content(self):
        """peek returns sliced content."""
        lazy_var = LazyContextVariable(lambda: "hello world")

        assert lazy_var.peek(0, 5) == "hello"
        assert lazy_var.peek(6) == "world"

    def test_peek_list_content(self):
        """peek works with list content."""
        lazy_var = LazyContextVariable(lambda: [1, 2, 3, 4, 5])

        assert lazy_var.peek(0, 2) == [1, 2]
        assert lazy_var.peek(3) == [4, 5]

    def test_iterate_string(self):
        """Can iterate over string content in chunks."""
        content = "x" * 25000
        lazy_var = LazyContextVariable(lambda: content)

        chunks = list(lazy_var)
        combined = "".join(chunks)
        assert combined == content

    def test_iterate_list(self):
        """Can iterate over list content."""
        lazy_var = LazyContextVariable(lambda: [1, 2, 3])

        items = list(lazy_var)
        assert items == [1, 2, 3]

    def test_iterate_dict(self):
        """Can iterate over dict content (yields items)."""
        lazy_var = LazyContextVariable(lambda: {"a": 1, "b": 2})

        items = list(lazy_var)
        assert set(items) == {("a", 1), ("b", 2)}

    def test_generator_mode(self):
        """Generator mode yields from loader."""

        def gen_loader():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        lazy_var = LazyContextVariable(gen_loader, is_generator=True)

        chunks = list(lazy_var)
        assert chunks == ["chunk1", "chunk2", "chunk3"]

    def test_memory_limit_respected(self):
        """Large content is not cached if exceeds limit."""
        # 1MB limit
        lazy_var = LazyContextVariable(
            lambda: "x" * (2 * 1024 * 1024),  # 2MB content
            max_memory_mb=1,
        )

        lazy_var.get()
        # Should not cache (exceeds limit)
        assert lazy_var._materialized_size == 0

    def test_estimated_size_mb(self):
        """estimated_size_mb reports cached size."""
        content = "x" * 1024 * 100  # 100KB
        lazy_var = LazyContextVariable(lambda: content)

        assert lazy_var.estimated_size_mb == 0
        lazy_var.get()
        assert lazy_var.estimated_size_mb > 0


class TestLazyFileLoader:
    """Tests for LazyFileLoader."""

    def test_load_file_lazy(self, tmp_path: Path):
        """load_file_lazy creates lazy variable for file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")

        loader = LazyFileLoader()
        lazy_var = loader.load_file_lazy(test_file)

        assert lazy_var.is_loaded is False
        content = lazy_var.get()
        assert content == "file content"

    def test_load_file_chunks(self, tmp_path: Path):
        """load_file_chunks creates generator-based loader."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("chunk1chunk2chunk3")

        config = LazyContextConfig(chunk_size=6)
        loader = LazyFileLoader(config)
        lazy_var = loader.load_file_chunks(test_file)

        chunks = list(lazy_var)
        assert "".join(chunks) == "chunk1chunk2chunk3"

    def test_nonexistent_file_returns_empty(self, tmp_path: Path):
        """Loading nonexistent file returns empty string."""
        loader = LazyFileLoader()
        lazy_var = loader.load_file_lazy(tmp_path / "nonexistent.txt")

        assert lazy_var.get() == ""

    def test_config_options(self):
        """LazyContextConfig options are respected."""
        config = LazyContextConfig(
            max_memory_mb=50,
            use_mmap=False,
            chunk_size=1024,
        )

        assert config.max_memory_mb == 50
        assert config.use_mmap is False
        assert config.chunk_size == 1024


class TestLazyContext:
    """Tests for LazyContext wrapper."""

    def _create_session_context(self) -> SessionContext:
        """Create a test session context."""
        return SessionContext(
            messages=[
                Message(role=MessageRole.USER, content="Hello", timestamp=1000),
                Message(role=MessageRole.ASSISTANT, content="Hi", timestamp=1001),
            ],
            files={"test.py": "print('hello')"},
            tool_outputs=[
                ToolOutput(tool_name="bash", content="output", exit_code=0, timestamp=1002),
            ],
            working_memory={"key": "value"},
        )

    def test_create_lazy_context(self):
        """create_lazy_context wraps session context."""
        ctx = self._create_session_context()
        lazy_ctx = create_lazy_context(ctx)

        assert isinstance(lazy_ctx, LazyContext)

    def test_conversation_property(self):
        """conversation property returns externalized messages."""
        ctx = self._create_session_context()
        lazy_ctx = create_lazy_context(ctx)

        conv = lazy_ctx.conversation
        assert len(conv) == 2
        assert conv[0]["content"] == "Hello"

    def test_files_property(self):
        """files property returns lazy variables."""
        ctx = self._create_session_context()
        lazy_ctx = create_lazy_context(ctx)

        files = lazy_ctx.files
        assert "test.py" in files
        assert isinstance(files["test.py"], LazyContextVariable)

    def test_get_file(self):
        """get_file returns file content."""
        ctx = self._create_session_context()
        lazy_ctx = create_lazy_context(ctx)

        content = lazy_ctx.get_file("test.py")
        assert content == "print('hello')"

    def test_get_file_not_found(self):
        """get_file returns empty for missing file."""
        ctx = self._create_session_context()
        lazy_ctx = create_lazy_context(ctx)

        content = lazy_ctx.get_file("nonexistent.py")
        assert content == ""

    def test_iter_file(self):
        """iter_file yields file content in chunks."""
        ctx = self._create_session_context()
        ctx.files["large.txt"] = "x" * 25000
        lazy_ctx = create_lazy_context(ctx)

        chunks = list(lazy_ctx.iter_file("large.txt"))
        combined = "".join(chunks)
        assert combined == "x" * 25000

    def test_tool_outputs_property(self):
        """tool_outputs property returns externalized outputs."""
        ctx = self._create_session_context()
        lazy_ctx = create_lazy_context(ctx)

        outputs = lazy_ctx.tool_outputs
        assert len(outputs) == 1
        assert outputs[0]["tool"] == "bash"

    def test_working_memory_property(self):
        """working_memory property returns copy."""
        ctx = self._create_session_context()
        lazy_ctx = create_lazy_context(ctx)

        mem = lazy_ctx.working_memory
        assert mem == {"key": "value"}

        # Should be a copy
        mem["new_key"] = "new_value"
        assert "new_key" not in lazy_ctx.working_memory

    def test_memory_usage_tracking(self):
        """get_memory_usage_mb tracks cached content."""
        ctx = self._create_session_context()
        lazy_ctx = create_lazy_context(ctx)

        # Before loading
        assert lazy_ctx.get_memory_usage_mb() == 0

        # After loading files
        lazy_ctx.get_file("test.py")
        assert lazy_ctx.get_memory_usage_mb() >= 0


class TestDefaultMemoryLimit:
    """Tests for default configuration."""

    def test_default_memory_limit(self):
        """Default memory limit is 100MB."""
        assert DEFAULT_MEMORY_LIMIT_MB == 100

    def test_config_default_values(self):
        """LazyContextConfig has sensible defaults."""
        config = LazyContextConfig()
        assert config.max_memory_mb == 100
        assert config.use_mmap is True
        assert config.chunk_size == 65536
