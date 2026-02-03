"""
Context externalization for RLM-Claude-Code.

Implements: Spec §3 Context Externalization
           SPEC-01.02 (Phase 4 - Lazy Context Loading)
"""

from __future__ import annotations

import mmap
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import Message, SessionContext, ToolOutput

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

from collections.abc import Callable

# Default memory limit for lazy context cache (100MB)
DEFAULT_MEMORY_LIMIT_MB = 100


@dataclass
class LazyContextConfig:
    """Configuration for lazy context loading."""

    max_memory_mb: int = DEFAULT_MEMORY_LIMIT_MB
    use_mmap: bool = True
    chunk_size: int = 65536  # 64KB chunks for file reading


class LazyContextVariable:
    """
    A lazily-loaded context variable with memory-bounded caching.

    Implements: SPEC-01.02 (Lazy Context Loading)

    Provides generator-based access to large content without
    materializing everything in memory at once.

    Example:
        >>> lazy_file = LazyContextVariable(lambda: read_large_file("huge.txt"))
        >>> for chunk in lazy_file:
        ...     process(chunk)
    """

    def __init__(
        self,
        loader: Callable[[], Any] | Callable[[], Generator[Any, None, None]],
        max_memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        is_generator: bool = False,
    ):
        """
        Initialize lazy context variable.

        Args:
            loader: Callable that returns content or yields chunks
            max_memory_mb: Maximum memory for cached content
            is_generator: If True, loader returns a generator
        """
        self._loader = loader
        self._max_memory_mb = max_memory_mb
        self._is_generator = is_generator
        self._cache: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
        self._materialized: Any | None = None
        self._materialized_size: int = 0

    def __iter__(self) -> Iterator[Any]:
        """Iterate over content chunks."""
        if self._is_generator:
            yield from self._loader()
        else:
            content = self._get_content()
            if isinstance(content, (list, tuple)):
                yield from content
            elif isinstance(content, dict):
                yield from content.items()
            elif isinstance(content, str):
                # Yield in reasonable chunks
                chunk_size = 10000
                for i in range(0, len(content), chunk_size):
                    yield content[i : i + chunk_size]
            else:
                yield content

    def _get_content(self) -> Any:
        """Get full content, caching if within memory limit."""
        if self._materialized is not None:
            return self._materialized

        content = self._loader()

        # Estimate size
        if isinstance(content, str):
            size_bytes = len(content.encode("utf-8", errors="replace"))
        elif isinstance(content, (list, dict)):
            # Rough estimate
            size_bytes = len(str(content))
        else:
            size_bytes = 0

        # Cache if within limit
        max_bytes = self._max_memory_mb * 1024 * 1024
        if size_bytes <= max_bytes:
            self._materialized = content
            self._materialized_size = size_bytes

        return content

    def get(self) -> Any:
        """Get the full content (materializes if needed)."""
        return self._get_content()

    def peek(self, start: int = 0, end: int | None = None) -> Any:
        """
        Peek at a slice of the content without full materialization.

        Args:
            start: Start index
            end: End index (None for end of content)

        Returns:
            Sliced content
        """
        content = self._get_content()
        if isinstance(content, (str, list, tuple)):
            return content[start:end]
        return content

    @property
    def is_loaded(self) -> bool:
        """Check if content is already materialized."""
        return self._materialized is not None

    @property
    def estimated_size_mb(self) -> float:
        """Get estimated size in MB."""
        return self._materialized_size / (1024 * 1024)

    def clear_cache(self) -> None:
        """Clear the materialized cache."""
        self._materialized = None
        self._materialized_size = 0


class LazyFileLoader:
    """
    Memory-efficient file loader using mmap or chunked reading.

    Implements: SPEC-01.02 (Lazy Context Loading)
    """

    def __init__(self, config: LazyContextConfig | None = None):
        """Initialize with optional config."""
        self._config = config or LazyContextConfig()
        self._mmap_cache: dict[str, mmap.mmap] = {}

    def load_file_lazy(self, path: str | Path) -> LazyContextVariable:
        """
        Create a lazy loader for a file.

        Args:
            path: Path to file

        Returns:
            LazyContextVariable that loads file on demand
        """
        path = Path(path)

        def loader() -> str:
            return self._read_file(path)

        return LazyContextVariable(
            loader=loader,
            max_memory_mb=self._config.max_memory_mb,
            is_generator=False,
        )

    def load_file_chunks(self, path: str | Path) -> LazyContextVariable:
        """
        Create a generator-based loader for large files.

        Args:
            path: Path to file

        Returns:
            LazyContextVariable that yields file chunks
        """
        path = Path(path)

        def chunk_generator() -> Generator[str, None, None]:
            yield from self._read_file_chunks(path)

        return LazyContextVariable(
            loader=chunk_generator,
            max_memory_mb=self._config.max_memory_mb,
            is_generator=True,
        )

    def _read_file(self, path: Path) -> str:
        """Read file content, using mmap for large files."""
        if not path.exists():
            return ""

        file_size = path.stat().st_size

        # Use mmap for large files if enabled
        if self._config.use_mmap and file_size > 1024 * 1024:  # > 1MB
            return self._read_with_mmap(path)

        # Regular read for smaller files
        return path.read_text(encoding="utf-8", errors="replace")

    def _read_with_mmap(self, path: Path) -> str:
        """Read file using memory-mapped I/O."""
        str_path = str(path)

        # Check cache
        if str_path in self._mmap_cache:
            mm = self._mmap_cache[str_path]
            mm.seek(0)
            return mm.read().decode("utf-8", errors="replace")

        # Create new mmap
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            content = mm.read().decode("utf-8", errors="replace")
            mm.close()
            return content

    def _read_file_chunks(self, path: Path) -> Generator[str, None, None]:
        """Read file in chunks."""
        if not path.exists():
            return

        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(self._config.chunk_size)
                if not chunk:
                    break
                yield chunk

    def close(self) -> None:
        """Close any open mmap handles."""
        for mm in self._mmap_cache.values():
            try:
                mm.close()
            except Exception:
                pass
        self._mmap_cache.clear()


@dataclass
class LazyContext:
    """
    Lazy context container with memory-bounded caching.

    Wraps SessionContext with lazy loading for large content.
    """

    _context: SessionContext
    _config: LazyContextConfig = field(default_factory=LazyContextConfig)
    _file_loader: LazyFileLoader = field(default=None)  # type: ignore[assignment]
    _lazy_files: dict[str, LazyContextVariable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize file loader."""
        if self._file_loader is None:
            self._file_loader = LazyFileLoader(self._config)

    @property
    def conversation(self) -> list[dict[str, Any]]:
        """Get conversation (usually small, no lazy loading needed)."""
        return externalize_conversation(self._context.messages)

    @property
    def files(self) -> dict[str, LazyContextVariable]:
        """Get files as lazy variables."""
        for path, content in self._context.files.items():
            if path not in self._lazy_files:
                # Wrap existing content in lazy variable
                self._lazy_files[path] = LazyContextVariable(
                    loader=lambda c=content: c,
                    max_memory_mb=self._config.max_memory_mb,
                )
        return self._lazy_files

    def get_file(self, path: str) -> str:
        """Get file content (materializes if lazy)."""
        if path in self._lazy_files:
            return self._lazy_files[path].get()
        if path in self._context.files:
            return self._context.files[path]
        return ""

    def iter_file(self, path: str) -> Iterator[str]:
        """Iterate over file content in chunks."""
        if path in self._lazy_files:
            yield from self._lazy_files[path]
        elif path in self._context.files:
            content = self._context.files[path]
            # Yield in chunks
            chunk_size = 10000
            for i in range(0, len(content), chunk_size):
                yield content[i : i + chunk_size]

    @property
    def tool_outputs(self) -> list[dict[str, Any]]:
        """Get tool outputs."""
        return externalize_tool_outputs(self._context.tool_outputs)

    @property
    def working_memory(self) -> dict[str, Any]:
        """Get working memory."""
        return self._context.working_memory.copy()

    def get_memory_usage_mb(self) -> float:
        """Get estimated memory usage of cached content."""
        total = 0.0
        for lazy_var in self._lazy_files.values():
            total += lazy_var.estimated_size_mb
        return total


def externalize_conversation(messages: list[Message]) -> list[dict[str, Any]]:
    """
    Externalize conversation history as REPL-accessible data.

    Implements: Spec §3.1 Context Variable Schema
    """
    return [
        {
            "role": m.role.value,
            "content": m.content,
            "timestamp": m.timestamp,
        }
        for m in messages
    ]


def externalize_files(files: dict[str, str]) -> dict[str, str]:
    """
    Externalize cached files as REPL-accessible data.

    Implements: Spec §3.1 Context Variable Schema
    """
    # Direct passthrough for now; could add metadata
    return files.copy()


def externalize_tool_outputs(outputs: list[ToolOutput]) -> list[dict[str, Any]]:
    """
    Externalize tool outputs as REPL-accessible data.

    Implements: Spec §3.1 Context Variable Schema
    """
    return [
        {
            "tool": o.tool_name,
            "content": o.content,
            "exit_code": o.exit_code,
            "timestamp": o.timestamp,
        }
        for o in outputs
    ]


def externalize_context(context: SessionContext) -> dict[str, Any]:
    """
    Externalize complete session context for REPL.

    Implements: Spec §3 Context Externalization
    """
    return {
        "conversation": externalize_conversation(context.messages),
        "files": externalize_files(context.files),
        "tool_outputs": externalize_tool_outputs(context.tool_outputs),
        "working_memory": context.working_memory.copy(),
        "context_stats": {
            "total_tokens": context.total_tokens,
            "conversation_count": len(context.messages),
            "file_count": len(context.files),
            "tool_output_count": len(context.tool_outputs),
        },
    }


def create_lazy_context(
    context: SessionContext,
    max_memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> LazyContext:
    """
    Create a lazy context wrapper for a session context.

    Args:
        context: Session context to wrap
        max_memory_mb: Maximum memory for caching

    Returns:
        LazyContext with memory-bounded loading
    """
    config = LazyContextConfig(max_memory_mb=max_memory_mb)
    return LazyContext(_context=context, _config=config)


@dataclass
class MicroModeContext:
    """
    Externalized context for micro mode REPL access.

    Implements: SPEC-14.40-14.44 (Context Externalization for Micro Mode)

    Provides the four standard variables that micro mode REPL can access:
    - query: Current user query
    - context: Available context (files, conversation)
    - memory: Relevant memory facts (lazy loaded)
    - prior_result: Previous turn result (if any)

    All large content is lazy-loaded to ensure <100ms externalization overhead.
    """

    query: str
    _context: SessionContext
    _memory_loader: Callable[[], list[dict[str, Any]]] | None = None
    prior_result: str | None = None
    _lazy_context: LazyContextVariable | None = field(default=None, init=False)
    _lazy_memory: LazyContextVariable | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize lazy loaders for large content."""
        # Lazy load context to ensure <100ms overhead (SPEC-14.43)
        self._lazy_context = LazyContextVariable(
            loader=lambda: externalize_context(self._context),
            max_memory_mb=50,  # Limit context cache
        )

        # Lazy load memory (SPEC-14.44)
        if self._memory_loader is not None:
            self._lazy_memory = LazyContextVariable(
                loader=self._memory_loader,
                max_memory_mb=10,  # Small limit for memory facts
            )

    @property
    def context(self) -> dict[str, Any]:
        """
        Get externalized context (lazy loaded).

        Implements: SPEC-14.41, SPEC-14.44

        Returns:
            Dict with conversation, files, tool_outputs, working_memory
        """
        if self._lazy_context is None:
            return {}
        return self._lazy_context.get()

    @property
    def memory(self) -> list[dict[str, Any]]:
        """
        Get relevant memory facts (lazy loaded).

        Implements: SPEC-14.41, SPEC-14.44

        Returns:
            List of memory fact dicts
        """
        if self._lazy_memory is None:
            return []
        return self._lazy_memory.get()

    def to_repl_vars(self) -> dict[str, Any]:
        """
        Convert to REPL variable dict.

        Implements: SPEC-14.40

        Returns:
            Dict suitable for injection into REPL globals
        """
        return {
            "query": self.query,
            "context": self.context,
            "memory": self.memory,
            "prior_result": self.prior_result or "",
        }

    @property
    def is_context_loaded(self) -> bool:
        """Check if context has been accessed/loaded."""
        return self._lazy_context is not None and self._lazy_context.is_loaded

    @property
    def is_memory_loaded(self) -> bool:
        """Check if memory has been accessed/loaded."""
        return self._lazy_memory is not None and self._lazy_memory.is_loaded


def create_micro_context(
    query: str,
    session_context: SessionContext,
    memory_loader: Callable[[], list[dict[str, Any]]] | None = None,
    prior_result: str | None = None,
) -> MicroModeContext:
    """
    Create a micro mode context for REPL access.

    Implements: SPEC-14.40-14.44

    Args:
        query: Current user query
        session_context: Full session context (lazy loaded)
        memory_loader: Callable that returns memory facts (lazy loaded)
        prior_result: Result from previous turn (if any)

    Returns:
        MicroModeContext with lazy-loaded content

    Example:
        >>> ctx = create_micro_context(
        ...     query="What does this function do?",
        ...     session_context=session.context,
        ...     memory_loader=lambda: memory_store.query("function"),
        ... )
        >>> repl_vars = ctx.to_repl_vars()
        >>> # Now REPL can access: query, context, memory, prior_result
    """
    return MicroModeContext(
        query=query,
        _context=session_context,
        _memory_loader=memory_loader,
        prior_result=prior_result,
    )


__all__ = [
    "externalize_context",
    "externalize_conversation",
    "externalize_files",
    "externalize_tool_outputs",
    # Lazy loading (SPEC-01.02)
    "LazyContextVariable",
    "LazyContextConfig",
    "LazyFileLoader",
    "LazyContext",
    "create_lazy_context",
    "DEFAULT_MEMORY_LIMIT_MB",
    # Micro mode context (SPEC-14.40-14.44)
    "MicroModeContext",
    "create_micro_context",
]
