"""
REPL environment for RLM context manipulation.

Implements: Spec §4 REPL Environment Design
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import TYPE_CHECKING, Any

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    safer_getattr,
)

from .types import DeferredBatch, DeferredOperation, ExecutionResult, SessionContext

if TYPE_CHECKING:
    from .recursive_handler import RecursiveREPL

# Subprocess allowlist for sandbox
ALLOWED_SUBPROCESSES = frozenset({"ty", "ruff"})

# Blocked builtins that could be dangerous
BLOCKED_BUILTINS = frozenset({
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
})


class RLMSecurityError(Exception):
    """Raised when sandbox security is violated."""

    pass


class RLMEnvironment:
    """
    Sandboxed REPL for context manipulation.

    Implements: Spec §4.1 Sandbox Architecture

    The REPL provides:
    - Context variables (conversation, files, tool_outputs, working_memory)
    - Helper functions (peek, search, summarize, recursive_query)
    - Safe stdlib (re, json, string operations)
    - Extended tooling (pydantic, hypothesis, cpmpy)

    Security:
    - Uses RestrictedPython for code compilation
    - Blocks dangerous builtins (open, exec, eval, __import__)
    - Subprocess calls limited to allowlist (ty, ruff)
    """

    def __init__(
        self,
        context: SessionContext,
        recursive_handler: RecursiveREPL | None = None,
        use_restricted: bool = True,
    ):
        """
        Initialize REPL environment with context.

        Args:
            context: Session context to externalize
            recursive_handler: Handler for recursive calls (depth>0)
            use_restricted: Whether to use RestrictedPython (default True)

        Implements: Spec §4.1 Sandbox Architecture
        """
        self.context = context
        self.recursive_handler = recursive_handler
        self.use_restricted = use_restricted

        # Build safe builtins
        self._safe_builtins = self._build_safe_builtins()

        # Build globals with context and helpers
        self.globals: dict[str, Any] = {
            # Module-level names needed for class definitions
            "__name__": "__rlm_repl__",
            "__doc__": None,
            # Builtins
            "__builtins__": self._safe_builtins,
            # RestrictedPython guards
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_getattr_": safer_getattr,
            # Context variables
            "conversation": [
                {"role": m.role.value, "content": m.content} for m in context.messages
            ],
            "files": context.files.copy(),
            "tool_outputs": [
                {"tool": o.tool_name, "content": o.content} for o in context.tool_outputs
            ],
            "working_memory": context.working_memory.copy(),
            # Helper functions
            "peek": self._peek,
            "search": self._search,
            "summarize": self._summarize,
            "recursive_query": self._recursive_query,
            "recursive_llm": self._recursive_query,
            "llm": self._recursive_query,  # Shorthand alias
            "llm_batch": self._llm_batch,  # Parallel LLM calls
            # Safe subprocess execution
            "run_tool": self._run_tool,
            # Standard library (safe modules)
            "re": re,
            "json": json,
        }

        # Add extended tooling
        self._add_extended_tooling()

        # Local namespace for user variables
        self.locals: dict[str, Any] = {}

        # Execution history for debugging
        self.history: list[dict[str, Any]] = []

        # Pending async operations (deferred until after sync execution)
        self.pending_operations: list[DeferredOperation] = []
        self.pending_batches: list[DeferredBatch] = []
        self._operation_counter = 0

    def _build_safe_builtins(self) -> dict[str, Any]:
        """
        Build restricted builtins dict.

        Implements: Spec §4.1 Security Constraints
        """
        # Start with RestrictedPython's safe_builtins
        builtins = dict(safe_builtins)

        # Add common safe functions
        safe_additions = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "range": range,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "hasattr": hasattr,
            "getattr": getattr,
            "type": type,
            "repr": repr,
            "hash": hash,
            "id": id,
            "ord": ord,
            "chr": chr,
            "hex": hex,
            "bin": bin,
            "oct": oct,
            "format": format,
            "slice": slice,
            "print": self._safe_print,  # Captured print
        }

        builtins.update(safe_additions)

        # Explicitly remove dangerous builtins
        for blocked in BLOCKED_BUILTINS:
            builtins.pop(blocked, None)

        return builtins

    def _add_extended_tooling(self) -> None:
        """
        Add pydantic, hypothesis, and cpmpy to globals.

        Implements: Spec §4.1.1 Extended Python Tooling
        """
        try:
            import pydantic

            self.globals["pydantic"] = pydantic
            # Common pydantic imports for convenience
            self.globals["BaseModel"] = pydantic.BaseModel
            self.globals["Field"] = pydantic.Field
        except ImportError:
            pass

        try:
            import hypothesis
            from hypothesis import strategies as st

            self.globals["hypothesis"] = hypothesis
            self.globals["given"] = hypothesis.given
            self.globals["st"] = st
        except ImportError:
            pass

        try:
            import cpmpy

            self.globals["cp"] = cpmpy
            self.globals["cpmpy"] = cpmpy
        except ImportError:
            pass

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute code in sandbox.

        Implements: Spec §4.1 Sandbox Architecture

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output or error
        """
        start_time = time.time()
        output_capture: list[str] = []

        # Set up print capture
        self._print_buffer = output_capture

        try:
            if self.use_restricted:
                # Use RestrictedPython for compilation
                # compile_restricted returns a code object directly
                # Errors are raised as SyntaxError exceptions
                byte_code = compile_restricted(
                    code,
                    filename="<repl>",
                    mode="exec",
                )

                # Execute the restricted code
                exec(byte_code, self.globals, self.locals)
            else:
                # Fallback: use regular exec (for testing)
                compiled = compile(code, "<repl>", "exec")
                exec(compiled, self.globals, self.locals)

            # Get result - check for _ variable or last expression
            output = self.locals.get("_")
            if output is None and output_capture:
                output = "\n".join(output_capture)

            execution_time = (time.time() - start_time) * 1000

            # Record in history
            self.history.append({
                "code": code,
                "success": True,
                "output": output,
                "time_ms": execution_time,
            })

            return ExecutionResult(
                success=True,
                output=output,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            # Record in history
            self.history.append({
                "code": code,
                "success": False,
                "error": str(e),
                "time_ms": execution_time,
            })

            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
        finally:
            self._print_buffer = []

    def _safe_print(self, *args: Any, **_kwargs: Any) -> None:
        """Capture print output instead of writing to stdout."""
        output = " ".join(str(arg) for arg in args)
        if hasattr(self, "_print_buffer"):
            self._print_buffer.append(output)

    def _peek(self, var: Any, start: int = 0, end: int = 1000) -> Any:
        """
        View a portion of a context variable.

        Implements: Spec §3.2 Root Prompt Structure

        Args:
            var: Variable to peek into
            start: Start index
            end: End index

        Returns:
            Sliced portion of the input - str for strings, list slice for lists,
            dict subset for dicts, str representation slice for other types.
        """
        if isinstance(var, (str, list)):
            return var[start:end]
        elif isinstance(var, dict):
            items = list(var.items())[start:end]
            return dict(items)
        else:
            return str(var)[start:end]

    def _search(
        self, var: Any, pattern: str, regex: bool = False
    ) -> list[dict[str, Any]]:
        """
        Search for patterns in context variable.

        Implements: Spec §3.2 Root Prompt Structure

        Args:
            var: Variable to search in
            pattern: Pattern to search for
            regex: If True, treat pattern as regex

        Returns:
            List of match results with location info
        """
        results: list[dict[str, Any]] = []

        if regex:
            compiled = re.compile(pattern)

            def matcher(s: str) -> bool:
                return bool(compiled.search(s))
        else:

            def matcher(s: str) -> bool:
                return pattern.lower() in s.lower()

        if isinstance(var, str):
            if matcher(var):
                results.append({"match": var[:500], "type": "string"})
        elif isinstance(var, list):
            for i, item in enumerate(var):
                content = str(item) if not isinstance(item, str) else item
                if matcher(content):
                    results.append({"index": i, "match": content[:200]})
        elif isinstance(var, dict):
            for key, value in var.items():
                content = str(value) if not isinstance(value, str) else value
                if matcher(content) or matcher(str(key)):
                    results.append({"key": key, "match": content[:200]})

        return results

    def _summarize(self, var: Any, max_tokens: int = 500) -> DeferredOperation:
        """
        Summarize context variable via sub-call.

        Implements: Spec §4.2 Recursive Call Implementation

        Args:
            var: Variable to summarize
            max_tokens: Max tokens for summary

        Returns:
            DeferredOperation that will be resolved by orchestrator
        """
        self._operation_counter += 1
        op_id = f"sum_{self._operation_counter}"

        content = str(var)
        if len(content) > max_tokens * 4:
            content = content[: max_tokens * 4] + "..."

        op = DeferredOperation(
            operation_id=op_id,
            operation_type="summarize",
            query=f"Summarize this content in {max_tokens} tokens or less",
            context=content,
            spawn_repl=False,
        )
        self.pending_operations.append(op)
        return op

    def _recursive_query(
        self,
        query: str,
        context: Any = None,
        spawn_repl: bool = False,
    ) -> DeferredOperation:
        """
        Spawn a recursive sub-query.

        Implements: Spec §4.2 Recursive Call Implementation

        Args:
            query: Query string for sub-call
            context: Context to pass to sub-call (optional)
            spawn_repl: If True, child gets its own REPL

        Returns:
            DeferredOperation that will be resolved by orchestrator
        """
        self._operation_counter += 1
        op_id = f"rq_{self._operation_counter}"

        # Convert context to string if not None
        context_str = str(context)[:8000] if context is not None else ""

        op = DeferredOperation(
            operation_id=op_id,
            operation_type="recursive_query",
            query=query,
            context=context_str,
            spawn_repl=spawn_repl,
        )
        self.pending_operations.append(op)
        return op

    def _llm_batch(
        self,
        queries: list[tuple[str, Any]],
        spawn_repl: bool = False,
    ) -> DeferredBatch:
        """
        Execute multiple LLM queries in parallel.

        Implements: Spec §4.2 Parallel Sub-Calls

        Args:
            queries: List of (query, context) tuples
            spawn_repl: If True, each query gets its own REPL

        Returns:
            DeferredBatch that will be resolved by orchestrator

        Example:
            results = llm_batch([
                ("Analyze auth module", auth_code),
                ("Analyze db module", db_code),
                ("Analyze api module", api_code),
            ])
        """
        self._operation_counter += 1
        batch_id = f"batch_{self._operation_counter}"

        batch = DeferredBatch(batch_id=batch_id)

        for query, context in queries:
            self._operation_counter += 1
            op_id = f"rq_{self._operation_counter}"
            context_str = str(context)[:8000] if context is not None else ""

            op = DeferredOperation(
                operation_id=op_id,
                operation_type="recursive_query",
                query=query,
                context=context_str,
                spawn_repl=spawn_repl,
            )
            batch.operations.append(op)

        self.pending_batches.append(batch)
        return batch

    def has_pending_operations(self) -> bool:
        """Check if there are pending async operations to process."""
        return bool(self.pending_operations) or bool(self.pending_batches)

    def get_pending_operations(self) -> tuple[list[DeferredOperation], list[DeferredBatch]]:
        """Get all pending operations for processing by orchestrator."""
        ops = self.pending_operations.copy()
        batches = self.pending_batches.copy()
        return ops, batches

    def clear_pending_operations(self) -> None:
        """Clear pending operations after they've been processed."""
        self.pending_operations.clear()
        self.pending_batches.clear()

    def resolve_operation(self, op_id: str, result: Any) -> None:
        """
        Resolve a pending operation with its result.

        Args:
            op_id: Operation ID to resolve
            result: Result to inject
        """
        # Check individual operations
        for op in self.pending_operations:
            if op.operation_id == op_id:
                op.resolved = True
                op.result = result
                # Inject into working memory for REPL access
                self.globals["working_memory"][op_id] = result
                return

        # Check batch operations
        for batch in self.pending_batches:
            for op in batch.operations:
                if op.operation_id == op_id:
                    op.resolved = True
                    op.result = result
                    self.globals["working_memory"][op_id] = result
                    return

    def resolve_batch(self, batch_id: str, results: list[Any]) -> None:
        """
        Resolve a batch of operations with their results.

        Args:
            batch_id: Batch ID to resolve
            results: List of results in order
        """
        for batch in self.pending_batches:
            if batch.batch_id == batch_id:
                batch.resolved = True
                batch.results = results
                # Resolve individual operations
                for op, result in zip(batch.operations, results, strict=False):
                    op.resolved = True
                    op.result = result
                    self.globals["working_memory"][op.operation_id] = result
                # Also store full batch results
                self.globals["working_memory"][batch_id] = results
                return

    def _run_tool(
        self,
        tool: str,
        *args: str,
        timeout: float = 30.0,
        stdin_input: str | None = None,
    ) -> dict[str, Any]:
        """
        Run an allowed subprocess tool.

        Implements: Spec §4.1.1 CLI Tools (via subprocess in sandbox)

        Args:
            tool: Tool name (must be in ALLOWED_SUBPROCESSES)
            *args: Arguments to pass to tool
            timeout: Timeout in seconds
            stdin_input: Optional input to pass via stdin

        Returns:
            Dict with stdout, stderr, returncode

        Raises:
            RLMSecurityError: If tool is not in allowlist
        """
        if tool not in ALLOWED_SUBPROCESSES:
            raise RLMSecurityError(
                f"Tool '{tool}' not allowed. Allowed: {sorted(ALLOWED_SUBPROCESSES)}"
            )

        try:
            result = subprocess.run(
                [tool, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin_input,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Tool '{tool}' timed out after {timeout}s",
                "returncode": -1,
                "success": False,
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": f"Tool '{tool}' not found",
                "returncode": -1,
                "success": False,
            }

    def get_context_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current context.

        Implements: Spec §3.1 Context Variable Schema

        Returns:
            Dict with context statistics
        """
        conv = self.globals.get("conversation", [])
        files = self.globals.get("files", {})
        tool_outputs = self.globals.get("tool_outputs", [])

        conv_chars = sum(len(m.get("content", "")) for m in conv)
        file_chars = sum(len(content) for content in files.values())
        output_chars = sum(len(o.get("content", "")) for o in tool_outputs)

        return {
            "conversation_count": len(conv),
            "conversation_tokens": conv_chars // 4,
            "file_count": len(files),
            "file_tokens": file_chars // 4,
            "tool_output_count": len(tool_outputs),
            "tool_output_tokens": output_chars // 4,
            "total_tokens": (conv_chars + file_chars + output_chars) // 4,
        }

    def inject_result(self, name: str, value: Any) -> None:
        """
        Inject a result back into the REPL namespace.

        Args:
            name: Variable name
            value: Value to inject
        """
        self.locals[name] = value

    def get_variable(self, name: str) -> Any:
        """
        Get a variable from the REPL namespace.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        # Check locals first
        if name in self.locals:
            return self.locals[name]
        # Then globals
        if name in self.globals:
            return self.globals[name]
        # Check working_memory
        working_memory = self.globals.get("working_memory", {})
        if name in working_memory:
            return working_memory[name]
        raise KeyError(f"Variable '{name}' not found")

    def get_execution_history(self) -> list[dict[str, Any]]:
        """Get execution history for debugging."""
        return self.history.copy()


async def typecheck_snippet(code: str, timeout: float = 30.0) -> dict[str, Any]:
    """
    Run ty type checker on a code snippet.

    Implements: Spec §4.1.1 CLI Tools

    Args:
        code: Python code to type check
        timeout: Timeout in seconds

    Returns:
        TypeCheckResult with success status and errors
    """
    env = RLMEnvironment(SessionContext())
    result = env._run_tool("ty", "check", "-", timeout=timeout, stdin_input=code)

    return {
        "success": result["success"],
        "errors": result["stderr"] if not result["success"] else "",
        "output": result["stdout"],
    }


async def lint_snippet(code: str, timeout: float = 30.0) -> dict[str, Any]:
    """
    Run ruff linter on a code snippet.

    Implements: Spec §4.1.1 CLI Tools

    Args:
        code: Python code to lint
        timeout: Timeout in seconds

    Returns:
        LintResult with issues found
    """
    env = RLMEnvironment(SessionContext())
    result = env._run_tool(
        "ruff",
        "check",
        "--stdin-filename=snippet.py",
        "-",
        timeout=timeout,
        stdin_input=code,
    )

    return {
        "success": result["success"],
        "issues": result["stdout"],
        "errors": result["stderr"],
    }


__all__ = [
    "RLMEnvironment",
    "RLMSecurityError",
    "ALLOWED_SUBPROCESSES",
    "typecheck_snippet",
    "lint_snippet",
]
