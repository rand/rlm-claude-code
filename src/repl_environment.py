"""
REPL environment for RLM context manipulation.

Implements: Spec §4 REPL Environment Design
Implements: Spec SPEC-14.03-14.04 for micro mode restrictions
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import TYPE_CHECKING, Any, Literal

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    safer_getattr,
)

from .tokenization import partition_content_by_tokens
from .types import DeferredBatch, DeferredOperation, ExecutionResult, SessionContext

# REPL function access levels (SPEC-14.03-14.04)
REPLAccessLevel = Literal["micro", "standard", "full"]

# Functions available in micro mode (SPEC-14.03, SPEC-14.52)
MICRO_MODE_FUNCTIONS = frozenset(
    {
        "peek",
        "search",
        "summarize_local",
        "memory_query",
        "memory_add_fact",  # SPEC-14.52: Micro mode can store insights
    }
)

# Functions NOT available in micro mode (SPEC-14.04)
MICRO_MODE_BLOCKED = frozenset(
    {
        "llm",
        "recursive_llm",
        "recursive_query",
        "llm_batch",
        "summarize",  # Uses LLM
        "map_reduce",
        "find_relevant",  # Can use LLM scoring
        "verify_claim",  # Uses LLM for verification
        "evidence_dependence",  # Uses LLM for consistency checking
        "audit_reasoning",  # Uses LLM for reasoning trace verification
        "detect_hallucinations",  # Uses LLM for claim extraction and verification
    }
)

if TYPE_CHECKING:
    from .memory_store import MemoryStore
    from .recursive_handler import RecursiveREPL

# Subprocess allowlist for sandbox
ALLOWED_SUBPROCESSES = frozenset({"uv", "ty", "ruff"})

# Blocked builtins that could be dangerous
BLOCKED_BUILTINS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
    }
)


class RLMSecurityError(Exception):
    """Raised when sandbox security is violated."""

    pass


class _REPLPrintCollector:
    """Print collector for RestrictedPython that captures output to a buffer."""

    def __init__(self, buffer: list[str]):
        self._buffer = buffer

    def _call_print(self, *args: Any, **kwargs: Any) -> None:
        """Called by RestrictedPython for print() statements."""
        output = " ".join(str(arg) for arg in args)
        self._buffer.append(output)

    def write(self, text: str) -> None:
        """Write method for compatibility."""
        self._buffer.append(text)


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
        access_level: REPLAccessLevel = "standard",
    ):
        """
        Initialize REPL environment with context.

        Args:
            context: Session context to externalize
            recursive_handler: Handler for recursive calls (depth>0)
            use_restricted: Whether to use RestrictedPython (default True)
            access_level: Function access level ("micro", "standard", "full")
                          - "micro": SPEC-14.03 restricted functions only
                          - "standard": All REPL functions except tool access
                          - "full": All REPL functions including tool access

        Implements: Spec §4.1 Sandbox Architecture
        Implements: Spec SPEC-14.03-14.04 for micro mode
        """
        self.context = context
        self.recursive_handler = recursive_handler
        self.use_restricted = use_restricted
        self.access_level = access_level

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
            "_getitem_": self._guarded_getitem,
            "_write_": self._guarded_write,
            # Context variables
            "context": context,  # Full SessionContext object
            "conversation": [
                {"role": m.role.value, "content": m.content} for m in context.messages
            ],
            "files": context.files.copy(),
            "tool_outputs": [
                {"tool": o.tool_name, "content": o.content} for o in context.tool_outputs
            ],
            "working_memory": context.working_memory.copy(),
            # Standard library (safe modules)
            "re": re,
            "json": json,
        }

        # Add helper functions based on access level (SPEC-14.03-14.04)
        self._add_helper_functions()

        # Add extended tooling (except in micro mode)
        if access_level != "micro":
            self._add_extended_tooling()

        # Local namespace for user variables
        self.locals: dict[str, Any] = {}

        # Execution history for debugging
        self.history: list[dict[str, Any]] = []

        # Pending async operations (deferred until after sync execution)
        self.pending_operations: list[DeferredOperation] = []
        self.pending_batches: list[DeferredBatch] = []
        self._operation_counter = 0

        # Memory store (initialized via enable_memory())
        self._memory_store: MemoryStore | None = None
        self._session_id: str | None = None

    def _guarded_getitem(self, obj: Any, key: Any) -> Any:
        """
        Safe getitem guard for RestrictedPython.

        Allows subscript access (obj[key]) for safe container types.
        """
        # Allow access to standard container types
        if isinstance(obj, (dict, list, tuple, str, bytes)):
            return obj[key]
        # Allow access to dataclass-like objects via dict interface
        if hasattr(obj, "__getitem__"):
            return obj[key]
        msg = f"Subscript access not allowed on {type(obj).__name__}"
        raise TypeError(msg)

    def _guarded_write(self, obj: Any) -> Any:
        """
        Safe write guard for RestrictedPython.

        Returns a wrapper that allows item/attribute assignment on safe types.
        """
        # For RestrictedPython, _write_ returns the object itself for safe types
        # The actual write operation is then performed by the generated code
        if isinstance(obj, (dict, list)):
            return obj
        # Allow writes to working_memory dict
        if obj is self.context.working_memory or obj is self.globals.get("working_memory"):
            return obj
        msg = f"Write access not allowed on {type(obj).__name__}"
        raise TypeError(msg)

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

    def _add_helper_functions(self) -> None:
        """
        Add helper functions to globals based on access level.

        Implements: Spec SPEC-14.03-14.04 for micro mode restrictions

        Access levels:
        - "micro": Only peek, search, summarize_local, memory_query
        - "standard": All REPL functions
        - "full": All REPL functions including tool access
        """
        # Functions available at all access levels (SPEC-14.03)
        self.globals["peek"] = self._peek
        self.globals["search"] = self._search
        self.globals["summarize_local"] = self._summarize_local
        # Note: memory_query is added via enable_memory()

        # Functions NOT available in micro mode (SPEC-14.04)
        if self.access_level != "micro":
            # LLM-based functions
            self.globals["summarize"] = self._summarize
            self.globals["recursive_query"] = self._recursive_query
            self.globals["recursive_llm"] = self._recursive_query
            self.globals["llm"] = self._recursive_query  # Shorthand alias
            self.globals["llm_batch"] = self._llm_batch  # Parallel LLM calls
            # Advanced REPL functions (SPEC-01)
            self.globals["map_reduce"] = self._map_reduce
            self.globals["find_relevant"] = self._find_relevant
            self.globals["extract_functions"] = self._extract_functions
            # Safe subprocess execution (standard and full modes)
            self.globals["run_tool"] = self._run_tool
            # Epistemic verification functions (SPEC-16)
            self.globals["verify_claim"] = self._verify_claim
            self.globals["evidence_dependence"] = self._evidence_dependence
            self.globals["audit_reasoning"] = self._audit_reasoning
            self.globals["detect_hallucinations"] = self._detect_hallucinations

    def _summarize_local(self, var: Any, max_chars: int = 500) -> str:
        """
        Summarize a variable locally without LLM calls.

        Implements: Spec SPEC-14.03 - Truncation-based summary for micro mode

        This function provides a quick summary by:
        1. Converting to string if not already
        2. Truncating to max_chars
        3. Adding ellipsis if truncated
        4. For lists/dicts, showing count and sample

        Args:
            var: Variable to summarize
            max_chars: Maximum characters in summary (default 500)

        Returns:
            String summary of the variable
        """
        if isinstance(var, str):
            if len(var) <= max_chars:
                return var
            return var[:max_chars] + f"... [{len(var)} chars total]"

        elif isinstance(var, list):
            count = len(var)
            if count == 0:
                return "[] (empty list)"
            # Show first few items
            preview_items = []
            chars_used = 0
            for item in var[:10]:
                item_str = repr(item)[:100]
                if chars_used + len(item_str) > max_chars - 50:
                    break
                preview_items.append(item_str)
                chars_used += len(item_str)
            preview = ", ".join(preview_items)
            if count > len(preview_items):
                return f"[{preview}, ...] ({count} items total)"
            return f"[{preview}]"

        elif isinstance(var, dict):
            count = len(var)
            if count == 0:
                return "{} (empty dict)"
            # Show first few key-value pairs
            preview_items = []
            chars_used = 0
            for key, value in list(var.items())[:10]:
                item_str = f"{repr(key)}: {repr(value)[:50]}"
                if chars_used + len(item_str) > max_chars - 50:
                    break
                preview_items.append(item_str)
                chars_used += len(item_str)
            preview = ", ".join(preview_items)
            if count > len(preview_items):
                return f"{{{preview}, ...}} ({count} keys total)"
            return f"{{{preview}}}"

        else:
            # For other types, convert to string and truncate
            s = str(var)
            type_name = type(var).__name__
            if len(s) <= max_chars:
                return f"({type_name}) {s}"
            return f"({type_name}) {s[:max_chars]}... [{len(s)} chars total]"

    def _add_extended_tooling(self) -> None:
        """
        Add pydantic, hypothesis, cpmpy, and data science libraries to globals.

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

        # Data science libraries
        try:
            import numpy

            self.globals["numpy"] = numpy
            self.globals["np"] = numpy  # Common alias
        except ImportError:
            pass

        try:
            import pandas

            self.globals["pandas"] = pandas
            self.globals["pd"] = pandas  # Common alias
        except ImportError:
            pass

        try:
            import polars

            self.globals["polars"] = polars
            self.globals["pl"] = polars  # Common alias
        except ImportError:
            pass

        try:
            import seaborn

            self.globals["seaborn"] = seaborn
            self.globals["sns"] = seaborn  # Common alias
        except ImportError:
            pass

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute code in sandbox.

        Implements: Spec §4.1 Sandbox Architecture

        Uses eval-first approach: tries to evaluate code as an expression first
        to capture return values (e.g., `1+1` returns `2`). Falls back to exec
        mode for statements (e.g., `x = 1`).

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output or error
        """
        start_time = time.time()
        output_capture: list[str] = []

        # Set up print capture
        self._print_buffer = output_capture

        # Create print collector for RestrictedPython
        print_collector = _REPLPrintCollector(output_capture)
        self.globals["_print"] = print_collector
        self.globals["_print_"] = lambda _: print_collector

        try:
            result = None

            if self.use_restricted:
                # Try eval mode first for expression values
                try:
                    byte_code = compile_restricted(
                        code,
                        filename="<repl>",
                        mode="eval",
                    )
                    result = eval(byte_code, self.globals, self.locals)
                except SyntaxError:
                    # Fall back to exec mode for statements
                    byte_code = compile_restricted(
                        code,
                        filename="<repl>",
                        mode="exec",
                    )
                    exec(byte_code, self.globals, self.locals)
            else:
                # Fallback: use regular compile (for testing)
                try:
                    compiled = compile(code, "<repl>", "eval")
                    result = eval(compiled, self.globals, self.locals)
                except SyntaxError:
                    # Fall back to exec mode for statements
                    compiled = compile(code, "<repl>", "exec")
                    exec(compiled, self.globals, self.locals)

            # Use eval result if we got one, otherwise check for _ variable
            output = result
            if output is None:
                output = self.locals.get("_")

            # Capture stdout separately
            stdout = "\n".join(output_capture) if output_capture else ""

            # For backward compatibility: if no return value, use stdout as output
            final_output = output if output is not None else (stdout or None)

            execution_time = (time.time() - start_time) * 1000

            # Record in history
            self.history.append(
                {
                    "code": code,
                    "success": True,
                    "output": final_output,
                    "stdout": stdout,
                    "time_ms": execution_time,
                }
            )

            return ExecutionResult(
                success=True,
                output=final_output,
                stdout=stdout,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            # Enhance error message with suggestions
            enhanced_error = self._enhance_error(e, code)

            # Record in history
            self.history.append(
                {
                    "code": code,
                    "success": False,
                    "error": enhanced_error,
                    "time_ms": execution_time,
                }
            )

            return ExecutionResult(
                success=False,
                error=enhanced_error,
                execution_time_ms=execution_time,
            )
        finally:
            self._print_buffer = []

    def _safe_print(self, *args: Any, **_kwargs: Any) -> None:
        """Capture print output instead of writing to stdout."""
        output = " ".join(str(arg) for arg in args)
        if hasattr(self, "_print_buffer"):
            self._print_buffer.append(output)

    def _enhance_error(self, error: Exception, code: str) -> str:
        """
        Enhance error message with helpful suggestions.

        Implements: SPEC-12.12 Enhanced Error Messages

        Args:
            error: The exception that occurred
            code: The code that caused the error

        Returns:
            Enhanced error message with suggestions
        """
        error_type = type(error).__name__
        error_msg = str(error)
        suggestions: list[str] = []

        if isinstance(error, NameError):
            # Extract undefined name and suggest similar ones
            undefined = self._extract_undefined_name(error_msg)
            if undefined:
                similar = self._find_similar_names(undefined)
                if similar:
                    suggestions.append(f"Did you mean: {', '.join(similar[:3])}?")
            suggestions.append(f"Available: {self._list_available_names()}")

        elif isinstance(error, KeyError):
            # Suggest available keys
            dict_keys = self._find_dict_keys_in_code(code)
            if dict_keys:
                suggestions.append(f"Available keys: {', '.join(dict_keys[:10])}")
            suggestions.append("Use .keys() to list all available keys")

        elif isinstance(error, AttributeError):
            # Suggest available methods
            obj_type = self._extract_object_type(error_msg)
            if obj_type:
                suggestions.append(f"Check available methods with dir({obj_type})")
            helpers = self._get_available_helpers()
            suggestions.append(f"Common helpers: {', '.join(f'{h}()' for h in helpers[:4])}")

        elif isinstance(error, TypeError):
            suggestions.append("Check function signature and argument types")
            suggestions.append("Common helpers: peek(var, start, end), search(var, pattern)")

        elif isinstance(error, IndexError):
            suggestions.append("Use len() to check collection size first")
            suggestions.append("Use peek(var, start, end) for safe slicing")

        # Build enhanced message
        result = f"{error_type}: {error_msg}"
        if suggestions:
            result += "\n\nSuggestions:\n• " + "\n• ".join(suggestions)

        return result

    def _extract_undefined_name(self, error_msg: str) -> str | None:
        """Extract the undefined name from a NameError message."""
        # Pattern: "name 'foo' is not defined"
        match = re.search(r"name ['\"](\w+)['\"] is not defined", error_msg)
        return match.group(1) if match else None

    def _find_similar_names(self, name: str, threshold: float = 0.6) -> list[str]:
        """Find similar names in the environment."""
        from difflib import SequenceMatcher

        all_names = set()
        # Collect all available names
        all_names.update(self.globals.get("working_memory", {}).keys())
        all_names.update(k for k in self.globals if not k.startswith("_"))
        all_names.update(self.locals.keys())

        # Find similar names
        similar = []
        for candidate in all_names:
            ratio = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
            if ratio >= threshold and candidate != name:
                similar.append((ratio, candidate))

        # Sort by similarity and return names
        similar.sort(reverse=True)
        return [name for _, name in similar]

    def _get_available_helpers(self) -> list[str]:
        """Get helper functions available at current access level."""
        # Core helpers always available
        helpers = ["peek", "search", "find_relevant"]

        # LLM-based helpers only in standard/full modes (not micro)
        if self.access_level != "micro":
            helpers.extend(["llm", "summarize", "map_reduce", "llm_batch"])
            # Epistemic verification helpers (SPEC-16)
            helpers.extend(["verify_claim", "evidence_dependence", "audit_reasoning", "detect_hallucinations"])

        return helpers

    def _list_available_names(self) -> str:
        """List commonly available names in the environment."""
        names = ["files", "conversation", "tool_outputs", "working_memory"]
        names.extend(self._get_available_helpers())
        wm_keys = list(self.globals.get("working_memory", {}).keys())[:5]
        if wm_keys:
            names.extend(wm_keys)
        return ", ".join(names[:15])

    def _find_dict_keys_in_code(self, code: str) -> list[str]:
        """Find dictionary keys that might be relevant to the error."""
        keys: list[str] = []
        # Check if accessing 'files'
        if "files" in code:
            keys.extend(list(self.globals.get("files", {}).keys())[:10])
        # Check working_memory
        if "working_memory" in code:
            keys.extend(list(self.globals.get("working_memory", {}).keys())[:10])
        return keys

    def _extract_object_type(self, error_msg: str) -> str | None:
        """Extract object type from AttributeError message."""
        # Pattern: "'str' object has no attribute 'foo'"
        match = re.search(r"['\"](\w+)['\"] object has no attribute", error_msg)
        return match.group(1) if match else None

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

    def _search(self, var: Any, pattern: str, regex: bool = False) -> list[dict[str, Any]]:
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

    def _map_reduce(
        self,
        content: str,
        map_prompt: str,
        reduce_prompt: str,
        n_chunks: int = 4,
        model: str = "auto",
    ) -> DeferredBatch:
        """
        Apply map-reduce pattern to large content.

        Implements: Spec SPEC-01.01 through SPEC-01.06

        Args:
            content: Content to process
            map_prompt: Prompt to apply to each chunk (can use {chunk} placeholder)
            reduce_prompt: Prompt to combine map results
            n_chunks: Number of chunks to split content into
            model: Model tier to use ("fast", "balanced", "powerful", "auto")

        Returns:
            DeferredBatch with map operations and reduce metadata
        """
        # Validate model parameter (SPEC-01.05)
        valid_models = {"fast", "balanced", "powerful", "auto"}
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of: {valid_models}")

        # Handle empty content (SPEC-01.06 graceful handling)
        if not content:
            content = ""

        # Partition content into n_chunks (SPEC-01.02)
        chunks = self._partition_content(content, n_chunks)

        # Create batch of map operations (SPEC-01.03)
        self._operation_counter += 1
        batch_id = f"mapreduce_{self._operation_counter}"

        batch = DeferredBatch(batch_id=batch_id)

        for _i, chunk in enumerate(chunks):
            self._operation_counter += 1
            op_id = f"map_{self._operation_counter}"

            # Format map_prompt with chunk if it has {chunk} placeholder
            if "{chunk}" in map_prompt:
                formatted_prompt = map_prompt.format(chunk=chunk)
            else:
                formatted_prompt = f"{map_prompt}\n\nContent:\n{chunk}"

            op = DeferredOperation(
                operation_id=op_id,
                operation_type="map",
                query=formatted_prompt,
                context=chunk,
                spawn_repl=False,
            )
            batch.operations.append(op)

        # Store reduce_prompt in batch metadata for later processing
        # The orchestrator will use this to create the reduce operation
        batch.metadata["reduce_prompt"] = reduce_prompt
        batch.metadata["model"] = model

        self.pending_batches.append(batch)
        return batch

    def _partition_content(self, content: str, n_chunks: int) -> list[str]:
        """
        Partition content into roughly equal token-sized chunks.

        Implements: Spec SPEC-01.02

        Uses token-aware chunking (SPEC-01.01) for accurate LLM context handling.
        Chunks at semantic boundaries (function/class definitions) when possible.

        Args:
            content: Content to partition
            n_chunks: Target number of chunks

        Returns:
            List of content chunks
        """
        return partition_content_by_tokens(content, n_chunks)

    def _find_relevant(
        self,
        content: str,
        query: str,
        top_k: int = 5,
        use_llm_scoring: bool = False,
    ) -> list[tuple[str, float]]:
        """
        Find sections most relevant to query.

        Implements: Spec SPEC-01.07 through SPEC-01.12

        Args:
            content: Content to search
            query: Query to find relevant sections for
            top_k: Number of top results to return
            use_llm_scoring: Whether to use LLM for scoring (if candidates > top_k * 2)

        Returns:
            List of (chunk, score) tuples sorted by relevance descending
        """
        if not content:
            return []

        # Partition into ~50-line chunks with 5-line overlap (SPEC-01.08)
        chunks = self._chunk_with_overlap(content, chunk_lines=50, overlap_lines=5)

        if not chunks:
            return []

        # Extract query keywords for filtering (SPEC-01.09)
        query_lower = query.lower()
        keywords = set(re.findall(r"\b\w{3,}\b", query_lower))

        # Keyword pre-filtering
        scored_chunks: list[tuple[str, float]] = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in chunk_lower)
            if matches > 0 or not keywords:
                # Score based on keyword density
                score = matches / max(len(keywords), 1)
                scored_chunks.append((chunk, score))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Optional LLM scoring if candidates exceed threshold (SPEC-01.10)
        if use_llm_scoring and len(scored_chunks) > top_k * 2:
            # Create deferred batch for LLM scoring
            # For now, we just use keyword scoring
            # LLM scoring would be implemented by creating a DeferredBatch
            # and processing results in the orchestrator
            pass

        # Return top_k results (SPEC-01.11)
        return scored_chunks[:top_k]

    def _chunk_with_overlap(
        self, content: str, chunk_lines: int = 50, overlap_lines: int = 5
    ) -> list[str]:
        """
        Split content into chunks with line overlap.

        Args:
            content: Content to chunk
            chunk_lines: Target lines per chunk
            overlap_lines: Lines to overlap between chunks

        Returns:
            List of overlapping chunks
        """
        lines = content.split("\n")
        if not lines:
            return []

        chunks = []
        start = 0

        while start < len(lines):
            end = min(start + chunk_lines, len(lines))
            chunk = "\n".join(lines[start:end])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            start = end - overlap_lines

            # Ensure we make progress and don't loop forever
            if end >= len(lines):
                break  # Reached end of content

        return chunks

    def _extract_functions(
        self,
        content: str,
        language: str = "python",
    ) -> list[dict[str, Any]]:
        """
        Extract function definitions from code.

        Implements: Spec SPEC-01.13 through SPEC-01.17

        Args:
            content: Source code content
            language: Programming language ("python", "go", "javascript", "typescript")

        Returns:
            List of dicts with keys: "name", "signature", "start_line", "end_line"
        """
        # Validate language (SPEC-01.14)
        valid_languages = {"python", "go", "javascript", "typescript"}
        if language not in valid_languages:
            raise ValueError(f"Invalid language: {language}. Must be one of: {valid_languages}")

        # Handle empty/malformed input gracefully (SPEC-01.17)
        if not content:
            return []

        # Get regex patterns for the language (SPEC-01.16)
        patterns = self._get_function_patterns(language)

        functions: list[dict[str, Any]] = []
        lines = content.split("\n")

        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            for match in re.finditer(pattern, content, re.MULTILINE):
                try:
                    name = match.group("name")
                    signature = match.group(0).strip()

                    # Calculate line numbers
                    # Note: \s* in pattern can match leading newlines, so we need
                    # to find where the actual function keyword starts
                    match_text = match.group(0)
                    leading_ws = len(match_text) - len(match_text.lstrip())
                    start_pos = match.start() + leading_ws
                    start_line = content[:start_pos].count("\n") + 1

                    # Estimate end line (simple heuristic)
                    end_line = self._estimate_function_end(lines, start_line - 1, language)

                    functions.append(
                        {
                            "name": name,
                            "signature": signature,
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    )
                except (IndexError, AttributeError):
                    # Handle malformed input gracefully (SPEC-01.17)
                    continue

        # Remove duplicates (same function might match multiple patterns)
        seen = set()
        unique_functions = []
        for func in functions:
            key = (func["name"], func["start_line"])
            if key not in seen:
                seen.add(key)
                unique_functions.append(func)

        # Sort by start_line
        unique_functions.sort(key=lambda x: x["start_line"])
        return unique_functions

    def _get_function_patterns(self, language: str) -> list[dict[str, Any]]:
        """
        Get regex patterns for function extraction by language.

        Implements: Spec SPEC-01.16
        """
        patterns: dict[str, list[dict[str, Any]]] = {
            "python": [
                {
                    "pattern": r"^\s*def\s+(?P<name>\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
                    "type": "function",
                },
                {
                    "pattern": r"^\s*async\s+def\s+(?P<name>\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
                    "type": "async_function",
                },
            ],
            "go": [
                {
                    "pattern": r"^func\s+(?P<name>\w+)\s*\([^)]*\)",
                    "type": "function",
                },
                {
                    "pattern": r"^func\s+\([^)]+\)\s+(?P<name>\w+)\s*\([^)]*\)",
                    "type": "method",
                },
            ],
            "javascript": [
                {
                    "pattern": r"^\s*function\s+(?P<name>\w+)\s*\([^)]*\)",
                    "type": "function",
                },
                {
                    "pattern": r"^\s*async\s+function\s+(?P<name>\w+)\s*\([^)]*\)",
                    "type": "async_function",
                },
                {
                    "pattern": r"^\s*(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
                    "type": "arrow_function",
                },
                {
                    "pattern": r"^\s*(?:const|let|var)\s+(?P<name>\w+)\s*=\s*function\s*\([^)]*\)",
                    "type": "function_expression",
                },
            ],
            "typescript": [
                {
                    "pattern": r"^\s*function\s+(?P<name>\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)",
                    "type": "function",
                },
                {
                    "pattern": r"^\s*async\s+function\s+(?P<name>\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)",
                    "type": "async_function",
                },
                {
                    "pattern": r"^\s*(?:const|let|var)\s+(?P<name>\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
                    "type": "arrow_function",
                },
                {
                    "pattern": r"^\s*(?:const|let|var)\s+(?P<name>\w+)\s*(?::\s*[^=]+)?\s*=\s*function\s*\([^)]*\)",
                    "type": "function_expression",
                },
            ],
        }
        return patterns.get(language, [])

    def _estimate_function_end(self, lines: list[str], start_idx: int, language: str) -> int:
        """
        Estimate the end line of a function.

        Simple heuristic based on indentation/braces.
        """
        if start_idx >= len(lines):
            return start_idx + 1

        if language == "python":
            # Find end by indentation
            start_line = lines[start_idx]
            base_indent = len(start_line) - len(start_line.lstrip())

            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                if not line.strip():  # Skip empty lines
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and line.strip():
                    return i  # Found line at same or lower indent
            return len(lines)
        else:
            # For brace-based languages, count braces
            brace_count = 0
            started = False

            for i in range(start_idx, len(lines)):
                line = lines[i]
                brace_count += line.count("{") - line.count("}")
                if "{" in line:
                    started = True
                if started and brace_count <= 0:
                    return i + 1

            return len(lines)

    def _verify_claim(
        self,
        claim: str,
        evidence: str | dict[str, str],
        threshold: float = 0.7,
    ) -> DeferredOperation:
        """
        Verify a claim against provided evidence.

        Implements: Spec SPEC-16.02

        Uses epistemic verification to check if a claim is supported by evidence:
        - Evidence support: Does the evidence support the claim?
        - Evidence dependence: Would the answer change without evidence?
        - Consistency: Is the claim internally consistent?

        Args:
            claim: The claim to verify
            evidence: Evidence to verify against (string or dict of evidence_id -> content)
            threshold: Support threshold for flagging (default 0.7)

        Returns:
            DeferredOperation that will resolve to ClaimVerification with:
            - evidence_support: Score from evidence auditing (0.0-1.0)
            - evidence_dependence: Score from consistency checking (0.0-1.0)
            - consistency_score: Internal consistency score (0.0-1.0)
            - is_flagged: Whether the claim is flagged for review
            - flag_reason: Reason for flagging (if applicable)

        Example:
            result = verify_claim(
                "The function returns 42",
                "def func(): return 42",
                threshold=0.7
            )
        """
        self._operation_counter += 1
        op_id = f"verify_{self._operation_counter}"

        # Normalize evidence to string
        if isinstance(evidence, dict):
            evidence_str = "\n\n".join(f"[{k}]:\n{v}" for k, v in evidence.items())
        else:
            evidence_str = str(evidence)

        op = DeferredOperation(
            operation_id=op_id,
            operation_type="verify_claim",
            query=claim,
            context=evidence_str,
            spawn_repl=False,
        )
        # Store threshold in metadata for orchestrator
        op.metadata = {"threshold": threshold, "claim": claim, "evidence": evidence_str}
        self.pending_operations.append(op)
        return op

    def _evidence_dependence(
        self,
        question: str,
        answer: str,
        evidence: str,
    ) -> DeferredOperation:
        """
        Compute how much an answer depends on the given evidence.

        Implements: Spec SPEC-16.04

        Uses consistency checking to determine if the answer would change
        significantly without the evidence (evidence scrubbing technique).

        Args:
            question: The question that was answered
            answer: The answer to evaluate
            evidence: The evidence the answer should depend on

        Returns:
            DeferredOperation that will resolve to float (0.0-1.0):
            - 0.0 = answer independent of evidence (potential hallucination)
            - 1.0 = answer fully dependent on evidence (good evidence use)

        Example:
            dependence = evidence_dependence(
                "What color is the widget?",
                "The widget is blue.",
                "According to the spec, widgets are blue."
            )
        """
        self._operation_counter += 1
        op_id = f"dep_{self._operation_counter}"

        # Combine question, answer, and evidence for context
        context = f"Question: {question}\n\nAnswer: {answer}\n\nEvidence: {evidence}"

        op = DeferredOperation(
            operation_id=op_id,
            operation_type="evidence_dependence",
            query=question,
            context=context,
            spawn_repl=False,
        )
        # Store components in metadata for orchestrator
        op.metadata = {"question": question, "answer": answer, "evidence": evidence}
        self.pending_operations.append(op)
        return op

    def _audit_reasoning(
        self,
        steps: list[dict[str, Any]],
        sources: dict[str, str],
    ) -> DeferredOperation:
        """
        Audit a reasoning trace for epistemic validity.

        Implements: Spec SPEC-16.03

        Verifies that each step in a reasoning trace properly cites its sources
        and that the claims are supported by the cited evidence.

        Args:
            steps: List of reasoning steps, each with:
                - "claim": The claim being made in this step
                - "cites": List of source span_ids supporting the claim
            sources: Dict mapping span_id -> content for all available sources

        Returns:
            DeferredOperation that will resolve to list of ClaimVerification,
            one for each step in the reasoning trace:
            - evidence_support: How well the cited sources support the claim
            - evidence_dependence: Whether the claim relies on the evidence
            - is_flagged: Whether the step is flagged for review
            - flag_reason: Why it was flagged (if applicable)

        Example:
            steps = [
                {"claim": "The function returns 42", "cites": ["src1"]},
                {"claim": "This matches the specification", "cites": ["src2"]},
            ]
            sources = {
                "src1": "def func(): return 42",
                "src2": "Spec: Function should return 42",
            }
            results = audit_reasoning(steps, sources)
        """
        self._operation_counter += 1
        op_id = f"audit_{self._operation_counter}"

        # Validate input structure
        validated_steps = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise TypeError(f"Step {i} must be a dict, got {type(step).__name__}")
            if "claim" not in step:
                raise ValueError(f"Step {i} missing required 'claim' key")
            claim = str(step.get("claim", ""))
            cites = step.get("cites", [])
            if not isinstance(cites, list):
                cites = [str(cites)]
            validated_steps.append({"claim": claim, "cites": cites})

        # Build context string with all sources
        sources_str = "\n\n".join(f"[{k}]:\n{v}" for k, v in sources.items())

        # Build query that describes the audit task
        query = f"Audit {len(validated_steps)} reasoning steps against provided sources"

        op = DeferredOperation(
            operation_id=op_id,
            operation_type="audit_reasoning",
            query=query,
            context=sources_str,
            spawn_repl=False,
        )
        # Store structured data in metadata for orchestrator
        op.metadata = {
            "steps": validated_steps,
            "sources": sources,
            "step_count": len(validated_steps),
        }
        self.pending_operations.append(op)
        return op

    def _detect_hallucinations(
        self,
        response: str,
        context: str | dict[str, str],
        support_threshold: float = 0.7,
        dependence_threshold: float = 0.3,
    ) -> DeferredOperation:
        """
        Detect hallucinations in a response by verifying claims against context.

        Implements: Spec SPEC-16.05

        This high-level function orchestrates the hallucination detection process:
        1. Extract claims from the response
        2. Map claims to evidence in the context
        3. Verify each claim against its evidence
        4. Return a HallucinationReport with results

        Args:
            response: The LLM response text to check for hallucinations
            context: Evidence context, either as string or dict of source_id -> content
            support_threshold: Minimum evidence support score (0.0-1.0, default 0.7)
            dependence_threshold: Minimum evidence dependence (0.0-1.0, default 0.3)

        Returns:
            DeferredOperation that will resolve to HallucinationReport with:
            - total_claims: Number of claims extracted
            - verified_claims: Number passing verification
            - flagged_claims: Number failing verification
            - claims: List of ClaimVerification for each claim
            - gaps: List of EpistemicGap for any issues found
            - overall_confidence: Aggregate confidence score
            - should_retry: Whether response should be regenerated

        Example:
            report = detect_hallucinations(
                response="The function returns 42 and handles errors gracefully.",
                context="def func(): return 42",
                support_threshold=0.7
            )
        """
        self._operation_counter += 1
        op_id = f"halluc_{self._operation_counter}"

        # Validate thresholds
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError(f"support_threshold must be between 0.0 and 1.0, got {support_threshold}")
        if not 0.0 <= dependence_threshold <= 1.0:
            raise ValueError(f"dependence_threshold must be between 0.0 and 1.0, got {dependence_threshold}")

        # Normalize context to string
        if isinstance(context, dict):
            context_str = "\n\n".join(f"[{k}]:\n{v}" for k, v in context.items())
            context_sources = context
        else:
            context_str = str(context)
            context_sources = {"main": context_str}

        # Build query for orchestrator
        query = "Detect hallucinations in response by extracting and verifying claims"

        op = DeferredOperation(
            operation_id=op_id,
            operation_type="detect_hallucinations",
            query=query,
            context=context_str,
            spawn_repl=False,
        )
        # Store all data needed for orchestrator processing
        op.metadata = {
            "response": response,
            "context_sources": context_sources,
            "support_threshold": support_threshold,
            "dependence_threshold": dependence_threshold,
        }
        self.pending_operations.append(op)
        return op

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

    def enable_memory(self, store: MemoryStore, session_id: str | None = None) -> None:
        """
        Enable memory functions in the REPL environment.

        Implements: Spec SPEC-02.27-31

        Args:
            store: MemoryStore instance to use for persistence
            session_id: Session ID for memory isolation (task/session tiers)
        """
        self._memory_store = store
        self._session_id = session_id

        # Add memory functions to globals
        self.globals["memory_query"] = self._memory_query
        self.globals["memory_add_fact"] = self._memory_add_fact
        self.globals["memory_add_experience"] = self._memory_add_experience
        self.globals["memory_get_context"] = self._memory_get_context
        self.globals["memory_relate"] = self._memory_relate

    def _memory_query(
        self, query: str, limit: int = 10, tier: str | None = None
    ) -> list[Any]:
        """
        Search for nodes matching a query.

        Implements: Spec SPEC-02.27

        Args:
            query: Search query string
            limit: Maximum number of results to return
            tier: Optional tier filter (task, session, longterm, archive)

        Returns:
            List of matching Node objects
        """
        if self._memory_store is None:
            return []

        # Query nodes with session isolation for task/session tiers
        all_nodes = self._memory_store.query_nodes(
            limit=limit * 3,
            tier=tier,
            session_id=self._session_id,
        )

        # Filter by query terms
        query_lower = query.lower()
        keywords = set(query_lower.split())

        matching = []
        for node in all_nodes:
            content_lower = node.content.lower()
            if any(kw in content_lower for kw in keywords):
                matching.append(node)

        # Sort by confidence descending and limit
        matching.sort(key=lambda n: n.confidence, reverse=True)
        return matching[:limit]

    def _memory_add_fact(
        self, content: str, confidence: float = 0.5, tier: str = "task"
    ) -> str:
        """
        Add a fact to memory.

        Implements: Spec SPEC-02.28

        Args:
            content: Fact content
            confidence: Confidence score (0.0 to 1.0)
            tier: Memory tier (task, session, longterm). Default is task.

        Returns:
            Node ID of created fact
        """
        if self._memory_store is None:
            raise RuntimeError("Memory not enabled. Call enable_memory() first.")

        # Include session_id in metadata for session isolation
        metadata = {}
        if self._session_id:
            metadata["session_id"] = self._session_id

        return self._memory_store.create_node(
            node_type="fact",
            content=content,
            confidence=confidence,
            tier=tier,
            metadata=metadata,
        )

    def _memory_add_experience(
        self, content: str, outcome: str, success: bool, tier: str = "session"
    ) -> str:
        """
        Add an experience to memory.

        Implements: Spec SPEC-02.29

        Args:
            content: Experience description
            outcome: Outcome description
            success: Whether the experience was successful
            tier: Memory tier (task, session, longterm). Default is session.

        Returns:
            Node ID of created experience
        """
        if self._memory_store is None:
            raise RuntimeError("Memory not enabled. Call enable_memory() first.")

        # Include session_id in metadata for session isolation
        metadata = {"outcome": outcome, "success": success}
        if self._session_id:
            metadata["session_id"] = self._session_id

        return self._memory_store.create_node(
            node_type="experience",
            content=content,
            tier=tier,
            metadata=metadata,
        )

    def _memory_get_context(self, limit: int = 10, tier: str | None = None) -> list[Any]:
        """
        Get recent/relevant context nodes.

        Implements: Spec SPEC-02.30

        Args:
            limit: Maximum number of nodes to return
            tier: Optional tier filter (task, session, longterm, archive)

        Returns:
            List of Node objects sorted by confidence
        """
        if self._memory_store is None:
            return []

        # Get nodes sorted by confidence (high to low) with session isolation
        nodes = self._memory_store.query_nodes(
            limit=limit * 2,
            tier=tier,
            session_id=self._session_id,
        )

        # Sort by confidence descending
        nodes.sort(key=lambda n: n.confidence, reverse=True)
        return nodes[:limit]

    def _memory_relate(self, label: str, node_id1: str, node_id2: str) -> str:
        """
        Create a relation between two nodes.

        Implements: Spec SPEC-02.31

        Args:
            label: Relation label (e.g., "implies", "causes", "connects")
            node_id1: First node ID
            node_id2: Second node ID

        Returns:
            Edge ID of created relation
        """
        if self._memory_store is None:
            raise RuntimeError("Memory not enabled. Call enable_memory() first.")

        return self._memory_store.create_edge(
            edge_type="relation",
            label=label,
            members=[
                {"node_id": node_id1, "role": "subject", "position": 0},
                {"node_id": node_id2, "role": "object", "position": 1},
            ],
        )

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

    def get_state(self) -> dict[str, Any]:
        """
        Get serializable state for checkpointing.

        Implements: SPEC-12.10 Error Recovery (checkpoint support)

        Returns:
            Dictionary containing:
            - working_memory: Current working memory contents
            - locals: Local variable names (not values, for security)
            - history_length: Number of executed commands
        """
        import copy as copy_module

        return {
            "working_memory": copy_module.deepcopy(self.globals.get("working_memory", {})),
            "locals": list(self.locals.keys()),
            "history_length": len(self.history),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """
        Restore state from a checkpoint.

        Implements: SPEC-12.10 Error Recovery (checkpoint support)

        Args:
            state: State dictionary from get_state()

        Note:
            This restores working_memory but clears locals for security.
            The execution history is not modified.
        """
        import copy as copy_module

        # Restore working memory
        if "working_memory" in state:
            self.globals["working_memory"] = copy_module.deepcopy(state["working_memory"])

        # Clear locals (can't safely restore arbitrary objects)
        self.locals.clear()


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


def create_micro_environment(
    context: SessionContext,
    use_restricted: bool = True,
) -> RLMEnvironment:
    """
    Create a micro-mode REPL environment with restricted functions.

    Implements: Spec SPEC-14.03-14.04

    Micro mode only exposes low-cost functions:
    - peek(var, start, end) - View context slice
    - search(var, pattern) - Pattern search (no LLM)
    - summarize_local(var, max_chars) - Truncation-based summary
    - memory_query(query) - Memory retrieval (if enabled)

    Functions NOT available in micro mode:
    - llm() - Recursive sub-queries
    - llm_batch() - Parallel sub-queries
    - map_reduce() - LLM-based aggregation

    Args:
        context: Session context to externalize
        use_restricted: Whether to use RestrictedPython (default True)

    Returns:
        RLMEnvironment configured for micro mode
    """
    return RLMEnvironment(
        context=context,
        recursive_handler=None,  # No recursive calls in micro mode
        use_restricted=use_restricted,
        access_level="micro",
    )


__all__ = [
    "RLMEnvironment",
    "RLMSecurityError",
    "ALLOWED_SUBPROCESSES",
    "MICRO_MODE_FUNCTIONS",
    "MICRO_MODE_BLOCKED",
    "REPLAccessLevel",
    "create_micro_environment",
    "typecheck_snippet",
    "lint_snippet",
]
