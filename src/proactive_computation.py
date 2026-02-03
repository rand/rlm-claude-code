"""
Proactive REPL computation for programmatic reasoning.

Implements: SPEC-06.01-06.05

Detects queries where REPL computation is more reliable than LLM reasoning
and provides computational helpers for common operations.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComputationTrigger(Enum):
    """
    Types of computation triggers.

    Implements: SPEC-06.02
    """

    ARITHMETIC = "arithmetic"
    COUNTING = "counting"
    SORTING = "sorting"
    FILTERING = "filtering"
    AGGREGATION = "aggregation"
    DATE_MATH = "date_math"
    STRING_OPS = "string_ops"


# Trigger detection patterns
TRIGGER_PATTERNS: dict[ComputationTrigger, list[re.Pattern[str]]] = {
    ComputationTrigger.ARITHMETIC: [
        re.compile(r"\b(calculate|compute|add|subtract|multiply|divide)\b", re.I),
        re.compile(r"\bwhat\s+is\s+\d+\s*[\+\-\*\/\^]", re.I),
        re.compile(r"\b\d+\s*[\+\-\*\/]\s*\d+", re.I),
        re.compile(r"\bresult\s+of\b", re.I),
        re.compile(r"\b\d+\s*\*\*\s*\d+", re.I),
    ],
    ComputationTrigger.COUNTING: [
        re.compile(r"\bhow\s+many\b", re.I),
        re.compile(r"\bcount\s+(the|of|all)?\b", re.I),
        re.compile(r"\bnumber\s+of\b", re.I),
        re.compile(r"\btotal\s+(number|count)?\b", re.I),
    ],
    ComputationTrigger.SORTING: [
        re.compile(r"\bsort(ed|ing)?\s+(by|the)?\b", re.I),
        re.compile(r"\border(ed|ing)?\s+(by|the)?\b", re.I),
        re.compile(r"\brank(ed|ing)?\b", re.I),
        re.compile(r"\btop\s+\d+\b", re.I),
        re.compile(r"\b(largest|smallest|biggest|highest|lowest)\b", re.I),
    ],
    ComputationTrigger.FILTERING: [
        re.compile(r"\bfilter(ed|ing)?\b", re.I),
        re.compile(r"\bwhere\s+\w+\s*(>|<|=|!=)", re.I),
        re.compile(r"\bmatching\b", re.I),
        re.compile(r"\bcontaining\b", re.I),
        re.compile(r"\bonly\s+(files|items|entries)\b", re.I),
    ],
    ComputationTrigger.AGGREGATION: [
        re.compile(r"\bsum\s+of\b", re.I),
        re.compile(r"\b(average|mean)\b", re.I),
        re.compile(r"\bmedian\b", re.I),
        re.compile(r"\b(max|min|maximum|minimum)\b", re.I),
        re.compile(r"\bstandard\s+deviation\b", re.I),
    ],
    ComputationTrigger.DATE_MATH: [
        re.compile(r"\bdays?\s+(since|ago|before|after)\b", re.I),
        re.compile(r"\bweeks?\s+(since|ago|before|after)\b", re.I),
        re.compile(r"\bmonths?\s+(since|ago|before|after)\b", re.I),
        re.compile(r"\blast\s+\d+\s+(days?|weeks?|months?)\b", re.I),
        re.compile(r"\b(before|after)\s+\d{4}-\d{2}-\d{2}\b", re.I),
        re.compile(r"\bmodified\s+(in|within)\s+the\s+last\b", re.I),
        re.compile(
            r"\b(before|after)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b",
            re.I,
        ),
    ],
    ComputationTrigger.STRING_OPS: [
        re.compile(r"\bextract\s+(the|all)?\b", re.I),
        re.compile(r"\bparse\s+(the|this)?\b", re.I),
        re.compile(r"\bsplit\s+(the|into|by)?\b", re.I),
        re.compile(r"\bformat\s+(this|as|the)?\b", re.I),
        re.compile(r"\bregex\b", re.I),
        re.compile(r"\bpattern\b", re.I),
    ],
}


@dataclass
class DetectionResult:
    """Result of computation trigger detection."""

    triggers: list[ComputationTrigger]
    confidence: float
    matched_patterns: dict[ComputationTrigger, list[str]] = field(default_factory=dict)


@dataclass
class ComputationSuggestion:
    """
    A suggestion for REPL computation.

    Implements: SPEC-06.03
    """

    triggers: list[ComputationTrigger]
    code: str
    explanation: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "triggers": [t.value for t in self.triggers],
            "code": self.code,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


class ComputationHelper:
    """
    Safe computational helpers for REPL.

    Implements: SPEC-06.04, SPEC-06.05
    """

    # Safe math functions allowed in calc()
    SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "sqrt": math.sqrt,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    # Safe constants allowed in calc()
    SAFE_CONSTANTS: dict[str, float] = {
        "pi": math.pi,
        "e": math.e,
    }

    # Safe binary operators
    SAFE_BINARY_OPS: dict[type, Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    # Safe unary operators
    SAFE_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def __init__(self, timeout_ms: int = 1000):
        """
        Initialize computation helper.

        Args:
            timeout_ms: Timeout for computations in milliseconds
        """
        self.timeout_ms = timeout_ms

    def calc(self, expression: str) -> Any:
        """
        Safely evaluate a mathematical expression.

        Implements: SPEC-06.04

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Result of the computation

        Raises:
            ValueError: If expression contains unsafe operations
        """
        # Check for obviously dangerous patterns
        dangerous_patterns = [
            "__",
            "import",
            "exec",
            "eval",
            "open",
            "file",
            "globals",
            "locals",
            "dir(",
            "type(",
            "getattr",
            "setattr",
            "delattr",
            "compile",
            "breakpoint",
        ]

        expr_lower = expression.lower()
        for pattern in dangerous_patterns:
            if pattern in expr_lower:
                raise ValueError(f"Unsafe expression: contains '{pattern}'")

        # Check for resource exhaustion
        if "**" in expression:
            # Check for chained exponentiation (e.g., 10**10**10)
            if expression.count("**") >= 2:
                raise ValueError("Chained exponentiation not allowed")
            # Check for huge exponents
            match = re.search(r"(\d+)\s*\*\*\s*(\d+)", expression)
            if match:
                base, exp = int(match.group(1)), int(match.group(2))
                # Reject if result would be very large
                if base > 2 and exp > 30:
                    raise ValueError("Expression too large to compute safely")
                if base > 10 and exp > 20:
                    raise ValueError("Expression too large to compute safely")

        try:
            tree = ast.parse(expression, mode="eval")
            return self._eval_node(tree.body)
        except (SyntaxError, TypeError) as e:
            raise ValueError(f"Invalid expression: {e}")

    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.SAFE_BINARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.SAFE_UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(operand)

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.SAFE_FUNCTIONS:
                    args = [self._eval_node(arg) for arg in node.args]
                    return self.SAFE_FUNCTIONS[func_name](*args)
                raise ValueError(f"Unknown function: {func_name}")
            raise ValueError("Complex function calls not allowed")

        if isinstance(node, ast.Name):
            # Check constants first
            if node.id in self.SAFE_CONSTANTS:
                return self.SAFE_CONSTANTS[node.id]
            # Then functions (as references, not calls)
            if node.id in self.SAFE_FUNCTIONS:
                return self.SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Unknown name: {node.id}")

        if isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def stats(self, data: list[float | int]) -> dict[str, float]:
        """
        Compute statistical measures for data.

        Implements: SPEC-06.04

        Args:
            data: List of numeric values

        Returns:
            Dictionary of statistical measures
        """
        if not data:
            return {
                "count": 0,
                "sum": 0,
                "mean": 0,
                "min": 0,
                "max": 0,
                "median": 0,
                "std": 0,
            }

        n = len(data)
        total = sum(data)
        mean = total / n

        return {
            "count": n,
            "sum": total,
            "mean": mean,
            "min": min(data),
            "max": max(data),
            "median": statistics.median(data),
            "std": statistics.stdev(data) if n > 1 else 0,
        }

    def group_by(
        self,
        data: list[Any],
        key: str | Callable[[Any], Any],
    ) -> dict[Any, list[Any]]:
        """
        Group data by a key.

        Implements: SPEC-06.04

        Args:
            data: List of items to group
            key: Key name (for dicts) or function

        Returns:
            Dictionary mapping keys to lists of items
        """
        result: dict[Any, list[Any]] = {}

        for item in data:
            if callable(key):
                k = key(item)
            elif isinstance(key, str):
                if isinstance(item, dict):
                    k = item.get(key)
                else:
                    k = getattr(item, key, None)
            else:
                k = None

            if k not in result:
                result[k] = []
            result[k].append(item)

        return result

    def find_imports(self, content: str) -> list[str]:
        """
        Find all imports in Python code.

        Implements: SPEC-06.04

        Args:
            content: Python source code

        Returns:
            List of imported module names
        """
        imports: set[str] = set()

        # import X, from X import Y
        import_pattern = re.compile(
            r"^\s*(?:from\s+(\S+)|import\s+(\S+))",
            re.MULTILINE,
        )

        for match in import_pattern.finditer(content):
            module = match.group(1) or match.group(2)
            if module:
                # Get base module (before any dots)
                base = module.split(".")[0]
                # Handle 'import X as Y' and 'import X, Y, Z'
                base = base.split(",")[0].strip()
                # Remove 'as alias' if present
                parts = base.split()
                if parts:
                    base = parts[0]
                if base and not base.startswith("."):
                    imports.add(base)

        return sorted(imports)


class ComputationAdvisor:
    """
    Advises when to use REPL computation.

    Implements: SPEC-06.01-06.03
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        helper: ComputationHelper | None = None,
    ):
        """
        Initialize computation advisor.

        Args:
            confidence_threshold: Minimum confidence to suggest REPL
            helper: Computation helper instance
        """
        self.confidence_threshold = confidence_threshold
        self.helper = helper or ComputationHelper()

    def detect_triggers(self, query: str) -> DetectionResult:
        """
        Detect computation triggers in a query.

        Implements: SPEC-06.01

        Args:
            query: User query text

        Returns:
            DetectionResult with triggers and confidence
        """
        triggers: list[ComputationTrigger] = []
        matched: dict[ComputationTrigger, list[str]] = {}

        for trigger, patterns in TRIGGER_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(query)
                if match:
                    if trigger not in triggers:
                        triggers.append(trigger)
                    if trigger not in matched:
                        matched[trigger] = []
                    matched[trigger].append(match.group())

        # Calculate confidence based on number and clarity of matches
        if not triggers:
            confidence = 0.0
        else:
            # Base confidence from having triggers
            confidence = 0.6

            # More triggers = higher confidence
            confidence += min(0.25, len(triggers) * 0.1)

            # More pattern matches = higher confidence
            total_matches = sum(len(m) for m in matched.values())
            confidence += min(0.15, total_matches * 0.075)

            confidence = min(1.0, confidence)

        return DetectionResult(
            triggers=triggers,
            confidence=confidence,
            matched_patterns=matched,
        )

    def should_use_repl(self, query: str) -> bool:
        """
        Check if REPL computation should be used.

        Args:
            query: User query

        Returns:
            True if REPL computation is recommended
        """
        result = self.detect_triggers(query)
        return result.confidence >= self.confidence_threshold

    def suggest_computation(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> ComputationSuggestion | None:
        """
        Generate a computation suggestion for a query.

        Implements: SPEC-06.03

        Args:
            query: User query
            context: Optional context with variables

        Returns:
            ComputationSuggestion or None
        """
        result = self.detect_triggers(query)

        if not result.triggers:
            return None

        # Generate code based on primary trigger
        primary = result.triggers[0]
        code, explanation = self._generate_code(primary, query, context)

        return ComputationSuggestion(
            triggers=result.triggers,
            code=code,
            explanation=explanation,
            confidence=result.confidence,
        )

    def _generate_code(
        self,
        trigger: ComputationTrigger,
        query: str,
        context: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Generate code and explanation for a trigger."""
        context = context or {}

        if trigger == ComputationTrigger.ARITHMETIC:
            # Extract numbers and operations from query
            numbers = re.findall(r"\d+(?:\.\d+)?", query)
            if len(numbers) >= 2:
                expr = query
                # Try to find the expression
                match = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/^])\s*(\d+(?:\.\d+)?)", query)
                if match:
                    a, op, b = match.groups()
                    op = "**" if op == "^" else op
                    code = f"calc('{a} {op} {b}')"
                    return code, f"Computing {a} {op} {b} using safe math evaluation"
            return "calc('...')", "Arithmetic computation using calc()"

        if trigger == ComputationTrigger.COUNTING:
            if "list" in context:
                return "len(list)", "Counting items in the list"
            return "len(data)", "Counting items using len()"

        if trigger == ComputationTrigger.SORTING:
            # Check for "top N" pattern
            match = re.search(r"top\s+(\d+)", query, re.I)
            if match:
                n = match.group(1)
                return f"sorted(data, reverse=True)[:{n}]", f"Getting top {n} items"
            return "sorted(data)", "Sorting data"

        if trigger == ComputationTrigger.AGGREGATION:
            if "values" in context or "data" in context:
                return "stats(values)", "Computing statistics on values"
            return "stats(data)", "Computing statistics"

        if trigger == ComputationTrigger.FILTERING:
            return "filter(condition, data)", "Filtering data based on condition"

        if trigger == ComputationTrigger.DATE_MATH:
            return "date_diff(start, end)", "Computing date difference"

        if trigger == ComputationTrigger.STRING_OPS:
            return "extract_pattern(text, pattern)", "Extracting from text"

        return "# computation", "Generic computation"

    def execute_suggestion(
        self,
        suggestion: ComputationSuggestion,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute a computation suggestion.

        Args:
            suggestion: The suggestion to execute
            context: Variables for execution

        Returns:
            Result of computation
        """
        context = context or {}
        primary = suggestion.triggers[0] if suggestion.triggers else None

        # Handle specific cases
        if primary == ComputationTrigger.ARITHMETIC:
            # Extract expression from calc('...')
            match = re.search(r"calc\('([^']+)'\)", suggestion.code)
            if match:
                return self.helper.calc(match.group(1))

        if primary == ComputationTrigger.AGGREGATION:
            data = context.get("data") or context.get("values", [])
            return self.helper.stats(data)

        if primary == ComputationTrigger.COUNTING:
            data = context.get("list") or context.get("data", [])
            return len(data)

        # Generic evaluation for simple cases
        if suggestion.code.startswith("calc("):
            match = re.search(r"calc\('([^']+)'\)", suggestion.code)
            if match:
                return self.helper.calc(match.group(1))

        return None


__all__ = [
    "ComputationAdvisor",
    "ComputationHelper",
    "ComputationSuggestion",
    "ComputationTrigger",
    "DetectionResult",
]
