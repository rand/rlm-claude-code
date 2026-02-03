"""
Programmatic context enrichment.

Implements: SPEC-06.30-06.35

Proactively enriches context before LLM reasoning based on
query intent classification and task-specific gathering strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(Enum):
    """
    Query intent classification.

    Implements: SPEC-06.31
    """

    CODE_TASK = "code_task"
    DEBUG_TASK = "debug_task"
    ANALYSIS_TASK = "analysis_task"
    QUESTION = "question"


@dataclass
class EnrichmentConfig:
    """Configuration for context enrichment."""

    max_tokens: int = 8000
    include_types: bool = True
    include_tests: bool = True
    include_changes: bool = True
    context_lines: int = 20


@dataclass
class EnrichmentStrategy:
    """
    Strategy for gathering enrichment data.

    Implements: SPEC-06.31
    """

    intent: QueryIntent
    gather_types: list[str]
    priority_order: list[str] = field(default_factory=list)

    @classmethod
    def for_intent(cls, intent: QueryIntent) -> EnrichmentStrategy:
        """
        Get enrichment strategy for an intent.

        Implements: SPEC-06.31
        """
        strategies = {
            QueryIntent.CODE_TASK: cls(
                intent=intent,
                gather_types=["dependencies", "types", "tests", "recent_changes"],
                priority_order=["types", "dependencies", "tests", "recent_changes"],
            ),
            QueryIntent.DEBUG_TASK: cls(
                intent=intent,
                gather_types=["error_context", "blame", "similar_experiences"],
                priority_order=["error_context", "blame", "similar_experiences"],
            ),
            QueryIntent.ANALYSIS_TASK: cls(
                intent=intent,
                gather_types=["documentation", "examples", "related_code"],
                priority_order=["documentation", "examples", "related_code"],
            ),
            QueryIntent.QUESTION: cls(
                intent=intent,
                gather_types=["memories", "facts", "documentation"],
                priority_order=["facts", "memories", "documentation"],
            ),
        }
        return strategies.get(intent, cls(intent=intent, gather_types=[]))


@dataclass
class EnrichmentResult:
    """
    Result of context enrichment.

    Implements: SPEC-06.35
    """

    enriched_context: dict[str, Any]
    reasoning: str
    additions: list[str]
    detected_intent: QueryIntent
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enriched_context": self.enriched_context,
            "reasoning": self.reasoning,
            "additions": self.additions,
            "detected_intent": self.detected_intent.value,
            "token_count": self.token_count,
        }


class IntentClassifier:
    """
    Classifies query intent for enrichment strategy selection.

    Implements: SPEC-06.31
    """

    def __init__(self) -> None:
        """Initialize classifier with patterns."""
        self.patterns = {
            QueryIntent.CODE_TASK: [
                r"\b(implement|add|create|write|build|develop)\b",
                r"\b(method|function|class|feature|component)\b",
                r"\b(refactor|update|modify|change)\b",
            ],
            QueryIntent.DEBUG_TASK: [
                r"\b(debug|fix|error|bug|crash|exception)\b",
                r"\b(why|failing|broken|wrong|issue)\b",
                r"\b(traceback|stacktrace|null|undefined)\b",
            ],
            QueryIntent.ANALYSIS_TASK: [
                r"\b(analyze|evaluate|assess|review|examine)\b",
                r"\b(performance|complexity|quality|coverage)\b",
                r"\b(compare|contrast|difference)\b",
            ],
            QueryIntent.QUESTION: [
                r"^(what|how|why|when|where|who)\b",
                r"\b(explain|describe|tell me|purpose)\b",
                r"\?$",
            ],
        }

    def classify(self, query: str) -> QueryIntent:
        """
        Classify query into intent category.

        Args:
            query: User query

        Returns:
            Detected QueryIntent
        """
        query_lower = query.lower()
        scores: dict[QueryIntent, int] = dict.fromkeys(QueryIntent, 0)

        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    scores[intent] += 1

        # Prioritize task intents over QUESTION (debug/code/analysis are more specific)
        # Give bonus to task-specific intents to break ties with question patterns
        for intent in [QueryIntent.DEBUG_TASK, QueryIntent.CODE_TASK, QueryIntent.ANALYSIS_TASK]:
            if scores[intent] > 0:
                scores[intent] += 1  # Bonus for specificity

        # Return highest scoring intent
        best_intent = max(scores, key=lambda i: scores[i])

        # Default to QUESTION if no matches
        if scores[best_intent] == 0:
            return QueryIntent.QUESTION

        return best_intent


class CodeTaskEnricher:
    """
    Enricher for code-related tasks.

    Implements: SPEC-06.32
    """

    def gather_imports(self, content: str) -> dict[str, Any]:
        """
        Gather import graph from content.

        Implements: SPEC-06.32 - Import graph (local dependencies only)

        Args:
            content: Source code content

        Returns:
            Dictionary of imports with locality info
        """
        imports: dict[str, Any] = {}

        # Match import statements
        import_patterns = [
            (r"^from\s+(\.+\w*(?:\.\w+)*)\s+import", True),  # Relative imports
            (r"^from\s+(\w+(?:\.\w+)*)\s+import", False),  # Absolute imports
            (r"^import\s+(\w+(?:\.\w+)*)", False),  # Simple imports
        ]

        for line in content.split("\n"):
            line = line.strip()
            for pattern, is_local in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1)
                    imports[module] = {
                        "local": is_local or module.startswith("."),
                        "line": line,
                    }

        return imports

    def gather_types(self, content: str) -> dict[str, Any]:
        """
        Gather type definitions from content.

        Implements: SPEC-06.32 - Type definitions

        Args:
            content: Source code content

        Returns:
            Dictionary of type definitions
        """
        types: dict[str, Any] = {}

        # Match class definitions
        class_pattern = r"class\s+(\w+)(?:\([^)]*\))?:"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            # Find class body (simplified)
            start = match.end()
            types[class_name] = {"kind": "class", "position": match.start()}

        # Match type annotations
        annotation_pattern = r"(\w+)\s*:\s*(\w+(?:\[[\w\[\], ]+\])?)"
        for match in re.finditer(annotation_pattern, content):
            var_name, type_name = match.groups()
            if var_name not in types:
                types[var_name] = {"kind": "annotation", "type": type_name}

        return types

    def find_related_tests(self, file_path: str) -> list[str]:
        """
        Find test files related to source file.

        Implements: SPEC-06.32 - Related test files

        Args:
            file_path: Path to source file

        Returns:
            List of potential test file paths
        """
        tests = []

        # Extract filename without extension
        parts = file_path.replace("\\", "/").split("/")
        filename = parts[-1] if parts else file_path

        if filename.endswith(".py"):
            base = filename[:-3]

            # Common test file patterns
            tests.extend(
                [
                    f"tests/test_{base}.py",
                    f"tests/unit/test_{base}.py",
                    f"test_{base}.py",
                    f"tests/{base}_test.py",
                ]
            )

            # If in src/, check parallel tests/
            if "src/" in file_path:
                test_path = file_path.replace("src/", "tests/").replace(
                    f"{base}.py", f"test_{base}.py"
                )
                tests.append(test_path)

        return tests

    def gather_recent_changes(self, file_path: str) -> list[dict[str, Any]]:
        """
        Gather recent git changes for file.

        Implements: SPEC-06.32 - Recent git changes

        Args:
            file_path: Path to file

        Returns:
            List of recent changes (mock implementation)
        """
        # Mock implementation - in production, use git log
        return []


class DebugTaskEnricher:
    """
    Enricher for debugging tasks.

    Implements: SPEC-06.33
    """

    def parse_stack_locations(self, stack_trace: str) -> list[dict[str, Any]]:
        """
        Parse locations from stack trace.

        Implements: SPEC-06.33 - Parse error stack trace locations

        Args:
            stack_trace: Error stack trace

        Returns:
            List of location dictionaries
        """
        locations = []

        # Match Python stack trace format
        pattern = r'File "([^"]+)", line (\d+)(?:, in (\w+))?'

        for match in re.finditer(pattern, stack_trace):
            file_path, line_num, func_name = match.groups()
            locations.append(
                {
                    "file": file_path,
                    "line": int(line_num),
                    "function": func_name,
                }
            )

        return locations

    def get_context_window(self) -> dict[str, int]:
        """
        Get context window configuration.

        Implements: SPEC-06.33 - ±20 lines context

        Returns:
            Context window settings
        """
        return {
            "lines_before": 20,
            "lines_after": 20,
        }

    def gather_blame(self, file_path: str, line: int) -> dict[str, Any]:
        """
        Gather git blame for location.

        Implements: SPEC-06.33 - Git blame for error locations

        Args:
            file_path: Path to file
            line: Line number

        Returns:
            Blame information (mock implementation)
        """
        # Mock implementation - in production, use git blame
        return {
            "file": file_path,
            "line": line,
            "author": "unknown",
            "commit": "unknown",
        }

    def find_similar_experiences(
        self,
        error_type: str,
        context: str,
    ) -> list[dict[str, Any]]:
        """
        Find similar debugging experiences from memory.

        Implements: SPEC-06.33 - Similar past debugging experiences

        Args:
            error_type: Type of error
            context: Error context

        Returns:
            List of similar experiences (mock implementation)
        """
        # Mock implementation - in production, query memory store
        return []


class ContextEnricher:
    """
    Main context enricher.

    Implements: SPEC-06.30-06.35
    """

    def __init__(self, config: EnrichmentConfig | None = None) -> None:
        """
        Initialize context enricher.

        Args:
            config: Enrichment configuration
        """
        self.config = config or EnrichmentConfig()
        self.classifier = IntentClassifier()
        self.code_enricher = CodeTaskEnricher()
        self.debug_enricher = DebugTaskEnricher()

    def enrich(
        self,
        query: str,
        context: dict[str, Any],
    ) -> EnrichmentResult:
        """
        Enrich context before LLM reasoning.

        Implements: SPEC-06.30

        Args:
            query: User query
            context: Original context

        Returns:
            EnrichmentResult with enriched context
        """
        # Classify intent
        intent = self.classifier.classify(query)

        # Get strategy for intent
        strategy = EnrichmentStrategy.for_intent(intent)

        # Gather enrichment data
        enriched = dict(context)
        additions: list[str] = []
        reasoning_parts: list[str] = [f"Detected intent: {intent.value}"]

        # Apply strategy-specific gathering
        if intent == QueryIntent.CODE_TASK:
            self._enrich_code_task(enriched, context, additions, reasoning_parts)
        elif intent == QueryIntent.DEBUG_TASK:
            self._enrich_debug_task(enriched, context, additions, reasoning_parts)
        elif intent == QueryIntent.ANALYSIS_TASK:
            self._enrich_analysis_task(enriched, context, additions, reasoning_parts)
        else:
            self._enrich_question(enriched, context, additions, reasoning_parts)

        # Calculate token count (rough estimate)
        token_count = self._estimate_tokens(enriched)

        # Truncate if needed
        if token_count > self.config.max_tokens:
            enriched, token_count = self._truncate_to_budget(enriched, self.config.max_tokens)
            reasoning_parts.append(f"Truncated to {self.config.max_tokens} tokens")

        return EnrichmentResult(
            enriched_context=enriched,
            reasoning=". ".join(reasoning_parts),
            additions=additions,
            detected_intent=intent,
            token_count=token_count,
        )

    def _enrich_code_task(
        self,
        enriched: dict[str, Any],
        context: dict[str, Any],
        additions: list[str],
        reasoning: list[str],
    ) -> None:
        """Enrich for code task."""
        content = context.get("content", "")
        file_path = context.get("file", "")

        if content and self.config.include_types:
            types = self.code_enricher.gather_types(content)
            if types:
                enriched["types"] = types
                additions.append("types")
                reasoning.append(f"Added {len(types)} type definitions")

            imports = self.code_enricher.gather_imports(content)
            if imports:
                enriched["imports"] = imports
                additions.append("dependencies")
                reasoning.append(f"Added {len(imports)} imports")

        if file_path and self.config.include_tests:
            tests = self.code_enricher.find_related_tests(file_path)
            if tests:
                enriched["related_tests"] = tests
                additions.append("tests")
                reasoning.append(f"Found {len(tests)} potential test files")

    def _enrich_debug_task(
        self,
        enriched: dict[str, Any],
        context: dict[str, Any],
        additions: list[str],
        reasoning: list[str],
    ) -> None:
        """Enrich for debug task."""
        error = context.get("error", "")
        stack_trace = context.get("stack_trace", error)

        if stack_trace:
            locations = self.debug_enricher.parse_stack_locations(stack_trace)
            if locations:
                enriched["error_locations"] = locations
                additions.append("error_context")
                reasoning.append(f"Parsed {len(locations)} stack locations")

                # Add context window config
                enriched["context_window"] = self.debug_enricher.get_context_window()

    def _enrich_analysis_task(
        self,
        enriched: dict[str, Any],
        context: dict[str, Any],
        additions: list[str],
        reasoning: list[str],
    ) -> None:
        """Enrich for analysis task."""
        # Add documentation hints
        enriched["documentation_hints"] = [
            "Check docstrings",
            "Review README",
            "Look for examples in tests",
        ]
        additions.append("documentation")
        reasoning.append("Added documentation hints")

    def _enrich_question(
        self,
        enriched: dict[str, Any],
        context: dict[str, Any],
        additions: list[str],
        reasoning: list[str],
    ) -> None:
        """Enrich for question."""
        # Add memory/fact retrieval hints
        enriched["retrieval_hints"] = {
            "search_memories": True,
            "search_facts": True,
        }
        additions.append("memories")
        additions.append("facts")
        reasoning.append("Enabled memory and fact retrieval")

    def _estimate_tokens(self, data: dict[str, Any]) -> int:
        """Estimate token count for data."""
        # Rough estimate: 1.3 tokens per word
        text = str(data)
        words = len(text.split())
        return int(words * 1.3)

    def _truncate_to_budget(
        self,
        data: dict[str, Any],
        max_tokens: int,
    ) -> tuple[dict[str, Any], int]:
        """Truncate data to fit token budget."""
        current_tokens = self._estimate_tokens(data)

        if current_tokens <= max_tokens:
            return data, current_tokens

        # Prioritize keeping query-relevant content
        truncated: dict[str, Any] = {}
        tokens = 0

        # Keep essential fields first, truncating if needed
        essential = ["query", "file", "content", "error"]
        budget_per_field = max_tokens // max(len(essential), 1)

        for key in essential:
            if key in data:
                value = data[key]
                field_tokens = self._estimate_tokens({key: value})

                if field_tokens > budget_per_field and isinstance(value, str):
                    # Truncate string to fit budget
                    chars_per_token = len(value) / max(field_tokens, 1)
                    max_chars = int(budget_per_field * chars_per_token)
                    value = value[:max_chars] + "..."
                    field_tokens = self._estimate_tokens({key: value})

                if tokens + field_tokens <= max_tokens:
                    truncated[key] = value
                    tokens += field_tokens

        # Add other fields if budget allows
        for key, value in data.items():
            if key not in essential:
                field_tokens = self._estimate_tokens({key: value})
                if tokens + field_tokens <= max_tokens:
                    truncated[key] = value
                    tokens += field_tokens

        return truncated, tokens


__all__ = [
    "CodeTaskEnricher",
    "ContextEnricher",
    "DebugTaskEnricher",
    "EnrichmentConfig",
    "EnrichmentResult",
    "EnrichmentStrategy",
    "IntentClassifier",
    "QueryIntent",
]
