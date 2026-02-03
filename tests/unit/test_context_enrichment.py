"""
Tests for programmatic context enrichment (SPEC-06.30-06.35).

Tests cover:
- Proactive context enrichment
- Intent classification
- Enrichment strategies for different task types
- Token budget respect
- Enrichment logging
"""

from src.context_enrichment import (
    CodeTaskEnricher,
    ContextEnricher,
    DebugTaskEnricher,
    EnrichmentConfig,
    EnrichmentResult,
    EnrichmentStrategy,
    IntentClassifier,
    QueryIntent,
)


class TestProactiveEnrichment:
    """Tests for proactive context enrichment (SPEC-06.30)."""

    def test_enricher_adds_context_before_reasoning(self):
        """SPEC-06.30: System enriches context before LLM reasoning."""
        enricher = ContextEnricher()

        result = enricher.enrich(
            query="What does this function do?",
            context={"file": "test.py", "content": "def foo(): pass"},
        )

        assert isinstance(result, EnrichmentResult)
        assert result.enriched_context is not None
        assert len(result.enriched_context) > 0

    def test_enricher_returns_original_plus_additions(self):
        """Enriched context should include original plus additions."""
        enricher = ContextEnricher()

        original = {"query": "Explain this code"}
        result = enricher.enrich("Explain this code", context=original)

        # Should have additions
        assert "additions" in result.enriched_context or len(result.enriched_context) > len(
            original
        )


class TestIntentClassification:
    """Tests for intent classification (SPEC-06.31)."""

    def test_classifies_code_task(self):
        """SPEC-06.31: Classify code tasks."""
        classifier = IntentClassifier()

        intent = classifier.classify("Add a new method to handle authentication")

        assert intent == QueryIntent.CODE_TASK

    def test_classifies_debug_task(self):
        """SPEC-06.31: Classify debug tasks."""
        classifier = IntentClassifier()

        intent = classifier.classify("Why is this throwing a NullPointerException?")

        assert intent == QueryIntent.DEBUG_TASK

    def test_classifies_analysis_task(self):
        """SPEC-06.31: Classify analysis tasks."""
        classifier = IntentClassifier()

        intent = classifier.classify("Analyze the performance of this algorithm")

        assert intent == QueryIntent.ANALYSIS_TASK

    def test_classifies_question(self):
        """SPEC-06.31: Classify questions."""
        classifier = IntentClassifier()

        intent = classifier.classify("What is the purpose of this module?")

        assert intent == QueryIntent.QUESTION


class TestEnrichmentStrategies:
    """Tests for enrichment strategies (SPEC-06.31)."""

    def test_code_task_strategy(self):
        """SPEC-06.31: Code task adds dependencies, types, tests, changes."""
        strategy = EnrichmentStrategy.for_intent(QueryIntent.CODE_TASK)

        assert "dependencies" in strategy.gather_types
        assert "types" in strategy.gather_types
        assert "tests" in strategy.gather_types
        assert "recent_changes" in strategy.gather_types

    def test_debug_task_strategy(self):
        """SPEC-06.31: Debug task adds error context, blame, experiences."""
        strategy = EnrichmentStrategy.for_intent(QueryIntent.DEBUG_TASK)

        assert "error_context" in strategy.gather_types
        assert "blame" in strategy.gather_types
        assert "similar_experiences" in strategy.gather_types

    def test_analysis_task_strategy(self):
        """SPEC-06.31: Analysis task adds documentation, examples."""
        strategy = EnrichmentStrategy.for_intent(QueryIntent.ANALYSIS_TASK)

        assert "documentation" in strategy.gather_types
        assert "examples" in strategy.gather_types

    def test_question_strategy(self):
        """SPEC-06.31: Question adds memories and facts."""
        strategy = EnrichmentStrategy.for_intent(QueryIntent.QUESTION)

        assert "memories" in strategy.gather_types
        assert "facts" in strategy.gather_types


class TestCodeTaskEnrichment:
    """Tests for code task enrichment (SPEC-06.32)."""

    def test_gathers_import_graph(self):
        """SPEC-06.32: Gather import graph (local dependencies)."""
        enricher = CodeTaskEnricher()

        content = """
import os
from pathlib import Path
from .utils import helper
from ..common import shared
"""
        result = enricher.gather_imports(content)

        assert "utils" in result or ".utils" in str(result)
        assert "common" in result or "..common" in str(result)
        # Should only include local imports
        assert "os" not in result or result.get("os", {}).get("local", False) is False

    def test_gathers_type_definitions(self):
        """SPEC-06.32: Gather type definitions."""
        enricher = CodeTaskEnricher()

        content = """
class UserModel:
    name: str
    age: int

def process(user: UserModel) -> str:
    pass
"""
        result = enricher.gather_types(content)

        assert "UserModel" in result
        assert "str" in str(result) or "name" in str(result)

    def test_finds_related_test_files(self):
        """SPEC-06.32: Find related test files."""
        enricher = CodeTaskEnricher()

        # Given a source file path
        result = enricher.find_related_tests("src/models/user.py")

        # Should suggest test file locations
        assert any("test" in path.lower() for path in result)

    def test_gathers_recent_changes(self):
        """SPEC-06.32: Gather recent git changes."""
        enricher = CodeTaskEnricher()

        # Mock file path
        result = enricher.gather_recent_changes("src/example.py")

        # Should return list of changes (may be empty)
        assert isinstance(result, list)


class TestDebugTaskEnrichment:
    """Tests for debug task enrichment (SPEC-06.33)."""

    def test_parses_stack_trace_locations(self):
        """SPEC-06.33: Parse error stack trace locations."""
        enricher = DebugTaskEnricher()

        stack_trace = """
Traceback (most recent call last):
  File "/app/src/handler.py", line 42, in process
    result = compute(data)
  File "/app/src/compute.py", line 15, in compute
    return value / divisor
ZeroDivisionError: division by zero
"""
        locations = enricher.parse_stack_locations(stack_trace)

        assert len(locations) >= 2
        assert any("handler.py" in loc["file"] for loc in locations)
        assert any(loc["line"] == 42 for loc in locations)

    def test_provides_context_around_error(self):
        """SPEC-06.33: ±20 lines context around error location."""
        enricher = DebugTaskEnricher()

        # Should provide context window
        config = enricher.get_context_window()
        assert config["lines_before"] >= 20
        assert config["lines_after"] >= 20

    def test_gathers_git_blame(self):
        """SPEC-06.33: Git blame for error locations."""
        enricher = DebugTaskEnricher()

        result = enricher.gather_blame("src/example.py", line=42)

        # Should return blame info (may be mock)
        assert isinstance(result, dict)

    def test_finds_similar_experiences(self):
        """SPEC-06.33: Find similar debugging experiences from memory."""
        enricher = DebugTaskEnricher()

        result = enricher.find_similar_experiences(
            error_type="ZeroDivisionError",
            context="division operation",
        )

        # Should return list of experiences
        assert isinstance(result, list)


class TestTokenBudget:
    """Tests for token budget respect (SPEC-06.34)."""

    def test_enrichment_respects_token_limit(self):
        """SPEC-06.34: Enrichment respects token budgets."""
        config = EnrichmentConfig(max_tokens=1000)
        enricher = ContextEnricher(config=config)

        # Provide large context
        large_content = "x " * 5000  # Very large
        result = enricher.enrich(
            query="Analyze this",
            context={"content": large_content},
        )

        # Result should be truncated
        assert result.token_count <= config.max_tokens

    def test_truncation_preserves_relevance(self):
        """Truncation should preserve most relevant content."""
        config = EnrichmentConfig(max_tokens=500)
        enricher = ContextEnricher(config=config)

        result = enricher.enrich(
            query="Implement the main function",  # Code task to preserve content
            context={"content": "def main(): pass\n" + "# comment\n" * 1000},
        )

        # Main content should be preserved (content is essential for code tasks)
        assert (
            "content" in result.enriched_context or "main" in str(result.enriched_context).lower()
        )

    def test_default_token_budget(self):
        """Default token budget should be reasonable."""
        config = EnrichmentConfig()

        # Should have sensible default (e.g., 4000-8000 tokens)
        assert 1000 <= config.max_tokens <= 16000


class TestEnrichmentLogging:
    """Tests for enrichment logging (SPEC-06.35)."""

    def test_logs_enrichment_reasoning(self):
        """SPEC-06.35: Enrichment reasoning is logged."""
        enricher = ContextEnricher()

        result = enricher.enrich(
            query="Debug this error",
            context={"error": "NullPointerException"},
        )

        # Should have reasoning log
        assert result.reasoning is not None
        assert len(result.reasoning) > 0

    def test_logs_what_was_added(self):
        """Log should indicate what was added."""
        enricher = ContextEnricher()

        result = enricher.enrich(
            query="Implement feature X",
            context={},
        )

        # Should list additions
        assert result.additions is not None
        assert isinstance(result.additions, list)

    def test_logs_intent_classification(self):
        """Log should include intent classification."""
        enricher = ContextEnricher()

        result = enricher.enrich(
            query="Fix the bug in authentication",
            context={},
        )

        # Should include detected intent
        assert result.detected_intent is not None


class TestEnrichmentResult:
    """Tests for EnrichmentResult structure."""

    def test_result_has_enriched_context(self):
        """Result should have enriched context."""
        result = EnrichmentResult(
            enriched_context={"original": "data", "added": "more"},
            reasoning="Added related context",
            additions=["types", "tests"],
            detected_intent=QueryIntent.CODE_TASK,
            token_count=500,
        )

        assert result.enriched_context is not None

    def test_result_has_token_count(self):
        """Result should track token count."""
        result = EnrichmentResult(
            enriched_context={},
            reasoning="",
            additions=[],
            detected_intent=QueryIntent.QUESTION,
            token_count=250,
        )

        assert result.token_count == 250

    def test_result_to_dict(self):
        """Result should be serializable."""
        result = EnrichmentResult(
            enriched_context={"data": "value"},
            reasoning="Test reasoning",
            additions=["item1"],
            detected_intent=QueryIntent.DEBUG_TASK,
            token_count=100,
        )

        data = result.to_dict()
        assert "enriched_context" in data
        assert "reasoning" in data
        assert "detected_intent" in data


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_code_task_intent(self):
        """CODE_TASK intent exists."""
        assert QueryIntent.CODE_TASK.value == "code_task"

    def test_debug_task_intent(self):
        """DEBUG_TASK intent exists."""
        assert QueryIntent.DEBUG_TASK.value == "debug_task"

    def test_analysis_task_intent(self):
        """ANALYSIS_TASK intent exists."""
        assert QueryIntent.ANALYSIS_TASK.value == "analysis_task"

    def test_question_intent(self):
        """QUESTION intent exists."""
        assert QueryIntent.QUESTION.value == "question"


class TestContextEnricherIntegration:
    """Integration tests for ContextEnricher."""

    def test_full_enrichment_pipeline(self):
        """Test complete enrichment pipeline."""
        enricher = ContextEnricher()

        result = enricher.enrich(
            query="Add error handling to this function",
            context={
                "file": "handler.py",
                "content": "def handle(x): return x.process()",
            },
        )

        # Should have all components
        assert result.enriched_context is not None
        assert result.detected_intent is not None
        assert result.reasoning is not None
        assert result.token_count > 0

    def test_enrichment_for_different_intents(self):
        """Different intents should produce different enrichments."""
        enricher = ContextEnricher()

        code_result = enricher.enrich("Implement login feature", {})
        debug_result = enricher.enrich("Why is this crashing?", {})

        # Should have different detected intents
        assert code_result.detected_intent != debug_result.detected_intent
