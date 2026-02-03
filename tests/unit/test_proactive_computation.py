"""
Tests for proactive REPL computation (SPEC-06.01-06.05).

Tests cover:
- Computation trigger detection
- Code template generation
- Computational helpers
- Sandbox security
"""

import pytest

from src.proactive_computation import (
    ComputationAdvisor,
    ComputationHelper,
    ComputationSuggestion,
    ComputationTrigger,
)


class TestComputationTriggerDetection:
    """Tests for computation trigger detection (SPEC-06.01, SPEC-06.02)."""

    def test_detects_arithmetic_operations(self):
        """SPEC-06.02: Detect arithmetic operations."""
        advisor = ComputationAdvisor()

        queries = [
            "Calculate 15 * 23 + 47",
            "What is 1024 / 16?",
            "Compute the result of 2^10 - 100",
            "Add 125 and 375",
        ]

        for query in queries:
            result = advisor.detect_triggers(query)
            assert ComputationTrigger.ARITHMETIC in result.triggers, f"Failed for: {query}"

    def test_detects_counting_operations(self):
        """SPEC-06.02: Detect counting operations."""
        advisor = ComputationAdvisor()

        queries = [
            "How many files are in the src directory?",
            "Count the number of TODO comments",
            "Total number of test cases",
            "How many imports does this file have?",
        ]

        for query in queries:
            result = advisor.detect_triggers(query)
            assert ComputationTrigger.COUNTING in result.triggers, f"Failed for: {query}"

    def test_detects_sorting_operations(self):
        """SPEC-06.02: Detect sorting operations."""
        advisor = ComputationAdvisor()

        queries = [
            "Sort the files by size",
            "Order these functions by line count",
            "What are the top 5 largest files?",
            "Rank these methods by complexity",
            "Find the smallest value",
        ]

        for query in queries:
            result = advisor.detect_triggers(query)
            assert ComputationTrigger.SORTING in result.triggers, f"Failed for: {query}"

    def test_detects_filtering_operations(self):
        """SPEC-06.02: Detect filtering operations."""
        advisor = ComputationAdvisor()

        queries = [
            "Filter files containing 'test'",
            "Show only functions where complexity > 10",
            "Find files matching *.py",
            "List classes containing 'Handler'",
        ]

        for query in queries:
            result = advisor.detect_triggers(query)
            assert ComputationTrigger.FILTERING in result.triggers, f"Failed for: {query}"

    def test_detects_aggregation_operations(self):
        """SPEC-06.02: Detect aggregation operations."""
        advisor = ComputationAdvisor()

        queries = [
            "What is the sum of all file sizes?",
            "Calculate the average line count",
            "Find the median complexity score",
            "What is the maximum depth?",
            "Mean time to completion",
        ]

        for query in queries:
            result = advisor.detect_triggers(query)
            assert ComputationTrigger.AGGREGATION in result.triggers, f"Failed for: {query}"

    def test_detects_date_math_operations(self):
        """SPEC-06.02: Detect date math operations."""
        advisor = ComputationAdvisor()

        queries = [
            "How many days since the last commit?",
            "Files modified in the last 2 weeks",
            "Changes before January 1st",
            "Commits after 2024-01-01",
            "What was changed 3 weeks ago?",
        ]

        for query in queries:
            result = advisor.detect_triggers(query)
            assert ComputationTrigger.DATE_MATH in result.triggers, f"Failed for: {query}"

    def test_detects_string_operations(self):
        """SPEC-06.02: Detect string operations."""
        advisor = ComputationAdvisor()

        queries = [
            "Extract the function names from this code",
            "Parse the error message",
            "Split the path into components",
            "Format this data as JSON",
            "Find all matches for the regex pattern",
        ]

        for query in queries:
            result = advisor.detect_triggers(query)
            assert ComputationTrigger.STRING_OPS in result.triggers, f"Failed for: {query}"

    def test_multiple_triggers_detected(self):
        """Multiple triggers can be detected in one query."""
        advisor = ComputationAdvisor()

        query = "Count the files and sort them by size, then show the top 10"
        result = advisor.detect_triggers(query)

        assert ComputationTrigger.COUNTING in result.triggers
        assert ComputationTrigger.SORTING in result.triggers

    def test_no_triggers_for_non_computational_query(self):
        """Non-computational queries should return no triggers."""
        advisor = ComputationAdvisor()

        queries = [
            "Explain this code",
            "What does this function do?",
            "Can you help me understand this?",
            "Review this pull request",
        ]

        for query in queries:
            result = advisor.detect_triggers(query)
            assert len(result.triggers) == 0, f"False positive for: {query}"


class TestCodeSuggestionGeneration:
    """Tests for REPL code suggestion generation (SPEC-06.03)."""

    def test_generates_arithmetic_code(self):
        """SPEC-06.03: Generate REPL code for arithmetic."""
        advisor = ComputationAdvisor()

        suggestion = advisor.suggest_computation("Calculate 15 * 23 + 47")

        assert suggestion is not None
        assert suggestion.code is not None
        assert "calc" in suggestion.code or "*" in suggestion.code
        assert suggestion.explanation is not None

    def test_generates_counting_code(self):
        """SPEC-06.03: Generate REPL code for counting."""
        advisor = ComputationAdvisor()

        suggestion = advisor.suggest_computation(
            "How many items in the list?",
            context={"list": [1, 2, 3, 4, 5]},
        )

        assert suggestion is not None
        assert "len" in suggestion.code or "count" in suggestion.code

    def test_generates_sorting_code(self):
        """SPEC-06.03: Generate REPL code for sorting."""
        advisor = ComputationAdvisor()

        suggestion = advisor.suggest_computation(
            "Sort these numbers: 5, 2, 8, 1, 9",
        )

        assert suggestion is not None
        assert "sort" in suggestion.code.lower()

    def test_generates_aggregation_code(self):
        """SPEC-06.03: Generate REPL code for aggregation."""
        advisor = ComputationAdvisor()

        suggestion = advisor.suggest_computation(
            "What is the average of these values?",
            context={"values": [10, 20, 30, 40, 50]},
        )

        assert suggestion is not None
        assert "stats" in suggestion.code or "mean" in suggestion.code or "sum" in suggestion.code

    def test_suggestion_includes_explanation(self):
        """SPEC-06.03: Suggestions should include explanation."""
        advisor = ComputationAdvisor()

        suggestion = advisor.suggest_computation("Calculate 100 / 4")

        assert suggestion.explanation is not None
        assert len(suggestion.explanation) > 10

    def test_no_suggestion_for_non_computational(self):
        """Non-computational queries should not get suggestions."""
        advisor = ComputationAdvisor()

        suggestion = advisor.suggest_computation("Explain this code")

        assert suggestion is None


class TestComputationHelper:
    """Tests for computational helpers (SPEC-06.04)."""

    def test_calc_basic_arithmetic(self):
        """SPEC-06.04: calc() for safe math evaluation."""
        helper = ComputationHelper()

        assert helper.calc("2 + 3") == 5
        assert helper.calc("10 * 5") == 50
        assert helper.calc("100 / 4") == 25
        assert helper.calc("2 ** 10") == 1024

    def test_calc_with_math_functions(self):
        """calc() should support math functions."""
        helper = ComputationHelper()

        assert helper.calc("sqrt(16)") == 4.0
        assert helper.calc("abs(-5)") == 5
        assert abs(helper.calc("sin(0)")) < 0.0001
        assert helper.calc("max(1, 5, 3)") == 5

    def test_calc_rejects_dangerous_expressions(self):
        """calc() should reject dangerous expressions."""
        helper = ComputationHelper()

        dangerous = [
            "__import__('os')",
            "open('/etc/passwd')",
            "exec('print(1)')",
            "eval('1+1')",
        ]

        for expr in dangerous:
            with pytest.raises(ValueError):
                helper.calc(expr)

    def test_stats_basic(self):
        """SPEC-06.04: stats() for statistical computation."""
        helper = ComputationHelper()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        result = helper.stats(data)

        assert result["count"] == 10
        assert result["sum"] == 55
        assert result["mean"] == 5.5
        assert result["min"] == 1
        assert result["max"] == 10
        assert "median" in result
        assert "std" in result

    def test_stats_empty_data(self):
        """stats() should handle empty data."""
        helper = ComputationHelper()

        result = helper.stats([])
        assert result["count"] == 0

    def test_group_by_basic(self):
        """SPEC-06.04: group_by() for data grouping."""
        helper = ComputationHelper()
        data = [
            {"type": "A", "value": 1},
            {"type": "B", "value": 2},
            {"type": "A", "value": 3},
            {"type": "B", "value": 4},
        ]

        result = helper.group_by(data, "type")

        assert "A" in result
        assert "B" in result
        assert len(result["A"]) == 2
        assert len(result["B"]) == 2

    def test_group_by_with_function_key(self):
        """group_by() should support function keys."""
        helper = ComputationHelper()
        data = ["apple", "banana", "apricot", "blueberry"]

        result = helper.group_by(data, lambda x: x[0])

        assert "a" in result
        assert "b" in result
        assert len(result["a"]) == 2

    def test_find_imports(self):
        """SPEC-06.04: find_imports() for import analysis."""
        helper = ComputationHelper()
        content = """
import os
from pathlib import Path
import json as j
from typing import Any, Optional
from . import local_module
from ..parent import something
"""
        imports = helper.find_imports(content)

        assert "os" in imports
        assert "pathlib" in imports
        assert "json" in imports
        assert "typing" in imports

    def test_find_imports_handles_complex_imports(self):
        """find_imports() handles multi-line and complex imports."""
        helper = ComputationHelper()
        content = """
from module import (
    thing1,
    thing2,
    thing3,
)
"""
        imports = helper.find_imports(content)
        assert "module" in imports


class TestComputationHelperSecurity:
    """Tests for computational helper security (SPEC-06.05)."""

    def test_calc_sandboxed(self):
        """SPEC-06.05: calc() is sandboxed."""
        helper = ComputationHelper()

        # These should all fail or return safe results
        dangerous_cases = [
            ("__builtins__", ValueError),
            ("globals()", ValueError),
            ("locals()", ValueError),
            ("dir()", ValueError),
            ("type.__subclasses__(type)", ValueError),
        ]

        for expr, expected_error in dangerous_cases:
            with pytest.raises(expected_error):
                helper.calc(expr)

    def test_helpers_no_file_access(self):
        """Helpers should not allow file access."""
        helper = ComputationHelper()

        with pytest.raises(ValueError):
            helper.calc("open('/etc/passwd').read()")

    def test_helpers_no_network_access(self):
        """Helpers should not allow network access."""
        helper = ComputationHelper()

        with pytest.raises(ValueError):
            helper.calc("__import__('socket').socket()")

    def test_helpers_timeout_protection(self):
        """Helpers should have timeout protection."""
        helper = ComputationHelper(timeout_ms=100)

        # This should timeout or be rejected
        with pytest.raises((TimeoutError, ValueError)):
            helper.calc("10**10**10")  # Huge computation


class TestComputationAdvisorIntegration:
    """Integration tests for ComputationAdvisor."""

    def test_full_workflow_arithmetic(self):
        """Full workflow for arithmetic computation."""
        advisor = ComputationAdvisor()

        # 1. Detect
        detection = advisor.detect_triggers("What is 125 * 8?")
        assert ComputationTrigger.ARITHMETIC in detection.triggers
        assert detection.confidence > 0.8

        # 2. Suggest
        suggestion = advisor.suggest_computation("What is 125 * 8?")
        assert suggestion is not None

        # 3. Execute (via helper)
        result = advisor.execute_suggestion(suggestion)
        assert result == 1000

    def test_full_workflow_statistics(self):
        """Full workflow for statistical computation."""
        advisor = ComputationAdvisor()

        context = {"data": [10, 20, 30, 40, 50]}

        # 1. Detect
        detection = advisor.detect_triggers("What is the average?")
        assert ComputationTrigger.AGGREGATION in detection.triggers

        # 2. Suggest
        suggestion = advisor.suggest_computation(
            "What is the average of the data?",
            context=context,
        )
        assert suggestion is not None

        # 3. Execute
        result = advisor.execute_suggestion(suggestion, context=context)
        assert result["mean"] == 30.0

    def test_should_use_repl_high_confidence(self):
        """should_use_repl returns True for high confidence."""
        advisor = ComputationAdvisor()

        assert advisor.should_use_repl("Calculate 15 + 27")
        assert advisor.should_use_repl("Count the items in [1,2,3,4,5]")

    def test_should_use_repl_low_confidence(self):
        """should_use_repl returns False for low confidence."""
        advisor = ComputationAdvisor()

        assert not advisor.should_use_repl("Explain this code")
        assert not advisor.should_use_repl("What does this function do?")

    def test_detection_confidence_varies(self):
        """Detection confidence varies by query clarity."""
        advisor = ComputationAdvisor()

        # Clear computation
        clear = advisor.detect_triggers("Calculate 10 * 5")
        # Ambiguous
        ambiguous = advisor.detect_triggers("What about the numbers?")

        assert clear.confidence > ambiguous.confidence


class TestComputationTriggerEnum:
    """Tests for ComputationTrigger enum."""

    def test_all_triggers_defined(self):
        """All required triggers from SPEC-06.02 are defined."""
        assert hasattr(ComputationTrigger, "ARITHMETIC")
        assert hasattr(ComputationTrigger, "COUNTING")
        assert hasattr(ComputationTrigger, "SORTING")
        assert hasattr(ComputationTrigger, "FILTERING")
        assert hasattr(ComputationTrigger, "AGGREGATION")
        assert hasattr(ComputationTrigger, "DATE_MATH")
        assert hasattr(ComputationTrigger, "STRING_OPS")


class TestComputationSuggestion:
    """Tests for ComputationSuggestion structure."""

    def test_suggestion_structure(self):
        """ComputationSuggestion has required fields."""
        suggestion = ComputationSuggestion(
            triggers=[ComputationTrigger.ARITHMETIC],
            code="calc('15 * 23')",
            explanation="Using REPL to compute 15 * 23",
            confidence=0.95,
        )

        assert suggestion.triggers == [ComputationTrigger.ARITHMETIC]
        assert suggestion.code == "calc('15 * 23')"
        assert suggestion.explanation == "Using REPL to compute 15 * 23"
        assert suggestion.confidence == 0.95

    def test_suggestion_to_dict(self):
        """ComputationSuggestion serializes correctly."""
        suggestion = ComputationSuggestion(
            triggers=[ComputationTrigger.COUNTING],
            code="len(data)",
            explanation="Count items",
            confidence=0.9,
        )

        data = suggestion.to_dict()
        assert data["code"] == "len(data)"
        assert data["confidence"] == 0.9
