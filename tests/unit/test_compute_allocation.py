"""
Tests for compute-optimal allocation.

@trace SPEC-07.10-07.15
"""

from __future__ import annotations

import pytest

from src.compute_allocation import (
    AllocationReasoning,
    ComputeAllocation,
    ComputeAllocator,
    ModelTier,
    TaskType,
)

# --- Test fixtures ---


def create_simple_query() -> str:
    """Create a simple query."""
    return "What does this function do?"


def create_complex_query() -> str:
    """Create a complex multi-step query."""
    return """Analyze the entire authentication system across all modules.
    Find all security vulnerabilities. Suggest fixes with implementation details.
    Verify the fixes don't break existing functionality."""


def create_code_context(num_files: int = 1, lines_per_file: int = 100) -> dict[str, str]:
    """Create mock code context."""
    return {f"file{i}.py": f"# File {i}\n" + "x = 1\n" * lines_per_file for i in range(num_files)}


# --- SPEC-07.10: Dynamic compute allocation ---


class TestDynamicAllocation:
    """Tests for dynamic compute allocation based on query difficulty."""

    def test_simple_query_gets_minimal_allocation(self) -> None:
        """
        @trace SPEC-07.10
        Simple queries should get minimal compute allocation.
        """
        allocator = ComputeAllocator()
        query = create_simple_query()

        allocation = allocator.allocate(query=query, context={})

        # Simple query should get low depth and cheaper model
        assert allocation.depth_budget <= 2
        assert allocation.model_tier in [ModelTier.HAIKU, ModelTier.SONNET]

    def test_complex_query_gets_higher_allocation(self) -> None:
        """
        @trace SPEC-07.10
        Complex queries should get higher compute allocation.
        """
        allocator = ComputeAllocator()
        query = create_complex_query()
        context = create_code_context(num_files=10, lines_per_file=500)

        allocation = allocator.allocate(query=query, context=context)

        # Complex query should get higher depth
        assert allocation.depth_budget >= 2
        # And potentially higher-tier model
        assert allocation.model_tier in [ModelTier.SONNET, ModelTier.OPUS]

    def test_allocation_varies_with_context_size(self) -> None:
        """
        @trace SPEC-07.10
        Allocation should increase with context size.
        """
        allocator = ComputeAllocator()
        query = "Analyze all files"

        small_context = create_code_context(num_files=2)
        large_context = create_code_context(num_files=20)

        small_alloc = allocator.allocate(query=query, context=small_context)
        large_alloc = allocator.allocate(query=query, context=large_context)

        # Larger context should get more resources
        assert (
            large_alloc.depth_budget >= small_alloc.depth_budget
            or large_alloc.parallel_calls >= small_alloc.parallel_calls
        )


# --- SPEC-07.11: ComputeAllocation structure ---


class TestComputeAllocationStructure:
    """Tests for ComputeAllocation dataclass structure."""

    def test_allocation_has_required_fields(self) -> None:
        """
        @trace SPEC-07.11
        ComputeAllocation should have all required fields.
        """
        allocation = ComputeAllocation(
            depth_budget=3,
            model_tier=ModelTier.SONNET,
            parallel_calls=5,
            timeout_ms=30000,
            estimated_cost=0.15,
        )

        assert allocation.depth_budget == 3
        assert allocation.model_tier == ModelTier.SONNET
        assert allocation.parallel_calls == 5
        assert allocation.timeout_ms == 30000
        assert allocation.estimated_cost == pytest.approx(0.15)

    def test_model_tier_enum(self) -> None:
        """
        @trace SPEC-07.11
        ModelTier should have expected values.
        """
        assert ModelTier.HAIKU.value == "haiku"
        assert ModelTier.SONNET.value == "sonnet"
        assert ModelTier.OPUS.value == "opus"


# --- SPEC-07.12: Difficulty estimation ---


class TestDifficultyEstimation:
    """Tests for difficulty estimation."""

    def test_estimate_uses_complexity_signals(self) -> None:
        """
        @trace SPEC-07.12
        Difficulty estimation should use complexity signals.
        """
        allocator = ComputeAllocator()

        # Multi-step query
        query = "First analyze, then refactor, finally test"
        estimate = allocator.estimate_difficulty(query=query, context={})

        assert estimate is not None
        assert 0.0 <= estimate.score <= 1.0

    def test_estimate_considers_context_complexity(self) -> None:
        """
        @trace SPEC-07.12
        Difficulty estimation should consider context complexity.
        """
        allocator = ComputeAllocator()
        query = "Analyze this code"

        simple_context = {"file.py": "x = 1"}
        complex_context = create_code_context(num_files=50, lines_per_file=500)

        simple_estimate = allocator.estimate_difficulty(query=query, context=simple_context)
        complex_estimate = allocator.estimate_difficulty(query=query, context=complex_context)

        assert complex_estimate.score >= simple_estimate.score

    def test_estimate_considers_task_type(self) -> None:
        """
        @trace SPEC-07.12
        Difficulty estimation should consider task type.
        """
        allocator = ComputeAllocator()

        code_query = "Write a function to sort a list"
        debug_query = "Debug this segfault in the memory allocator"
        analysis_query = "What does this code do?"

        code_estimate = allocator.estimate_difficulty(query=code_query, context={})
        debug_estimate = allocator.estimate_difficulty(query=debug_query, context={})
        analysis_estimate = allocator.estimate_difficulty(query=analysis_query, context={})

        # Debug tasks typically harder than simple questions
        assert debug_estimate.task_type == TaskType.DEBUG
        assert analysis_estimate.task_type == TaskType.QUESTION


# --- SPEC-07.13: Budget constraints ---


class TestBudgetConstraints:
    """Tests for total budget constraint."""

    def test_allocation_respects_budget(self) -> None:
        """
        @trace SPEC-07.13
        Allocation should respect total_budget constraint.
        """
        allocator = ComputeAllocator()
        query = create_complex_query()
        context = create_code_context(num_files=10)

        allocation = allocator.allocate(
            query=query,
            context=context,
            total_budget=0.10,  # Low budget
        )

        assert allocation.estimated_cost <= 0.10

    def test_budget_reduces_allocation(self) -> None:
        """
        @trace SPEC-07.13
        Lower budget should result in reduced allocation.
        """
        allocator = ComputeAllocator()
        query = create_complex_query()
        context = create_code_context(num_files=10)

        high_budget_alloc = allocator.allocate(
            query=query,
            context=context,
            total_budget=5.0,
        )

        low_budget_alloc = allocator.allocate(
            query=query,
            context=context,
            total_budget=0.05,
        )

        # Low budget should have fewer resources
        assert (
            low_budget_alloc.depth_budget <= high_budget_alloc.depth_budget
            or low_budget_alloc.model_tier.value <= high_budget_alloc.model_tier.value
        )


# --- SPEC-07.14: Model+depth tradeoffs ---


class TestModelDepthTradeoffs:
    """Tests for model+depth tradeoff consideration."""

    def test_considers_haiku_deep_vs_opus_shallow(self) -> None:
        """
        @trace SPEC-07.14
        Should consider tradeoffs between model tier and depth.
        """
        allocator = ComputeAllocator()

        # With constrained budget, should make tradeoff decision
        query = create_complex_query()
        context = create_code_context(num_files=5)

        allocation = allocator.allocate(
            query=query,
            context=context,
            total_budget=0.20,
        )

        # Either deep with cheap model or shallow with expensive model
        if allocation.model_tier == ModelTier.HAIKU:
            assert allocation.depth_budget >= 2
        elif allocation.model_tier == ModelTier.OPUS:
            assert allocation.depth_budget <= 2

    def test_quality_vs_cost_optimization(self) -> None:
        """
        @trace SPEC-07.14
        Should optimize for quality vs cost based on mode.
        """
        allocator = ComputeAllocator()
        query = "Analyze security vulnerabilities"
        context = create_code_context(num_files=5)

        # Quality-focused allocation
        quality_alloc = allocator.allocate(
            query=query,
            context=context,
            optimize_for="quality",
        )

        # Cost-focused allocation
        cost_alloc = allocator.allocate(
            query=query,
            context=context,
            optimize_for="cost",
        )

        # Quality should prefer higher-tier models
        # Cost should prefer cheaper options
        assert quality_alloc.estimated_cost >= cost_alloc.estimated_cost


# --- SPEC-07.15: Allocation reasoning ---


class TestAllocationReasoning:
    """Tests for allocation reasoning transparency."""

    def test_allocation_includes_reasoning(self) -> None:
        """
        @trace SPEC-07.15
        Allocation should include reasoning for transparency.
        """
        allocator = ComputeAllocator()
        query = create_complex_query()

        allocation = allocator.allocate(query=query, context={})

        assert allocation.reasoning is not None
        assert isinstance(allocation.reasoning, AllocationReasoning)

    def test_reasoning_explains_decisions(self) -> None:
        """
        @trace SPEC-07.15
        Reasoning should explain allocation decisions.
        """
        allocator = ComputeAllocator()
        query = create_complex_query()

        allocation = allocator.allocate(query=query, context={})

        reasoning = allocation.reasoning
        assert reasoning.difficulty_factors is not None
        assert reasoning.tradeoff_explanation is not None

    def test_reasoning_is_loggable(self) -> None:
        """
        @trace SPEC-07.15
        Reasoning should be convertible to log format.
        """
        allocator = ComputeAllocator()
        query = "Test query"

        allocation = allocator.allocate(query=query, context={})

        log_dict = allocation.reasoning.to_dict()
        assert isinstance(log_dict, dict)
        assert "difficulty_score" in log_dict or "factors" in log_dict


# --- Integration tests ---


class TestComputeAllocatorIntegration:
    """Integration tests for ComputeAllocator."""

    def test_full_allocation_workflow(self) -> None:
        """
        Test complete allocation workflow.
        """
        allocator = ComputeAllocator()

        # Estimate difficulty
        query = "Refactor the authentication module to use JWT"
        context = create_code_context(num_files=5)

        difficulty = allocator.estimate_difficulty(query=query, context=context)
        assert difficulty.score > 0

        # Get allocation
        allocation = allocator.allocate(
            query=query,
            context=context,
            total_budget=1.0,
        )

        # Verify allocation is reasonable
        assert allocation.depth_budget >= 1
        assert allocation.model_tier in ModelTier
        assert allocation.timeout_ms > 0
        assert allocation.estimated_cost <= 1.0
        assert allocation.reasoning is not None

    def test_allocation_determinism(self) -> None:
        """
        Same inputs should produce same allocation.
        """
        allocator = ComputeAllocator()
        query = "Simple question"
        context = {"file.py": "x = 1"}

        alloc1 = allocator.allocate(query=query, context=context)
        alloc2 = allocator.allocate(query=query, context=context)

        assert alloc1.depth_budget == alloc2.depth_budget
        assert alloc1.model_tier == alloc2.model_tier
        assert alloc1.parallel_calls == alloc2.parallel_calls
