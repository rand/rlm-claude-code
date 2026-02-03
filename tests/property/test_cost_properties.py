"""
Property-based tests for cost tracking and token estimation.

Implements: Plan verification for tiktoken accuracy.
"""

import sys
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cost_tracker import (
    CostComponent,
    CostEstimate,
    CostTracker,
    compute_affordable_depth,
    compute_affordable_tokens,
    estimate_call_cost,
    estimate_tokens,
    get_model_costs,
)


class TestTokenEstimationProperties:
    """Property-based tests for token estimation."""

    @given(st.text(min_size=0, max_size=10000))
    @settings(max_examples=100)
    def test_estimate_tokens_non_negative(self, text: str):
        """Token count is always non-negative."""
        tokens = estimate_tokens(text)
        assert tokens >= 0

    @given(st.text(min_size=0, max_size=10000, alphabet=st.characters(codec="ascii")))
    @settings(max_examples=100)
    def test_estimate_tokens_bounded_by_length_ascii(self, text: str):
        """Token count never exceeds character count for ASCII text."""
        tokens = estimate_tokens(text)
        # For ASCII, each character can be at most 1 token
        assert tokens <= len(text) + 1  # +1 for potential BOS token

    @given(st.text(min_size=1, max_size=5000))
    @settings(max_examples=50)
    def test_estimate_tokens_positive_for_nonempty(self, text: str):
        """Non-empty text produces at least 1 token."""
        assume(len(text.strip()) > 0)  # Skip whitespace-only
        tokens = estimate_tokens(text)
        assert tokens >= 1

    @given(
        st.lists(
            st.text(min_size=2, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=15,
            max_size=100,
        ).map(lambda words: " ".join(words))
    )
    @settings(max_examples=50)
    def test_estimate_tokens_accuracy_bounds(self, text: str):
        """Token estimate is within reasonable bounds for English-like text.

        For English text, tiktoken typically produces ~4 chars/token.
        We allow 1-10 chars/token range for various content types.
        """
        tokens = estimate_tokens(text)
        chars_per_token = len(text) / tokens if tokens > 0 else 0

        # Reasonable bounds for English-like text: 1-10 chars per token
        assert chars_per_token >= 1, f"Too many tokens: {tokens} for {len(text)} chars"
        assert chars_per_token <= 10, f"Too few tokens: {tokens} for {len(text)} chars"

    @given(st.text(min_size=0, max_size=5000))
    @settings(max_examples=50)
    def test_estimate_tokens_deterministic(self, text: str):
        """Same input always produces same output."""
        tokens1 = estimate_tokens(text)
        tokens2 = estimate_tokens(text)
        assert tokens1 == tokens2

    @given(st.text(min_size=0, max_size=1000), st.text(min_size=0, max_size=1000))
    @settings(max_examples=50)
    def test_estimate_tokens_concatenation(self, text1: str, text2: str):
        """Tokens of concatenation roughly equals sum of individual tokens.

        Due to BPE merging at boundaries, concatenation might have
        slightly fewer tokens than the sum. We allow 10% variance.
        """
        tokens1 = estimate_tokens(text1)
        tokens2 = estimate_tokens(text2)
        tokens_combined = estimate_tokens(text1 + text2)

        # Combined should be within 10% of sum, or at least as many as max
        expected_sum = tokens1 + tokens2
        assert tokens_combined >= max(tokens1, tokens2)  # At least as many as largest part
        # Allow some merging at boundaries
        assert tokens_combined <= expected_sum + 10


class TestCostEstimateProperties:
    """Property-based tests for cost estimation."""

    @given(
        st.integers(min_value=0, max_value=1_000_000),
        st.integers(min_value=0, max_value=100_000),
        st.sampled_from(["opus", "sonnet", "haiku"]),
    )
    @settings(max_examples=100)
    def test_cost_estimate_non_negative(self, input_tokens: int, output_tokens: int, model: str):
        """Cost estimate is always non-negative."""
        estimate = CostEstimate(
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            model=model,
            confidence=0.8,
            component=CostComponent.ROOT_PROMPT,
        )
        assert estimate.estimated_cost >= 0

    @given(
        st.integers(min_value=0, max_value=1_000_000),
        st.integers(min_value=0, max_value=100_000),
    )
    @settings(max_examples=50)
    def test_opus_costs_more_than_haiku(self, input_tokens: int, output_tokens: int):
        """Opus always costs more than Haiku for same token count."""
        opus_estimate = CostEstimate(
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            model="opus",
            confidence=0.8,
            component=CostComponent.ROOT_PROMPT,
        )
        haiku_estimate = CostEstimate(
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            model="haiku",
            confidence=0.8,
            component=CostComponent.ROOT_PROMPT,
        )

        if input_tokens > 0 or output_tokens > 0:
            assert opus_estimate.estimated_cost > haiku_estimate.estimated_cost

    @given(
        st.integers(min_value=1, max_value=1_000_000),
        st.integers(min_value=1, max_value=100_000),
        st.sampled_from(["opus", "sonnet", "haiku"]),
    )
    @settings(max_examples=50)
    def test_cost_scales_with_tokens(self, input_tokens: int, output_tokens: int, model: str):
        """Cost increases when tokens increase."""
        estimate1 = CostEstimate(
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            model=model,
            confidence=0.8,
            component=CostComponent.ROOT_PROMPT,
        )
        estimate2 = CostEstimate(
            estimated_input_tokens=input_tokens * 2,
            estimated_output_tokens=output_tokens * 2,
            model=model,
            confidence=0.8,
            component=CostComponent.ROOT_PROMPT,
        )

        assert estimate2.estimated_cost > estimate1.estimated_cost


class TestCostTrackerProperties:
    """Property-based tests for CostTracker."""

    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=10000),
                st.integers(min_value=0, max_value=5000),
                st.sampled_from(["opus", "sonnet", "haiku"]),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_total_tokens_equals_sum(self, usages: list[tuple[int, int, str]]):
        """Total tokens equals sum of all recorded usages."""
        tracker = CostTracker()

        expected_total = 0
        for input_tokens, output_tokens, model in usages:
            tracker.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                component=CostComponent.ROOT_PROMPT,
            )
            expected_total += input_tokens + output_tokens

        assert tracker.total_tokens == expected_total

    @given(
        st.integers(min_value=1000, max_value=1_000_000),
        st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=30)
    def test_remaining_budget_decreases(self, budget_tokens: int, budget_dollars: float):
        """Remaining budget decreases after recording usage."""
        tracker = CostTracker(budget_tokens=budget_tokens, budget_dollars=budget_dollars)

        initial_remaining_tokens = tracker.remaining_tokens
        initial_remaining_dollars = tracker.remaining_budget

        tracker.record_usage(
            input_tokens=100,
            output_tokens=50,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        assert tracker.remaining_tokens < initial_remaining_tokens
        assert tracker.remaining_budget < initial_remaining_dollars

    @given(st.integers(min_value=100, max_value=10000))
    @settings(max_examples=30)
    def test_would_exceed_budget_accurate(self, budget_tokens: int):
        """would_exceed_budget correctly predicts budget violations."""
        tracker = CostTracker(budget_tokens=budget_tokens, budget_dollars=100.0)

        # Estimate that would exceed
        large_estimate = CostEstimate(
            estimated_input_tokens=budget_tokens + 1000,
            estimated_output_tokens=0,
            model="sonnet",
            confidence=0.8,
            component=CostComponent.ROOT_PROMPT,
        )
        would_exceed, reason = tracker.would_exceed_budget(large_estimate)
        assert would_exceed
        assert reason is not None

        # Estimate that wouldn't exceed
        small_estimate = CostEstimate(
            estimated_input_tokens=budget_tokens // 10,
            estimated_output_tokens=0,
            model="sonnet",
            confidence=0.8,
            component=CostComponent.ROOT_PROMPT,
        )
        would_exceed, reason = tracker.would_exceed_budget(small_estimate)
        assert not would_exceed
        assert reason is None


class TestModelCostProperties:
    """Property-based tests for model cost functions."""

    @given(st.sampled_from(["opus", "sonnet", "haiku", "gpt-4o", "gpt-4o-mini"]))
    @settings(max_examples=20)
    def test_get_model_costs_returns_valid_structure(self, model: str):
        """get_model_costs always returns input and output keys."""
        costs = get_model_costs(model)
        assert "input" in costs
        assert "output" in costs
        assert costs["input"] >= 0
        assert costs["output"] >= 0

    @given(
        st.integers(min_value=0, max_value=1_000_000),
        st.integers(min_value=0, max_value=100_000),
        st.sampled_from(["opus", "sonnet", "haiku"]),
    )
    @settings(max_examples=50)
    def test_estimate_call_cost_non_negative(
        self, input_tokens: int, output_tokens: int, model: str
    ):
        """estimate_call_cost always returns non-negative value."""
        cost = estimate_call_cost(input_tokens, output_tokens, model)
        assert cost >= 0

    @given(
        st.integers(min_value=1000, max_value=100_000),
        st.integers(min_value=100, max_value=10_000),
    )
    @settings(max_examples=30)
    def test_opus_more_expensive_than_haiku(self, input_tokens: int, output_tokens: int):
        """Opus is always more expensive than Haiku for same tokens."""
        opus_cost = estimate_call_cost(input_tokens, output_tokens, "opus")
        haiku_cost = estimate_call_cost(input_tokens, output_tokens, "haiku")
        assert opus_cost > haiku_cost

    @given(
        st.floats(min_value=0.01, max_value=100.0),
        st.sampled_from(["opus", "sonnet", "haiku"]),
    )
    @settings(max_examples=30)
    def test_compute_affordable_tokens_positive(self, budget: float, model: str):
        """compute_affordable_tokens returns positive value for positive budget."""
        tokens = compute_affordable_tokens(budget, model)
        assert tokens > 0

    @given(st.floats(min_value=0.1, max_value=50.0))
    @settings(max_examples=20)
    def test_haiku_affords_more_tokens_than_opus(self, budget: float):
        """Haiku affords more tokens than Opus for same budget."""
        haiku_tokens = compute_affordable_tokens(budget, "haiku")
        opus_tokens = compute_affordable_tokens(budget, "opus")
        assert haiku_tokens > opus_tokens

    @given(
        st.floats(min_value=0.1, max_value=20.0),
        st.sampled_from(["opus", "sonnet", "haiku"]),
    )
    @settings(max_examples=30)
    def test_compute_affordable_depth_bounded(self, budget: float, model: str):
        """compute_affordable_depth is bounded by 0-3."""
        depth = compute_affordable_depth(budget, model)
        assert 0 <= depth <= 3

    @given(st.floats(min_value=0.5, max_value=10.0))
    @settings(max_examples=20)
    def test_haiku_affords_same_or_more_depth_than_opus(self, budget: float):
        """Haiku affords same or more depth than Opus for same budget."""
        haiku_depth = compute_affordable_depth(budget, "haiku")
        opus_depth = compute_affordable_depth(budget, "opus")
        assert haiku_depth >= opus_depth
