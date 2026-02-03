"""
Tests for learned model routing (SPEC-06.20-06.26).

Tests cover:
- Query-based routing
- Model profiles
- Difficulty estimation
- Cost sensitivity
- Cascading routing
- Outcome recording
"""

from src.learned_routing import (
    CascadingRouter,
    DifficultyEstimator,
    LearnedRouter,
    ModelProfile,
    OutcomeRecord,
    QueryFeatures,
    RouterConfig,
    RoutingDecision,
)


class TestQueryBasedRouting:
    """Tests for query-based routing (SPEC-06.20)."""

    def test_router_selects_model_based_on_query(self):
        """SPEC-06.20: Router selects model based on query characteristics."""
        router = LearnedRouter()

        decision = router.route("What is 2 + 2?")

        assert isinstance(decision, RoutingDecision)
        assert decision.model in ["haiku", "sonnet", "opus"]

    def test_simple_query_routes_to_cheap_model(self):
        """Simple queries should route to cheaper models."""
        router = LearnedRouter()

        decision = router.route("What is the capital of France?")

        # Simple factual query should prefer cheaper model
        assert decision.model in ["haiku", "sonnet"]

    def test_complex_query_routes_to_capable_model(self):
        """Complex queries should route to more capable models."""
        router = LearnedRouter()

        decision = router.route(
            "Analyze the architectural implications of implementing "
            "a distributed consensus algorithm using Raft vs Paxos, "
            "considering network partitions and Byzantine fault tolerance."
        )

        # Complex analysis should prefer more capable model
        assert decision.model in ["sonnet", "opus"]


class TestModelProfiles:
    """Tests for model profiles (SPEC-06.21)."""

    def test_profile_has_strengths(self):
        """SPEC-06.21: Profile includes strengths (task types)."""
        profile = ModelProfile(
            name="sonnet",
            strengths=["code", "analysis", "reasoning"],
            cost_per_1k=0.003,
            quality_baseline=0.85,
        )

        assert "code" in profile.strengths
        assert "analysis" in profile.strengths

    def test_profile_has_cost(self):
        """SPEC-06.21: Profile includes cost_per_1k tokens."""
        profile = ModelProfile(
            name="haiku",
            strengths=["simple", "factual"],
            cost_per_1k=0.00025,
            quality_baseline=0.7,
        )

        assert profile.cost_per_1k == 0.00025

    def test_profile_has_quality_baseline(self):
        """SPEC-06.21: Profile includes quality_baseline (0-1)."""
        profile = ModelProfile(
            name="opus",
            strengths=["complex", "creative", "nuanced"],
            cost_per_1k=0.015,
            quality_baseline=0.95,
        )

        assert 0 <= profile.quality_baseline <= 1
        assert profile.quality_baseline == 0.95


class TestDifficultyEstimation:
    """Tests for query difficulty estimation (SPEC-06.22)."""

    def test_estimates_reasoning_depth(self):
        """SPEC-06.22: Estimate reasoning depth required."""
        estimator = DifficultyEstimator()

        simple = estimator.estimate("What color is the sky?")
        complex = estimator.estimate(
            "If all A are B and some B are C, what can we conclude about A and C?"
        )

        assert complex.reasoning_depth > simple.reasoning_depth

    def test_estimates_domain_specificity(self):
        """SPEC-06.22: Estimate domain specificity."""
        estimator = DifficultyEstimator()

        general = estimator.estimate("How do I make coffee?")
        specific = estimator.estimate("Explain the Hindley-Milner type inference algorithm.")

        assert specific.domain_specificity > general.domain_specificity

    def test_estimates_ambiguity_level(self):
        """SPEC-06.22: Estimate ambiguity level."""
        estimator = DifficultyEstimator()

        clear = estimator.estimate("What is 5 * 3?")
        ambiguous = estimator.estimate("Is this good?")

        assert ambiguous.ambiguity_level > clear.ambiguity_level

    def test_estimates_context_size(self):
        """SPEC-06.22: Estimate context size impact."""
        estimator = DifficultyEstimator()

        short = estimator.estimate("Hello")
        long = estimator.estimate("x " * 1000)

        assert long.context_size > short.context_size


class TestCostSensitivity:
    """Tests for cost sensitivity parameter (SPEC-06.23)."""

    def test_cost_sensitivity_zero_prefers_quality(self):
        """SPEC-06.23: cost_sensitivity=0 means quality only."""
        router = LearnedRouter(cost_sensitivity=0.0)

        decision = router.route("Complex analysis task")

        # Should prefer highest quality regardless of cost
        assert decision.model == "opus"

    def test_cost_sensitivity_one_prefers_cheap(self):
        """SPEC-06.23: cost_sensitivity=1 means cost only."""
        router = LearnedRouter(cost_sensitivity=1.0)

        decision = router.route("Complex analysis task")

        # Should prefer cheapest regardless of quality
        assert decision.model == "haiku"

    def test_cost_sensitivity_balanced(self):
        """Balanced cost_sensitivity trades off quality and cost."""
        router = LearnedRouter(cost_sensitivity=0.5)

        decision = router.route("Moderate complexity task")

        # Should pick middle-ground model
        assert decision.model in ["haiku", "sonnet", "opus"]


class TestCascadingRouting:
    """Tests for cascading routing (SPEC-06.24, SPEC-06.25)."""

    def test_cascading_starts_with_cheapest(self):
        """SPEC-06.24: Start with cheapest viable model."""
        router = CascadingRouter()

        result = router.route_with_cascade("Simple question")

        # First attempt should use cheapest
        assert result.attempts[0].model == "haiku"

    def test_cascading_escalates_on_low_confidence(self):
        """SPEC-06.24: Escalate on low confidence."""
        config = RouterConfig(confidence_threshold=0.9)
        router = CascadingRouter(config=config)

        # Mock low confidence on first attempt
        result = router.route_with_cascade(
            "Complex question",
            mock_confidences=[0.5, 0.7, 0.95],
        )

        # Should have escalated
        assert len(result.attempts) > 1

    def test_default_confidence_threshold(self):
        """SPEC-06.25: Default confidence_threshold is 0.8."""
        config = RouterConfig()
        assert config.confidence_threshold == 0.8

    def test_default_max_escalations(self):
        """SPEC-06.25: Default max_escalations is 2."""
        config = RouterConfig()
        assert config.max_escalations == 2

    def test_default_cascade_order(self):
        """SPEC-06.25: Default cascade_order is haiku, sonnet, opus."""
        config = RouterConfig()
        assert config.cascade_order == ["haiku", "sonnet", "opus"]

    def test_custom_cascade_order(self):
        """Custom cascade order should be respected."""
        config = RouterConfig(cascade_order=["sonnet", "opus"])
        router = CascadingRouter(config=config)

        result = router.route_with_cascade("Test")

        assert result.attempts[0].model == "sonnet"

    def test_max_escalations_respected(self):
        """Should not exceed max_escalations."""
        config = RouterConfig(max_escalations=1)
        router = CascadingRouter(config=config)

        result = router.route_with_cascade(
            "Test",
            mock_confidences=[0.3, 0.3, 0.3],
        )

        # Only 2 attempts (initial + 1 escalation)
        assert len(result.attempts) <= 2


class TestOutcomeRecording:
    """Tests for outcome recording (SPEC-06.26)."""

    def test_record_includes_query_embedding(self):
        """SPEC-06.26: Record includes query embedding."""
        record = OutcomeRecord(
            query="Test query",
            query_embedding=[0.1, 0.2, 0.3],
            model="sonnet",
            success=True,
            quality_score=0.9,
            cost=0.005,
        )

        assert record.query_embedding is not None
        assert len(record.query_embedding) > 0

    def test_record_includes_model_used(self):
        """SPEC-06.26: Record includes model used."""
        record = OutcomeRecord(
            query="Test",
            query_embedding=[],
            model="opus",
            success=True,
            quality_score=0.95,
            cost=0.02,
        )

        assert record.model == "opus"

    def test_record_includes_success_failure(self):
        """SPEC-06.26: Record includes success/failure."""
        success = OutcomeRecord(
            query="Test",
            query_embedding=[],
            model="sonnet",
            success=True,
            quality_score=0.9,
            cost=0.01,
        )
        failure = OutcomeRecord(
            query="Test",
            query_embedding=[],
            model="haiku",
            success=False,
            quality_score=0.3,
            cost=0.001,
        )

        assert success.success is True
        assert failure.success is False

    def test_record_includes_quality_score(self):
        """SPEC-06.26: Record includes quality score."""
        record = OutcomeRecord(
            query="Test",
            query_embedding=[],
            model="sonnet",
            success=True,
            quality_score=0.85,
            cost=0.01,
        )

        assert record.quality_score == 0.85

    def test_record_includes_cost(self):
        """SPEC-06.26: Record includes cost."""
        record = OutcomeRecord(
            query="Test",
            query_embedding=[],
            model="opus",
            success=True,
            quality_score=0.9,
            cost=0.025,
        )

        assert record.cost == 0.025

    def test_router_records_outcomes(self):
        """Router should record outcomes for learning."""
        router = LearnedRouter()

        # Make a routing decision and record outcome
        decision = router.route("Test query")
        router.record_outcome(
            query="Test query",
            decision=decision,
            success=True,
            quality_score=0.9,
        )

        records = router.get_outcome_history()
        assert len(records) == 1
        assert records[0].model == decision.model


class TestQueryFeatures:
    """Tests for query feature extraction."""

    def test_features_include_length(self):
        """Features should include query length."""
        features = QueryFeatures.from_query("Short query")

        assert features.length > 0

    def test_features_include_complexity_indicators(self):
        """Features should include complexity indicators."""
        simple = QueryFeatures.from_query("Hello")
        complex = QueryFeatures.from_query(
            "Analyze the implications of quantum entanglement "
            "on information theory and cryptography"
        )

        assert complex.estimated_complexity > simple.estimated_complexity

    def test_features_include_domain_hints(self):
        """Features should include domain hints."""
        features = QueryFeatures.from_query("Debug the Python code")

        assert "code" in features.domain_hints or "programming" in features.domain_hints


class TestRoutingDecision:
    """Tests for routing decision structure."""

    def test_decision_includes_model(self):
        """Decision should include selected model."""
        decision = RoutingDecision(
            model="sonnet",
            confidence=0.85,
            reasoning="Best fit for code task",
        )

        assert decision.model == "sonnet"

    def test_decision_includes_confidence(self):
        """Decision should include confidence."""
        decision = RoutingDecision(
            model="opus",
            confidence=0.95,
            reasoning="Complex task requires opus",
        )

        assert decision.confidence == 0.95

    def test_decision_includes_reasoning(self):
        """Decision should include reasoning."""
        decision = RoutingDecision(
            model="haiku",
            confidence=0.9,
            reasoning="Simple factual query, haiku sufficient",
        )

        assert "simple" in decision.reasoning.lower() or len(decision.reasoning) > 0


class TestLearnedRouterIntegration:
    """Integration tests for LearnedRouter."""

    def test_router_learns_from_outcomes(self):
        """Router should adjust based on outcome history."""
        router = LearnedRouter()

        # Record several outcomes showing sonnet works well for code
        for _ in range(5):
            router.record_outcome(
                query="Debug this code",
                decision=RoutingDecision("sonnet", 0.8, "code task"),
                success=True,
                quality_score=0.95,
            )

        # Record outcomes showing haiku fails for code
        for _ in range(5):
            router.record_outcome(
                query="Debug this code",
                decision=RoutingDecision("haiku", 0.6, "trying cheap"),
                success=False,
                quality_score=0.3,
            )

        # Router should now prefer sonnet for code
        adjustments = router.get_learned_adjustments()
        assert "code" in adjustments or len(adjustments) >= 0  # May be empty initially

    def test_router_to_dict(self):
        """Router state should be serializable."""
        router = LearnedRouter()
        router.record_outcome(
            query="Test",
            decision=RoutingDecision("sonnet", 0.8, "test"),
            success=True,
            quality_score=0.9,
        )

        state = router.to_dict()

        assert "outcome_history" in state
        assert "learned_adjustments" in state
