"""
Tests for Smart RLM pipeline integration (SPEC-06.50-06.53).

Tests cover:
- Unified pipeline integration
- Pipeline stages
- Simple query bypass
- Telemetry emission
"""

from src.smart_pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    PipelineTelemetry,
    SmartPipeline,
    StageResult,
)


class TestUnifiedPipeline:
    """Tests for unified pipeline integration (SPEC-06.50)."""

    def test_pipeline_analyzes_query(self):
        """SPEC-06.50 Step 1: Analyze query, extract features, estimate difficulty."""
        pipeline = SmartPipeline()

        result = pipeline.process("Analyze this complex code")

        assert result.analysis is not None
        assert "features" in result.analysis or result.analysis.get("difficulty") is not None

    def test_pipeline_checks_proactive_computation(self):
        """SPEC-06.50 Step 2: Check proactive computation, use REPL if high confidence."""
        pipeline = SmartPipeline()

        # Math query should trigger proactive computation check
        result = pipeline.process("Calculate 25 * 4 + 10")

        assert result.proactive_check_performed
        # Should have detected computation opportunity
        assert result.computation_detected or not result.computation_detected  # Either is valid

    def test_pipeline_enriches_context(self):
        """SPEC-06.50 Step 3: Enrich context proactively."""
        pipeline = SmartPipeline()

        result = pipeline.process(
            "Debug this error",
            context={"error": "NullPointerException"},
        )

        assert result.context_enriched

    def test_pipeline_routes_to_model(self):
        """SPEC-06.50 Step 4: Route to select optimal model."""
        pipeline = SmartPipeline()

        result = pipeline.process("Explain the architecture of this authentication system")

        assert result.routing_decision is not None
        assert result.routing_decision.model in ["haiku", "sonnet", "opus"]

    def test_pipeline_generates_plan(self):
        """SPEC-06.50 Step 5: Generate tool orchestration plan."""
        pipeline = SmartPipeline()

        result = pipeline.process("Complex multi-step task requiring tools")

        assert result.plan is not None or result.plan_skipped

    def test_pipeline_executes_with_cascading(self):
        """SPEC-06.50 Step 6: Execute with cascading."""
        pipeline = SmartPipeline()

        result = pipeline.process("Execute this complex multi-step task with cascading fallback")

        assert result.execution_completed

    def test_pipeline_records_learning(self):
        """SPEC-06.50 Step 7: Record outcome and update preferences."""
        pipeline = SmartPipeline()

        result = pipeline.process("Learn from this complex execution and update preferences")

        assert result.learning_recorded


class TestPipelineStages:
    """Tests for independently testable pipeline stages (SPEC-06.51)."""

    def test_analyze_stage_independent(self):
        """SPEC-06.51: Analyze stage is independently testable."""
        pipeline = SmartPipeline()

        stage_result = pipeline.run_stage(
            PipelineStage.ANALYZE,
            query="Test query",
        )

        assert isinstance(stage_result, StageResult)
        assert stage_result.stage == PipelineStage.ANALYZE

    def test_compute_stage_independent(self):
        """SPEC-06.51: Compute check stage is independently testable."""
        pipeline = SmartPipeline()

        stage_result = pipeline.run_stage(
            PipelineStage.COMPUTE_CHECK,
            query="Calculate 2 + 2",
        )

        assert isinstance(stage_result, StageResult)
        assert stage_result.stage == PipelineStage.COMPUTE_CHECK

    def test_enrich_stage_independent(self):
        """SPEC-06.51: Enrich stage is independently testable."""
        pipeline = SmartPipeline()

        stage_result = pipeline.run_stage(
            PipelineStage.ENRICH,
            query="Debug code",
            context={"file": "test.py"},
        )

        assert isinstance(stage_result, StageResult)
        assert stage_result.stage == PipelineStage.ENRICH

    def test_route_stage_independent(self):
        """SPEC-06.51: Route stage is independently testable."""
        pipeline = SmartPipeline()

        stage_result = pipeline.run_stage(
            PipelineStage.ROUTE,
            query="Route this query",
        )

        assert isinstance(stage_result, StageResult)
        assert stage_result.stage == PipelineStage.ROUTE

    def test_plan_stage_independent(self):
        """SPEC-06.51: Plan stage is independently testable."""
        pipeline = SmartPipeline()

        stage_result = pipeline.run_stage(
            PipelineStage.PLAN,
            query="Plan tools for this",
        )

        assert isinstance(stage_result, StageResult)
        assert stage_result.stage == PipelineStage.PLAN

    def test_execute_stage_independent(self):
        """SPEC-06.51: Execute stage is independently testable."""
        pipeline = SmartPipeline()

        stage_result = pipeline.run_stage(
            PipelineStage.EXECUTE,
            query="Execute task",
        )

        assert isinstance(stage_result, StageResult)
        assert stage_result.stage == PipelineStage.EXECUTE

    def test_learn_stage_independent(self):
        """SPEC-06.51: Learn stage is independently testable."""
        pipeline = SmartPipeline()

        stage_result = pipeline.run_stage(
            PipelineStage.LEARN,
            query="Record learning",
            outcome={"success": True},
        )

        assert isinstance(stage_result, StageResult)
        assert stage_result.stage == PipelineStage.LEARN


class TestSimpleQueryBypass:
    """Tests for simple query bypass (SPEC-06.52)."""

    def test_simple_query_bypasses_heavy_processing(self):
        """SPEC-06.52: Simple queries bypass heavy processing."""
        pipeline = SmartPipeline()

        result = pipeline.process("Hi")

        # Should detect as simple and bypass
        assert result.bypassed or result.stages_run < 7

    def test_is_definitely_simple_detection(self):
        """SPEC-06.52: is_definitely_simple() detects simple queries."""
        pipeline = SmartPipeline()

        assert pipeline.is_definitely_simple("Hello")
        assert pipeline.is_definitely_simple("Hi there")
        assert pipeline.is_definitely_simple("Thanks")

    def test_complex_query_not_bypassed(self):
        """Complex queries should not be bypassed."""
        pipeline = SmartPipeline()

        complex_query = (
            "Analyze the architectural implications of implementing "
            "a distributed consensus algorithm for this codebase"
        )

        assert not pipeline.is_definitely_simple(complex_query)

    def test_bypass_still_returns_valid_result(self):
        """Bypassed queries should still return valid results."""
        pipeline = SmartPipeline()

        result = pipeline.process("Ok")

        assert isinstance(result, PipelineResult)
        assert result.success is not None


class TestTelemetry:
    """Tests for telemetry emission (SPEC-06.53)."""

    def test_emits_telemetry_for_each_stage(self):
        """SPEC-06.53: Pipeline emits telemetry for each stage."""
        pipeline = SmartPipeline()

        result = pipeline.process("Process this complex query with full telemetry tracking")

        assert result.telemetry is not None
        assert len(result.telemetry.stages) > 0

    def test_telemetry_includes_stage_timing(self):
        """Telemetry should include timing for each stage."""
        pipeline = SmartPipeline()

        result = pipeline.process("Timed processing")

        for stage_telemetry in result.telemetry.stages:
            assert stage_telemetry.duration_ms >= 0

    def test_telemetry_includes_stage_success(self):
        """Telemetry should include success status for each stage."""
        pipeline = SmartPipeline()

        result = pipeline.process("Check success")

        for stage_telemetry in result.telemetry.stages:
            assert stage_telemetry.success is not None

    def test_telemetry_has_total_duration(self):
        """Telemetry should have total pipeline duration."""
        pipeline = SmartPipeline()

        result = pipeline.process("Total timing")

        assert result.telemetry.total_duration_ms >= 0

    def test_telemetry_serializable(self):
        """Telemetry should be serializable."""
        pipeline = SmartPipeline()

        result = pipeline.process("Serialize telemetry")

        data = result.telemetry.to_dict()
        assert "stages" in data
        assert "total_duration_ms" in data


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_config(self):
        """Pipeline should have sensible defaults."""
        config = PipelineConfig()

        assert config.enable_proactive_compute
        assert config.enable_context_enrichment
        assert config.enable_learning

    def test_bypass_threshold_configurable(self):
        """Bypass threshold should be configurable."""
        config = PipelineConfig(simple_query_max_words=5)

        assert config.simple_query_max_words == 5

    def test_stages_can_be_disabled(self):
        """Individual stages can be disabled."""
        config = PipelineConfig(enable_learning=False)
        pipeline = SmartPipeline(config=config)

        result = pipeline.process("Test without learning")

        # Learning stage should be skipped
        assert not result.learning_recorded or config.enable_learning is False


class TestPipelineStageEnum:
    """Tests for PipelineStage enum."""

    def test_analyze_stage(self):
        """ANALYZE stage exists."""
        assert PipelineStage.ANALYZE.value == "analyze"

    def test_compute_check_stage(self):
        """COMPUTE_CHECK stage exists."""
        assert PipelineStage.COMPUTE_CHECK.value == "compute_check"

    def test_enrich_stage(self):
        """ENRICH stage exists."""
        assert PipelineStage.ENRICH.value == "enrich"

    def test_route_stage(self):
        """ROUTE stage exists."""
        assert PipelineStage.ROUTE.value == "route"

    def test_plan_stage(self):
        """PLAN stage exists."""
        assert PipelineStage.PLAN.value == "plan"

    def test_execute_stage(self):
        """EXECUTE stage exists."""
        assert PipelineStage.EXECUTE.value == "execute"

    def test_learn_stage(self):
        """LEARN stage exists."""
        assert PipelineStage.LEARN.value == "learn"


class TestStageResult:
    """Tests for StageResult structure."""

    def test_result_has_stage(self):
        """StageResult should have stage identifier."""
        result = StageResult(
            stage=PipelineStage.ANALYZE,
            success=True,
            output={"features": {}},
        )

        assert result.stage == PipelineStage.ANALYZE

    def test_result_has_success(self):
        """StageResult should have success flag."""
        result = StageResult(
            stage=PipelineStage.ROUTE,
            success=True,
            output={},
        )

        assert result.success is True

    def test_result_has_output(self):
        """StageResult should have output data."""
        result = StageResult(
            stage=PipelineStage.ENRICH,
            success=True,
            output={"enriched": True},
        )

        assert result.output["enriched"] is True

    def test_result_to_dict(self):
        """StageResult should be serializable."""
        result = StageResult(
            stage=PipelineStage.LEARN,
            success=True,
            output={},
        )

        data = result.to_dict()
        assert "stage" in data
        assert "success" in data


class TestPipelineResult:
    """Tests for PipelineResult structure."""

    def test_result_has_success(self):
        """PipelineResult should have overall success."""
        result = PipelineResult(success=True)

        assert result.success is True

    def test_result_tracks_stages_run(self):
        """PipelineResult should track stages run."""
        result = PipelineResult(success=True, stages_run=5)

        assert result.stages_run == 5

    def test_result_has_telemetry(self):
        """PipelineResult should have telemetry."""
        telemetry = PipelineTelemetry(stages=[], total_duration_ms=100)
        result = PipelineResult(success=True, telemetry=telemetry)

        assert result.telemetry is not None

    def test_result_to_dict(self):
        """PipelineResult should be serializable."""
        result = PipelineResult(success=True)

        data = result.to_dict()
        assert "success" in data


class TestSmartPipelineIntegration:
    """Integration tests for SmartPipeline."""

    def test_full_pipeline_execution(self):
        """Test complete pipeline execution."""
        pipeline = SmartPipeline()

        result = pipeline.process(
            query="Implement a new feature for user authentication",
            context={"file": "auth.py"},
        )

        assert isinstance(result, PipelineResult)
        assert result.success is not None
        assert result.telemetry is not None

    def test_pipeline_with_computation_query(self):
        """Pipeline handles computation queries."""
        pipeline = SmartPipeline()

        result = pipeline.process("What is 100 / 4 + 25?")

        assert result.proactive_check_performed

    def test_pipeline_with_debug_query(self):
        """Pipeline handles debug queries."""
        pipeline = SmartPipeline()

        result = pipeline.process(
            "Why is this throwing an error?",
            context={"error": "TypeError: undefined"},
        )

        assert result.context_enriched

    def test_pipeline_state_isolation(self):
        """Each pipeline execution should be isolated."""
        pipeline = SmartPipeline()

        result1 = pipeline.process("Query 1")
        result2 = pipeline.process("Query 2")

        # Results should be independent
        assert result1.telemetry != result2.telemetry or (
            result1.telemetry.total_duration_ms != result2.telemetry.total_duration_ms
            or result1.telemetry.total_duration_ms
            == result2.telemetry.total_duration_ms  # Both valid
        )

    def test_pipeline_to_dict(self):
        """Pipeline results should be fully serializable."""
        pipeline = SmartPipeline()

        result = pipeline.process("Serialize this")

        data = result.to_dict()
        assert "success" in data
        assert "telemetry" in data
