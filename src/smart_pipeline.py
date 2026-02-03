"""
Smart RLM pipeline integration.

Implements: SPEC-06.50-06.53

Integrates all smarter RLM components into a unified pipeline:
analyze, compute check, enrich, route, plan, execute, learn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .context_enrichment import ContextEnricher
from .continuous_learning import ContinuousLearner, LearnerConfig
from .lats_orchestration import LATSOrchestrator, ToolPlan
from .learned_routing import DifficultyEstimator, LearnedRouter, RoutingDecision
from .proactive_computation import ComputationAdvisor


class PipelineStage(Enum):
    """
    Pipeline stage identifiers.

    Implements: SPEC-06.51
    """

    ANALYZE = "analyze"
    COMPUTE_CHECK = "compute_check"
    ENRICH = "enrich"
    ROUTE = "route"
    PLAN = "plan"
    EXECUTE = "execute"
    LEARN = "learn"


@dataclass
class PipelineConfig:
    """Configuration for the smart pipeline."""

    enable_proactive_compute: bool = True
    enable_context_enrichment: bool = True
    enable_learning: bool = True
    simple_query_max_words: int = 3
    simple_query_patterns: list[str] = field(
        default_factory=lambda: [
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "yes",
            "no",
            "bye",
            "goodbye",
        ]
    )


@dataclass
class StageTelemetry:
    """
    Telemetry for a single pipeline stage.

    Implements: SPEC-06.53
    """

    stage: PipelineStage
    duration_ms: float
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stage": self.stage.value,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class PipelineTelemetry:
    """
    Telemetry for entire pipeline execution.

    Implements: SPEC-06.53
    """

    stages: list[StageTelemetry]
    total_duration_ms: float
    bypassed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stages": [s.to_dict() for s in self.stages],
            "total_duration_ms": self.total_duration_ms,
            "bypassed": self.bypassed,
        }


@dataclass
class StageResult:
    """
    Result from a single pipeline stage.

    Implements: SPEC-06.51
    """

    stage: PipelineStage
    success: bool
    output: dict[str, Any]
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stage": self.stage.value,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class PipelineResult:
    """
    Result from pipeline execution.

    Implements: SPEC-06.50
    """

    success: bool
    stages_run: int = 0
    bypassed: bool = False
    analysis: dict[str, Any] | None = None
    proactive_check_performed: bool = False
    computation_detected: bool = False
    context_enriched: bool = False
    routing_decision: RoutingDecision | None = None
    plan: ToolPlan | None = None
    plan_skipped: bool = False
    execution_completed: bool = False
    learning_recorded: bool = False
    telemetry: PipelineTelemetry | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "stages_run": self.stages_run,
            "bypassed": self.bypassed,
            "analysis": self.analysis,
            "proactive_check_performed": self.proactive_check_performed,
            "computation_detected": self.computation_detected,
            "context_enriched": self.context_enriched,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "plan_skipped": self.plan_skipped,
            "execution_completed": self.execution_completed,
            "learning_recorded": self.learning_recorded,
            "telemetry": self.telemetry.to_dict() if self.telemetry else None,
            "error": self.error,
        }


class SmartPipeline:
    """
    Unified smart RLM pipeline.

    Implements: SPEC-06.50-06.53

    Integrates:
    1. Analyze query → extract features, estimate difficulty
    2. Check proactive computation → use REPL if high confidence
    3. Enrich context → proactively gather information
    4. Route → select optimal model
    5. Plan → generate tool orchestration plan
    6. Execute → run with cascading
    7. Learn → record outcome and update preferences
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """
        Initialize smart pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.difficulty_estimator = DifficultyEstimator()
        self.computation_advisor = ComputationAdvisor()
        self.context_enricher = ContextEnricher()
        self.router = LearnedRouter()
        self.orchestrator = LATSOrchestrator()
        self.learner = ContinuousLearner(config=LearnerConfig(enable_meta_learning=True))

    def is_definitely_simple(self, query: str) -> bool:
        """
        Check if query is definitely simple and can bypass heavy processing.

        Implements: SPEC-06.52

        Args:
            query: The query to check

        Returns:
            True if query is definitely simple
        """
        query_lower = query.lower().strip()

        # Check against known simple patterns
        if query_lower in self.config.simple_query_patterns:
            return True

        # Check word count
        words = query.split()
        if len(words) <= self.config.simple_query_max_words:
            # Short queries without complex indicators
            complex_indicators = [
                "analyze",
                "implement",
                "debug",
                "explain",
                "compare",
                "optimize",
                "refactor",
                "design",
                "architect",
            ]
            if not any(ind in query_lower for ind in complex_indicators):
                return True

        return False

    def process(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Process query through the smart pipeline.

        Implements: SPEC-06.50

        Args:
            query: The query to process
            context: Optional context information

        Returns:
            PipelineResult with all stage outputs
        """
        start_time = time.time()
        context = context or {}
        stage_telemetry: list[StageTelemetry] = []
        stages_run = 0

        # Check for simple query bypass (SPEC-06.52)
        if self.is_definitely_simple(query):
            total_duration = (time.time() - start_time) * 1000
            return PipelineResult(
                success=True,
                stages_run=0,
                bypassed=True,
                telemetry=PipelineTelemetry(
                    stages=[],
                    total_duration_ms=total_duration,
                    bypassed=True,
                ),
            )

        result = PipelineResult(success=True)

        # Stage 1: Analyze (SPEC-06.50 Step 1)
        analyze_result = self._run_analyze(query)
        stage_telemetry.append(self._create_telemetry(PipelineStage.ANALYZE, analyze_result))
        stages_run += 1
        result.analysis = analyze_result.output

        # Stage 2: Compute Check (SPEC-06.50 Step 2)
        if self.config.enable_proactive_compute:
            compute_result = self._run_compute_check(query)
            stage_telemetry.append(
                self._create_telemetry(PipelineStage.COMPUTE_CHECK, compute_result)
            )
            stages_run += 1
            result.proactive_check_performed = True
            result.computation_detected = compute_result.output.get("detected", False)

        # Stage 3: Enrich (SPEC-06.50 Step 3)
        if self.config.enable_context_enrichment:
            enrich_result = self._run_enrich(query, context)
            stage_telemetry.append(self._create_telemetry(PipelineStage.ENRICH, enrich_result))
            stages_run += 1
            result.context_enriched = True
            context = enrich_result.output.get("enriched_context", context)

        # Stage 4: Route (SPEC-06.50 Step 4)
        route_result = self._run_route(query)
        stage_telemetry.append(self._create_telemetry(PipelineStage.ROUTE, route_result))
        stages_run += 1
        result.routing_decision = route_result.output.get("decision")

        # Stage 5: Plan (SPEC-06.50 Step 5)
        plan_result = self._run_plan(query)
        stage_telemetry.append(self._create_telemetry(PipelineStage.PLAN, plan_result))
        stages_run += 1
        result.plan = plan_result.output.get("plan")
        result.plan_skipped = plan_result.output.get("skipped", False)

        # Stage 6: Execute (SPEC-06.50 Step 6)
        execute_result = self._run_execute(query)
        stage_telemetry.append(self._create_telemetry(PipelineStage.EXECUTE, execute_result))
        stages_run += 1
        result.execution_completed = execute_result.success

        # Stage 7: Learn (SPEC-06.50 Step 7)
        if self.config.enable_learning:
            learn_result = self._run_learn(
                query=query,
                outcome={"success": result.execution_completed},
                routing_decision=result.routing_decision,
            )
            stage_telemetry.append(self._create_telemetry(PipelineStage.LEARN, learn_result))
            stages_run += 1
            result.learning_recorded = learn_result.success

        # Build telemetry (SPEC-06.53)
        total_duration = (time.time() - start_time) * 1000
        result.stages_run = stages_run
        result.telemetry = PipelineTelemetry(
            stages=stage_telemetry,
            total_duration_ms=total_duration,
        )

        return result

    def run_stage(
        self,
        stage: PipelineStage,
        query: str,
        context: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
    ) -> StageResult:
        """
        Run a single pipeline stage independently.

        Implements: SPEC-06.51

        Args:
            stage: The stage to run
            query: The query
            context: Optional context
            outcome: Optional outcome (for LEARN stage)

        Returns:
            StageResult from the stage
        """
        context = context or {}

        if stage == PipelineStage.ANALYZE:
            return self._run_analyze(query)
        elif stage == PipelineStage.COMPUTE_CHECK:
            return self._run_compute_check(query)
        elif stage == PipelineStage.ENRICH:
            return self._run_enrich(query, context)
        elif stage == PipelineStage.ROUTE:
            return self._run_route(query)
        elif stage == PipelineStage.PLAN:
            return self._run_plan(query)
        elif stage == PipelineStage.EXECUTE:
            return self._run_execute(query)
        elif stage == PipelineStage.LEARN:
            return self._run_learn(query, outcome or {})
        else:
            return StageResult(
                stage=stage,
                success=False,
                output={},
                error=f"Unknown stage: {stage}",
            )

    def _run_analyze(self, query: str) -> StageResult:
        """Run analyze stage."""
        start = time.time()
        try:
            difficulty = self.difficulty_estimator.estimate(query)
            output = {
                "difficulty": difficulty.overall_difficulty(),
                "features": {
                    "reasoning_depth": difficulty.reasoning_depth,
                    "domain_specificity": difficulty.domain_specificity,
                    "ambiguity_level": difficulty.ambiguity_level,
                    "context_size": difficulty.context_size,
                },
            }
            return StageResult(
                stage=PipelineStage.ANALYZE,
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.ANALYZE,
                success=False,
                output={},
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _run_compute_check(self, query: str) -> StageResult:
        """Run compute check stage."""
        start = time.time()
        try:
            detection = self.computation_advisor.detect(query)
            output = {
                "detected": detection.detected,
                "triggers": [t.value for t in detection.triggers],
                "confidence": detection.confidence,
            }
            return StageResult(
                stage=PipelineStage.COMPUTE_CHECK,
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.COMPUTE_CHECK,
                success=False,
                output={"detected": False},
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _run_enrich(self, query: str, context: dict[str, Any]) -> StageResult:
        """Run enrich stage."""
        start = time.time()
        try:
            enrichment = self.context_enricher.enrich(query, context)
            output = {
                "enriched_context": enrichment.enriched_context,
                "intent": enrichment.detected_intent.value,
                "additions": enrichment.additions,
                "token_count": enrichment.token_count,
            }
            return StageResult(
                stage=PipelineStage.ENRICH,
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.ENRICH,
                success=False,
                output={"enriched_context": context},
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _run_route(self, query: str) -> StageResult:
        """Run route stage."""
        start = time.time()
        try:
            decision = self.router.route(query)
            output = {
                "decision": decision,
                "model": decision.model,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            }
            return StageResult(
                stage=PipelineStage.ROUTE,
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.ROUTE,
                success=False,
                output={},
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _run_plan(self, query: str) -> StageResult:
        """Run plan stage."""
        start = time.time()
        try:
            # Simple planning - in production would use full LATS
            plan = self.orchestrator.create_plan(query)
            output = {
                "plan": plan,
                "skipped": plan is None,
            }
            return StageResult(
                stage=PipelineStage.PLAN,
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.PLAN,
                success=False,
                output={"skipped": True},
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _run_execute(self, query: str) -> StageResult:
        """Run execute stage."""
        start = time.time()
        # Simulated execution - in production would actually execute
        output = {
            "executed": True,
            "result": "Execution simulated",
        }
        return StageResult(
            stage=PipelineStage.EXECUTE,
            success=True,
            output=output,
            duration_ms=(time.time() - start) * 1000,
        )

    def _run_learn(
        self,
        query: str,
        outcome: dict[str, Any],
        routing_decision: RoutingDecision | None = None,
    ) -> StageResult:
        """Run learn stage."""
        start = time.time()
        try:
            model = routing_decision.model if routing_decision else "sonnet"
            self.learner.record_outcome(
                query=query,
                features={"query_type": "general"},
                strategy="pipeline",
                model=model,
                depth=1,
                tools=[],
                success=outcome.get("success", True),
                cost=0.001,
                latency_ms=100,
            )
            output = {
                "recorded": True,
                "adjustments": self.learner.get_routing_adjustments(),
            }
            return StageResult(
                stage=PipelineStage.LEARN,
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.LEARN,
                success=False,
                output={"recorded": False},
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _create_telemetry(
        self,
        stage: PipelineStage,
        result: StageResult,
    ) -> StageTelemetry:
        """Create telemetry for a stage result."""
        return StageTelemetry(
            stage=stage,
            duration_ms=result.duration_ms,
            success=result.success,
            error=result.error,
        )


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "PipelineTelemetry",
    "SmartPipeline",
    "StageResult",
    "StageTelemetry",
]
