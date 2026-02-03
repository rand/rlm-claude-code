"""
RLM-Claude-Code: Recursive Language Model integration for Claude Code.

Transform Claude Code into an RLM agent for unbounded context handling
and improved reasoning via REPL-based context decomposition.

Implements intelligent orchestration with:
- Automatic complexity-based RLM activation
- Claude-powered orchestration decisions
- Strategy learning from successful trajectories
- User-configurable preferences and budgets
"""

__version__ = "0.7.0"

# Async execution (SPEC-08.01-08.06) - from orchestrator package
# Core orchestration
from .auto_activation import AutoActivator, check_auto_activation

# Circuit breaker (SPEC-10.20-10.26)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    FailureRecord,
    FallbackResult,
    RecoveryTest,
    TierCircuitBreaker,
)

# Complexity and routing
from .complexity_classifier import should_activate_rlm

# Compute allocation (SPEC-07.10-07.15)
from .compute_allocation import (
    AllocationReasoning,
    ComputeAllocation,
    ComputeAllocator,
    DifficultyEstimate,
    ModelTier,
    TaskType,
)

# Confidence-weighted synthesis (SPEC-10.01-10.06)
from .confidence_synthesis import (
    ConfidenceEstimator,
    ConfidenceFlags,
    RecursiveResult,
    SynthesisResult,
    SynthesisStrategy,
    WeightedSynthesizer,
)

# Context compression (SPEC-08.20-08.25)
from .context_compression import (
    AbstractiveCompressor,
    AutoCompressor,
    CompressionConfig,
    CompressionMetrics,
    CompressionResult,
    CompressionStage,
    ContextCompressor,
    ExtractiveCompressor,
    KeyInfoPreserver,
    RelevanceScorer,
    TwoStageCompressor,
)

# Context enrichment (SPEC-06.30-06.35)
from .context_enrichment import (
    CodeTaskEnricher,
    ContextEnricher,
    DebugTaskEnricher,
    EnrichmentConfig,
    EnrichmentResult,
    EnrichmentStrategy,
    IntentClassifier,
    QueryIntent,
)

# Context indexing (SPEC-01.04 - Phase 4)
from .context_index import ContextIndex, FileIndex, IndexStats

# Context and REPL
from .context_manager import (
    LazyContext,
    LazyContextConfig,
    LazyContextVariable,
    LazyFileLoader,
    create_lazy_context,
    externalize_context,
)

# Continuous learning (SPEC-06.40-06.45)
from .continuous_learning import (
    ContinuousLearner,
    ExecutionOutcome,
    LearnerConfig,
    LearnerState,
    LearningSignal,
    MetaLearner,
    OutcomeRecorder,
    PredictionRecord,
    SignalExtractor,
    SignalType,
)

# Cross-session memory promotion (SPEC-09.10-09.15)
from .cross_session_promotion import (
    AccessRecord,
    CrossSessionPromoter,
    CrossSessionTracker,
    PromotionCandidate,
    PromotionConfig,
    PromotionCriteria,
    PromotionDecision,
    PromotionLog,
    PromotionReason,
)

# Embedding retrieval (SPEC-09.01-09.07)
from .embedding_retrieval import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingStore,
    HybridRetriever,
    HybridSearchResult,
    MockEmbeddingProvider,
)

# Enhanced budget tracking (SPEC-05)
from .enhanced_budget import (
    AdaptiveDepthRecommendation,
    BudgetAlert,
    BudgetLimits,
    BurnRateAlert,
    BurnRateMetrics,
    CostSample,
    DepthBudgetWarning,
    EnhancedBudgetMetrics,
    EnhancedBudgetTracker,
)

# Epistemic verification (SPEC-16)
from .epistemic import (
    AuditResult,
    BatchAuditResult,
    ClaimExtractor,
    ClaimVerification,
    EpistemicGap,
    EvidenceAuditor,
    ExtractedClaim,
    ExtractionResult,
    FlagReason,
    GapType,
    HallucinationReport,
    OnFailureAction,
    VerificationConfig,
    VerificationMode,
    compute_evidence_support,
    extract_evidence_references,
)

# Execution guarantees (SPEC-10.10-10.15)
from .execution_guarantees import (
    CheckResult,
    ExecutionGuarantees,
    GracefulDegradationPlan,
    GuaranteeChecker,
    GuaranteeViolation,
    ViolationType,
)

# Formal verification (SPEC-07.20-07.25)
from .formal_verification import (
    AutoCorrector,
    CodeChange,
    Constraint,
    ConstraintType,
    CorrectionAttempt,
    CorrectionResult,
    PostconditionGenerator,
    PreconditionGenerator,
    RefactoringPostconditions,
    VerificationChain,
    VerificationResult,
    VerificationStatus,
    Verifier,
)

# LATS orchestration (SPEC-06.10-06.15)
from .lats_orchestration import (
    ActionNode,
    LATSConfig,
    LATSOrchestrator,
    LATSPhase,
    ToolCapability,
    ToolCapabilityMatrix,
    ToolPlan,
    compute_ucb1,
)

# Learned routing (SPEC-06.20-06.26)
from .learned_routing import (
    CascadingRouter,
    DifficultyEstimate,
    DifficultyEstimator,
    LearnedRouter,
    ModelProfile,
    OutcomeRecord,
    QueryFeatures,
    RouterConfig,
    RoutingDecision,
)
from .local_orchestrator import RECOMMENDED_CONFIGS, LocalModelConfig, LocalOrchestrator

# Memory backend abstraction (SPEC-12.20-12.25)
from .memory_backend import (
    Edge,
    EdgeType,
    InMemoryBackend,
    MemoryBackend,
    MemoryBackendConfig,
    MemoryBackendType,
    MemoryMigrator,
    MigrationStats,
    NodeType,
    SQLiteBackend,
    create_backend,
)
from .memory_evolution import ConsolidationResult, DecayResult, MemoryEvolution, PromotionResult

# Memory system (SPEC-02, SPEC-03)
from .memory_store import ConfidenceUpdate, Hyperedge, MemoryStore, Node, SearchResult
from .orchestration_logger import (
    LoggerConfig,
    OrchestrationDecisionLog,
    OrchestrationLogger,
    TrainingDataExporter,
    get_logger,
    set_logger,
)
from .orchestration_schema import ExecutionMode, OrchestrationPlan, ToolAccessLevel

# Orchestration telemetry (Feature 3e0.6)
from .orchestration_telemetry import (
    HeuristicAccuracy,
    HeuristicOutcome,
    OrchestrationTelemetry,
    TelemetryConfig,
    TelemetryDecisionLog,
    TelemetryReport,
)

# Intelligent orchestration - from orchestrator package
# Core orchestration - from orchestrator package
# Checkpointing and steering - from orchestrator package (SPEC-12.05-12.06)
from .orchestrator import (
    AsyncExecutor,
    AsyncRLMOrchestrator,
    AutoSteeringPolicy,
    BudgetChecker,
    CheckpointingOrchestrator,
    ExecutionResult,
    IntelligentOrchestrator,
    InteractiveOrchestrator,
    OrchestrationState,
    OrchestratorConfig,
    PartialFailureResult,
    RLMCheckpoint,
    RLMOrchestrator,
    SpeculativeExecution,
    SpeculativeResult,
    SteeringDecision,
    SteeringPoint,
)

# Proactive computation (SPEC-06.01-06.05)
from .proactive_computation import (
    ComputationAdvisor,
    ComputationHelper,
    ComputationSuggestion,
    ComputationTrigger,
    DetectionResult,
)

# Progress reporting (SPEC-01.05 - Phase 4)
from .progress import (
    CancellationToken,
    CancelledException,
    ConsoleProgressCallback,
    NullProgressCallback,
    OperationType,
    ProgressCallback,
    ProgressContext,
    ProgressStats,
    ProgressUpdate,
    ThrottledProgressCallback,
    create_progress_context,
)

# Progressive trajectory (SPEC-11.01-11.06)
from .progressive_trajectory import (
    CostAttribution,
    CostBreakdown,
    ProgressiveTrajectory,
    render_cost_breakdown,
    render_detail,
    render_overview,
    render_summary,
)

# Prompt caching (SPEC-08.10-08.15)
from .prompt_caching import (
    CacheablePrompt,
    CacheMetrics,
    CachePrefix,
    CachePrefixRegistry,
    PromptCacheManager,
    build_cacheable_prompt,
)

# Reasoning traces (SPEC-04)
from .reasoning_traces import DecisionNode, DecisionTree, EvidenceScore, ReasoningTraces
from .repl_environment import RLMEnvironment

# REPL plugin system (SPEC-12.10-12.16)
from .repl_plugin import (
    BuiltinPlugins,
    PluginConflictError,
    PluginRegistry,
    REPLPlugin,
    REPLPluginManager,
)

# Smart RLM pipeline (SPEC-06.50-06.53)
from .smart_pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    PipelineTelemetry,
    SmartPipeline,
    StageResult,
    StageTelemetry,
)
from .strategy_cache import StrategyCache

# Tokenization (SPEC-01.01 - Phase 4)
from .tokenization import (
    Chunk,
    ChunkingConfig,
    chunk_by_tokens,
    count_tokens,
    detect_language,
    find_semantic_boundaries,
    partition_content_by_tokens,
    token_aware_chunk,
)
from .tool_bridge import ToolBridge, ToolPermissions

# Trajectory and analysis
from .trajectory import TrajectoryEvent, TrajectoryRenderer
from .trajectory_analysis import StrategyType, TrajectoryAnalyzer

# Tree of Thoughts (SPEC-07.01-07.07)
from .tree_of_thoughts import (
    SearchStrategy,
    ThoughtNode,
    ThoughtTree,
    ToTConfig,
)
from .types import SessionContext

# User corrections (SPEC-11.20-11.25)
from .user_corrections import (
    ClassifierAdjustment,
    Correction,
    CorrectionAnalyzer,
    CorrectionRecorder,
    CorrectionType,
    UserCorrectionLearner,
)

# User preferences and tools
from .user_preferences import PreferencesManager, UserPreferences

__all__ = [
    # Async execution (SPEC-08.01-08.06)
    "AsyncExecutor",
    "AsyncRLMOrchestrator",
    "BudgetChecker",
    "ExecutionResult",
    "PartialFailureResult",
    "SpeculativeExecution",
    "SpeculativeResult",
    # Core orchestration
    "RLMOrchestrator",
    "OrchestrationState",
    "IntelligentOrchestrator",
    "OrchestratorConfig",
    # Checkpointing (SPEC-12.05)
    "CheckpointingOrchestrator",
    "RLMCheckpoint",
    # Steering (SPEC-12.06)
    "AutoSteeringPolicy",
    "InteractiveOrchestrator",
    "SteeringDecision",
    "SteeringPoint",
    "AutoActivator",
    "check_auto_activation",
    # Compute allocation (SPEC-07.10-07.15)
    "AllocationReasoning",
    "ComputeAllocation",
    "ComputeAllocator",
    "DifficultyEstimate",
    "ModelTier",
    "TaskType",
    # Context
    "SessionContext",
    "externalize_context",
    "RLMEnvironment",
    # Lazy context loading (SPEC-01.02)
    "LazyContextVariable",
    "LazyContextConfig",
    "LazyFileLoader",
    "LazyContext",
    "create_lazy_context",
    # Complexity
    "should_activate_rlm",
    "ExecutionMode",
    # Confidence-weighted synthesis (SPEC-10.01-10.06)
    "ConfidenceEstimator",
    "ConfidenceFlags",
    "RecursiveResult",
    "SynthesisResult",
    "SynthesisStrategy",
    "WeightedSynthesizer",
    "OrchestrationPlan",
    "ToolAccessLevel",
    # Local orchestration
    "LocalOrchestrator",
    "LocalModelConfig",
    "RECOMMENDED_CONFIGS",
    # Orchestration logging
    "OrchestrationLogger",
    "LoggerConfig",
    "OrchestrationDecisionLog",
    "TrainingDataExporter",
    "get_logger",
    "set_logger",
    # Trajectory
    "TrajectoryEvent",
    "TrajectoryRenderer",
    "TrajectoryAnalyzer",
    "StrategyType",
    # Preferences and tools
    "UserPreferences",
    "PreferencesManager",
    "ToolBridge",
    "ToolPermissions",
    "StrategyCache",
    # Memory system (SPEC-02, SPEC-03)
    "MemoryStore",
    "Node",
    "Hyperedge",
    "ConfidenceUpdate",
    "SearchResult",
    "MemoryEvolution",
    "ConsolidationResult",
    "PromotionResult",
    "DecayResult",
    # Reasoning traces (SPEC-04)
    "ReasoningTraces",
    "DecisionNode",
    "DecisionTree",
    "EvidenceScore",
    # Embedding retrieval (SPEC-09.01-09.07)
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingStore",
    "HybridRetriever",
    "HybridSearchResult",
    "MockEmbeddingProvider",
    # Execution guarantees (SPEC-10.10-10.15)
    "CheckResult",
    "ExecutionGuarantees",
    "GracefulDegradationPlan",
    "GuaranteeChecker",
    "GuaranteeViolation",
    "ViolationType",
    # Enhanced budget (SPEC-05)
    "EnhancedBudgetTracker",
    "EnhancedBudgetMetrics",
    "BudgetLimits",
    "BudgetAlert",
    # Burn rate monitoring (Feature 3e0.5)
    "BurnRateMetrics",
    "BurnRateAlert",
    "CostSample",
    # Adaptive depth budgeting (Feature 3e0.4)
    "AdaptiveDepthRecommendation",
    "DepthBudgetWarning",
    # Orchestration telemetry (Feature 3e0.6)
    "OrchestrationTelemetry",
    "TelemetryConfig",
    "TelemetryDecisionLog",
    "TelemetryReport",
    "HeuristicAccuracy",
    "HeuristicOutcome",
    # Tokenization (SPEC-01.01 - Phase 4)
    "Chunk",
    "ChunkingConfig",
    "count_tokens",
    "token_aware_chunk",
    "chunk_by_tokens",
    "partition_content_by_tokens",
    "detect_language",
    "find_semantic_boundaries",
    # Context indexing (SPEC-01.04 - Phase 4)
    "ContextIndex",
    "FileIndex",
    "IndexStats",
    # Prompt caching (SPEC-08.10-08.15)
    "CacheablePrompt",
    "CacheMetrics",
    "CachePrefix",
    "CachePrefixRegistry",
    "PromptCacheManager",
    "build_cacheable_prompt",
    # Progressive trajectory (SPEC-11.01-11.06)
    "CostAttribution",
    "CostBreakdown",
    "ProgressiveTrajectory",
    "render_cost_breakdown",
    "render_detail",
    "render_overview",
    "render_summary",
    # Progress reporting (SPEC-01.05 - Phase 4)
    "OperationType",
    "ProgressUpdate",
    "ProgressStats",
    "ProgressCallback",
    "CancellationToken",
    "CancelledException",
    "ProgressContext",
    "create_progress_context",
    "ConsoleProgressCallback",
    "NullProgressCallback",
    "ThrottledProgressCallback",
    # Tree of Thoughts (SPEC-07.01-07.07)
    "SearchStrategy",
    "ThoughtNode",
    "ThoughtTree",
    "ToTConfig",
    # Formal verification (SPEC-07.20-07.25)
    "AutoCorrector",
    "CodeChange",
    "Constraint",
    "ConstraintType",
    "CorrectionAttempt",
    "CorrectionResult",
    "PostconditionGenerator",
    "PreconditionGenerator",
    "RefactoringPostconditions",
    "VerificationChain",
    "VerificationResult",
    "VerificationStatus",
    "Verifier",
    # Proactive computation (SPEC-06.01-06.05)
    "ComputationAdvisor",
    "ComputationHelper",
    "ComputationSuggestion",
    "ComputationTrigger",
    "DetectionResult",
    # LATS orchestration (SPEC-06.10-06.15)
    "ActionNode",
    "LATSConfig",
    "LATSOrchestrator",
    "LATSPhase",
    "ToolCapability",
    "ToolCapabilityMatrix",
    "ToolPlan",
    "compute_ucb1",
    # Learned routing (SPEC-06.20-06.26)
    "CascadingRouter",
    "DifficultyEstimate",
    "DifficultyEstimator",
    "LearnedRouter",
    "ModelProfile",
    "OutcomeRecord",
    "QueryFeatures",
    "RouterConfig",
    "RoutingDecision",
    # Context enrichment (SPEC-06.30-06.35)
    "CodeTaskEnricher",
    "ContextEnricher",
    "DebugTaskEnricher",
    "EnrichmentConfig",
    "EnrichmentResult",
    "EnrichmentStrategy",
    "IntentClassifier",
    "QueryIntent",
    # Continuous learning (SPEC-06.40-06.45)
    "ContinuousLearner",
    "ExecutionOutcome",
    "LearnerConfig",
    "LearnerState",
    "LearningSignal",
    "MetaLearner",
    "OutcomeRecorder",
    "PredictionRecord",
    "SignalExtractor",
    "SignalType",
    # Smart RLM pipeline (SPEC-06.50-06.53)
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "PipelineTelemetry",
    "SmartPipeline",
    "StageResult",
    "StageTelemetry",
    # Context compression (SPEC-08.20-08.25)
    "AbstractiveCompressor",
    "AutoCompressor",
    "CompressionConfig",
    "CompressionMetrics",
    "CompressionResult",
    "CompressionStage",
    "ContextCompressor",
    "ExtractiveCompressor",
    "KeyInfoPreserver",
    "RelevanceScorer",
    "TwoStageCompressor",
    # Cross-session memory promotion (SPEC-09.10-09.15)
    "AccessRecord",
    "CrossSessionPromoter",
    "CrossSessionTracker",
    "PromotionCandidate",
    "PromotionConfig",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionLog",
    "PromotionReason",
    # Circuit breaker (SPEC-10.20-10.26)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitState",
    "FailureRecord",
    "FallbackResult",
    "RecoveryTest",
    "TierCircuitBreaker",
    # User corrections (SPEC-11.20-11.25)
    "ClassifierAdjustment",
    "Correction",
    "CorrectionAnalyzer",
    "CorrectionRecorder",
    "CorrectionType",
    "UserCorrectionLearner",
    # REPL plugin system (SPEC-12.10-12.16)
    "BuiltinPlugins",
    "PluginConflictError",
    "PluginRegistry",
    "REPLPlugin",
    "REPLPluginManager",
    # Memory backend abstraction (SPEC-12.20-12.25)
    "Edge",
    "EdgeType",
    "InMemoryBackend",
    "MemoryBackend",
    "MemoryBackendConfig",
    "MemoryBackendType",
    "MemoryMigrator",
    "MigrationStats",
    "NodeType",
    "SQLiteBackend",
    "create_backend",
    # Epistemic verification (SPEC-16)
    "AuditResult",
    "BatchAuditResult",
    "ClaimExtractor",
    "ClaimVerification",
    "EpistemicGap",
    "EvidenceAuditor",
    "ExtractedClaim",
    "ExtractionResult",
    "FlagReason",
    "GapType",
    "HallucinationReport",
    "OnFailureAction",
    "VerificationConfig",
    "VerificationMode",
    "compute_evidence_support",
    "extract_evidence_references",
]
