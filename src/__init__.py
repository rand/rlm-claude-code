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

__version__ = "0.2.0"

# Async execution (SPEC-08.01-08.06)
from .async_executor import (
    AsyncExecutor,
    AsyncRLMOrchestrator,
    BudgetChecker,
    ExecutionResult,
    PartialFailureResult,
    SpeculativeExecution,
    SpeculativeResult,
)

# Core orchestration
from .auto_activation import AutoActivator, check_auto_activation

# Complexity and routing
from .complexity_classifier import should_activate_rlm

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
from .intelligent_orchestrator import IntelligentOrchestrator
from .local_orchestrator import RECOMMENDED_CONFIGS, LocalModelConfig, LocalOrchestrator
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
from .orchestrator import RLMOrchestrator

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

# Reasoning traces (SPEC-04)
from .reasoning_traces import DecisionNode, DecisionTree, EvidenceScore, ReasoningTraces
from .repl_environment import RLMEnvironment
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
from .types import SessionContext

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
    # Core
    "RLMOrchestrator",
    "IntelligentOrchestrator",
    "AutoActivator",
    "check_auto_activation",
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
]
