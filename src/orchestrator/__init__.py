"""
Modular orchestrator package.

Implements: SPEC-12.01-12.07

This package provides the modular orchestrator architecture:
- core: Base RLM orchestration loop
- intelligent: Claude-powered decision making
- async_executor: Parallel and speculative execution
- checkpointing: Session persistence
- steering: User interaction support

All public classes are re-exported here for backward compatibility.
"""

# Core orchestration (SPEC-12.02)
# Async execution (SPEC-12.04)
from .async_executor import (
    AsyncExecutor,
    AsyncRLMOrchestrator,
    BudgetChecker,
    ExecutionResult,
    PartialFailureResult,
    SpeculativeExecution,
    SpeculativeResult,
)

# Checkpointing (SPEC-12.05)
from .checkpointing import (
    CheckpointingOrchestrator,
    RLMCheckpoint,
)
from .core import OrchestrationState, RLMOrchestrator

# Intelligent orchestration (SPEC-12.03)
from .intelligent import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    IntelligentOrchestrator,
    OrchestratorConfig,
    create_orchestration_plan,
)

# Steering (SPEC-12.06, SPEC-11.10-11.16)
from .steering import (
    AutoSteeringPolicy,
    InteractiveOrchestrator,
    SteeringCallback,
    SteeringDecision,
    SteeringPoint,
    SteeringPointType,
    SteeringResponse,
)

__all__ = [
    # Core (SPEC-12.02)
    "OrchestrationState",
    "RLMOrchestrator",
    # Intelligent (SPEC-12.03)
    "IntelligentOrchestrator",
    "OrchestratorConfig",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "create_orchestration_plan",
    # Async (SPEC-12.04)
    "AsyncExecutor",
    "AsyncRLMOrchestrator",
    "BudgetChecker",
    "ExecutionResult",
    "PartialFailureResult",
    "SpeculativeExecution",
    "SpeculativeResult",
    # Checkpointing (SPEC-12.05)
    "CheckpointingOrchestrator",
    "RLMCheckpoint",
    # Steering (SPEC-12.06, SPEC-11.10-11.16)
    "AutoSteeringPolicy",
    "InteractiveOrchestrator",
    "SteeringCallback",
    "SteeringDecision",
    "SteeringPoint",
    "SteeringPointType",
    "SteeringResponse",
]
