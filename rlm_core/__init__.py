"""rlm_core — Rust-native RLM orchestration library with Python bindings.

This package provides Python access to the rlm-core Rust library via PyO3.
All types are defined in the compiled extension module; this file re-exports
them for convenient access and provides a graceful error message if the
native extension is unavailable.
"""

from __future__ import annotations

try:
    from rlm_core.rlm_core import *  # noqa: F401, F403
    from rlm_core.rlm_core import (
        # Module-level functions
        available_features,
        has_feature,
        quick_hallucination_check,
        version,
        version_tuple,
        # Context
        Message,
        Role,
        SessionContext,
        ToolOutput,
        # Memory
        HyperEdge,
        MemoryStats,
        MemoryStore,
        Node,
        NodeType,
        Tier,
        # LLM
        ChatMessage,
        CompletionRequest,
        CompletionResponse,
        CostTracker,
        ModelSpec,
        ModelTier,
        Provider,
        QueryType,
        RoutingContext,
        RoutingDecision,
        SmartRouter,
        TokenUsage,
        # Trajectory
        TrajectoryEvent,
        TrajectoryEventType,
        # Complexity
        ActivationDecision,
        PatternClassifier,
        # Epistemic
        BudgetResult,
        Claim,
        ClaimCategory,
        ClaimExtractor,
        EvidenceRef,
        EvidenceType,
        GroundingStatus,
        KL,
        Probability,
        VerificationConfig,
        VerificationStats,
        VerificationVerdict,
    )
except ImportError as _e:
    _msg = (
        "rlm_core native extension is not available. "
        "This typically means the Rust library was not compiled for your platform. "
        "Install from a pre-built wheel: pip install rlm-claude-code\n"
        f"Original error: {_e}"
    )
    raise ImportError(_msg) from _e

__all__ = [
    # Functions
    "available_features",
    "has_feature",
    "quick_hallucination_check",
    "version",
    "version_tuple",
    # Context
    "Message",
    "Role",
    "SessionContext",
    "ToolOutput",
    # Memory
    "HyperEdge",
    "MemoryStats",
    "MemoryStore",
    "Node",
    "NodeType",
    "Tier",
    # LLM
    "ChatMessage",
    "CompletionRequest",
    "CompletionResponse",
    "CostTracker",
    "ModelSpec",
    "ModelTier",
    "Provider",
    "QueryType",
    "RoutingContext",
    "RoutingDecision",
    "SmartRouter",
    "TokenUsage",
    # Trajectory
    "TrajectoryEvent",
    "TrajectoryEventType",
    # Complexity
    "ActivationDecision",
    "PatternClassifier",
    # Epistemic
    "BudgetResult",
    "Claim",
    "ClaimCategory",
    "ClaimExtractor",
    "EvidenceRef",
    "EvidenceType",
    "GroundingStatus",
    "KL",
    "Probability",
    "VerificationConfig",
    "VerificationStats",
    "VerificationVerdict",
]
