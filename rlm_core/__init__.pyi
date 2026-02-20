"""Type stubs for the rlm_core native extension module."""

from __future__ import annotations
from enum import IntEnum

# Re-export everything from submodules for flat access
from rlm_core._context import Message as Message
from rlm_core._context import Role as Role
from rlm_core._context import SessionContext as SessionContext
from rlm_core._context import ToolOutput as ToolOutput
from rlm_core._memory import HyperEdge as HyperEdge
from rlm_core._memory import MemoryStats as MemoryStats
from rlm_core._memory import MemoryStore as MemoryStore
from rlm_core._memory import Node as Node
from rlm_core._memory import NodeType as NodeType
from rlm_core._memory import Tier as Tier
from rlm_core._llm import ChatMessage as ChatMessage
from rlm_core._llm import CompletionRequest as CompletionRequest
from rlm_core._llm import CompletionResponse as CompletionResponse
from rlm_core._llm import CostTracker as CostTracker
from rlm_core._llm import ModelSpec as ModelSpec
from rlm_core._llm import ModelTier as ModelTier
from rlm_core._llm import Provider as Provider
from rlm_core._llm import QueryType as QueryType
from rlm_core._llm import RoutingContext as RoutingContext
from rlm_core._llm import RoutingDecision as RoutingDecision
from rlm_core._llm import SmartRouter as SmartRouter
from rlm_core._llm import TokenUsage as TokenUsage
from rlm_core._trajectory import TrajectoryEvent as TrajectoryEvent
from rlm_core._trajectory import TrajectoryEventType as TrajectoryEventType
from rlm_core._complexity import ActivationDecision as ActivationDecision
from rlm_core._complexity import PatternClassifier as PatternClassifier
from rlm_core._epistemic import BudgetResult as BudgetResult
from rlm_core._epistemic import Claim as Claim
from rlm_core._epistemic import ClaimCategory as ClaimCategory
from rlm_core._epistemic import ClaimExtractor as ClaimExtractor
from rlm_core._epistemic import EvidenceRef as EvidenceRef
from rlm_core._epistemic import EvidenceType as EvidenceType
from rlm_core._epistemic import GroundingStatus as GroundingStatus
from rlm_core._epistemic import KL as KL
from rlm_core._epistemic import Probability as Probability
from rlm_core._epistemic import VerificationConfig as VerificationConfig
from rlm_core._epistemic import VerificationStats as VerificationStats
from rlm_core._epistemic import VerificationVerdict as VerificationVerdict

__version__: str

def version() -> str: ...
def version_tuple() -> tuple[int, int, int]: ...
def has_feature(feature_name: str) -> bool: ...
def available_features() -> list[str]: ...
def quick_hallucination_check(response: str) -> float: ...

class IssueSeverity(IntEnum):
    Critical = 0
    High = 1
    Medium = 2
    Low = 3
    Info = 4

class ValidationContext: ...
class ValidationResult: ...
class AdversarialConfig: ...
