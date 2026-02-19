from __future__ import annotations

class ClaimCategory:
    Factual: ClaimCategory
    CodeBehavior: ClaimCategory
    Relational: ClaimCategory
    Numerical: ClaimCategory
    Temporal: ClaimCategory
    UserIntent: ClaimCategory
    MetaReasoning: ClaimCategory
    Unknown: ClaimCategory

class GroundingStatus:
    Grounded: GroundingStatus
    WeaklyGrounded: GroundingStatus
    Ungrounded: GroundingStatus
    Uncertain: GroundingStatus

class EvidenceType:
    Citation: EvidenceType
    CodeRef: EvidenceType
    ToolOutput: EvidenceType
    UserStatement: EvidenceType
    Inference: EvidenceType
    Prior: EvidenceType

class VerificationVerdict:
    Verified: VerificationVerdict
    PartiallyVerified: VerificationVerdict
    Unverified: VerificationVerdict
    Error: VerificationVerdict

class Probability:
    estimate: float
    lower: float
    upper: float
    n_samples: int

    def __init__(self, p: float) -> None: ...
    @staticmethod
    def point(p: float) -> Probability: ...
    @staticmethod
    def from_samples(agreeing: int, total: int) -> Probability: ...
    def kl_divergence(self, other: Probability) -> float: ...
    def uncertainty(self) -> float: ...

class EvidenceRef:
    id: str
    evidence_type: EvidenceType
    description: str
    strength: float

    def __init__(self, id: str, evidence_type: EvidenceType, description: str) -> None: ...
    def with_strength(self, strength: float) -> EvidenceRef: ...

class Claim:
    id: str
    text: str
    category: ClaimCategory
    specificity: float

    def __init__(self, text: str, category: ClaimCategory) -> None: ...
    def with_specificity(self, specificity: float) -> Claim: ...
    def required_bits(self) -> float: ...

class BudgetResult:
    claim_id: str
    p0: Probability
    p1: Probability
    observed_bits: float
    required_bits: float
    budget_gap: float
    status: GroundingStatus
    confidence: float

    def __init__(
        self,
        claim_id: str,
        p0: Probability,
        p1: Probability,
        required_bits: float,
    ) -> None: ...
    def is_grounded(self) -> bool: ...
    def should_flag(self, threshold: float) -> bool: ...

class VerificationConfig:
    n_samples: int
    hallucination_threshold: float
    max_latency_ms: int

    def __init__(self) -> None: ...
    @staticmethod
    def fast() -> VerificationConfig: ...
    @staticmethod
    def thorough() -> VerificationConfig: ...

class VerificationStats:
    total_claims: int
    grounded_claims: int
    weakly_grounded_claims: int
    ungrounded_claims: int
    uncertain_claims: int
    avg_budget_gap: float
    max_budget_gap: float

    def hallucination_rate(self) -> float: ...
    def grounding_rate(self) -> float: ...

class ClaimExtractor:
    def __init__(self) -> None: ...
    def extract(self, text: str) -> list[Claim]: ...

class KL:
    @staticmethod
    def bernoulli_kl_bits(p: float, q: float) -> float: ...
    @staticmethod
    def binary_entropy_bits(p: float) -> float: ...
    @staticmethod
    def required_bits_for_specificity(specificity: float) -> float: ...
    @staticmethod
    def jensen_shannon_bits(p: float, q: float) -> float: ...
    @staticmethod
    def surprise_bits(p: float) -> float: ...
    @staticmethod
    def aggregate_evidence_bits(bits_list: list[float]) -> float: ...
