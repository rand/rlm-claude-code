"""
Deciduous-style reasoning traces with git integration.

Implements: Spec SPEC-04 Reasoning Traces
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .epistemic.types import ClaimVerification, HallucinationReport
    from .memory_store import MemoryStore
    from .trajectory import TrajectoryEvent


class VerificationModel(Protocol):
    """
    Protocol for epistemic verification models.

    Implements: SPEC-16.16

    A verification model takes a claim and its cited evidence,
    and returns a ClaimVerification result with scores indicating
    how well the evidence supports the claim.
    """

    def __call__(
        self,
        claim_text: str,
        evidence: list[dict[str, Any]],
    ) -> ClaimVerification:
        """
        Verify a claim against evidence.

        Args:
            claim_text: The claim to verify
            evidence: List of evidence items, each with:
                - id: Evidence node ID
                - content: Evidence text content
                - confidence: Evidence confidence score (optional)

        Returns:
            ClaimVerification with verification scores
        """
        ...


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class DecisionNode:
    """
    A node in the decision graph.

    Implements: Spec SPEC-04.01-03
    Implements: Spec SPEC-16.11 (claim decision type)
    """

    id: str
    decision_type: str  # goal, decision, option, action, outcome, observation, claim, verification
    content: str
    confidence: float = 0.5
    prompt: str | None = None
    files: list[str] = field(default_factory=list)
    branch: str | None = None
    commit_hash: str | None = None
    parent_id: str | None = None
    # Claim-specific fields (SPEC-16.11)
    claim_text: str | None = None  # The actual claim being verified
    evidence_ids: list[str] = field(default_factory=list)  # Evidence source IDs
    verification_status: str | None = None  # pending, verified, flagged, refuted
    # Verification-specific fields (SPEC-16.12)
    verified_claim_id: str | None = None  # ID of the claim being verified
    support_score: float | None = None  # How well evidence supports claim (0.0-1.0)
    dependence_score: float | None = None  # How much answer changed without evidence (0.0-1.0)
    consistency_score: float | None = None  # Semantic consistency across variations (0.0-1.0)
    is_flagged: bool = False  # Whether this verification requires attention
    flag_reason: str | None = None  # Reason for flagging


@dataclass
class DecisionTree:
    """
    A tree of decision nodes.

    Implements: Spec SPEC-04.16
    """

    root: DecisionNode
    children: list[DecisionTree] = field(default_factory=list)


@dataclass
class RejectedOption:
    """An option that was rejected with a reason."""

    id: str
    content: str
    reason: str


@dataclass
class EvidenceScore:
    """
    Score for an option based on supporting/contradicting evidence.

    Implements: Phase 3 Memory Integration
    """

    option_id: str
    score: float  # Positive means more support, negative means more contradiction
    support_score: float  # Sum of weighted supporting evidence
    contradiction_score: float  # Sum of weighted contradicting evidence
    supporting_facts: list[tuple[str, float, float]]  # (fact_id, weight, confidence)
    contradicting_facts: list[tuple[str, float, float]]  # (fact_id, weight, confidence)

    @property
    def net_evidence_count(self) -> int:
        """Number of supporting facts minus contradicting facts."""
        return len(self.supporting_facts) - len(self.contradicting_facts)

    @property
    def total_evidence_count(self) -> int:
        """Total number of evidence facts."""
        return len(self.supporting_facts) + len(self.contradicting_facts)


# =============================================================================
# Schema Extension
# =============================================================================

DECISIONS_SCHEMA_SQL = """
-- Decisions extension table
CREATE TABLE IF NOT EXISTS decisions (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    decision_type TEXT NOT NULL CHECK(decision_type IN
        ('goal', 'decision', 'option', 'action', 'outcome', 'observation', 'claim', 'verification')),
    confidence REAL DEFAULT 0.5,
    prompt TEXT,
    files JSON DEFAULT '[]',
    branch TEXT,
    commit_hash TEXT,
    parent_id TEXT REFERENCES decisions(node_id),
    -- Claim-specific fields (SPEC-16.11)
    claim_text TEXT,  -- The actual claim text being verified
    evidence_ids JSON DEFAULT '[]',  -- List of evidence source IDs
    verification_status TEXT CHECK(verification_status IS NULL OR verification_status IN
        ('pending', 'verified', 'flagged', 'refuted')),
    -- Verification-specific fields (SPEC-16.12)
    verified_claim_id TEXT REFERENCES decisions(node_id),  -- Links to claim being verified
    support_score REAL CHECK(support_score IS NULL OR (support_score >= 0.0 AND support_score <= 1.0)),
    dependence_score REAL CHECK(dependence_score IS NULL OR (dependence_score >= 0.0 AND dependence_score <= 1.0)),
    consistency_score REAL CHECK(consistency_score IS NULL OR (consistency_score >= 0.0 AND consistency_score <= 1.0)),
    is_flagged INTEGER DEFAULT 0,  -- Boolean: 0=false, 1=true
    flag_reason TEXT CHECK(flag_reason IS NULL OR flag_reason IN
        ('unsupported', 'phantom_citation', 'low_dependence', 'contradiction', 'over_extrapolation', 'confidence_mismatch'))
);

CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_decisions_parent ON decisions(parent_id);
CREATE INDEX IF NOT EXISTS idx_decisions_commit ON decisions(commit_hash);
CREATE INDEX IF NOT EXISTS idx_decisions_verification ON decisions(verification_status);
CREATE INDEX IF NOT EXISTS idx_decisions_verified_claim ON decisions(verified_claim_id);
"""


# =============================================================================
# Valid Types
# =============================================================================

VALID_DECISION_TYPES = frozenset(
    {
        "goal",
        "decision",
        "option",
        "action",
        "outcome",
        "observation",
        "claim",  # SPEC-16.11: Epistemic verification claim
        "verification",  # SPEC-16.12: Epistemic verification result
    }
)

# Epistemic edge labels (SPEC-16.13, 16.14)
EPISTEMIC_EDGE_LABELS = frozenset(
    {
        "cites",  # claim → evidence_source (claim cites evidence)
        "verifies",  # verification → claim (positive verification result)
        "refutes",  # verification → claim (negative verification result)
    }
)


# =============================================================================
# ReasoningTraces Class
# =============================================================================


class ReasoningTraces:
    """
    Deciduous-style reasoning traces with git integration.

    Implements: Spec SPEC-04

    Features:
    - Decision node types: goal, decision, option, action, outcome, observation
    - Hyperedge relationships: spawns, considers, chooses, rejects, implements, produces, informs
    - Git integration: branch and commit linking
    - Trajectory mapping: auto-create decisions from trajectory events
    """

    def __init__(self, store: MemoryStore):
        """
        Initialize reasoning traces.

        Args:
            store: MemoryStore instance for persistence
        """
        self.store = store
        self._trajectory_mapping_enabled = True
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize the decisions table schema.

        The decisions table is created by MemoryStore._init_database()
        before rlm_core opens. This method handles migration of old
        databases that lack epistemic columns.
        """
        conn = self.store._get_connection()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'"
            )
            if cursor.fetchone() is not None:
                self._migrate_epistemic_schema(conn)
                # Create indexes that depend on migrated columns
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_decisions_verification ON decisions(verification_status)",
                    "CREATE INDEX IF NOT EXISTS idx_decisions_verified_claim ON decisions(verified_claim_id)",
                ]:
                    try:
                        conn.execute(idx_sql)
                    except Exception:
                        pass
                conn.commit()
        finally:
            conn.close()

    def _migrate_epistemic_schema(self, conn: Any) -> None:
        """
        Migrate decisions table for epistemic verification support.

        Implements: SPEC-16.19

        Adds claim and verification columns to existing databases
        that were created before SPEC-16 implementation. Ensures
        backward compatibility with older databases.
        """
        # Check current columns
        cursor = conn.execute("PRAGMA table_info(decisions)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Columns needed for claims (SPEC-16.11)
        claim_columns = [
            ("claim_text", "TEXT"),
            ("evidence_ids", "JSON DEFAULT '[]'"),
            ("verification_status", "TEXT"),
        ]

        # Columns needed for verifications (SPEC-16.12)
        verification_columns = [
            ("verified_claim_id", "TEXT"),
            ("support_score", "REAL"),
            ("dependence_score", "REAL"),
            ("consistency_score", "REAL"),
            ("is_flagged", "INTEGER DEFAULT 0"),
            ("flag_reason", "TEXT"),
        ]

        # Add missing columns (SQLite ALTER TABLE is limited, no CHECK constraints)
        migration_needed = False
        for col_name, col_def in claim_columns + verification_columns:
            if col_name not in existing_columns:
                migration_needed = True
                try:
                    conn.execute(f"ALTER TABLE decisions ADD COLUMN {col_name} {col_def}")
                except Exception:
                    # Column might already exist or other error - continue
                    pass

        # Create indexes if they don't exist
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decisions_verification "
            "ON decisions(verification_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decisions_verified_claim "
            "ON decisions(verified_claim_id)"
        )

        if migration_needed:
            conn.commit()

    # =========================================================================
    # Node Creation (SPEC-04.01-03)
    # =========================================================================

    def create_goal(
        self,
        content: str,
        prompt: str | None = None,
        files: list[str] | None = None,
        branch: str | None = None,
    ) -> str:
        """
        Create a goal node.

        Implements: Spec SPEC-04.01

        Args:
            content: Goal description
            prompt: Optional user prompt that triggered this goal
            files: Optional list of relevant files
            branch: Optional git branch

        Returns:
            Node ID
        """
        return self._create_decision_node(
            decision_type="goal",
            content=content,
            prompt=prompt,
            files=files or [],
            branch=branch,
            parent_id=None,
        )

    def create_decision(
        self,
        goal_id: str,
        content: str,
        prompt: str | None = None,
        parent_id: str | None = None,
    ) -> str:
        """
        Create a decision node under a goal or another decision.

        Implements: Spec SPEC-04.04

        Args:
            goal_id: Goal ID to associate with (for edge creation)
            content: Decision description
            prompt: Optional prompt that triggered this decision
            parent_id: Optional parent decision ID for sub-decisions.
                       If None, uses goal_id as parent (backward compatible).

        Returns:
            Node ID
        """
        # Use goal_id as parent if parent_id not specified (backward compatible)
        actual_parent_id = parent_id if parent_id is not None else goal_id

        decision_id = self._create_decision_node(
            decision_type="decision",
            content=content,
            prompt=prompt,
            parent_id=actual_parent_id,
        )

        # Create "spawns" edge from parent to decision
        self.store.create_edge(
            edge_type="relation",
            label="spawns",
            members=[
                {"node_id": actual_parent_id, "role": "subject", "position": 0},
                {"node_id": decision_id, "role": "object", "position": 1},
            ],
        )

        return decision_id

    def add_option(self, decision_id: str, content: str) -> str:
        """
        Add an option to a decision.

        Implements: Spec SPEC-04.05

        Args:
            decision_id: Parent decision ID
            content: Option description

        Returns:
            Node ID
        """
        option_id = self._create_decision_node(
            decision_type="option",
            content=content,
            parent_id=decision_id,
        )

        # Create "considers" edge from decision to option
        self.store.create_edge(
            edge_type="relation",
            label="considers",
            members=[
                {"node_id": decision_id, "role": "subject", "position": 0},
                {"node_id": option_id, "role": "object", "position": 1},
            ],
        )

        return option_id

    def choose_option(self, decision_id: str, option_id: str) -> None:
        """
        Mark an option as chosen.

        Implements: Spec SPEC-04.06

        Args:
            decision_id: Decision ID
            option_id: Chosen option ID
        """
        # Create "chooses" edge
        self.store.create_edge(
            edge_type="relation",
            label="chooses",
            members=[
                {"node_id": decision_id, "role": "subject", "position": 0},
                {"node_id": option_id, "role": "object", "position": 1},
            ],
        )

        # Update option confidence
        self.store.update_node(option_id, confidence=1.0)

    def reject_option(self, decision_id: str, option_id: str, reason: str) -> None:
        """
        Mark an option as rejected with reason.

        Implements: Spec SPEC-04.07

        Args:
            decision_id: Decision ID
            option_id: Rejected option ID
            reason: Reason for rejection
        """
        # Create "rejects" edge with reason as metadata
        self.store.create_edge(
            edge_type="relation",
            label="rejects",
            members=[
                {"node_id": decision_id, "role": "subject", "position": 0},
                {"node_id": option_id, "role": "object", "position": 1},
            ],
        )

        # Store reason in option metadata
        option_node = self.store.get_node(option_id)
        if option_node:
            metadata = option_node.metadata.copy()
            metadata["rejection_reason"] = reason
            self.store.update_node(option_id, metadata=metadata, confidence=0.0)

    def create_action(self, decision_id: str, content: str) -> str:
        """
        Create an action implementing a decision.

        Implements: Spec SPEC-04.08

        Args:
            decision_id: Parent decision ID
            content: Action description

        Returns:
            Node ID
        """
        action_id = self._create_decision_node(
            decision_type="action",
            content=content,
            parent_id=decision_id,
        )

        # Create "implements" edge
        self.store.create_edge(
            edge_type="relation",
            label="implements",
            members=[
                {"node_id": decision_id, "role": "subject", "position": 0},
                {"node_id": action_id, "role": "object", "position": 1},
            ],
        )

        return action_id

    def create_outcome(
        self,
        action_id: str,
        content: str,
        success: bool,
    ) -> str:
        """
        Create an outcome from an action.

        Implements: Spec SPEC-04.09

        Args:
            action_id: Parent action ID
            content: Outcome description
            success: Whether the outcome was successful

        Returns:
            Node ID
        """
        outcome_id = self._create_decision_node(
            decision_type="outcome",
            content=content,
            parent_id=action_id,
            confidence=1.0 if success else 0.3,
        )

        # Store success in metadata
        self._update_decision_metadata(outcome_id, {"success": success})

        # Create "produces" edge
        self.store.create_edge(
            edge_type="relation",
            label="produces",
            members=[
                {"node_id": action_id, "role": "subject", "position": 0},
                {"node_id": outcome_id, "role": "object", "position": 1},
            ],
        )

        return outcome_id

    def create_observation(self, content: str) -> str:
        """
        Create an observation node.

        Implements: Spec SPEC-04.10

        Args:
            content: Observation content

        Returns:
            Node ID
        """
        return self._create_decision_node(
            decision_type="observation",
            content=content,
        )

    def create_claim(
        self,
        claim_text: str,
        evidence_ids: list[str] | None = None,
        confidence: float = 0.5,
        parent_id: str | None = None,
        verification_status: str = "pending",
    ) -> str:
        """
        Create a claim node for epistemic verification.

        Implements: Spec SPEC-16.11

        Args:
            claim_text: The claim being verified
            evidence_ids: List of evidence source IDs supporting/relating to claim
            confidence: Initial confidence score (0.0-1.0)
            parent_id: Optional parent decision ID
            verification_status: Status of verification ('pending', 'verified', 'flagged', 'refuted')

        Returns:
            Node ID
        """
        # Validate verification_status
        valid_statuses = {"pending", "verified", "flagged", "refuted"}
        if verification_status not in valid_statuses:
            raise ValueError(
                f"Invalid verification_status: {verification_status}. "
                f"Must be one of: {valid_statuses}"
            )

        return self._create_decision_node(
            decision_type="claim",
            content=claim_text,  # Use claim_text as content for searchability
            claim_text=claim_text,
            evidence_ids=evidence_ids or [],
            confidence=confidence,
            parent_id=parent_id,
            verification_status=verification_status,
        )

    def update_claim_status(
        self,
        claim_id: str,
        verification_status: str,
        confidence: float | None = None,
    ) -> bool:
        """
        Update a claim's verification status.

        Implements: Spec SPEC-16.11

        Args:
            claim_id: The claim node ID
            verification_status: New status ('pending', 'verified', 'flagged', 'refuted')
            confidence: Optional new confidence score

        Returns:
            True if updated successfully
        """
        valid_statuses = {"pending", "verified", "flagged", "refuted"}
        if verification_status not in valid_statuses:
            raise ValueError(
                f"Invalid verification_status: {verification_status}. "
                f"Must be one of: {valid_statuses}"
            )

        updates: dict[str, Any] = {"verification_status": verification_status}
        if confidence is not None:
            updates["confidence"] = confidence
            # Also update the base node confidence
            self.store.update_node(claim_id, confidence=confidence)

        return self._update_decision(claim_id, **updates)

    def link_claim_to_evidence(
        self,
        claim_id: str,
        evidence_id: str,
    ) -> None:
        """
        Link a claim to an evidence source (cites relationship).

        Implements: Spec SPEC-16.13

        Creates a "cites" edge from the claim to the evidence source,
        indicating that the claim references or is based on this evidence.

        Args:
            claim_id: The claim node ID
            evidence_id: The evidence source node ID (fact, observation, etc.)
        """
        self.store.create_edge(
            edge_type="relation",
            label="cites",
            members=[
                {"node_id": claim_id, "role": "subject", "position": 0},
                {"node_id": evidence_id, "role": "object", "position": 1},
            ],
        )

    def add_claim(
        self,
        decision_id: str,
        claim: str,
        evidence_ids: list[str] | None = None,
        confidence: float = 0.5,
    ) -> str:
        """
        Add a claim to a decision with evidence links.

        Implements: Spec SPEC-16.15

        This is a convenience method that:
        1. Creates a claim node with the decision as parent
        2. Links the claim to all provided evidence sources via "cites" edges

        Args:
            decision_id: The parent decision node ID
            claim: The claim text being made
            evidence_ids: List of evidence source node IDs (facts, observations, etc.)
            confidence: Initial confidence score for the claim (0.0-1.0)

        Returns:
            The claim node ID
        """
        # Create the claim with decision as parent
        claim_id = self.create_claim(
            claim_text=claim,
            evidence_ids=evidence_ids,
            confidence=confidence,
            parent_id=decision_id,
        )

        # Create "cites" edges to each evidence source
        if evidence_ids:
            for evidence_id in evidence_ids:
                self.link_claim_to_evidence(claim_id, evidence_id)

        return claim_id

    def verify_claim(
        self,
        claim_id: str,
        verification_model: VerificationModel,
    ) -> str:
        """
        Verify a claim using an epistemic verification model.

        Implements: Spec SPEC-16.16

        This method:
        1. Retrieves the claim and its cited evidence
        2. Calls the verification model with claim text and evidence
        3. Creates a verification node with the result
        4. Links via 'verifies' or 'refutes' edge (handled by create_verification)

        Args:
            claim_id: The ID of the claim to verify
            verification_model: A callable that verifies claims against evidence.
                Must conform to the VerificationModel protocol.

        Returns:
            The verification node ID

        Raises:
            ValueError: If claim_id doesn't exist or isn't a claim node
        """
        # 1. Retrieve the claim node
        claim_node = self.get_decision_node(claim_id)
        if claim_node is None:
            raise ValueError(f"Claim not found: {claim_id}")
        if claim_node.decision_type != "claim":
            raise ValueError(f"Node {claim_id} is not a claim (type: {claim_node.decision_type})")

        # 2. Get evidence from the claim's evidence_ids and/or "cites" edges
        evidence_ids = set(claim_node.evidence_ids or [])

        # Also check for "cites" edges to find additional evidence
        edges = self.store.query_edges(label="cites")
        for edge in edges:
            members = self.store.get_edge_members(edge.id)
            # Check if this edge originates from our claim
            subject = next((m for m in members if m["role"] == "subject"), None)
            obj = next((m for m in members if m["role"] == "object"), None)
            if subject and subject["node_id"] == claim_id and obj:
                evidence_ids.add(obj["node_id"])

        # 3. Retrieve evidence content
        evidence: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            evidence_node = self.store.get_node(evidence_id)
            if evidence_node:
                evidence.append(
                    {
                        "id": evidence_id,
                        "content": evidence_node.content,
                        "confidence": evidence_node.confidence,
                    }
                )

        # 4. Call the verification model
        claim_text = claim_node.claim_text or claim_node.content
        verification_result = verification_model(claim_text, evidence)

        # 5. Create verification node with result
        verification_id = self.create_verification(
            claim_id=claim_id,
            support_score=verification_result.evidence_support,
            dependence_score=verification_result.evidence_dependence,
            consistency_score=verification_result.consistency_score,
            is_flagged=verification_result.is_flagged,
            flag_reason=verification_result.flag_reason,
            confidence=verification_result.combined_score,
            parent_id=claim_node.parent_id,
        )

        # 6. Update claim verification status based on result
        if verification_result.is_flagged:
            new_status = verification_result.flag_reason or "flagged"
            if new_status in ("unsupported", "phantom_citation", "contradiction"):
                self.update_claim_status(claim_id, "refuted")
            else:
                self.update_claim_status(claim_id, "flagged")
        else:
            self.update_claim_status(claim_id, "verified")

        return verification_id

    def create_verification(
        self,
        claim_id: str,
        support_score: float,
        dependence_score: float,
        consistency_score: float = 1.0,
        is_flagged: bool = False,
        flag_reason: str | None = None,
        confidence: float = 0.5,
        parent_id: str | None = None,
    ) -> str:
        """
        Create a verification node for an epistemic claim.

        Implements: Spec SPEC-16.12

        Args:
            claim_id: ID of the claim being verified
            support_score: How well evidence supports claim (0.0-1.0)
            dependence_score: How much answer changed without evidence (0.0-1.0)
            consistency_score: Semantic consistency across variations (0.0-1.0)
            is_flagged: Whether this verification flags a potential issue
            flag_reason: Reason for flagging (if flagged)
            confidence: Confidence in the verification result (0.0-1.0)
            parent_id: Optional parent decision ID

        Returns:
            Node ID of the created verification

        Raises:
            ValueError: If scores are out of range or flag_reason is invalid
        """
        # Validate score ranges
        for name, value in [
            ("support_score", support_score),
            ("dependence_score", dependence_score),
            ("consistency_score", consistency_score),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")

        # Validate flag_reason if provided
        valid_flag_reasons = {
            "unsupported",
            "phantom_citation",
            "low_dependence",
            "contradiction",
            "over_extrapolation",
            "confidence_mismatch",
        }
        if flag_reason is not None and flag_reason not in valid_flag_reasons:
            raise ValueError(
                f"Invalid flag_reason: {flag_reason}. Must be one of: {valid_flag_reasons}"
            )

        # Generate verification content/summary
        status = "flagged" if is_flagged else "verified"
        content = (
            f"Verification ({status}): support={support_score:.2f}, "
            f"dependence={dependence_score:.2f}, consistency={consistency_score:.2f}"
        )
        if flag_reason:
            content += f" [reason: {flag_reason}]"

        verification_id = self._create_decision_node(
            decision_type="verification",
            content=content,
            verified_claim_id=claim_id,
            support_score=support_score,
            dependence_score=dependence_score,
            consistency_score=consistency_score,
            is_flagged=is_flagged,
            flag_reason=flag_reason,
            confidence=confidence,
            parent_id=parent_id,
        )

        # Create edge linking verification to claim
        # Use "refutes" for flagged verifications, "verifies" for positive ones (SPEC-16.14)
        edge_label = "refutes" if is_flagged else "verifies"
        self.store.create_edge(
            edge_type="relation",
            label=edge_label,
            members=[
                {"node_id": verification_id, "role": "subject", "position": 0},
                {"node_id": claim_id, "role": "object", "position": 1},
            ],
        )

        return verification_id

    def get_epistemic_gaps(
        self,
        goal_id: str,
        support_threshold: float = 0.7,
    ) -> list:
        """
        Find all epistemic gaps in a decision tree.

        Implements: Spec SPEC-16.17

        Traverses the decision tree rooted at goal_id, finds all claims
        with insufficient evidence support, and returns EpistemicGap objects.

        A claim has an epistemic gap if:
        - It has a verification that flagged it
        - Its verification support_score is below the threshold
        - It has no verification but is marked with low confidence

        Args:
            goal_id: Root goal ID to traverse
            support_threshold: Minimum support score to not be flagged (default 0.7)

        Returns:
            List of EpistemicGap objects for claims with insufficient evidence

        Raises:
            ValueError: If goal_id doesn't exist
        """
        from .epistemic import EpistemicGap

        # Verify goal exists
        goal = self.get_decision_node(goal_id)
        if goal is None:
            raise ValueError(f"Goal not found: {goal_id}")

        gaps: list[EpistemicGap] = []

        # Collect all claims that are descendants of this goal
        claims = self._collect_claims_under_goal(goal_id)

        # For each claim, check for verification issues
        for claim in claims:
            verification = self._get_verification_for_claim(claim.id)

            if verification is not None:
                # Claim has been verified
                if verification.is_flagged:
                    # Verification flagged an issue
                    gap_type = self._flag_reason_to_gap_type(verification.flag_reason)
                    suggested_action = self._get_suggested_action(gap_type)

                    # Calculate gap_bits from support score
                    # Higher gap_bits means lower support
                    support = verification.support_score or 0.5
                    gap_bits = max(0.0, -2.0 * (support - 1.0))  # 0 support = 2 bits gap

                    gaps.append(
                        EpistemicGap(
                            claim_id=claim.id,
                            claim_text=claim.claim_text or claim.content,
                            gap_type=gap_type,
                            gap_bits=gap_bits,
                            suggested_action=suggested_action,
                        )
                    )
                elif (verification.support_score or 1.0) < support_threshold:
                    # Support score is below threshold but not flagged
                    gaps.append(
                        EpistemicGap(
                            claim_id=claim.id,
                            claim_text=claim.claim_text or claim.content,
                            gap_type="partial_support",
                            gap_bits=max(
                                0.0, support_threshold - (verification.support_score or 0.0)
                            ),
                            suggested_action="Provide additional evidence to strengthen claim",
                        )
                    )
            else:
                # Claim has no verification - check if it has low confidence
                if claim.confidence is not None and claim.confidence < 0.5:
                    gaps.append(
                        EpistemicGap(
                            claim_id=claim.id,
                            claim_text=claim.claim_text or claim.content,
                            gap_type="unsupported",
                            gap_bits=1.0,  # Default gap for unverified low-confidence claims
                            suggested_action="Verify claim against available evidence",
                        )
                    )

        return gaps

    def get_verification_report(self, goal_id: str) -> HallucinationReport:
        """
        Generate a full verification report for a goal tree.

        Implements: Spec SPEC-16.18

        Aggregates all claims under the goal, their verification status,
        and epistemic gaps into a comprehensive HallucinationReport.

        Args:
            goal_id: Root goal ID to generate report for

        Returns:
            HallucinationReport with all claims, gaps, and metrics

        Raises:
            ValueError: If goal_id doesn't exist
        """
        from .epistemic import ClaimVerification, HallucinationReport

        # Verify goal exists
        goal = self.get_decision_node(goal_id)
        if goal is None:
            raise ValueError(f"Goal not found: {goal_id}")

        # Create report with goal_id as response_id
        report = HallucinationReport(response_id=goal_id)

        # Collect all claims
        claims = self._collect_claims_under_goal(goal_id)

        # Build ClaimVerification for each claim
        for claim in claims:
            verification = self._get_verification_for_claim(claim.id)

            # Build ClaimVerification from claim and verification data
            claim_verification = ClaimVerification(
                claim_id=claim.id,
                claim_text=claim.claim_text or claim.content,
                evidence_ids=claim.evidence_ids or [],
                evidence_support=verification.support_score if verification else 0.5,
                evidence_dependence=verification.dependence_score if verification else 0.5,
                consistency_score=verification.consistency_score if verification else 1.0,
                is_flagged=verification.is_flagged if verification else False,
                flag_reason=verification.flag_reason if verification else None,
            )

            # Calculate evidence gap bits
            if verification and verification.support_score is not None:
                # Higher gap bits for lower support
                claim_verification.evidence_gap_bits = max(
                    0.0, 2.0 * (1.0 - verification.support_score)
                )

            report.add_claim(claim_verification)

        # Get and add epistemic gaps
        gaps = self.get_epistemic_gaps(goal_id)
        for gap in gaps:
            report.add_gap(gap)

        # Determine if retry should be recommended
        if report.has_critical_gaps or report.verification_rate < 0.5:
            report.should_retry = True
            if report.has_critical_gaps:
                critical_types = [g.gap_type for g in gaps if g.severity == "critical"]
                report.retry_guidance = (
                    f"Critical issues found: {', '.join(set(critical_types))}. "
                    "Please verify claims against actual evidence."
                )
            else:
                report.retry_guidance = (
                    f"Low verification rate ({report.verification_rate:.0%}). "
                    "Consider providing more evidence or qualifying claims."
                )

        return report

    def _collect_claims_under_goal(self, goal_id: str) -> list[DecisionNode]:
        """
        Collect all claim nodes that are descendants of a goal.

        Uses recursive CTE to find all descendants in the parent hierarchy,
        then filters for claim nodes.

        Args:
            goal_id: The root goal ID

        Returns:
            List of DecisionNode objects with decision_type='claim'
        """
        import sqlite3

        conn = self.store._get_connection()
        conn.row_factory = sqlite3.Row
        try:
            # Use recursive CTE to find all descendants
            cursor = conn.execute(
                """
                WITH RECURSIVE descendants(node_id) AS (
                    -- Base case: direct children of goal
                    SELECT node_id FROM decisions WHERE parent_id = ?
                    UNION ALL
                    -- Recursive case: children of descendants
                    SELECT d.node_id FROM decisions d
                    JOIN descendants desc ON d.parent_id = desc.node_id
                )
                SELECT d.*, n.content
                FROM decisions d
                JOIN nodes n ON d.node_id = n.id
                WHERE d.node_id IN (SELECT node_id FROM descendants)
                AND d.decision_type = 'claim'
                """,
                (goal_id,),
            )

            claims = []
            for row in cursor.fetchall():
                claims.append(self._row_to_decision_node(row))
            return claims
        finally:
            conn.close()

    def _get_verification_for_claim(self, claim_id: str) -> DecisionNode | None:
        """
        Find the most recent verification for a claim.

        Args:
            claim_id: The claim to find verification for

        Returns:
            DecisionNode with the verification, or None if not verified
        """
        import sqlite3

        # Query for verification nodes that reference this claim
        conn = self.store._get_connection()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT node_id FROM decisions
                WHERE decision_type = 'verification'
                AND verified_claim_id = ?
                ORDER BY node_id DESC
                LIMIT 1
                """,
                (claim_id,),
            )
            row = cursor.fetchone()
            if row:
                return self.get_decision_node(row["node_id"])
            return None
        finally:
            conn.close()

    def _flag_reason_to_gap_type(self, flag_reason: str | None) -> str:
        """
        Map flag_reason to GapType.

        Args:
            flag_reason: The flag reason from verification

        Returns:
            Corresponding GapType string
        """
        mapping = {
            "unsupported": "unsupported",
            "phantom_citation": "phantom_citation",
            "low_dependence": "evidence_independent",
            "contradiction": "contradicted",
            "over_extrapolation": "over_extrapolation",
            "confidence_mismatch": "partial_support",
        }
        return mapping.get(flag_reason or "", "unsupported")

    def _get_suggested_action(self, gap_type: str) -> str:
        """
        Get suggested remediation action for a gap type.

        Args:
            gap_type: The type of epistemic gap

        Returns:
            Suggested action string
        """
        actions = {
            "unsupported": "Find evidence that supports this claim or remove it",
            "phantom_citation": "Correct the citation to reference actual evidence",
            "evidence_independent": "Verify that evidence actually informs the claim",
            "contradicted": "Resolve contradiction between claim and evidence",
            "over_extrapolation": "Qualify claim or find additional supporting evidence",
            "partial_support": "Provide additional evidence to strengthen claim",
        }
        return actions.get(gap_type, "Review claim against available evidence")

    def link_observation(self, observation_id: str, decision_id: str) -> None:
        """
        Link an observation to a decision (informs).

        Implements: Spec SPEC-04.10

        Args:
            observation_id: Observation node ID
            decision_id: Decision node ID
        """
        self.store.create_edge(
            edge_type="relation",
            label="informs",
            members=[
                {"node_id": observation_id, "role": "subject", "position": 0},
                {"node_id": decision_id, "role": "object", "position": 1},
            ],
        )

    # =========================================================================
    # Evidence Linking (Memory + Reasoning Integration)
    # =========================================================================

    def add_supporting_fact(
        self,
        fact_id: str,
        option_id: str,
        weight: float = 1.0,
    ) -> str:
        """
        Link a fact as supporting evidence for an option.

        Args:
            fact_id: The fact node ID providing evidence
            option_id: The option node ID being supported
            weight: Evidence strength (0.0-1.0)

        Returns:
            Edge ID
        """
        return self.store.create_evidence_edge(
            label="supports",
            source_id=fact_id,
            target_id=option_id,
            weight=weight,
        )

    def add_contradicting_fact(
        self,
        fact_id: str,
        option_id: str,
        weight: float = 1.0,
    ) -> str:
        """
        Link a fact as contradicting evidence against an option.

        Args:
            fact_id: The fact node ID providing evidence
            option_id: The option node ID being contradicted
            weight: Evidence strength (0.0-1.0)

        Returns:
            Edge ID
        """
        return self.store.create_evidence_edge(
            label="contradicts",
            source_id=fact_id,
            target_id=option_id,
            weight=weight,
        )

    def record_outcome_validates_fact(
        self,
        outcome_id: str,
        fact_id: str,
        weight: float = 1.0,
    ) -> str:
        """
        Record that an outcome validates (confirms) a fact.

        This is used to update fact confidence based on decision outcomes.
        When an outcome confirms what a fact predicted, the fact's
        reliability increases.

        Args:
            outcome_id: The outcome node ID
            fact_id: The fact node ID being validated
            weight: Validation strength (0.0-1.0)

        Returns:
            Edge ID
        """
        return self.store.create_evidence_edge(
            label="validates",
            source_id=outcome_id,
            target_id=fact_id,
            weight=weight,
        )

    def record_outcome_invalidates_fact(
        self,
        outcome_id: str,
        fact_id: str,
        weight: float = 1.0,
    ) -> str:
        """
        Record that an outcome invalidates (refutes) a fact.

        This is used to update fact confidence based on decision outcomes.
        When an outcome contradicts what a fact predicted, the fact's
        reliability decreases.

        Args:
            outcome_id: The outcome node ID
            fact_id: The fact node ID being invalidated
            weight: Invalidation strength (0.0-1.0)

        Returns:
            Edge ID
        """
        return self.store.create_evidence_edge(
            label="invalidates",
            source_id=outcome_id,
            target_id=fact_id,
            weight=weight,
        )

    def get_evidence_for_option(self, option_id: str) -> dict[str, list[tuple[str, float]]]:
        """
        Get all evidence (supporting and contradicting facts) for an option.

        Args:
            option_id: The option node ID

        Returns:
            Dict with 'supporting' and 'contradicting' lists of (fact_id, weight)
        """
        return {
            "supporting": self.store.get_supporting_facts(option_id),
            "contradicting": self.store.get_contradicting_facts(option_id),
        }

    def get_outcome_evidence_for_fact(self, fact_id: str) -> dict[str, list[tuple[str, float]]]:
        """
        Get all outcome evidence (validating and invalidating) for a fact.

        Args:
            fact_id: The fact node ID

        Returns:
            Dict with 'validating' and 'invalidating' lists of (outcome_id, weight)
        """
        return {
            "validating": self.store.get_validating_outcomes(fact_id),
            "invalidating": self.store.get_invalidating_outcomes(fact_id),
        }

    def _compute_recency_weight(self, timestamp_ms: int, half_life_days: float = 7.0) -> float:
        """
        Compute recency weight using exponential decay.

        More recent facts have higher weight. Weight decays by half
        every `half_life_days` days.

        Args:
            timestamp_ms: Timestamp in milliseconds
            half_life_days: Half-life for decay in days

        Returns:
            Weight between 0.0 and 1.0
        """
        import math
        import time

        now_ms = int(time.time() * 1000)
        age_ms = now_ms - timestamp_ms
        age_days = age_ms / (1000 * 60 * 60 * 24)

        # Exponential decay: weight = 0.5^(age/half_life)
        return math.pow(0.5, age_days / half_life_days)

    def score_option_with_evidence(
        self,
        option_id: str,
        recency_half_life_days: float = 7.0,
    ) -> EvidenceScore:
        """
        Score an option based on supporting and contradicting evidence.

        Computes a weighted score where each fact's contribution is:
        edge_weight * fact_confidence * recency_weight

        Args:
            option_id: The option node ID to score
            recency_half_life_days: Half-life for recency decay (default 7 days)

        Returns:
            EvidenceScore with detailed breakdown
        """
        # Get evidence links
        supporting = self.store.get_supporting_facts(option_id)
        contradicting = self.store.get_contradicting_facts(option_id)

        # Calculate weighted scores
        support_score = 0.0
        supporting_details: list[tuple[str, float, float]] = []

        for fact_id, edge_weight in supporting:
            node = self.store.get_node(fact_id)
            if node:
                recency = self._compute_recency_weight(node.created_at, recency_half_life_days)
                weighted = edge_weight * node.confidence * recency
                support_score += weighted
                supporting_details.append((fact_id, edge_weight, node.confidence))

        contradiction_score = 0.0
        contradicting_details: list[tuple[str, float, float]] = []

        for fact_id, edge_weight in contradicting:
            node = self.store.get_node(fact_id)
            if node:
                recency = self._compute_recency_weight(node.created_at, recency_half_life_days)
                weighted = edge_weight * node.confidence * recency
                contradiction_score += weighted
                contradicting_details.append((fact_id, edge_weight, node.confidence))

        return EvidenceScore(
            option_id=option_id,
            score=support_score - contradiction_score,
            support_score=support_score,
            contradiction_score=contradiction_score,
            supporting_facts=supporting_details,
            contradicting_facts=contradicting_details,
        )

    def get_evidence_chain_for_option(self, option_id: str) -> list[dict[str, Any]]:
        """
        Get a detailed evidence chain for decision trace output.

        Args:
            option_id: The option node ID

        Returns:
            List of evidence details with fact content, confidence, and direction
        """
        evidence_chain = []

        # Get supporting facts
        for fact_id, weight in self.store.get_supporting_facts(option_id):
            node = self.store.get_node(fact_id)
            if node:
                evidence_chain.append(
                    {
                        "fact_id": fact_id,
                        "content": node.content,
                        "confidence": node.confidence,
                        "weight": weight,
                        "direction": "supports",
                        "timestamp": node.created_at,
                    }
                )

        # Get contradicting facts
        for fact_id, weight in self.store.get_contradicting_facts(option_id):
            node = self.store.get_node(fact_id)
            if node:
                evidence_chain.append(
                    {
                        "fact_id": fact_id,
                        "content": node.content,
                        "confidence": node.confidence,
                        "weight": weight,
                        "direction": "contradicts",
                        "timestamp": node.created_at,
                    }
                )

        # Sort by timestamp (most recent first)
        evidence_chain.sort(key=lambda x: x["timestamp"], reverse=True)
        return evidence_chain

    # =========================================================================
    # Bayesian Confidence Propagation (Phase 3: Memory Integration)
    # =========================================================================

    def update_fact_confidence_from_outcome(
        self,
        fact_id: str,
        outcome_success: bool,
        damping: float = 0.1,
        min_confidence: float = 0.1,
        max_confidence: float = 0.95,
        trigger_id: str | None = None,
    ) -> float:
        """
        Update a fact's confidence based on decision outcome using Bayesian update.

        Uses damping to prevent oscillation - confidence changes by at most
        `damping` fraction per update.

        Args:
            fact_id: The fact node ID to update
            outcome_success: Whether the outcome was successful
            damping: Maximum adjustment per update (default 0.1 = 10%)
            min_confidence: Minimum allowed confidence (default 0.1)
            max_confidence: Maximum allowed confidence (default 0.95)
            trigger_id: Optional ID of the triggering outcome (for audit trail)

        Returns:
            New confidence value after update
        """
        # Get current fact
        node = self.store.get_node(fact_id)
        if not node:
            raise ValueError(f"Fact not found: {fact_id}")

        current_confidence = node.confidence

        # Get all outcomes linked to this fact (validating and invalidating)
        all_outcomes = self._get_all_outcomes_for_fact(fact_id)

        if not all_outcomes:
            # No historical outcomes, use simple damped update
            target = 1.0 if outcome_success else 0.0
            new_confidence = current_confidence * (1 - damping) + target * damping
        else:
            # Calculate recency-weighted success rate
            weighted_success = 0.0
            total_weight = 0.0

            for outcome_id, is_validating, outcome_timestamp in all_outcomes:
                recency = self._compute_recency_weight(outcome_timestamp)
                # Validating outcomes count as success, invalidating as failure
                success_value = 1.0 if is_validating else 0.0
                weighted_success += success_value * recency
                total_weight += recency

            # Include the new outcome
            new_recency = 1.0  # Current outcome has full recency weight
            new_success = 1.0 if outcome_success else 0.0
            weighted_success += new_success * new_recency
            total_weight += new_recency

            observed_rate = weighted_success / total_weight if total_weight > 0 else 0.5

            # Bayesian update with damping
            new_confidence = current_confidence * (1 - damping) + observed_rate * damping

        # Clamp to allowed range
        new_confidence = max(min_confidence, min(max_confidence, new_confidence))

        # Update the node
        self.store.update_node(fact_id, confidence=new_confidence)

        # Log the confidence update for audit trail
        self.store.log_confidence_update(
            node_id=fact_id,
            old_confidence=current_confidence,
            new_confidence=new_confidence,
            trigger_type="outcome",
            trigger_id=trigger_id,
        )

        return new_confidence

    def _get_all_outcomes_for_fact(self, fact_id: str) -> list[tuple[str, bool, int]]:
        """
        Get all outcomes linked to a fact with their validation status.

        Args:
            fact_id: The fact node ID

        Returns:
            List of (outcome_id, is_validating, timestamp) tuples
        """
        outcomes = []

        # Get validating outcomes
        for outcome_id, weight in self.store.get_validating_outcomes(fact_id):
            node = self.store.get_node(outcome_id)
            if node:
                outcomes.append((outcome_id, True, node.created_at))

        # Get invalidating outcomes
        for outcome_id, weight in self.store.get_invalidating_outcomes(fact_id):
            node = self.store.get_node(outcome_id)
            if node:
                outcomes.append((outcome_id, False, node.created_at))

        return outcomes

    def propagate_outcome_to_facts(
        self,
        outcome_id: str,
        success: bool,
        damping: float = 0.1,
    ) -> dict[str, float]:
        """
        Propagate an outcome's success/failure to all linked facts.

        Updates confidence of all facts that this outcome validates or invalidates.

        Args:
            outcome_id: The outcome node ID
            success: Whether the outcome was successful
            damping: Maximum adjustment per update (default 0.1)

        Returns:
            Dict mapping fact_id to new confidence value
        """
        updated_facts = {}

        # Get facts this outcome validates (outcome validates fact means success confirms it)
        for fact_id, weight in self.store.get_evidence_targets(outcome_id, "validates"):
            # Successful outcome + validates = increase confidence
            # Failed outcome + validates = no clear signal (skip or slight decrease)
            if success:
                new_conf = self.update_fact_confidence_from_outcome(
                    fact_id, outcome_success=True, damping=damping
                )
                updated_facts[fact_id] = new_conf

        # Get facts this outcome invalidates (outcome invalidates fact means success refutes it)
        for fact_id, weight in self.store.get_evidence_targets(outcome_id, "invalidates"):
            # Successful outcome + invalidates = decrease confidence
            # Failed outcome + invalidates = no clear signal (skip or slight increase)
            if success:
                new_conf = self.update_fact_confidence_from_outcome(
                    fact_id, outcome_success=False, damping=damping
                )
                updated_facts[fact_id] = new_conf

        return updated_facts

    # =========================================================================
    # Cross-Reference API (Phase 3: Memory Integration)
    # =========================================================================

    def get_decisions_for_fact(self, fact_id: str) -> list[DecisionNode]:
        """
        Get all decisions that used this fact as evidence.

        Finds decisions whose options were supported or contradicted by this fact.

        Args:
            fact_id: The fact node ID

        Returns:
            List of DecisionNode objects that used this fact
        """
        decisions: dict[str, DecisionNode] = {}

        # Get all options this fact supports or contradicts
        supported_options = self.store.get_evidence_targets(fact_id, "supports")
        contradicted_options = self.store.get_evidence_targets(fact_id, "contradicts")

        all_options = [(oid, w) for oid, w in supported_options]
        all_options.extend([(oid, w) for oid, w in contradicted_options])

        # For each option, find the decision that considers it
        for option_id, _ in all_options:
            # Query edges where this option is considered by a decision
            edges = self.store.query_edges(label="considers")
            for edge in edges:
                members = self.store.get_edge_members(edge.id)
                for member in members:
                    if member["node_id"] == option_id and member["role"] == "object":
                        # Find the subject (the decision)
                        for m in members:
                            if m["role"] == "subject":
                                decision_id = m["node_id"]
                                if decision_id not in decisions:
                                    node = self.get_decision_node(decision_id)
                                    if node:
                                        decisions[decision_id] = node
                                break

        return list(decisions.values())

    def get_supporting_facts_for_decision(self, decision_id: str) -> list[tuple[str, str, float]]:
        """
        Get all facts supporting options in this decision.

        Args:
            decision_id: The decision node ID

        Returns:
            List of (fact_id, option_id, weight) tuples for supporting facts
        """
        supporting_facts = []

        # Get all options considered by this decision
        edges = self.store.query_edges(label="considers")
        option_ids = []

        for edge in edges:
            members = self.store.get_edge_members(edge.id)
            for member in members:
                if member["role"] == "subject":
                    if member["node_id"] == decision_id:
                        # Found an edge from this decision
                        for m in members:
                            if m["role"] == "object":
                                option_ids.append(m["node_id"])
                        break

        # For each option, get supporting facts
        for option_id in option_ids:
            for fact_id, weight in self.store.get_supporting_facts(option_id):
                supporting_facts.append((fact_id, option_id, weight))

        return supporting_facts

    def get_fact_success_rate(self, fact_id: str) -> dict[str, Any]:
        """
        Calculate how often this fact led to successful outcomes.

        Args:
            fact_id: The fact node ID

        Returns:
            Dict with success rate statistics:
            - total_outcomes: Number of linked outcomes
            - validating_outcomes: Number of validating outcomes
            - invalidating_outcomes: Number of invalidating outcomes
            - success_rate: Ratio of validating to total (0.0-1.0)
            - confidence_trend: Whether fact confidence is increasing/decreasing
        """
        validating = self.store.get_validating_outcomes(fact_id)
        invalidating = self.store.get_invalidating_outcomes(fact_id)

        total = len(validating) + len(invalidating)
        success_rate = len(validating) / total if total > 0 else 0.5

        # Get confidence drift from history
        drift_report = self.store.get_confidence_drift_report(fact_id)
        total_drift = drift_report.get("total_drift", 0.0)

        if total_drift > 0.05:
            trend = "increasing"
        elif total_drift < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "fact_id": fact_id,
            "total_outcomes": total,
            "validating_outcomes": len(validating),
            "invalidating_outcomes": len(invalidating),
            "success_rate": success_rate,
            "confidence_trend": trend,
            "total_confidence_drift": total_drift,
        }

    def get_fact_usage_summary(self, fact_id: str) -> dict[str, Any]:
        """
        Get a comprehensive summary of how a fact has been used.

        Args:
            fact_id: The fact node ID

        Returns:
            Dict with usage statistics
        """
        node = self.store.get_node(fact_id)
        if not node:
            return {"error": f"Fact not found: {fact_id}"}

        decisions = self.get_decisions_for_fact(fact_id)
        success_stats = self.get_fact_success_rate(fact_id)

        # Get supporting/contradicting counts
        supported_options = self.store.get_evidence_targets(fact_id, "supports")
        contradicted_options = self.store.get_evidence_targets(fact_id, "contradicts")

        return {
            "fact_id": fact_id,
            "content": node.content,
            "current_confidence": node.confidence,
            "decisions_using_fact": len(decisions),
            "options_supported": len(supported_options),
            "options_contradicted": len(contradicted_options),
            **success_stats,
        }

    # =========================================================================
    # Git Integration (SPEC-04.11-15)
    # =========================================================================

    def link_commit(self, decision_id: str, commit_hash: str) -> None:
        """
        Link a decision to a git commit.

        Implements: Spec SPEC-04.11

        Args:
            decision_id: Decision node ID
            commit_hash: Git commit hash
        """
        self._update_decision(decision_id, commit_hash=commit_hash)

    def get_decisions_for_commit(self, commit_hash: str) -> list[DecisionNode]:
        """
        Get all decisions linked to a commit.

        Implements: Spec SPEC-04.14

        Args:
            commit_hash: Git commit hash

        Returns:
            List of DecisionNode objects
        """
        import sqlite3

        conn = self.store._get_connection()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT d.*, n.content
                FROM decisions d
                JOIN nodes n ON d.node_id = n.id
                WHERE d.commit_hash = ?
                """,
                (commit_hash,),
            )

            results = []
            for row in cursor.fetchall():
                results.append(self._row_to_decision_node(row))

            return results
        finally:
            conn.close()

    # =========================================================================
    # Query Interface (SPEC-04.16-19)
    # =========================================================================

    def get_decision_node(self, node_id: str) -> DecisionNode | None:
        """
        Get a decision node by ID.

        Args:
            node_id: Node ID

        Returns:
            DecisionNode or None if not found
        """
        import sqlite3

        conn = self.store._get_connection()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT d.*, n.content
                FROM decisions d
                JOIN nodes n ON d.node_id = n.id
                WHERE d.node_id = ?
                """,
                (node_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_decision_node(row)
        finally:
            conn.close()

    def get_decision_tree(self, goal_id: str) -> DecisionTree | None:
        """
        Get the decision tree rooted at a goal.

        Implements: Spec SPEC-04.16

        Args:
            goal_id: Root goal ID

        Returns:
            DecisionTree or None if goal not found
        """
        root = self.get_decision_node(goal_id)
        if root is None:
            return None

        return self._build_tree(root)

    def _build_tree(self, node: DecisionNode) -> DecisionTree:
        """Recursively build a decision tree."""
        # Get children via hyperedges
        related = self.store.get_related_nodes(node.id)
        children = []

        for rel in related:
            child_node = self.get_decision_node(rel.id)
            if child_node and child_node.parent_id == node.id:
                children.append(self._build_tree(child_node))

        return DecisionTree(root=node, children=children)

    def get_rejected_options(self, decision_id: str) -> list[RejectedOption]:
        """
        Get rejected options for a decision.

        Implements: Spec SPEC-04.17

        Args:
            decision_id: Decision ID

        Returns:
            List of RejectedOption objects
        """
        # Find "rejects" edges from this decision
        edges = self.store.query_edges(label="rejects")

        rejected = []
        for edge in edges:
            members = self.store.get_edge_members(edge.id)
            # Check if this edge is from our decision
            subject = next((m for m in members if m["role"] == "subject"), None)
            obj = next((m for m in members if m["role"] == "object"), None)

            if subject and subject["node_id"] == decision_id and obj:
                option_node = self.store.get_node(obj["node_id"])
                if option_node:
                    reason = option_node.metadata.get("rejection_reason", "")
                    rejected.append(
                        RejectedOption(
                            id=option_node.id,
                            content=option_node.content,
                            reason=reason,
                        )
                    )

        return rejected

    def get_outcome(self, goal_id: str) -> DecisionNode | None:
        """
        Get the outcome for a goal (if any).

        Implements: Spec SPEC-04.18

        Args:
            goal_id: Goal ID

        Returns:
            Outcome DecisionNode or None
        """
        # Traverse the tree to find outcome nodes
        tree = self.get_decision_tree(goal_id)
        if tree is None:
            return None

        return self._find_outcome_in_tree(tree)

    def _find_outcome_in_tree(self, tree: DecisionTree) -> DecisionNode | None:
        """Recursively search for an outcome node."""
        if tree.root.decision_type == "outcome":
            return tree.root

        for child in tree.children:
            result = self._find_outcome_in_tree(child)
            if result:
                return result

        return None

    def get_informing_observations(self, decision_id: str) -> list[DecisionNode]:
        """
        Get observations that inform a decision.

        Implements: Spec SPEC-04.19

        Args:
            decision_id: Decision ID

        Returns:
            List of observation DecisionNodes
        """
        # Find "informs" edges to this decision
        edges = self.store.query_edges(label="informs")

        observations = []
        for edge in edges:
            members = self.store.get_edge_members(edge.id)
            # Check if this edge points to our decision
            obj = next((m for m in members if m["role"] == "object"), None)
            subject = next((m for m in members if m["role"] == "subject"), None)

            if obj and obj["node_id"] == decision_id and subject:
                obs_node = self.get_decision_node(subject["node_id"])
                if obs_node and obs_node.decision_type == "observation":
                    observations.append(obs_node)

        return observations

    # =========================================================================
    # Trajectory Integration (SPEC-04.20-24)
    # =========================================================================

    def set_trajectory_mapping(self, enabled: bool) -> None:
        """
        Enable or disable trajectory-to-decision mapping.

        Implements: Spec SPEC-04.24

        Args:
            enabled: Whether to enable mapping
        """
        self._trajectory_mapping_enabled = enabled

    def from_trajectory_event(self, event: TrajectoryEvent) -> str | None:
        """
        Create a decision node from a trajectory event.

        Implements: Spec SPEC-04.20-23

        Args:
            event: TrajectoryEvent to convert

        Returns:
            Node ID or None if mapping is disabled
        """
        if not self._trajectory_mapping_enabled:
            return None

        from .trajectory import TrajectoryEventType

        # Map event type to decision type
        type_mapping = {
            TrajectoryEventType.RLM_START: "goal",
            TrajectoryEventType.RECURSE_START: "goal",
            TrajectoryEventType.ANALYZE: "decision",
            TrajectoryEventType.REASON: "decision",
            TrajectoryEventType.REPL_EXEC: "action",
            TrajectoryEventType.TOOL_USE: "action",
            TrajectoryEventType.FINAL: "outcome",
            TrajectoryEventType.RECURSE_END: "outcome",
            TrajectoryEventType.REPL_RESULT: "observation",
            TrajectoryEventType.ERROR: "observation",
        }

        decision_type = type_mapping.get(event.type, "observation")

        # Get parent from metadata
        parent_id = None
        if hasattr(event, "metadata") and event.metadata:
            parent_id = event.metadata.get("parent_id")

        # Create the node
        return self._create_decision_node(
            decision_type=decision_type,
            content=event.content,
            parent_id=parent_id,
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _create_decision_node(
        self,
        decision_type: str,
        content: str,
        prompt: str | None = None,
        files: list[str] | None = None,
        branch: str | None = None,
        commit_hash: str | None = None,
        parent_id: str | None = None,
        confidence: float = 0.5,
        # Claim-specific fields (SPEC-16.11)
        claim_text: str | None = None,
        evidence_ids: list[str] | None = None,
        verification_status: str | None = None,
        # Verification-specific fields (SPEC-16.12)
        verified_claim_id: str | None = None,
        support_score: float | None = None,
        dependence_score: float | None = None,
        consistency_score: float | None = None,
        is_flagged: bool = False,
        flag_reason: str | None = None,
    ) -> str:
        """Create a decision node in both nodes and decisions tables."""

        # Create the base node
        metadata: dict[str, Any] = {
            "decision_type": decision_type,
            "prompt": prompt,
            "files": files or [],
        }
        # Add claim-specific metadata if this is a claim
        if decision_type == "claim":
            metadata["claim_text"] = claim_text
            metadata["evidence_ids"] = evidence_ids or []
            metadata["verification_status"] = verification_status
        # Add verification-specific metadata if this is a verification
        elif decision_type == "verification":
            metadata["verified_claim_id"] = verified_claim_id
            metadata["support_score"] = support_score
            metadata["dependence_score"] = dependence_score
            metadata["consistency_score"] = consistency_score
            metadata["is_flagged"] = is_flagged
            metadata["flag_reason"] = flag_reason

        node_id = self.store.create_node(
            node_type="decision",
            content=content,
            subtype=decision_type,
            confidence=confidence,
            metadata=metadata,
        )

        # Create the decision extension record
        conn = self.store._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO decisions (
                    node_id, decision_type, confidence, prompt, files,
                    branch, commit_hash, parent_id,
                    claim_text, evidence_ids, verification_status,
                    verified_claim_id, support_score, dependence_score,
                    consistency_score, is_flagged, flag_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    decision_type,
                    confidence,
                    prompt,
                    json.dumps(files or []),
                    branch,
                    commit_hash,
                    parent_id,
                    claim_text,
                    json.dumps(evidence_ids or []),
                    verification_status,
                    verified_claim_id,
                    support_score,
                    dependence_score,
                    consistency_score,
                    1 if is_flagged else 0,
                    flag_reason,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return node_id

    def _update_decision(self, node_id: str, **kwargs: Any) -> bool:
        """Update decision-specific fields."""

        if not kwargs:
            return False

        conn = self.store._get_connection()
        try:
            updates = []
            values = []
            for key, value in kwargs.items():
                if key == "files":
                    updates.append(f"{key} = ?")
                    values.append(json.dumps(value))
                else:
                    updates.append(f"{key} = ?")
                    values.append(value)

            values.append(node_id)

            query = f"UPDATE decisions SET {', '.join(updates)} WHERE node_id = ?"
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def _update_decision_metadata(self, node_id: str, extra: dict[str, Any]) -> None:
        """Update decision metadata in the nodes table."""
        node = self.store.get_node(node_id)
        if node:
            metadata = node.metadata.copy()
            metadata.update(extra)
            self.store.update_node(node_id, metadata=metadata)

    def _row_to_decision_node(self, row) -> DecisionNode:
        """Convert a database row to a DecisionNode."""
        files = row["files"]
        if isinstance(files, str):
            files = json.loads(files)

        # Parse evidence_ids (SPEC-16.11)
        evidence_ids = row["evidence_ids"]
        if isinstance(evidence_ids, str):
            evidence_ids = json.loads(evidence_ids)

        return DecisionNode(
            id=row["node_id"],
            decision_type=row["decision_type"],
            content=row["content"],
            confidence=row["confidence"] or 0.5,
            prompt=row["prompt"],
            files=files or [],
            branch=row["branch"],
            commit_hash=row["commit_hash"],
            parent_id=row["parent_id"],
            # Claim-specific fields (SPEC-16.11)
            claim_text=row["claim_text"],
            evidence_ids=evidence_ids or [],
            verification_status=row["verification_status"],
            # Verification-specific fields (SPEC-16.12)
            verified_claim_id=row["verified_claim_id"],
            support_score=row["support_score"],
            dependence_score=row["dependence_score"],
            consistency_score=row["consistency_score"],
            is_flagged=bool(row["is_flagged"]),
            flag_reason=row["flag_reason"],
        )


__all__ = [
    "ReasoningTraces",
    "DecisionNode",
    "DecisionTree",
    "RejectedOption",
    "EvidenceScore",
    "VALID_DECISION_TYPES",
    "EPISTEMIC_EDGE_LABELS",
    "VerificationModel",
]
