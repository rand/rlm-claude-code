"""
Persistent hypergraph memory with SQLite storage.

Implements: Spec SPEC-02 Memory Foundation

Uses rlm_core.MemoryStore as the primary storage backend.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

import rlm_core

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Node:
    """
    A node in the hypergraph memory.

    Implements: Spec SPEC-02.05-10
    """

    id: str
    type: str  # entity, fact, experience, decision, snippet
    content: str
    tier: str = "task"  # task, session, longterm, archive
    confidence: float = 0.5
    subtype: str | None = None
    embedding: bytes | None = None
    provenance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0
    last_accessed: int = 0
    access_count: int = 0


@dataclass
class Hyperedge:
    """
    A hyperedge connecting multiple nodes.

    Implements: Spec SPEC-02.11-13
    """

    id: str
    type: str  # relation, composition, causation, context
    label: str | None = None
    weight: float = 1.0


@dataclass
class EvolutionLogEntry:
    """Entry in the evolution log for tier transitions."""

    id: int
    timestamp: int
    operation: str
    node_ids: list[str]
    from_tier: str | None
    to_tier: str | None
    reasoning: str | None


@dataclass
class ConfidenceUpdate:
    """
    Entry in the confidence updates audit log.

    Implements: Phase 3 Memory Integration
    """

    id: int
    node_id: str
    old_confidence: float
    new_confidence: float
    trigger_type: str  # outcome, decay, consolidation, manual
    trigger_id: str | None
    timestamp: int


# Valid trigger types for confidence updates
VALID_CONFIDENCE_TRIGGERS = frozenset({"outcome", "decay", "consolidation", "manual"})


@dataclass
class SearchResult:
    """
    Result from FTS5 full-text search.

    Implements: Phase 4 Massive Context (SPEC-01.03)
    """

    node_id: str
    content: str
    node_type: str
    bm25_score: float
    snippet: str | None = None  # Highlighted snippet if available


# =============================================================================
# Schema Definition
# =============================================================================

# =============================================================================
# MemoryStore Class
# =============================================================================


class _NoCloseConnection:
    """Wrapper around sqlite3.Connection that makes close() a no-op."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def close(self) -> None:
        pass  # Don't actually close the persistent connection

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class MemoryStore:
    """
    Persistent hypergraph memory store using SQLite.

    Implements: Spec SPEC-02.01-26

    Features:
    - SQLite with WAL mode for concurrent access
    - Node CRUD operations with tier system
    - Hyperedge support for many-to-many relationships
    - Evolution logging for tier transitions

    Delegates core node operations to rlm_core.MemoryStore while maintaining
    API compatibility for auxiliary features (hyperedges, evolution logging).
    """

    # Valid node types (SPEC-02.05)
    VALID_NODE_TYPES = frozenset({"entity", "fact", "experience", "decision", "snippet"})

    # Valid tiers (SPEC-02.17)
    VALID_TIERS = frozenset({"task", "session", "longterm", "archive"})

    # Valid edge types (SPEC-02.11)
    VALID_EDGE_TYPES = frozenset({"relation", "composition", "causation", "context"})

    # Valid membership roles (SPEC-02.16)
    VALID_ROLES = frozenset({"subject", "object", "context", "participant", "cause", "effect"})

    # Valid edge labels for reasoning/memory integration
    # Decision flow labels (SPEC-04)
    DECISION_LABELS = frozenset(
        {
            "spawns",  # goal → decision
            "considers",  # decision → option
            "chooses",  # decision → option (selected)
            "rejects",  # decision → option (rejected)
            "implements",  # option → action
            "produces",  # action → outcome
            "informs",  # observation → any
        }
    )

    # Evidence linking labels (for bidirectional fact ↔ decision linking)
    EVIDENCE_LABELS = frozenset(
        {
            "supports",  # fact → option (evidence for a choice)
            "contradicts",  # fact → option (evidence against a choice)
            "validates",  # outcome → fact (outcome confirms fact)
            "invalidates",  # outcome → fact (outcome refutes fact)
        }
    )

    # All valid edge labels
    VALID_EDGE_LABELS = DECISION_LABELS | EVIDENCE_LABELS

    def __init__(self, db_path: str | None = None):
        """
        Initialize memory store.

        Args:
            db_path: Path to SQLite database. If None, uses default or env var.

        Implements: Spec SPEC-02.01-04
        """
        if db_path is None:
            db_path = os.environ.get("RLM_MEMORY_DB") or self._get_default_db_path()

        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._persistent_conn: sqlite3.Connection | None = None

        # Python SQLite for auxiliary tables (hyperedges, confidence_updates)
        # Must be created BEFORE rlm_core.open() — opening multiple Python
        # connections after rlm_core corrupts its schema cache.
        if not self._is_memory:
            self._ensure_directory()
        self._init_database()

        # rlm_core is the primary backend for node CRUD
        # Opens after Python auxiliary tables so schema changes don't conflict.
        if self._is_memory:
            self._core_store = rlm_core.MemoryStore.in_memory()
        else:
            self._core_store = rlm_core.MemoryStore.open(db_path)

    @property
    def uses_rlm_core(self) -> bool:
        """Return True if using rlm_core backend."""
        return True

    def _get_default_db_path(self) -> str:
        """
        Get default database path.

        Implements: Spec SPEC-02.02
        """
        return str(Path.home() / ".claude" / "rlm-memory.db")

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self) -> None:
        """
        Initialize auxiliary tables (hyperedges, evolution_log, confidence_updates).

        Node CRUD is handled by rlm_core.MemoryStore.
        """
        if self._is_memory:
            conn = sqlite3.connect(":memory:")
        else:
            conn = sqlite3.connect(self.db_path)
        # Keep a persistent connection for all stores to avoid opening
        # multiple connections that corrupt rlm_core's WAL state.
        self._persistent_conn = conn
        conn.execute("PRAGMA journal_mode=WAL")
        # FK enforcement disabled: rlm_core manages nodes via its own Rust connection,
        # so membership.node_id FK against nodes(id) can't be validated here.
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.executescript("""
            -- Hyperedges table
            CREATE TABLE IF NOT EXISTS hyperedges (
                id TEXT PRIMARY KEY,
                edge_type TEXT NOT NULL CHECK(edge_type IN ('relation', 'composition', 'causation', 'context')),
                label TEXT,
                weight REAL DEFAULT 1.0 CHECK(weight >= 0.0)
            );

            -- Membership table (many-to-many)
            CREATE TABLE IF NOT EXISTS membership (
                hyperedge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
                node_id TEXT NOT NULL,
                role TEXT NOT NULL,
                position INTEGER DEFAULT 0,
                PRIMARY KEY (hyperedge_id, node_id, role)
            );

            -- evolution_log is created by rlm_core (node_id, operation, from_tier, to_tier, reason, created_at)

            -- Confidence updates audit table
            CREATE TABLE IF NOT EXISTS confidence_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                old_confidence REAL NOT NULL,
                new_confidence REAL NOT NULL,
                trigger_type TEXT NOT NULL CHECK(trigger_type IN ('outcome', 'decay', 'consolidation', 'manual')),
                trigger_id TEXT,
                timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000)
            );

            -- Decisions extension table (used by ReasoningTraces)
            -- Created here so all schema init happens before rlm_core.open()
            CREATE TABLE IF NOT EXISTS decisions (
                node_id TEXT PRIMARY KEY,
                decision_type TEXT NOT NULL CHECK(decision_type IN
                    ('goal', 'decision', 'option', 'action', 'outcome', 'observation', 'claim', 'verification')),
                confidence REAL DEFAULT 0.5,
                prompt TEXT,
                files JSON DEFAULT '[]',
                branch TEXT,
                commit_hash TEXT,
                parent_id TEXT,
                claim_text TEXT,
                evidence_ids JSON DEFAULT '[]',
                verification_status TEXT CHECK(verification_status IS NULL OR verification_status IN
                    ('pending', 'verified', 'flagged', 'refuted')),
                verified_claim_id TEXT,
                support_score REAL CHECK(support_score IS NULL OR (support_score >= 0.0 AND support_score <= 1.0)),
                dependence_score REAL CHECK(dependence_score IS NULL OR (dependence_score >= 0.0 AND dependence_score <= 1.0)),
                consistency_score REAL CHECK(consistency_score IS NULL OR (consistency_score >= 0.0 AND consistency_score <= 1.0)),
                is_flagged INTEGER DEFAULT 0,
                flag_reason TEXT CHECK(flag_reason IS NULL OR flag_reason IN
                    ('unsupported', 'phantom_citation', 'low_dependence', 'contradiction', 'over_extrapolation', 'confidence_mismatch'))
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_membership_node ON membership(node_id);
            CREATE INDEX IF NOT EXISTS idx_membership_edge ON membership(hyperedge_id);
            CREATE INDEX IF NOT EXISTS idx_confidence_updates_node ON confidence_updates(node_id);
            CREATE INDEX IF NOT EXISTS idx_confidence_updates_trigger ON confidence_updates(trigger_type);
            CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type);
            CREATE INDEX IF NOT EXISTS idx_decisions_parent ON decisions(parent_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_commit ON decisions(commit_hash);
        """)
        conn.commit()
        # These indexes depend on columns that may not exist in old DBs
        # (added by migration). Create them best-effort.
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_decisions_verification ON decisions(verification_status)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_verified_claim ON decisions(verified_claim_id)",
        ]:
            try:
                conn.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        conn.commit()

    def _init_auxiliary_tables(self) -> None:
        """Initialize Python-specific tables when rlm_core manages core node/edge schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript("""
                -- Evolution log
                CREATE TABLE IF NOT EXISTS evolution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000),
                    operation TEXT NOT NULL,
                    node_ids JSON NOT NULL,
                    from_tier TEXT,
                    to_tier TEXT,
                    reasoning TEXT
                );

                -- Confidence updates audit
                CREATE TABLE IF NOT EXISTS confidence_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    old_confidence REAL NOT NULL,
                    new_confidence REAL NOT NULL,
                    trigger_type TEXT NOT NULL CHECK(trigger_type IN ('outcome', 'decay', 'consolidation', 'manual')),
                    trigger_id TEXT,
                    timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000)
                );
                CREATE INDEX IF NOT EXISTS idx_confidence_updates_node ON confidence_updates(node_id);
                CREATE INDEX IF NOT EXISTS idx_confidence_updates_trigger ON confidence_updates(trigger_type);

                -- Hyperedges (rlm_core uses different edge types, keep Python edges separate)
                CREATE TABLE IF NOT EXISTS hyperedges (
                    id TEXT PRIMARY KEY,
                    edge_type TEXT NOT NULL CHECK(edge_type IN ('relation', 'composition', 'causation', 'context')),
                    label TEXT,
                    weight REAL DEFAULT 1.0 CHECK(weight >= 0.0)
                );
                CREATE TABLE IF NOT EXISTS membership (
                    hyperedge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
                    node_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    position INTEGER DEFAULT 0,
                    PRIMARY KEY (hyperedge_id, node_id, role)
                );
                CREATE INDEX IF NOT EXISTS idx_membership_node ON membership(node_id);
                CREATE INDEX IF NOT EXISTS idx_membership_edge ON membership(hyperedge_id);
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings.

        Returns a non-closeable wrapper around a persistent connection
        to avoid opening multiple connections that corrupt rlm_core's WAL state.
        """
        if self._persistent_conn is not None:
            self._persistent_conn.row_factory = sqlite3.Row
            return _NoCloseConnection(self._persistent_conn)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # rlm_core Delegation Helpers
    # =========================================================================

    def _map_node_type_to_core(self, node_type: str) -> Any:
        """Map Python node type string to rlm_core.NodeType enum."""
        mapping = {
            "entity": rlm_core.NodeType.Entity,
            "fact": rlm_core.NodeType.Fact,
            "experience": rlm_core.NodeType.Experience,
            "decision": rlm_core.NodeType.Decision,
            "snippet": rlm_core.NodeType.Snippet,
        }
        return mapping.get(node_type, rlm_core.NodeType.Fact)

    def _map_tier_to_core(self, tier: str) -> Any:
        """Map Python tier string to rlm_core.Tier enum."""
        mapping = {
            "task": rlm_core.Tier.Task,
            "session": rlm_core.Tier.Session,
            "longterm": rlm_core.Tier.LongTerm,
            "archive": rlm_core.Tier.Archive,
        }
        return mapping.get(tier, rlm_core.Tier.Task)

    def _map_core_node_type_to_str(self, node_type: Any) -> str:
        """Map rlm_core.NodeType enum to Python string."""
        name = str(node_type).split(".")[-1].lower()
        return name if name in self.VALID_NODE_TYPES else "fact"

    def _map_core_tier_to_str(self, tier: Any) -> str:
        """Map rlm_core.Tier enum to Python string."""
        name = str(tier).split(".")[-1].lower()
        return name if name in self.VALID_TIERS else "task"

    @staticmethod
    def _rfc3339_to_epoch_ms(rfc3339_str: str) -> int:
        """Convert RFC3339 timestamp string to epoch milliseconds."""
        from datetime import datetime

        try:
            # Handle various RFC3339 formats
            dt = datetime.fromisoformat(rfc3339_str.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, AttributeError):
            return 0

    def _core_node_to_python(self, node: Any) -> Node:
        """Convert rlm_core.Node to Python Node dataclass."""
        meta = node.metadata or {}
        # Retrieve original provenance string from metadata if stored there
        provenance = meta.pop("_provenance", None) or node.provenance_source
        return Node(
            id=node.id,
            type=self._map_core_node_type_to_str(node.node_type),
            content=node.content,
            tier=self._map_core_tier_to_str(node.tier),
            confidence=node.confidence if node.confidence is not None else 0.5,
            subtype=node.subtype,
            embedding=None,  # Rust stores Vec<f32>, Python expects bytes
            provenance=provenance,
            metadata=meta,
            created_at=self._rfc3339_to_epoch_ms(node.created_at),
            updated_at=self._rfc3339_to_epoch_ms(node.updated_at),
            last_accessed=self._rfc3339_to_epoch_ms(node.last_accessed),
            access_count=node.access_count,
        )

    def _create_node_core(
        self,
        node_type: str,
        content: str,
        tier: str,
        confidence: float = 0.5,
        subtype: str | None = None,
        metadata: dict[str, Any] | None = None,
        provenance: str | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Create a node using rlm_core backend."""

        kwargs: dict[str, Any] = {}
        if subtype is not None:
            kwargs["subtype"] = subtype
        kwargs["confidence"] = confidence
        if metadata:
            kwargs["metadata"] = metadata
        if provenance:
            # Store original provenance string in metadata for round-trip fidelity
            if "metadata" not in kwargs or kwargs["metadata"] is None:
                kwargs["metadata"] = {}
            kwargs["metadata"]["_provenance"] = provenance
            # Also set rlm_core provenance for its internal use
            kwargs["provenance_source"] = "inference"
            kwargs["provenance_ref"] = provenance
        if embedding:
            kwargs["embedding"] = embedding

        node = rlm_core.Node(
            node_type=self._map_node_type_to_core(node_type),
            content=content,
            tier=self._map_tier_to_core(tier),
            **kwargs,
        )
        return self._core_store.add_node(node)

    def _get_node_core(self, node_id: str) -> Node | None:
        """Get a node using rlm_core backend."""
        try:
            result = self._core_store.get_node(node_id)
        except (ValueError, RuntimeError):
            return None
        if result is None:
            return None
        return self._core_node_to_python(result)

    def _delete_node_core(self, node_id: str) -> bool:
        """Delete a node using rlm_core backend."""
        try:
            self._core_store.delete_node(node_id)
            return True
        except Exception:
            return False

    def _query_nodes_core(
        self,
        node_type: str | None = None,
        tier: str | None = None,
        limit: int = 100,
    ) -> list[Node]:
        """Query nodes using rlm_core backend."""

        if node_type is not None:
            results = self._core_store.query_by_type(self._map_node_type_to_core(node_type), limit)
        elif tier is not None:
            results = self._core_store.query_by_tier(self._map_tier_to_core(tier), limit)
        else:
            # No specific filter — query all types
            results = []
            for nt in self.VALID_NODE_TYPES:
                results.extend(
                    self._core_store.query_by_type(self._map_node_type_to_core(nt), limit)
                )

        nodes = [self._core_node_to_python(n) for n in results]

        # Apply tier filter if both node_type and tier are specified
        # (rlm_core query_by_type doesn't filter by tier)
        if node_type is not None and tier is not None:
            nodes = [n for n in nodes if n.tier == tier]

        return nodes

    def _search_core(self, query: str, limit: int) -> list[SearchResult]:
        """Search using rlm_core backend."""

        results = self._core_store.search_content(query, limit)
        return [
            SearchResult(
                node_id=node.id,
                content=node.content,
                node_type=self._map_core_node_type_to_str(node.node_type),
                bm25_score=node.confidence if node.confidence is not None else 0.5,
                snippet=node.content[:100] if len(node.content) > 100 else node.content,
            )
            for node in results
        ]

    # =========================================================================
    # Node CRUD Operations (SPEC-02.20-24)
    # =========================================================================

    def create_node(
        self,
        node_type: str | None = None,
        content: str | None = None,
        tier: str = "task",
        confidence: float = 0.5,
        subtype: str | None = None,
        embedding: bytes | None = None,
        provenance: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        type: str | None = None,
    ) -> str:
        """
        Create a new node in the memory store.

        Implements: Spec SPEC-02.20

        Args:
            node_type: Type of node (alias: type). Valid types:
                - "entity": Named entities (people, places, concepts)
                - "fact": Verified facts about the codebase
                - "experience": Past interactions and their outcomes
                - "decision": Architectural/design decisions
                - "snippet": Code snippets
            content: Node content (required)
            tier: Memory tier (task, session, longterm, archive)
            confidence: Confidence score [0.0, 1.0]
            subtype: Optional subtype for categorization
            embedding: Optional embedding blob for vector search
            provenance: Optional source/origin string
            metadata: Optional metadata dict
            type: Alias for node_type

        Returns:
            Node ID (UUID string)

        Raises:
            ValueError: If node_type is invalid or content is missing

        Example:
            >>> store = MemoryStore(":memory:")
            >>> node_id = store.create_node("fact", "Auth uses JWT tokens")
            >>> node_id = store.create_node(type="experience", content="Refactored auth module")
        """
        # Handle type alias
        if type is not None:
            node_type = type

        # Validate required parameters
        if node_type is None:
            raise ValueError("node_type (or type=) is required")
        if content is None:
            raise ValueError("content is required")
        # Validate node type (SPEC-02.05)
        if node_type not in self.VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node type: {node_type}. Must be one of: {self.VALID_NODE_TYPES}"
            )

        # Validate tier (SPEC-02.17)
        if tier not in self.VALID_TIERS:
            raise ValueError(f"Invalid tier: {tier}. Must be one of: {self.VALID_TIERS}")

        # Validate confidence (SPEC-02.10)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {confidence}")

        # Convert embedding bytes to float list if needed
        embedding_list = None
        if embedding is not None:
            import struct

            embedding_list = list(struct.unpack(f"{len(embedding) // 4}f", embedding))

        return self._create_node_core(
            node_type=node_type,
            content=content,
            tier=tier,
            confidence=confidence,
            subtype=subtype,
            metadata=metadata,
            provenance=provenance,
            embedding=embedding_list,
        )

    def get_node(self, node_id: str, include_archived: bool = False) -> Node | None:
        """
        Get a node by ID.

        Implements: Spec SPEC-02.21

        Args:
            node_id: Node ID to retrieve
            include_archived: If True, include archived nodes

        Returns:
            Node object or None if not found
        """
        node = self._get_node_core(node_id)
        if node is None:
            return None
        if not include_archived and node.tier == "archive":
            return None
        # Track access (best-effort, don't fail reads)
        try:
            self._core_store.update_fields(
                node_id,
                access_count=node.access_count + 1,
            )
            node.access_count += 1
        except Exception:
            pass
        return node

    def update_node(self, node_id: str, **kwargs: Any) -> bool:
        """
        Update a node.

        Implements: Spec SPEC-02.22

        Args:
            node_id: Node ID to update
            **kwargs: Fields to update (content, confidence, tier, etc.)

        Returns:
            True if updated, False if node not found
        """
        if not kwargs:
            return False

        # Validate confidence if provided
        if "confidence" in kwargs:
            confidence = kwargs["confidence"]
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {confidence}")

        # Validate tier if provided
        if "tier" in kwargs:
            tier = kwargs["tier"]
            if tier not in self.VALID_TIERS:
                raise ValueError(f"Invalid tier: {tier}")

        # Get current node from rlm_core
        current = self._core_store.get_node(node_id)
        if current is None:
            return False

        old_tier = self._map_core_tier_to_str(current.tier)

        # Use update_fields to avoid Node immutability issues
        update_kwargs: dict[str, Any] = {}
        if "content" in kwargs:
            update_kwargs["content"] = kwargs["content"]
        if "confidence" in kwargs:
            update_kwargs["confidence"] = kwargs["confidence"]
        if "tier" in kwargs:
            update_kwargs["tier"] = self._map_tier_to_core(kwargs["tier"])
        if "subtype" in kwargs:
            update_kwargs["subtype"] = kwargs["subtype"]
        if "metadata" in kwargs:
            update_kwargs["metadata"] = kwargs["metadata"]

        self._core_store.update_fields(node_id, **update_kwargs)

        # Log tier transition if tier changed (SPEC-02.19)
        if "tier" in kwargs and kwargs["tier"] != old_tier:
            conn = self._get_connection()
            try:
                self._log_evolution(
                    conn,
                    operation="tier_transition",
                    node_ids=[node_id],
                    from_tier=old_tier,
                    to_tier=kwargs["tier"],
                )
                conn.commit()
            finally:
                conn.close()

        return True

    def delete_node(self, node_id: str) -> bool:
        """
        Soft delete a node (move to archive tier).

        Implements: Spec SPEC-02.23

        Args:
            node_id: Node ID to delete

        Returns:
            True if deleted, False if node not found
        """
        return self.update_node(node_id, tier="archive")

    def hard_delete_node(self, node_id: str) -> bool:
        """
        Permanently delete a node.

        Args:
            node_id: Node ID to delete

        Returns:
            True if deleted, False if node not found
        """
        result = self._delete_node_core(node_id)
        if result:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM membership WHERE node_id = ?", (node_id,))
                conn.commit()
            finally:
                conn.close()
        return result

    def query_nodes(
        self,
        node_type: str | None = None,
        tier: str | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
        include_archived: bool = False,
        session_id: str | None = None,
    ) -> list[Node]:
        """
        Query nodes with filters.

        Implements: Spec SPEC-02.24

        Args:
            node_type: Filter by node type
            tier: Filter by tier
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results (default 100 to prevent context flooding)
            include_archived: Include archived nodes
            session_id: Filter by session for task/session tiers (ignored for longterm/archive)

        Returns:
            List of matching nodes
        """
        effective_limit = limit if limit is not None else 100
        nodes = self._query_nodes_core(
            node_type=node_type,
            tier=tier,
            limit=effective_limit,
        )

        # Apply filters that rlm_core doesn't support natively
        if not include_archived:
            nodes = [n for n in nodes if n.tier != "archive"]
        if min_confidence is not None:
            nodes = [n for n in nodes if n.confidence >= min_confidence]
        if tier in ("task", "session") and session_id is not None:
            nodes = [n for n in nodes if n.metadata.get("session_id") == session_id]

        # Sort by confidence DESC
        nodes.sort(key=lambda n: n.confidence, reverse=True)
        return nodes[:effective_limit]

    # =========================================================================
    # Hyperedge Operations (SPEC-02.25-26)
    # =========================================================================

    def create_edge(
        self,
        edge_type: str,
        label: str | None,
        members: list[dict[str, Any]],
        weight: float = 1.0,
    ) -> str:
        """
        Create a hyperedge connecting multiple nodes.

        Implements: Spec SPEC-02.25

        This is the low-level API for creating hyperedges (edges that can connect
        more than two nodes). For simple two-node relationships, use the `link()`
        convenience method instead.

        Args:
            edge_type: Type of edge. Valid types:
                - "relation": General relationship between nodes
                - "composition": Part-whole relationships
                - "causation": Cause-effect relationships
                - "context": Contextual associations
            label: Semantic label for the edge (e.g., "supports", "implements")
            members: List of member dicts, each with:
                - node_id (str): The node ID
                - role (str): Role in the relationship (subject, object, context, etc.)
                - position (int, optional): Order in the edge
            weight: Edge weight >= 0.0 (default 1.0)

        Returns:
            Edge ID (UUID string)

        Example:
            >>> # For simple two-node edges, prefer link():
            >>> store.link(fact_id, decision_id, "supports")
            >>>
            >>> # For multi-node hyperedges, use create_edge():
            >>> store.create_edge(
            ...     edge_type="causation",
            ...     label="caused_by",
            ...     members=[
            ...         {"node_id": effect_id, "role": "effect"},
            ...         {"node_id": cause1_id, "role": "cause"},
            ...         {"node_id": cause2_id, "role": "cause"},
            ...     ],
            ... )
        """
        # Validate edge type
        if edge_type not in self.VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge type: {edge_type}. Must be one of: {self.VALID_EDGE_TYPES}"
            )

        # Validate weight
        if weight < 0.0:
            raise ValueError(f"Edge weight must be >= 0.0, got: {weight}")

        edge_id = str(uuid.uuid4())

        conn = self._get_connection()
        try:
            # Create edge
            conn.execute(
                "INSERT INTO hyperedges (id, edge_type, label, weight) VALUES (?, ?, ?, ?)",
                (edge_id, edge_type, label, weight),
            )

            # Create memberships
            for i, member in enumerate(members):
                node_id = member["node_id"]
                role = member["role"]
                position = member.get("position", i)

                conn.execute(
                    """
                    INSERT INTO membership (hyperedge_id, node_id, role, position)
                    VALUES (?, ?, ?, ?)
                    """,
                    (edge_id, node_id, role, position),
                )

            conn.commit()
        finally:
            conn.close()

        return edge_id

    def get_edge(self, edge_id: str) -> Hyperedge | None:
        """Get a hyperedge by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM hyperedges WHERE id = ?", (edge_id,))
            row = cursor.fetchone()
            if row is None:
                return None

            return Hyperedge(
                id=row["id"],
                type=row["edge_type"],
                label=row["label"],
                weight=row["weight"],
            )
        finally:
            conn.close()

    def get_edge_members(self, edge_id: str) -> list[dict[str, Any]]:
        """Get members of a hyperedge."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT node_id, role, position
                FROM membership
                WHERE hyperedge_id = ?
                ORDER BY position
                """,
                (edge_id,),
            )
            return [
                {"node_id": row["node_id"], "role": row["role"], "position": row["position"]}
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def update_edge(self, edge_id: str, **kwargs: Any) -> bool:
        """Update a hyperedge."""
        if not kwargs:
            return False

        if "weight" in kwargs and kwargs["weight"] < 0.0:
            raise ValueError("Edge weight must be >= 0.0")

        conn = self._get_connection()
        try:
            updates = []
            values = []
            for key, value in kwargs.items():
                updates.append(f"{key} = ?")
                values.append(value)

            values.append(edge_id)

            query = f"UPDATE hyperedges SET {', '.join(updates)} WHERE id = ?"
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def delete_edge(self, edge_id: str) -> bool:
        """Delete a hyperedge and its membership entries."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM membership WHERE hyperedge_id = ?", (edge_id,))
            cursor = conn.execute("DELETE FROM hyperedges WHERE id = ?", (edge_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # =========================================================================
    # Convenience Methods for Node Creation
    # =========================================================================

    def add_fact(
        self,
        content: str,
        confidence: float = 0.8,
        tier: str = "session",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a fact to memory (convenience method).

        Args:
            content: The factual statement
            confidence: Confidence score 0.0-1.0 (default 0.8)
            tier: Memory tier - "task", "session", or "long_term" (default "session")
            metadata: Optional additional metadata

        Returns:
            Node ID (UUID string)

        Example:
            >>> store = MemoryStore(":memory:")
            >>> fact_id = store.add_fact("Auth uses JWT tokens", confidence=0.9)
            >>> fact_id = store.add_fact("Redis cache TTL is 5 minutes")
        """
        return self.create_node(
            node_type="fact",
            content=content,
            tier=tier,
            confidence=confidence,
            metadata=metadata,
        )

    def add_experience(
        self,
        content: str,
        outcome: str,
        success: bool = True,
        confidence: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add an experience/learning to memory (convenience method).

        Args:
            content: Description of what was tried
            outcome: What happened as a result
            success: Whether the outcome was positive (default True)
            confidence: Confidence score 0.0-1.0 (default 0.7)
            metadata: Optional additional metadata

        Returns:
            Node ID (UUID string)

        Example:
            >>> store = MemoryStore(":memory:")
            >>> exp_id = store.add_experience(
            ...     "Tried map_reduce on large file",
            ...     outcome="Faster than sequential processing",
            ...     success=True,
            ... )
        """
        full_metadata = {"outcome": outcome, "success": success}
        if metadata:
            full_metadata.update(metadata)
        return self.create_node(
            node_type="experience",
            content=content,
            tier="session",
            confidence=confidence,
            metadata=full_metadata,
        )

    def add_entity(
        self,
        name: str,
        entity_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add an entity (person, file, concept) to memory (convenience method).

        Args:
            name: Entity name/identifier
            entity_type: Type like "file", "function", "person", "concept"
            metadata: Optional additional metadata

        Returns:
            Node ID (UUID string)

        Example:
            >>> store = MemoryStore(":memory:")
            >>> file_id = store.add_entity("auth.py", entity_type="file")
            >>> person_id = store.add_entity("Alice", entity_type="person")
        """
        full_metadata = {}
        if entity_type:
            full_metadata["entity_type"] = entity_type
        if metadata:
            full_metadata.update(metadata)
        return self.create_node(
            node_type="entity",
            content=name,
            tier="session",
            confidence=1.0,
            metadata=full_metadata if full_metadata else None,
        )

    def find(
        self,
        query: str,
        k: int = 10,
        node_type: str | None = None,
        min_confidence: float | None = None,
    ) -> list[SearchResult]:
        """
        Search memory (alias for search with intuitive parameter names).

        Args:
            query: Search query (uses FTS with stemming)
            k: Maximum number of results (default 10)
            node_type: Filter by node type (e.g., "fact", "experience")
            min_confidence: Minimum confidence threshold

        Returns:
            List of SearchResult objects

        Example:
            >>> store = MemoryStore(":memory:")
            >>> results = store.find("authentication JWT")
            >>> results = store.find("error handling", k=5, node_type="experience")
        """
        return self.search(
            query=query,
            limit=k,
            node_type=node_type,
            min_confidence=min_confidence,
        )

    # =========================================================================
    # Convenience Methods for Two-Node Relationships
    # =========================================================================

    def link(
        self,
        source_id: str,
        target_id: str,
        label: str,
        edge_type: str = "relation",
        weight: float = 1.0,
    ) -> str:
        """
        Create a simple two-node edge (convenience method).

        This is the recommended way to create relationships between two nodes.
        For hyperedges connecting 3+ nodes, use `create_edge()` instead.

        Args:
            source_id: Source node ID (the "from" node)
            target_id: Target node ID (the "to" node)
            label: Semantic label describing the relationship. Common labels:
                - "supports", "contradicts" (evidence relationships)
                - "implements", "extends" (code relationships)
                - "causes", "blocks" (causal relationships)
                - "related_to", "similar_to" (general associations)
            edge_type: Edge type (default "relation"). Valid types:
                - "relation": General relationships (default)
                - "composition": Part-whole (source is part of target)
                - "causation": Cause-effect (source causes target)
                - "context": Contextual association
            weight: Edge weight 0.0-1.0 (default 1.0)

        Returns:
            Edge ID (UUID string)

        Example:
            >>> store = MemoryStore(":memory:")
            >>> fact_id = store.create_node("fact", "Auth uses JWT")
            >>> decision_id = store.create_node("decision", "Use refresh tokens")
            >>> edge_id = store.link(fact_id, decision_id, "supports")
            >>>
            >>> # With explicit edge type
            >>> store.link(cause_id, effect_id, "triggers", edge_type="causation")
        """
        return self.create_edge(
            edge_type=edge_type,
            label=label,
            members=[
                {"node_id": source_id, "role": "subject", "position": 0},
                {"node_id": target_id, "role": "object", "position": 1},
            ],
            weight=weight,
        )

    def unlink(
        self,
        source_id: str,
        target_id: str,
        label: str | None = None,
    ) -> int:
        """
        Remove edges between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            label: Optional label filter. If None, removes all edges between the nodes.

        Returns:
            Number of edges removed

        Example:
            >>> store.unlink(fact_id, decision_id, "supports")  # Remove specific edge
            >>> store.unlink(node1, node2)  # Remove all edges between nodes
        """
        conn = self._get_connection()
        try:
            # Find edges where source is subject and target is object
            if label:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT h.id FROM hyperedges h
                    JOIN membership m1 ON h.id = m1.hyperedge_id
                    JOIN membership m2 ON h.id = m2.hyperedge_id
                    WHERE m1.node_id = ? AND m1.role = 'subject'
                    AND m2.node_id = ? AND m2.role = 'object'
                    AND h.label = ?
                    """,
                    (source_id, target_id, label),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT h.id FROM hyperedges h
                    JOIN membership m1 ON h.id = m1.hyperedge_id
                    JOIN membership m2 ON h.id = m2.hyperedge_id
                    WHERE m1.node_id = ? AND m1.role = 'subject'
                    AND m2.node_id = ? AND m2.role = 'object'
                    """,
                    (source_id, target_id),
                )

            edge_ids = [row[0] for row in cursor.fetchall()]

            # Delete the edges
            for edge_id in edge_ids:
                conn.execute("DELETE FROM hyperedges WHERE id = ?", (edge_id,))

            conn.commit()
            return len(edge_ids)
        finally:
            conn.close()

    def get_links(
        self,
        node_id: str,
        direction: str = "both",
        label: str | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """
        Get all links for a node.

        Args:
            node_id: Node ID to get links for
            direction: "outgoing" (node is source), "incoming" (node is target),
                      or "both" (default)
            label: Optional label filter

        Returns:
            List of (edge_id, other_node_id, label, weight) tuples

        Example:
            >>> links = store.get_links(fact_id, direction="outgoing")
            >>> for edge_id, target_id, label, weight in links:
            ...     print(f"{label} -> {target_id}")
        """
        conn = self._get_connection()
        try:
            results = []

            if direction in ("outgoing", "both"):
                # Node is subject, find objects
                query = """
                    SELECT h.id, m2.node_id, h.label, h.weight
                    FROM hyperedges h
                    JOIN membership m1 ON h.id = m1.hyperedge_id
                    JOIN membership m2 ON h.id = m2.hyperedge_id
                    WHERE m1.node_id = ? AND m1.role = 'subject'
                    AND m2.role = 'object'
                """
                params: list[Any] = [node_id]
                if label:
                    query += " AND h.label = ?"
                    params.append(label)

                cursor = conn.execute(query, params)
                for row in cursor.fetchall():
                    results.append((row[0], row[1], row[2], row[3]))

            if direction in ("incoming", "both"):
                # Node is object, find subjects
                query = """
                    SELECT h.id, m1.node_id, h.label, h.weight
                    FROM hyperedges h
                    JOIN membership m1 ON h.id = m1.hyperedge_id
                    JOIN membership m2 ON h.id = m2.hyperedge_id
                    WHERE m2.node_id = ? AND m2.role = 'object'
                    AND m1.role = 'subject'
                """
                params = [node_id]
                if label:
                    query += " AND h.label = ?"
                    params.append(label)

                cursor = conn.execute(query, params)
                for row in cursor.fetchall():
                    # For incoming, the "other" node is the subject
                    results.append((row[0], row[1], row[2], row[3]))

            return results
        finally:
            conn.close()

    def get_related_nodes(
        self,
        node_id: str,
        edge_type: str | None = None,
    ) -> list[Node]:
        """
        Get nodes related to a given node via hyperedges.

        Implements: Spec SPEC-02.26

        Args:
            node_id: Source node ID
            edge_type: Optional filter by edge type

        Returns:
            List of related nodes
        """
        conn = self._get_connection()
        try:
            # Find related node IDs via membership table
            if edge_type is not None:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT m1.node_id FROM membership m1
                    JOIN membership m2 ON m1.hyperedge_id = m2.hyperedge_id
                    JOIN hyperedges h ON m1.hyperedge_id = h.id
                    WHERE m2.node_id = ? AND m1.node_id != ?
                    AND h.edge_type = ?
                    """,
                    (node_id, node_id, edge_type),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT m1.node_id FROM membership m1
                    JOIN membership m2 ON m1.hyperedge_id = m2.hyperedge_id
                    WHERE m2.node_id = ? AND m1.node_id != ?
                    """,
                    (node_id, node_id),
                )

            related_ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        # Fetch nodes from rlm_core
        nodes = []
        for nid in related_ids:
            node = self._get_node_core(nid)
            if node is not None and node.tier != "archive":
                nodes.append(node)
        return nodes

    def query_edges(
        self,
        edge_type: str | None = None,
        label: str | None = None,
    ) -> list[Hyperedge]:
        """Query hyperedges with filters."""
        conditions = []
        values: list[Any] = []

        if edge_type is not None:
            conditions.append("edge_type = ?")
            values.append(edge_type)

        if label is not None:
            conditions.append("label = ?")
            values.append(label)

        query = "SELECT * FROM hyperedges"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        conn = self._get_connection()
        try:
            cursor = conn.execute(query, values)
            return [
                Hyperedge(
                    id=row["id"],
                    type=row["edge_type"],
                    label=row["label"],
                    weight=row["weight"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_edges_for_node(self, node_id: str) -> list[Hyperedge]:
        """Get all edges a node participates in."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT DISTINCT h.* FROM hyperedges h
                JOIN membership m ON h.id = m.hyperedge_id
                WHERE m.node_id = ?
                """,
                (node_id,),
            )
            return [
                Hyperedge(
                    id=row["id"],
                    type=row["edge_type"],
                    label=row["label"],
                    weight=row["weight"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    # =========================================================================
    # Evidence Linking (for Memory + Reasoning Integration)
    # =========================================================================

    def create_evidence_edge(
        self,
        label: str,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
    ) -> str:
        """
        Create an evidence edge linking facts to decisions or outcomes to facts.

        Evidence labels:
        - supports: fact → option (evidence for a choice)
        - contradicts: fact → option (evidence against a choice)
        - validates: outcome → fact (outcome confirms fact)
        - invalidates: outcome → fact (outcome refutes fact)

        Args:
            label: Evidence label (supports, contradicts, validates, invalidates)
            source_id: Source node ID (fact or outcome)
            target_id: Target node ID (option or fact)
            weight: Edge weight representing evidence strength (0.0-1.0)

        Returns:
            Edge ID

        Raises:
            ValueError: If label is not a valid evidence label
        """
        if label not in self.EVIDENCE_LABELS:
            raise ValueError(
                f"Invalid evidence label: {label}. Must be one of: {self.EVIDENCE_LABELS}"
            )

        return self.create_edge(
            edge_type="relation",
            label=label,
            members=[
                {"node_id": source_id, "role": "subject", "position": 0},
                {"node_id": target_id, "role": "object", "position": 1},
            ],
            weight=weight,
        )

    def get_supporting_facts(self, option_id: str) -> list[tuple[str, float]]:
        """
        Get facts that support an option.

        Args:
            option_id: The option node ID

        Returns:
            List of (fact_id, weight) tuples
        """
        return self._get_evidence_for_target(option_id, "supports")

    def get_contradicting_facts(self, option_id: str) -> list[tuple[str, float]]:
        """
        Get facts that contradict an option.

        Args:
            option_id: The option node ID

        Returns:
            List of (fact_id, weight) tuples
        """
        return self._get_evidence_for_target(option_id, "contradicts")

    def get_validating_outcomes(self, fact_id: str) -> list[tuple[str, float]]:
        """
        Get outcomes that validate a fact.

        Args:
            fact_id: The fact node ID

        Returns:
            List of (outcome_id, weight) tuples
        """
        return self._get_evidence_for_target(fact_id, "validates")

    def get_invalidating_outcomes(self, fact_id: str) -> list[tuple[str, float]]:
        """
        Get outcomes that invalidate a fact.

        Args:
            fact_id: The fact node ID

        Returns:
            List of (outcome_id, weight) tuples
        """
        return self._get_evidence_for_target(fact_id, "invalidates")

    def _get_evidence_for_target(self, target_id: str, label: str) -> list[tuple[str, float]]:
        """
        Get evidence nodes pointing to a target with a specific label.

        Args:
            target_id: Target node ID
            label: Evidence label

        Returns:
            List of (source_id, weight) tuples
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT m_source.node_id, h.weight
                FROM hyperedges h
                JOIN membership m_target ON h.id = m_target.hyperedge_id
                JOIN membership m_source ON h.id = m_source.hyperedge_id
                WHERE h.label = ?
                  AND m_target.node_id = ?
                  AND m_target.role = 'object'
                  AND m_source.role = 'subject'
                """,
                (label, target_id),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_evidence_targets(self, source_id: str, label: str) -> list[tuple[str, float]]:
        """
        Get nodes that a source provides evidence for.

        Args:
            source_id: Source node ID (fact or outcome)
            label: Evidence label

        Returns:
            List of (target_id, weight) tuples
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT m_target.node_id, h.weight
                FROM hyperedges h
                JOIN membership m_source ON h.id = m_source.hyperedge_id
                JOIN membership m_target ON h.id = m_target.hyperedge_id
                WHERE h.label = ?
                  AND m_source.node_id = ?
                  AND m_source.role = 'subject'
                  AND m_target.role = 'object'
                """,
                (label, source_id),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]
        finally:
            conn.close()

    # =========================================================================
    # Evolution Log (SPEC-02.19)
    # =========================================================================

    def _log_evolution(
        self,
        conn: sqlite3.Connection,
        operation: str,
        node_ids: list[str],
        from_tier: str | None = None,
        to_tier: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        """Log an evolution event."""
        tier_to_int = {"task": 0, "session": 1, "longterm": 2, "archive": 3}
        from_tier_int = tier_to_int.get(from_tier) if from_tier else None
        to_tier_int = tier_to_int.get(to_tier) if to_tier else None
        # rlm_core's evolution_log has single node_id, not node_ids array
        for nid in node_ids:
            conn.execute(
                """
                INSERT INTO evolution_log (node_id, operation, from_tier, to_tier, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (nid, operation, from_tier_int, to_tier_int, reasoning),
            )

    def get_evolution_log(
        self,
        node_id: str | None = None,
        operation_type: str | None = None,
    ) -> list[EvolutionLogEntry]:
        """
        Get evolution log entries.

        Args:
            node_id: Optional filter by node ID
            operation_type: Optional filter by operation type

        Returns:
            List of EvolutionLogEntry objects
        """
        conn = self._get_connection()
        try:
            int_to_tier = {0: "task", 1: "session", 2: "longterm", 3: "archive"}
            conditions = []
            values: list[Any] = []

            if node_id is not None:
                conditions.append("node_id = ?")
                values.append(node_id)

            if operation_type is not None:
                conditions.append("operation = ?")
                values.append(operation_type)

            query = "SELECT * FROM evolution_log"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY created_at DESC"

            cursor = conn.execute(query, values)

            return [
                EvolutionLogEntry(
                    id=row["id"],
                    timestamp=0,  # rlm_core uses created_at text, not integer timestamp
                    operation=row["operation"],
                    node_ids=[row["node_id"]],
                    from_tier=int_to_tier.get(row["from_tier"])
                    if row["from_tier"] is not None
                    else None,
                    to_tier=int_to_tier.get(row["to_tier"]) if row["to_tier"] is not None else None,
                    reasoning=row["reason"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def log_evolution(
        self,
        operation: str,
        node_ids: list[str],
        from_tier: str | None = None,
        to_tier: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        """
        Public method to log an evolution event.

        Implements: Spec SPEC-03.06, SPEC-03.13, SPEC-03.20

        Args:
            operation: Type of operation (consolidate, promote, decay, etc.)
            node_ids: List of affected node IDs
            from_tier: Original tier (if applicable)
            to_tier: New tier (if applicable)
            reasoning: Explanation of why this operation occurred
        """
        conn = self._get_connection()
        try:
            self._log_evolution(conn, operation, node_ids, from_tier, to_tier, reasoning)
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Confidence Update Logging (Phase 3: Memory Integration)
    # =========================================================================

    def log_confidence_update(
        self,
        node_id: str,
        old_confidence: float,
        new_confidence: float,
        trigger_type: str,
        trigger_id: str | None = None,
    ) -> int:
        """
        Log a confidence update to the audit trail.

        Args:
            node_id: Node ID being updated
            old_confidence: Previous confidence value
            new_confidence: New confidence value
            trigger_type: What caused this update (outcome, decay, consolidation, manual)
            trigger_id: Optional ID of the trigger (e.g., outcome_id for outcome triggers)

        Returns:
            ID of the log entry
        """
        if trigger_type not in VALID_CONFIDENCE_TRIGGERS:
            raise ValueError(
                f"Invalid trigger type: {trigger_type}. Must be one of: {VALID_CONFIDENCE_TRIGGERS}"
            )

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO confidence_updates
                (node_id, old_confidence, new_confidence, trigger_type, trigger_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (node_id, old_confidence, new_confidence, trigger_type, trigger_id),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            conn.close()

    def get_confidence_history(
        self,
        node_id: str,
        limit: int | None = None,
    ) -> list[ConfidenceUpdate]:
        """
        Get confidence update history for a node.

        Args:
            node_id: Node ID to get history for
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of ConfidenceUpdate entries, most recent first
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        try:
            query = """
                SELECT id, node_id, old_confidence, new_confidence,
                       trigger_type, trigger_id, timestamp
                FROM confidence_updates
                WHERE node_id = ?
                ORDER BY timestamp DESC, id DESC
            """
            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query, (node_id,))
            return [
                ConfidenceUpdate(
                    id=row["id"],
                    node_id=row["node_id"],
                    old_confidence=row["old_confidence"],
                    new_confidence=row["new_confidence"],
                    trigger_type=row["trigger_type"],
                    trigger_id=row["trigger_id"],
                    timestamp=row["timestamp"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_confidence_updates_by_trigger(
        self,
        trigger_type: str,
        trigger_id: str | None = None,
    ) -> list[ConfidenceUpdate]:
        """
        Get all confidence updates for a specific trigger type/ID.

        Args:
            trigger_type: Type of trigger (outcome, decay, consolidation, manual)
            trigger_id: Optional specific trigger ID

        Returns:
            List of ConfidenceUpdate entries
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        try:
            if trigger_id:
                cursor = conn.execute(
                    """
                    SELECT id, node_id, old_confidence, new_confidence,
                           trigger_type, trigger_id, timestamp
                    FROM confidence_updates
                    WHERE trigger_type = ? AND trigger_id = ?
                    ORDER BY timestamp DESC, id DESC
                    """,
                    (trigger_type, trigger_id),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, node_id, old_confidence, new_confidence,
                           trigger_type, trigger_id, timestamp
                    FROM confidence_updates
                    WHERE trigger_type = ?
                    ORDER BY timestamp DESC, id DESC
                    """,
                    (trigger_type,),
                )

            return [
                ConfidenceUpdate(
                    id=row["id"],
                    node_id=row["node_id"],
                    old_confidence=row["old_confidence"],
                    new_confidence=row["new_confidence"],
                    trigger_type=row["trigger_type"],
                    trigger_id=row["trigger_id"],
                    timestamp=row["timestamp"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_confidence_drift_report(
        self,
        node_id: str,
    ) -> dict[str, Any]:
        """
        Get a summary of confidence changes for a node.

        Args:
            node_id: Node ID to analyze

        Returns:
            Dict with statistics about confidence changes
        """
        history = self.get_confidence_history(node_id)

        if not history:
            return {
                "node_id": node_id,
                "total_updates": 0,
                "current_confidence": None,
                "initial_confidence": None,
                "total_drift": 0.0,
                "updates_by_trigger": {},
            }

        # Count updates by trigger type
        updates_by_trigger: dict[str, int] = {}
        for update in history:
            updates_by_trigger[update.trigger_type] = (
                updates_by_trigger.get(update.trigger_type, 0) + 1
            )

        # History is ordered by timestamp DESC, so first is most recent
        current = history[0].new_confidence
        initial = history[-1].old_confidence

        return {
            "node_id": node_id,
            "total_updates": len(history),
            "current_confidence": current,
            "initial_confidence": initial,
            "total_drift": current - initial,
            "updates_by_trigger": updates_by_trigger,
        }

    def _set_last_accessed(self, node_id: str, timestamp: Any) -> bool:
        """
        Set the last_accessed time for a node (for testing).

        Args:
            node_id: Node ID to update
            timestamp: datetime object

        Returns:
            True if updated, False if node not found
        """
        from datetime import datetime

        if isinstance(timestamp, datetime):
            rfc3339 = timestamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000000+00:00")
        else:
            rfc3339 = str(timestamp)
        return self._core_store.update_fields(node_id, last_accessed=rfc3339)

    def get_nodes_by_metadata(
        self,
        key: str,
        value: Any,
        tier: str | None = None,
        include_archived: bool = False,
    ) -> list[Node]:
        """
        Get nodes by metadata key-value.

        Args:
            key: Metadata key to search
            value: Metadata value to match
            tier: Optional tier filter
            include_archived: Include archived nodes

        Returns:
            List of matching nodes
        """
        # Query all nodes from rlm_core and filter by metadata
        all_nodes = self._query_nodes_core(tier=tier, limit=1000)
        results = []
        for node in all_nodes:
            if not include_archived and node.tier == "archive":
                continue
            if node.metadata.get(key) == value:
                results.append(node)
        return results

    # =========================================================================
    # FTS5 Full-Text Search (Phase 4: Massive Context)
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 20,
        node_type: str | None = None,
        min_confidence: float | None = None,
        *,
        k: int | None = None,
        type: str | None = None,
    ) -> list[SearchResult]:
        """
        Search node content using FTS5 with BM25 ranking.

        Implements: Phase 4 Massive Context (SPEC-01.03)

        Args:
            query: Search query (supports FTS5 syntax: AND, OR, NOT, prefix*, "phrase")
            limit: Maximum results to return (alias: k)
            node_type: Filter by node type (alias: type)
            min_confidence: Minimum confidence threshold
            k: Alias for limit (common in ML/vector store APIs)
            type: Alias for node_type

        Returns:
            List of SearchResult objects sorted by BM25 relevance

        Example:
            >>> store = MemoryStore(":memory:")
            >>> results = store.search("error handling", k=5)
            >>> results = store.search("auth", type="fact", limit=10)
        """
        # Handle parameter aliases
        if k is not None:
            limit = k
        if type is not None:
            node_type = type
        if not query or not query.strip():
            return []

        try:
            results = self._search_core(query, limit)

            # Apply filters
            if node_type:
                results = [r for r in results if r.node_type == node_type]
            if min_confidence is not None:
                results = [r for r in results if r.bm25_score >= min_confidence]

            return results[:limit]
        except Exception:
            return []

    def search_prefix(self, prefix: str, limit: int = 20) -> list[SearchResult]:
        """
        Search for nodes matching a prefix.

        Args:
            prefix: Prefix to search for (will add * for prefix matching)
            limit: Maximum results

        Returns:
            List of SearchResult objects
        """
        if not prefix or not prefix.strip():
            return []

        # Add prefix matching syntax
        query = f"{prefix.strip()}*"
        return self.search(query, limit=limit)

    def search_phrase(self, phrase: str, limit: int = 20) -> list[SearchResult]:
        """
        Search for an exact phrase.

        Args:
            phrase: Exact phrase to search for
            limit: Maximum results

        Returns:
            List of SearchResult objects
        """
        if not phrase or not phrase.strip():
            return []

        # Wrap in quotes for exact phrase matching
        query = f'"{phrase.strip()}"'
        return self.search(query, limit=limit)

    def rebuild_fts_index(self) -> int:
        """
        Rebuild the FTS index. Managed by rlm_core internally.

        Returns:
            Number of nodes indexed
        """
        # rlm_core manages its own search index
        all_nodes = self._query_nodes_core(limit=10000)
        return len(all_nodes)

    def get_fts_stats(self) -> dict[str, Any]:
        """
        Get statistics about the search index.

        Returns:
            Dict with index statistics
        """
        all_nodes = self._query_nodes_core(limit=10000)
        total_chars = sum(len(n.content) for n in all_nodes)
        return {
            "indexed_documents": len(all_nodes),
            "total_characters": total_chars,
            "estimated_tokens": total_chars // 4,
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    # =========================================================================
    # Micro Mode Memory Access (SPEC-14.50-14.55)
    # =========================================================================

    def retrieve_for_query(
        self,
        query: str,
        k: int = 5,
        use_embedding: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant facts for a query.

        Implements: SPEC-14.51, SPEC-14.53, SPEC-14.54

        In micro mode (use_embedding=False), uses keyword-based FTS5 search.
        On escalation (use_embedding=True), could use embedding-based retrieval.

        Args:
            query: The user's query to find relevant facts for
            k: Maximum number of facts to return
            use_embedding: If True, use embedding-based retrieval (SPEC-14.55)

        Returns:
            List of fact dicts with id, content, confidence, score
        """
        if use_embedding:
            # SPEC-14.55: Full embedding-based retrieval on escalation
            # For now, fall back to FTS (embedding implementation is separate)
            return self._retrieve_with_fts(query, k)

        # SPEC-14.54: Keyword matching in micro mode
        return self._retrieve_with_fts(query, k)

    def _retrieve_with_fts(self, query: str, k: int) -> list[dict[str, Any]]:
        """
        Retrieve relevant facts using FTS5 keyword search.

        Implements: SPEC-14.53, SPEC-14.54
        """
        # Extract keywords from query for FTS
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # Search for facts matching keywords
        results = self.search(
            query=" OR ".join(keywords),
            k=k,
            type="fact",
            min_confidence=0.3,  # Include lower-confidence facts
        )

        return [
            {
                "id": r.node_id,
                "content": r.content,
                "score": r.bm25_score,
                "snippet": r.snippet,
            }
            for r in results
        ]

    def _extract_keywords(self, query: str) -> list[str]:
        """
        Extract significant keywords from a query for FTS search.

        Simple keyword extraction without LLM (SPEC-14.53).
        """
        # Common stop words to filter out
        stop_words = frozenset(
            {
                "a",
                "an",
                "the",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "shall",
                "can",
                "need",
                "dare",
                "ought",
                "used",
                "to",
                "of",
                "in",
                "for",
                "on",
                "with",
                "at",
                "by",
                "from",
                "as",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "under",
                "again",
                "further",
                "then",
                "once",
                "here",
                "there",
                "when",
                "where",
                "why",
                "how",
                "all",
                "each",
                "few",
                "more",
                "most",
                "other",
                "some",
                "such",
                "no",
                "nor",
                "not",
                "only",
                "own",
                "same",
                "so",
                "than",
                "too",
                "very",
                "just",
                "and",
                "but",
                "if",
                "or",
                "because",
                "until",
                "while",
                "this",
                "that",
                "these",
                "those",
                "what",
                "which",
                "who",
                "whom",
                "whose",
                "i",
                "me",
                "my",
                "we",
                "us",
                "you",
                "your",
                "he",
                "him",
                "his",
                "she",
                "her",
                "it",
                "its",
                "they",
                "them",
                "their",
            }
        )

        # Tokenize and filter
        import re

        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords[:10]  # Limit to top 10 keywords

    def micro_add_fact(
        self,
        content: str,
        confidence: float = 0.7,
    ) -> str:
        """
        Add a fact in micro mode (no LLM processing).

        Implements: SPEC-14.52

        Args:
            content: The factual statement to store
            confidence: Confidence score 0.0-1.0 (default 0.7)

        Returns:
            Node ID
        """
        return self.add_fact(
            content=content,
            confidence=confidence,
            tier="task",  # Micro mode facts start at task tier
        )


def create_micro_memory_loader(
    store: MemoryStore,
    query: str,
    k: int = 5,
) -> Callable[[], list[dict[str, Any]]]:
    """
    Create a lazy memory loader for micro mode context.

    Implements: SPEC-14.51

    Args:
        store: Memory store instance
        query: Query to retrieve relevant facts for
        k: Maximum number of facts to return

    Returns:
        Callable that retrieves memory when invoked
    """

    def loader() -> list[dict[str, Any]]:
        return store.retrieve_for_query(query, k=k, use_embedding=False)

    return loader


__all__ = [
    "MemoryStore",
    "Node",
    "Hyperedge",
    "EvolutionLogEntry",
    "SearchResult",
    "ConfidenceUpdate",
    # Micro mode (SPEC-14.50-14.55)
    "create_micro_memory_loader",
]
