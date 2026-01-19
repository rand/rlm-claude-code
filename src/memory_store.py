"""
Persistent hypergraph memory with SQLite storage.

Implements: Spec SPEC-02 Memory Foundation
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

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

SCHEMA_SQL = """
-- Nodes table
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('entity', 'fact', 'experience', 'decision', 'snippet')),
    subtype TEXT,
    content TEXT NOT NULL,
    embedding BLOB,
    tier TEXT DEFAULT 'task' CHECK(tier IN ('task', 'session', 'longterm', 'archive')),
    confidence REAL DEFAULT 0.5 CHECK(confidence >= 0.0 AND confidence <= 1.0),
    provenance TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    updated_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    last_accessed INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    access_count INTEGER DEFAULT 0,
    metadata JSON DEFAULT '{}'
);

-- Hyperedges table
CREATE TABLE IF NOT EXISTS hyperedges (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('relation', 'composition', 'causation', 'context')),
    label TEXT,
    weight REAL DEFAULT 1.0 CHECK(weight >= 0.0)
);

-- Membership table (many-to-many)
CREATE TABLE IF NOT EXISTS membership (
    hyperedge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    position INTEGER DEFAULT 0,
    PRIMARY KEY (hyperedge_id, node_id, role)
);

-- Evolution log table
CREATE TABLE IF NOT EXISTS evolution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    operation TEXT NOT NULL,
    node_ids JSON NOT NULL,
    from_tier TEXT,
    to_tier TEXT,
    reasoning TEXT
);

-- Confidence updates audit table (Phase 3: Memory Integration)
CREATE TABLE IF NOT EXISTS confidence_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    old_confidence REAL NOT NULL,
    new_confidence REAL NOT NULL,
    trigger_type TEXT NOT NULL CHECK(trigger_type IN ('outcome', 'decay', 'consolidation', 'manual')),
    trigger_id TEXT,
    timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_nodes_tier ON nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence);
CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON nodes(last_accessed);
CREATE INDEX IF NOT EXISTS idx_membership_node ON membership(node_id);
CREATE INDEX IF NOT EXISTS idx_membership_edge ON membership(hyperedge_id);
CREATE INDEX IF NOT EXISTS idx_confidence_updates_node ON confidence_updates(node_id);
CREATE INDEX IF NOT EXISTS idx_confidence_updates_trigger ON confidence_updates(trigger_type);

-- FTS5 full-text search index (Phase 4: Massive Context)
-- Uses Porter stemming for better matching and BM25 ranking
CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
    node_id UNINDEXED,
    content,
    type UNINDEXED,
    tokenize='porter'
);

-- Triggers to keep FTS index in sync with nodes table
CREATE TRIGGER IF NOT EXISTS nodes_fts_insert AFTER INSERT ON nodes BEGIN
    INSERT INTO content_fts(node_id, content, type) VALUES (NEW.id, NEW.content, NEW.type);
END;

CREATE TRIGGER IF NOT EXISTS nodes_fts_update AFTER UPDATE OF content ON nodes BEGIN
    DELETE FROM content_fts WHERE node_id = OLD.id;
    INSERT INTO content_fts(node_id, content, type) VALUES (NEW.id, NEW.content, NEW.type);
END;

CREATE TRIGGER IF NOT EXISTS nodes_fts_delete AFTER DELETE ON nodes BEGIN
    DELETE FROM content_fts WHERE node_id = OLD.id;
END;
"""


# =============================================================================
# MemoryStore Class
# =============================================================================


class MemoryStore:
    """
    Persistent hypergraph memory store using SQLite.

    Implements: Spec SPEC-02.01-26

    Features:
    - SQLite with WAL mode for concurrent access
    - Node CRUD operations with tier system
    - Hyperedge support for many-to-many relationships
    - Evolution logging for tier transitions
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
        self._ensure_directory()
        self._init_database()

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
        Initialize database with schema.

        Implements: Spec SPEC-02.04 (WAL mode)
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            # Create schema
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

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

        # Generate UUID (SPEC-02.09)
        node_id = str(uuid.uuid4())

        # Serialize metadata
        metadata_json = json.dumps(metadata or {})

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO nodes (id, type, subtype, content, embedding, tier,
                                   confidence, provenance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    node_type,
                    subtype,
                    content,
                    embedding,
                    tier,
                    confidence,
                    provenance,
                    metadata_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return node_id

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
        conn = self._get_connection()
        try:
            # Update access tracking
            conn.execute(
                """
                UPDATE nodes
                SET last_accessed = strftime('%s', 'now') * 1000,
                    access_count = access_count + 1
                WHERE id = ?
                """,
                (node_id,),
            )
            conn.commit()

            # Build query
            query = "SELECT * FROM nodes WHERE id = ?"
            if not include_archived:
                query += " AND tier != 'archive'"

            cursor = conn.execute(query, (node_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_node(row)
        finally:
            conn.close()

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

        conn = self._get_connection()
        try:
            # Check if node exists and get current tier for logging
            cursor = conn.execute("SELECT tier FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row is None:
                return False

            old_tier = row["tier"]

            # Build update query
            updates = []
            values = []
            for key, value in kwargs.items():
                if key == "metadata":
                    updates.append(f"{key} = ?")
                    values.append(json.dumps(value))
                else:
                    updates.append(f"{key} = ?")
                    values.append(value)

            # Always update updated_at
            updates.append("updated_at = strftime('%s', 'now') * 1000")

            values.append(node_id)

            query = f"UPDATE nodes SET {', '.join(updates)} WHERE id = ?"
            conn.execute(query, values)

            # Log tier transition if tier changed (SPEC-02.19)
            if "tier" in kwargs and kwargs["tier"] != old_tier:
                self._log_evolution(
                    conn,
                    operation="tier_transition",
                    node_ids=[node_id],
                    from_tier=old_tier,
                    to_tier=kwargs["tier"],
                )

            conn.commit()
            return True
        finally:
            conn.close()

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
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

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
        conditions = []
        values: list[Any] = []

        if not include_archived:
            conditions.append("tier != 'archive'")

        if node_type is not None:
            conditions.append("type = ?")
            values.append(node_type)

        if tier is not None:
            conditions.append("tier = ?")
            values.append(tier)

            # Session isolation for task/session tiers only
            # Longterm and archive tiers are shared across all sessions
            if tier in ("task", "session") and session_id is not None:
                conditions.append("json_extract(metadata, '$.session_id') = ?")
                values.append(session_id)

        if min_confidence is not None:
            conditions.append("confidence >= ?")
            values.append(min_confidence)

        query = "SELECT * FROM nodes"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Order by confidence DESC first (most relevant), then last_accessed DESC
        query += " ORDER BY confidence DESC, last_accessed DESC"

        # Default limit to prevent context flooding
        effective_limit = limit if limit is not None else 100
        query += " LIMIT ?"
        values.append(effective_limit)

        conn = self._get_connection()
        try:
            cursor = conn.execute(query, values)
            return [self._row_to_node(row) for row in cursor.fetchall()]
        finally:
            conn.close()

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
            ...     ]
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
                "INSERT INTO hyperedges (id, type, label, weight) VALUES (?, ?, ?, ?)",
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
                type=row["type"],
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
        """Delete a hyperedge."""
        conn = self._get_connection()
        try:
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
            ...     success=True
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
            # Find all edges this node participates in
            query = """
                SELECT DISTINCT n.* FROM nodes n
                JOIN membership m1 ON n.id = m1.node_id
                JOIN membership m2 ON m1.hyperedge_id = m2.hyperedge_id
                WHERE m2.node_id = ? AND n.id != ? AND n.tier != 'archive'
            """
            values: list[Any] = [node_id, node_id]

            if edge_type is not None:
                query = """
                    SELECT DISTINCT n.* FROM nodes n
                    JOIN membership m1 ON n.id = m1.node_id
                    JOIN membership m2 ON m1.hyperedge_id = m2.hyperedge_id
                    JOIN hyperedges h ON m1.hyperedge_id = h.id
                    WHERE m2.node_id = ? AND n.id != ? AND n.tier != 'archive'
                    AND h.type = ?
                """
                values.append(edge_type)

            cursor = conn.execute(query, values)
            return [self._row_to_node(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def query_edges(
        self,
        edge_type: str | None = None,
        label: str | None = None,
    ) -> list[Hyperedge]:
        """Query hyperedges with filters."""
        conditions = []
        values: list[Any] = []

        if edge_type is not None:
            conditions.append("type = ?")
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
                    type=row["type"],
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
                    type=row["type"],
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
        conn.execute(
            """
            INSERT INTO evolution_log (operation, node_ids, from_tier, to_tier, reasoning)
            VALUES (?, ?, ?, ?, ?)
            """,
            (operation, json.dumps(node_ids), from_tier, to_tier, reasoning),
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
            conditions = []
            values: list[Any] = []

            if node_id is not None:
                conditions.append("node_ids LIKE ?")
                values.append(f'%"{node_id}"%')

            if operation_type is not None:
                conditions.append("operation = ?")
                values.append(operation_type)

            query = "SELECT * FROM evolution_log"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"

            cursor = conn.execute(query, values)

            return [
                EvolutionLogEntry(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    operation=row["operation"],
                    node_ids=json.loads(row["node_ids"]),
                    from_tier=row["from_tier"],
                    to_tier=row["to_tier"],
                    reasoning=row["reasoning"],
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
        # Convert datetime to milliseconds timestamp
        from datetime import datetime

        if isinstance(timestamp, datetime):
            ts_ms = int(timestamp.timestamp() * 1000)
        else:
            ts_ms = timestamp

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "UPDATE nodes SET last_accessed = ? WHERE id = ?",
                (ts_ms, node_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

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
        conn = self._get_connection()
        try:
            # Use JSON extract for metadata search
            # json_extract returns the raw value, so compare directly
            conditions = [f"json_extract(metadata, '$.{key}') = ?"]
            values: list[Any] = [value]

            if not include_archived:
                conditions.append("tier != 'archive'")

            if tier is not None:
                conditions.append("tier = ?")
                values.append(tier)

            query = "SELECT * FROM nodes WHERE " + " AND ".join(conditions)
            cursor = conn.execute(query, values)
            return [self._row_to_node(row) for row in cursor.fetchall()]
        finally:
            conn.close()

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

        conn = self._get_connection()
        try:
            # Build query with optional filters
            # BM25 scoring: SQLite returns negative values (more negative = better match)
            # We negate the score so users see positive values (higher = better)
            # Always join with nodes table to exclude archived nodes (soft-deleted)
            base_query = """
                SELECT
                    f.node_id,
                    f.content,
                    f.type,
                    -bm25(content_fts) as score,
                    snippet(content_fts, 1, '<b>', '</b>', '...', 32) as snippet
                FROM content_fts f
                JOIN nodes n ON f.node_id = n.id
                WHERE content_fts MATCH ?
                AND n.tier != 'archive'
            """

            params: list[Any] = [query]
            conditions = []

            if node_type:
                conditions.append("f.type = ?")
                params.append(node_type)

            if min_confidence is not None:
                conditions.append("n.confidence >= ?")
                params.append(min_confidence)

            if conditions:
                base_query += " AND " + " AND ".join(conditions)

            # Higher score = better match (after negation), so order descending
            base_query += " ORDER BY score DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(base_query, params)
            results = []
            for row in cursor.fetchall():
                results.append(
                    SearchResult(
                        node_id=row["node_id"],
                        content=row["content"],
                        node_type=row["type"],
                        bm25_score=row["score"],
                        snippet=row["snippet"],
                    )
                )
            return results

        except Exception:
            # FTS query syntax error or other issue
            return []
        finally:
            conn.close()

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
        Rebuild the FTS index from the nodes table.

        Useful after bulk imports or if index gets out of sync.

        Returns:
            Number of nodes indexed
        """
        conn = self._get_connection()
        try:
            # Clear and rebuild
            conn.execute("DELETE FROM content_fts")
            cursor = conn.execute(
                "INSERT INTO content_fts(node_id, content, type) "
                "SELECT id, content, type FROM nodes WHERE tier != 'archive'"
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_fts_stats(self) -> dict[str, Any]:
        """
        Get statistics about the FTS index.

        Returns:
            Dict with index statistics
        """
        conn = self._get_connection()
        try:
            # Count indexed documents
            cursor = conn.execute("SELECT COUNT(*) as count FROM content_fts")
            doc_count = cursor.fetchone()["count"]

            # Get index size (approximate)
            cursor = conn.execute("SELECT SUM(length(content)) as total_chars FROM content_fts")
            row = cursor.fetchone()
            total_chars = row["total_chars"] if row["total_chars"] else 0

            return {
                "indexed_documents": doc_count,
                "total_characters": total_chars,
                "estimated_tokens": total_chars // 4,
            }
        finally:
            conn.close()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        """Convert a database row to a Node object."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Node(
            id=row["id"],
            type=row["type"],
            content=row["content"],
            tier=row["tier"],
            confidence=row["confidence"],
            subtype=row["subtype"],
            embedding=row["embedding"],
            provenance=row["provenance"],
            metadata=metadata or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
        )


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
        stop_words = frozenset({
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "this", "that", "these", "those", "what",
            "which", "who", "whom", "whose", "i", "me", "my", "we", "us",
            "you", "your", "he", "him", "his", "she", "her", "it", "its",
            "they", "them", "their",
        })

        # Tokenize and filter
        import re
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower())
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
