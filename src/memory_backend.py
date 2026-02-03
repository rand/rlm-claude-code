"""
Memory backend abstraction for pluggable storage.

Implements: SPEC-12.20-12.25

Provides MemoryBackend protocol with SQLite and InMemory implementations,
configurable backend selection, and migration tooling.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class NodeType(Enum):
    """Type of memory node."""

    FACT = "fact"
    EXPERIENCE = "experience"
    CONCEPT = "concept"


class EdgeType(Enum):
    """Type of edge between nodes."""

    RELATED = "related"
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"


@dataclass
class Node:
    """
    Memory node.

    Implements: SPEC-12.21
    """

    id: str
    content: str
    node_type: NodeType
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class Edge:
    """
    Edge between memory nodes.

    Implements: SPEC-12.21
    """

    id: str
    from_node: str
    to_node: str
    edge_type: EdgeType
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """
    Search result from memory backend.

    Implements: SPEC-12.21
    """

    id: str
    content: str
    node_type: NodeType
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryBackendType(Enum):
    """Type of memory backend."""

    SQLITE = "sqlite"
    INMEMORY = "inmemory"
    POSTGRES = "postgres"  # Future


@dataclass
class MemoryBackendConfig:
    """
    Configuration for memory backend.

    Implements: SPEC-12.23
    """

    backend_type: MemoryBackendType = MemoryBackendType.INMEMORY
    connection_string: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


class MemoryBackend(ABC):
    """
    Protocol for memory storage backends.

    Implements: SPEC-12.20, SPEC-12.21
    """

    @abstractmethod
    def create_node(
        self,
        content: str,
        node_type: NodeType,
        metadata: dict[str, Any],
    ) -> str:
        """
        Create a new memory node.

        Args:
            content: Node content
            node_type: Type of node
            metadata: Additional metadata

        Returns:
            Node ID
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Node | None:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node or None if not found
        """
        pass

    @abstractmethod
    def update_node(
        self,
        node_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update a node.

        Args:
            node_id: Node ID
            content: New content (optional)
            metadata: New metadata (optional)
        """
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node ID

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        node_type: NodeType | None = None,
    ) -> list[SearchResult]:
        """
        Search for nodes.

        Args:
            query: Search query
            limit: Maximum results
            node_type: Filter by type

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def create_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> str:
        """
        Create an edge between nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            edge_type: Type of edge
            weight: Edge weight

        Returns:
            Edge ID
        """
        pass

    @abstractmethod
    def get_edges(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        """
        Get edges for a node.

        Args:
            node_id: Node ID
            edge_type: Filter by type

        Returns:
            List of edges
        """
        pass

    @abstractmethod
    def get_all_nodes(self) -> list[Node]:
        """Get all nodes (for migration)."""
        pass

    @abstractmethod
    def get_all_edges(self) -> list[Edge]:
        """Get all edges (for migration)."""
        pass


class InMemoryBackend(MemoryBackend):
    """
    In-memory implementation for testing.

    Implements: SPEC-12.22
    """

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

    def create_node(
        self,
        content: str,
        node_type: NodeType,
        metadata: dict[str, Any],
    ) -> str:
        """Create a new memory node."""
        node_id = str(uuid.uuid4())
        now = datetime.now()
        self._nodes[node_id] = Node(
            id=node_id,
            content=content,
            node_type=node_type,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )
        return node_id

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def update_node(
        self,
        node_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update a node."""
        if node_id not in self._nodes:
            return

        node = self._nodes[node_id]
        if content is not None:
            self._nodes[node_id] = Node(
                id=node.id,
                content=content,
                node_type=node.node_type,
                metadata=node.metadata if metadata is None else metadata,
                created_at=node.created_at,
                updated_at=datetime.now(),
            )
        elif metadata is not None:
            self._nodes[node_id] = Node(
                id=node.id,
                content=node.content,
                node_type=node.node_type,
                metadata=metadata,
                created_at=node.created_at,
                updated_at=datetime.now(),
            )

    def delete_node(self, node_id: str) -> bool:
        """Delete a node."""
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        # Also remove related edges
        self._edges = {
            eid: e
            for eid, e in self._edges.items()
            if e.from_node != node_id and e.to_node != node_id
        }
        return True

    def search(
        self,
        query: str,
        limit: int = 10,
        node_type: NodeType | None = None,
    ) -> list[SearchResult]:
        """Search for nodes."""
        results = []
        query_lower = query.lower()

        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue

            if query_lower in node.content.lower():
                # Simple relevance score based on position
                pos = node.content.lower().find(query_lower)
                score = 1.0 - (pos / max(len(node.content), 1))

                results.append(
                    SearchResult(
                        id=node.id,
                        content=node.content,
                        node_type=node.node_type,
                        score=score,
                        metadata=node.metadata,
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def create_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> str:
        """Create an edge between nodes."""
        edge_id = str(uuid.uuid4())
        self._edges[edge_id] = Edge(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            edge_type=edge_type,
            weight=weight,
            created_at=datetime.now(),
        )
        return edge_id

    def get_edges(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        """Get edges for a node."""
        edges = []
        for edge in self._edges.values():
            if edge.from_node != node_id and edge.to_node != node_id:
                continue
            if edge_type is not None and edge.edge_type != edge_type:
                continue
            edges.append(edge)
        return edges

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_all_edges(self) -> list[Edge]:
        """Get all edges."""
        return list(self._edges.values())


class SQLiteBackend(MemoryBackend):
    """
    SQLite implementation for persistence.

    Implements: SPEC-12.22
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to database file
        """
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                from_node TEXT NOT NULL,
                to_node TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (from_node) REFERENCES nodes(id),
                FOREIGN KEY (to_node) REFERENCES nodes(id)
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_node);
        """
        )
        conn.commit()

    def create_node(
        self,
        content: str,
        node_type: NodeType,
        metadata: dict[str, Any],
    ) -> str:
        """Create a new memory node."""
        node_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO nodes (id, content, node_type, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (node_id, content, node_type.value, json.dumps(metadata), now, now),
        )
        conn.commit()
        return node_id

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()

        if row is None:
            return None

        return Node(
            id=row["id"],
            content=row["content"],
            node_type=NodeType(row["node_type"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def update_node(
        self,
        node_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update a node."""
        conn = self._get_conn()
        now = datetime.now().isoformat()

        if content is not None:
            conn.execute(
                "UPDATE nodes SET content = ?, updated_at = ? WHERE id = ?",
                (content, now, node_id),
            )
        if metadata is not None:
            conn.execute(
                "UPDATE nodes SET metadata = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata), now, node_id),
            )
        conn.commit()

    def delete_node(self, node_id: str) -> bool:
        """Delete a node."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM nodes WHERE id = ?",
            (node_id,),
        )
        # Also delete related edges
        conn.execute(
            "DELETE FROM edges WHERE from_node = ? OR to_node = ?",
            (node_id, node_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def search(
        self,
        query: str,
        limit: int = 10,
        node_type: NodeType | None = None,
    ) -> list[SearchResult]:
        """Search for nodes."""
        conn = self._get_conn()

        if node_type is not None:
            rows = conn.execute(
                """
                SELECT * FROM nodes
                WHERE content LIKE ? AND node_type = ?
                LIMIT ?
                """,
                (f"%{query}%", node_type.value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM nodes
                WHERE content LIKE ?
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()

        results = []
        for row in rows:
            # Simple relevance score
            content = row["content"]
            pos = content.lower().find(query.lower())
            score = 1.0 - (pos / max(len(content), 1)) if pos >= 0 else 0.0

            results.append(
                SearchResult(
                    id=row["id"],
                    content=content,
                    node_type=NodeType(row["node_type"]),
                    score=score,
                    metadata=json.loads(row["metadata"]),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def create_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> str:
        """Create an edge between nodes."""
        edge_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO edges (id, from_node, to_node, edge_type, weight, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (edge_id, from_node, to_node, edge_type.value, weight, now),
        )
        conn.commit()
        return edge_id

    def get_edges(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        """Get edges for a node."""
        conn = self._get_conn()

        if edge_type is not None:
            rows = conn.execute(
                """
                SELECT * FROM edges
                WHERE (from_node = ? OR to_node = ?) AND edge_type = ?
                """,
                (node_id, node_id, edge_type.value),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM edges
                WHERE from_node = ? OR to_node = ?
                """,
                (node_id, node_id),
            ).fetchall()

        return [
            Edge(
                id=row["id"],
                from_node=row["from_node"],
                to_node=row["to_node"],
                edge_type=EdgeType(row["edge_type"]),
                weight=row["weight"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM nodes").fetchall()
        return [
            Node(
                id=row["id"],
                content=row["content"],
                node_type=NodeType(row["node_type"]),
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def get_all_edges(self) -> list[Edge]:
        """Get all edges."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM edges").fetchall()
        return [
            Edge(
                id=row["id"],
                from_node=row["from_node"],
                to_node=row["to_node"],
                edge_type=EdgeType(row["edge_type"]),
                weight=row["weight"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]


@dataclass
class MigrationStats:
    """
    Statistics from migration.

    Implements: SPEC-12.25
    """

    nodes_migrated: int
    edges_migrated: int
    duration_ms: float
    errors: list[str] = field(default_factory=list)


class MemoryMigrator:
    """
    Migrate data between memory backends.

    Implements: SPEC-12.25
    """

    def __init__(self, source: MemoryBackend, target: MemoryBackend) -> None:
        """
        Initialize migrator.

        Args:
            source: Source backend
            target: Target backend
        """
        self._source = source
        self._target = target

    def migrate(self) -> MigrationStats:
        """
        Migrate all data from source to target.

        Returns:
            Migration statistics
        """
        start_time = time.time()
        errors: list[str] = []
        node_id_map: dict[str, str] = {}

        # Migrate nodes
        nodes = self._source.get_all_nodes()
        nodes_migrated = 0

        for node in nodes:
            try:
                new_id = self._target.create_node(
                    content=node.content,
                    node_type=node.node_type,
                    metadata=node.metadata,
                )
                node_id_map[node.id] = new_id
                nodes_migrated += 1
            except Exception as e:
                errors.append(f"Failed to migrate node {node.id}: {e}")

        # Migrate edges
        edges = self._source.get_all_edges()
        edges_migrated = 0

        for edge in edges:
            try:
                # Map old node IDs to new ones
                from_node = node_id_map.get(edge.from_node)
                to_node = node_id_map.get(edge.to_node)

                if from_node and to_node:
                    self._target.create_edge(
                        from_node=from_node,
                        to_node=to_node,
                        edge_type=edge.edge_type,
                        weight=edge.weight,
                    )
                    edges_migrated += 1
                else:
                    errors.append(f"Skipped edge {edge.id}: missing node mapping")
            except Exception as e:
                errors.append(f"Failed to migrate edge {edge.id}: {e}")

        duration_ms = (time.time() - start_time) * 1000

        return MigrationStats(
            nodes_migrated=nodes_migrated,
            edges_migrated=edges_migrated,
            duration_ms=duration_ms,
            errors=errors,
        )


def create_backend(config: MemoryBackendConfig) -> MemoryBackend:
    """
    Create a memory backend from configuration.

    Implements: SPEC-12.23

    Args:
        config: Backend configuration

    Returns:
        MemoryBackend instance
    """
    if config.backend_type == MemoryBackendType.SQLITE:
        if config.connection_string is None:
            raise ValueError("SQLite backend requires connection_string")
        return SQLiteBackend(db_path=config.connection_string)
    elif config.backend_type == MemoryBackendType.INMEMORY:
        return InMemoryBackend()
    else:
        raise ValueError(f"Unsupported backend type: {config.backend_type}")


__all__ = [
    "Edge",
    "EdgeType",
    "InMemoryBackend",
    "MemoryBackend",
    "MemoryBackendConfig",
    "MemoryBackendType",
    "MemoryMigrator",
    "MigrationStats",
    "Node",
    "NodeType",
    "SQLiteBackend",
    "SearchResult",
    "create_backend",
]
