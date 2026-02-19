from __future__ import annotations

from typing import Any

class NodeType:
    Entity: NodeType
    Fact: NodeType
    Experience: NodeType
    Decision: NodeType
    Snippet: NodeType

class Tier:
    Task: Tier
    Session: Tier
    LongTerm: Tier
    Archive: Tier

    def next(self) -> Tier | None: ...
    def previous(self) -> Tier | None: ...

class Node:
    id: str
    node_type: NodeType
    subtype: str | None
    content: str
    tier: Tier
    confidence: float
    created_at: str
    updated_at: str
    last_accessed: str
    access_count: int
    metadata: dict[str, Any] | None
    provenance_source: str | None
    provenance_ref: str | None
    embedding: list[float] | None

    def __init__(
        self,
        node_type: NodeType,
        content: str,
        id: str | None = None,
        subtype: str | None = None,
        tier: Tier | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        provenance_source: str | None = None,
        provenance_ref: str | None = None,
        embedding: list[float] | None = None,
    ) -> None: ...
    def record_access(self) -> None: ...
    def is_decayed(self, min_confidence: float) -> bool: ...
    def age_hours(self) -> int: ...

class HyperEdge:
    id: str
    edge_type: str
    label: str | None
    weight: float
    created_at: str

    def __init__(
        self,
        edge_type: str,
        node_ids: list[str] | None = None,
        label: str | None = None,
        weight: float | None = None,
    ) -> None: ...
    def node_ids(self) -> list[str]: ...
    def contains(self, node_id: str) -> bool: ...
    def with_member(self, node_id: str, role: str) -> HyperEdge: ...
    @staticmethod
    def binary(edge_type: str, subject_id: str, object_id: str, label: str) -> HyperEdge: ...

class MemoryStore:
    @staticmethod
    def in_memory() -> MemoryStore: ...
    @staticmethod
    def open(path: str) -> MemoryStore: ...
    def add_node(self, node: Node) -> str: ...
    def get_node(self, node_id: str) -> Node | None: ...
    def query_by_type(self, node_type: NodeType, limit: int) -> list[Node]: ...
    def query_by_tier(self, tier: Tier, limit: int) -> list[Node]: ...
    def search_content(self, query: str, limit: int) -> list[Node]: ...
    def update_node(self, node: Node) -> None: ...
    def update_fields(
        self,
        node_id: str,
        content: str | None = None,
        confidence: float | None = None,
        tier: Tier | None = None,
        subtype: str | None = None,
        metadata: dict[str, Any] | None = None,
        last_accessed: str | None = None,
        access_count: int | None = None,
    ) -> bool: ...
    def delete_node(self, node_id: str) -> bool: ...
    def promote(self, node_ids: list[str], reason: str) -> list[str]: ...
    def decay(self, factor: float, min_confidence: float) -> list[str]: ...
    def stats(self) -> MemoryStats: ...
    def add_edge(self, edge: HyperEdge) -> str: ...
    def get_edges_for_node(self, node_id: str) -> list[HyperEdge]: ...
    def delete_edge(self, edge_id: str) -> bool: ...

class MemoryStats:
    total_nodes: int
    total_edges: int
    nodes_by_tier: dict[str, int]
    nodes_by_type: dict[str, int]
