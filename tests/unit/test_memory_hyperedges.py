"""
Unit tests for hyperedge operations in memory store.

Implements: Spec SPEC-02 tests for hyperedges and membership.
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)
    for suffix in ["-wal", "-shm"]:
        wal_path = path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
def memory_store(temp_db_path):
    """Create a MemoryStore instance with temporary database."""
    from src.memory_store import MemoryStore

    store = MemoryStore(db_path=temp_db_path)
    return store


@pytest.fixture
def sample_nodes(memory_store):
    """Create sample nodes for testing hyperedges."""
    node1 = memory_store.create_node(node_type="entity", content="Python")
    node2 = memory_store.create_node(node_type="entity", content="Programming Language")
    node3 = memory_store.create_node(node_type="fact", content="Python is interpreted")
    return {"node1": node1, "node2": node2, "node3": node3}


# =============================================================================
# SPEC-02.11-13: Hyperedge Types and Constraints
# =============================================================================


class TestHyperedgeTypes:
    """Tests for hyperedge type support."""

    @pytest.mark.parametrize(
        "edge_type",
        ["relation", "composition", "causation", "context"],
    )
    def test_supports_all_edge_types(self, memory_store, sample_nodes, edge_type):
        """
        System should support all specified hyperedge types.

        @trace SPEC-02.11
        """
        edge_id = memory_store.create_edge(
            edge_type=edge_type,
            label=f"test_{edge_type}",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )

        assert edge_id is not None

        edge = memory_store.get_edge(edge_id)
        assert edge is not None
        assert edge.type == edge_type

    def test_invalid_edge_type_rejected(self, memory_store, sample_nodes):
        """
        Invalid edge types should be rejected.

        @trace SPEC-02.11
        """
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            memory_store.create_edge(
                edge_type="invalid_type",
                label="test",
                members=[
                    {"node_id": sample_nodes["node1"], "role": "subject"},
                ],
            )


class TestHyperedgeFields:
    """Tests for hyperedge required fields."""

    def test_edge_has_required_fields(self, memory_store, sample_nodes):
        """
        Each hyperedge should have id, type, label, weight.

        @trace SPEC-02.12
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="is_a",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )

        edge = memory_store.get_edge(edge_id)

        assert hasattr(edge, "id") and edge.id is not None
        assert hasattr(edge, "type") and edge.type == "relation"
        assert hasattr(edge, "label") and edge.label == "is_a"
        assert hasattr(edge, "weight") and edge.weight is not None

    def test_edge_id_is_uuid(self, memory_store, sample_nodes):
        """
        Edge IDs should be UUIDs.

        @trace SPEC-02.12
        """
        import uuid

        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
        )

        parsed = uuid.UUID(edge_id)
        assert str(parsed) == edge_id


class TestHyperedgeWeight:
    """Tests for hyperedge weight constraints."""

    def test_default_weight_is_one(self, memory_store, sample_nodes):
        """
        Default edge weight should be 1.0.

        @trace SPEC-02.12
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
        )

        edge = memory_store.get_edge(edge_id)
        assert edge.weight == 1.0

    def test_weight_can_be_set(self, memory_store, sample_nodes):
        """
        Edge weight should be settable.

        @trace SPEC-02.12
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
            weight=2.5,
        )

        edge = memory_store.get_edge(edge_id)
        assert edge.weight == 2.5

    def test_weight_zero_allowed(self, memory_store, sample_nodes):
        """
        Edge weight of 0.0 should be allowed.

        @trace SPEC-02.13
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
            weight=0.0,
        )

        edge = memory_store.get_edge(edge_id)
        assert edge.weight == 0.0

    def test_weight_large_values_allowed(self, memory_store, sample_nodes):
        """
        Large positive weight values should be allowed.

        @trace SPEC-02.13
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
            weight=1000.0,
        )

        edge = memory_store.get_edge(edge_id)
        assert edge.weight == 1000.0

    def test_weight_negative_rejected(self, memory_store, sample_nodes):
        """
        Negative edge weights should be rejected.

        @trace SPEC-02.13
        """
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            memory_store.create_edge(
                edge_type="relation",
                label="test",
                members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
                weight=-1.0,
            )


# =============================================================================
# SPEC-02.14-16: Membership
# =============================================================================


class TestMembership:
    """Tests for hyperedge membership."""

    def test_many_to_many_relationship(self, memory_store, sample_nodes):
        """
        System should support many-to-many relationships.

        @trace SPEC-02.14
        """
        # One node in multiple edges
        edge1 = memory_store.create_edge(
            edge_type="relation",
            label="edge1",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )
        edge2 = memory_store.create_edge(
            edge_type="relation",
            label="edge2",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node3"], "role": "object"},
            ],
        )

        # Multiple nodes in one edge
        edge3 = memory_store.create_edge(
            edge_type="context",
            label="context",
            members=[
                {"node_id": sample_nodes["node1"], "role": "participant"},
                {"node_id": sample_nodes["node2"], "role": "participant"},
                {"node_id": sample_nodes["node3"], "role": "participant"},
            ],
        )

        assert edge1 is not None
        assert edge2 is not None
        assert edge3 is not None

    def test_membership_has_required_fields(self, memory_store, sample_nodes):
        """
        Membership should specify hyperedge_id, node_id, role, position.

        @trace SPEC-02.15
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject", "position": 0},
                {"node_id": sample_nodes["node2"], "role": "object", "position": 1},
            ],
        )

        members = memory_store.get_edge_members(edge_id)

        assert len(members) == 2
        for member in members:
            assert "node_id" in member
            assert "role" in member
            assert "position" in member

    @pytest.mark.parametrize(
        "role",
        ["subject", "object", "context", "participant", "cause", "effect"],
    )
    def test_supports_all_membership_roles(self, memory_store, sample_nodes, role):
        """
        System should support all specified membership roles.

        @trace SPEC-02.16
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[{"node_id": sample_nodes["node1"], "role": role}],
        )

        members = memory_store.get_edge_members(edge_id)
        assert len(members) == 1
        assert members[0]["role"] == role

    def test_position_ordering(self, memory_store, sample_nodes):
        """
        Members should be ordered by position.

        @trace SPEC-02.15
        """
        edge_id = memory_store.create_edge(
            edge_type="composition",
            label="ordered",
            members=[
                {"node_id": sample_nodes["node3"], "role": "participant", "position": 2},
                {"node_id": sample_nodes["node1"], "role": "participant", "position": 0},
                {"node_id": sample_nodes["node2"], "role": "participant", "position": 1},
            ],
        )

        members = memory_store.get_edge_members(edge_id)

        assert members[0]["node_id"] == sample_nodes["node1"]
        assert members[1]["node_id"] == sample_nodes["node2"]
        assert members[2]["node_id"] == sample_nodes["node3"]


# =============================================================================
# SPEC-02.25-26: Edge CRUD Operations
# =============================================================================


class TestCreateEdge:
    """Tests for create_edge operation."""

    def test_create_edge_returns_id(self, memory_store, sample_nodes):
        """
        create_edge should return the edge ID.

        @trace SPEC-02.25
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="is_a",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )

        assert edge_id is not None
        assert isinstance(edge_id, str)
        assert len(edge_id) == 36  # UUID length

    def test_create_edge_with_single_member(self, memory_store, sample_nodes):
        """
        create_edge should allow single-member edges.

        @trace SPEC-02.25
        """
        edge_id = memory_store.create_edge(
            edge_type="context",
            label="standalone",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
        )

        assert edge_id is not None

    def test_create_edge_with_many_members(self, memory_store):
        """
        create_edge should support many members (hyperedge).

        @trace SPEC-02.25
        """
        # Create many nodes
        nodes = []
        for i in range(10):
            node_id = memory_store.create_node(node_type="entity", content=f"Node {i}")
            nodes.append(node_id)

        # Create edge with all nodes
        edge_id = memory_store.create_edge(
            edge_type="context",
            label="large_hyperedge",
            members=[
                {"node_id": node_id, "role": "participant", "position": i}
                for i, node_id in enumerate(nodes)
            ],
        )

        members = memory_store.get_edge_members(edge_id)
        assert len(members) == 10

    def test_create_edge_with_empty_label(self, memory_store, sample_nodes):
        """
        create_edge should allow empty/None label.

        @trace SPEC-02.25
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label=None,
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
        )

        edge = memory_store.get_edge(edge_id)
        assert edge.label is None or edge.label == ""


class TestGetRelatedNodes:
    """Tests for get_related_nodes operation."""

    def test_get_related_nodes_basic(self, memory_store, sample_nodes):
        """
        get_related_nodes should return nodes connected by edges.

        @trace SPEC-02.26
        """
        # Create edge: node1 -> node2
        memory_store.create_edge(
            edge_type="relation",
            label="is_a",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )

        related = memory_store.get_related_nodes(sample_nodes["node1"])

        assert len(related) >= 1
        node_ids = [n.id for n in related]
        assert sample_nodes["node2"] in node_ids

    def test_get_related_nodes_by_edge_type(self, memory_store, sample_nodes):
        """
        get_related_nodes should filter by edge type.

        @trace SPEC-02.26
        """
        # Create relation edge
        memory_store.create_edge(
            edge_type="relation",
            label="related",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )
        # Create causation edge
        memory_store.create_edge(
            edge_type="causation",
            label="causes",
            members=[
                {"node_id": sample_nodes["node1"], "role": "cause"},
                {"node_id": sample_nodes["node3"], "role": "effect"},
            ],
        )

        # Filter by relation type
        related = memory_store.get_related_nodes(sample_nodes["node1"], edge_type="relation")
        node_ids = [n.id for n in related]

        assert sample_nodes["node2"] in node_ids
        assert sample_nodes["node3"] not in node_ids

    def test_get_related_nodes_no_relations(self, memory_store, sample_nodes):  # noqa: ARG002
        """
        get_related_nodes should return empty list for unconnected nodes.

        @trace SPEC-02.26
        """
        # Create isolated node
        isolated = memory_store.create_node(node_type="entity", content="Isolated")

        related = memory_store.get_related_nodes(isolated)

        assert len(related) == 0

    def test_get_related_nodes_bidirectional(self, memory_store, sample_nodes):
        """
        get_related_nodes should find relations in both directions.

        @trace SPEC-02.26
        """
        # Create edge: node1 -> node2
        memory_store.create_edge(
            edge_type="relation",
            label="connects",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )

        # Query from node2 should still find node1
        related = memory_store.get_related_nodes(sample_nodes["node2"])
        node_ids = [n.id for n in related]

        assert sample_nodes["node1"] in node_ids


# =============================================================================
# Edge Deletion and Cascade
# =============================================================================


class TestEdgeDeletion:
    """Tests for edge deletion behavior."""

    def test_delete_edge(self, memory_store, sample_nodes):
        """
        Edges should be deletable.

        @trace SPEC-02.25
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )

        result = memory_store.delete_edge(edge_id)
        assert result is True

        edge = memory_store.get_edge(edge_id)
        assert edge is None

    def test_delete_edge_removes_membership(self, memory_store, sample_nodes):
        """
        Deleting edge should remove membership entries.

        @trace SPEC-02.25
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )

        memory_store.delete_edge(edge_id)

        members = memory_store.get_edge_members(edge_id)
        assert len(members) == 0

    def test_delete_node_cascades_to_membership(self, memory_store, sample_nodes):
        """
        Deleting a node should cascade to membership.

        @trace SPEC-02.14
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )

        # Hard delete node1 (not soft delete)
        memory_store.hard_delete_node(sample_nodes["node1"])

        # Edge should still exist
        edge = memory_store.get_edge(edge_id)
        assert edge is not None

        # But membership should only have node2
        members = memory_store.get_edge_members(edge_id)
        node_ids = [m["node_id"] for m in members]
        assert sample_nodes["node1"] not in node_ids
        assert sample_nodes["node2"] in node_ids


# =============================================================================
# Edge Update Operations
# =============================================================================


class TestEdgeUpdate:
    """Tests for edge update operations."""

    def test_update_edge_weight(self, memory_store, sample_nodes):
        """
        Edge weight should be updatable.

        @trace SPEC-02.12
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="test",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
            weight=1.0,
        )

        memory_store.update_edge(edge_id, weight=5.0)

        edge = memory_store.get_edge(edge_id)
        assert edge.weight == 5.0

    def test_update_edge_label(self, memory_store, sample_nodes):
        """
        Edge label should be updatable.

        @trace SPEC-02.12
        """
        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="original",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
        )

        memory_store.update_edge(edge_id, label="updated")

        edge = memory_store.get_edge(edge_id)
        assert edge.label == "updated"


# =============================================================================
# Query Edges
# =============================================================================


class TestQueryEdges:
    """Tests for querying edges."""

    def test_query_edges_by_type(self, memory_store, sample_nodes):
        """
        Should be able to query edges by type.

        @trace SPEC-02.11
        """
        memory_store.create_edge(
            edge_type="relation",
            label="rel1",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
        )
        memory_store.create_edge(
            edge_type="causation",
            label="cause1",
            members=[{"node_id": sample_nodes["node1"], "role": "cause"}],
        )

        relation_edges = memory_store.query_edges(edge_type="relation")
        causation_edges = memory_store.query_edges(edge_type="causation")

        assert len(relation_edges) == 1
        assert len(causation_edges) == 1
        assert relation_edges[0].label == "rel1"
        assert causation_edges[0].label == "cause1"

    def test_query_edges_by_label(self, memory_store, sample_nodes):
        """
        Should be able to query edges by label.

        @trace SPEC-02.12
        """
        memory_store.create_edge(
            edge_type="relation",
            label="is_a",
            members=[{"node_id": sample_nodes["node1"], "role": "subject"}],
        )
        memory_store.create_edge(
            edge_type="relation",
            label="has_property",
            members=[{"node_id": sample_nodes["node2"], "role": "subject"}],
        )

        results = memory_store.query_edges(label="is_a")

        assert len(results) == 1
        assert results[0].label == "is_a"

    def test_get_edges_for_node(self, memory_store, sample_nodes):
        """
        Should be able to get all edges for a node.

        @trace SPEC-02.26
        """
        edge1 = memory_store.create_edge(
            edge_type="relation",
            label="edge1",
            members=[
                {"node_id": sample_nodes["node1"], "role": "subject"},
                {"node_id": sample_nodes["node2"], "role": "object"},
            ],
        )
        edge2 = memory_store.create_edge(
            edge_type="causation",
            label="edge2",
            members=[
                {"node_id": sample_nodes["node1"], "role": "cause"},
                {"node_id": sample_nodes["node3"], "role": "effect"},
            ],
        )

        edges = memory_store.get_edges_for_node(sample_nodes["node1"])

        assert len(edges) == 2
        edge_ids = [e.id for e in edges]
        assert edge1 in edge_ids
        assert edge2 in edge_ids
