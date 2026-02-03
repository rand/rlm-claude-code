"""
Tests for memory backend abstraction (SPEC-12.20-12.25).

Tests cover:
- MemoryBackend protocol
- SQLiteBackend implementation
- InMemoryBackend implementation
- Backend configuration
- Same tests pass for all backends
- Migration tooling
"""

from datetime import datetime

import pytest

from src.memory_backend import (
    EdgeType,
    InMemoryBackend,
    MemoryBackend,
    MemoryBackendConfig,
    MemoryBackendType,
    MemoryMigrator,
    NodeType,
    SQLiteBackend,
    create_backend,
)


# Parametrize tests to run against all backend implementations
@pytest.fixture(params=["sqlite", "inmemory"])
def backend(request, tmp_path) -> MemoryBackend:
    """Create a backend for testing."""
    if request.param == "sqlite":
        return SQLiteBackend(db_path=tmp_path / "test.db")
    else:
        return InMemoryBackend()


class TestMemoryBackendProtocol:
    """Tests for MemoryBackend protocol (SPEC-12.20, SPEC-12.21)."""

    def test_protocol_has_create_node(self):
        """SPEC-12.21: create_node(...) -> str."""
        backend = InMemoryBackend()
        assert hasattr(backend, "create_node")

    def test_protocol_has_get_node(self):
        """SPEC-12.21: get_node(node_id: str) -> Node | None."""
        backend = InMemoryBackend()
        assert hasattr(backend, "get_node")

    def test_protocol_has_update_node(self):
        """SPEC-12.21: update_node(node_id: str, ...) -> None."""
        backend = InMemoryBackend()
        assert hasattr(backend, "update_node")

    def test_protocol_has_delete_node(self):
        """SPEC-12.21: delete_node(node_id: str) -> bool."""
        backend = InMemoryBackend()
        assert hasattr(backend, "delete_node")

    def test_protocol_has_search(self):
        """SPEC-12.21: search(query: str, ...) -> list[SearchResult]."""
        backend = InMemoryBackend()
        assert hasattr(backend, "search")

    def test_protocol_has_create_edge(self):
        """SPEC-12.21: create_edge(...) -> str."""
        backend = InMemoryBackend()
        assert hasattr(backend, "create_edge")

    def test_protocol_has_get_edges(self):
        """SPEC-12.21: get_edges(node_id: str, ...) -> list[Edge]."""
        backend = InMemoryBackend()
        assert hasattr(backend, "get_edges")


class TestNodeOperations:
    """Tests for node CRUD operations (SPEC-12.21)."""

    def test_create_node_returns_id(self, backend):
        """create_node returns node ID."""
        node_id = backend.create_node(
            content="Test content",
            node_type=NodeType.FACT,
            metadata={"key": "value"},
        )
        assert node_id is not None
        assert isinstance(node_id, str)

    def test_get_node_returns_node(self, backend):
        """get_node returns created node."""
        node_id = backend.create_node(
            content="Test content",
            node_type=NodeType.FACT,
            metadata={},
        )
        node = backend.get_node(node_id)
        assert node is not None
        assert node.content == "Test content"
        assert node.node_type == NodeType.FACT

    def test_get_node_returns_none_for_missing(self, backend):
        """get_node returns None for non-existent node."""
        node = backend.get_node("non-existent-id")
        assert node is None

    def test_update_node_changes_content(self, backend):
        """update_node changes node content."""
        node_id = backend.create_node(
            content="Original",
            node_type=NodeType.FACT,
            metadata={},
        )
        backend.update_node(node_id, content="Updated")
        node = backend.get_node(node_id)
        assert node.content == "Updated"

    def test_update_node_changes_metadata(self, backend):
        """update_node changes node metadata."""
        node_id = backend.create_node(
            content="Content",
            node_type=NodeType.FACT,
            metadata={"old": "value"},
        )
        backend.update_node(node_id, metadata={"new": "value"})
        node = backend.get_node(node_id)
        assert node.metadata == {"new": "value"}

    def test_delete_node_removes_node(self, backend):
        """delete_node removes node."""
        node_id = backend.create_node(
            content="To delete",
            node_type=NodeType.FACT,
            metadata={},
        )
        result = backend.delete_node(node_id)
        assert result is True
        assert backend.get_node(node_id) is None

    def test_delete_node_returns_false_for_missing(self, backend):
        """delete_node returns False for non-existent node."""
        result = backend.delete_node("non-existent-id")
        assert result is False

    def test_node_has_created_at(self, backend):
        """Node has created_at timestamp."""
        node_id = backend.create_node(
            content="Content",
            node_type=NodeType.FACT,
            metadata={},
        )
        node = backend.get_node(node_id)
        assert node.created_at is not None
        assert isinstance(node.created_at, datetime)

    def test_node_has_updated_at(self, backend):
        """Node has updated_at timestamp."""
        node_id = backend.create_node(
            content="Content",
            node_type=NodeType.FACT,
            metadata={},
        )
        node = backend.get_node(node_id)
        assert node.updated_at is not None


class TestSearchOperations:
    """Tests for search operations (SPEC-12.21)."""

    def test_search_finds_matching_content(self, backend):
        """search finds nodes with matching content."""
        backend.create_node(
            content="Python programming is fun",
            node_type=NodeType.FACT,
            metadata={},
        )
        backend.create_node(
            content="Java programming is verbose",
            node_type=NodeType.FACT,
            metadata={},
        )

        results = backend.search("Python")
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_search_returns_empty_for_no_match(self, backend):
        """search returns empty list for no matches."""
        backend.create_node(
            content="Hello world",
            node_type=NodeType.FACT,
            metadata={},
        )

        results = backend.search("nonexistent")
        assert len(results) == 0

    def test_search_respects_limit(self, backend):
        """search respects limit parameter."""
        for i in range(10):
            backend.create_node(
                content=f"Test content {i}",
                node_type=NodeType.FACT,
                metadata={},
            )

        results = backend.search("Test", limit=3)
        assert len(results) <= 3

    def test_search_can_filter_by_type(self, backend):
        """search can filter by node type."""
        backend.create_node(
            content="Fact content",
            node_type=NodeType.FACT,
            metadata={},
        )
        backend.create_node(
            content="Experience content",
            node_type=NodeType.EXPERIENCE,
            metadata={},
        )

        results = backend.search("content", node_type=NodeType.FACT)
        assert all(r.node_type == NodeType.FACT for r in results)

    def test_search_result_has_score(self, backend):
        """SearchResult has relevance score."""
        backend.create_node(
            content="Searchable content",
            node_type=NodeType.FACT,
            metadata={},
        )

        results = backend.search("Searchable")
        if results:
            assert results[0].score is not None


class TestEdgeOperations:
    """Tests for edge operations (SPEC-12.21)."""

    def test_create_edge_returns_id(self, backend):
        """create_edge returns edge ID."""
        node1_id = backend.create_node(
            content="Node 1",
            node_type=NodeType.FACT,
            metadata={},
        )
        node2_id = backend.create_node(
            content="Node 2",
            node_type=NodeType.FACT,
            metadata={},
        )

        edge_id = backend.create_edge(
            from_node=node1_id,
            to_node=node2_id,
            edge_type=EdgeType.RELATED,
        )
        assert edge_id is not None
        assert isinstance(edge_id, str)

    def test_get_edges_returns_edges(self, backend):
        """get_edges returns edges for node."""
        node1_id = backend.create_node(
            content="Node 1",
            node_type=NodeType.FACT,
            metadata={},
        )
        node2_id = backend.create_node(
            content="Node 2",
            node_type=NodeType.FACT,
            metadata={},
        )

        backend.create_edge(
            from_node=node1_id,
            to_node=node2_id,
            edge_type=EdgeType.RELATED,
        )

        edges = backend.get_edges(node1_id)
        assert len(edges) >= 1

    def test_get_edges_empty_for_no_edges(self, backend):
        """get_edges returns empty list for node without edges."""
        node_id = backend.create_node(
            content="Isolated node",
            node_type=NodeType.FACT,
            metadata={},
        )

        edges = backend.get_edges(node_id)
        assert len(edges) == 0

    def test_edge_has_weight(self, backend):
        """Edge can have weight."""
        node1_id = backend.create_node(
            content="Node 1",
            node_type=NodeType.FACT,
            metadata={},
        )
        node2_id = backend.create_node(
            content="Node 2",
            node_type=NodeType.FACT,
            metadata={},
        )

        backend.create_edge(
            from_node=node1_id,
            to_node=node2_id,
            edge_type=EdgeType.RELATED,
            weight=0.8,
        )

        edges = backend.get_edges(node1_id)
        assert edges[0].weight == 0.8

    def test_get_edges_can_filter_by_type(self, backend):
        """get_edges can filter by edge type."""
        node1_id = backend.create_node(
            content="Node 1",
            node_type=NodeType.FACT,
            metadata={},
        )
        node2_id = backend.create_node(
            content="Node 2",
            node_type=NodeType.FACT,
            metadata={},
        )
        node3_id = backend.create_node(
            content="Node 3",
            node_type=NodeType.FACT,
            metadata={},
        )

        backend.create_edge(
            from_node=node1_id,
            to_node=node2_id,
            edge_type=EdgeType.RELATED,
        )
        backend.create_edge(
            from_node=node1_id,
            to_node=node3_id,
            edge_type=EdgeType.DERIVED_FROM,
        )

        edges = backend.get_edges(node1_id, edge_type=EdgeType.RELATED)
        assert all(e.edge_type == EdgeType.RELATED for e in edges)


class TestSQLiteBackend:
    """Tests for SQLiteBackend implementation (SPEC-12.22)."""

    def test_sqlite_creates_database(self, tmp_path):
        """SQLiteBackend creates database file."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db_path)
        backend.create_node(
            content="Test",
            node_type=NodeType.FACT,
            metadata={},
        )
        assert db_path.exists()

    def test_sqlite_persists_data(self, tmp_path):
        """SQLiteBackend persists data between instances."""
        db_path = tmp_path / "test.db"

        # Create node in first instance
        backend1 = SQLiteBackend(db_path=db_path)
        node_id = backend1.create_node(
            content="Persistent data",
            node_type=NodeType.FACT,
            metadata={},
        )

        # Read in second instance
        backend2 = SQLiteBackend(db_path=db_path)
        node = backend2.get_node(node_id)
        assert node is not None
        assert node.content == "Persistent data"


class TestInMemoryBackend:
    """Tests for InMemoryBackend implementation (SPEC-12.22)."""

    def test_inmemory_does_not_persist(self):
        """InMemoryBackend does not persist between instances."""
        backend1 = InMemoryBackend()
        backend1.create_node(
            content="Non-persistent",
            node_type=NodeType.FACT,
            metadata={},
        )

        backend2 = InMemoryBackend()
        results = backend2.search("Non-persistent")
        assert len(results) == 0


class TestBackendConfiguration:
    """Tests for backend configuration (SPEC-12.23)."""

    def test_config_has_backend_type(self):
        """SPEC-12.23: Config has backend_type."""
        config = MemoryBackendConfig(backend_type=MemoryBackendType.SQLITE)
        assert config.backend_type == MemoryBackendType.SQLITE

    def test_config_has_connection_string(self):
        """Config has connection_string."""
        config = MemoryBackendConfig(
            backend_type=MemoryBackendType.SQLITE,
            connection_string="/path/to/db",
        )
        assert config.connection_string == "/path/to/db"

    def test_create_backend_from_config_sqlite(self, tmp_path):
        """create_backend creates SQLiteBackend from config."""
        config = MemoryBackendConfig(
            backend_type=MemoryBackendType.SQLITE,
            connection_string=str(tmp_path / "config.db"),
        )
        backend = create_backend(config)
        assert isinstance(backend, SQLiteBackend)

    def test_create_backend_from_config_inmemory(self):
        """create_backend creates InMemoryBackend from config."""
        config = MemoryBackendConfig(
            backend_type=MemoryBackendType.INMEMORY,
        )
        backend = create_backend(config)
        assert isinstance(backend, InMemoryBackend)

    def test_default_backend_is_inmemory(self):
        """Default backend is InMemory when no config provided."""
        config = MemoryBackendConfig()
        backend = create_backend(config)
        assert isinstance(backend, InMemoryBackend)


class TestMigration:
    """Tests for migration tooling (SPEC-12.25)."""

    def test_migrator_copies_nodes(self, tmp_path):
        """SPEC-12.25: Migration copies nodes between backends."""
        source = InMemoryBackend()
        source.create_node(
            content="Node to migrate",
            node_type=NodeType.FACT,
            metadata={"key": "value"},
        )

        target = SQLiteBackend(db_path=tmp_path / "target.db")

        migrator = MemoryMigrator(source=source, target=target)
        stats = migrator.migrate()

        assert stats.nodes_migrated >= 1

    def test_migrator_copies_edges(self, tmp_path):
        """Migration copies edges between backends."""
        source = InMemoryBackend()
        node1_id = source.create_node(
            content="Node 1",
            node_type=NodeType.FACT,
            metadata={},
        )
        node2_id = source.create_node(
            content="Node 2",
            node_type=NodeType.FACT,
            metadata={},
        )
        source.create_edge(
            from_node=node1_id,
            to_node=node2_id,
            edge_type=EdgeType.RELATED,
        )

        target = SQLiteBackend(db_path=tmp_path / "target.db")

        migrator = MemoryMigrator(source=source, target=target)
        stats = migrator.migrate()

        assert stats.edges_migrated >= 1

    def test_migrator_preserves_content(self, tmp_path):
        """Migration preserves node content."""
        source = InMemoryBackend()
        source.create_node(
            content="Original content",
            node_type=NodeType.EXPERIENCE,
            metadata={"preserved": True},
        )

        target = SQLiteBackend(db_path=tmp_path / "target.db")

        migrator = MemoryMigrator(source=source, target=target)
        migrator.migrate()

        results = target.search("Original content")
        assert len(results) >= 1
        assert results[0].content == "Original content"

    def test_migrator_has_stats(self, tmp_path):
        """Migration returns statistics."""
        source = InMemoryBackend()
        target = SQLiteBackend(db_path=tmp_path / "target.db")

        migrator = MemoryMigrator(source=source, target=target)
        stats = migrator.migrate()

        assert hasattr(stats, "nodes_migrated")
        assert hasattr(stats, "edges_migrated")
        assert hasattr(stats, "duration_ms")


class TestNodeTypes:
    """Tests for node type enum."""

    def test_fact_type(self):
        """NodeType has FACT."""
        assert NodeType.FACT.value == "fact"

    def test_experience_type(self):
        """NodeType has EXPERIENCE."""
        assert NodeType.EXPERIENCE.value == "experience"

    def test_concept_type(self):
        """NodeType has CONCEPT."""
        assert NodeType.CONCEPT.value == "concept"


class TestEdgeTypes:
    """Tests for edge type enum."""

    def test_related_type(self):
        """EdgeType has RELATED."""
        assert EdgeType.RELATED.value == "related"

    def test_derived_from_type(self):
        """EdgeType has DERIVED_FROM."""
        assert EdgeType.DERIVED_FROM.value == "derived_from"

    def test_supports_type(self):
        """EdgeType has SUPPORTS."""
        assert EdgeType.SUPPORTS.value == "supports"


class TestIntegration:
    """Integration tests for memory backend."""

    def test_complete_workflow(self, backend):
        """Test complete node and edge workflow."""
        # Create nodes
        node1_id = backend.create_node(
            content="First concept",
            node_type=NodeType.CONCEPT,
            metadata={"importance": "high"},
        )
        node2_id = backend.create_node(
            content="Supporting fact",
            node_type=NodeType.FACT,
            metadata={"source": "test"},
        )

        # Create edge
        backend.create_edge(
            from_node=node2_id,
            to_node=node1_id,
            edge_type=EdgeType.SUPPORTS,
            weight=0.9,
        )

        # Search
        results = backend.search("concept")
        assert len(results) >= 1

        # Get edges
        edges = backend.get_edges(node2_id)
        assert len(edges) >= 1

        # Update
        backend.update_node(node1_id, content="Updated concept")
        node = backend.get_node(node1_id)
        assert node.content == "Updated concept"

        # Delete
        backend.delete_node(node2_id)
        assert backend.get_node(node2_id) is None
