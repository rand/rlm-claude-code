"""Tests for session isolation in memory store (SPEC-02 extension)."""
import pytest
from src.memory_store import MemoryStore


class TestSessionIsolation:
    """Tests for session-scoped memory isolation."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a memory store with temp database."""
        db_path = tmp_path / "test_memory.db"
        return MemoryStore(db_path=str(db_path))

    def test_task_tier_isolated_by_session(self, store):
        """Task-tier nodes should only be visible to their session."""
        # Session A stores a fact
        node_a = store.create_node(
            node_type="fact",
            content="Session A working on auth",
            tier="task",
            metadata={"session_id": "session-a"}
        )

        # Session B stores a different fact
        node_b = store.create_node(
            node_type="fact",
            content="Session B working on billing",
            tier="task",
            metadata={"session_id": "session-b"}
        )

        # Query from Session A perspective
        results_a = store.query_nodes(
            tier="task",
            session_id="session-a"
        )

        # Should only see Session A's node
        assert len(results_a) == 1
        assert results_a[0].content == "Session A working on auth"

    def test_session_tier_isolated_by_session(self, store):
        """Session-tier nodes should only be visible to their session."""
        store.create_node(
            node_type="fact",
            content="Auth uses JWT tokens",
            tier="session",
            metadata={"session_id": "session-a"}
        )

        store.create_node(
            node_type="fact",
            content="Billing uses Stripe",
            tier="session",
            metadata={"session_id": "session-b"}
        )

        results_a = store.query_nodes(tier="session", session_id="session-a")
        results_b = store.query_nodes(tier="session", session_id="session-b")

        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0].content == "Auth uses JWT tokens"
        assert results_b[0].content == "Billing uses Stripe"

    def test_longterm_tier_shared_across_sessions(self, store):
        """Longterm-tier nodes should be visible to all sessions."""
        store.create_node(
            node_type="fact",
            content="PostgreSQL runs on port 5433",
            tier="longterm",
            metadata={"session_id": "session-a"}  # Origin session
        )

        # Both sessions should see longterm facts
        results_a = store.query_nodes(tier="longterm", session_id="session-a")
        results_b = store.query_nodes(tier="longterm", session_id="session-b")

        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_archive_tier_shared_across_sessions(self, store):
        """Archive-tier nodes should be visible to all sessions."""
        store.create_node(
            node_type="fact",
            content="Old API endpoint deprecated",
            tier="archive",
            metadata={"session_id": "session-a"}
        )

        # Need to include_archived to see archive tier
        results_b = store.query_nodes(
            tier="archive",
            session_id="session-b",
            include_archived=True
        )
        assert len(results_b) == 1

    def test_no_session_id_returns_all(self, store):
        """Query without session_id should return all nodes (backward compatibility)."""
        store.create_node(
            node_type="fact",
            content="Fact from session A",
            tier="task",
            metadata={"session_id": "session-a"}
        )
        store.create_node(
            node_type="fact",
            content="Fact from session B",
            tier="task",
            metadata={"session_id": "session-b"}
        )

        # No session_id filter should return all
        results = store.query_nodes(tier="task")
        assert len(results) == 2

    def test_session_id_with_no_matches(self, store):
        """Query with non-existent session_id should return empty."""
        store.create_node(
            node_type="fact",
            content="Some fact",
            tier="task",
            metadata={"session_id": "session-a"}
        )

        results = store.query_nodes(tier="task", session_id="session-nonexistent")
        assert len(results) == 0

    def test_nodes_without_session_id_visible_to_all(self, store):
        """Nodes without session_id in metadata should not match session filters."""
        # Node without session_id
        store.create_node(
            node_type="fact",
            content="Legacy fact without session",
            tier="task",
            metadata={}  # No session_id
        )

        # Node with session_id
        store.create_node(
            node_type="fact",
            content="New fact with session",
            tier="task",
            metadata={"session_id": "session-a"}
        )

        # Querying with session_id should only return nodes that match
        results = store.query_nodes(tier="task", session_id="session-a")
        assert len(results) == 1
        assert results[0].content == "New fact with session"

        # Querying without session_id should return all
        all_results = store.query_nodes(tier="task")
        assert len(all_results) == 2


class TestContextVolumeLimits:
    """Prevent context flooding with too many memories."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = tmp_path / "test_memory.db"
        return MemoryStore(db_path=str(db_path))

    def test_query_respects_limit(self, store):
        """Query should respect limit even with many nodes."""
        # Add 50 nodes
        for i in range(50):
            store.create_node(
                node_type="fact",
                content=f"Fact number {i}",
                tier="session",
                metadata={"session_id": "session-a"}
            )

        # Query with limit
        results = store.query_nodes(
            tier="session",
            session_id="session-a",
            limit=10
        )

        assert len(results) == 10

    def test_query_returns_most_relevant_first(self, store):
        """Higher confidence nodes should come first."""
        store.create_node(
            node_type="fact",
            content="Low confidence fact",
            tier="session",
            confidence=0.3,
            metadata={"session_id": "session-a"}
        )
        store.create_node(
            node_type="fact",
            content="High confidence fact",
            tier="session",
            confidence=0.9,
            metadata={"session_id": "session-a"}
        )

        results = store.query_nodes(
            tier="session",
            session_id="session-a",
            limit=1
        )

        assert results[0].content == "High confidence fact"

    def test_default_limit_prevents_flooding(self, store):
        """Default query should not return unlimited results."""
        # Add 200 nodes
        for i in range(200):
            store.create_node(
                node_type="fact",
                content=f"Fact {i}",
                tier="session",
                metadata={"session_id": "session-a"}
            )

        results = store.query_nodes(tier="session", session_id="session-a")

        # Should be limited to default (100)
        assert len(results) <= 100
