"""Integration tests for parallel session isolation."""
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from src.memory_store import MemoryStore


class TestParallelSessions:
    """Test isolation when multiple sessions run concurrently."""

    @pytest.fixture
    def shared_store(self, tmp_path):
        """Create a shared memory store (simulates single DB)."""
        db_path = tmp_path / "shared_memory.db"
        return MemoryStore(db_path=str(db_path))

    def test_concurrent_writes_isolated(self, shared_store):
        """Concurrent sessions writing should not cross-contaminate."""
        errors = []

        def session_worker(session_id: str, topic: str, store: MemoryStore):
            try:
                # Each session writes 20 facts about its topic
                for i in range(20):
                    store.create_node(
                        node_type="fact",
                        content=f"{topic} fact {i}",
                        tier="session",
                        metadata={"session_id": session_id}
                    )
                    time.sleep(0.01)  # Simulate real work

                # Verify isolation - should only see own facts
                results = store.query_nodes(
                    tier="session",
                    session_id=session_id,
                    limit=50
                )

                for node in results:
                    if topic not in node.content:
                        errors.append(
                            f"Session {session_id} saw foreign content: {node.content}"
                        )
            except Exception as e:
                errors.append(str(e))

        # Run 3 parallel sessions
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(session_worker, "session-auth", "auth", shared_store),
                executor.submit(session_worker, "session-billing", "billing", shared_store),
                executor.submit(session_worker, "session-users", "users", shared_store),
            ]
            # Wait for all to complete
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Isolation failures: {errors}"

    def test_longterm_visible_to_all_concurrent_sessions(self, shared_store):
        """Longterm facts should be visible across concurrent sessions."""
        # Pre-populate longterm knowledge
        shared_store.create_node(
            node_type="fact",
            content="Database runs on port 5432",
            tier="longterm",
            metadata={"session_id": "setup"}
        )

        visible_count = {"count": 0}
        lock = threading.Lock()

        def check_longterm(session_id: str, store: MemoryStore):
            results = store.query_nodes(tier="longterm", session_id=session_id)
            if any("5432" in n.content for n in results):
                with lock:
                    visible_count["count"] += 1

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(check_longterm, "session-a", shared_store),
                executor.submit(check_longterm, "session-b", shared_store),
                executor.submit(check_longterm, "session-c", shared_store),
            ]
            for future in futures:
                future.result()

        # All 3 sessions should see the longterm fact
        assert visible_count["count"] == 3

    def test_rapid_concurrent_access(self, shared_store):
        """Rapid concurrent access should not cause race conditions."""
        errors = []
        write_counts = {"session-a": 0, "session-b": 0}

        def rapid_writer(session_id: str, store: MemoryStore, count: int):
            try:
                for i in range(count):
                    store.create_node(
                        node_type="fact",
                        content=f"Rapid fact {i} from {session_id}",
                        tier="task",
                        metadata={"session_id": session_id}
                    )
                write_counts[session_id] = count
            except Exception as e:
                errors.append(f"{session_id}: {e}")

        # Two sessions writing rapidly
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(rapid_writer, "session-a", shared_store, 50),
                executor.submit(rapid_writer, "session-b", shared_store, 50),
            ]
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Race condition errors: {errors}"

        # Verify each session sees correct count
        results_a = shared_store.query_nodes(tier="task", session_id="session-a", limit=100)
        results_b = shared_store.query_nodes(tier="task", session_id="session-b", limit=100)

        assert len(results_a) == 50, f"Session A expected 50, got {len(results_a)}"
        assert len(results_b) == 50, f"Session B expected 50, got {len(results_b)}"

    def test_mixed_tier_concurrent_access(self, shared_store):
        """Different tiers accessed concurrently should work correctly."""
        errors = []

        def task_tier_worker(store: MemoryStore):
            try:
                for i in range(10):
                    store.create_node(
                        node_type="fact",
                        content=f"Task fact {i}",
                        tier="task",
                        metadata={"session_id": "task-session"}
                    )
                results = store.query_nodes(tier="task", session_id="task-session")
                assert len(results) == 10
            except Exception as e:
                errors.append(f"task: {e}")

        def longterm_worker(store: MemoryStore):
            try:
                for i in range(10):
                    store.create_node(
                        node_type="fact",
                        content=f"Longterm fact {i}",
                        tier="longterm",
                        metadata={"session_id": "longterm-session"}
                    )
                # Longterm should be visible regardless of session_id
                results = store.query_nodes(tier="longterm", session_id="any-session")
                assert len(results) == 10
            except Exception as e:
                errors.append(f"longterm: {e}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(task_tier_worker, shared_store),
                executor.submit(longterm_worker, shared_store),
            ]
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Mixed tier errors: {errors}"


class TestCrossContaminationPrevention:
    """Focused tests on preventing cross-contamination between sessions."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = tmp_path / "test_memory.db"
        return MemoryStore(db_path=str(db_path))

    def test_no_leakage_between_sessions(self, store):
        """Facts from one session must never appear in another session's query."""
        # Create facts for different sessions
        sessions = ["session-1", "session-2", "session-3"]
        for session_id in sessions:
            for i in range(5):
                store.create_node(
                    node_type="fact",
                    content=f"Secret fact {i} for {session_id}",
                    tier="session",
                    metadata={"session_id": session_id}
                )

        # Query each session and verify no cross-contamination
        for session_id in sessions:
            results = store.query_nodes(tier="session", session_id=session_id)
            for node in results:
                assert session_id in node.content, (
                    f"Session {session_id} received fact from another session: {node.content}"
                )
                # Also verify metadata matches
                assert node.metadata.get("session_id") == session_id

    def test_session_deletion_does_not_affect_other_sessions(self, store):
        """Deleting one session's data should not affect others."""
        # Create facts for two sessions
        store.create_node(
            node_type="fact",
            content="Session A fact",
            tier="session",
            metadata={"session_id": "session-a"}
        )
        node_b = store.create_node(
            node_type="fact",
            content="Session B fact",
            tier="session",
            metadata={"session_id": "session-b"}
        )

        # Delete session A's fact by archiving it
        results_a = store.query_nodes(tier="session", session_id="session-a")
        for node in results_a:
            store.delete_node(node.id)

        # Session B should be unaffected
        results_b = store.query_nodes(tier="session", session_id="session-b")
        assert len(results_b) == 1
        assert results_b[0].content == "Session B fact"
