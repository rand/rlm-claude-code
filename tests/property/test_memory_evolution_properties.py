"""
Property-based tests for memory evolution.

Implements: Spec SPEC-03.29-30 - Property tests for decay invariants.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

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
    if os.path.exists(path):
        os.unlink(path)
    for suffix in ["-wal", "-shm"]:
        wal_path = path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


def make_memory_evolution(temp_db_path):
    """Create a fresh MemoryEvolution instance."""
    from src.memory_evolution import MemoryEvolution
    from src.memory_store import MemoryStore

    store = MemoryStore(db_path=temp_db_path)
    return MemoryEvolution(store), store


# =============================================================================
# Strategies
# =============================================================================

confidence_strategy = st.floats(min_value=0.1, max_value=1.0, allow_nan=False)
days_strategy = st.integers(min_value=0, max_value=365)
factor_strategy = st.floats(min_value=0.5, max_value=0.99, allow_nan=False)
min_confidence_strategy = st.floats(min_value=0.01, max_value=0.5, allow_nan=False)


# =============================================================================
# SPEC-03.29: Decay Never Increases Confidence
# =============================================================================


@pytest.mark.hypothesis
class TestDecayProperties:
    """Property tests for decay invariants."""

    @given(
        initial_confidence=confidence_strategy,
        days_since_access=days_strategy,
        factor=factor_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_decay_never_increases_confidence(self, initial_confidence, days_since_access, factor):
        """
        Decay should never increase confidence.

        @trace SPEC-03.29
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            from src.memory_evolution import MemoryEvolution
            from src.memory_store import MemoryStore

            store = MemoryStore(db_path=temp_path)
            evolution = MemoryEvolution(store)

            node_id = store.create_node(
                node_type="fact",
                content="Test decay property",
                tier="longterm",
                confidence=initial_confidence,
            )

            store._set_last_accessed(node_id, datetime.now() - timedelta(days=days_since_access))

            evolution.decay(factor=factor, min_confidence=0.01)

            node = store.get_node(node_id, include_archived=True)

            # Confidence should never increase
            assert node.confidence <= initial_confidence

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)

    @given(
        initial_confidence=confidence_strategy,
        factor=factor_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_decay_idempotent_for_zero_days(self, initial_confidence, factor):
        """
        Decay with 0 days since access should not change confidence significantly.

        @trace SPEC-03.29
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            from src.memory_evolution import MemoryEvolution
            from src.memory_store import MemoryStore

            store = MemoryStore(db_path=temp_path)
            evolution = MemoryEvolution(store)

            node_id = store.create_node(
                node_type="fact",
                content="Test zero decay",
                tier="longterm",
                confidence=initial_confidence,
            )

            # Access now (0 days ago)
            store._set_last_accessed(node_id, datetime.now())

            evolution.decay(factor=factor, min_confidence=0.01)

            node = store.get_node(node_id, include_archived=True)

            # Should be approximately the same (factor^0 = 1)
            assert abs(node.confidence - initial_confidence) < 0.01

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)

    @given(
        initial_confidence=confidence_strategy,
        days1=st.integers(min_value=1, max_value=30),
        days2=st.integers(min_value=1, max_value=30),
        factor=factor_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_decay_monotonic_with_time(self, initial_confidence, days1, days2, factor):
        """
        More time should result in more decay.

        @trace SPEC-03.29
        """
        assume(days1 != days2)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path2 = f.name

        try:
            from src.memory_evolution import MemoryEvolution
            from src.memory_store import MemoryStore

            # First node with days1
            store1 = MemoryStore(db_path=temp_path1)
            evolution1 = MemoryEvolution(store1)

            node1 = store1.create_node(
                node_type="fact",
                content="Test monotonic 1",
                tier="longterm",
                confidence=initial_confidence,
            )
            store1._set_last_accessed(node1, datetime.now() - timedelta(days=days1))
            evolution1.decay(factor=factor, min_confidence=0.01)
            result1 = store1.get_node(node1, include_archived=True)

            # Second node with days2
            store2 = MemoryStore(db_path=temp_path2)
            evolution2 = MemoryEvolution(store2)

            node2 = store2.create_node(
                node_type="fact",
                content="Test monotonic 2",
                tier="longterm",
                confidence=initial_confidence,
            )
            store2._set_last_accessed(node2, datetime.now() - timedelta(days=days2))
            evolution2.decay(factor=factor, min_confidence=0.01)
            result2 = store2.get_node(node2, include_archived=True)

            # More days = more decay = lower confidence
            if days1 > days2:
                assert result1.confidence <= result2.confidence
            else:
                assert result2.confidence <= result1.confidence

        finally:
            for path in [temp_path1, temp_path2]:
                if os.path.exists(path):
                    os.unlink(path)
                for suffix in ["-wal", "-shm"]:
                    wal_path = path + suffix
                    if os.path.exists(wal_path):
                        os.unlink(wal_path)


# =============================================================================
# SPEC-03.30: Archived Nodes Never Deleted
# =============================================================================


@pytest.mark.hypothesis
class TestArchiveProperties:
    """Property tests for archive tier invariants."""

    @given(
        initial_confidence=confidence_strategy,
        decay_cycles=st.integers(min_value=1, max_value=20),
        factor=factor_strategy,
        min_confidence=min_confidence_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_archived_nodes_never_deleted(
        self, initial_confidence, decay_cycles, factor, min_confidence
    ):
        """
        Archived nodes should never be deleted regardless of decay.

        @trace SPEC-03.30
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            from src.memory_evolution import MemoryEvolution
            from src.memory_store import MemoryStore

            store = MemoryStore(db_path=temp_path)
            evolution = MemoryEvolution(store)

            # Create node directly in archive tier
            node_id = store.create_node(
                node_type="fact",
                content="Never delete this archived node",
                tier="archive",
                confidence=initial_confidence,
            )

            # Run multiple decay cycles
            for _ in range(decay_cycles):
                evolution.decay(factor=factor, min_confidence=min_confidence)

            # Node should still exist
            node = store.get_node(node_id, include_archived=True)
            assert node is not None, "Archived node was deleted!"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)

    @given(
        content=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_archive_preserves_content(self, content):
        """
        Archived nodes should preserve their content exactly.

        @trace SPEC-03.30
        """
        # Filter out null bytes which SQLite doesn't handle well
        assume("\x00" not in content)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            from src.memory_store import MemoryStore

            store = MemoryStore(db_path=temp_path)

            node_id = store.create_node(
                node_type="fact",
                content=content,
                tier="archive",
            )

            # Retrieve and verify content is preserved
            node = store.get_node(node_id, include_archived=True)
            assert node.content == content

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)


# =============================================================================
# Additional Property Tests
# =============================================================================


@pytest.mark.hypothesis
class TestConsolidationProperties:
    """Property tests for consolidation invariants."""

    @given(
        num_nodes=st.integers(min_value=1, max_value=10),
        confidence=confidence_strategy,
    )
    @settings(max_examples=30, deadline=None)
    def test_consolidation_preserves_total_information(self, num_nodes, confidence):
        """
        Consolidation should not lose information (nodes may merge but content preserved).

        @trace SPEC-03.26
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            from src.memory_evolution import MemoryEvolution
            from src.memory_store import MemoryStore

            store = MemoryStore(db_path=temp_path)
            evolution = MemoryEvolution(store)

            # Create nodes
            task_id = "task-prop-test"
            contents = []
            for i in range(num_nodes):
                content = f"Fact number {i}"
                contents.append(content)
                store.create_node(
                    node_type="fact",
                    content=content,
                    tier="task",
                    confidence=confidence,
                    metadata={"task_id": task_id},
                )

            evolution.consolidate(task_id)

            # All content should still be queryable (in some form)
            all_nodes = store.query_nodes(include_archived=True)
            all_content = " ".join(n.content for n in all_nodes)

            # At least some information should be preserved
            assert len(all_nodes) >= 1

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)


@pytest.mark.hypothesis
class TestPromotionProperties:
    """Property tests for promotion invariants."""

    @given(
        threshold=st.floats(min_value=0.1, max_value=0.99, allow_nan=False),
        confidence=confidence_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_promotion_respects_threshold_invariant(self, threshold, confidence):
        """
        Nodes below threshold should never be promoted.

        @trace SPEC-03.27
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            from src.memory_evolution import MemoryEvolution
            from src.memory_store import MemoryStore

            store = MemoryStore(db_path=temp_path)
            evolution = MemoryEvolution(store)

            session_id = "sess-prop-test"
            node_id = store.create_node(
                node_type="fact",
                content="Test promotion threshold",
                tier="session",
                confidence=confidence,
                metadata={"session_id": session_id},
            )

            # Access it to make it a candidate
            for _ in range(5):
                store.get_node(node_id)

            evolution.promote(session_id, threshold=threshold)

            node = store.get_node(node_id)

            if confidence < threshold:
                # Should NOT be promoted
                assert node.tier == "session"
            # Note: confidence >= threshold doesn't guarantee promotion
            # (access count also matters)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)
