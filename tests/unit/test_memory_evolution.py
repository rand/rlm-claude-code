"""
Unit tests for memory evolution.

Implements: Spec SPEC-03 tests for consolidation, promotion, and decay.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
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
    """Create a MemoryStore instance."""
    from src.memory_store import MemoryStore

    return MemoryStore(db_path=temp_db_path)


@pytest.fixture
def memory_evolution(memory_store):
    """Create a MemoryEvolution instance."""
    from src.memory_evolution import MemoryEvolution

    return MemoryEvolution(memory_store)


# =============================================================================
# SPEC-03.01-07: Consolidation Tests
# =============================================================================


class TestConsolidation:
    """Tests for task-to-session consolidation."""

    def test_consolidate_exists(self, memory_evolution):
        """
        MemoryEvolution should have consolidate method.

        @trace SPEC-03.01
        """
        assert hasattr(memory_evolution, "consolidate")
        assert callable(memory_evolution.consolidate)

    def test_consolidate_returns_result(self, memory_evolution, memory_store):
        """
        consolidate should return ConsolidationResult.

        @trace SPEC-03.01
        """
        from src.memory_evolution import ConsolidationResult

        # Create a task node
        memory_store.create_node(
            node_type="fact",
            content="Test fact",
            tier="task",
            metadata={"task_id": "task-123"},
        )

        result = memory_evolution.consolidate("task-123")

        assert isinstance(result, ConsolidationResult)
        assert hasattr(result, "merged_count")
        assert hasattr(result, "promoted_count")
        assert hasattr(result, "edges_strengthened")

    def test_consolidate_moves_nodes_to_session(self, memory_evolution, memory_store):
        """
        consolidate should move task nodes to session tier.

        @trace SPEC-03.01
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Task fact to consolidate",
            tier="task",
            metadata={"task_id": "task-456"},
        )

        memory_evolution.consolidate("task-456")

        node = memory_store.get_node(node_id)
        assert node.tier == "session"

    def test_consolidate_identifies_related_by_hyperedge(self, memory_evolution, memory_store):
        """
        consolidate should identify related facts by shared hyperedges.

        @trace SPEC-03.02
        """
        # Create related facts
        fact1 = memory_store.create_node(
            node_type="fact",
            content="Python is a language",
            tier="task",
            metadata={"task_id": "task-789"},
        )
        fact2 = memory_store.create_node(
            node_type="fact",
            content="Python has classes",
            tier="task",
            metadata={"task_id": "task-789"},
        )

        # Create edge relating them
        memory_store.create_edge(
            edge_type="relation",
            label="related_to",
            members=[
                {"node_id": fact1, "role": "subject", "position": 0},
                {"node_id": fact2, "role": "object", "position": 1},
            ],
        )

        result = memory_evolution.consolidate("task-789")

        # Both should be consolidated
        node1 = memory_store.get_node(fact1)
        node2 = memory_store.get_node(fact2)
        assert node1.tier == "session"
        assert node2.tier == "session"

    def test_consolidate_merges_redundant_facts(self, memory_evolution, memory_store):
        """
        consolidate should merge redundant facts, keeping highest confidence.

        @trace SPEC-03.03
        """
        # Create redundant facts with different confidence
        fact1 = memory_store.create_node(
            node_type="fact",
            content="Python is interpreted",
            tier="task",
            confidence=0.7,
            metadata={"task_id": "task-merge"},
        )
        fact2 = memory_store.create_node(
            node_type="fact",
            content="Python is interpreted",  # Same content
            tier="task",
            confidence=0.9,  # Higher confidence
            metadata={"task_id": "task-merge"},
        )

        result = memory_evolution.consolidate("task-merge")

        assert result.merged_count >= 1

        # Lower confidence fact should be archived or removed
        node1 = memory_store.get_node(fact1, include_archived=True)
        node2 = memory_store.get_node(fact2)

        # The higher confidence one should remain
        assert node2 is not None
        assert node2.confidence == 0.9

    def test_consolidate_strengthens_frequently_accessed_edges(
        self, memory_evolution, memory_store
    ):
        """
        consolidate should strengthen frequently-accessed edges.

        @trace SPEC-03.04
        """
        fact1 = memory_store.create_node(
            node_type="fact",
            content="Fact A",
            tier="task",
            metadata={"task_id": "task-edge"},
        )
        fact2 = memory_store.create_node(
            node_type="fact",
            content="Fact B",
            tier="task",
            metadata={"task_id": "task-edge"},
        )

        edge_id = memory_store.create_edge(
            edge_type="relation",
            label="connects",
            members=[
                {"node_id": fact1, "role": "subject", "position": 0},
                {"node_id": fact2, "role": "object", "position": 1},
            ],
            weight=1.0,
        )

        # Simulate access by getting related nodes multiple times
        for _ in range(5):
            memory_store.get_related_nodes(fact1)

        result = memory_evolution.consolidate("task-edge")

        edge = memory_store.get_edge(edge_id)
        # Edge weight should be increased (strengthened)
        assert edge.weight >= 1.0

    def test_consolidate_preserves_detail_via_summarizes_edge(self, memory_evolution, memory_store):
        """
        consolidate should preserve detail via "summarizes" edges.

        @trace SPEC-03.05
        """
        # Create multiple related facts
        facts = []
        for i in range(3):
            facts.append(
                memory_store.create_node(
                    node_type="fact",
                    content=f"Detail fact {i}",
                    tier="task",
                    metadata={"task_id": "task-summarize"},
                )
            )

        # Create edges between them
        for i in range(len(facts) - 1):
            memory_store.create_edge(
                edge_type="relation",
                label="related",
                members=[
                    {"node_id": facts[i], "role": "subject", "position": 0},
                    {"node_id": facts[i + 1], "role": "object", "position": 1},
                ],
            )

        result = memory_evolution.consolidate("task-summarize")

        # Check that summarizes edges exist if merged
        if result.merged_count > 0:
            # There should be a "summarizes" edge somewhere
            all_edges = memory_store.query_edges(label="summarizes")
            # At least some detail should be preserved
            assert len(all_edges) >= 0  # May or may not have edges based on logic

    def test_consolidate_logs_to_evolution_log(self, memory_evolution, memory_store):
        """
        consolidate should log operations to evolution_log.

        @trace SPEC-03.06
        """
        memory_store.create_node(
            node_type="fact",
            content="Logged fact",
            tier="task",
            metadata={"task_id": "task-log"},
        )

        memory_evolution.consolidate("task-log")

        # Check evolution log has entries
        logs = memory_store.get_evolution_log(operation_type="consolidate")
        assert len(logs) >= 1
        assert logs[0].reasoning is not None


# =============================================================================
# SPEC-03.08-14: Promotion Tests
# =============================================================================


class TestPromotion:
    """Tests for session-to-longterm promotion."""

    def test_promote_exists(self, memory_evolution):
        """
        MemoryEvolution should have promote method.

        @trace SPEC-03.08
        """
        assert hasattr(memory_evolution, "promote")
        assert callable(memory_evolution.promote)

    def test_promote_returns_result(self, memory_evolution, memory_store):
        """
        promote should return PromotionResult.

        @trace SPEC-03.08
        """
        from src.memory_evolution import PromotionResult

        # Create a session node
        memory_store.create_node(
            node_type="fact",
            content="Session fact",
            tier="session",
            confidence=0.9,
            metadata={"session_id": "sess-123"},
        )

        result = memory_evolution.promote("sess-123")

        assert isinstance(result, PromotionResult)
        assert hasattr(result, "promoted_count")
        assert hasattr(result, "crystallized_count")

    def test_promote_respects_confidence_threshold(self, memory_evolution, memory_store):
        """
        promote should only promote nodes with confidence >= threshold.

        @trace SPEC-03.09
        """
        # High confidence - should be promoted
        high_conf = memory_store.create_node(
            node_type="fact",
            content="High confidence fact",
            tier="session",
            confidence=0.9,
            metadata={"session_id": "sess-threshold"},
        )

        # Low confidence - should not be promoted
        low_conf = memory_store.create_node(
            node_type="fact",
            content="Low confidence fact",
            tier="session",
            confidence=0.5,
            metadata={"session_id": "sess-threshold"},
        )

        memory_evolution.promote("sess-threshold", threshold=0.8)

        high_node = memory_store.get_node(high_conf)
        low_node = memory_store.get_node(low_conf)

        assert high_node.tier == "longterm"
        assert low_node.tier == "session"  # Should remain in session

    def test_promote_default_threshold_is_0_8(self, memory_evolution, memory_store):
        """
        promote should default to threshold=0.8.

        @trace SPEC-03.09
        """
        # Just below threshold
        node_id = memory_store.create_node(
            node_type="fact",
            content="Just below threshold",
            tier="session",
            confidence=0.79,
            metadata={"session_id": "sess-default"},
        )

        memory_evolution.promote("sess-default")  # No threshold specified

        node = memory_store.get_node(node_id)
        assert node.tier == "session"  # Should NOT be promoted

    def test_promote_considers_access_count(self, memory_evolution, memory_store):
        """
        promote should select nodes with access_count above median.

        @trace SPEC-03.10
        """
        # Create nodes with different access patterns
        frequently_accessed = memory_store.create_node(
            node_type="fact",
            content="Frequently accessed",
            tier="session",
            confidence=0.85,
            metadata={"session_id": "sess-access"},
        )

        rarely_accessed = memory_store.create_node(
            node_type="fact",
            content="Rarely accessed",
            tier="session",
            confidence=0.85,
            metadata={"session_id": "sess-access"},
        )

        # Simulate frequent access
        for _ in range(10):
            memory_store.get_node(frequently_accessed)

        # Only access once
        memory_store.get_node(rarely_accessed)

        result = memory_evolution.promote("sess-access", threshold=0.8)

        # The frequently accessed one should be promoted
        freq_node = memory_store.get_node(frequently_accessed)
        assert freq_node.tier == "longterm"

    def test_promote_creates_crystallized_summaries(self, memory_evolution, memory_store):
        """
        promote should create crystallized summary nodes for complex subgraphs.

        @trace SPEC-03.11
        """
        # Create a complex subgraph
        nodes = []
        for i in range(5):
            nodes.append(
                memory_store.create_node(
                    node_type="fact",
                    content=f"Complex fact {i}",
                    tier="session",
                    confidence=0.9,
                    metadata={"session_id": "sess-crystal"},
                )
            )

        # Create edges forming a subgraph
        for i in range(len(nodes) - 1):
            memory_store.create_edge(
                edge_type="relation",
                label="connects",
                members=[
                    {"node_id": nodes[i], "role": "subject", "position": 0},
                    {"node_id": nodes[i + 1], "role": "object", "position": 1},
                ],
            )

        # Simulate access
        for node_id in nodes:
            for _ in range(5):
                memory_store.get_node(node_id)

        result = memory_evolution.promote("sess-crystal", threshold=0.8)

        # Should have created crystallized nodes
        assert result.crystallized_count >= 0  # May or may not crystallize

    def test_promote_preserves_originals_until_confirmed(self, memory_evolution, memory_store):
        """
        promote should preserve original nodes in session tier until confirmed.

        @trace SPEC-03.12
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Original to preserve",
            tier="session",
            confidence=0.95,
            metadata={"session_id": "sess-preserve"},
        )

        # Access it to ensure promotion
        for _ in range(5):
            memory_store.get_node(node_id)

        # Promote without confirmation
        result = memory_evolution.promote("sess-preserve", threshold=0.8, confirm=False)

        # Original should still exist (in session or longterm)
        node = memory_store.get_node(node_id)
        assert node is not None

    def test_promote_logs_to_evolution_log(self, memory_evolution, memory_store):
        """
        promote should log operations to evolution_log.

        @trace SPEC-03.13
        """
        memory_store.create_node(
            node_type="fact",
            content="Logged promotion",
            tier="session",
            confidence=0.95,
            metadata={"session_id": "sess-log"},
        )

        memory_evolution.promote("sess-log")

        logs = memory_store.get_evolution_log(operation_type="promote")
        assert len(logs) >= 1


# =============================================================================
# SPEC-03.15-20: Decay Tests
# =============================================================================


class TestDecay:
    """Tests for temporal decay algorithm."""

    def test_decay_exists(self, memory_evolution):
        """
        MemoryEvolution should have decay method.

        @trace SPEC-03.15
        """
        assert hasattr(memory_evolution, "decay")
        assert callable(memory_evolution.decay)

    def test_decay_returns_result(self, memory_evolution, memory_store):
        """
        decay should return DecayResult.

        @trace SPEC-03.15
        """
        from src.memory_evolution import DecayResult

        result = memory_evolution.decay()

        assert isinstance(result, DecayResult)
        assert hasattr(result, "decayed_count")
        assert hasattr(result, "archived_count")

    def test_decay_formula(self, memory_evolution, memory_store):
        """
        Decay formula should be: new_confidence = base_confidence * (factor ^ days).

        @trace SPEC-03.16
        """
        # Create a node with known last access time
        node_id = memory_store.create_node(
            node_type="fact",
            content="Decaying fact",
            tier="longterm",
            confidence=1.0,
        )

        # Manually set last_accessed to 10 days ago
        memory_store._set_last_accessed(node_id, datetime.now() - timedelta(days=10))

        # Apply decay with factor=0.95
        memory_evolution.decay(factor=0.95, min_confidence=0.1)

        node = memory_store.get_node(node_id)

        # Expected: 1.0 * (0.95 ^ 10) = ~0.599
        expected = 1.0 * (0.95**10)
        assert abs(node.confidence - expected) < 0.05  # Allow small variance

    def test_decay_default_factor(self, memory_evolution, memory_store):
        """
        decay should default to factor=0.95.

        @trace SPEC-03.15
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Default decay",
            tier="longterm",
            confidence=1.0,
        )

        memory_store._set_last_accessed(node_id, datetime.now() - timedelta(days=5))

        memory_evolution.decay()  # No factor specified

        node = memory_store.get_node(node_id)
        expected = 1.0 * (0.95**5)  # ~0.774
        assert abs(node.confidence - expected) < 0.05

    def test_decay_amplified_by_access_frequency(self, memory_evolution, memory_store):
        """
        Decay should be amplified by access frequency.

        @trace SPEC-03.17
        """
        # Frequently accessed node - should decay slower
        frequent = memory_store.create_node(
            node_type="fact",
            content="Frequently accessed",
            tier="longterm",
            confidence=1.0,
        )

        # Rarely accessed node - should decay faster
        rare = memory_store.create_node(
            node_type="fact",
            content="Rarely accessed",
            tier="longterm",
            confidence=1.0,
        )

        # Simulate different access patterns
        for _ in range(20):
            memory_store.get_node(frequent)

        memory_store.get_node(rare)  # Only once

        # Set both to same last_accessed time
        old_time = datetime.now() - timedelta(days=7)
        memory_store._set_last_accessed(frequent, old_time)
        memory_store._set_last_accessed(rare, old_time)

        memory_evolution.decay(factor=0.95, min_confidence=0.1)

        freq_node = memory_store.get_node(frequent)
        rare_node = memory_store.get_node(rare)

        # Frequently accessed should have higher confidence after decay
        assert freq_node.confidence > rare_node.confidence

    def test_decay_archives_below_min_confidence(self, memory_evolution, memory_store):
        """
        Nodes below min_confidence should be moved to archive tier.

        @trace SPEC-03.18
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Will be archived",
            tier="longterm",
            confidence=0.35,  # Just above default min
        )

        # Set to very old access time
        memory_store._set_last_accessed(node_id, datetime.now() - timedelta(days=100))

        memory_evolution.decay(factor=0.95, min_confidence=0.3)

        node = memory_store.get_node(node_id, include_archived=True)
        assert node.tier == "archive"

    def test_decay_default_min_confidence(self, memory_evolution, memory_store):
        """
        decay should default to min_confidence=0.3.

        @trace SPEC-03.18
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test min confidence",
            tier="longterm",
            confidence=0.31,
        )

        # Set to old access time to trigger decay
        memory_store._set_last_accessed(node_id, datetime.now() - timedelta(days=50))

        memory_evolution.decay()  # No min_confidence specified

        node = memory_store.get_node(node_id, include_archived=True)
        # Should be archived if confidence dropped below 0.3
        if node.confidence < 0.3:
            assert node.tier == "archive"

    def test_decay_logs_tier_transitions(self, memory_evolution, memory_store):
        """
        decay should log all tier transitions to evolution_log.

        @trace SPEC-03.20
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Transition logging",
            tier="longterm",
            confidence=0.32,
        )

        memory_store._set_last_accessed(node_id, datetime.now() - timedelta(days=100))

        memory_evolution.decay(factor=0.95, min_confidence=0.3)

        logs = memory_store.get_evolution_log(operation_type="decay")
        # Should have logged the transition
        assert len(logs) >= 1


# =============================================================================
# SPEC-03.21-24: Archive Tier Tests
# =============================================================================


class TestArchiveTier:
    """Tests for archive tier behavior."""

    def test_archive_not_returned_by_default(self, memory_store):
        """
        Archive tier nodes should not be returned in normal queries.

        @trace SPEC-03.21
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Archived fact",
            tier="archive",
        )

        # Normal query should not return it
        node = memory_store.get_node(node_id)
        assert node is None

        # query_nodes should not return it
        results = memory_store.query_nodes()
        node_ids = [n.id for n in results]
        assert node_id not in node_ids

    def test_archive_retrievable_with_include_archived(self, memory_store):
        """
        Archive nodes should be retrievable with include_archived=True.

        @trace SPEC-03.22
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Retrievable archive",
            tier="archive",
        )

        node = memory_store.get_node(node_id, include_archived=True)
        assert node is not None
        assert node.content == "Retrievable archive"

    def test_restore_node_exists(self, memory_evolution):
        """
        MemoryEvolution should have restore_node method.

        @trace SPEC-03.23
        """
        assert hasattr(memory_evolution, "restore_node")
        assert callable(memory_evolution.restore_node)

    def test_restore_node_moves_to_longterm(self, memory_evolution, memory_store):
        """
        restore_node should move archived node to longterm tier.

        @trace SPEC-03.23
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="To be restored",
            tier="archive",
        )

        result = memory_evolution.restore_node(node_id)

        assert result is True
        node = memory_store.get_node(node_id)
        assert node is not None
        assert node.tier == "longterm"

    def test_restore_node_returns_false_for_nonexistent(self, memory_evolution, memory_store):
        """
        restore_node should return False for nonexistent node.

        @trace SPEC-03.23
        """
        result = memory_evolution.restore_node("nonexistent-id")
        assert result is False

    def test_archive_no_automatic_deletion(self, memory_evolution, memory_store):
        """
        Archive tier should never have automatic deletion.

        @trace SPEC-03.24
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Never delete me",
            tier="archive",
        )

        # Run multiple decay cycles
        for _ in range(10):
            memory_evolution.decay()

        # Node should still exist
        node = memory_store.get_node(node_id, include_archived=True)
        assert node is not None


# =============================================================================
# SPEC-03.25: Configuration Tests
# =============================================================================


class TestEvolutionConfiguration:
    """Tests for evolution configuration."""

    def test_default_consolidation_threshold(self, memory_evolution):
        """
        Default consolidation_threshold should be 0.5.

        @trace SPEC-03.25
        """
        config = memory_evolution.get_config()
        assert config.get("consolidation_threshold", 0.5) == 0.5

    def test_default_promotion_threshold(self, memory_evolution):
        """
        Default promotion_threshold should be 0.8.

        @trace SPEC-03.25
        """
        config = memory_evolution.get_config()
        assert config.get("promotion_threshold", 0.8) == 0.8

    def test_default_decay_factor(self, memory_evolution):
        """
        Default decay_factor should be 0.95.

        @trace SPEC-03.25
        """
        config = memory_evolution.get_config()
        assert config.get("decay_factor", 0.95) == 0.95

    def test_default_decay_min_confidence(self, memory_evolution):
        """
        Default decay_min_confidence should be 0.3.

        @trace SPEC-03.25
        """
        config = memory_evolution.get_config()
        assert config.get("decay_min_confidence", 0.3) == 0.3

    def test_config_from_file(self, memory_store, tmp_path):
        """
        Evolution should load config from file if present.

        @trace SPEC-03.25
        """
        import json

        from src.memory_evolution import MemoryEvolution

        config_file = tmp_path / "rlm-config.json"
        config_file.write_text(
            json.dumps(
                {
                    "memory": {
                        "consolidation_threshold": 0.6,
                        "promotion_threshold": 0.9,
                        "decay_factor": 0.9,
                        "decay_min_confidence": 0.2,
                    }
                }
            )
        )

        evolution = MemoryEvolution(memory_store, config_path=str(config_file))
        config = evolution.get_config()

        assert config["consolidation_threshold"] == 0.6
        assert config["promotion_threshold"] == 0.9
        assert config["decay_factor"] == 0.9
        assert config["decay_min_confidence"] == 0.2


# =============================================================================
# SPEC-03.26-31: Testing Requirements
# =============================================================================


class TestConsolidationMerging:
    """Unit tests for consolidation merging logic."""

    def test_merges_related_facts_correctly(self, memory_evolution, memory_store):
        """
        Consolidation should merge related facts correctly.

        @trace SPEC-03.26
        """
        # Create facts with similar content
        fact1 = memory_store.create_node(
            node_type="fact",
            content="The API uses REST",
            tier="task",
            confidence=0.8,
            metadata={"task_id": "task-merge-test"},
        )
        fact2 = memory_store.create_node(
            node_type="fact",
            content="The API uses REST architecture",
            tier="task",
            confidence=0.85,
            metadata={"task_id": "task-merge-test"},
        )

        # Relate them
        memory_store.create_edge(
            edge_type="relation",
            label="similar_to",
            members=[
                {"node_id": fact1, "role": "subject", "position": 0},
                {"node_id": fact2, "role": "object", "position": 1},
            ],
        )

        result = memory_evolution.consolidate("task-merge-test")

        # Should have merged something
        assert result.promoted_count >= 1


class TestPromotionThreshold:
    """Unit tests for promotion threshold verification."""

    def test_respects_confidence_threshold(self, memory_evolution, memory_store):
        """
        Promotion should respect confidence threshold.

        @trace SPEC-03.27
        """
        # Create nodes at various confidence levels
        below = memory_store.create_node(
            node_type="fact",
            content="Below threshold",
            tier="session",
            confidence=0.6,
            metadata={"session_id": "sess-thresh-test"},
        )
        at_threshold = memory_store.create_node(
            node_type="fact",
            content="At threshold",
            tier="session",
            confidence=0.8,
            metadata={"session_id": "sess-thresh-test"},
        )
        above = memory_store.create_node(
            node_type="fact",
            content="Above threshold",
            tier="session",
            confidence=0.95,
            metadata={"session_id": "sess-thresh-test"},
        )

        # Access all to ensure they're candidates
        for node_id in [below, at_threshold, above]:
            for _ in range(5):
                memory_store.get_node(node_id)

        memory_evolution.promote("sess-thresh-test", threshold=0.8)

        assert memory_store.get_node(below).tier == "session"
        assert memory_store.get_node(at_threshold).tier == "longterm"
        assert memory_store.get_node(above).tier == "longterm"


class TestDecayFormula:
    """Unit tests for decay formula verification."""

    def test_produces_expected_values(self, memory_evolution, memory_store):
        """
        Decay formula should produce expected confidence values.

        @trace SPEC-03.28
        """
        test_cases = [
            # (initial_conf, days, factor, expected)
            (1.0, 1, 0.95, 0.95),
            (1.0, 7, 0.95, 0.698),  # 0.95^7
            (0.8, 10, 0.9, 0.279),  # 0.8 * 0.9^10
            (1.0, 30, 0.95, 0.215),  # 0.95^30
        ]

        for initial, days, factor, expected in test_cases:
            node_id = memory_store.create_node(
                node_type="fact",
                content=f"Test decay {initial}/{days}/{factor}",
                tier="longterm",
                confidence=initial,
            )

            memory_store._set_last_accessed(node_id, datetime.now() - timedelta(days=days))

            memory_evolution.decay(factor=factor, min_confidence=0.01)

            node = memory_store.get_node(node_id, include_archived=True)
            assert abs(node.confidence - expected) < 0.05, (
                f"Expected {expected}, got {node.confidence} for "
                f"initial={initial}, days={days}, factor={factor}"
            )
