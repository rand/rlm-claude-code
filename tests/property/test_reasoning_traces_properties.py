"""
Property-based tests for reasoning traces.

Implements: Spec SPEC-04.30 - Property tests for decision tree acyclicity.
"""

import os
import sys
import tempfile
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


def make_reasoning_traces(temp_db_path):
    """Create a fresh ReasoningTraces instance."""
    from src.memory_store import MemoryStore
    from src.reasoning_traces import ReasoningTraces

    store = MemoryStore(db_path=temp_db_path)
    return ReasoningTraces(store), store


# =============================================================================
# Strategies
# =============================================================================

content_strategy = st.text(min_size=1, max_size=100)
depth_strategy = st.integers(min_value=1, max_value=5)
options_strategy = st.integers(min_value=1, max_value=5)


# =============================================================================
# SPEC-04.30: Decision Trees Are Acyclic
# =============================================================================


@pytest.mark.hypothesis
class TestDecisionTreeProperties:
    """Property tests for decision tree invariants."""

    @given(
        goal_content=content_strategy,
        num_decisions=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_decision_tree_is_acyclic(self, goal_content, num_decisions):
        """
        Decision trees should always be acyclic.

        @trace SPEC-04.30
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            traces, store = make_reasoning_traces(temp_path)

            # Create a goal and several decisions
            goal_id = traces.create_goal(goal_content)

            decision_ids = []
            for i in range(num_decisions):
                decision_id = traces.create_decision(goal_id, f"Decision {i}")
                decision_ids.append(decision_id)

            # Get the tree and verify acyclicity
            tree = traces.get_decision_tree(goal_id)
            assert tree is not None

            # Verify no cycles by checking that we can't reach any node from itself
            visited = set()
            has_cycle = _check_for_cycles(traces, goal_id, visited, set())
            assert not has_cycle, "Decision tree has a cycle!"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)

    @given(
        num_options=options_strategy,
    )
    @settings(max_examples=20, deadline=None)
    def test_only_one_option_chosen(self, num_options):
        """
        Only one option should be marked as chosen per decision.

        @trace SPEC-04.06
        """
        assume(num_options >= 2)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            traces, store = make_reasoning_traces(temp_path)

            goal_id = traces.create_goal("Test goal")
            decision_id = traces.create_decision(goal_id, "Test decision")

            # Add multiple options
            option_ids = []
            for i in range(num_options):
                opt_id = traces.add_option(decision_id, f"Option {i}")
                option_ids.append(opt_id)

            # Choose one option
            chosen_idx = 0
            traces.choose_option(decision_id, option_ids[chosen_idx])

            # Verify only one is chosen
            chosen_edges = store.query_edges(label="chooses")
            # Filter to this decision
            decision_chosen = [
                e
                for e in chosen_edges
                if any(m["node_id"] == decision_id for m in store.get_edge_members(e.id))
            ]
            assert len(decision_chosen) == 1

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)

    @given(
        depth=depth_strategy,
    )
    @settings(max_examples=15, deadline=None)
    def test_parent_child_relationship_maintained(self, depth):
        """
        Parent-child relationships should be maintained at all depths.

        @trace SPEC-04.26
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        try:
            traces, store = make_reasoning_traces(temp_path)

            # Create a chain of decisions
            goal_id = traces.create_goal("Root goal")
            parent_id = goal_id

            for i in range(depth):
                decision_id = traces.create_decision(parent_id, f"Decision level {i}")
                parent_id = decision_id

            # Verify tree structure
            tree = traces.get_decision_tree(goal_id)
            assert tree is not None

            # Count depth by traversing
            actual_depth = _count_tree_depth(tree)
            assert actual_depth >= depth

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            for suffix in ["-wal", "-shm"]:
                wal_path = temp_path + suffix
                if os.path.exists(wal_path):
                    os.unlink(wal_path)


# =============================================================================
# Helper Functions
# =============================================================================


def _check_for_cycles(traces, node_id, visited, path):
    """Check if there's a cycle starting from node_id."""
    if node_id in path:
        return True  # Cycle detected

    if node_id in visited:
        return False  # Already checked this branch

    visited.add(node_id)
    path.add(node_id)

    # Get children
    node = traces.get_decision_node(node_id)
    if node is None:
        path.remove(node_id)
        return False

    # Get related nodes (children)
    tree = traces.get_decision_tree(node_id)
    if tree and hasattr(tree, "children"):
        for child in tree.children:
            if _check_for_cycles(traces, child.root.id, visited, path):
                return True

    path.remove(node_id)
    return False


def _count_tree_depth(tree, current_depth=1):
    """Count the maximum depth of a decision tree."""
    if not hasattr(tree, "children") or not tree.children:
        return current_depth

    max_child_depth = current_depth
    for child in tree.children:
        child_depth = _count_tree_depth(child, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)

    return max_child_depth
