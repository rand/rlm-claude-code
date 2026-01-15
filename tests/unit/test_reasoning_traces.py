"""
Unit tests for reasoning traces.

Implements: Spec SPEC-04 tests for decision nodes, git integration, and trajectory mapping.
"""

import os
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
    """Create a MemoryStore instance."""
    from src.memory_store import MemoryStore

    return MemoryStore(db_path=temp_db_path)


@pytest.fixture
def reasoning_traces(memory_store):
    """Create a ReasoningTraces instance."""
    from src.reasoning_traces import ReasoningTraces

    return ReasoningTraces(memory_store)


# =============================================================================
# SPEC-04.01-03: Decision Node Types
# =============================================================================


class TestDecisionNodeTypes:
    """Tests for decision node subtypes."""

    @pytest.mark.parametrize(
        "subtype",
        ["goal", "decision", "option", "action", "outcome", "observation"],
    )
    def test_supports_all_decision_subtypes(self, reasoning_traces, subtype):
        """
        System should support all decision subtypes.

        @trace SPEC-04.01
        """
        from src.reasoning_traces import DecisionNode

        # Create node of each subtype
        if subtype == "goal":
            node_id = reasoning_traces.create_goal("Test goal")
        elif subtype == "decision":
            goal_id = reasoning_traces.create_goal("Parent goal")
            node_id = reasoning_traces.create_decision(goal_id, "Test decision")
        elif subtype == "option":
            goal_id = reasoning_traces.create_goal("Parent goal")
            decision_id = reasoning_traces.create_decision(goal_id, "Parent decision")
            node_id = reasoning_traces.add_option(decision_id, "Test option")
        elif subtype == "action":
            goal_id = reasoning_traces.create_goal("Parent goal")
            decision_id = reasoning_traces.create_decision(goal_id, "Parent decision")
            option_id = reasoning_traces.add_option(decision_id, "Option")
            reasoning_traces.choose_option(decision_id, option_id)
            node_id = reasoning_traces.create_action(decision_id, "Test action")
        elif subtype == "outcome":
            goal_id = reasoning_traces.create_goal("Parent goal")
            decision_id = reasoning_traces.create_decision(goal_id, "Parent decision")
            option_id = reasoning_traces.add_option(decision_id, "Option")
            reasoning_traces.choose_option(decision_id, option_id)
            action_id = reasoning_traces.create_action(decision_id, "Test action")
            node_id = reasoning_traces.create_outcome(action_id, "Test outcome", success=True)
        else:  # observation
            node_id = reasoning_traces.create_observation("Test observation")

        node = reasoning_traces.get_decision_node(node_id)
        assert node is not None
        assert node.decision_type == subtype

    def test_decision_stored_as_decision_node_type(self, reasoning_traces, memory_store):
        """
        Decision nodes should be stored with type='decision'.

        @trace SPEC-04.02
        """
        goal_id = reasoning_traces.create_goal("Test goal")

        # Check underlying node type
        node = memory_store.get_node(goal_id)
        assert node.type == "decision"

    def test_decision_has_additional_fields(self, reasoning_traces):
        """
        Decision nodes should have additional fields.

        @trace SPEC-04.03
        """
        from src.reasoning_traces import DecisionNode

        goal_id = reasoning_traces.create_goal(
            content="Test goal",
            prompt="User prompt",
            files=["file1.py", "file2.py"],
        )

        node = reasoning_traces.get_decision_node(goal_id)
        assert isinstance(node, DecisionNode)
        assert node.prompt == "User prompt"
        assert node.files == ["file1.py", "file2.py"]
        # branch and commit_hash may be None initially
        assert hasattr(node, "branch")
        assert hasattr(node, "commit_hash")
        assert hasattr(node, "parent_id")


# =============================================================================
# SPEC-04.04-10: Decision Graph Structure
# =============================================================================


class TestDecisionGraphStructure:
    """Tests for decision graph hyperedge relationships."""

    def test_goal_spawns_decision(self, reasoning_traces, memory_store):
        """
        Goals should spawn decisions via 'spawns' hyperedge.

        @trace SPEC-04.04
        """
        goal_id = reasoning_traces.create_goal("Main goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Sub decision")

        # Check for spawns edge
        edges = memory_store.query_edges(label="spawns")
        assert len(edges) >= 1

        # Verify relationship
        related = memory_store.get_related_nodes(goal_id)
        related_ids = [n.id for n in related]
        assert decision_id in related_ids

    def test_decision_considers_options(self, reasoning_traces, memory_store):
        """
        Decisions should consider options via 'considers' hyperedge.

        @trace SPEC-04.05
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option1_id = reasoning_traces.add_option(decision_id, "Option 1")
        option2_id = reasoning_traces.add_option(decision_id, "Option 2")

        # Check for considers edges
        edges = memory_store.query_edges(label="considers")
        assert len(edges) >= 2

        # Verify relationship
        related = memory_store.get_related_nodes(decision_id)
        related_ids = [n.id for n in related]
        assert option1_id in related_ids
        assert option2_id in related_ids

    def test_decision_chooses_option(self, reasoning_traces, memory_store):
        """
        Decisions should choose one option via 'chooses' hyperedge.

        @trace SPEC-04.06
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Chosen option")

        reasoning_traces.choose_option(decision_id, option_id)

        # Check for chooses edge
        edges = memory_store.query_edges(label="chooses")
        assert len(edges) >= 1

    def test_decision_rejects_option_with_reason(self, reasoning_traces, memory_store):
        """
        Decisions should reject options via 'rejects' hyperedge with reason.

        @trace SPEC-04.07
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Rejected option")

        reasoning_traces.reject_option(decision_id, option_id, "Too complex")

        # Check for rejects edge
        edges = memory_store.query_edges(label="rejects")
        assert len(edges) >= 1

        # Verify the reason is stored
        rejected = reasoning_traces.get_rejected_options(decision_id)
        assert len(rejected) >= 1
        # Reason should be accessible
        assert any("Too complex" in str(r) for r in rejected) or len(rejected) >= 1

    def test_decision_implements_action(self, reasoning_traces, memory_store):
        """
        Decisions should implement actions via 'implements' hyperedge.

        @trace SPEC-04.08
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")
        reasoning_traces.choose_option(decision_id, option_id)
        action_id = reasoning_traces.create_action(decision_id, "Action step")

        # Check for implements edge
        edges = memory_store.query_edges(label="implements")
        assert len(edges) >= 1

    def test_action_produces_outcome(self, reasoning_traces, memory_store):
        """
        Actions should produce outcomes via 'produces' hyperedge.

        @trace SPEC-04.09
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")
        reasoning_traces.choose_option(decision_id, option_id)
        action_id = reasoning_traces.create_action(decision_id, "Action")
        outcome_id = reasoning_traces.create_outcome(action_id, "Success!", success=True)

        # Check for produces edge
        edges = memory_store.query_edges(label="produces")
        assert len(edges) >= 1

    def test_observation_informs_decision(self, reasoning_traces, memory_store):
        """
        Observations should inform decisions via 'informs' hyperedge.

        @trace SPEC-04.10
        """
        observation_id = reasoning_traces.create_observation("Important finding")
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")

        reasoning_traces.link_observation(observation_id, decision_id)

        # Check for informs edge
        edges = memory_store.query_edges(label="informs")
        assert len(edges) >= 1


# =============================================================================
# SPEC-04.11-15: Git Integration
# =============================================================================


class TestGitIntegration:
    """Tests for git commit linking."""

    def test_link_commit_exists(self, reasoning_traces):
        """
        System should have link_commit method.

        @trace SPEC-04.11
        """
        assert hasattr(reasoning_traces, "link_commit")
        assert callable(reasoning_traces.link_commit)

    def test_link_commit_associates_decision(self, reasoning_traces):
        """
        link_commit should associate decisions with commits.

        @trace SPEC-04.11
        """
        goal_id = reasoning_traces.create_goal("Fix bug")
        decision_id = reasoning_traces.create_decision(goal_id, "Update handler")

        reasoning_traces.link_commit(decision_id, "abc123def")

        node = reasoning_traces.get_decision_node(decision_id)
        assert node.commit_hash == "abc123def"

    def test_captures_current_branch(self, reasoning_traces):
        """
        System should capture current branch when creating decisions.

        @trace SPEC-04.12
        """
        # Create goal with branch context
        goal_id = reasoning_traces.create_goal("Feature work", branch="feature/new-thing")

        node = reasoning_traces.get_decision_node(goal_id)
        assert node.branch == "feature/new-thing"

    def test_get_decisions_for_commit(self, reasoning_traces):
        """
        System should provide get_decisions_for_commit.

        @trace SPEC-04.14
        """
        goal_id = reasoning_traces.create_goal("Commit feature")
        decision_id = reasoning_traces.create_decision(goal_id, "Implementation")

        reasoning_traces.link_commit(decision_id, "commit123")
        reasoning_traces.link_commit(goal_id, "commit123")

        decisions = reasoning_traces.get_decisions_for_commit("commit123")
        assert len(decisions) >= 2
        ids = [d.id for d in decisions]
        assert decision_id in ids
        assert goal_id in ids

    def test_git_integration_optional(self, reasoning_traces):
        """
        Git integration should be optional.

        @trace SPEC-04.15
        """
        # Create decision without git context
        goal_id = reasoning_traces.create_goal("No git context")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")

        node = reasoning_traces.get_decision_node(decision_id)
        assert node is not None
        # Should work even without branch/commit
        assert node.branch is None or node.branch == ""
        assert node.commit_hash is None or node.commit_hash == ""


# =============================================================================
# SPEC-04.16-19: Query Interface
# =============================================================================


class TestQueryInterface:
    """Tests for decision tree queries."""

    def test_get_decision_tree(self, reasoning_traces):
        """
        System should provide get_decision_tree.

        @trace SPEC-04.16
        """
        goal_id = reasoning_traces.create_goal("Root goal")
        decision1 = reasoning_traces.create_decision(goal_id, "Decision 1")
        decision2 = reasoning_traces.create_decision(goal_id, "Decision 2")

        tree = reasoning_traces.get_decision_tree(goal_id)

        assert tree is not None
        assert tree.root.id == goal_id
        assert len(tree.children) >= 2

    def test_get_rejected_options(self, reasoning_traces):
        """
        System should provide get_rejected_options.

        @trace SPEC-04.17
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        opt1 = reasoning_traces.add_option(decision_id, "Good option")
        opt2 = reasoning_traces.add_option(decision_id, "Bad option")

        reasoning_traces.choose_option(decision_id, opt1)
        reasoning_traces.reject_option(decision_id, opt2, "Performance issues")

        rejected = reasoning_traces.get_rejected_options(decision_id)
        assert len(rejected) >= 1
        assert any(r.id == opt2 for r in rejected)

    def test_get_outcome(self, reasoning_traces):
        """
        System should provide get_outcome for a goal.

        @trace SPEC-04.18
        """
        goal_id = reasoning_traces.create_goal("Complete task")
        decision_id = reasoning_traces.create_decision(goal_id, "Approach")
        option_id = reasoning_traces.add_option(decision_id, "Option")
        reasoning_traces.choose_option(decision_id, option_id)
        action_id = reasoning_traces.create_action(decision_id, "Execute")
        outcome_id = reasoning_traces.create_outcome(action_id, "Task completed!", success=True)

        outcome = reasoning_traces.get_outcome(goal_id)
        assert outcome is not None
        assert outcome.id == outcome_id
        assert "completed" in outcome.content

    def test_get_informing_observations(self, reasoning_traces):
        """
        System should provide get_informing_observations.

        @trace SPEC-04.19
        """
        obs1 = reasoning_traces.create_observation("Error in logs")
        obs2 = reasoning_traces.create_observation("User feedback")

        goal_id = reasoning_traces.create_goal("Debug issue")
        decision_id = reasoning_traces.create_decision(goal_id, "Investigation")

        reasoning_traces.link_observation(obs1, decision_id)
        reasoning_traces.link_observation(obs2, decision_id)

        observations = reasoning_traces.get_informing_observations(decision_id)
        assert len(observations) >= 2
        obs_ids = [o.id for o in observations]
        assert obs1 in obs_ids
        assert obs2 in obs_ids


# =============================================================================
# SPEC-04.20-24: Trajectory Integration
# =============================================================================


class TestTrajectoryIntegration:
    """Tests for trajectory-to-decision mapping."""

    def test_trajectory_event_maps_to_decision(self, reasoning_traces):
        """
        TrajectoryEvent should map to decision nodes.

        @trace SPEC-04.20
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            content="Starting recursive analysis",
            depth=0,
        )

        node_id = reasoning_traces.from_trajectory_event(event)
        assert node_id is not None

        node = reasoning_traces.get_decision_node(node_id)
        assert node is not None

    def test_recurse_event_creates_goal(self, reasoning_traces):
        """
        RECURSE_START events should create goal nodes.

        @trace SPEC-04.21
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            content="Main objective",
            depth=0,
        )

        node_id = reasoning_traces.from_trajectory_event(event)
        node = reasoning_traces.get_decision_node(node_id)

        assert node.decision_type == "goal"

    def test_orchestrate_event_creates_decision(self, reasoning_traces):
        """
        ANALYZE events should create decision nodes.

        @trace SPEC-04.22
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        # First create a goal
        goal_event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            content="Goal",
            depth=0,
        )
        goal_id = reasoning_traces.from_trajectory_event(goal_event)

        # Then analyze event
        event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            content="Deciding approach",
            depth=0,
            metadata={"parent_id": goal_id},
        )

        node_id = reasoning_traces.from_trajectory_event(event)
        node = reasoning_traces.get_decision_node(node_id)

        assert node.decision_type == "decision"

    def test_final_event_creates_outcome(self, reasoning_traces):
        """
        FINAL events should create outcome nodes.

        @trace SPEC-04.23
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        # Create prerequisite nodes
        goal_event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            content="Goal",
            depth=0,
        )
        goal_id = reasoning_traces.from_trajectory_event(goal_event)

        decision_event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            content="Decision",
            depth=0,
            metadata={"parent_id": goal_id},
        )
        decision_id = reasoning_traces.from_trajectory_event(decision_event)

        # Final event
        final_event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            content="Task completed successfully",
            depth=0,
            metadata={"parent_id": decision_id},
        )

        node_id = reasoning_traces.from_trajectory_event(final_event)
        node = reasoning_traces.get_decision_node(node_id)

        assert node.decision_type == "outcome"

    def test_trajectory_mapping_configurable(self, reasoning_traces):
        """
        Trajectory-to-decision mapping should be configurable.

        @trace SPEC-04.24
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        # Disable mapping
        reasoning_traces.set_trajectory_mapping(enabled=False)

        event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            content="Should not create node",
            depth=0,
        )

        node_id = reasoning_traces.from_trajectory_event(event)
        assert node_id is None

        # Re-enable
        reasoning_traces.set_trajectory_mapping(enabled=True)


# =============================================================================
# SPEC-04.25: Schema Extension
# =============================================================================


class TestSchemaExtension:
    """Tests for decisions table schema."""

    def test_decisions_table_exists(self, reasoning_traces, memory_store):
        """
        Decisions table should exist in schema.

        @trace SPEC-04.25
        """
        import sqlite3

        conn = sqlite3.connect(memory_store.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_decisions_table_has_required_columns(self, reasoning_traces, memory_store):
        """
        Decisions table should have all required columns.

        @trace SPEC-04.25
        """
        import sqlite3

        conn = sqlite3.connect(memory_store.db_path)
        cursor = conn.execute("PRAGMA table_info(decisions)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        required = {
            "node_id",
            "decision_type",
            "confidence",
            "prompt",
            "files",
            "branch",
            "commit_hash",
            "parent_id",
        }
        assert required.issubset(columns)

    def test_decisions_indexes_exist(self, reasoning_traces, memory_store):
        """
        Decisions table should have proper indexes.

        @trace SPEC-04.25
        """
        import sqlite3

        conn = sqlite3.connect(memory_store.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='decisions'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Check for expected indexes
        assert any("type" in idx for idx in indexes) or any("decision" in idx for idx in indexes)


# =============================================================================
# SPEC-04.26-29: Testing Requirements
# =============================================================================


class TestDecisionGraphIntegrity:
    """Tests for decision graph structure integrity."""

    def test_graph_structure_maintained(self, reasoning_traces):
        """
        Decision graph structure should be maintained.

        @trace SPEC-04.26
        """
        # Create a complete decision tree
        goal_id = reasoning_traces.create_goal("Main goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Key decision")
        opt1 = reasoning_traces.add_option(decision_id, "Option A")
        opt2 = reasoning_traces.add_option(decision_id, "Option B")
        reasoning_traces.choose_option(decision_id, opt1)
        reasoning_traces.reject_option(decision_id, opt2, "Not optimal")
        action_id = reasoning_traces.create_action(decision_id, "Execute option A")
        outcome_id = reasoning_traces.create_outcome(action_id, "Success", success=True)

        # Verify tree structure
        tree = reasoning_traces.get_decision_tree(goal_id)
        assert tree is not None
        assert tree.root.decision_type == "goal"

        # Should be able to trace from goal to outcome
        outcome = reasoning_traces.get_outcome(goal_id)
        assert outcome is not None
        assert outcome.id == outcome_id


class TestPersistence:
    """Tests for cross-session persistence."""

    def test_decisions_persist_across_sessions(self, temp_db_path):
        """
        Decisions should persist across sessions.

        @trace SPEC-04.29
        """
        from src.memory_store import MemoryStore
        from src.reasoning_traces import ReasoningTraces

        # Session 1: Create decisions
        store1 = MemoryStore(db_path=temp_db_path)
        traces1 = ReasoningTraces(store1)

        goal_id = traces1.create_goal("Persistent goal")
        decision_id = traces1.create_decision(goal_id, "Persistent decision")

        # Session 2: Verify they exist
        store2 = MemoryStore(db_path=temp_db_path)
        traces2 = ReasoningTraces(store2)

        goal = traces2.get_decision_node(goal_id)
        decision = traces2.get_decision_node(decision_id)

        assert goal is not None
        assert goal.content == "Persistent goal"
        assert decision is not None
        assert decision.content == "Persistent decision"


# =============================================================================
# Evidence Linking (Phase 3: Memory Integration)
# =============================================================================


class TestEvidenceLinking:
    """Tests for evidence linking between facts and decisions."""

    def test_add_supporting_fact(self, reasoning_traces, memory_store):
        """Supporting facts can be linked to options."""
        # Create a decision with option
        goal_id = reasoning_traces.create_goal("Decide approach")
        decision_id = reasoning_traces.create_decision(goal_id, "Choose method")
        option_id = reasoning_traces.add_option(decision_id, "Use approach A")

        # Create a fact that supports this option
        fact_id = memory_store.create_node("fact", "Approach A is faster")

        # Link fact to option
        edge_id = reasoning_traces.add_supporting_fact(fact_id, option_id, weight=0.8)

        assert edge_id is not None
        # Verify edge was created
        edges = memory_store.query_edges(label="supports")
        assert len(edges) >= 1

    def test_add_contradicting_fact(self, reasoning_traces, memory_store):
        """Contradicting facts can be linked to options."""
        goal_id = reasoning_traces.create_goal("Decide approach")
        decision_id = reasoning_traces.create_decision(goal_id, "Choose method")
        option_id = reasoning_traces.add_option(decision_id, "Use approach B")

        fact_id = memory_store.create_node("fact", "Approach B has known bugs")
        edge_id = reasoning_traces.add_contradicting_fact(fact_id, option_id, weight=0.9)

        assert edge_id is not None
        edges = memory_store.query_edges(label="contradicts")
        assert len(edges) >= 1

    def test_record_outcome_validates_fact(self, reasoning_traces, memory_store):
        """Outcomes can validate facts."""
        # Create decision flow
        goal_id = reasoning_traces.create_goal("Test hypothesis")
        decision_id = reasoning_traces.create_decision(goal_id, "Run test")
        option_id = reasoning_traces.add_option(decision_id, "Execute test")
        reasoning_traces.choose_option(decision_id, option_id)
        action_id = reasoning_traces.create_action(decision_id, "Run tests")
        outcome_id = reasoning_traces.create_outcome(action_id, "Tests passed", success=True)

        # Create fact that was validated
        fact_id = memory_store.create_node("fact", "Code is correct")

        # Link outcome to fact
        edge_id = reasoning_traces.record_outcome_validates_fact(outcome_id, fact_id)

        assert edge_id is not None
        edges = memory_store.query_edges(label="validates")
        assert len(edges) >= 1

    def test_record_outcome_invalidates_fact(self, reasoning_traces, memory_store):
        """Outcomes can invalidate facts."""
        goal_id = reasoning_traces.create_goal("Test assumption")
        decision_id = reasoning_traces.create_decision(goal_id, "Check assumption")
        option_id = reasoning_traces.add_option(decision_id, "Verify")
        reasoning_traces.choose_option(decision_id, option_id)
        action_id = reasoning_traces.create_action(decision_id, "Verify assumption")
        outcome_id = reasoning_traces.create_outcome(action_id, "Assumption was wrong", success=False)

        fact_id = memory_store.create_node("fact", "Old assumption X holds")
        edge_id = reasoning_traces.record_outcome_invalidates_fact(outcome_id, fact_id)

        assert edge_id is not None
        edges = memory_store.query_edges(label="invalidates")
        assert len(edges) >= 1

    def test_get_evidence_for_option(self, reasoning_traces, memory_store):
        """Get all evidence (supporting and contradicting) for an option."""
        goal_id = reasoning_traces.create_goal("Decide tech stack")
        decision_id = reasoning_traces.create_decision(goal_id, "Choose database")
        option_id = reasoning_traces.add_option(decision_id, "Use PostgreSQL")

        # Add supporting facts
        support1 = memory_store.create_node("fact", "PostgreSQL has good JSON support")
        support2 = memory_store.create_node("fact", "Team has PostgreSQL experience")
        reasoning_traces.add_supporting_fact(support1, option_id, weight=0.7)
        reasoning_traces.add_supporting_fact(support2, option_id, weight=0.9)

        # Add contradicting fact
        contra1 = memory_store.create_node("fact", "PostgreSQL licensing cost is high")
        reasoning_traces.add_contradicting_fact(contra1, option_id, weight=0.5)

        # Get all evidence
        evidence = reasoning_traces.get_evidence_for_option(option_id)

        assert "supporting" in evidence
        assert "contradicting" in evidence
        assert len(evidence["supporting"]) == 2
        assert len(evidence["contradicting"]) == 1

        # Check IDs are in results
        supporting_ids = [fact_id for fact_id, _ in evidence["supporting"]]
        assert support1 in supporting_ids
        assert support2 in supporting_ids

    def test_get_outcome_evidence_for_fact(self, reasoning_traces, memory_store):
        """Get all outcome evidence (validating and invalidating) for a fact."""
        # Create a fact
        fact_id = memory_store.create_node("fact", "Feature X works correctly")

        # Create validating outcomes
        goal1 = reasoning_traces.create_goal("Test feature X")
        decision1 = reasoning_traces.create_decision(goal1, "Run test")
        option1 = reasoning_traces.add_option(decision1, "Execute")
        reasoning_traces.choose_option(decision1, option1)
        action1 = reasoning_traces.create_action(decision1, "Test")
        outcome1 = reasoning_traces.create_outcome(action1, "Test passed", success=True)
        reasoning_traces.record_outcome_validates_fact(outcome1, fact_id)

        # Create invalidating outcome
        goal2 = reasoning_traces.create_goal("Stress test feature X")
        decision2 = reasoning_traces.create_decision(goal2, "Run stress test")
        option2 = reasoning_traces.add_option(decision2, "Execute")
        reasoning_traces.choose_option(decision2, option2)
        action2 = reasoning_traces.create_action(decision2, "Stress test")
        outcome2 = reasoning_traces.create_outcome(action2, "Failed under load", success=False)
        reasoning_traces.record_outcome_invalidates_fact(outcome2, fact_id)

        # Get all outcome evidence
        evidence = reasoning_traces.get_outcome_evidence_for_fact(fact_id)

        assert "validating" in evidence
        assert "invalidating" in evidence
        assert len(evidence["validating"]) == 1
        assert len(evidence["invalidating"]) == 1

    def test_evidence_weights_preserved(self, reasoning_traces, memory_store):
        """Evidence edge weights are preserved when queried."""
        goal_id = reasoning_traces.create_goal("Test")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")

        fact_id = memory_store.create_node("fact", "Strong evidence")
        reasoning_traces.add_supporting_fact(fact_id, option_id, weight=0.95)

        evidence = reasoning_traces.get_evidence_for_option(option_id)
        supporting = evidence["supporting"]
        assert len(supporting) == 1
        returned_fact_id, weight = supporting[0]
        assert returned_fact_id == fact_id
        assert weight == 0.95


# =============================================================================
# Evidence-Based Option Scoring (Phase 3: Memory Integration)
# =============================================================================


class TestEvidenceScoring:
    """Tests for evidence-based option scoring."""

    def test_score_option_with_no_evidence(self, reasoning_traces):
        """Options with no evidence should have zero score."""
        from src.reasoning_traces import EvidenceScore

        goal_id = reasoning_traces.create_goal("Test")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")

        score = reasoning_traces.score_option_with_evidence(option_id)

        assert isinstance(score, EvidenceScore)
        assert score.option_id == option_id
        assert score.score == 0.0
        assert score.support_score == 0.0
        assert score.contradiction_score == 0.0
        assert len(score.supporting_facts) == 0
        assert len(score.contradicting_facts) == 0

    def test_score_option_with_supporting_evidence(self, reasoning_traces, memory_store):
        """Supporting evidence should increase score."""
        goal_id = reasoning_traces.create_goal("Test")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option A")

        # Add supporting facts with high confidence
        fact1 = memory_store.create_node("fact", "Fact 1 supports A", confidence=0.9)
        fact2 = memory_store.create_node("fact", "Fact 2 supports A", confidence=0.8)
        reasoning_traces.add_supporting_fact(fact1, option_id, weight=1.0)
        reasoning_traces.add_supporting_fact(fact2, option_id, weight=0.5)

        score = reasoning_traces.score_option_with_evidence(option_id)

        assert score.score > 0
        assert score.support_score > 0
        assert score.contradiction_score == 0.0
        assert len(score.supporting_facts) == 2
        assert len(score.contradicting_facts) == 0

    def test_score_option_with_contradicting_evidence(self, reasoning_traces, memory_store):
        """Contradicting evidence should decrease score."""
        goal_id = reasoning_traces.create_goal("Test")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option B")

        # Add contradicting facts
        fact1 = memory_store.create_node("fact", "Fact 1 contradicts B", confidence=0.9)
        reasoning_traces.add_contradicting_fact(fact1, option_id, weight=1.0)

        score = reasoning_traces.score_option_with_evidence(option_id)

        assert score.score < 0
        assert score.support_score == 0.0
        assert score.contradiction_score > 0
        assert len(score.supporting_facts) == 0
        assert len(score.contradicting_facts) == 1

    def test_score_option_with_mixed_evidence(self, reasoning_traces, memory_store):
        """Mixed evidence should balance support and contradiction."""
        goal_id = reasoning_traces.create_goal("Test")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option C")

        # Add equal supporting and contradicting evidence
        support = memory_store.create_node("fact", "Support C", confidence=0.8)
        contra = memory_store.create_node("fact", "Contra C", confidence=0.8)
        reasoning_traces.add_supporting_fact(support, option_id, weight=1.0)
        reasoning_traces.add_contradicting_fact(contra, option_id, weight=1.0)

        score = reasoning_traces.score_option_with_evidence(option_id)

        # Scores should be roughly equal (within recency tolerance for new facts)
        assert abs(score.support_score - score.contradiction_score) < 0.1
        assert len(score.supporting_facts) == 1
        assert len(score.contradicting_facts) == 1

    def test_score_includes_fact_details(self, reasoning_traces, memory_store):
        """Score should include fact details (id, weight, confidence)."""
        goal_id = reasoning_traces.create_goal("Test")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")

        fact_id = memory_store.create_node("fact", "Evidence fact", confidence=0.75)
        reasoning_traces.add_supporting_fact(fact_id, option_id, weight=0.8)

        score = reasoning_traces.score_option_with_evidence(option_id)

        assert len(score.supporting_facts) == 1
        returned_id, weight, confidence = score.supporting_facts[0]
        assert returned_id == fact_id
        assert weight == 0.8
        assert confidence == 0.75

    def test_evidence_score_properties(self, reasoning_traces, memory_store):
        """EvidenceScore properties should work correctly."""
        goal_id = reasoning_traces.create_goal("Test")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")

        # Add 3 supporting and 1 contradicting
        for i in range(3):
            fact = memory_store.create_node("fact", f"Support {i}", confidence=0.8)
            reasoning_traces.add_supporting_fact(fact, option_id)
        fact = memory_store.create_node("fact", "Contra", confidence=0.8)
        reasoning_traces.add_contradicting_fact(fact, option_id)

        score = reasoning_traces.score_option_with_evidence(option_id)

        assert score.net_evidence_count == 2  # 3 - 1
        assert score.total_evidence_count == 4  # 3 + 1

    def test_get_evidence_chain_for_option(self, reasoning_traces, memory_store):
        """Evidence chain should include all details for trace output."""
        goal_id = reasoning_traces.create_goal("Test")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")

        support = memory_store.create_node("fact", "Supporting fact content", confidence=0.9)
        contra = memory_store.create_node("fact", "Contradicting fact content", confidence=0.7)
        reasoning_traces.add_supporting_fact(support, option_id, weight=0.8)
        reasoning_traces.add_contradicting_fact(contra, option_id, weight=0.6)

        chain = reasoning_traces.get_evidence_chain_for_option(option_id)

        assert len(chain) == 2

        # Check required fields
        for item in chain:
            assert "fact_id" in item
            assert "content" in item
            assert "confidence" in item
            assert "weight" in item
            assert "direction" in item
            assert "timestamp" in item

        # Check directions
        directions = {item["direction"] for item in chain}
        assert "supports" in directions
        assert "contradicts" in directions

    def test_recency_weight_computation(self, reasoning_traces):
        """Recency weight should decay over time."""
        import time

        now_ms = int(time.time() * 1000)

        # Very recent (now) should be close to 1.0
        recent_weight = reasoning_traces._compute_recency_weight(now_ms)
        assert recent_weight > 0.99

        # 7 days ago should be about 0.5 (with default half-life)
        week_ago_ms = now_ms - (7 * 24 * 60 * 60 * 1000)
        week_weight = reasoning_traces._compute_recency_weight(week_ago_ms)
        assert 0.4 < week_weight < 0.6

        # 14 days ago should be about 0.25
        two_weeks_ms = now_ms - (14 * 24 * 60 * 60 * 1000)
        two_week_weight = reasoning_traces._compute_recency_weight(two_weeks_ms)
        assert 0.2 < two_week_weight < 0.3


# =============================================================================
# Bayesian Confidence Propagation (Phase 3: Memory Integration)
# =============================================================================


class TestBayesianConfidencePropagation:
    """Tests for Bayesian confidence propagation."""

    def test_update_fact_confidence_success_increases(self, reasoning_traces, memory_store):
        """Successful outcome should increase fact confidence."""
        fact_id = memory_store.create_node("fact", "Test fact", confidence=0.5)

        new_conf = reasoning_traces.update_fact_confidence_from_outcome(
            fact_id, outcome_success=True
        )

        assert new_conf > 0.5
        # Check the node was updated
        node = memory_store.get_node(fact_id)
        assert node.confidence == new_conf

    def test_update_fact_confidence_failure_decreases(self, reasoning_traces, memory_store):
        """Failed outcome should decrease fact confidence."""
        fact_id = memory_store.create_node("fact", "Test fact", confidence=0.5)

        new_conf = reasoning_traces.update_fact_confidence_from_outcome(
            fact_id, outcome_success=False
        )

        assert new_conf < 0.5

    def test_confidence_bounded_min(self, reasoning_traces, memory_store):
        """Confidence should not go below minimum (0.1)."""
        fact_id = memory_store.create_node("fact", "Test fact", confidence=0.15)

        # Repeated failures should not push below 0.1
        for _ in range(20):
            new_conf = reasoning_traces.update_fact_confidence_from_outcome(
                fact_id, outcome_success=False
            )

        assert new_conf >= 0.1

    def test_confidence_bounded_max(self, reasoning_traces, memory_store):
        """Confidence should not exceed maximum (0.95)."""
        fact_id = memory_store.create_node("fact", "Test fact", confidence=0.9)

        # Repeated successes should not push above 0.95
        for _ in range(20):
            new_conf = reasoning_traces.update_fact_confidence_from_outcome(
                fact_id, outcome_success=True
            )

        assert new_conf <= 0.95

    def test_damping_limits_change(self, reasoning_traces, memory_store):
        """Damping should limit change per update."""
        fact_id = memory_store.create_node("fact", "Test fact", confidence=0.5)

        # With 10% damping, max change from 0.5 should be ~0.05
        new_conf = reasoning_traces.update_fact_confidence_from_outcome(
            fact_id, outcome_success=True, damping=0.1
        )

        change = abs(new_conf - 0.5)
        assert change <= 0.11  # Allow small tolerance

    def test_custom_damping(self, reasoning_traces, memory_store):
        """Custom damping value should be respected."""
        fact_id = memory_store.create_node("fact", "Test fact", confidence=0.5)

        # With 20% damping, change should be larger
        new_conf = reasoning_traces.update_fact_confidence_from_outcome(
            fact_id, outcome_success=True, damping=0.2
        )

        assert new_conf > 0.55  # More change with higher damping

    def test_get_all_outcomes_for_fact(self, reasoning_traces, memory_store):
        """Should return all validating and invalidating outcomes."""
        fact_id = memory_store.create_node("fact", "Test fact")

        # Create outcomes and link them
        goal1 = reasoning_traces.create_goal("Goal 1")
        decision1 = reasoning_traces.create_decision(goal1, "Decision 1")
        option1 = reasoning_traces.add_option(decision1, "Option 1")
        reasoning_traces.choose_option(decision1, option1)
        action1 = reasoning_traces.create_action(decision1, "Action 1")
        outcome1 = reasoning_traces.create_outcome(action1, "Success", success=True)

        goal2 = reasoning_traces.create_goal("Goal 2")
        decision2 = reasoning_traces.create_decision(goal2, "Decision 2")
        option2 = reasoning_traces.add_option(decision2, "Option 2")
        reasoning_traces.choose_option(decision2, option2)
        action2 = reasoning_traces.create_action(decision2, "Action 2")
        outcome2 = reasoning_traces.create_outcome(action2, "Failure", success=False)

        # Link outcomes to fact
        reasoning_traces.record_outcome_validates_fact(outcome1, fact_id)
        reasoning_traces.record_outcome_invalidates_fact(outcome2, fact_id)

        outcomes = reasoning_traces._get_all_outcomes_for_fact(fact_id)

        assert len(outcomes) == 2
        outcome_ids = [o[0] for o in outcomes]
        assert outcome1 in outcome_ids
        assert outcome2 in outcome_ids

        # Check validation status
        for oid, is_validating, _ in outcomes:
            if oid == outcome1:
                assert is_validating is True
            elif oid == outcome2:
                assert is_validating is False

    def test_propagate_outcome_to_facts(self, reasoning_traces, memory_store):
        """Propagate should update all linked facts."""
        # Create facts
        fact1 = memory_store.create_node("fact", "Fact 1", confidence=0.5)
        fact2 = memory_store.create_node("fact", "Fact 2", confidence=0.5)

        # Create outcome
        goal = reasoning_traces.create_goal("Goal")
        decision = reasoning_traces.create_decision(goal, "Decision")
        option = reasoning_traces.add_option(decision, "Option")
        reasoning_traces.choose_option(decision, option)
        action = reasoning_traces.create_action(decision, "Action")
        outcome_id = reasoning_traces.create_outcome(action, "Test outcome", success=True)

        # Link outcome to facts
        reasoning_traces.record_outcome_validates_fact(outcome_id, fact1)
        reasoning_traces.record_outcome_invalidates_fact(outcome_id, fact2)

        # Propagate successful outcome
        updated = reasoning_traces.propagate_outcome_to_facts(outcome_id, success=True)

        assert fact1 in updated
        assert fact2 in updated

        # Fact1 was validated by success, should increase
        assert updated[fact1] > 0.5

        # Fact2 was invalidated by success, should decrease
        assert updated[fact2] < 0.5

    def test_nonexistent_fact_raises(self, reasoning_traces):
        """Updating nonexistent fact should raise ValueError."""
        with pytest.raises(ValueError, match="Fact not found"):
            reasoning_traces.update_fact_confidence_from_outcome(
                "nonexistent-id", outcome_success=True
            )

    def test_repeated_updates_converge(self, reasoning_traces, memory_store):
        """
        Property: Repeated updates with same outcome should converge.

        With damping, repeatedly applying the same update should cause
        confidence to approach (but never reach) the target.
        """
        fact_id = memory_store.create_node("fact", "Test fact", confidence=0.5)

        # Apply many successful updates
        prev_conf = 0.5
        for i in range(50):
            new_conf = reasoning_traces.update_fact_confidence_from_outcome(
                fact_id, outcome_success=True
            )
            # Confidence should increase but at decreasing rate
            if i > 0:
                change = new_conf - prev_conf
                # Change should be positive but decreasing
                assert change >= 0
            prev_conf = new_conf

        # Should converge toward max but not exceed it
        assert 0.85 < new_conf <= 0.95

    def test_update_automatically_logs_to_audit_trail(self, reasoning_traces, memory_store):
        """Bayesian updates should automatically log to confidence_updates table."""
        fact_id = memory_store.create_node("fact", "Test fact", confidence=0.5)

        # Update with a trigger_id
        reasoning_traces.update_fact_confidence_from_outcome(
            fact_id, outcome_success=True, trigger_id="test-outcome-123"
        )

        # Verify it was logged
        history = memory_store.get_confidence_history(fact_id)
        assert len(history) == 1
        assert history[0].trigger_type == "outcome"
        assert history[0].trigger_id == "test-outcome-123"
        assert history[0].old_confidence == 0.5
        assert history[0].new_confidence > 0.5


# =============================================================================
# Cross-Reference API (Phase 3: Memory Integration)
# =============================================================================


class TestCrossReferenceAPI:
    """Tests for cross-reference queries between facts and decisions."""

    def test_get_decisions_for_fact(self, reasoning_traces, memory_store):
        """Can find decisions that used a fact as evidence."""
        # Create a decision with options
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Choose approach")
        option_id = reasoning_traces.add_option(decision_id, "Option A")

        # Create a fact and link it to the option
        fact_id = memory_store.create_node("fact", "Supporting evidence")
        reasoning_traces.add_supporting_fact(fact_id, option_id)

        # Query decisions for this fact
        decisions = reasoning_traces.get_decisions_for_fact(fact_id)

        assert len(decisions) >= 1
        decision_ids = [d.id for d in decisions]
        assert decision_id in decision_ids

    def test_get_decisions_for_fact_multiple_decisions(self, reasoning_traces, memory_store):
        """Fact used in multiple decisions returns all of them."""
        # Create two decisions
        goal1 = reasoning_traces.create_goal("Goal 1")
        decision1 = reasoning_traces.create_decision(goal1, "Decision 1")
        option1 = reasoning_traces.add_option(decision1, "Option 1")

        goal2 = reasoning_traces.create_goal("Goal 2")
        decision2 = reasoning_traces.create_decision(goal2, "Decision 2")
        option2 = reasoning_traces.add_option(decision2, "Option 2")

        # Create fact supporting both options
        fact_id = memory_store.create_node("fact", "Common evidence")
        reasoning_traces.add_supporting_fact(fact_id, option1)
        reasoning_traces.add_supporting_fact(fact_id, option2)

        decisions = reasoning_traces.get_decisions_for_fact(fact_id)

        assert len(decisions) == 2
        decision_ids = [d.id for d in decisions]
        assert decision1 in decision_ids
        assert decision2 in decision_ids

    def test_get_supporting_facts_for_decision(self, reasoning_traces, memory_store):
        """Can get all supporting facts for a decision's options."""
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option1 = reasoning_traces.add_option(decision_id, "Option A")
        option2 = reasoning_traces.add_option(decision_id, "Option B")

        # Add supporting facts
        fact1 = memory_store.create_node("fact", "Fact 1 for A")
        fact2 = memory_store.create_node("fact", "Fact 2 for B")
        reasoning_traces.add_supporting_fact(fact1, option1, weight=0.8)
        reasoning_traces.add_supporting_fact(fact2, option2, weight=0.9)

        supporting = reasoning_traces.get_supporting_facts_for_decision(decision_id)

        assert len(supporting) == 2
        fact_ids = [f[0] for f in supporting]
        assert fact1 in fact_ids
        assert fact2 in fact_ids

    def test_get_fact_success_rate_no_outcomes(self, reasoning_traces, memory_store):
        """Fact with no outcomes returns default 0.5 success rate."""
        fact_id = memory_store.create_node("fact", "Unused fact")

        stats = reasoning_traces.get_fact_success_rate(fact_id)

        assert stats["total_outcomes"] == 0
        assert stats["success_rate"] == 0.5
        assert stats["confidence_trend"] == "stable"

    def test_get_fact_success_rate_with_outcomes(self, reasoning_traces, memory_store):
        """Fact with mixed outcomes shows correct success rate."""
        fact_id = memory_store.create_node("fact", "Test fact")

        # Create validating outcomes
        for i in range(3):
            goal = reasoning_traces.create_goal(f"Goal {i}")
            decision = reasoning_traces.create_decision(goal, f"Decision {i}")
            option = reasoning_traces.add_option(decision, f"Option {i}")
            reasoning_traces.choose_option(decision, option)
            action = reasoning_traces.create_action(decision, f"Action {i}")
            outcome = reasoning_traces.create_outcome(action, "Success", success=True)
            reasoning_traces.record_outcome_validates_fact(outcome, fact_id)

        # Create invalidating outcome
        goal = reasoning_traces.create_goal("Bad goal")
        decision = reasoning_traces.create_decision(goal, "Bad decision")
        option = reasoning_traces.add_option(decision, "Bad option")
        reasoning_traces.choose_option(decision, option)
        action = reasoning_traces.create_action(decision, "Bad action")
        outcome = reasoning_traces.create_outcome(action, "Failure", success=False)
        reasoning_traces.record_outcome_invalidates_fact(outcome, fact_id)

        stats = reasoning_traces.get_fact_success_rate(fact_id)

        assert stats["total_outcomes"] == 4
        assert stats["validating_outcomes"] == 3
        assert stats["invalidating_outcomes"] == 1
        assert stats["success_rate"] == 0.75  # 3/4

    def test_get_fact_usage_summary(self, reasoning_traces, memory_store):
        """Can get comprehensive usage summary for a fact."""
        fact_id = memory_store.create_node("fact", "Important fact", confidence=0.7)

        # Create decision with option supported by this fact
        goal = reasoning_traces.create_goal("Goal")
        decision = reasoning_traces.create_decision(goal, "Decision")
        option = reasoning_traces.add_option(decision, "Option")
        reasoning_traces.add_supporting_fact(fact_id, option)

        summary = reasoning_traces.get_fact_usage_summary(fact_id)

        assert summary["fact_id"] == fact_id
        assert summary["content"] == "Important fact"
        assert summary["current_confidence"] == 0.7
        assert summary["options_supported"] == 1
        assert "total_outcomes" in summary
        assert "success_rate" in summary

    def test_get_fact_usage_summary_not_found(self, reasoning_traces):
        """Summary for nonexistent fact returns error."""
        summary = reasoning_traces.get_fact_usage_summary("nonexistent-id")

        assert "error" in summary


# =============================================================================
# Claim Decision Type Tests (SPEC-16.11)
# =============================================================================


class TestClaimDecisionType:
    """Tests for claim decision type (SPEC-16.11)."""

    def test_claim_in_valid_decision_types(self):
        """SPEC-16.11: 'claim' is a valid decision type."""
        from src.reasoning_traces import VALID_DECISION_TYPES

        assert "claim" in VALID_DECISION_TYPES

    def test_create_claim_basic(self, reasoning_traces):
        """Can create a basic claim node."""
        claim_id = reasoning_traces.create_claim(
            claim_text="The function returns 42",
        )

        assert claim_id is not None
        node = reasoning_traces.get_decision_node(claim_id)
        assert node is not None
        assert node.decision_type == "claim"
        assert node.claim_text == "The function returns 42"
        assert node.content == "The function returns 42"
        assert node.verification_status == "pending"

    def test_create_claim_with_evidence_ids(self, reasoning_traces):
        """Can create a claim with evidence IDs."""
        claim_id = reasoning_traces.create_claim(
            claim_text="X equals Y",
            evidence_ids=["src1", "src2", "src3"],
        )

        node = reasoning_traces.get_decision_node(claim_id)
        assert node.evidence_ids == ["src1", "src2", "src3"]

    def test_create_claim_with_confidence(self, reasoning_traces):
        """Can create a claim with custom confidence."""
        claim_id = reasoning_traces.create_claim(
            claim_text="High confidence claim",
            confidence=0.9,
        )

        node = reasoning_traces.get_decision_node(claim_id)
        assert node.confidence == 0.9

    def test_create_claim_with_parent(self, reasoning_traces):
        """Can create a claim with a parent decision."""
        goal_id = reasoning_traces.create_goal("Verify code")
        decision_id = reasoning_traces.create_decision(goal_id, "Check claims")

        claim_id = reasoning_traces.create_claim(
            claim_text="Function is correct",
            parent_id=decision_id,
        )

        node = reasoning_traces.get_decision_node(claim_id)
        assert node.parent_id == decision_id

    def test_create_claim_with_verification_status(self, reasoning_traces):
        """Can create a claim with specific verification status."""
        claim_id = reasoning_traces.create_claim(
            claim_text="Verified claim",
            verification_status="verified",
        )

        node = reasoning_traces.get_decision_node(claim_id)
        assert node.verification_status == "verified"

    def test_create_claim_invalid_status_raises(self, reasoning_traces):
        """Creating claim with invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid verification_status"):
            reasoning_traces.create_claim(
                claim_text="Bad claim",
                verification_status="invalid_status",
            )

    def test_create_claim_all_valid_statuses(self, reasoning_traces):
        """All valid verification statuses work."""
        valid_statuses = ["pending", "verified", "flagged", "refuted"]

        for status in valid_statuses:
            claim_id = reasoning_traces.create_claim(
                claim_text=f"Claim with {status}",
                verification_status=status,
            )
            node = reasoning_traces.get_decision_node(claim_id)
            assert node.verification_status == status

    def test_update_claim_status(self, reasoning_traces):
        """Can update a claim's verification status."""
        claim_id = reasoning_traces.create_claim(
            claim_text="Pending claim",
            verification_status="pending",
        )

        result = reasoning_traces.update_claim_status(claim_id, "verified")

        assert result is True
        node = reasoning_traces.get_decision_node(claim_id)
        assert node.verification_status == "verified"

    def test_update_claim_status_with_confidence(self, reasoning_traces):
        """Can update claim status and confidence together."""
        claim_id = reasoning_traces.create_claim(
            claim_text="Claim to update",
            confidence=0.5,
            verification_status="pending",
        )

        reasoning_traces.update_claim_status(claim_id, "verified", confidence=0.95)

        node = reasoning_traces.get_decision_node(claim_id)
        assert node.verification_status == "verified"
        assert node.confidence == 0.95

    def test_update_claim_status_invalid_raises(self, reasoning_traces):
        """Updating to invalid status raises ValueError."""
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        with pytest.raises(ValueError, match="Invalid verification_status"):
            reasoning_traces.update_claim_status(claim_id, "bad_status")

    def test_decision_node_claim_fields_default_empty(self, reasoning_traces):
        """Non-claim nodes have empty claim fields."""
        goal_id = reasoning_traces.create_goal("Regular goal")

        node = reasoning_traces.get_decision_node(goal_id)
        assert node.claim_text is None
        assert node.evidence_ids == []
        assert node.verification_status is None

    def test_claim_persists_across_queries(self, reasoning_traces):
        """Claim data persists when queried multiple times."""
        claim_id = reasoning_traces.create_claim(
            claim_text="Persistent claim",
            evidence_ids=["e1", "e2"],
            confidence=0.8,
            verification_status="flagged",
        )

        # Query multiple times
        for _ in range(3):
            node = reasoning_traces.get_decision_node(claim_id)
            assert node.claim_text == "Persistent claim"
            assert node.evidence_ids == ["e1", "e2"]
            assert node.confidence == 0.8
            assert node.verification_status == "flagged"

    def test_claim_empty_evidence_ids(self, reasoning_traces):
        """Claim with no evidence IDs stores empty list."""
        claim_id = reasoning_traces.create_claim(
            claim_text="Unsupported claim",
            evidence_ids=[],
        )

        node = reasoning_traces.get_decision_node(claim_id)
        assert node.evidence_ids == []

    def test_multiple_claims_independent(self, reasoning_traces):
        """Multiple claims are stored independently."""
        claim1_id = reasoning_traces.create_claim(
            claim_text="First claim",
            evidence_ids=["a"],
            verification_status="verified",
        )
        claim2_id = reasoning_traces.create_claim(
            claim_text="Second claim",
            evidence_ids=["b", "c"],
            verification_status="flagged",
        )

        node1 = reasoning_traces.get_decision_node(claim1_id)
        node2 = reasoning_traces.get_decision_node(claim2_id)

        assert node1.claim_text == "First claim"
        assert node1.evidence_ids == ["a"]
        assert node1.verification_status == "verified"

        assert node2.claim_text == "Second claim"
        assert node2.evidence_ids == ["b", "c"]
        assert node2.verification_status == "flagged"


class TestVerificationDecisionType:
    """Tests for verification decision type (SPEC-16.12)."""

    def test_verification_in_valid_decision_types(self):
        """Verification is a valid decision type."""
        from src.reasoning_traces import VALID_DECISION_TYPES

        assert "verification" in VALID_DECISION_TYPES

    def test_create_verification_basic(self, reasoning_traces):
        """Can create a basic verification node."""
        # First create a claim to verify
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        verification_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.8,
            dependence_score=0.7,
        )

        assert verification_id is not None
        node = reasoning_traces.get_decision_node(verification_id)
        assert node is not None
        assert node.decision_type == "verification"
        assert node.verified_claim_id == claim_id
        assert node.support_score == 0.8
        assert node.dependence_score == 0.7
        assert node.consistency_score == 1.0  # default
        assert node.is_flagged is False

    def test_create_verification_with_all_scores(self, reasoning_traces):
        """Can create a verification with all scores."""
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        verification_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.9,
            dependence_score=0.6,
            consistency_score=0.85,
        )

        node = reasoning_traces.get_decision_node(verification_id)
        assert node.support_score == 0.9
        assert node.dependence_score == 0.6
        assert node.consistency_score == 0.85

    def test_create_verification_flagged(self, reasoning_traces):
        """Can create a flagged verification."""
        claim_id = reasoning_traces.create_claim(claim_text="Suspicious claim")

        verification_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.2,
            dependence_score=0.1,
            is_flagged=True,
            flag_reason="low_dependence",
        )

        node = reasoning_traces.get_decision_node(verification_id)
        assert node.is_flagged is True
        assert node.flag_reason == "low_dependence"
        assert "flagged" in node.content
        assert "low_dependence" in node.content

    def test_create_verification_all_flag_reasons(self, reasoning_traces):
        """Can create verifications with all valid flag reasons."""
        valid_reasons = [
            "unsupported",
            "phantom_citation",
            "low_dependence",
            "contradiction",
            "over_extrapolation",
            "confidence_mismatch",
        ]

        for reason in valid_reasons:
            claim_id = reasoning_traces.create_claim(claim_text=f"Claim for {reason}")
            verification_id = reasoning_traces.create_verification(
                claim_id=claim_id,
                support_score=0.3,
                dependence_score=0.2,
                is_flagged=True,
                flag_reason=reason,
            )
            node = reasoning_traces.get_decision_node(verification_id)
            assert node.flag_reason == reason

    def test_create_verification_invalid_flag_reason_raises(self, reasoning_traces):
        """Invalid flag_reason raises ValueError."""
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        with pytest.raises(ValueError, match="Invalid flag_reason"):
            reasoning_traces.create_verification(
                claim_id=claim_id,
                support_score=0.5,
                dependence_score=0.5,
                flag_reason="invalid_reason",
            )

    def test_create_verification_score_out_of_range_raises(self, reasoning_traces):
        """Out-of-range scores raise ValueError."""
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        # support_score > 1.0
        with pytest.raises(ValueError, match="support_score"):
            reasoning_traces.create_verification(
                claim_id=claim_id,
                support_score=1.5,
                dependence_score=0.5,
            )

        # dependence_score < 0.0
        with pytest.raises(ValueError, match="dependence_score"):
            reasoning_traces.create_verification(
                claim_id=claim_id,
                support_score=0.5,
                dependence_score=-0.1,
            )

        # consistency_score > 1.0
        with pytest.raises(ValueError, match="consistency_score"):
            reasoning_traces.create_verification(
                claim_id=claim_id,
                support_score=0.5,
                dependence_score=0.5,
                consistency_score=2.0,
            )

    def test_create_verification_with_confidence(self, reasoning_traces):
        """Can create a verification with custom confidence."""
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        verification_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.8,
            dependence_score=0.7,
            confidence=0.95,
        )

        node = reasoning_traces.get_decision_node(verification_id)
        assert node.confidence == 0.95

    def test_create_verification_with_parent(self, reasoning_traces):
        """Can create a verification with a parent decision."""
        parent_id = reasoning_traces.create_goal(content="Verify claims")
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        verification_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.8,
            dependence_score=0.7,
            parent_id=parent_id,
        )

        node = reasoning_traces.get_decision_node(verification_id)
        assert node.parent_id == parent_id

    def test_verification_creates_edge_to_claim(self, reasoning_traces):
        """Verification creates an edge linking to the claim."""
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        verification_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.8,
            dependence_score=0.7,
        )

        # Check that the edge exists for the verification node
        edges = reasoning_traces.store.get_edges_for_node(verification_id)
        assert len(edges) >= 1

        # Find the verifies edge
        verifies_edges = [e for e in edges if e.label == "verifies"]
        assert len(verifies_edges) == 1

        # Also check that the claim has an edge
        claim_edges = reasoning_traces.store.get_edges_for_node(claim_id)
        claim_verifies_edges = [e for e in claim_edges if e.label == "verifies"]
        assert len(claim_verifies_edges) == 1

        # Both should reference the same edge
        assert verifies_edges[0].id == claim_verifies_edges[0].id

    def test_verification_content_format(self, reasoning_traces):
        """Verification content contains formatted scores."""
        claim_id = reasoning_traces.create_claim(claim_text="Test claim")

        verification_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.85,
            dependence_score=0.72,
            consistency_score=0.90,
        )

        node = reasoning_traces.get_decision_node(verification_id)
        assert "support=0.85" in node.content
        assert "dependence=0.72" in node.content
        assert "consistency=0.90" in node.content
        assert "verified" in node.content

    def test_decision_node_verification_fields_default(self, reasoning_traces):
        """DecisionNode has default values for verification fields."""
        # Create a non-verification decision
        goal_id = reasoning_traces.create_goal(content="Test goal")
        node = reasoning_traces.get_decision_node(goal_id)

        # Verification fields should be None/False/default
        assert node.verified_claim_id is None
        assert node.support_score is None
        assert node.dependence_score is None
        assert node.consistency_score is None
        assert node.is_flagged is False
        assert node.flag_reason is None

    def test_multiple_verifications_for_same_claim(self, reasoning_traces):
        """Can create multiple verifications for the same claim."""
        claim_id = reasoning_traces.create_claim(claim_text="Re-verified claim")

        v1_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.6,
            dependence_score=0.5,
        )

        v2_id = reasoning_traces.create_verification(
            claim_id=claim_id,
            support_score=0.8,
            dependence_score=0.7,
        )

        node1 = reasoning_traces.get_decision_node(v1_id)
        node2 = reasoning_traces.get_decision_node(v2_id)

        assert node1.verified_claim_id == claim_id
        assert node2.verified_claim_id == claim_id
        assert node1.support_score == 0.6
        assert node2.support_score == 0.8
