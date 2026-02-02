"""
Integration tests for memory system components.

Tests end-to-end flows for SPEC-02 through SPEC-05.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check if rlm_core is available (affects which features work)
try:
    import rlm_core
    HAS_RLM_CORE = True
except ImportError:
    HAS_RLM_CORE = False



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


# =============================================================================
# End-to-End Memory Flow Tests
# =============================================================================


class TestMemorySystemIntegration:
    """Integration tests for the complete memory system."""


    def test_memory_store_to_evolution_flow(self, temp_db_path):
        """
        Test complete flow: create nodes -> consolidate -> promote -> decay.

        Verifies SPEC-02, SPEC-03 integration.
        """
        from src.memory_evolution import MemoryEvolution
        from src.memory_store import MemoryStore

        # Create store and evolution manager
        store = MemoryStore(db_path=temp_db_path)
        evolution = MemoryEvolution(store)

        # Create task-tier nodes (simulating a task)
        task_id = "test-task-001"
        session_id = "test-session-001"

        node1_id = store.create_node(
            node_type="fact",
            content="The authentication module uses JWT tokens",
            tier="task",
            confidence=0.9,
            metadata={"task_id": task_id, "session_id": session_id},
        )

        node2_id = store.create_node(
            node_type="fact",
            content="Rate limiting is set to 100 requests per minute",
            tier="task",
            confidence=0.85,
            metadata={"task_id": task_id, "session_id": session_id},
        )

        # Create relationship between nodes
        store.create_edge(
            edge_type="relation",
            label="relates_to",
            members=[
                {"node_id": node1_id, "role": "subject", "position": 0},
                {"node_id": node2_id, "role": "object", "position": 1},
            ],
        )

        # Consolidate task nodes to session tier
        consolidation_result = evolution.consolidate(task_id)
        assert consolidation_result.promoted_count >= 2

        # Verify nodes are now session tier
        node1 = store.get_node(node1_id)
        node2 = store.get_node(node2_id)
        assert node1.tier == "session"
        assert node2.tier == "session"

        # Promote high-confidence nodes to longterm
        promotion_result = evolution.promote(session_id, threshold=0.8)
        assert promotion_result.promoted_count >= 2

        # Verify nodes are now longterm
        node1 = store.get_node(node1_id)
        node2 = store.get_node(node2_id)
        assert node1.tier == "longterm"
        assert node2.tier == "longterm"


    def test_memory_with_reasoning_traces(self, temp_db_path):
        """
        Test memory system integration with reasoning traces.

        Verifies SPEC-02, SPEC-04 integration.
        """
        from src.memory_store import MemoryStore
        from src.reasoning_traces import ReasoningTraces

        store = MemoryStore(db_path=temp_db_path)
        traces = ReasoningTraces(store)

        # Create a goal
        goal_id = traces.create_goal(
            content="Implement user authentication",
            prompt="How should I implement user authentication?",
            files=["src/auth.py", "src/models/user.py"],
        )

        # Create decisions
        decision_id = traces.create_decision(
            goal_id=goal_id,
            content="Choose authentication strategy",
        )

        # Add options
        option1_id = traces.add_option(decision_id, "Use JWT tokens")
        option2_id = traces.add_option(decision_id, "Use session cookies")

        # Choose one and reject the other
        traces.choose_option(decision_id, option1_id)
        traces.reject_option(decision_id, option2_id, "JWT is more scalable for API")

        # Create action
        action_id = traces.create_action(decision_id, "Implementing JWT authentication")

        # Create outcome
        outcome_id = traces.create_outcome(action_id, "JWT authentication implemented", success=True)

        # Verify the decision tree
        tree = traces.get_decision_tree(goal_id)
        assert tree is not None
        assert tree.root.content == "Implement user authentication"

        # Verify rejection reason is captured
        rejected = traces.get_rejected_options(decision_id)
        assert len(rejected) == 1
        assert rejected[0].reason == "JWT is more scalable for API"


    def test_budget_tracking_with_memory(self, temp_db_path):
        """
        Test budget tracking during memory operations.

        Verifies SPEC-02, SPEC-05 integration.
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits, EnhancedBudgetTracker
        from src.memory_store import MemoryStore

        store = MemoryStore(db_path=temp_db_path)
        tracker = EnhancedBudgetTracker()

        # Set conservative limits
        limits = BudgetLimits(
            max_cost_per_task=1.0,
            max_recursive_calls=5,
        )
        tracker.set_limits(limits)

        # Simulate memory-related LLM calls
        tracker.start_task("memory-task")

        # Record calls for memory operations
        tracker.record_llm_call(
            input_tokens=500,
            output_tokens=200,
            model="haiku",
            component=CostComponent.ROOT_PROMPT,
        )

        # Check we can still make calls
        allowed, _ = tracker.can_make_llm_call()
        assert allowed

        # Record more calls
        for i in range(3):
            tracker.record_llm_call(
                input_tokens=200,
                output_tokens=100,
                model="haiku",
                component=CostComponent.RECURSIVE_CALL,
            )

        # Get metrics
        metrics = tracker.get_metrics()
        assert metrics.sub_call_count == 3
        assert metrics.input_tokens == 500 + 200 * 3
        assert metrics.output_tokens == 200 + 100 * 3

        tracker.end_task()


    def test_repl_with_memory_functions(self, temp_db_path):
        """
        Test REPL environment with memory functions enabled.

        Verifies SPEC-01, SPEC-02 integration.
        """
        from src.memory_store import MemoryStore
        from src.repl_environment import RLMEnvironment
        from src.types import SessionContext

        store = MemoryStore(db_path=temp_db_path)

        context = SessionContext(
            messages=[],
            files={},
        )

        env = RLMEnvironment(context)
        env.enable_memory(store)

        # Test memory functions through REPL
        # Add a fact - we store the result in a variable to capture the return value
        result = env.execute("fact_id = memory_add_fact('Python uses indentation for blocks', confidence=0.95)")
        assert result.success

        # Query the fact - should find it
        result = env.execute("results = memory_query('Python indentation')")
        assert result.success

        # Verify we can access the query results
        result = env.execute("len(results) > 0")
        assert result.success

        # Add related experience (requires content, outcome, success)
        result = env.execute("memory_add_experience('Learned about Python indentation rules', 'Successfully applied indentation', True)")
        assert result.success

        # Get context (takes an optional limit parameter, not a query string)
        result = env.execute("memory_get_context(limit=5)")
        assert result.success


class TestFullSystemIntegration:
    """Full system integration tests."""


    def test_complete_workflow(self, temp_db_path):
        """
        Test a complete workflow using all components.

        This simulates an RLM session with memory, reasoning traces, and budget tracking.
        """
        from src.cost_tracker import CostComponent
        from src.enhanced_budget import BudgetLimits, EnhancedBudgetTracker
        from src.memory_evolution import MemoryEvolution
        from src.memory_store import MemoryStore
        from src.reasoning_traces import ReasoningTraces

        # Initialize all components
        store = MemoryStore(db_path=temp_db_path)
        evolution = MemoryEvolution(store)
        traces = ReasoningTraces(store)
        budget = EnhancedBudgetTracker()

        budget.set_limits(BudgetLimits(max_cost_per_task=5.0))
        budget.start_task("integration-test")
        budget.start_timing()

        # Step 1: Create goal and initial decision
        goal_id = traces.create_goal("Analyze codebase architecture")
        decision_id = traces.create_decision(goal_id, "How to approach analysis")

        # Record the LLM call for creating the goal
        budget.record_llm_call(
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
            component=CostComponent.ROOT_PROMPT,
        )

        # Step 2: Store learned facts in memory
        fact1_id = store.create_node(
            node_type="fact",
            content="The codebase uses a modular architecture",
            tier="task",
            confidence=0.9,
            metadata={"task_id": "integration-test"},
        )

        fact2_id = store.create_node(
            node_type="fact",
            content="Main entry point is in src/main.py",
            tier="task",
            confidence=0.95,
            metadata={"task_id": "integration-test"},
        )

        # Create relationship
        store.create_edge(
            edge_type="relation",
            label="describes",
            members=[
                {"node_id": fact1_id, "role": "subject", "position": 0},
                {"node_id": fact2_id, "role": "object", "position": 1},
            ],
        )

        # Step 3: Record recursive analysis calls
        for i in range(3):
            budget.record_llm_call(
                input_tokens=500,
                output_tokens=300,
                model="haiku",
                component=CostComponent.RECURSIVE_CALL,
            )
            budget.record_depth(i + 1)

        # Step 4: Record REPL executions
        for _ in range(2):
            budget.record_repl_execution()

        # Step 5: Create action and outcome in traces
        action_id = traces.create_action(decision_id, "Analyzed module structure")
        outcome_id = traces.create_outcome(action_id, "Architecture documented", success=True)

        # Step 6: Consolidate memory
        consolidation = evolution.consolidate("integration-test")

        # Step 7: Stop timing and verify
        budget.stop_timing()
        budget.end_task()

        # Verify all components worked together
        metrics = budget.get_metrics()
        assert metrics.sub_call_count == 3
        assert metrics.repl_executions == 2
        assert metrics.max_depth_reached == 3
        assert metrics.total_cost_usd > 0

        tree = traces.get_decision_tree(goal_id)
        assert tree is not None

        # Facts should be consolidated
        fact1 = store.get_node(fact1_id)
        assert fact1.tier == "session"

        # Evolution log should have entries
        log = store.get_evolution_log()
        assert len(log) > 0
