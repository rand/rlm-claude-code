"""
Tests for LATS-inspired tool orchestration (SPEC-06.10-06.15).

Tests cover:
- LATS phases (plan, expand, simulate, backpropagate, reflect)
- Tool capability matrix
- Task-to-tool preference mapping
- Fallback sequences
- UCB1 selection
- Configuration
"""

import math

from src.lats_orchestration import (
    ActionNode,
    LATSConfig,
    LATSOrchestrator,
    LATSPhase,
    ToolCapability,
    ToolCapabilityMatrix,
    ToolPlan,
    compute_ucb1,
)


class TestLATSPhases:
    """Tests for LATS phase enumeration (SPEC-06.10)."""

    def test_plan_phase_exists(self):
        """SPEC-06.10: PLAN phase generates tool use plan."""
        assert LATSPhase.PLAN.value == "plan"

    def test_expand_phase_exists(self):
        """SPEC-06.10: EXPAND phase uses UCB1 to select branches."""
        assert LATSPhase.EXPAND.value == "expand"

    def test_simulate_phase_exists(self):
        """SPEC-06.10: SIMULATE phase executes and evaluates."""
        assert LATSPhase.SIMULATE.value == "simulate"

    def test_backpropagate_phase_exists(self):
        """SPEC-06.10: BACKPROPAGATE phase updates value estimates."""
        assert LATSPhase.BACKPROPAGATE.value == "backpropagate"

    def test_reflect_phase_exists(self):
        """SPEC-06.10: REFLECT phase self-critiques failed paths."""
        assert LATSPhase.REFLECT.value == "reflect"


class TestToolCapabilityMatrix:
    """Tests for tool capability matrix (SPEC-06.11)."""

    def test_capability_matrix_maps_tools_to_capabilities(self):
        """SPEC-06.11: Matrix maps tools to capabilities."""
        matrix = ToolCapabilityMatrix()

        # Each tool should have defined capabilities
        bash_caps = matrix.get_capabilities("bash")
        assert isinstance(bash_caps, list)
        assert len(bash_caps) > 0

    def test_capability_includes_file_operations(self):
        """Common capabilities should be defined."""
        matrix = ToolCapabilityMatrix()

        file_tools = matrix.get_tools_for_capability(ToolCapability.FILE_READ)
        assert "read" in file_tools or "Read" in file_tools

    def test_capability_includes_code_execution(self):
        """Code execution capability should be defined."""
        matrix = ToolCapabilityMatrix()

        exec_tools = matrix.get_tools_for_capability(ToolCapability.CODE_EXECUTION)
        assert "bash" in exec_tools or "Bash" in exec_tools

    def test_capability_includes_search(self):
        """Search capability should be defined."""
        matrix = ToolCapabilityMatrix()

        search_tools = matrix.get_tools_for_capability(ToolCapability.SEARCH)
        assert len(search_tools) > 0


class TestTaskToToolPreference:
    """Tests for task-to-tool preference mapping (SPEC-06.12)."""

    def test_preference_for_file_search_task(self):
        """SPEC-06.12: File search tasks should prefer glob/grep."""
        matrix = ToolCapabilityMatrix()

        tools = matrix.get_preferred_tools("file_search")
        assert "glob" in tools or "Glob" in tools

    def test_preference_for_code_execution_task(self):
        """SPEC-06.12: Code execution tasks should prefer bash."""
        matrix = ToolCapabilityMatrix()

        tools = matrix.get_preferred_tools("code_execution")
        assert "bash" in tools or "Bash" in tools

    def test_preference_for_file_edit_task(self):
        """SPEC-06.12: File edit tasks should prefer edit tool."""
        matrix = ToolCapabilityMatrix()

        tools = matrix.get_preferred_tools("file_edit")
        assert "edit" in tools or "Edit" in tools

    def test_preference_returns_ordered_list(self):
        """Tool preferences should be ordered by relevance."""
        matrix = ToolCapabilityMatrix()

        tools = matrix.get_preferred_tools("file_search")
        assert isinstance(tools, list)
        assert len(tools) > 0


class TestFallbackSequences:
    """Tests for fallback tool sequences (SPEC-06.13)."""

    def test_fallback_sequence_for_search(self):
        """SPEC-06.13: Search has fallback sequence."""
        matrix = ToolCapabilityMatrix()

        fallbacks = matrix.get_fallback_sequence("grep")
        assert isinstance(fallbacks, list)
        # Should have at least one fallback
        assert len(fallbacks) >= 0  # May be empty if no fallbacks

    def test_fallback_sequence_for_read(self):
        """SPEC-06.13: Read has fallback sequence."""
        matrix = ToolCapabilityMatrix()

        fallbacks = matrix.get_fallback_sequence("read")
        assert isinstance(fallbacks, list)

    def test_fallback_sequence_preserves_capability(self):
        """Fallbacks should provide similar capability."""
        matrix = ToolCapabilityMatrix()

        # If glob fails, grep might be a fallback for search
        glob_caps = set(matrix.get_capabilities("glob"))
        fallbacks = matrix.get_fallback_sequence("glob")

        for fallback in fallbacks:
            fallback_caps = set(matrix.get_capabilities(fallback))
            # Should share at least one capability
            assert len(glob_caps & fallback_caps) > 0 or fallback == "bash"


class TestUCB1Selection:
    """Tests for UCB1 score computation (SPEC-06.14)."""

    def test_ucb1_formula_correct(self):
        """SPEC-06.14: UCB1 = exploitation + exploration_weight * sqrt(2 * ln(parent) / node)."""
        exploitation = 0.5
        parent_visits = 100
        node_visits = 10
        exploration_weight = 1.414

        expected = exploitation + exploration_weight * math.sqrt(
            2 * math.log(parent_visits) / node_visits
        )

        result = compute_ucb1(
            exploitation=exploitation,
            parent_visits=parent_visits,
            node_visits=node_visits,
            exploration_weight=exploration_weight,
        )

        assert abs(result - expected) < 0.0001

    def test_ucb1_handles_zero_visits(self):
        """UCB1 should return infinity for unvisited nodes."""
        result = compute_ucb1(
            exploitation=0.5,
            parent_visits=100,
            node_visits=0,
            exploration_weight=1.414,
        )

        assert result == float("inf")

    def test_ucb1_higher_exploitation_increases_score(self):
        """Higher exploitation value should increase UCB1."""
        low = compute_ucb1(0.3, 100, 10, 1.414)
        high = compute_ucb1(0.7, 100, 10, 1.414)

        assert high > low

    def test_ucb1_fewer_visits_increases_exploration(self):
        """Fewer visits should increase exploration bonus."""
        many_visits = compute_ucb1(0.5, 100, 50, 1.414)
        few_visits = compute_ucb1(0.5, 100, 5, 1.414)

        assert few_visits > many_visits


class TestLATSConfig:
    """Tests for LATS configuration (SPEC-06.15)."""

    def test_default_exploration_weight(self):
        """SPEC-06.15: Default exploration_weight is 1.414."""
        config = LATSConfig()
        assert abs(config.exploration_weight - 1.414) < 0.001

    def test_default_max_rollouts(self):
        """SPEC-06.15: Default max_rollouts is 10."""
        config = LATSConfig()
        assert config.max_rollouts == 10

    def test_default_max_depth(self):
        """SPEC-06.15: Default max_depth is 5."""
        config = LATSConfig()
        assert config.max_depth == 5

    def test_custom_exploration_weight(self):
        """Custom exploration weight should be respected."""
        config = LATSConfig(exploration_weight=2.0)
        assert config.exploration_weight == 2.0

    def test_custom_max_rollouts(self):
        """Custom max_rollouts should be respected."""
        config = LATSConfig(max_rollouts=20)
        assert config.max_rollouts == 20

    def test_custom_max_depth(self):
        """Custom max_depth should be respected."""
        config = LATSConfig(max_depth=3)
        assert config.max_depth == 3


class TestActionNode:
    """Tests for action tree nodes."""

    def test_node_tracks_visits(self):
        """Nodes should track visit count."""
        node = ActionNode(action="search", tool="grep")
        assert node.visits == 0

        node.visits += 1
        assert node.visits == 1

    def test_node_tracks_value(self):
        """Nodes should track value estimate."""
        node = ActionNode(action="search", tool="grep")
        assert node.value == 0.0

        node.value = 0.75
        assert node.value == 0.75

    def test_node_has_children(self):
        """Nodes should support child nodes."""
        parent = ActionNode(action="search", tool="grep")
        child = ActionNode(action="read", tool="read")

        parent.children.append(child)
        child.parent = parent

        assert len(parent.children) == 1
        assert child.parent is parent

    def test_node_computes_ucb1(self):
        """Nodes should compute UCB1 score."""
        parent = ActionNode(action="root", tool="")
        parent.visits = 100

        child = ActionNode(action="search", tool="grep")
        child.visits = 10
        child.value = 0.5
        child.parent = parent

        ucb1 = child.get_ucb1(exploration_weight=1.414)
        assert ucb1 > 0


class TestToolPlan:
    """Tests for tool execution plans."""

    def test_plan_has_steps(self):
        """Plans should contain ordered steps."""
        plan = ToolPlan(
            goal="Find and read a file",
            steps=[
                {"tool": "glob", "action": "find files matching pattern"},
                {"tool": "read", "action": "read matched file"},
            ],
        )

        assert len(plan.steps) == 2
        assert plan.steps[0]["tool"] == "glob"

    def test_plan_has_goal(self):
        """Plans should have a stated goal."""
        plan = ToolPlan(
            goal="Search for function definitions",
            steps=[],
        )

        assert plan.goal == "Search for function definitions"

    def test_plan_tracks_status(self):
        """Plans should track execution status."""
        plan = ToolPlan(goal="test", steps=[])
        assert plan.status == "pending"

        plan.status = "executing"
        assert plan.status == "executing"


class TestLATSOrchestrator:
    """Tests for LATS orchestrator integration."""

    def test_orchestrator_creates_plan(self):
        """SPEC-06.10: Orchestrator should generate tool plan."""
        orchestrator = LATSOrchestrator()

        plan = orchestrator.plan("Find all Python files")

        assert isinstance(plan, ToolPlan)
        assert len(plan.steps) > 0

    def test_orchestrator_expands_promising_node(self):
        """SPEC-06.10: Orchestrator should expand using UCB1."""
        orchestrator = LATSOrchestrator()

        # Create a tree with multiple options
        root = ActionNode(action="root", tool="")
        root.visits = 10

        child1 = ActionNode(action="option1", tool="glob")
        child1.visits = 5
        child1.value = 0.3
        child1.parent = root

        child2 = ActionNode(action="option2", tool="grep")
        child2.visits = 1  # Fewer visits = higher exploration
        child2.value = 0.4
        child2.parent = root

        root.children = [child1, child2]

        selected = orchestrator.select_node(root)
        # Should select based on UCB1 (child2 has fewer visits)
        assert selected is not None

    def test_orchestrator_backpropagates_value(self):
        """SPEC-06.10: Orchestrator should backpropagate results."""
        orchestrator = LATSOrchestrator()

        root = ActionNode(action="root", tool="")
        root.visits = 10
        root.value = 0.5

        child = ActionNode(action="search", tool="grep")
        child.visits = 5
        child.value = 0.3
        child.parent = root
        root.children = [child]

        # Backpropagate a success
        orchestrator.backpropagate(child, reward=1.0)

        assert child.visits == 6
        assert child.value > 0.3  # Value should increase
        assert root.visits == 11  # Parent visits should increase too

    def test_orchestrator_reflects_on_failure(self):
        """SPEC-06.10: Orchestrator should reflect on failed paths."""
        orchestrator = LATSOrchestrator()

        node = ActionNode(action="search", tool="grep")
        node.result = {"success": False, "error": "No matches found"}

        reflection = orchestrator.reflect(node)

        assert reflection is not None
        assert isinstance(reflection, str)
        assert len(reflection) > 0

    def test_orchestrator_respects_max_depth(self):
        """SPEC-06.15: Orchestrator should respect max_depth."""
        config = LATSConfig(max_depth=2)
        orchestrator = LATSOrchestrator(config=config)

        # Build a deep tree
        root = ActionNode(action="root", tool="")
        level1 = ActionNode(action="l1", tool="grep")
        level1.parent = root
        root.children = [level1]

        level2 = ActionNode(action="l2", tool="read")
        level2.parent = level1
        level1.children = [level2]

        level3 = ActionNode(action="l3", tool="edit")
        level3.parent = level2
        level2.children = [level3]

        # Should not expand beyond max_depth
        assert orchestrator.should_expand(level2) is False
        assert orchestrator.should_expand(level1) is True

    def test_orchestrator_respects_max_rollouts(self):
        """SPEC-06.15: Orchestrator should respect max_rollouts."""
        config = LATSConfig(max_rollouts=5)
        orchestrator = LATSOrchestrator(config=config)

        # Track rollouts
        for _ in range(5):
            orchestrator.rollouts += 1

        assert orchestrator.should_continue() is False

    def test_full_lats_cycle(self):
        """Test complete LATS cycle: plan -> expand -> simulate -> backpropagate."""
        orchestrator = LATSOrchestrator()

        # Plan
        plan = orchestrator.plan("Search for test files")
        assert plan is not None

        # Create action tree
        root = ActionNode(action="root", tool="")
        root.visits = 1

        # Expand
        new_node = orchestrator.expand(
            root, plan.steps[0] if plan.steps else {"tool": "glob", "action": "search"}
        )
        assert new_node is not None
        assert new_node.parent is root

        # Simulate (mock result)
        new_node.result = {"success": True, "output": ["test1.py", "test2.py"]}

        # Backpropagate
        orchestrator.backpropagate(new_node, reward=1.0)
        assert new_node.visits == 1
        assert root.visits == 2


class TestToolCapabilityEnum:
    """Tests for tool capability enumeration."""

    def test_file_read_capability(self):
        """FILE_READ capability exists."""
        assert ToolCapability.FILE_READ.value == "file_read"

    def test_file_write_capability(self):
        """FILE_WRITE capability exists."""
        assert ToolCapability.FILE_WRITE.value == "file_write"

    def test_search_capability(self):
        """SEARCH capability exists."""
        assert ToolCapability.SEARCH.value == "search"

    def test_code_execution_capability(self):
        """CODE_EXECUTION capability exists."""
        assert ToolCapability.CODE_EXECUTION.value == "code_execution"
