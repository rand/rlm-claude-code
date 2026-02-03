"""
Tests for Tree of Thoughts integration (SPEC-07.01-07.07).

Tests cover:
- ThoughtNode structure
- Branching and backtracking
- Search strategies
- Configuration options
"""

from src.tree_of_thoughts import (
    SearchStrategy,
    ThoughtNode,
    ThoughtTree,
    ToTConfig,
)


class TestThoughtNode:
    """Tests for ThoughtNode structure (SPEC-07.01)."""

    def test_thought_node_has_required_fields(self):
        """SPEC-07.01: ThoughtNode SHALL have thought, state, children, value_estimate, is_terminal."""
        node = ThoughtNode(
            thought="Analyze the problem",
            state={"step": 1, "data": [1, 2, 3]},
            value_estimate=0.8,
            is_terminal=False,
        )

        assert node.thought == "Analyze the problem"
        assert node.state == {"step": 1, "data": [1, 2, 3]}
        assert node.children == []
        assert node.value_estimate == 0.8
        assert node.is_terminal is False

    def test_thought_node_default_children_empty(self):
        """Children should default to empty list."""
        node = ThoughtNode(
            thought="Initial thought",
            state={},
        )
        assert node.children == []

    def test_thought_node_default_value_estimate(self):
        """Value estimate should default to 0.0."""
        node = ThoughtNode(
            thought="Initial thought",
            state={},
        )
        assert node.value_estimate == 0.0

    def test_thought_node_default_not_terminal(self):
        """is_terminal should default to False."""
        node = ThoughtNode(
            thought="Initial thought",
            state={},
        )
        assert node.is_terminal is False

    def test_thought_node_with_children(self):
        """Node can have child nodes."""
        child1 = ThoughtNode(thought="Child 1", state={"c": 1})
        child2 = ThoughtNode(thought="Child 2", state={"c": 2})
        parent = ThoughtNode(
            thought="Parent",
            state={"p": 0},
            children=[child1, child2],
        )

        assert len(parent.children) == 2
        assert parent.children[0].thought == "Child 1"
        assert parent.children[1].thought == "Child 2"

    def test_thought_node_depth_calculation(self):
        """Node should be able to calculate its depth in tree."""
        root = ThoughtNode(thought="Root", state={})
        child = ThoughtNode(thought="Child", state={}, parent=root)
        grandchild = ThoughtNode(thought="Grandchild", state={}, parent=child)

        assert root.depth() == 0
        assert child.depth() == 1
        assert grandchild.depth() == 2


class TestThoughtNodeState:
    """Tests for REPL state preservation (SPEC-07.05)."""

    def test_state_is_copied_not_referenced(self):
        """State should be copied to prevent mutation."""
        original_state = {"x": 1, "data": [1, 2, 3]}
        node = ThoughtNode(thought="Test", state=original_state)

        # Modifying original should not affect node
        original_state["x"] = 999
        original_state["data"].append(4)

        assert node.state["x"] == 1
        assert node.state["data"] == [1, 2, 3]

    def test_get_state_returns_copy(self):
        """get_state should return a copy."""
        node = ThoughtNode(thought="Test", state={"x": 1})
        state = node.get_state()

        # Modifying returned state should not affect node
        state["x"] = 999

        assert node.state["x"] == 1


class TestBranching:
    """Tests for thought branching (SPEC-07.02)."""

    def test_branch_creates_child_nodes(self):
        """SPEC-07.02: branch(thoughts) creates child nodes."""
        tree = ThoughtTree(ToTConfig())
        root = ThoughtNode(thought="Initial", state={"step": 0})

        children = tree.branch(
            root,
            thoughts=[
                "Approach A: Use recursion",
                "Approach B: Use iteration",
                "Approach C: Use memoization",
            ],
        )

        assert len(children) == 3
        assert children[0].thought == "Approach A: Use recursion"
        assert children[1].thought == "Approach B: Use iteration"
        assert children[2].thought == "Approach C: Use memoization"

    def test_branch_links_to_parent(self):
        """Children should have parent reference for backtracking."""
        tree = ThoughtTree(ToTConfig())
        root = ThoughtNode(thought="Initial", state={"step": 0})

        children = tree.branch(root, thoughts=["Child 1", "Child 2"])

        assert children[0].parent is root
        assert children[1].parent is root

    def test_branch_adds_to_parent_children(self):
        """Branch should add children to parent node."""
        tree = ThoughtTree(ToTConfig())
        root = ThoughtNode(thought="Initial", state={"step": 0})

        tree.branch(root, thoughts=["Child 1", "Child 2"])

        assert len(root.children) == 2
        assert root.children[0].thought == "Child 1"

    def test_branch_inherits_state(self):
        """Children inherit parent state by default."""
        tree = ThoughtTree(ToTConfig())
        root = ThoughtNode(thought="Initial", state={"x": 1, "y": 2})

        children = tree.branch(root, thoughts=["Child 1"])

        # Child should have copy of parent state
        assert children[0].state["x"] == 1
        assert children[0].state["y"] == 2

    def test_branch_with_state_updates(self):
        """Branch can provide state updates for children."""
        tree = ThoughtTree(ToTConfig())
        root = ThoughtNode(thought="Initial", state={"step": 0})

        children = tree.branch(
            root,
            thoughts=["Child 1", "Child 2"],
            state_updates=[{"step": 1, "choice": "A"}, {"step": 1, "choice": "B"}],
        )

        assert children[0].state["step"] == 1
        assert children[0].state["choice"] == "A"
        assert children[1].state["step"] == 1
        assert children[1].state["choice"] == "B"

    def test_branch_respects_max_branches(self):
        """SPEC-07.07: Branch should respect max_branches config."""
        config = ToTConfig(max_branches=2)
        tree = ThoughtTree(config)
        root = ThoughtNode(thought="Initial", state={})

        # Try to create 4 branches, should be limited to 2
        children = tree.branch(
            root,
            thoughts=["A", "B", "C", "D"],
        )

        assert len(children) == 2


class TestStateEvaluation:
    """Tests for state evaluation (SPEC-07.03)."""

    def test_evaluate_state_returns_float(self):
        """SPEC-07.03: evaluate_state returns float value estimate."""
        tree = ThoughtTree(ToTConfig())
        node = ThoughtNode(thought="Solution found", state={"solved": True})

        value = tree.evaluate_state(node)

        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_evaluate_state_updates_node_value(self):
        """Evaluation should update node's value_estimate."""
        tree = ThoughtTree(ToTConfig())
        node = ThoughtNode(thought="Promising approach", state={})

        tree.evaluate_state(node)

        assert node.value_estimate >= 0.0

    def test_terminal_state_evaluation(self):
        """Terminal states should be evaluated appropriately."""
        tree = ThoughtTree(ToTConfig())
        terminal_node = ThoughtNode(
            thought="Found the answer: 42",
            state={"answer": 42},
            is_terminal=True,
        )

        value = tree.evaluate_state(terminal_node)

        assert value >= 0.0


class TestBacktracking:
    """Tests for backtracking (SPEC-07.04, SPEC-07.05)."""

    def test_backtrack_returns_to_node(self):
        """SPEC-07.04: backtrack restores to specified node."""
        tree = ThoughtTree(ToTConfig())
        root = ThoughtNode(thought="Root", state={"step": 0})
        children = tree.branch(root, thoughts=["Child"])
        grandchildren = tree.branch(children[0], thoughts=["Grandchild"])

        tree.set_current(grandchildren[0])
        tree.backtrack(to_node=root)

        assert tree.current_node is root

    def test_backtrack_restores_state(self):
        """SPEC-07.05: backtrack preserves REPL state at branch points."""
        tree = ThoughtTree(ToTConfig())

        root_state = {"x": 1, "data": [1, 2, 3]}
        root = ThoughtNode(thought="Root", state=root_state)

        children = tree.branch(
            root,
            thoughts=["Child"],
            state_updates=[{"x": 100, "data": [4, 5, 6]}],
        )

        tree.set_current(children[0])
        tree.backtrack(to_node=root)

        # State should be restored to root's state
        assert tree.get_current_state()["x"] == 1
        assert tree.get_current_state()["data"] == [1, 2, 3]

    def test_backtrack_to_parent(self):
        """Backtrack without target goes to parent."""
        tree = ThoughtTree(ToTConfig())
        root = ThoughtNode(thought="Root", state={})
        children = tree.branch(root, thoughts=["Child"])

        tree.set_current(children[0])
        tree.backtrack()

        assert tree.current_node is root

    def test_backtrack_at_root_does_nothing(self):
        """Backtrack at root should not fail."""
        tree = ThoughtTree(ToTConfig())
        root = ThoughtNode(thought="Root", state={"x": 1})

        tree.set_current(root)
        tree.backtrack()

        assert tree.current_node is root


class TestSearchStrategies:
    """Tests for search strategies (SPEC-07.06)."""

    def test_search_strategy_enum(self):
        """SPEC-07.06: System shall support configurable search strategies."""
        assert SearchStrategy.BFS.value == "bfs"
        assert SearchStrategy.DFS.value == "dfs"
        assert SearchStrategy.BEST_FIRST.value == "best_first"

    def test_bfs_search_order(self):
        """BFS explores breadth-first."""
        config = ToTConfig(search_strategy=SearchStrategy.BFS)
        tree = ThoughtTree(config)
        root = tree.create_root(thought="Root", state={})

        # Create tree structure
        level1 = tree.branch(root, thoughts=["L1-A", "L1-B"])
        tree.branch(level1[0], thoughts=["L2-A1", "L2-A2"])
        tree.branch(level1[1], thoughts=["L2-B1", "L2-B2"])

        # Get nodes in exploration order
        order = list(tree.iter_nodes(strategy=SearchStrategy.BFS))
        thoughts = [n.thought for n in order]

        # BFS: Root -> L1-A -> L1-B -> L2-*
        assert thoughts.index("L1-A") < thoughts.index("L2-A1")
        assert thoughts.index("L1-B") < thoughts.index("L2-B1")

    def test_dfs_search_order(self):
        """DFS explores depth-first."""
        config = ToTConfig(search_strategy=SearchStrategy.DFS)
        tree = ThoughtTree(config)
        root = tree.create_root(thought="Root", state={})

        # Create tree structure
        level1 = tree.branch(root, thoughts=["L1-A", "L1-B"])
        tree.branch(level1[0], thoughts=["L2-A1"])
        tree.branch(level1[1], thoughts=["L2-B1"])

        # Get nodes in exploration order
        order = list(tree.iter_nodes(strategy=SearchStrategy.DFS))
        thoughts = [n.thought for n in order]

        # DFS: Root -> L1-A -> L2-A1 -> L1-B -> L2-B1
        assert thoughts.index("L2-A1") < thoughts.index("L1-B")

    def test_best_first_search_order(self):
        """Best-first explores by value estimate."""
        config = ToTConfig(search_strategy=SearchStrategy.BEST_FIRST)
        tree = ThoughtTree(config)
        root = ThoughtNode(thought="Root", state={}, value_estimate=0.5)

        # Create children with different values
        child_low = ThoughtNode(thought="Low", state={}, value_estimate=0.2, parent=root)
        child_high = ThoughtNode(thought="High", state={}, value_estimate=0.9, parent=root)
        child_mid = ThoughtNode(thought="Mid", state={}, value_estimate=0.5, parent=root)
        root.children = [child_low, child_high, child_mid]

        # Get unexplored children in best-first order
        unexplored = tree.get_unexplored_children(root, strategy=SearchStrategy.BEST_FIRST)

        # Should be ordered by value: High, Mid, Low
        assert unexplored[0].thought == "High"
        assert unexplored[1].thought == "Mid"
        assert unexplored[2].thought == "Low"


class TestToTConfig:
    """Tests for configuration (SPEC-07.07)."""

    def test_default_max_branches(self):
        """SPEC-07.07: max_branches default is 3."""
        config = ToTConfig()
        assert config.max_branches == 3

    def test_default_max_depth(self):
        """SPEC-07.07: max_depth default is 4."""
        config = ToTConfig()
        assert config.max_depth == 4

    def test_default_pruning_threshold(self):
        """SPEC-07.07: pruning_threshold default is 0.3."""
        config = ToTConfig()
        assert config.pruning_threshold == 0.3

    def test_custom_config(self):
        """Config should accept custom values."""
        config = ToTConfig(
            max_branches=5,
            max_depth=6,
            pruning_threshold=0.5,
            search_strategy=SearchStrategy.BEST_FIRST,
        )

        assert config.max_branches == 5
        assert config.max_depth == 6
        assert config.pruning_threshold == 0.5
        assert config.search_strategy == SearchStrategy.BEST_FIRST


class TestTreeDepthLimits:
    """Tests for depth limiting (SPEC-07.07)."""

    def test_branch_respects_max_depth(self):
        """Branch should not exceed max_depth."""
        config = ToTConfig(max_depth=2)
        tree = ThoughtTree(config)
        root = ThoughtNode(thought="Root", state={})

        # Depth 0 -> 1
        level1 = tree.branch(root, thoughts=["L1"])
        # Depth 1 -> 2
        level2 = tree.branch(level1[0], thoughts=["L2"])
        # Depth 2 -> 3 should be empty (exceeds max_depth)
        level3 = tree.branch(level2[0], thoughts=["L3"])

        assert len(level3) == 0

    def test_can_branch_returns_false_at_max_depth(self):
        """can_branch should return False at max depth."""
        config = ToTConfig(max_depth=1)
        tree = ThoughtTree(config)
        root = ThoughtNode(thought="Root", state={})
        level1 = tree.branch(root, thoughts=["L1"])

        assert tree.can_branch(root) is True
        assert tree.can_branch(level1[0]) is False


class TestPruning:
    """Tests for pruning (SPEC-07.07)."""

    def test_prune_low_value_nodes(self):
        """Nodes below pruning_threshold should be prunable."""
        config = ToTConfig(pruning_threshold=0.3)
        tree = ThoughtTree(config)

        low_node = ThoughtNode(thought="Low value", state={}, value_estimate=0.1)
        high_node = ThoughtNode(thought="High value", state={}, value_estimate=0.8)

        assert tree.should_prune(low_node) is True
        assert tree.should_prune(high_node) is False

    def test_pruning_threshold_boundary(self):
        """Node exactly at threshold should not be pruned."""
        config = ToTConfig(pruning_threshold=0.3)
        tree = ThoughtTree(config)

        boundary_node = ThoughtNode(thought="Boundary", state={}, value_estimate=0.3)

        assert tree.should_prune(boundary_node) is False


class TestThoughtTreeIntegration:
    """Integration tests for complete ToT workflow."""

    def test_complete_tot_workflow(self):
        """Test full ToT exploration workflow."""
        config = ToTConfig(
            max_branches=2,
            max_depth=3,
            search_strategy=SearchStrategy.BFS,
        )
        tree = ThoughtTree(config)

        # Initialize with root
        root = tree.create_root(
            thought="Solve the problem",
            state={"problem": "2 + 2"},
        )

        # Branch into approaches
        approaches = tree.branch(
            root,
            thoughts=["Use arithmetic", "Use lookup table"],
        )

        # Evaluate branches
        for approach in approaches:
            tree.evaluate_state(approach)

        # Continue best branch
        best = max(approaches, key=lambda n: n.value_estimate)
        solutions = tree.branch(
            best,
            thoughts=["Calculate: 2 + 2 = 4"],
            state_updates=[{"answer": 4}],
        )

        # Mark terminal
        solutions[0].is_terminal = True
        tree.evaluate_state(solutions[0])

        # Get best terminal
        terminal = tree.get_best_terminal()
        assert terminal is not None
        assert terminal.state["answer"] == 4

    def test_tot_with_backtracking(self):
        """Test ToT with backtracking on dead end."""
        config = ToTConfig(max_branches=2, max_depth=3)
        tree = ThoughtTree(config)

        root = tree.create_root(thought="Start", state={"attempts": 0})

        # Try first approach
        approaches = tree.branch(root, thoughts=["Approach A", "Approach B"])
        tree.set_current(approaches[0])

        # Approach A leads to dead end
        dead_end = tree.branch(
            approaches[0],
            thoughts=["Dead end"],
            state_updates=[{"dead": True}],
        )
        dead_end[0].value_estimate = 0.1
        dead_end[0].is_terminal = True

        # Backtrack and try Approach B
        tree.backtrack(to_node=root)
        tree.set_current(approaches[1])

        # Approach B succeeds
        success = tree.branch(
            approaches[1],
            thoughts=["Success!"],
            state_updates=[{"solved": True}],
        )
        success[0].value_estimate = 0.95
        success[0].is_terminal = True

        # Best terminal should be success
        best = tree.get_best_terminal()
        assert best is not None
        assert best.state.get("solved") is True
