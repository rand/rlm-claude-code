"""
Tree of Thoughts integration for RLM.

Implements: SPEC-07.01-07.07

Provides structured exploration of reasoning paths with:
- ThoughtNode for tracking state
- Branching and backtracking
- Configurable search strategies
- Pruning based on value estimates
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchStrategy(Enum):
    """
    Search strategies for tree exploration.

    Implements: SPEC-07.06
    """

    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search
    BEST_FIRST = "best_first"  # Best-first by value estimate


@dataclass
class ToTConfig:
    """
    Configuration for Tree of Thoughts.

    Implements: SPEC-07.07
    """

    max_branches: int = 3
    max_depth: int = 4
    pruning_threshold: float = 0.3
    search_strategy: SearchStrategy = SearchStrategy.BFS


@dataclass
class ThoughtNode:
    """
    A node in the Tree of Thoughts.

    Implements: SPEC-07.01

    Represents a single thought/reasoning step with:
    - thought: The reasoning text
    - state: REPL/context state at this point
    - children: Child thoughts (alternatives explored)
    - value_estimate: Estimated quality (0.0 to 1.0)
    - is_terminal: Whether this is a terminal/solution state
    """

    thought: str
    state: dict[str, Any]
    children: list[ThoughtNode] = field(default_factory=list)
    value_estimate: float = 0.0
    is_terminal: bool = False
    parent: ThoughtNode | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Copy state to prevent mutation."""
        self.state = copy.deepcopy(self.state)

    def get_state(self) -> dict[str, Any]:
        """
        Get a copy of the node's state.

        Implements: SPEC-07.05 (state preservation)

        Returns:
            Deep copy of state dictionary
        """
        return copy.deepcopy(self.state)

    def depth(self) -> int:
        """
        Calculate depth of this node in the tree.

        Returns:
            Depth (root = 0)
        """
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth


class ThoughtTree:
    """
    Tree of Thoughts manager.

    Implements: SPEC-07.01-07.07

    Provides:
    - Thought branching (SPEC-07.02)
    - State evaluation (SPEC-07.03)
    - Backtracking (SPEC-07.04)
    - Search strategies (SPEC-07.06)
    - Configurable limits (SPEC-07.07)
    """

    def __init__(self, config: ToTConfig | None = None):
        """
        Initialize ThoughtTree.

        Args:
            config: Configuration options
        """
        self.config = config or ToTConfig()
        self.root: ThoughtNode | None = None
        self.current_node: ThoughtNode | None = None

    def create_root(self, thought: str, state: dict[str, Any]) -> ThoughtNode:
        """
        Create and set the root node.

        Args:
            thought: Initial thought/problem statement
            state: Initial REPL state

        Returns:
            Root ThoughtNode
        """
        self.root = ThoughtNode(thought=thought, state=state)
        self.current_node = self.root
        return self.root

    def branch(
        self,
        parent: ThoughtNode,
        thoughts: list[str],
        state_updates: list[dict[str, Any]] | None = None,
    ) -> list[ThoughtNode]:
        """
        Create child nodes for alternative thoughts.

        Implements: SPEC-07.02

        Args:
            parent: Parent node to branch from
            thoughts: List of alternative thoughts
            state_updates: Optional state modifications for each branch

        Returns:
            List of created child nodes
        """
        # Check depth limit (SPEC-07.07)
        if parent.depth() >= self.config.max_depth:
            return []

        # Limit branches (SPEC-07.07)
        limited_thoughts = thoughts[: self.config.max_branches]

        children: list[ThoughtNode] = []
        for i, thought in enumerate(limited_thoughts):
            # Start with parent's state
            child_state = parent.get_state()

            # Apply state updates if provided
            if state_updates and i < len(state_updates):
                child_state.update(state_updates[i])

            child = ThoughtNode(
                thought=thought,
                state=child_state,
                parent=parent,
            )
            children.append(child)
            parent.children.append(child)

        return children

    def evaluate_state(self, node: ThoughtNode) -> float:
        """
        Evaluate the quality/promise of a thought node.

        Implements: SPEC-07.03

        This is a placeholder that returns a simple heuristic.
        In practice, this would use Claude to evaluate the thought.

        Args:
            node: Node to evaluate

        Returns:
            Value estimate between 0.0 and 1.0
        """
        # Simple heuristic based on state and thought content
        value = 0.5  # Base value

        # Terminal nodes with solution indicators get higher value
        if node.is_terminal:
            if node.state.get("solved") or node.state.get("answer") is not None:
                value = 0.9
            elif node.state.get("dead"):
                value = 0.1

        # Deeper thoughts that haven't failed are promising
        if not node.is_terminal and node.depth() > 0:
            value = min(0.7, 0.5 + node.depth() * 0.1)

        node.value_estimate = value
        return value

    def backtrack(self, to_node: ThoughtNode | None = None) -> None:
        """
        Backtrack to a previous node.

        Implements: SPEC-07.04

        Args:
            to_node: Target node to return to (defaults to parent)
        """
        if self.current_node is None:
            return

        if to_node is not None:
            self.current_node = to_node
        elif self.current_node.parent is not None:
            self.current_node = self.current_node.parent

    def set_current(self, node: ThoughtNode) -> None:
        """
        Set the current exploration node.

        Args:
            node: Node to set as current
        """
        self.current_node = node

    def get_current_state(self) -> dict[str, Any]:
        """
        Get the current node's state.

        Implements: SPEC-07.05

        Returns:
            Copy of current state
        """
        if self.current_node is None:
            return {}
        return self.current_node.get_state()

    def can_branch(self, node: ThoughtNode) -> bool:
        """
        Check if node can be branched.

        Args:
            node: Node to check

        Returns:
            True if branching is allowed
        """
        return node.depth() < self.config.max_depth

    def should_prune(self, node: ThoughtNode) -> bool:
        """
        Check if node should be pruned.

        Implements: SPEC-07.07 (pruning_threshold)

        Args:
            node: Node to check

        Returns:
            True if node should be pruned
        """
        return node.value_estimate < self.config.pruning_threshold

    def iter_nodes(
        self,
        strategy: SearchStrategy | None = None,
    ) -> Iterator[ThoughtNode]:
        """
        Iterate over nodes in specified order.

        Implements: SPEC-07.06

        Args:
            strategy: Search strategy (defaults to config)

        Yields:
            Nodes in exploration order
        """
        if self.root is None:
            return

        strategy = strategy or self.config.search_strategy

        if strategy == SearchStrategy.BFS:
            yield from self._bfs_iter()
        elif strategy == SearchStrategy.DFS:
            yield from self._dfs_iter()
        elif strategy == SearchStrategy.BEST_FIRST:
            yield from self._best_first_iter()

    def _bfs_iter(self) -> Iterator[ThoughtNode]:
        """Breadth-first iteration."""
        if self.root is None:
            return

        queue: list[ThoughtNode] = [self.root]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    def _dfs_iter(self) -> Iterator[ThoughtNode]:
        """Depth-first iteration."""
        if self.root is None:
            return

        stack: list[ThoughtNode] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            # Add children in reverse order so first child is processed first
            stack.extend(reversed(node.children))

    def _best_first_iter(self) -> Iterator[ThoughtNode]:
        """Best-first iteration by value estimate."""
        if self.root is None:
            return

        # Use list as priority queue (simple implementation)
        queue: list[ThoughtNode] = [self.root]
        while queue:
            # Sort by value estimate descending
            queue.sort(key=lambda n: n.value_estimate, reverse=True)
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    def get_unexplored_children(
        self,
        node: ThoughtNode,
        strategy: SearchStrategy | None = None,
    ) -> list[ThoughtNode]:
        """
        Get children ordered by exploration strategy.

        Args:
            node: Parent node
            strategy: Search strategy

        Returns:
            Children in exploration order
        """
        strategy = strategy or self.config.search_strategy

        if strategy == SearchStrategy.BEST_FIRST:
            return sorted(
                node.children,
                key=lambda n: n.value_estimate,
                reverse=True,
            )
        else:
            # BFS and DFS use natural order
            return list(node.children)

    def get_best_terminal(self) -> ThoughtNode | None:
        """
        Get the best terminal node found.

        Returns:
            Terminal node with highest value, or None
        """
        best: ThoughtNode | None = None
        best_value = -1.0

        for node in self.iter_nodes():
            if node.is_terminal and node.value_estimate > best_value:
                best = node
                best_value = node.value_estimate

        return best

    def get_all_terminals(self) -> list[ThoughtNode]:
        """
        Get all terminal nodes.

        Returns:
            List of terminal nodes
        """
        return [node for node in self.iter_nodes() if node.is_terminal]


__all__ = [
    "SearchStrategy",
    "ThoughtNode",
    "ThoughtTree",
    "ToTConfig",
]
