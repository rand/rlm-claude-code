"""
LATS-inspired tool orchestration.

Implements: SPEC-06.10-06.15

Language Agent Tree Search (LATS) approach for intelligent tool selection
and execution with exploration/exploitation balance via UCB1.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LATSPhase(Enum):
    """
    LATS execution phases.

    Implements: SPEC-06.10
    """

    PLAN = "plan"
    EXPAND = "expand"
    SIMULATE = "simulate"
    BACKPROPAGATE = "backpropagate"
    REFLECT = "reflect"


class ToolCapability(Enum):
    """
    Capabilities that tools can provide.

    Implements: SPEC-06.11
    """

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    SEARCH = "search"
    CODE_EXECUTION = "code_execution"
    WEB_FETCH = "web_fetch"
    DIRECTORY_LIST = "directory_list"
    PATTERN_MATCH = "pattern_match"


# Tool capability mapping
TOOL_CAPABILITIES: dict[str, list[ToolCapability]] = {
    "read": [ToolCapability.FILE_READ],
    "Read": [ToolCapability.FILE_READ],
    "write": [ToolCapability.FILE_WRITE],
    "Write": [ToolCapability.FILE_WRITE],
    "edit": [ToolCapability.FILE_EDIT, ToolCapability.FILE_WRITE],
    "Edit": [ToolCapability.FILE_EDIT, ToolCapability.FILE_WRITE],
    "bash": [ToolCapability.CODE_EXECUTION, ToolCapability.FILE_READ, ToolCapability.SEARCH],
    "Bash": [ToolCapability.CODE_EXECUTION, ToolCapability.FILE_READ, ToolCapability.SEARCH],
    "glob": [ToolCapability.PATTERN_MATCH, ToolCapability.SEARCH, ToolCapability.DIRECTORY_LIST],
    "Glob": [ToolCapability.PATTERN_MATCH, ToolCapability.SEARCH, ToolCapability.DIRECTORY_LIST],
    "grep": [ToolCapability.SEARCH, ToolCapability.PATTERN_MATCH],
    "Grep": [ToolCapability.SEARCH, ToolCapability.PATTERN_MATCH],
    "webfetch": [ToolCapability.WEB_FETCH],
    "WebFetch": [ToolCapability.WEB_FETCH],
    "ls": [ToolCapability.DIRECTORY_LIST],
    "LS": [ToolCapability.DIRECTORY_LIST],
}

# Task to tool preference mapping
TASK_TOOL_PREFERENCES: dict[str, list[str]] = {
    "file_search": ["glob", "grep", "bash"],
    "content_search": ["grep", "glob", "bash"],
    "file_read": ["read", "bash"],
    "file_edit": ["edit", "write"],
    "file_write": ["write", "edit"],
    "code_execution": ["bash"],
    "directory_list": ["glob", "ls", "bash"],
    "web_fetch": ["webfetch"],
    "pattern_match": ["glob", "grep"],
}

# Fallback sequences when primary tool fails
FALLBACK_SEQUENCES: dict[str, list[str]] = {
    "glob": ["grep", "bash"],
    "grep": ["glob", "bash"],
    "read": ["bash"],
    "edit": ["write", "bash"],
    "write": ["bash"],
    "webfetch": [],
}


def compute_ucb1(
    exploitation: float,
    parent_visits: int,
    node_visits: int,
    exploration_weight: float = 1.414,
) -> float:
    """
    Compute UCB1 score for node selection.

    Implements: SPEC-06.14

    UCB1 = exploitation + exploration_weight * sqrt(2 * ln(parent_visits) / node_visits)

    Args:
        exploitation: The average value/reward of this node
        parent_visits: Number of times parent was visited
        node_visits: Number of times this node was visited
        exploration_weight: Weight for exploration term (default: sqrt(2))

    Returns:
        UCB1 score (infinity if node_visits is 0)
    """
    if node_visits == 0:
        return float("inf")

    exploration = exploration_weight * math.sqrt(2 * math.log(parent_visits) / node_visits)

    return exploitation + exploration


@dataclass
class LATSConfig:
    """
    Configuration for LATS orchestration.

    Implements: SPEC-06.15
    """

    exploration_weight: float = 1.414  # sqrt(2)
    max_rollouts: int = 10
    max_depth: int = 5
    reflection_enabled: bool = True
    fallback_enabled: bool = True


@dataclass
class ActionNode:
    """
    Node in the action tree for LATS exploration.

    Represents a tool action that can be expanded and evaluated.
    """

    action: str
    tool: str
    visits: int = 0
    value: float = 0.0
    children: list[ActionNode] = field(default_factory=list)
    parent: ActionNode | None = field(default=None, repr=False)
    result: dict[str, Any] | None = None
    reflection: str | None = None

    def depth(self) -> int:
        """Get depth of this node in the tree."""
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def get_ucb1(self, exploration_weight: float = 1.414) -> float:
        """Compute UCB1 score for this node."""
        if self.parent is None:
            return 0.0

        return compute_ucb1(
            exploitation=self.value,
            parent_visits=self.parent.visits,
            node_visits=self.visits,
            exploration_weight=exploration_weight,
        )


@dataclass
class ToolPlan:
    """
    A plan for executing tools to achieve a goal.

    Implements: SPEC-06.10 (PLAN phase output)
    """

    goal: str
    steps: list[dict[str, str]]
    status: str = "pending"
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "goal": self.goal,
            "steps": self.steps,
            "status": self.status,
            "reasoning": self.reasoning,
        }


class ToolCapabilityMatrix:
    """
    Matrix mapping tools to capabilities.

    Implements: SPEC-06.11, SPEC-06.12, SPEC-06.13
    """

    def __init__(
        self,
        capabilities: dict[str, list[ToolCapability]] | None = None,
        preferences: dict[str, list[str]] | None = None,
        fallbacks: dict[str, list[str]] | None = None,
    ):
        """
        Initialize capability matrix.

        Args:
            capabilities: Custom tool-to-capability mapping
            preferences: Custom task-to-tool preferences
            fallbacks: Custom fallback sequences
        """
        self.capabilities = capabilities or TOOL_CAPABILITIES
        self.preferences = preferences or TASK_TOOL_PREFERENCES
        self.fallbacks = fallbacks or FALLBACK_SEQUENCES

    def get_capabilities(self, tool: str) -> list[ToolCapability]:
        """
        Get capabilities for a tool.

        Implements: SPEC-06.11

        Args:
            tool: Tool name

        Returns:
            List of capabilities the tool provides
        """
        return self.capabilities.get(tool, [])

    def get_tools_for_capability(self, capability: ToolCapability) -> list[str]:
        """
        Get tools that provide a capability.

        Args:
            capability: The capability to search for

        Returns:
            List of tool names
        """
        tools = []
        for tool, caps in self.capabilities.items():
            if capability in caps:
                tools.append(tool)
        return tools

    def get_preferred_tools(self, task_type: str) -> list[str]:
        """
        Get preferred tools for a task type.

        Implements: SPEC-06.12

        Args:
            task_type: Type of task (e.g., "file_search", "code_execution")

        Returns:
            Ordered list of preferred tools
        """
        return self.preferences.get(task_type, [])

    def get_fallback_sequence(self, tool: str) -> list[str]:
        """
        Get fallback tools when primary tool fails.

        Implements: SPEC-06.13

        Args:
            tool: Primary tool that failed

        Returns:
            List of fallback tools to try
        """
        return self.fallbacks.get(tool, [])


class LATSOrchestrator:
    """
    LATS-inspired tool orchestrator.

    Implements: SPEC-06.10-06.15

    Orchestrates tool selection and execution using Language Agent Tree Search:
    - PLAN: Generate tool use plan before execution
    - EXPAND: Use UCB1 to select promising action branches
    - SIMULATE: Execute and evaluate results
    - BACKPROPAGATE: Update value estimates
    - REFLECT: Self-critique failed paths
    """

    def __init__(
        self,
        config: LATSConfig | None = None,
        capability_matrix: ToolCapabilityMatrix | None = None,
    ):
        """
        Initialize LATS orchestrator.

        Args:
            config: LATS configuration
            capability_matrix: Tool capability matrix
        """
        self.config = config or LATSConfig()
        self.matrix = capability_matrix or ToolCapabilityMatrix()
        self.rollouts = 0
        self.current_phase = LATSPhase.PLAN

    def plan(self, query: str, context: dict[str, Any] | None = None) -> ToolPlan:
        """
        Generate a tool use plan.

        Implements: SPEC-06.10 (PLAN phase)

        Args:
            query: User query or task description
            context: Optional context information

        Returns:
            ToolPlan with steps to execute
        """
        self.current_phase = LATSPhase.PLAN

        # Analyze query to determine task type
        task_type = self._classify_task(query)

        # Get preferred tools for this task
        preferred_tools = self.matrix.get_preferred_tools(task_type)

        # Generate plan steps
        steps = []
        if preferred_tools:
            primary_tool = preferred_tools[0]
            steps.append(
                {
                    "tool": primary_tool,
                    "action": f"Use {primary_tool} to {task_type.replace('_', ' ')}",
                }
            )

            # Add fallback steps if enabled
            if self.config.fallback_enabled:
                fallbacks = self.matrix.get_fallback_sequence(primary_tool)
                for fallback in fallbacks[:2]:  # Limit fallbacks
                    steps.append(
                        {
                            "tool": fallback,
                            "action": f"Fallback: use {fallback} if {primary_tool} fails",
                        }
                    )

        return ToolPlan(
            goal=query,
            steps=steps,
            reasoning=f"Task classified as {task_type}",
        )

    def _classify_task(self, query: str) -> str:
        """Classify query into task type."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["find", "search", "locate", "where"]):
            if "file" in query_lower or "." in query:
                return "file_search"
            return "content_search"

        if any(w in query_lower for w in ["read", "show", "display", "cat"]):
            return "file_read"

        if any(w in query_lower for w in ["edit", "modify", "change", "update"]):
            return "file_edit"

        if any(w in query_lower for w in ["write", "create", "new"]):
            return "file_write"

        if any(w in query_lower for w in ["run", "execute", "test", "build"]):
            return "code_execution"

        if any(w in query_lower for w in ["list", "ls", "directory"]):
            return "directory_list"

        if any(w in query_lower for w in ["fetch", "url", "http", "web"]):
            return "web_fetch"

        return "file_search"  # Default

    def select_node(self, root: ActionNode) -> ActionNode | None:
        """
        Select most promising node using UCB1.

        Implements: SPEC-06.10 (EXPAND phase)

        Args:
            root: Root of the action tree

        Returns:
            Selected node or None
        """
        self.current_phase = LATSPhase.EXPAND

        if not root.children:
            return root

        best_node = None
        best_ucb1 = -float("inf")

        for child in root.children:
            ucb1 = child.get_ucb1(self.config.exploration_weight)
            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best_node = child

        return best_node

    def expand(self, parent: ActionNode, step: dict[str, str]) -> ActionNode:
        """
        Expand tree with new action node.

        Args:
            parent: Parent node to expand from
            step: Step to add as child

        Returns:
            New child node
        """
        child = ActionNode(
            action=step.get("action", ""),
            tool=step.get("tool", ""),
            parent=parent,
        )
        parent.children.append(child)
        return child

    def simulate(
        self,
        node: ActionNode,
        executor: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Execute action and get result.

        Implements: SPEC-06.10 (SIMULATE phase)

        Args:
            node: Node to simulate
            executor: Optional function to execute tool

        Returns:
            Execution result
        """
        self.current_phase = LATSPhase.SIMULATE

        if executor:
            result = executor(node.tool, node.action)
        else:
            # Mock result for testing
            result = {"success": True, "output": f"Executed {node.tool}"}

        node.result = result
        return result

    def backpropagate(self, node: ActionNode, reward: float) -> None:
        """
        Backpropagate reward up the tree.

        Implements: SPEC-06.10 (BACKPROPAGATE phase)

        Args:
            node: Leaf node to start from
            reward: Reward value (0-1)
        """
        self.current_phase = LATSPhase.BACKPROPAGATE

        current = node
        while current is not None:
            current.visits += 1
            # Incremental mean update
            current.value += (reward - current.value) / current.visits
            current = current.parent

    def reflect(self, node: ActionNode) -> str:
        """
        Generate reflection on failed path.

        Implements: SPEC-06.10 (REFLECT phase)

        Args:
            node: Failed node to reflect on

        Returns:
            Reflection text
        """
        self.current_phase = LATSPhase.REFLECT

        if not self.config.reflection_enabled:
            return ""

        result = node.result or {}
        error = result.get("error", "Unknown error")

        reflection = f"Action '{node.action}' using tool '{node.tool}' failed: {error}. "

        # Suggest alternatives
        fallbacks = self.matrix.get_fallback_sequence(node.tool)
        if fallbacks:
            reflection += f"Consider trying: {', '.join(fallbacks)}"
        else:
            reflection += "No fallback tools available."

        node.reflection = reflection
        return reflection

    def should_expand(self, node: ActionNode) -> bool:
        """
        Check if node should be expanded further.

        Args:
            node: Node to check

        Returns:
            True if expansion should continue
        """
        return node.depth() < self.config.max_depth

    def should_continue(self) -> bool:
        """
        Check if search should continue.

        Returns:
            True if more rollouts allowed
        """
        return self.rollouts < self.config.max_rollouts

    def run_search(
        self,
        query: str,
        executor: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> ActionNode:
        """
        Run full LATS search.

        Args:
            query: Task to accomplish
            executor: Tool execution function

        Returns:
            Best action node found
        """
        # Plan
        plan = self.plan(query)

        # Initialize tree
        root = ActionNode(action="root", tool="")
        root.visits = 1

        best_node = root
        best_value = -float("inf")

        # Main LATS loop
        while self.should_continue():
            self.rollouts += 1

            # Select node to expand
            selected = self.select_node(root)
            if selected is None:
                break

            # Check depth limit
            if not self.should_expand(selected):
                continue

            # Expand with next step
            step_idx = selected.depth()
            if step_idx < len(plan.steps):
                step = plan.steps[step_idx]
            else:
                continue

            new_node = self.expand(selected, step)

            # Simulate
            result = self.simulate(new_node, executor)

            # Compute reward
            reward = 1.0 if result.get("success", False) else 0.0

            # Backpropagate
            self.backpropagate(new_node, reward)

            # Reflect on failures
            if reward == 0.0:
                self.reflect(new_node)

            # Track best
            if new_node.value > best_value:
                best_value = new_node.value
                best_node = new_node

        return best_node


__all__ = [
    "ActionNode",
    "LATSConfig",
    "LATSOrchestrator",
    "LATSPhase",
    "ToolCapability",
    "ToolCapabilityMatrix",
    "ToolPlan",
    "compute_ucb1",
]
