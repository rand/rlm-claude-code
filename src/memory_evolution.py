"""
Memory evolution: consolidation, promotion, and decay algorithms.

Implements: Spec SPEC-03 Memory Evolution
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory_store import MemoryStore, Node


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class ConsolidationResult:
    """
    Result of consolidation operation.

    Implements: Spec SPEC-03.01
    """

    merged_count: int
    promoted_count: int
    edges_strengthened: int


@dataclass
class PromotionResult:
    """
    Result of promotion operation.

    Implements: Spec SPEC-03.08
    """

    promoted_count: int
    crystallized_count: int


@dataclass
class DecayResult:
    """
    Result of decay operation.

    Implements: Spec SPEC-03.15
    """

    decayed_count: int
    archived_count: int


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "consolidation_threshold": 0.5,
    "promotion_threshold": 0.8,
    "decay_factor": 0.95,
    "decay_min_confidence": 0.3,
    "decay_interval_hours": 24,
}


# =============================================================================
# MemoryEvolution Class
# =============================================================================


class MemoryEvolution:
    """
    Memory lifecycle management: consolidation, promotion, and decay.

    Implements: Spec SPEC-03

    Features:
    - Consolidation: Merge task tier nodes to session tier
    - Promotion: Move valuable session nodes to long-term
    - Decay: Reduce confidence of stale nodes over time
    - Archive: Soft delete for nodes below min confidence
    """

    def __init__(
        self,
        store: MemoryStore,
        config_path: str | None = None,
    ):
        """
        Initialize memory evolution.

        Args:
            store: MemoryStore instance
            config_path: Optional path to config file
        """
        self.store = store
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """
        Load configuration from file or use defaults.

        Implements: Spec SPEC-03.25
        """
        config = DEFAULT_CONFIG.copy()

        # Try loading from specified path
        if config_path:
            path = Path(config_path)
            if path.exists():
                try:
                    with open(path) as f:
                        file_config = json.load(f)
                        if "memory" in file_config:
                            config.update(file_config["memory"])
                except (json.JSONDecodeError, OSError):
                    pass  # Use defaults if file is invalid

        # Try loading from default location
        else:
            default_path = Path.home() / ".claude" / "rlm-config.json"
            if default_path.exists():
                try:
                    with open(default_path) as f:
                        file_config = json.load(f)
                        if "memory" in file_config:
                            config.update(file_config["memory"])
                except (json.JSONDecodeError, OSError):
                    pass

        return config

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    # =========================================================================
    # Consolidation (SPEC-03.01-07)
    # =========================================================================

    def consolidate(self, task_id: str) -> ConsolidationResult:
        """
        Consolidate task tier nodes to session tier.

        Implements: Spec SPEC-03.01-07

        Args:
            task_id: Task ID to consolidate nodes for

        Returns:
            ConsolidationResult with statistics
        """
        # Get all task nodes for this task
        task_nodes = self.store.get_nodes_by_metadata(
            key="task_id",
            value=task_id,
            tier="task",
        )

        if not task_nodes:
            return ConsolidationResult(merged_count=0, promoted_count=0, edges_strengthened=0)

        merged_count = 0
        promoted_count = 0
        edges_strengthened = 0

        # Group nodes by content similarity for merging (SPEC-03.03)
        content_groups: dict[str, list[Node]] = {}
        for node in task_nodes:
            # Normalize content for comparison
            normalized = node.content.strip().lower()
            if normalized not in content_groups:
                content_groups[normalized] = []
            content_groups[normalized].append(node)

        # Process each group
        for content, nodes in content_groups.items():
            if len(nodes) > 1:
                # Merge redundant facts - keep highest confidence (SPEC-03.03)
                nodes.sort(key=lambda n: n.confidence, reverse=True)
                keeper = nodes[0]

                # Archive the others
                for node in nodes[1:]:
                    self.store.update_node(node.id, tier="archive")
                    merged_count += 1

                    # Log the merge
                    self.store.log_evolution(
                        operation="consolidate",
                        node_ids=[node.id, keeper.id],
                        from_tier="task",
                        to_tier="archive",
                        reasoning="Merged duplicate with higher confidence version",
                    )

                # Promote the keeper to session
                self.store.update_node(keeper.id, tier="session")
                promoted_count += 1
            else:
                # Single node - just promote to session
                self.store.update_node(nodes[0].id, tier="session")
                promoted_count += 1

        # Strengthen frequently-accessed edges (SPEC-03.04)
        for node in task_nodes:
            edges = self.store.get_edges_for_node(node.id)
            for edge in edges:
                # Increase weight based on node access count
                node_data = self.store.get_node(node.id, include_archived=True)
                if node_data and node_data.access_count > 1:
                    new_weight = edge.weight * (1 + 0.1 * min(node_data.access_count, 10))
                    self.store.update_edge(edge.id, weight=new_weight)
                    edges_strengthened += 1

        # Log overall consolidation
        self.store.log_evolution(
            operation="consolidate",
            node_ids=[n.id for n in task_nodes],
            from_tier="task",
            to_tier="session",
            reasoning=f"Consolidated {len(task_nodes)} nodes from task {task_id}",
        )

        return ConsolidationResult(
            merged_count=merged_count,
            promoted_count=promoted_count,
            edges_strengthened=edges_strengthened,
        )

    # =========================================================================
    # Promotion (SPEC-03.08-14)
    # =========================================================================

    def promote(
        self,
        session_id: str,
        threshold: float | None = None,
        confirm: bool = True,
    ) -> PromotionResult:
        """
        Promote session tier nodes to long-term tier.

        Implements: Spec SPEC-03.08-14

        Args:
            session_id: Session ID to promote nodes for
            threshold: Confidence threshold (default 0.8)
            confirm: If True, actually move nodes; if False, just identify candidates

        Returns:
            PromotionResult with statistics
        """
        if threshold is None:
            threshold = self.config["promotion_threshold"]

        # Get session nodes for this session
        session_nodes = self.store.get_nodes_by_metadata(
            key="session_id",
            value=session_id,
            tier="session",
        )

        if not session_nodes:
            return PromotionResult(promoted_count=0, crystallized_count=0)

        # Calculate median access count (SPEC-03.10)
        access_counts = sorted(n.access_count for n in session_nodes)
        median_idx = len(access_counts) // 2
        median_access = access_counts[median_idx] if access_counts else 0

        promoted_count = 0
        crystallized_count = 0

        # Identify candidates based on confidence and access (SPEC-03.09, SPEC-03.10)
        candidates = [
            n
            for n in session_nodes
            if n.confidence >= threshold and n.access_count >= median_access
        ]

        if confirm:
            for node in candidates:
                self.store.update_node(node.id, tier="longterm")
                promoted_count += 1

                # Log the promotion
                self.store.log_evolution(
                    operation="promote",
                    node_ids=[node.id],
                    from_tier="session",
                    to_tier="longterm",
                    reasoning=f"Confidence {node.confidence:.2f} >= {threshold}, access {node.access_count} >= median {median_access}",
                )

            # Check for complex subgraphs to crystallize (SPEC-03.11)
            # Group nodes by their edge connections
            if len(candidates) >= 3:
                # Find densely connected subgroups
                connected_groups = self._find_connected_subgraphs(candidates)
                for group in connected_groups:
                    if len(group) >= 3:
                        # Create crystallized summary node
                        summary_content = self._create_summary(group)
                        summary_id = self.store.create_node(
                            node_type="fact",
                            content=summary_content,
                            tier="longterm",
                            confidence=max(n.confidence for n in group),
                            metadata={
                                "crystallized": True,
                                "source_nodes": [n.id for n in group],
                                "session_id": session_id,
                            },
                        )

                        # Create "summarizes" edges (SPEC-03.05)
                        for node in group:
                            self.store.create_edge(
                                edge_type="composition",
                                label="summarizes",
                                members=[
                                    {"node_id": summary_id, "role": "subject", "position": 0},
                                    {"node_id": node.id, "role": "object", "position": 1},
                                ],
                            )

                        crystallized_count += 1

        return PromotionResult(
            promoted_count=promoted_count,
            crystallized_count=crystallized_count,
        )

    def promote_node(
        self,
        node_id: str,
        target_tier: str | None = None,
    ) -> bool:
        """
        Promote a specific node to a higher tier.

        Implements: Spec SPEC-03.08

        Args:
            node_id: ID of the node to promote
            target_tier: Target tier ('session', 'longterm'). If None, promotes
                        to the next tier (task→session, session→longterm)

        Returns:
            True if promoted successfully, False if node not found or invalid tier
        """
        # Get the node
        node = self.store.get_node(node_id)
        if node is None:
            return False

        # Determine target tier
        tier_progression = {
            "task": "session",
            "session": "longterm",
        }

        if target_tier is None:
            target_tier = tier_progression.get(node.tier)
            if target_tier is None:
                return False  # Can't promote from longterm or archive

        # Validate target tier is a valid promotion
        valid_promotions = {
            "task": ["session", "longterm"],
            "session": ["longterm"],
        }

        if node.tier not in valid_promotions:
            return False  # Can't promote from longterm or archive

        if target_tier not in valid_promotions.get(node.tier, []):
            return False  # Invalid target tier for current tier

        # Perform the promotion
        result = self.store.update_node(node_id, tier=target_tier)

        if result:
            self.store.log_evolution(
                operation="promote",
                node_ids=[node_id],
                from_tier=node.tier,
                to_tier=target_tier,
                reasoning="Single node promotion via promote_node()",
            )

        return result

    def _find_connected_subgraphs(self, nodes: list[Node]) -> list[list[Node]]:
        """Find groups of connected nodes."""
        if not nodes:
            return []

        node_ids = {n.id for n in nodes}
        node_map = {n.id: n for n in nodes}
        visited: set[str] = set()
        groups: list[list[Node]] = []

        for node in nodes:
            if node.id in visited:
                continue

            # BFS to find connected component
            group: list[Node] = []
            queue = [node.id]

            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)

                if current_id in node_map:
                    group.append(node_map[current_id])

                    # Find related nodes
                    related = self.store.get_related_nodes(current_id)
                    for rel in related:
                        if rel.id in node_ids and rel.id not in visited:
                            queue.append(rel.id)

            if group:
                groups.append(group)

        return groups

    def _create_summary(self, nodes: list[Node]) -> str:
        """Create a summary of node contents."""
        contents = [n.content[:100] for n in nodes[:5]]  # Limit to first 5, truncate each
        return "Summary: " + "; ".join(contents)

    # =========================================================================
    # Decay (SPEC-03.15-20)
    # =========================================================================

    def decay(
        self,
        factor: float | None = None,
        min_confidence: float | None = None,
    ) -> DecayResult:
        """
        Apply temporal decay to long-term memory.

        Implements: Spec SPEC-03.15-20

        Formula: new_confidence = base_confidence * (factor ^ days_since_access)
        With amplifier: amplifier = access_count / (1 + log(access_count))

        Args:
            factor: Decay factor per day (default 0.95)
            min_confidence: Minimum confidence before archiving (default 0.3)

        Returns:
            DecayResult with statistics
        """
        if factor is None:
            factor = self.config["decay_factor"]
        if min_confidence is None:
            min_confidence = self.config["decay_min_confidence"]

        # Get all longterm nodes (archived nodes are not decayed)
        longterm_nodes = self.store.query_nodes(tier="longterm")

        decayed_count = 0
        archived_count = 0
        now = datetime.now()

        for node in longterm_nodes:
            # Calculate days since last access
            last_accessed_ms = node.last_accessed
            last_accessed = datetime.fromtimestamp(last_accessed_ms / 1000)
            days_since = (now - last_accessed).days

            if days_since <= 0:
                continue  # Recently accessed, no decay

            # Access frequency amplifier (SPEC-03.17)
            # Higher access count = slower decay
            access_count = max(1, node.access_count)
            amplifier = access_count / (1 + math.log(access_count))

            # Apply decay with amplifier
            # More accesses = higher amplifier = less decay
            adjusted_factor = factor ** (days_since / max(1, amplifier))
            new_confidence = node.confidence * adjusted_factor

            # Ensure confidence doesn't increase
            new_confidence = min(new_confidence, node.confidence)

            if new_confidence < min_confidence:
                # Archive the node (SPEC-03.18)
                self.store.update_node(node.id, tier="archive", confidence=new_confidence)
                archived_count += 1

                self.store.log_evolution(
                    operation="decay",
                    node_ids=[node.id],
                    from_tier="longterm",
                    to_tier="archive",
                    reasoning=f"Confidence {new_confidence:.3f} < {min_confidence} after {days_since} days",
                )
            else:
                # Just update confidence
                self.store.update_node(node.id, confidence=new_confidence)
                decayed_count += 1

        return DecayResult(
            decayed_count=decayed_count,
            archived_count=archived_count,
        )

    # =========================================================================
    # Archive Operations (SPEC-03.21-24)
    # =========================================================================

    def restore_node(self, node_id: str) -> bool:
        """
        Restore an archived node to long-term tier.

        Implements: Spec SPEC-03.23

        Args:
            node_id: Node ID to restore

        Returns:
            True if restored, False if node not found or not archived
        """
        # Get node including archived
        node = self.store.get_node(node_id, include_archived=True)

        if node is None:
            return False

        if node.tier != "archive":
            return False  # Not archived

        # Restore to longterm
        result = self.store.update_node(node_id, tier="longterm")

        if result:
            self.store.log_evolution(
                operation="restore",
                node_ids=[node_id],
                from_tier="archive",
                to_tier="longterm",
                reasoning="Manual restoration",
            )

        return result


__all__ = [
    "MemoryEvolution",
    "ConsolidationResult",
    "PromotionResult",
    "DecayResult",
]
