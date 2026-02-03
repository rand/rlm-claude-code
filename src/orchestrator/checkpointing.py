"""
Checkpointing support for RLM sessions.

Implements: SPEC-12.05

Contains:
- RLMCheckpoint dataclass
- Serialization/deserialization
- CheckpointingOrchestrator
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RLMCheckpoint:
    """
    Checkpoint for RLM session state.

    Implements: SPEC-12.05, SPEC-09.20-09.26

    Captures all state needed to resume an RLM session:
    - Session identification
    - Current orchestration state (depth, turn)
    - Conversation messages
    - REPL variable state
    - Trajectory events for replay
    - Working memory (SPEC-09.21)
    - Pending operations (SPEC-09.21)
    - Cost tracking (SPEC-09.21)
    """

    session_id: str
    depth: int
    turn: int
    messages: list[dict[str, str]]
    repl_state: dict[str, Any]
    trajectory_events: list[dict[str, Any]]
    # SPEC-09.21: Additional state for multi-turn checkpointing
    working_memory: dict[str, Any] = field(default_factory=dict)
    pending_operations: list[dict[str, Any]] = field(default_factory=list)
    cost_so_far: float = 0.0
    timestamp: float = field(default_factory=time.time)
    version: str = "1.1"  # Version 1.1 for expanded format
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize checkpoint to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLMCheckpoint:
        """Deserialize checkpoint from dictionary with backward compatibility."""
        return cls(
            session_id=data["session_id"],
            depth=data["depth"],
            turn=data["turn"],
            messages=data["messages"],
            repl_state=data["repl_state"],
            trajectory_events=data["trajectory_events"],
            # SPEC-09.21: New fields with defaults for backward compatibility
            working_memory=data.get("working_memory", {}),
            pending_operations=data.get("pending_operations", []),
            cost_so_far=data.get("cost_so_far", 0.0),
            timestamp=data.get("timestamp", time.time()),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> RLMCheckpoint:
        """Deserialize checkpoint from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: str | Path) -> None:
        """Save checkpoint to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> RLMCheckpoint:
        """Load checkpoint from file."""
        path = Path(path)
        return cls.from_json(path.read_text())

    # SPEC-09.23: Restoration support methods

    def get_repl_state(self) -> dict[str, Any]:
        """Get REPL state for restoration."""
        return self.repl_state.copy()

    def get_pending_operations(self) -> list[dict[str, Any]]:
        """Get all pending operations."""
        return self.pending_operations.copy()

    def get_resumable_operations(self) -> list[dict[str, Any]]:
        """Get operations that can be resumed (not completed)."""
        return [op for op in self.pending_operations if op.get("status") != "completed"]

    # SPEC-09.26: Version compatibility

    def is_compatible(self, runtime_version: str) -> bool:
        """
        Check if checkpoint is compatible with runtime version.

        Args:
            runtime_version: Current runtime version string (e.g., "1.1")

        Returns:
            True if checkpoint can be loaded by this runtime
        """
        try:
            checkpoint_major = int(self.version.split(".")[0])
            runtime_major = int(runtime_version.split(".")[0])

            # Checkpoint major version must not exceed runtime major version
            return checkpoint_major <= runtime_major
        except (ValueError, IndexError):
            # Invalid version format - assume incompatible
            return False


class CheckpointingOrchestrator:
    """
    Orchestrator wrapper with checkpointing support.

    Implements: SPEC-12.05

    Provides:
    - Automatic checkpointing every N turns
    - Manual checkpoint creation
    - Session restoration from checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "~/.rlm/checkpoints",
        auto_checkpoint_interval: int = 5,
    ):
        """
        Initialize checkpointing orchestrator.

        Args:
            checkpoint_dir: Directory for checkpoint storage
            auto_checkpoint_interval: Create checkpoint every N turns (0 to disable)
        """
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self._current_checkpoint: RLMCheckpoint | None = None

    def create_checkpoint(
        self,
        session_id: str,
        depth: int,
        turn: int,
        messages: list[dict[str, str]],
        repl_state: dict[str, Any],
        trajectory_events: list[dict[str, Any]] | None = None,
        working_memory: dict[str, Any] | None = None,
        pending_operations: list[dict[str, Any]] | None = None,
        cost_so_far: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> RLMCheckpoint:
        """
        Create a checkpoint from current state.

        Implements: SPEC-09.20-09.21

        Args:
            session_id: Session identifier
            depth: Current recursion depth
            turn: Current turn number
            messages: Conversation messages
            repl_state: REPL variable state
            trajectory_events: Optional trajectory events
            working_memory: Current working memory state (SPEC-09.21)
            pending_operations: Pending async operations (SPEC-09.21)
            cost_so_far: Accumulated cost in USD (SPEC-09.21)
            metadata: Optional additional metadata

        Returns:
            Created checkpoint
        """
        checkpoint = RLMCheckpoint(
            session_id=session_id,
            depth=depth,
            turn=turn,
            messages=messages,
            repl_state=repl_state,
            trajectory_events=trajectory_events or [],
            working_memory=working_memory or {},
            pending_operations=pending_operations or [],
            cost_so_far=cost_so_far,
            metadata=metadata or {},
        )

        self._current_checkpoint = checkpoint
        return checkpoint

    def save_checkpoint(self, checkpoint: RLMCheckpoint) -> Path:
        """
        Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            Path to saved checkpoint file
        """
        filename = f"{checkpoint.session_id}_{int(checkpoint.timestamp)}.json"
        path = self.checkpoint_dir / filename
        checkpoint.save(path)
        return path

    def load_checkpoint(self, session_id: str) -> RLMCheckpoint | None:
        """
        Load most recent checkpoint for a session.

        Args:
            session_id: Session identifier

        Returns:
            Most recent checkpoint or None if not found
        """
        if not self.checkpoint_dir.exists():
            return None

        # Find all checkpoints for this session
        checkpoints = list(self.checkpoint_dir.glob(f"{session_id}_*.json"))
        if not checkpoints:
            return None

        # Sort by timestamp (embedded in filename) and load most recent
        checkpoints.sort(reverse=True)
        return RLMCheckpoint.load(checkpoints[0])

    def list_checkpoints(self, session_id: str | None = None) -> list[Path]:
        """
        List available checkpoints.

        Args:
            session_id: Optional session filter

        Returns:
            List of checkpoint file paths
        """
        if not self.checkpoint_dir.exists():
            return []

        if session_id:
            return list(self.checkpoint_dir.glob(f"{session_id}_*.json"))
        return list(self.checkpoint_dir.glob("*.json"))

    def should_checkpoint(self, turn: int) -> bool:
        """
        Check if auto-checkpoint should trigger.

        Args:
            turn: Current turn number

        Returns:
            True if checkpoint should be created
        """
        if self.auto_checkpoint_interval <= 0:
            return False
        return turn > 0 and turn % self.auto_checkpoint_interval == 0

    def cleanup_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 3,
    ) -> int:
        """
        Remove old checkpoints, keeping most recent.

        Args:
            session_id: Session to clean up
            keep_count: Number of checkpoints to keep

        Returns:
            Number of checkpoints removed
        """
        checkpoints = self.list_checkpoints(session_id)
        if len(checkpoints) <= keep_count:
            return 0

        # Sort by timestamp (newest first) and remove old ones
        checkpoints.sort(reverse=True)
        to_remove = checkpoints[keep_count:]

        for path in to_remove:
            path.unlink()

        return len(to_remove)

    def restore_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Restore session state from checkpoint.

        Implements: SPEC-09.23

        Loads the most recent checkpoint and returns restoration info
        including REPL state and resumable operations.

        Args:
            session_id: Session to restore

        Returns:
            Dictionary with restoration info or None if not found
        """
        checkpoint = self.load_checkpoint(session_id)
        if checkpoint is None:
            return None

        return {
            "session_id": checkpoint.session_id,
            "depth": checkpoint.depth,
            "turn": checkpoint.turn,
            "messages": checkpoint.messages,
            "repl_state": checkpoint.get_repl_state(),
            "working_memory": checkpoint.working_memory,
            "resumable_operations": checkpoint.get_resumable_operations(),
            "trajectory_events": checkpoint.trajectory_events,
            "cost_so_far": checkpoint.cost_so_far,
            "timestamp": checkpoint.timestamp,
            "version": checkpoint.version,
            "metadata": checkpoint.metadata,
        }


__all__ = [
    "CheckpointingOrchestrator",
    "RLMCheckpoint",
]
