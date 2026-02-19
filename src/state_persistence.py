"""
State persistence for RLM-Claude-Code.

Implements: Spec §5.2 Hook Integration (state management)

Handles:
- Save RLM state on session end
- Restore RLM state on session resume
- Clear state on /clear
- Tool state synchronization
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import RLMConfig, default_config
from .types import Message, MessageRole, SessionContext, ToolOutput


@dataclass
class RLMSessionState:
    """
    Persistent state for an RLM session.

    Implements: Spec §5.2 Hook Integration
    """

    # Session identification
    session_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # RLM mode state
    rlm_active: bool = False
    current_depth: int = 0
    total_recursive_calls: int = 0
    total_tokens_used: int = 0

    # Context state
    working_memory: dict[str, Any] = field(default_factory=dict)
    file_cache: dict[str, str] = field(default_factory=dict)
    tool_outputs_count: int = 0

    # Trajectory state
    trajectory_path: str | None = None
    trajectory_events_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLMSessionState:
        """Create from dictionary, tolerating unknown fields."""
        from dataclasses import fields as dc_fields

        known = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class StatePersistence:
    """
    Manages RLM state persistence across sessions.

    Implements: Spec §5.2 Hook Integration

    State is stored in ~/.claude/rlm-state/
    """

    def __init__(self, config: RLMConfig | None = None):
        """
        Initialize state persistence.

        Args:
            config: RLM configuration
        """
        self.config = config or default_config
        self.state_dir = Path.home() / ".claude" / "rlm-state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Current session state (in-memory)
        self._current_state: RLMSessionState | None = None
        self._current_context: SessionContext | None = None

    @property
    def current_state(self) -> RLMSessionState | None:
        """Get current session state."""
        return self._current_state

    @property
    def current_context(self) -> SessionContext | None:
        """Get current session context."""
        return self._current_context

    def get_state_file(self, session_id: str) -> Path:
        """Get state file path for session."""
        return self.state_dir / f"{session_id}.json"

    def get_context_file(self, session_id: str) -> Path:
        """Get context file path for session."""
        return self.state_dir / f"{session_id}_context.json"

    def init_session(self, session_id: str) -> RLMSessionState:
        """
        Initialize new session or restore existing.

        Args:
            session_id: Unique session identifier

        Returns:
            Session state (new or restored)
        """
        state_file = self.get_state_file(session_id)

        if state_file.exists():
            # Restore existing session
            return self.restore_state(session_id)
        else:
            # Create new session
            self._current_state = RLMSessionState(session_id=session_id)
            self._current_context = SessionContext()
            return self._current_state

    def save_state(self, session_id: str | None = None) -> Path:
        """
        Save current state to disk using atomic write.

        Uses write-to-temp-then-rename pattern to prevent corruption
        from concurrent writes by multiple hook processes.

        Args:
            session_id: Session ID (uses current if not provided)

        Returns:
            Path to saved state file
        """
        if self._current_state is None:
            raise ValueError("No active session to save")

        session_id = session_id or self._current_state.session_id
        self._current_state.updated_at = time.time()

        state_file = self.get_state_file(session_id)
        self._atomic_json_write(state_file, self._current_state.to_dict())

        # Also save context if available
        if self._current_context is not None:
            self._save_context(session_id)

        return state_file

    def _atomic_json_write(self, target_path: Path, data: dict) -> None:
        """
        Write JSON atomically using temp file + rename.

        This prevents corruption when multiple processes write concurrently.
        The rename operation is atomic on POSIX systems.
        """
        # Write to temp file in same directory (ensures same filesystem for rename)
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=target_path.stem + "_",
            dir=target_path.parent,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            # Atomic rename
            os.rename(temp_path, target_path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _save_context(self, session_id: str) -> Path:
        """Save context to disk using atomic write."""
        if self._current_context is None:
            raise ValueError("No context to save")

        context_file = self.get_context_file(session_id)

        # Serialize context
        context_data = {
            "messages": [
                {"role": m.role.value, "content": m.content, "timestamp": m.timestamp}
                for m in self._current_context.messages
            ],
            "files": self._current_context.files,
            "tool_outputs": [
                {
                    "tool_name": o.tool_name,
                    "content": o.content,
                    "exit_code": o.exit_code,
                    "timestamp": o.timestamp,
                }
                for o in self._current_context.tool_outputs
            ],
            "working_memory": self._current_context.working_memory,
        }

        self._atomic_json_write(context_file, context_data)

        return context_file

    def restore_state(self, session_id: str) -> RLMSessionState:
        """
        Restore state from disk.

        Args:
            session_id: Session to restore

        Returns:
            Restored session state

        Raises:
            FileNotFoundError: If session state not found
        """
        state_file = self.get_state_file(session_id)

        if not state_file.exists():
            raise FileNotFoundError(f"No state found for session {session_id}")

        with open(state_file) as f:
            data = json.load(f)

        self._current_state = RLMSessionState.from_dict(data)

        # Also restore context if available
        context_file = self.get_context_file(session_id)
        if context_file.exists():
            self._restore_context(context_file)
        else:
            self._current_context = SessionContext()

        return self._current_state

    def _restore_context(self, context_file: Path) -> None:
        """Restore context from disk."""
        with open(context_file) as f:
            data = json.load(f)

        self._current_context = SessionContext(
            messages=[
                Message(
                    role=MessageRole(m["role"]),
                    content=m["content"],
                    timestamp=m.get("timestamp"),
                )
                for m in data.get("messages", [])
            ],
            files=data.get("files", {}),
            tool_outputs=[
                ToolOutput(
                    tool_name=o["tool_name"],
                    content=o["content"],
                    exit_code=o.get("exit_code"),
                    timestamp=o.get("timestamp"),
                )
                for o in data.get("tool_outputs", [])
            ],
            working_memory=data.get("working_memory", {}),
        )

    def clear_state(self, session_id: str | None = None) -> None:
        """
        Clear session state (on /clear).

        Args:
            session_id: Session to clear (uses current if not provided)
        """
        if session_id is None and self._current_state is not None:
            session_id = self._current_state.session_id

        if session_id:
            # Remove state files
            state_file = self.get_state_file(session_id)
            context_file = self.get_context_file(session_id)

            if state_file.exists():
                state_file.unlink()
            if context_file.exists():
                context_file.unlink()

        # Reset in-memory state
        self._current_state = None
        self._current_context = None

    def update_rlm_active(self, active: bool) -> None:
        """Update RLM active status."""
        if self._current_state:
            self._current_state.rlm_active = active
            self._current_state.updated_at = time.time()

    def update_depth(self, depth: int) -> None:
        """Update current recursion depth."""
        if self._current_state:
            self._current_state.current_depth = depth
            self._current_state.updated_at = time.time()

    def increment_recursive_calls(self, count: int = 1) -> None:
        """Increment recursive call counter."""
        if self._current_state:
            self._current_state.total_recursive_calls += count
            self._current_state.updated_at = time.time()

    def add_tokens_used(self, tokens: int) -> None:
        """Add to total tokens used."""
        if self._current_state:
            self._current_state.total_tokens_used += tokens
            self._current_state.updated_at = time.time()

    def update_working_memory(self, key: str, value: Any) -> None:
        """Update working memory."""
        if self._current_state:
            self._current_state.working_memory[key] = value
            self._current_state.updated_at = time.time()

        if self._current_context:
            self._current_context.working_memory[key] = value

    def add_file_to_cache(self, path: str, content: str) -> None:
        """Add file to cache."""
        if self._current_state:
            self._current_state.file_cache[path] = content
            self._current_state.updated_at = time.time()

        if self._current_context:
            self._current_context.files[path] = content

    def add_tool_output(self, tool_name: str, content: str, exit_code: int | None = None) -> None:
        """Add tool output to context."""
        if self._current_state:
            self._current_state.tool_outputs_count += 1
            self._current_state.updated_at = time.time()

        if self._current_context:
            self._current_context.tool_outputs.append(
                ToolOutput(
                    tool_name=tool_name,
                    content=content,
                    exit_code=exit_code,
                    timestamp=time.time(),
                )
            )

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add message to context."""
        if self._current_context:
            self._current_context.messages.append(
                Message(role=role, content=content, timestamp=time.time())
            )

    def set_trajectory_path(self, path: str) -> None:
        """Set trajectory export path."""
        if self._current_state:
            self._current_state.trajectory_path = path
            self._current_state.updated_at = time.time()

    def increment_trajectory_events(self, count: int = 1) -> None:
        """Increment trajectory events counter."""
        if self._current_state:
            self._current_state.trajectory_events_count += count
            self._current_state.updated_at = time.time()

    def list_sessions(self) -> list[str]:
        """List all saved session IDs."""
        return [f.stem for f in self.state_dir.glob("*.json") if not f.stem.endswith("_context")]

    def cleanup_old_sessions(self, max_age_days: int = 7) -> int:
        """
        Clean up sessions older than max_age_days.

        Returns:
            Number of sessions cleaned up
        """
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned = 0

        for state_file in self.state_dir.glob("*.json"):
            if state_file.stem.endswith("_context"):
                continue

            try:
                with open(state_file) as f:
                    data = json.load(f)

                if data.get("updated_at", 0) < cutoff:
                    session_id = state_file.stem
                    self.clear_state(session_id)
                    cleaned += 1
            except (json.JSONDecodeError, KeyError):
                # Invalid state file, clean it up
                state_file.unlink()
                cleaned += 1

        return cleaned


# Global instance for hook scripts
_persistence: StatePersistence | None = None


def get_persistence() -> StatePersistence:
    """Get global persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = StatePersistence()
    return _persistence


__all__ = [
    "RLMSessionState",
    "StatePersistence",
    "get_persistence",
]
