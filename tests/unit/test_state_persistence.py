"""
Unit tests for state_persistence module.

Implements: Spec §5.2 Hook Integration tests
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.state_persistence import (
    RLMSessionState,
    StatePersistence,
    get_persistence,
)
from src.types import MessageRole


class TestRLMSessionState:
    """Tests for RLMSessionState dataclass."""

    def test_default_values(self):
        """Has expected default values."""
        state = RLMSessionState()

        assert state.session_id == ""
        assert state.rlm_active is False
        assert state.current_depth == 0
        assert state.total_recursive_calls == 0
        assert state.total_tokens_used == 0
        assert state.working_memory == {}
        assert state.file_cache == {}

    def test_create_with_values(self):
        """Can create with custom values."""
        state = RLMSessionState(
            session_id="test-123",
            rlm_active=True,
            current_depth=2,
            total_recursive_calls=5,
        )

        assert state.session_id == "test-123"
        assert state.rlm_active is True
        assert state.current_depth == 2
        assert state.total_recursive_calls == 5

    def test_to_dict(self):
        """Can convert to dictionary."""
        state = RLMSessionState(
            session_id="test-456",
            rlm_active=True,
            working_memory={"key": "value"},
        )

        d = state.to_dict()

        assert d["session_id"] == "test-456"
        assert d["rlm_active"] is True
        assert d["working_memory"] == {"key": "value"}

    def test_from_dict(self):
        """Can create from dictionary."""
        data = {
            "session_id": "restored-789",
            "created_at": 1000.0,
            "updated_at": 2000.0,
            "rlm_active": True,
            "current_depth": 1,
            "total_recursive_calls": 10,
            "total_tokens_used": 5000,
            "working_memory": {"restored": True},
            "file_cache": {"test.py": "content"},
            "tool_outputs_count": 3,
            "trajectory_path": "/path/to/traj",
            "trajectory_events_count": 25,
        }

        state = RLMSessionState.from_dict(data)

        assert state.session_id == "restored-789"
        assert state.rlm_active is True
        assert state.total_tokens_used == 5000
        assert state.working_memory == {"restored": True}


class TestStatePersistence:
    """Tests for StatePersistence class."""

    @pytest.fixture
    def temp_state_dir(self):
        """Create temporary state directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def persistence(self, temp_state_dir, monkeypatch):
        """Create persistence with temp directory."""
        # Monkeypatch Path.home() to use temp dir
        monkeypatch.setattr(Path, "home", lambda: temp_state_dir)
        return StatePersistence()

    def test_init_creates_state_dir(self, temp_state_dir, monkeypatch):
        """Initialization creates state directory."""
        monkeypatch.setattr(Path, "home", lambda: temp_state_dir)
        persistence = StatePersistence()

        assert persistence.state_dir.exists()
        assert persistence.state_dir == temp_state_dir / ".claude" / "rlm-state"

    def test_init_session_creates_new(self, persistence):
        """init_session creates new session state."""
        state = persistence.init_session("new-session")

        assert state.session_id == "new-session"
        assert persistence.current_state is state
        assert persistence.current_context is not None

    def test_save_and_restore_state(self, persistence):
        """Can save and restore state."""
        # Initialize and modify session
        persistence.init_session("save-test")
        persistence.update_rlm_active(True)
        persistence.update_depth(2)
        persistence.add_tokens_used(1000)
        persistence.update_working_memory("test_key", "test_value")

        # Save
        state_file = persistence.save_state()
        assert state_file.exists()

        # Clear and restore
        persistence._current_state = None
        persistence._current_context = None

        restored = persistence.restore_state("save-test")

        assert restored.session_id == "save-test"
        assert restored.rlm_active is True
        assert restored.current_depth == 2
        assert restored.total_tokens_used == 1000
        assert restored.working_memory["test_key"] == "test_value"

    def test_restore_nonexistent_raises(self, persistence):
        """Restore nonexistent session raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            persistence.restore_state("nonexistent")

    def test_clear_state(self, persistence):
        """clear_state removes session files."""
        # Create and save session
        persistence.init_session("clear-test")
        persistence.save_state()

        state_file = persistence.get_state_file("clear-test")
        assert state_file.exists()

        # Clear
        persistence.clear_state("clear-test")

        assert not state_file.exists()
        assert persistence.current_state is None
        assert persistence.current_context is None

    def test_update_rlm_active(self, persistence):
        """Can update RLM active status."""
        persistence.init_session("active-test")

        persistence.update_rlm_active(True)
        assert persistence.current_state.rlm_active is True

        persistence.update_rlm_active(False)
        assert persistence.current_state.rlm_active is False

    def test_update_depth(self, persistence):
        """Can update current depth."""
        persistence.init_session("depth-test")

        persistence.update_depth(3)
        assert persistence.current_state.current_depth == 3

    def test_increment_recursive_calls(self, persistence):
        """Can increment recursive calls."""
        persistence.init_session("calls-test")

        persistence.increment_recursive_calls(2)
        assert persistence.current_state.total_recursive_calls == 2

        persistence.increment_recursive_calls(3)
        assert persistence.current_state.total_recursive_calls == 5

    def test_add_tokens_used(self, persistence):
        """Can add tokens used."""
        persistence.init_session("tokens-test")

        persistence.add_tokens_used(500)
        assert persistence.current_state.total_tokens_used == 500

        persistence.add_tokens_used(300)
        assert persistence.current_state.total_tokens_used == 800

    def test_update_working_memory(self, persistence):
        """Can update working memory."""
        persistence.init_session("memory-test")

        persistence.update_working_memory("key1", "value1")
        persistence.update_working_memory("key2", {"nested": True})

        assert persistence.current_state.working_memory["key1"] == "value1"
        assert persistence.current_state.working_memory["key2"] == {"nested": True}
        assert persistence.current_context.working_memory["key1"] == "value1"

    def test_add_file_to_cache(self, persistence):
        """Can add files to cache."""
        persistence.init_session("cache-test")

        persistence.add_file_to_cache("test.py", "print('hello')")

        assert persistence.current_state.file_cache["test.py"] == "print('hello')"
        assert persistence.current_context.files["test.py"] == "print('hello')"

    def test_add_tool_output(self, persistence):
        """Can add tool outputs."""
        persistence.init_session("output-test")

        persistence.add_tool_output("bash", "output content", exit_code=0)

        assert persistence.current_state.tool_outputs_count == 1
        assert len(persistence.current_context.tool_outputs) == 1
        assert persistence.current_context.tool_outputs[0].tool_name == "bash"
        assert persistence.current_context.tool_outputs[0].content == "output content"

    def test_add_message(self, persistence):
        """Can add messages to context."""
        persistence.init_session("message-test")

        persistence.add_message(MessageRole.USER, "Hello")
        persistence.add_message(MessageRole.ASSISTANT, "Hi there")

        assert len(persistence.current_context.messages) == 2
        assert persistence.current_context.messages[0].role == MessageRole.USER
        assert persistence.current_context.messages[1].content == "Hi there"

    def test_set_trajectory_path(self, persistence):
        """Can set trajectory path."""
        persistence.init_session("traj-test")

        persistence.set_trajectory_path("/path/to/trajectory.json")

        assert persistence.current_state.trajectory_path == "/path/to/trajectory.json"

    def test_increment_trajectory_events(self, persistence):
        """Can increment trajectory events."""
        persistence.init_session("events-test")

        persistence.increment_trajectory_events(5)
        assert persistence.current_state.trajectory_events_count == 5

        persistence.increment_trajectory_events(3)
        assert persistence.current_state.trajectory_events_count == 8

    def test_list_sessions(self, persistence):
        """Can list saved sessions."""
        # Create multiple sessions
        persistence.init_session("session-1")
        persistence.save_state()

        persistence.init_session("session-2")
        persistence.save_state()

        sessions = persistence.list_sessions()

        assert "session-1" in sessions
        assert "session-2" in sessions

    def test_cleanup_old_sessions(self, persistence):
        """Can clean up old sessions."""
        # Create old session and save
        persistence.init_session("old-session")
        persistence.save_state()

        # Manually modify the saved file to have old timestamp
        state_file = persistence.get_state_file("old-session")
        with open(state_file) as f:
            data = json.load(f)
        data["updated_at"] = time.time() - (10 * 24 * 60 * 60)  # 10 days ago
        with open(state_file, "w") as f:
            json.dump(data, f)

        # Create recent session
        persistence.init_session("new-session")
        persistence.save_state()

        # Clean up sessions older than 7 days
        cleaned = persistence.cleanup_old_sessions(max_age_days=7)

        assert cleaned == 1
        sessions = persistence.list_sessions()
        assert "old-session" not in sessions
        assert "new-session" in sessions

    def test_context_persistence(self, persistence):
        """Context is saved and restored correctly."""
        persistence.init_session("context-test")

        # Add various context items
        persistence.add_message(MessageRole.USER, "Test message")
        persistence.add_file_to_cache("main.py", "print('test')")
        persistence.add_tool_output("read", "file content")

        # Save
        persistence.save_state()

        # Clear and restore
        persistence._current_state = None
        persistence._current_context = None

        persistence.restore_state("context-test")

        # Verify context restored
        assert len(persistence.current_context.messages) == 1
        assert persistence.current_context.messages[0].content == "Test message"
        assert "main.py" in persistence.current_context.files
        assert len(persistence.current_context.tool_outputs) == 1


class TestGetPersistence:
    """Tests for get_persistence function."""

    def test_returns_singleton(self, monkeypatch):
        """get_persistence returns same instance."""
        import src.state_persistence as sp

        # Reset global
        sp._persistence = None

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            p1 = get_persistence()
            p2 = get_persistence()

            assert p1 is p2

            # Clean up
            sp._persistence = None
