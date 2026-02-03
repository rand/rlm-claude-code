"""
Tests for multi-turn checkpointing (SPEC-09.20-09.26).

Tests cover:
- Expanded checkpoint state capture
- Serialization/deserialization
- Checkpoint restoration
- Version compatibility
"""

import json
import tempfile
import time
from pathlib import Path

from src.orchestrator.checkpointing import (
    CheckpointingOrchestrator,
    RLMCheckpoint,
)


class TestRLMCheckpointExpanded:
    """Tests for expanded RLMCheckpoint per SPEC-09.21."""

    def test_checkpoint_includes_working_memory(self):
        """SPEC-09.21: Checkpoint SHALL include working_memory."""
        checkpoint = RLMCheckpoint(
            session_id="test-session",
            depth=1,
            turn=5,
            messages=[{"role": "user", "content": "test"}],
            repl_state={"x": 1},
            trajectory_events=[],
            working_memory={"key": "value", "facts": ["fact1", "fact2"]},
        )
        assert checkpoint.working_memory == {"key": "value", "facts": ["fact1", "fact2"]}

    def test_checkpoint_includes_pending_operations(self):
        """SPEC-09.21: Checkpoint SHALL include pending_operations."""
        pending = [
            {"op_id": "op1", "query": "analyze this", "status": "pending"},
            {"op_id": "op2", "query": "summarize that", "status": "started"},
        ]
        checkpoint = RLMCheckpoint(
            session_id="test-session",
            depth=1,
            turn=5,
            messages=[],
            repl_state={},
            trajectory_events=[],
            pending_operations=pending,
        )
        assert checkpoint.pending_operations == pending
        assert len(checkpoint.pending_operations) == 2

    def test_checkpoint_includes_cost_so_far(self):
        """SPEC-09.21: Checkpoint SHALL include cost_so_far."""
        checkpoint = RLMCheckpoint(
            session_id="test-session",
            depth=1,
            turn=5,
            messages=[],
            repl_state={},
            trajectory_events=[],
            cost_so_far=0.0523,
        )
        assert checkpoint.cost_so_far == 0.0523

    def test_checkpoint_defaults_for_new_fields(self):
        """New fields should have sensible defaults."""
        checkpoint = RLMCheckpoint(
            session_id="test-session",
            depth=0,
            turn=0,
            messages=[],
            repl_state={},
            trajectory_events=[],
        )
        assert checkpoint.working_memory == {}
        assert checkpoint.pending_operations == []
        assert checkpoint.cost_so_far == 0.0


class TestCheckpointSerialization:
    """Tests for checkpoint serialization (SPEC-09.22)."""

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all expanded fields."""
        checkpoint = RLMCheckpoint(
            session_id="sess-123",
            depth=2,
            turn=10,
            messages=[{"role": "user", "content": "hello"}],
            repl_state={"var1": [1, 2, 3]},
            trajectory_events=[{"type": "query", "content": "test"}],
            working_memory={"context": "analysis"},
            pending_operations=[{"op_id": "pending1"}],
            cost_so_far=0.15,
        )
        data = checkpoint.to_dict()

        assert data["session_id"] == "sess-123"
        assert data["depth"] == 2
        assert data["turn"] == 10
        assert data["working_memory"] == {"context": "analysis"}
        assert data["pending_operations"] == [{"op_id": "pending1"}]
        assert data["cost_so_far"] == 0.15

    def test_to_json_serializes_correctly(self):
        """to_json should produce valid JSON."""
        checkpoint = RLMCheckpoint(
            session_id="sess-456",
            depth=1,
            turn=3,
            messages=[],
            repl_state={"test": True},
            trajectory_events=[],
            working_memory={"facts": ["a", "b"]},
            pending_operations=[],
            cost_so_far=0.05,
        )
        json_str = checkpoint.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "sess-456"
        assert parsed["working_memory"] == {"facts": ["a", "b"]}

    def test_from_dict_restores_all_fields(self):
        """from_dict should restore all fields including new ones."""
        data = {
            "session_id": "restored-sess",
            "depth": 2,
            "turn": 15,
            "messages": [{"role": "assistant", "content": "done"}],
            "repl_state": {"result": 42},
            "trajectory_events": [{"event": "complete"}],
            "working_memory": {"key1": "val1"},
            "pending_operations": [{"op": "remaining"}],
            "cost_so_far": 0.25,
            "timestamp": 1234567890.0,
            "version": "1.1",
            "metadata": {"source": "test"},
        }
        checkpoint = RLMCheckpoint.from_dict(data)

        assert checkpoint.session_id == "restored-sess"
        assert checkpoint.depth == 2
        assert checkpoint.working_memory == {"key1": "val1"}
        assert checkpoint.pending_operations == [{"op": "remaining"}]
        assert checkpoint.cost_so_far == 0.25

    def test_from_dict_handles_missing_new_fields(self):
        """from_dict should handle checkpoints without new fields (backward compat)."""
        # Old checkpoint format without new fields
        data = {
            "session_id": "old-sess",
            "depth": 1,
            "turn": 5,
            "messages": [],
            "repl_state": {},
            "trajectory_events": [],
        }
        checkpoint = RLMCheckpoint.from_dict(data)

        # Should use defaults for missing fields
        assert checkpoint.working_memory == {}
        assert checkpoint.pending_operations == []
        assert checkpoint.cost_so_far == 0.0

    def test_roundtrip_serialization(self):
        """Serialization roundtrip should preserve all data."""
        original = RLMCheckpoint(
            session_id="roundtrip-test",
            depth=3,
            turn=25,
            messages=[{"role": "user", "content": "complex query"}],
            repl_state={"analysis": {"nested": "data"}},
            trajectory_events=[{"type": "reasoning", "content": "thinking..."}],
            working_memory={"context": "large", "items": list(range(100))},
            pending_operations=[
                {"op_id": "p1", "query": "q1"},
                {"op_id": "p2", "query": "q2"},
            ],
            cost_so_far=0.456789,
            metadata={"custom": "meta"},
        )

        # Serialize and deserialize
        json_str = original.to_json()
        restored = RLMCheckpoint.from_json(json_str)

        assert restored.session_id == original.session_id
        assert restored.depth == original.depth
        assert restored.turn == original.turn
        assert restored.working_memory == original.working_memory
        assert restored.pending_operations == original.pending_operations
        assert restored.cost_so_far == original.cost_so_far


class TestCheckpointRestoration:
    """Tests for checkpoint restoration (SPEC-09.23)."""

    def test_restore_repl_state(self):
        """SPEC-09.23: System should restore REPL state."""
        checkpoint = RLMCheckpoint(
            session_id="restore-test",
            depth=1,
            turn=10,
            messages=[],
            repl_state={
                "x": 100,
                "results": [1, 2, 3],
                "analysis": {"key": "value"},
            },
            trajectory_events=[],
        )

        # Should be able to extract REPL state for restoration
        repl_state = checkpoint.get_repl_state()
        assert repl_state == {"x": 100, "results": [1, 2, 3], "analysis": {"key": "value"}}

    def test_get_pending_operations(self):
        """SPEC-09.23: System should support resuming pending operations."""
        pending = [
            {"op_id": "op1", "query": "analyze", "status": "pending"},
            {"op_id": "op2", "query": "summarize", "status": "started"},
        ]
        checkpoint = RLMCheckpoint(
            session_id="pending-test",
            depth=2,
            turn=15,
            messages=[],
            repl_state={},
            trajectory_events=[],
            pending_operations=pending,
        )

        ops = checkpoint.get_pending_operations()
        assert len(ops) == 2
        assert ops[0]["op_id"] == "op1"

    def test_get_resumable_operations(self):
        """Should filter to only resumable (not completed) operations."""
        pending = [
            {"op_id": "op1", "query": "q1", "status": "pending"},
            {"op_id": "op2", "query": "q2", "status": "completed"},
            {"op_id": "op3", "query": "q3", "status": "started"},
        ]
        checkpoint = RLMCheckpoint(
            session_id="resume-test",
            depth=1,
            turn=10,
            messages=[],
            repl_state={},
            trajectory_events=[],
            pending_operations=pending,
        )

        resumable = checkpoint.get_resumable_operations()
        assert len(resumable) == 2
        assert all(op["status"] != "completed" for op in resumable)


class TestVersionCompatibility:
    """Tests for version compatibility (SPEC-09.26)."""

    def test_checkpoint_has_version(self):
        """SPEC-09.26: Checkpoints SHALL be versioned."""
        checkpoint = RLMCheckpoint(
            session_id="version-test",
            depth=0,
            turn=0,
            messages=[],
            repl_state={},
            trajectory_events=[],
        )
        assert checkpoint.version is not None
        assert isinstance(checkpoint.version, str)

    def test_default_version(self):
        """Default version should be 1.1 for expanded format."""
        checkpoint = RLMCheckpoint(
            session_id="test",
            depth=0,
            turn=0,
            messages=[],
            repl_state={},
            trajectory_events=[],
        )
        # Version 1.1 indicates expanded format with new fields
        assert checkpoint.version == "1.1"

    def test_check_compatibility_same_version(self):
        """Same version should be compatible."""
        checkpoint = RLMCheckpoint(
            session_id="compat-test",
            depth=0,
            turn=0,
            messages=[],
            repl_state={},
            trajectory_events=[],
            version="1.1",
        )
        assert checkpoint.is_compatible("1.1")

    def test_check_compatibility_older_version(self):
        """Older major version should be compatible (can upgrade)."""
        checkpoint = RLMCheckpoint(
            session_id="old-version",
            depth=0,
            turn=0,
            messages=[],
            repl_state={},
            trajectory_events=[],
            version="1.0",
        )
        # Version 1.0 can be loaded (with defaults for missing fields)
        assert checkpoint.is_compatible("1.1")

    def test_check_compatibility_newer_major_version(self):
        """Newer major version should not be compatible."""
        checkpoint = RLMCheckpoint(
            session_id="future-version",
            depth=0,
            turn=0,
            messages=[],
            repl_state={},
            trajectory_events=[],
            version="2.0",
        )
        # Version 2.0 checkpoint is too new for 1.1 runtime
        assert not checkpoint.is_compatible("1.1")

    def test_check_compatibility_minor_version_ok(self):
        """Minor version differences should be compatible."""
        checkpoint = RLMCheckpoint(
            session_id="minor-diff",
            depth=0,
            turn=0,
            messages=[],
            repl_state={},
            trajectory_events=[],
            version="1.2",
        )
        # Minor version 1.2 should be compatible with 1.1 runtime
        assert checkpoint.is_compatible("1.1")


class TestCheckpointingOrchestratorExpanded:
    """Tests for expanded CheckpointingOrchestrator functionality."""

    def test_create_checkpoint_with_all_state(self):
        """Create checkpoint should capture all state per SPEC-09.21."""
        orchestrator = CheckpointingOrchestrator(
            checkpoint_dir=tempfile.mkdtemp(),
            auto_checkpoint_interval=5,
        )

        checkpoint = orchestrator.create_checkpoint(
            session_id="full-state-test",
            depth=2,
            turn=15,
            messages=[{"role": "user", "content": "test"}],
            repl_state={"var": 123},
            trajectory_events=[{"type": "query"}],
            working_memory={"context": "important"},
            pending_operations=[{"op_id": "op1"}],
            cost_so_far=0.35,
            metadata={"reason": "manual"},
        )

        assert checkpoint.working_memory == {"context": "important"}
        assert checkpoint.pending_operations == [{"op_id": "op1"}]
        assert checkpoint.cost_so_far == 0.35

    def test_save_and_load_expanded_checkpoint(self):
        """Save/load should preserve all expanded state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = CheckpointingOrchestrator(checkpoint_dir=tmpdir)

            # Create checkpoint with all state
            checkpoint = orchestrator.create_checkpoint(
                session_id="save-load-test",
                depth=1,
                turn=10,
                messages=[],
                repl_state={"analysis": [1, 2, 3]},
                trajectory_events=[],
                working_memory={"facts": ["fact1"]},
                pending_operations=[{"op_id": "pending"}],
                cost_so_far=0.123,
            )

            # Save
            path = orchestrator.save_checkpoint(checkpoint)
            assert path.exists()

            # Load
            loaded = orchestrator.load_checkpoint("save-load-test")
            assert loaded is not None
            assert loaded.working_memory == {"facts": ["fact1"]}
            assert loaded.pending_operations == [{"op_id": "pending"}]
            assert loaded.cost_so_far == 0.123

    def test_restore_session_returns_restoration_info(self):
        """restore_session should return info needed to resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = CheckpointingOrchestrator(checkpoint_dir=tmpdir)

            # Create and save checkpoint
            checkpoint = orchestrator.create_checkpoint(
                session_id="restore-info-test",
                depth=2,
                turn=20,
                messages=[{"role": "user", "content": "query"}],
                repl_state={"x": 42},
                trajectory_events=[{"type": "step"}],
                working_memory={"ctx": "data"},
                pending_operations=[
                    {"op_id": "p1", "status": "pending"},
                    {"op_id": "p2", "status": "completed"},
                ],
                cost_so_far=0.5,
            )
            orchestrator.save_checkpoint(checkpoint)

            # Restore
            restoration = orchestrator.restore_session("restore-info-test")

            assert restoration is not None
            assert restoration["session_id"] == "restore-info-test"
            assert restoration["depth"] == 2
            assert restoration["turn"] == 20
            assert restoration["repl_state"] == {"x": 42}
            assert restoration["working_memory"] == {"ctx": "data"}
            # Only pending/started ops, not completed
            assert len(restoration["resumable_operations"]) == 1
            assert restoration["cost_so_far"] == 0.5

    def test_restore_nonexistent_session(self):
        """restore_session for non-existent session returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = CheckpointingOrchestrator(checkpoint_dir=tmpdir)
            result = orchestrator.restore_session("nonexistent")
            assert result is None

    def test_auto_checkpoint_interval(self):
        """Auto checkpoint should trigger at specified interval."""
        orchestrator = CheckpointingOrchestrator(
            checkpoint_dir=tempfile.mkdtemp(),
            auto_checkpoint_interval=5,
        )

        assert not orchestrator.should_checkpoint(0)
        assert not orchestrator.should_checkpoint(3)
        assert orchestrator.should_checkpoint(5)
        assert not orchestrator.should_checkpoint(7)
        assert orchestrator.should_checkpoint(10)
        assert orchestrator.should_checkpoint(15)


class TestCheckpointFileOperations:
    """Tests for checkpoint file operations."""

    def test_save_creates_directory_if_needed(self):
        """Save should create checkpoint directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "checkpoints"
            orchestrator = CheckpointingOrchestrator(checkpoint_dir=nested_dir)

            checkpoint = orchestrator.create_checkpoint(
                session_id="dir-create-test",
                depth=0,
                turn=5,
                messages=[],
                repl_state={},
                trajectory_events=[],
            )

            path = orchestrator.save_checkpoint(checkpoint)
            assert path.exists()
            assert nested_dir.exists()

    def test_checkpoint_file_size_reasonable(self):
        """Checkpoint file size should be reasonable per acceptance criteria."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = CheckpointingOrchestrator(checkpoint_dir=tmpdir)

            # Create checkpoint with substantial data
            large_repl_state = {f"var_{i}": list(range(100)) for i in range(10)}
            large_trajectory = [{"type": f"event_{i}", "data": "x" * 100} for i in range(100)]

            checkpoint = orchestrator.create_checkpoint(
                session_id="size-test",
                depth=3,
                turn=50,
                messages=[{"role": "user", "content": "x" * 1000} for _ in range(20)],
                repl_state=large_repl_state,
                trajectory_events=large_trajectory,
                working_memory={"large": "x" * 10000},
                pending_operations=[{"op": f"op_{i}"} for i in range(10)],
                cost_so_far=1.5,
            )

            path = orchestrator.save_checkpoint(checkpoint)
            size_mb = path.stat().st_size / (1024 * 1024)

            # Should be under 10MB for typical checkpoint
            assert size_mb < 10, f"Checkpoint size {size_mb:.2f}MB exceeds 10MB limit"

    def test_cleanup_preserves_most_recent(self):
        """Cleanup should preserve most recent checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = CheckpointingOrchestrator(checkpoint_dir=tmpdir)

            # Create multiple checkpoints with explicit different timestamps
            base_time = time.time()
            for i in range(5):
                checkpoint = RLMCheckpoint(
                    session_id="cleanup-test",
                    depth=0,
                    turn=i * 5,
                    messages=[],
                    repl_state={},
                    trajectory_events=[],
                    timestamp=base_time + i,  # Distinct timestamps
                )
                orchestrator.save_checkpoint(checkpoint)

            # Should have 5 checkpoints
            assert len(orchestrator.list_checkpoints("cleanup-test")) == 5

            # Cleanup, keep 2
            removed = orchestrator.cleanup_old_checkpoints("cleanup-test", keep_count=2)
            assert removed == 3
            assert len(orchestrator.list_checkpoints("cleanup-test")) == 2
