"""
Unit tests for orchestration decision logging.

Tests logging, export, and analysis of orchestration decisions
for training data collection.
"""

import json
from pathlib import Path

import pytest

from src.orchestration_logger import (
    LoggerConfig,
    OrchestrationDecisionLog,
    OrchestrationLogger,
    TrainingDataExporter,
    get_logger,
    set_logger,
)


class TestLoggerConfig:
    """Tests for LoggerConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = LoggerConfig()
        assert config.enabled is True
        assert config.log_heuristics is True
        assert config.log_cache_hits is False
        assert config.max_size_mb == 100.0

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = LoggerConfig(
            log_path="/custom/path.jsonl",
            enabled=False,
            log_heuristics=False,
        )
        assert config.log_path == "/custom/path.jsonl"
        assert config.enabled is False
        assert config.log_heuristics is False


class TestOrchestrationDecisionLog:
    """Tests for OrchestrationDecisionLog dataclass."""

    def test_create_log_entry(self):
        """Can create a log entry."""
        entry = OrchestrationDecisionLog(
            query="test query",
            query_length=10,
            context_tokens=5000,
            context_summary="test context",
            activate_rlm=True,
            activation_reason="discovery",
            execution_mode="balanced",
            model_tier="balanced",
            depth_budget=2,
            tool_access="read_only",
            query_type="analytical",
            complexity_score=0.7,
            confidence=0.8,
            signals=["discovery_required"],
            timestamp="2025-01-01T00:00:00",
            source="api",
            latency_ms=150.0,
        )
        assert entry.query == "test query"
        assert entry.activate_rlm is True
        assert entry.signals == ["discovery_required"]

    def test_to_dict(self):
        """Entry converts to dict."""
        entry = OrchestrationDecisionLog(
            query="test",
            query_length=4,
            context_tokens=1000,
            context_summary="ctx",
            activate_rlm=False,
            activation_reason="simple",
            execution_mode="fast",
            model_tier="fast",
            depth_budget=0,
            tool_access="none",
            query_type="factual",
            complexity_score=0.2,
            confidence=0.9,
            signals=[],
            timestamp="2025-01-01",
            source="heuristic",
            latency_ms=5.0,
        )
        d = entry.to_dict()
        assert d["query"] == "test"
        assert d["activate_rlm"] is False
        assert d["source"] == "heuristic"

    def test_from_dict(self):
        """Entry can be created from dict."""
        data = {
            "query": "test",
            "query_length": 4,
            "context_tokens": 1000,
            "context_summary": "ctx",
            "activate_rlm": True,
            "activation_reason": "test",
            "execution_mode": "balanced",
            "model_tier": "balanced",
            "depth_budget": 2,
            "tool_access": "read_only",
            "query_type": "code",
            "complexity_score": 0.5,
            "confidence": 0.7,
            "signals": ["multi_file"],
            "timestamp": "2025-01-01",
            "source": "api",
            "latency_ms": 100.0,
        }
        entry = OrchestrationDecisionLog.from_dict(data)
        assert entry.query == "test"
        assert entry.activate_rlm is True
        assert entry.signals == ["multi_file"]


class TestOrchestrationLogger:
    """Tests for OrchestrationLogger."""

    @pytest.fixture
    def temp_log_path(self, tmp_path):
        """Create temporary log path."""
        return str(tmp_path / "test_decisions.jsonl")

    @pytest.fixture
    def logger(self, temp_log_path):
        """Create logger with temp path."""
        config = LoggerConfig(log_path=temp_log_path, enabled=True)
        return OrchestrationLogger(config=config)

    def test_initialization(self, logger, temp_log_path):
        """Logger initializes correctly."""
        assert logger.config.enabled is True
        assert logger._decision_count == 0

    def test_log_decision(self, logger, temp_log_path):
        """Logs a decision to file."""
        decision = {
            "activate_rlm": True,
            "activation_reason": "test",
            "execution_mode": "balanced",
            "model_tier": "balanced",
            "depth_budget": 2,
            "tool_access": "read_only",
            "query_type": "analytical",
            "complexity_score": 0.6,
            "confidence": 0.8,
            "signals": ["discovery_required"],
        }

        logger.log_decision(
            query="test query",
            context_summary="test context",
            decision=decision,
            source="api",
            latency_ms=150.0,
            context_tokens=5000,
        )

        assert logger._decision_count == 1

        # Verify file contents
        with open(temp_log_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["query"] == "test query"
            assert data["activate_rlm"] is True
            assert data["source"] == "api"

    def test_skip_heuristics_when_disabled(self, temp_log_path):
        """Skips heuristic decisions when configured."""
        config = LoggerConfig(
            log_path=temp_log_path,
            enabled=True,
            log_heuristics=False,
        )
        logger = OrchestrationLogger(config=config)

        logger.log_decision(
            query="test",
            context_summary="ctx",
            decision={"activate_rlm": False},
            source="heuristic",
        )

        assert logger._decision_count == 0

    def test_skip_cache_hits_by_default(self, logger):
        """Skips cache hits by default."""
        logger.log_decision(
            query="test",
            context_summary="ctx",
            decision={"activate_rlm": True},
            source="cache",
        )

        assert logger._decision_count == 0

    def test_disabled_logger_no_op(self, temp_log_path):
        """Disabled logger does nothing."""
        config = LoggerConfig(log_path=temp_log_path, enabled=False)
        logger = OrchestrationLogger(config=config)

        logger.log_decision(
            query="test",
            context_summary="ctx",
            decision={"activate_rlm": True},
            source="api",
        )

        assert logger._decision_count == 0
        assert not Path(temp_log_path).exists()

    def test_log_outcome_feedback(self, logger, temp_log_path):
        """Logs outcome feedback."""
        logger.log_outcome(
            query="test query",
            success=True,
            feedback="worked well",
        )

        with open(temp_log_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["type"] == "outcome_feedback"
            assert data["success"] is True

    def test_get_statistics(self, logger, temp_log_path):
        """Gets logging statistics."""
        # Log some decisions
        for i in range(3):
            logger.log_decision(
                query=f"query {i}",
                context_summary="ctx",
                decision={"activate_rlm": True},
                source="api",
            )

        stats = logger.get_statistics()
        assert stats["enabled"] is True
        assert stats["decisions_logged"] == 3
        assert stats["log_lines"] == 3


class TestTrainingDataExporter:
    """Tests for TrainingDataExporter."""

    @pytest.fixture
    def populated_log(self, tmp_path):
        """Create log with sample data."""
        log_path = tmp_path / "decisions.jsonl"

        entries = [
            {
                "query": "Why is this failing?",
                "query_length": 20,
                "context_tokens": 5000,
                "context_summary": "ctx",
                "activate_rlm": True,
                "activation_reason": "discovery",
                "execution_mode": "balanced",
                "model_tier": "balanced",
                "depth_budget": 2,
                "tool_access": "read_only",
                "query_type": "debugging",
                "complexity_score": 0.7,
                "confidence": 0.8,
                "signals": ["discovery_required"],
                "timestamp": "2025-01-01",
                "source": "api",
                "latency_ms": 150.0,
                "session_id": "test",
                "model_used": "haiku",
            },
            {
                "query": "show config.py",
                "query_length": 14,
                "context_tokens": 1000,
                "context_summary": "ctx",
                "activate_rlm": False,
                "activation_reason": "simple",
                "execution_mode": "fast",
                "model_tier": "fast",
                "depth_budget": 0,
                "tool_access": "none",
                "query_type": "factual",
                "complexity_score": 0.1,
                "confidence": 0.95,
                "signals": [],
                "timestamp": "2025-01-01",
                "source": "heuristic",
                "latency_ms": 5.0,
                "session_id": "test",
                "model_used": "",
            },
        ]

        with open(log_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return str(log_path)

    def test_load_decisions(self, populated_log):
        """Loads decisions from log file."""
        exporter = TrainingDataExporter(populated_log)
        decisions = exporter.load_decisions()

        assert len(decisions) == 2
        assert decisions[0].query == "Why is this failing?"
        assert decisions[0].activate_rlm is True
        assert decisions[1].query == "show config.py"
        assert decisions[1].activate_rlm is False

    def test_export_csv(self, populated_log, tmp_path):
        """Exports to CSV format."""
        exporter = TrainingDataExporter(populated_log)
        output_path = str(tmp_path / "export.csv")

        count = exporter.export_csv(output_path)

        assert count == 2
        assert Path(output_path).exists()

        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 3  # Header + 2 rows
            assert "query" in lines[0]  # Header

    def test_export_huggingface(self, populated_log, tmp_path):
        """Exports to HuggingFace format."""
        exporter = TrainingDataExporter(populated_log)
        output_dir = str(tmp_path / "hf_dataset")

        count = exporter.export_huggingface(output_dir)

        assert count == 2
        assert (Path(output_dir) / "train.jsonl").exists()
        assert (Path(output_dir) / "test.jsonl").exists()
        assert (Path(output_dir) / "dataset_info.json").exists()

        # Check dataset info
        with open(Path(output_dir) / "dataset_info.json") as f:
            info = json.load(f)
            assert info["total_examples"] == 2

    def test_get_label_distribution(self, populated_log):
        """Gets label distribution for analysis."""
        exporter = TrainingDataExporter(populated_log)
        dist = exporter.get_label_distribution()

        assert dist["activate_rlm"]["true"] == 1
        assert dist["activate_rlm"]["false"] == 1
        assert dist["source"]["api"] == 1
        assert dist["source"]["heuristic"] == 1

    def test_export_setfit_binary(self, populated_log, tmp_path):
        """Exports to SetFit binary format."""
        exporter = TrainingDataExporter(populated_log)
        output_path = str(tmp_path / "setfit_binary.jsonl")

        count = exporter.export_setfit(output_path, label_type="binary")

        assert count == 2
        assert Path(output_path).exists()

        # Check format
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 2

            # First entry should be activate_rlm=True -> label=1
            entry1 = json.loads(lines[0])
            assert "text" in entry1
            assert entry1["label"] == 1

            # Second entry should be activate_rlm=False -> label=0
            entry2 = json.loads(lines[1])
            assert entry2["label"] == 0

        # Check metadata file
        meta_path = tmp_path / "setfit_binary.meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
            assert meta["format"] == "setfit"
            assert meta["label_type"] == "binary"
            assert meta["total_examples"] == 2

    def test_export_setfit_mode(self, populated_log, tmp_path):
        """Exports to SetFit mode format (3-class)."""
        exporter = TrainingDataExporter(populated_log)
        output_path = str(tmp_path / "setfit_mode.jsonl")

        count = exporter.export_setfit(output_path, label_type="mode")

        assert count == 2
        assert Path(output_path).exists()

        with open(output_path) as f:
            lines = f.readlines()
            entry1 = json.loads(lines[0])
            assert entry1["label"] in ["fast", "balanced", "thorough"]

    def test_export_setfit_complexity(self, populated_log, tmp_path):
        """Exports to SetFit complexity format (5-class)."""
        exporter = TrainingDataExporter(populated_log)
        output_path = str(tmp_path / "setfit_complexity.jsonl")

        count = exporter.export_setfit(output_path, label_type="complexity")

        assert count == 2
        assert Path(output_path).exists()

        with open(output_path) as f:
            lines = f.readlines()
            entry1 = json.loads(lines[0])
            assert entry1["label"] in ["trivial", "simple", "moderate", "complex", "unbounded"]


class TestGlobalLogger:
    """Tests for global logger functions."""

    def test_get_logger_creates_default(self):
        """get_logger creates default logger."""
        # Reset global
        set_logger(None)  # type: ignore

        logger = get_logger()
        assert logger is not None
        assert isinstance(logger, OrchestrationLogger)

    def test_set_logger_overrides(self, tmp_path):
        """set_logger overrides global logger."""
        custom_config = LoggerConfig(
            log_path=str(tmp_path / "custom.jsonl"),
            session_id="custom_session",
        )
        custom_logger = OrchestrationLogger(config=custom_config)

        set_logger(custom_logger)

        assert get_logger() is custom_logger
        assert get_logger().config.session_id == "custom_session"


class TestLogRotation:
    """Tests for log file rotation."""

    def test_rotation_triggers_at_size(self, tmp_path):
        """Log rotates when size limit reached."""
        log_path = tmp_path / "test.jsonl"

        # Create small size limit
        config = LoggerConfig(
            log_path=str(log_path),
            max_size_mb=0.0001,  # Very small, ~100 bytes
            max_files=3,
        )
        logger = OrchestrationLogger(config=config)

        # Log enough to trigger rotation
        for i in range(10):
            logger.log_decision(
                query=f"query {i} " * 50,  # Make it bigger
                context_summary="ctx",
                decision={"activate_rlm": True},
                source="api",
            )

        # Should have rotated files
        rotated_files = list(tmp_path.glob("test.jsonl.*"))
        assert len(rotated_files) >= 1


class TestIntegrationWithOrchestrator:
    """Integration tests with IntelligentOrchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_logs_decisions(self, tmp_path):
        """IntelligentOrchestrator logs decisions when enabled."""
        from unittest.mock import AsyncMock, patch

        from src.intelligent_orchestrator import (
            IntelligentOrchestrator,
            OrchestratorConfig,
        )
        from src.orchestration_schema import OrchestrationContext

        log_path = str(tmp_path / "decisions.jsonl")

        config = OrchestratorConfig(
            log_decisions=True,
            log_path=log_path,
            use_fallback=True,  # Will use heuristics on failure
        )
        orchestrator = IntelligentOrchestrator(config=config)

        context = OrchestrationContext(
            query="Why is this failing?",
            context_tokens=5000,
        )

        # Force heuristic path by making API call fail
        with patch.object(orchestrator, "_llm_orchestrate", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("Forced failure to trigger heuristics")
            await orchestrator.create_plan("Why is this failing?", context)

        # Check log was created
        assert Path(log_path).exists()
        with open(log_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["query"] == "Why is this failing?"
            assert data["source"] == "heuristic"

    @pytest.mark.asyncio
    async def test_orchestrator_respects_disabled_logging(self, tmp_path):
        """IntelligentOrchestrator respects disabled logging."""
        from src.intelligent_orchestrator import (
            IntelligentOrchestrator,
            OrchestratorConfig,
        )
        from src.orchestration_schema import OrchestrationContext

        log_path = str(tmp_path / "decisions.jsonl")

        config = OrchestratorConfig(
            log_decisions=False,  # Disabled
            log_path=log_path,
            use_fallback=True,
        )
        orchestrator = IntelligentOrchestrator(config=config)

        context = OrchestrationContext(
            query="simple query",
            context_tokens=1000,
        )

        orchestrator._client = None
        await orchestrator.create_plan("simple query", context)

        # Log should not exist
        assert not Path(log_path).exists()
