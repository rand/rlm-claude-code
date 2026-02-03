"""
Unit tests for config module.

Implements: Spec §5.3 Router Configuration tests
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    ActivationConfig,
    CostConfig,
    DepthConfig,
    HybridConfig,
    ModelConfig,
    RLMConfig,
    TrajectoryConfig,
    default_config,
)


class TestActivationConfig:
    """Tests for ActivationConfig."""

    def test_default_values(self):
        """Has expected default values (SPEC-14.10-14.15)."""
        config = ActivationConfig()

        # SPEC-14.11: Default mode is micro for always-on RLM
        assert config.mode == "micro"
        assert config.fallback_token_threshold == 80000
        assert config.complexity_score_threshold == 2
        # SPEC-14.30: Fast-path bypass enabled by default
        assert config.fast_path_enabled is True
        # SPEC-14.20: Escalation enabled by default
        assert config.escalation_enabled is True
        # SPEC-14.62: Session token budget
        assert config.session_budget_tokens == 500_000

    def test_custom_values(self):
        """Can create with custom values."""
        config = ActivationConfig(
            mode="always",
            fallback_token_threshold=50000,
            complexity_score_threshold=3,
        )

        assert config.mode == "always"
        assert config.fallback_token_threshold == 50000
        assert config.complexity_score_threshold == 3

    def test_mode_options(self):
        """All mode options are valid."""
        for mode in ["micro", "complexity", "always", "manual", "token"]:
            config = ActivationConfig(mode=mode)
            assert config.mode == mode


class TestDepthConfig:
    """Tests for DepthConfig."""

    def test_default_values(self):
        """Has expected default values."""
        config = DepthConfig()

        assert config.default == 2
        assert config.max == 3
        assert config.spawn_repl_at_depth_1 is True

    def test_custom_values(self):
        """Can create with custom values."""
        config = DepthConfig(
            default=1,
            max=5,
            spawn_repl_at_depth_1=False,
        )

        assert config.default == 1
        assert config.max == 5
        assert config.spawn_repl_at_depth_1 is False


class TestHybridConfig:
    """Tests for HybridConfig."""

    def test_default_values(self):
        """Has expected default values."""
        config = HybridConfig()

        assert config.enabled is True
        assert config.simple_query_bypass is True
        assert config.simple_confidence_threshold == 0.95

    def test_custom_values(self):
        """Can create with custom values."""
        config = HybridConfig(
            enabled=False,
            simple_query_bypass=False,
            simple_confidence_threshold=0.8,
        )

        assert config.enabled is False
        assert config.simple_query_bypass is False
        assert config.simple_confidence_threshold == 0.8


class TestTrajectoryConfig:
    """Tests for TrajectoryConfig."""

    def test_default_values(self):
        """Has expected default values."""
        config = TrajectoryConfig()

        assert config.verbosity == "normal"
        assert config.streaming is True
        assert config.colors is True
        assert config.export_enabled is True
        assert config.export_path == "~/.claude/rlm-trajectories/"

    def test_verbosity_options(self):
        """All verbosity options are valid."""
        for verbosity in ["minimal", "normal", "verbose", "debug"]:
            config = TrajectoryConfig(verbosity=verbosity)
            assert config.verbosity == verbosity


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Has expected default model values."""
        config = ModelConfig()

        assert config.root_model == "opus"
        assert config.recursive_depth_1 == "sonnet"
        assert config.recursive_depth_2 == "haiku"

    def test_custom_values(self):
        """Can create with custom model values."""
        config = ModelConfig(
            root_model="custom-root",
            recursive_depth_1="custom-d1",
            recursive_depth_2="custom-d2",
        )

        assert config.root_model == "custom-root"
        assert config.recursive_depth_1 == "custom-d1"
        assert config.recursive_depth_2 == "custom-d2"


class TestCostConfig:
    """Tests for CostConfig."""

    def test_default_values(self):
        """Has expected default cost values."""
        config = CostConfig()

        assert config.max_recursive_calls_per_turn == 10
        assert config.max_tokens_per_recursive_call == 8000
        assert config.abort_on_cost_threshold == 50000

    def test_custom_values(self):
        """Can create with custom cost values."""
        config = CostConfig(
            max_recursive_calls_per_turn=5,
            max_tokens_per_recursive_call=4000,
            abort_on_cost_threshold=25000,
        )

        assert config.max_recursive_calls_per_turn == 5
        assert config.max_tokens_per_recursive_call == 4000
        assert config.abort_on_cost_threshold == 25000


class TestRLMConfig:
    """Tests for RLMConfig."""

    def test_default_values(self):
        """Has all sub-configs with defaults."""
        config = RLMConfig()

        assert isinstance(config.activation, ActivationConfig)
        assert isinstance(config.depth, DepthConfig)
        assert isinstance(config.hybrid, HybridConfig)
        assert isinstance(config.trajectory, TrajectoryConfig)
        assert isinstance(config.models, ModelConfig)
        assert isinstance(config.cost_controls, CostConfig)

    def test_custom_sub_configs(self):
        """Can create with custom sub-configs."""
        config = RLMConfig(
            activation=ActivationConfig(mode="always"),
            depth=DepthConfig(max=5),
        )

        assert config.activation.mode == "always"
        assert config.depth.max == 5

    def test_load_nonexistent_returns_defaults(self):
        """Load from nonexistent file returns defaults."""
        config = RLMConfig.load(Path("/nonexistent/path/config.json"))

        # SPEC-14.11: Default mode is micro
        assert config.activation.mode == "micro"
        assert config.depth.max == 3

    def test_load_from_file(self):
        """Can load config from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "activation": {"mode": "always"},
                    "depth": {"max": 5, "default": 3},
                    "models": {"root_model": "custom-model"},
                },
                f,
            )
            f.flush()
            config_path = Path(f.name)

        try:
            config = RLMConfig.load(config_path)

            assert config.activation.mode == "always"
            assert config.depth.max == 5
            assert config.depth.default == 3
            assert config.models.root_model == "custom-model"
            # Non-specified values should be defaults
            assert config.hybrid.enabled is True
        finally:
            config_path.unlink()

    def test_save_and_load_roundtrip(self):
        """Config survives save/load roundtrip."""
        original = RLMConfig(
            activation=ActivationConfig(mode="manual", complexity_score_threshold=5),
            depth=DepthConfig(max=4, spawn_repl_at_depth_1=False),
            hybrid=HybridConfig(enabled=False),
            trajectory=TrajectoryConfig(verbosity="debug"),
            models=ModelConfig(root_model="test-model"),
            cost_controls=CostConfig(max_recursive_calls_per_turn=20),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test-config.json"
            original.save(config_path)
            loaded = RLMConfig.load(config_path)

        assert loaded.activation.mode == "manual"
        assert loaded.activation.complexity_score_threshold == 5
        assert loaded.depth.max == 4
        assert loaded.depth.spawn_repl_at_depth_1 is False
        assert loaded.hybrid.enabled is False
        assert loaded.trajectory.verbosity == "debug"
        assert loaded.models.root_model == "test-model"
        assert loaded.cost_controls.max_recursive_calls_per_turn == 20

    def test_save_creates_parent_dirs(self):
        """Save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nested" / "dir" / "config.json"
            config = RLMConfig()
            config.save(config_path)

            assert config_path.exists()

    def test_partial_config_file(self):
        """Can load config with only some sections."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "depth": {"max": 10},
                },
                f,
            )
            f.flush()
            config_path = Path(f.name)

        try:
            config = RLMConfig.load(config_path)

            assert config.depth.max == 10
            # Other sections should be defaults (SPEC-14.11: micro mode)
            assert config.activation.mode == "micro"
            assert config.models.root_model == "opus"
        finally:
            config_path.unlink()


class TestDefaultConfig:
    """Tests for default_config global instance."""

    def test_default_config_exists(self):
        """default_config is an RLMConfig instance."""
        assert isinstance(default_config, RLMConfig)

    def test_default_config_has_expected_values(self):
        """default_config has expected default values (SPEC-14.11)."""
        # SPEC-14.11: Default mode is micro for always-on RLM
        assert default_config.activation.mode == "micro"
        assert default_config.depth.max == 3
        assert default_config.models.root_model == "opus"


class TestVerificationConfig:
    """Tests for VerificationConfig in RLMConfig. Implements: SPEC-16.21"""

    def test_rlm_config_has_verification_field(self):
        """RLMConfig includes verification configuration."""
        from src.epistemic.types import VerificationConfig

        config = RLMConfig()
        assert hasattr(config, "verification")
        assert isinstance(config.verification, VerificationConfig)

    def test_verification_default_values(self):
        """Verification config has expected default values."""
        config = RLMConfig()

        assert config.verification.enabled is True
        assert config.verification.support_threshold == 0.7
        assert config.verification.dependence_threshold == 0.3
        assert config.verification.gap_threshold_bits == 2.0
        assert config.verification.on_failure == "retry"
        assert config.verification.verification_model == "haiku"
        assert config.verification.critical_model == "sonnet"
        assert config.verification.max_retries == 2
        assert config.verification.mode == "sample"

    def test_verification_load_from_file(self):
        """Can load verification config from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "verification": {
                        "enabled": False,
                        "support_threshold": 0.8,
                        "on_failure": "flag",
                        "max_retries": 5,
                    }
                },
                f,
            )
            f.flush()
            config_path = Path(f.name)

        try:
            config = RLMConfig.load(config_path)

            assert config.verification.enabled is False
            assert config.verification.support_threshold == 0.8
            assert config.verification.on_failure == "flag"
            assert config.verification.max_retries == 5
            # Non-specified values should be defaults
            assert config.verification.dependence_threshold == 0.3
        finally:
            config_path.unlink()

    def test_verification_save_and_load_roundtrip(self):
        """Verification config survives save/load roundtrip."""
        from src.epistemic.types import VerificationConfig

        original = RLMConfig(
            verification=VerificationConfig(
                enabled=False,
                support_threshold=0.9,
                dependence_threshold=0.4,
                gap_threshold_bits=3.0,
                on_failure="ask",
                max_retries=3,
                verification_model="sonnet",
                critical_model="opus",
                max_claims_per_response=20,
                parallel_verification=False,
                mode="full",
                sample_rate=0.5,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test-config.json"
            original.save(config_path)
            loaded = RLMConfig.load(config_path)

        assert loaded.verification.enabled is False
        assert loaded.verification.support_threshold == 0.9
        assert loaded.verification.dependence_threshold == 0.4
        assert loaded.verification.gap_threshold_bits == 3.0
        assert loaded.verification.on_failure == "ask"
        assert loaded.verification.max_retries == 3
        assert loaded.verification.verification_model == "sonnet"
        assert loaded.verification.critical_model == "opus"
        assert loaded.verification.max_claims_per_response == 20
        assert loaded.verification.parallel_verification is False
        assert loaded.verification.mode == "full"
        assert loaded.verification.sample_rate == 0.5

    def test_verification_absent_in_file_uses_defaults(self):
        """Loading config without verification section uses defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "activation": {"mode": "always"},
                },
                f,
            )
            f.flush()
            config_path = Path(f.name)

        try:
            config = RLMConfig.load(config_path)

            # Verification should be defaults
            assert config.verification.enabled is True
            assert config.verification.support_threshold == 0.7
            assert config.verification.on_failure == "retry"
        finally:
            config_path.unlink()

    def test_default_config_has_verification(self):
        """default_config includes verification configuration."""
        from src.epistemic.types import VerificationConfig

        assert hasattr(default_config, "verification")
        assert isinstance(default_config.verification, VerificationConfig)
        assert default_config.verification.enabled is True
