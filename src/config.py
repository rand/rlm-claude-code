"""
Configuration management for RLM-Claude-Code.

Implements: Spec §5.3 Router Configuration
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .epistemic.types import VerificationConfig as VerificationConfigType


@dataclass
class ActivationConfig:
    """
    Configuration for RLM activation.

    Implements: SPEC-14.10-14.15 for always-on micro mode.

    Modes:
    - "micro": Always-on with minimal cost, starts at micro level (SPEC-14.12 default)
    - "complexity": Original heuristic-based activation
    - "always": Always full RLM
    - "manual": Only when explicitly enabled
    - "token": Activate above token threshold
    """

    mode: Literal["micro", "complexity", "always", "manual", "token"] = "micro"
    fallback_token_threshold: int = 80000
    complexity_score_threshold: int = 2
    # SPEC-14.30: Fast-path bypass configuration
    fast_path_enabled: bool = True
    # SPEC-14.20: Escalation configuration
    escalation_enabled: bool = True
    # SPEC-14.62: Session token budget
    session_budget_tokens: int = 500_000


@dataclass
class DepthConfig:
    """Configuration for recursive depth."""

    default: int = 2
    max: int = 3
    spawn_repl_at_depth_1: bool = True


@dataclass
class HybridConfig:
    """Configuration for hybrid mode."""

    enabled: bool = True
    simple_query_bypass: bool = True
    simple_confidence_threshold: float = 0.95


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory output."""

    verbosity: Literal["minimal", "normal", "verbose", "debug"] = "normal"
    streaming: bool = True
    colors: bool = True
    export_enabled: bool = True
    export_path: str = "~/.claude/rlm-trajectories/"


@dataclass
class ModelConfig:
    """Configuration for model selection by depth."""

    root_model: str = "opus"  # Default to Opus
    recursive_depth_1: str = "sonnet"
    recursive_depth_2: str = "haiku"
    # Alternative models for OpenAI routing
    openai_root: str = "gpt-5.2-codex"
    openai_recursive: str = "gpt-4o-mini"


@dataclass
class CostConfig:
    """Configuration for cost controls."""

    max_recursive_calls_per_turn: int = 10
    max_tokens_per_recursive_call: int = 8000
    abort_on_cost_threshold: int = 50000  # tokens


def _default_verification_config() -> "VerificationConfigType":
    """Create default VerificationConfig. Implements: SPEC-16.21"""
    from .epistemic.types import VerificationConfig

    return VerificationConfig()


@dataclass
class RLMConfig:
    """
    Complete RLM configuration.

    Implements: Spec §5.3 Router Configuration
    """

    activation: ActivationConfig = field(default_factory=ActivationConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    cost_controls: CostConfig = field(default_factory=CostConfig)
    # SPEC-16.21: Epistemic verification configuration
    verification: "VerificationConfigType" = field(default_factory=_default_verification_config)

    @classmethod
    def load(cls, path: Path | None = None) -> "RLMConfig":
        """Load configuration from file."""
        if path is None:
            path = Path.home() / ".claude" / "rlm-config.json"

        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        # Backward compatibility: migrate old field names
        models_data = data.get("models", {})
        if "root" in models_data and "root_model" not in models_data:
            models_data["root_model"] = models_data.pop("root")

        # SPEC-16.21: Load verification config if present
        from .epistemic.types import VerificationConfig

        verification_data = data.get("verification", {})
        verification = (
            VerificationConfig(**verification_data) if verification_data else VerificationConfig()
        )

        return cls(
            activation=ActivationConfig(**data.get("activation", {})),
            depth=DepthConfig(**data.get("depth", {})),
            hybrid=HybridConfig(**data.get("hybrid", {})),
            trajectory=TrajectoryConfig(**data.get("trajectory", {})),
            models=ModelConfig(**models_data),
            cost_controls=CostConfig(**data.get("cost_controls", {})),
            verification=verification,
        )

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = Path.home() / ".claude" / "rlm-config.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        # SPEC-16.21: Serialize verification config
        # Need to convert dataclass to dict, handling any nested types
        verification_dict = {
            "enabled": self.verification.enabled,
            "support_threshold": self.verification.support_threshold,
            "dependence_threshold": self.verification.dependence_threshold,
            "gap_threshold_bits": self.verification.gap_threshold_bits,
            "on_failure": self.verification.on_failure,
            "max_retries": self.verification.max_retries,
            "verification_model": self.verification.verification_model,
            "critical_model": self.verification.critical_model,
            "max_claims_per_response": self.verification.max_claims_per_response,
            "parallel_verification": self.verification.parallel_verification,
            "mode": self.verification.mode,
            "sample_rate": self.verification.sample_rate,
        }

        with open(path, "w") as f:
            json.dump(
                {
                    "activation": self.activation.__dict__,
                    "depth": self.depth.__dict__,
                    "hybrid": self.hybrid.__dict__,
                    "trajectory": self.trajectory.__dict__,
                    "models": self.models.__dict__,
                    "cost_controls": self.cost_controls.__dict__,
                    "verification": verification_dict,
                },
                f,
                indent=2,
            )


# Default configuration instance
default_config = RLMConfig()
