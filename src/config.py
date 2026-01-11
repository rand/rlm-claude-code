"""
Configuration management for RLM-Claude-Code.

Implements: Spec §5.3 Router Configuration
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ActivationConfig:
    """Configuration for RLM activation."""

    mode: Literal["complexity", "always", "manual", "token"] = "complexity"
    fallback_token_threshold: int = 80000
    complexity_score_threshold: int = 2


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

        return cls(
            activation=ActivationConfig(**data.get("activation", {})),
            depth=DepthConfig(**data.get("depth", {})),
            hybrid=HybridConfig(**data.get("hybrid", {})),
            trajectory=TrajectoryConfig(**data.get("trajectory", {})),
            models=ModelConfig(**models_data),
            cost_controls=CostConfig(**data.get("cost_controls", {})),
        )

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = Path.home() / ".claude" / "rlm-config.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {
                    "activation": self.activation.__dict__,
                    "depth": self.depth.__dict__,
                    "hybrid": self.hybrid.__dict__,
                    "trajectory": self.trajectory.__dict__,
                    "models": self.models.__dict__,
                    "cost_controls": self.cost_controls.__dict__,
                },
                f,
                indent=2,
            )


# Default configuration instance
default_config = RLMConfig()
