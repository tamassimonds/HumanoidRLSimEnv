"""Configuration helpers for SB3 PPO training."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from robottraining.envs.humanoid import HumanoidEnvConfig


@dataclass(slots=True)
class HumanoidEnvSettings:
    """Serializable view of :class:`HumanoidEnvConfig`."""

    model_path: str | None = None
    frame_skip: int = 5
    episode_length: int = 1000

    def to_env_config(self) -> HumanoidEnvConfig:
        return HumanoidEnvConfig(
            model_path=Path(self.model_path) if self.model_path else None,
            frame_skip=self.frame_skip,
            episode_length=self.episode_length,
        )


@dataclass(slots=True)
class PPOSettings:
    """Parameters forwarded to :class:`stable_baselines3.PPO`."""

    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_sb3_kwargs(self, device: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "device": device,
        }
        if self.policy_kwargs:
            kwargs["policy_kwargs"] = self.policy_kwargs
        return kwargs


@dataclass(slots=True)
class TrainerConfig:
    """Top-level training file parsed from YAML/JSON configs."""

    seed: int = 0
    total_timesteps: int = 200_000
    eval_freq: int = 10_000
    eval_episodes: int = 5
    log_interval: int = 10
    run_name: str = "ppo_humanoid"
    device: str = "auto"
    tensorboard_log: str = "runs"
    checkpoint_dir: str = "checkpoints"
    env: HumanoidEnvSettings = field(default_factory=HumanoidEnvSettings)
    ppo: PPOSettings = field(default_factory=PPOSettings)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TrainerConfig":
        if payload is None:
            return cls()
        payload = dict(payload)
        env_cfg = _coerce_dataclass(HumanoidEnvSettings, payload.pop("env", None))
        ppo_cfg = _coerce_dataclass(PPOSettings, payload.pop("ppo", None))
        return cls(env=env_cfg, ppo=ppo_cfg, **payload)

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainerConfig":
        data = _load_mapping(path)
        return cls.from_dict(data)


def resolve_device(device_pref: str) -> str:
    """Resolve the `auto` token into an available device string."""

    if device_pref != "auto":
        return device_pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - optional during linting
        return "cpu"


def _load_mapping(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    import json

    return json.loads(text)


def _coerce_dataclass(cls, payload: Mapping[str, Any] | None):
    if payload is None:
        return cls()
    if isinstance(payload, cls):
        return payload
    allowed = {field.name for field in fields(cls)}
    filtered = {k: v for k, v in payload.items() if k in allowed}
    return cls(**filtered)


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for loading and rolling out a trained policy."""

    policy_path: str | Path
    env: HumanoidEnvSettings = field(default_factory=HumanoidEnvSettings)
    episodes: int = 1
    max_steps: int = 1000
    deterministic: bool = True
    render_mode: str | None = None
    video_path: str | Path | None = None
    video_fps: int = 60
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.policy_path:
            raise ValueError("policy_path is required for evaluation")
        self.policy_path = Path(self.policy_path)
        if self.video_path is not None:
            self.video_path = Path(self.video_path)
            if self.render_mode not in (None, "rgb_array"):
                raise ValueError("Video capture requires render_mode='rgb_array'")
            if self.render_mode is None:
                self.render_mode = "rgb_array"
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "EvaluationConfig":
        if payload is None:
            raise ValueError("EvaluationConfig requires a policy_path entry")
        payload = dict(payload)
        env_cfg = _coerce_dataclass(HumanoidEnvSettings, payload.pop("env", None))
        return cls(env=env_cfg, **payload)

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationConfig":
        data = _load_mapping(path)
        return cls.from_dict(data)


__all__ = [
    "TrainerConfig",
    "HumanoidEnvSettings",
    "PPOSettings",
    "EvaluationConfig",
    "resolve_device",
]
