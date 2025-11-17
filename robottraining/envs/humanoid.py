"""Mujoco humanoid environment with reward abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium.utils import seeding
import mujoco
import numpy as np

from robottraining.rewards import (
    ControlEffortPenalty,
    ForwardVelocityReward,
    RewardAggregator,
    RewardTerm,
    UprightPostureReward,
)
from robottraining.terminations import (
    FallOverCondition,
    HorizontalDriftCondition,
    StopCondition,
    StopConditionSet,
    TiltCondition,
)


@dataclass(slots=True)
class HumanoidEnvConfig:
    """Configuration for :class:`HumanoidEnv`."""

    model_path: Optional[Path | str] = None
    frame_skip: int = 5
    episode_length: int = 1000
    reward_terms: Optional[Sequence[RewardTerm]] = None
    termination_conditions: Optional[Sequence[StopCondition]] = None

    def __post_init__(self) -> None:
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")


InfoDict = Dict[str, tuple[str, ...] | Dict[str, float]]


class HumanoidEnv(gym.Env[np.ndarray, np.ndarray]):
    """Simple Mujoco humanoid setup running on a flat plane."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config: Optional[HumanoidEnvConfig] = None, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.config = config or HumanoidEnvConfig()
        self.render_mode = render_mode

        xml = self._load_model_xml(self.config.model_path)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._renderer: Optional[mujoco.Renderer] = None

        self._torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if self._torso_body_id < 0:
            raise RuntimeError("Torso body 'torso' not found in model")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_size = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        self.reward_aggregator = RewardAggregator(self._resolve_reward_terms(self.config.reward_terms))
        self.stop_conditions = StopConditionSet(
            self._resolve_stop_conditions(self.config.termination_conditions)
        )
        self.current_step = 0
        self.np_random, _ = seeding.np_random(None)
        self._last_stop_triggers: Tuple[str, ...] = ()

    @staticmethod
    def _load_model_xml(path: Optional[Path | str]) -> str:
        if path is not None:
            return Path(path).read_text()
        asset = resources.files("robottraining.assets").joinpath("flat_humanoid.xml")
        return asset.read_text()

    @staticmethod
    def _resolve_reward_terms(custom_terms: Optional[Sequence[RewardTerm]]) -> Sequence[RewardTerm]:
        if custom_terms:
            return list(custom_terms)
        return [
            ForwardVelocityReward(target_velocity=1.5, weight=1.0),
            UprightPostureReward(weight=0.5),
            ControlEffortPenalty(weight=0.05),
        ]

    @staticmethod
    def _resolve_stop_conditions(
        custom_conditions: Optional[Sequence[StopCondition]]
    ) -> Sequence[StopCondition]:
        if custom_conditions:
            return list(custom_conditions)
        return [
            FallOverCondition(min_height=0.6),
            TiltCondition(min_dot=0.2),
            HorizontalDriftCondition(max_radius=15.0),
        ]

    @property
    def torso_body_id(self) -> int:
        return self._torso_body_id

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, InfoDict]:
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self._last_stop_triggers = ()
        noise_mag = options.get("init_noise", 0.01) if options else 0.01
        qpos_noise = self.np_random.normal(scale=noise_mag, size=self.model.nq)
        qvel_noise = self.np_random.normal(scale=noise_mag, size=self.model.nv)
        self.data.qpos[:] = self.data.qpos + qpos_noise
        self.data.qvel[:] = qvel_noise
        observation = self._get_observation()
        info: InfoDict = {
            "reward_terms": {term.name: 0.0 for term in self.reward_aggregator.terms},
            "terminations": self._last_stop_triggers,
        }
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, InfoDict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self.config.frame_skip):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        observation = self._get_observation()
        reward, breakdown = self.reward_aggregator.evaluate(self, action)
        terminated = self._update_stop_conditions()
        truncated = self.current_step >= self.config.episode_length
        info: InfoDict = {
            "reward_terms": breakdown,
            "terminations": self._last_stop_triggers,
        }
        return observation, reward, terminated, truncated, info

    def _update_stop_conditions(self) -> bool:
        terminated, triggers = self.stop_conditions.evaluate(self)
        self._last_stop_triggers = triggers
        return terminated

    def _get_observation(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.ravel(), self.data.qvel.ravel()])

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            renderer = self._ensure_renderer()
            renderer.update_scene(self.data)
            return renderer.render()
        if self.render_mode == "human":
            renderer = self._ensure_renderer()
            renderer.update_scene(self.data)
            image = renderer.render()
            try:
                import matplotlib.pyplot as plt  # type: ignore

                plt.imshow(image)
                plt.axis("off")
                plt.show(block=False)
            except ImportError:
                pass
            return None
        raise NotImplementedError(f"Unsupported render_mode: {self.render_mode}")

    def close(self) -> None:
        if self._renderer is not None:
            closer = getattr(self._renderer, "close", None)
            if callable(closer):
                closer()
            else:
                self._renderer.free()
            self._renderer = None

    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, 640, 480)
            except Exception:
                self._renderer = None
                raise
        return self._renderer

    def seed(self, seed: Optional[int] = None) -> None:  # pragma: no cover - compatibility shim
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
