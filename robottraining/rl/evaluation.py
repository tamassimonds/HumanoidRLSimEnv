"""Policy evaluation utilities for humanoid agents."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO

from robottraining.envs.humanoid import HumanoidEnv
from robottraining.rl.config import EvaluationConfig, resolve_device


@dataclass(slots=True)
class EvaluationResult:
    """Summary statistics for rollout episodes."""

    episode_rewards: Tuple[float, ...]
    episode_lengths: Tuple[int, ...]
    video_path: Optional[Path]


class VideoRecorder:
    """Simple video recorder backed by imageio."""

    def __init__(self, path: Path, fps: int) -> None:
        import imageio

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=fps)

    def write(self, frame: np.ndarray) -> None:
        self._writer.append_data(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class PolicyEvaluator:
    """Load an SB3 PPO policy and roll it out inside Mujoco."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.env = HumanoidEnv(config.env.to_env_config(), render_mode=config.render_mode)
        self.model = PPO.load(str(config.policy_path), device=self.device)
        self.video_recorder: Optional[VideoRecorder] = None
        if config.video_path is not None:
            self.video_recorder = VideoRecorder(config.video_path, config.video_fps)

    def run(self) -> EvaluationResult:
        rewards: List[float] = []
        lengths: List[int] = []
        try:
            for _ in range(self.config.episodes):
                ep_reward, ep_len = self._run_episode()
                rewards.append(ep_reward)
                lengths.append(ep_len)
        finally:
            self.close()
        return EvaluationResult(tuple(rewards), tuple(lengths), self.config.video_path)

    def _run_episode(self) -> Tuple[float, int]:
        obs, _ = self.env.reset()
        total_reward = 0.0
        steps = 0
        while steps < self.config.max_steps:
            action, _ = self.model.predict(obs, deterministic=self.config.deterministic)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += float(reward)
            steps += 1
            self._handle_render()
            if terminated or truncated:
                break
        return total_reward, steps

    def _handle_render(self) -> None:
        if self.video_recorder is not None:
            frame = self.env.render()
            if frame is not None:
                self.video_recorder.write(frame)
        elif self.config.render_mode == "human":
            self.env.render()

    def close(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        self.env.close()


def evaluate_policy(config: EvaluationConfig) -> EvaluationResult:
    """Convenience wrapper that instantiates :class:`PolicyEvaluator`."""

    evaluator = PolicyEvaluator(config)
    return evaluator.run()


__all__ = ["EvaluationResult", "PolicyEvaluator", "evaluate_policy"]
