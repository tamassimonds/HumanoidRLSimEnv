"""Manual runner for the humanoid Mujoco environment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import numpy as np

from robottraining import HumanoidEnv, HumanoidEnvConfig


@dataclass
class RunnerConfig:
    episodes: int
    steps: int
    render: Optional[str]
    render_delay: float
    snapshot_path: Optional[Path]


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollouts to execute")
    parser.add_argument("--steps", type=int, default=200, help="Steps per rollout")
    parser.add_argument(
        "--render",
        choices=["rgb_array", "human"],
        default=None,
        help="Optional render mode",
    )
    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after each render call (useful to keep windows visible)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Optional path to save the first rendered frame (rgb_array mode recommended)",
    )
    args = parser.parse_args()
    return RunnerConfig(
        episodes=args.episodes,
        steps=args.steps,
        render=args.render,
        render_delay=args.render_delay,
        snapshot_path=args.snapshot,
    )


def run(config: RunnerConfig) -> None:
    env = HumanoidEnv(HumanoidEnvConfig(), render_mode=config.render)
    snapshot_saved = False
    for ep in range(config.episodes):
        obs, info = env.reset()
        print(f"Episode {ep + 1} reset with observation shape {obs.shape}")
        for step in range(config.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            breakdown = ", ".join(f"{k}: {v:+.3f}" for k, v in info["reward_terms"].items())
            stop_info = f" terminators={info['terminations']}" if info["terminations"] else ""
            print(
                f"step={step:04d} reward={reward:+.3f} done={done} "
                f"height={env.data.qpos[2]:.2f} | {breakdown}{stop_info}"
            )
            if config.render:
                frame = env.render()
                if config.render == "rgb_array" and frame is not None:
                    print(f"Rendered frame shape: {frame.shape}")
                if config.snapshot_path and frame is not None and not snapshot_saved:
                    save_frame(frame, config.snapshot_path)
                    snapshot_saved = True
                if config.render_delay > 0:
                    time.sleep(config.render_delay)
            if done:
                break
    env.close()


def save_frame(frame: np.ndarray, path: Path) -> None:
    """Persist an RGB frame to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - hard dependency for snapshots
        raise RuntimeError("imageio is required for saving snapshots") from exc
    iio.imwrite(path, frame)
    print(f"Saved snapshot to {path}")


if __name__ == "__main__":
    run(parse_args())
