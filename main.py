"""Manual runner for the humanoid Mujoco environment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from robottraining import HumanoidEnv, HumanoidEnvConfig


@dataclass
class RunnerConfig:
    episodes: int
    steps: int
    render: Optional[str]


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
    args = parser.parse_args()
    return RunnerConfig(episodes=args.episodes, steps=args.steps, render=args.render)


def run(config: RunnerConfig) -> None:
    env = HumanoidEnv(HumanoidEnvConfig(), render_mode=config.render)
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
            if done:
                break
    env.close()


if __name__ == "__main__":
    run(parse_args())
