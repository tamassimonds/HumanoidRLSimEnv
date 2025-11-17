"""Evaluate a trained PPO humanoid policy with optional video export."""
from __future__ import annotations

import argparse
from pathlib import Path

from robottraining.rl import EvaluationConfig, HumanoidEnvSettings, PolicyEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional path to an evaluation config file")
    parser.add_argument("--policy", type=Path, help="Path to a saved PPO policy (.zip)", nargs="?")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to rollout")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps per episode")
    parser.add_argument(
        "--render",
        choices=["human", "rgb_array"],
        default=None,
        help="Render mode for the environment",
    )
    parser.add_argument("--video", type=Path, default=None, help="Optional path to save a video (rgb_array mode)")
    parser.add_argument("--video-fps", type=int, default=None, help="Frames per second for the recorded video")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy outputs")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda/auto)")
    parser.add_argument("--frame-skip", type=int, default=None, help="Override env frame skip")
    parser.add_argument("--episode-length", type=int, default=None, help="Override env episode length")
    parser.add_argument("--model-path", type=Path, default=None, help="Custom MJCF path for the humanoid")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvaluationConfig:
    if args.config:
        base = EvaluationConfig.from_file(args.config)
    else:
        if args.policy is None:
            raise SystemExit("--policy is required when no config file is provided")
        default_env = HumanoidEnvSettings(
            model_path=str(args.model_path) if args.model_path else None,
            frame_skip=args.frame_skip or 5,
            episode_length=args.episode_length or 1000,
        )
        base = EvaluationConfig(
            policy_path=args.policy,
            env=default_env,
            episodes=args.episodes or 1,
            max_steps=args.max_steps or default_env.episode_length,
            deterministic=args.deterministic,
            render_mode=args.render,
            video_path=args.video,
            video_fps=args.video_fps or 60,
            device=args.device or "auto",
        )

    env_settings = HumanoidEnvSettings(
        model_path=str(args.model_path) if args.model_path else base.env.model_path,
        frame_skip=args.frame_skip or base.env.frame_skip,
        episode_length=args.episode_length or base.env.episode_length,
    )

    return EvaluationConfig(
        policy_path=args.policy or base.policy_path,
        env=env_settings,
        episodes=args.episodes or base.episodes,
        max_steps=args.max_steps or base.max_steps,
        deterministic=base.deterministic or args.deterministic,
        render_mode=args.render or base.render_mode,
        video_path=args.video or base.video_path,
        video_fps=args.video_fps or base.video_fps,
        device=args.device or base.device,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    evaluator = PolicyEvaluator(config)
    result = evaluator.run()
    for idx, (reward, length) in enumerate(zip(result.episode_rewards, result.episode_lengths), start=1):
        print(f"Episode {idx}: reward={reward:.3f}, length={length}")
    if result.video_path:
        print(f"Video saved to {result.video_path}")


if __name__ == "__main__":
    main()
