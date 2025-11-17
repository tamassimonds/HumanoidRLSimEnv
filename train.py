"""Training entrypoint for PPO humanoid agent using Stable-Baselines3."""
from __future__ import annotations

import argparse
from pathlib import Path

from robottraining.rl import TrainerConfig, train_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ppo_humanoid.yaml"),
        help="Path to a YAML/JSON TrainerConfig file",
    )
    parser.add_argument("--total-timesteps", type=int, default=None, help="Optional override for total timesteps")
    parser.add_argument("--run-name", type=str, default=None, help="Override the run name used for logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainerConfig.from_file(args.config)
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.run_name:
        config.run_name = args.run_name
    print(f"Loaded config from {args.config} (run_name={config.run_name}, steps={config.total_timesteps}).")
    train_ppo(config)


if __name__ == "__main__":
    main()
