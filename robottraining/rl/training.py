"""Helper utilities to launch PPO training via Stable-Baselines3."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from robottraining.envs.humanoid import HumanoidEnv
from robottraining.rl.config import HumanoidEnvSettings, TrainerConfig, resolve_device


EnvFactory = Callable[[], HumanoidEnv]


def make_env(settings: HumanoidEnvSettings, seed: int) -> EnvFactory:
    """Return a callable that instantiates :class:`HumanoidEnv` for SB3."""

    def _init() -> HumanoidEnv:
        env = HumanoidEnv(settings.to_env_config())
        env.reset(seed=seed)
        return env

    return _init


def train_ppo(config: TrainerConfig) -> None:
    """Train PPO with SB3 using the provided `config`."""

    device = resolve_device(config.device)
    set_random_seed(config.seed)

    train_env = VecMonitor(DummyVecEnv([make_env(config.env, config.seed)]))
    eval_env = VecMonitor(DummyVecEnv([make_env(config.env, config.seed + 9876)]))

    log_dir = Path(config.tensorboard_log) / config.run_name
    checkpoint_dir = Path(config.checkpoint_dir) / config.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(checkpoint_dir / "eval"),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy=config.ppo.policy,
        env=train_env,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=config.seed,
        **config.ppo.to_sb3_kwargs(device),
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
        log_interval=config.log_interval,
        tb_log_name=config.run_name,
        progress_bar=False,
    )

    final_model_path = checkpoint_dir / "final_model"
    model.save(str(final_model_path))
    train_env.close()
    eval_env.close()

    print(f"Training complete. Final model saved to {final_model_path}.")


__all__ = ["make_env", "train_ppo"]
