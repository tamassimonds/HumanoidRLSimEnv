"""Basic smoke tests for the humanoid environment."""
from __future__ import annotations

from typing import Optional

import numpy as np

from robottraining import (
    HumanoidEnv,
    HumanoidEnvConfig,
    RewardTerm,
    StopCondition,
)


class ConstantReward(RewardTerm):
    def __init__(self, value: float, name: str = "constant") -> None:
        super().__init__(name=name, weight=1.0)
        self.value = value

    def _compute(self, env: HumanoidEnv, action: Optional[np.ndarray] = None) -> float:
        return self.value


class ImmediateStop(StopCondition):
    def __init__(self) -> None:
        super().__init__(name="immediate")

    def triggered(self, env: HumanoidEnv) -> bool:
        return env.current_step >= 1


def test_env_reset_and_step_cycle() -> None:
    env = HumanoidEnv(HumanoidEnvConfig(frame_skip=1, episode_length=5))
    try:
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert set(info["reward_terms"].keys()) == {
            term.name for term in env.reward_aggregator.terms
        }
        assert info["terminations"] == ()

        action = env.action_space.sample()
        obs2, reward, terminated, truncated, step_info = env.step(action)
        assert obs2.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert set(step_info["reward_terms"].keys()) == {
            term.name for term in env.reward_aggregator.terms
        }
    finally:
        env.close()


def test_custom_reward_and_stop_condition() -> None:
    config = HumanoidEnvConfig(
        reward_terms=[ConstantReward(0.5, name="const")],
        termination_conditions=[ImmediateStop()],
        frame_skip=1,
        episode_length=5,
    )
    env = HumanoidEnv(config)
    try:
        env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isclose(reward, 0.5)
        assert terminated
        assert not truncated
        assert info["terminations"] == ("immediate",)
        assert set(info["reward_terms"].keys()) == {"const"}
    finally:
        env.close()
