from __future__ import annotations

import numpy as np

from robottraining.rewards import ForwardVelocityReward


class DummyEnv:
    def __init__(self, velocity: float) -> None:
        self.data = type("Data", (), {"qvel": np.array([velocity])})()


def test_forward_velocity_reward_positive_direction():
    env = DummyEnv(velocity=1.5)
    reward = ForwardVelocityReward(target_velocity=1.5)
    assert np.isclose(reward._compute(env), 1.0)


def test_forward_velocity_reward_penalises_backward():
    env = DummyEnv(velocity=-1.0)
    reward = ForwardVelocityReward(target_velocity=1.0)
    assert reward._compute(env) < 0
