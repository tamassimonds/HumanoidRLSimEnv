"""Robottraining package providing Mujoco humanoid environment scaffolding."""
from .envs.humanoid import HumanoidEnv, HumanoidEnvConfig
from .rewards import (
    ControlEffortPenalty,
    ForwardVelocityReward,
    RewardAggregator,
    RewardTerm,
    UprightPostureReward,
)

__all__ = [
    "HumanoidEnv",
    "HumanoidEnvConfig",
    "RewardTerm",
    "RewardAggregator",
    "ForwardVelocityReward",
    "UprightPostureReward",
    "ControlEffortPenalty",
]
