"""Robottraining package providing Mujoco humanoid environment scaffolding."""
from .envs.humanoid import HumanoidEnv, HumanoidEnvConfig
from .rewards import (
    ControlEffortPenalty,
    ForwardVelocityReward,
    RewardAggregator,
    RewardTerm,
    UprightPostureReward,
)
from .rl import HumanoidEnvSettings, PPOSettings, TrainerConfig, train_ppo
from .terminations import (
    FallOverCondition,
    HorizontalDriftCondition,
    StopCondition,
    StopConditionSet,
    TiltCondition,
)

__all__ = [
    "HumanoidEnv",
    "HumanoidEnvConfig",
    "RewardTerm",
    "RewardAggregator",
    "ForwardVelocityReward",
    "UprightPostureReward",
    "ControlEffortPenalty",
    "TrainerConfig",
    "PPOSettings",
    "HumanoidEnvSettings",
    "train_ppo",
    "StopCondition",
    "StopConditionSet",
    "FallOverCondition",
    "HorizontalDriftCondition",
    "TiltCondition",
]
