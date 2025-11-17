"""Robottraining package providing Mujoco humanoid environment scaffolding."""
from .envs.humanoid import HumanoidEnv, HumanoidEnvConfig
from .rewards import (
    ControlEffortPenalty,
    ForwardVelocityReward,
    RewardAggregator,
    RewardTerm,
    UprightPostureReward,
)
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
    "StopCondition",
    "StopConditionSet",
    "FallOverCondition",
    "HorizontalDriftCondition",
    "TiltCondition",
]
