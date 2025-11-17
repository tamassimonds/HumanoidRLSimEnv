"""Stop condition abstractions for resetting Mujoco environments."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

if False:  # pragma: no cover - circular import hints
    from robottraining.envs.humanoid import HumanoidEnv  # noqa: F401


class StopCondition:
    """Base class describing a termination trigger."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def triggered(self, env: "HumanoidEnv") -> bool:  # pragma: no cover - abstract
        raise NotImplementedError


class FallOverCondition(StopCondition):
    """Triggers when the torso height drops below a threshold."""

    def __init__(self, min_height: float = 0.6) -> None:
        super().__init__(name="fell_over")
        self.min_height = min_height

    def triggered(self, env: "HumanoidEnv") -> bool:
        torso_height = float(env.data.qpos[2])
        return bool(np.isnan(torso_height) or torso_height < self.min_height)


class HorizontalDriftCondition(StopCondition):
    """Triggers when the agent drifts too far from the origin on the plane."""

    def __init__(self, max_radius: float = 10.0) -> None:
        super().__init__(name="drifted_too_far")
        self.max_radius = max_radius

    def triggered(self, env: "HumanoidEnv") -> bool:
        pos = env.data.qpos[:2]
        radius = float(np.linalg.norm(pos))
        return bool(np.isnan(radius) or radius > self.max_radius)


class TiltCondition(StopCondition):
    """Triggers when the torso tilts outside an allowed cone relative to Z."""

    def __init__(self, min_dot: float = 0.2) -> None:
        super().__init__(name="tilted")
        self.min_dot = min_dot

    def triggered(self, env: "HumanoidEnv") -> bool:
        torso_matrix = env.data.xmat[env.torso_body_id].reshape(3, 3)
        z_axis = torso_matrix[:, 2]
        return bool(z_axis[2] < self.min_dot)


class StopConditionSet:
    """Evaluates a collection of stop conditions and reports triggers."""

    def __init__(self, conditions: Sequence[StopCondition]):
        self._conditions = list(conditions)

    def evaluate(self, env: "HumanoidEnv") -> Tuple[bool, Tuple[str, ...]]:
        triggered = tuple(cond.name for cond in self._conditions if cond.triggered(env))
        return bool(triggered), triggered

    @property
    def conditions(self) -> Tuple[StopCondition, ...]:
        return tuple(self._conditions)
