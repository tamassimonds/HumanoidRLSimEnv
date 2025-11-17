"""Reward system abstractions for humanoid Mujoco environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class RewardTerm:
    """Base class for weighted reward terms."""

    name: str
    weight: float

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    def score(self, env: "HumanoidEnv", action: np.ndarray | None = None) -> float:
        """Return the weighted contribution for the term."""

        raw = self._compute(env, action)
        return float(self.weight * raw)

    def _compute(self, env: "HumanoidEnv", action: np.ndarray | None = None) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


class ForwardVelocityReward(RewardTerm):
    """Encourages forward center-of-mass velocity along the +X axis."""

    def __init__(
        self,
        target_velocity: float = 2.0,
        weight: float = 1.0,
        velocity_index: int = 0,
    ) -> None:
        super().__init__(name="forward_velocity", weight=weight)
        if target_velocity <= 0:
            raise ValueError("target_velocity must be positive")
        self.target_velocity = target_velocity
        self.velocity_index = velocity_index

    def _compute(self, env: "HumanoidEnv", action: np.ndarray | None = None) -> float:
        velocity = float(env.data.qvel[self.velocity_index])
        forward_speed = max(0.0, velocity)
        normalized_forward = forward_speed / self.target_velocity
        backward_penalty = min(0.0, velocity) / self.target_velocity
        return float(np.clip(normalized_forward + backward_penalty, -1.0, 2.0))


class UprightPostureReward(RewardTerm):
    """Rewards the agent for aligning the torso with the world Z axis."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(name="upright", weight=weight)

    def _compute(self, env: "HumanoidEnv", action: np.ndarray | None = None) -> float:
        torso_body_id = env.torso_body_id
        torso_matrix = env.data.xmat[torso_body_id].reshape(3, 3)
        z_axis = torso_matrix[:, 2]
        return np.clip(z_axis[2], 0.0, 1.0)


class ControlEffortPenalty(RewardTerm):
    """Penalises large control actions to encourage smooth motion."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(name="control_effort", weight=-abs(weight))

    def _compute(self, env: "HumanoidEnv", action: np.ndarray | None = None) -> float:
        if action is None:
            return 0.0
        return float(np.mean(np.square(action)))


class RewardAggregator:
    """Aggregates reward components and exposes per-term breakdowns."""

    def __init__(self, terms: Sequence[RewardTerm]):
        if not terms:
            raise ValueError("RewardAggregator requires at least one term")
        self._terms: List[RewardTerm] = list(terms)

    def evaluate(self, env: "HumanoidEnv", action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        breakdown: Dict[str, float] = {}
        total = 0.0
        for term in self._terms:
            contribution = term.score(env, action)
            breakdown[term.name] = contribution
            total += contribution
        return total, breakdown

    @property
    def terms(self) -> Tuple[RewardTerm, ...]:
        return tuple(self._terms)
