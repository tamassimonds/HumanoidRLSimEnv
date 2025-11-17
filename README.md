# RobotTraining

Prototype Mujoco humanoid training environment with modular reward components.

## Setup

Create a virtual environment, install dependencies, and ensure Mujoco can load GPU/CPU drivers:

```bash
uv sync
```

(Any other standard Python tooling such as `pip install -e .` works as well.)

## Quickstart

```bash
python main.py --episodes 1 --steps 50
```

This command instantiates the humanoid environment, steps it with random actions, and prints the reward breakdown for each step along with simple diagnostics.

## Reward and penalty design

Rewards are implemented as composable `RewardTerm` objects (see `robottraining/rewards.py`). Each term exposes a `name`, `weight`, and `score` computed from the Mujoco state/action. The `RewardAggregator` combines these components and exposes per-term contributions per timestep. The humanoid environment wires in the following defaults:

- `ForwardVelocityReward` encourages forward center-of-mass velocity around 1.5 m/s.
- `UprightPostureReward` keeps the torso aligned with the world Z-axis.
- `ControlEffortPenalty` discourages large control signals.

Custom reward schedules can be supplied by instantiating `HumanoidEnv` with a tailored `HumanoidEnvConfig` and a custom list of `RewardTerm` objects, making it easy to iterate on task definitions without touching the environment core.

## Project structure

```
robottraining/
├── assets/flat_humanoid.xml   # Simplified humanoid MJCF asset (flat plane)
├── envs/humanoid.py           # Gymnasium-compatible Mujoco environment
├── rewards.py                 # Reward abstractions and common terms
└── __init__.py
```

`main.py` demonstrates how to construct the environment, reset/step it, and inspect reward diagnostics. Future RL agents can plug in directly via the standard Gymnasium interface.
