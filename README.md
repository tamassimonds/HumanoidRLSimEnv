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

Add `--render human` (or `--render rgb_array`) to preview Mujoco frames while it runs. Use `--render-delay 0.05` to slow the window update loop and `--snapshot preview.png` (best with `rgb_array`) to capture the first frame to disk.

## PPO training (Stable-Baselines3)

The `train.py` entrypoint wires the humanoid environment into Stable-Baselines3's PPO implementation and accepts a YAML/JSON config describing both the environment parameters and PPO hyper-parameters.

```bash
python train.py --config configs/ppo_humanoid.yaml
```

The default config (`configs/ppo_humanoid.yaml`) exposes:

- `env`: frame skip, episode length, and optional custom MJCF path for the Mujoco model.
- `ppo`: policy type plus learning rate, rollout length (`n_steps`), batch size, clipping, entropy/value weights, etc. Custom network topologies can be specified through `policy_kwargs`.
- `seed`, `total_timesteps`, `eval_freq`, logging directories, and run names so multiple ablations can be tracked independently.

During training, SB3 writes TensorBoard logs under `runs/<run_name>` and saves checkpoints/best models under `checkpoints/<run_name>`. Adjust these paths directly in the config if needed.
Add `--render human` (or `--render rgb_array`) to preview Mujoco frames while it runs. Rendering requires an offscreen buffer, already configured in `flat_humanoid.xml`.

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

## PPO training (Stable-Baselines3)

`train.py` wraps the humanoid environment with Stable-Baselines3's PPO implementation. Pass a YAML/JSON configuration describing the environment parameters plus PPO hyper-parameters to run a training job:

```bash
python train.py --config configs/ppo_humanoid.yaml
```

Key config sections:

- `env` &mdash; toggles Mujoco XML path, frame skip, and episode length (`HumanoidEnvConfig`).
- `ppo` &mdash; forwards settings to SB3 (e.g., `learning_rate`, `n_steps`, `clip_range`, `policy_kwargs.net_arch`).
- `seed`, `total_timesteps`, logging/checkpoint directories, evaluation cadence, and run names.

SB3 writes TensorBoard traces under `runs/<run_name>` and checkpoints/best models under `checkpoints/<run_name>`. Adjust these paths in the config for different experiments.

## Stop conditions

Termination logic is handled through composable `StopCondition` objects aggregated in a `StopConditionSet`. The default environment includes fall detection, torso tilt monitoring, and lateral drift limits; triggered conditions are exposed under the `terminations` entry of the Gymnasium `info` dict so agents or curriculum logic can react accordingly.

## Testing

Run the smoke tests to ensure the environment can reset, step, and trigger termination hooks:

```bash
uv run pytest
```
