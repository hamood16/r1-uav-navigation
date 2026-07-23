# r1-UAV-navigation

Autonomous UAV path planning and navigation using reinforcement learning, classical
planning, and Colosseum/AirSim-compatible 3D simulation.

## Project Goal

This repository builds a modular UAV navigation stack that progresses from small
2D Gymnasium environments to live 3D simulator control. It is designed to make each
stage testable and reproducible before adding more realism.

The current project demonstrates:

- 2D static and dynamic UAV navigation environments.
- DQN baselines for discrete grid navigation.
- TD3 baselines for continuous 2D and Colosseum 3D control.
- A* planning and static DQN-vs-A* comparison.
- Repeated evaluation summaries and comparison plotting.
- Colosseum Blocks connection, state reads, takeoff, movement, landing, and safe
  cleanup.
- Scripted 3D waypoint navigation.
- A Gymnasium-compatible Colosseum UAV wrapper.
- A live-validated TD3 training, checkpointing, evaluation, and cleanup pipeline
  for a simple fixed, obstacle-free 3D goal task.
- A supervised Colosseum capability probe validating scene surveys, debug markers,
  temporary-object lifecycle, raw LiDAR access, `SensorLocalFrame` behavior, and
  bounded RPC performance.

## Current M12 Baseline

The completed M12 Colosseum baseline uses `ColosseumUAVEnv` with a normalized
continuous 3D action space and a 10-value state-vector observation. The M12.5 TD3
baseline trains toward a fixed forward goal offset of `(3.0, 0.0, 0.0)` with no
obstacles.

The strongest documented M12.5 result is the 2,000-step TD3 Stage B run:

- evaluation episodes: 5
- success rate: 100%
- mean return: 12.305
- mean final distance: 0.459 m

This result validates the live simulator training pipeline for a simple fixed-goal
task. It does not demonstrate random-goal generalisation, obstacle avoidance,
LiDAR perception, or real-world readiness.

## Implemented Algorithms

- DQN: implemented through Stable-Baselines3 for discrete grid navigation.
- TD3: implemented through Stable-Baselines3 for continuous navigation.
- A*: implemented as a classical static-grid planning baseline.

DDPG, SAC, PPO, and other algorithms are future options only; they are not
implemented in the current repository.

## Current Limitations And M13 Direction

M13 is moving toward obstacle-aware 3D navigation. M13.1 validates raw LiDAR and
temporary scene-mutation capabilities, but it does not integrate them into a
navigation environment or policy. The current repository does not yet include:

- LiDAR observations in a Gymnasium environment or learned policy.
- Camera or depth perception.
- Reusable scene specifications or procedural obstacle courses.
- visible start/goal pads.
- obstacle-aware Colosseum Gymnasium environments.
- 3D obstacle-aware A* planning.
- curriculum training or replay-buffer resume.
- dynamic obstacle avoidance in Colosseum.

M13.2 is the next milestone and will build on the validated scene and marker
capabilities without changing the completed M12 interfaces.

## Tech Stack

- Python 3.11
- PyTorch
- Gymnasium
- Stable-Baselines3
- NumPy
- Matplotlib
- PyYAML
- pytest
- Ruff
- Black
- Colosseum Blocks v2.0.0-beta for live 3D simulator validation

## Results And Documentation

- [M10 dynamic RL results summary](docs/m10_dynamic_rl_results.md)
- [DQN vs TD3 dynamic navigation comparison](docs/results/dqn_vs_td3_dynamic.md)
- [M12 simulator stack decision](docs/m12_simulator_stack_decision.md)
- [M12 Colosseum setup and connection check](docs/m12_colosseum_setup.md)
- [M12 Colosseum waypoint navigation demo](docs/m12_colosseum_navigation_demo.md)
- [M12 Colosseum Gymnasium wrapper](docs/m12_colosseum_gym_wrapper.md)
- [M12 Colosseum TD3 baseline](docs/m12_colosseum_td3_baseline.md)
- [M13.0 baseline reproducibility freeze](docs/m13_0_baseline_reproducibility.md)
- [M13.1 Colosseum capability probe](docs/m13_colosseum_capability_probe.md)

## Project Structure

```text
r1-UAV-navigation/
|-- configs/
|   |-- env/
|   `-- training/
|-- docs/
|   `-- results/
|-- results/
|   |-- logs/
|   |-- plots/
|   |-- reports/
|   |-- trained_models/
|   `-- videos/
|-- scripts/
|-- src/
|   `-- r1_uav_nav/
|       |-- agents/
|       |-- envs/
|       |-- evaluation/
|       |-- planners/
|       |-- sim/
|       |-- training/
|       `-- utils/
`-- tests/
```

Generated models, logs, plots, reports, videos, simulator binaries, and local
virtual environments are intentionally not committed.
