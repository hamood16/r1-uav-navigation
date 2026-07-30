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
- A live-validated M13.2 deterministic scene system with calibrated Cube geometry,
  distinguishable start/goal pads, static layouts, reproducible reset, exact
  ownership cleanup, and measured-relative start-anchor positioning.
- M13.3 deterministic static-course generation with conservative 3D voxel
  occupancy, bounded solvability rejection, and reproducible 3D A* reference
  paths across training and held-out profiles.
- M13.4 fixed-size LiDAR extraction with 24-by-3 sensor-local sectors, an opt-in
  83-value environment observation, bounded sensor-failure handling, an accepted
  grounded live probe, accepted known-geometry distance evidence, and an accepted
  airborne observation smoke test with named cleanup.
- M13.5 Phase A provides an opt-in obstacle-aware Gymnasium environment that
  composes the validated LiDAR lifecycle with deterministic solvable courses,
  configurable safety rewards, exact workspace checks, episode metrics, and
  ordered scene cleanup. Its current acceptance evidence is fake-client/offline.
- M13.6 tooling provides offline-tested random and direct non-privileged
  baselines, a privileged A* waypoint reference controller, a strictly gated
  one-episode runner, and sanitized suite reporting. Supervised live evidence is
  pending.
- M13.7 Phase A provides offline long-run TD3 infrastructure: immutable
  model/replay/run-state bundles, strict resume and model-only warm start,
  deterministic validation-based best checkpoints, heartbeat supervision
  decisions, and throughput evidence. No long or live training has been run.
- M13.8 provides an offline six-stage static-obstacle curriculum,
  deterministic training/validation split enforcement, course and robustness
  samplers, promotion gates, route-shape evidence, and M13.7 curriculum-state
  checkpoint/resume support. Phase B adds strictly gated supervised-pilot
  tooling for bounded Stage 0/1 runs, but no curriculum training or live pilot
  has been run.

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
- A*: implemented for both classical 2D grids and deterministic 3D voxel-course
  solvability validation.

DDPG, SAC, PPO, and other algorithms are future options only; they are not
implemented in the current repository.

## Current Limitations And M13 Direction

M13 is moving toward obstacle-aware 3D navigation. M13.1 validates raw LiDAR and
temporary scene-mutation capabilities. M13.2 now validates deterministic live scene
materialization, same-seed reset, exact cleanup, and optional start-anchor
positioning. M13.3 now rejects impossible configured static courses offline and
records deterministic 3D A* reference paths before optional simulator use.
Physical collision response remains unverified, and the proof does not include
undocumented built-in Blocks geometry. The current repository does not yet
include:

- A trained obstacle-aware policy using the opt-in LiDAR observations.
- Camera or depth perception.
- occupancy built from live LiDAR.
- curriculum training or a live-validated long-run resume.
- dynamic obstacle avoidance in Colosseum.

M13.4 is complete: grounded extraction, one known-geometry comparison, and a
bounded airborne observation smoke test are live-validated. The legacy M12
environment remains a 10-value observation; the opt-in LiDAR environment adds 72
sector distances and one validity flag without exposing M13.3 obstacle
coordinates or reference paths. No obstacle-aware policy has been trained.
M13.3 is the next milestone after M13.2 in the roadmap sequence and is complete.
M13.5 is the next milestone after M13.4 in that sequence, and its Phase A
environment is implemented. M13.6 reference-controller software is also
implemented with fake-client/offline evidence. Supervised M13.6 course runs and
later obstacle-aware policy training remain future work. M13.7 Phase A
infrastructure is implemented offline; it does not demonstrate a trained
obstacle-aware policy or a recovered live run. M13.8 curriculum infrastructure
and supervised-pilot command software are implemented, but no live pilot has
been executed. Its M13.9 final-test seeds remain excluded from training and
curriculum promotion.

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
- [M13.2 deterministic scene specification](docs/m13_2_scene_specification.md)
- [M13.3 static-course solvability](docs/m13_3_static_course_solvability.md)
- [M13.4 LiDAR feature extraction](docs/m13_4_lidar_feature_extraction.md)
- [M13.5 obstacle-aware Gymnasium environment](docs/m13_5_obstacle_aware_gym_env.md)
- [M13.6 scripted reference controllers](docs/m13_6_scripted_reference_controllers.md)
- [M13.7 long-run training infrastructure](docs/m13_7_long_run_training_infrastructure.md)
- [M13.8 static-obstacle training curriculum](docs/m13_8_static_obstacle_curriculum.md)

## Project Structure

```text
r1-UAV-navigation/
|-- configs/
|   |-- evaluation/
|   |-- env/
|   |-- planning/
|   |-- scenes/
|   |-- sensing/
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
