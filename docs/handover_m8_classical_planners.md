# Handover for M8 — Classical Planner Comparison

This document is intended for the next Codex chat. It summarises the current state of `r1-UAV-navigation` after the reinforcement-learning baseline milestones and gives a clear starting point for M8.

## 1. Project overview

- Repo name: `r1-UAV-navigation`
- Python package: `src/r1_uav_nav`
- Project goal: build a portfolio-grade autonomous UAV reinforcement-learning navigation project.
- Current stack:
  - Python 3.11
  - Gymnasium
  - Stable-Baselines3
  - PyTorch/CUDA
  - TensorBoard
  - Matplotlib
  - pytest
  - Ruff
  - Black

The project is structured milestone-by-milestone. The current assumption for M8 is that M7.8 is complete and merged into `main`.

## 2. Completed milestones

### M1 — Static `GridUAVEnv`

Implemented a Gymnasium-compatible static 2D grid-world UAV environment in `src/r1_uav_nav/envs/grid_uav_env.py`.

Core static environment features:

- discrete UAV actions:
  - `0 = up`
  - `1 = down`
  - `2 = left`
  - `3 = right`
  - `4 = hover`
- static obstacles
- grid boundaries
- collision detection
- goal-reaching termination
- max-step truncation
- modern Gymnasium API:
  - `reset()` returns `(observation, info)`
  - `step()` returns `(observation, reward, terminated, truncated, info)`

### M2 — Static environment tests

Expanded the test suite for `GridUAVEnv`.

Coverage includes:

- constructor validation
- reset placement
- fixed start and fixed goal behaviour
- seeded reproducibility
- boundary handling
- valid movement
- hover
- obstacle collision
- goal reaching
- truncation
- invalid actions
- observation-space validity
- optional reward shaping and LiDAR behaviour

### M3 — YAML config loader

Implemented YAML config loading in `src/r1_uav_nav/utils/config_loader.py`.

Key helpers:

- `load_config`
- `create_grid_uav_env_from_config`
- `create_dynamic_grid_uav_env_from_config`

The environment classes do not load YAML directly. Config loading and environment construction are kept in utility/factory code.

### M4 — Debug DQN pipeline

Added a small debug DQN training pipeline using Stable-Baselines3.

Relevant files:

- `configs/training/dqn_debug.yaml`
- `scripts/train_dqn.py`
- `scripts/evaluate.py`
- `src/r1_uav_nav/agents/dqn_agent.py`

This pipeline is intentionally quick and remains separate from the full static/dynamic baselines.

### M5 — Evaluation metrics

Implemented reusable evaluation metrics in `src/r1_uav_nav/evaluation/metrics.py`.

Key structures/functions:

- `EpisodeResult`
- `EvaluationSummary`
- `calculate_path_length`
- `summarise_episode_results`

Metrics currently include:

- success rate
- collision rate
- average reward
- average steps per episode
- average path length

### M6 — Evaluation plots

Added portfolio-friendly plotting helpers in `src/r1_uav_nav/evaluation/plots.py`.

Static plotting includes:

- trajectory plot
- reward curve
- success-rate bar chart
- collision-rate bar chart

Dynamic plotting includes:

- dynamic trajectory PNGs
- dynamic trajectory GIFs
- dynamic obstacle trails and direction arrows

### M7 — Static full DQN baseline

Added a separate full static DQN baseline pipeline.

Relevant files:

- `configs/env/grid_2d_static_full.yaml`
- `configs/training/dqn_static_full.yaml`
- `scripts/train_static_dqn.py`
- `scripts/evaluate_static_dqn.py`

Important additions:

- reward shaping for the static full environment
- optional LiDAR observations in `GridUAVEnv`
- static full model/log/plot paths kept separate from the debug pipeline

### M7.5 — Dynamic grid environment and visualisation

Added a separate dynamic environment with moving obstacles.

Relevant files:

- `src/r1_uav_nav/envs/dynamic_grid_uav_env.py`
- `configs/env/dynamic_grid_2d.yaml`
- `scripts/visualize_dynamic_env.py`

M7.5 does not train a DQN. It provides dynamic environment correctness and safety-aware greedy rollout visualisations.

### M7.8 — Full dynamic DQN training/evaluation

Added the trained dynamic DQN baseline.

Relevant files:

- `configs/training/dqn_dynamic_full.yaml`
- `scripts/train_dynamic_dqn.py`
- `scripts/evaluate_dynamic_dqn.py`
- `tests/test_dynamic_dqn_config.py`

Dynamic DQN outputs are kept separate from static DQN outputs.

## 3. Static environment details

Static environment implementation:

- `src/r1_uav_nav/envs/grid_uav_env.py`

The static environment supports optional LiDAR:

- default/debug observation shape: `(5,)`
- LiDAR-enabled observation shape: `(9,)`

Default observation:

```text
[
  uav_x,
  uav_y,
  goal_x,
  goal_y,
  nearest_obstacle_distance
]
```

LiDAR-enabled observation:

```text
[
  uav_x,
  uav_y,
  goal_x,
  goal_y,
  nearest_obstacle_distance,
  lidar_up,
  lidar_down,
  lidar_left,
  lidar_right
]
```

Static full environment config:

- `configs/env/grid_2d_static_full.yaml`

Current shaped static environment values:

```yaml
env_name: GridUAVEnv
grid_size: 10
max_steps: 100
num_obstacles: 10
random_start: true
random_goal: true
use_lidar: true
step_penalty: -0.02
hover_penalty: -0.08
boundary_penalty: -0.50
collision_penalty: -8.0
goal_reward: 10.0
timeout_penalty: -3.0
progress_reward_scale: 0.3
```

Static DQN training config:

- `configs/training/dqn_static_full.yaml`

Static model path:

- `results/trained_models/dqn_static_full.zip`

Static logs:

- `results/logs/dqn_static_full/`

Static plots:

- `results/plots/static/`

## 4. Static DQN result

Latest static full DQN baseline result:

- Success rate: `0.70`
- Collision rate: `0.00`
- Average reward: `6.20`
- Average steps: `34.59`
- Average path length: `28.90`

Note: zero collisions were observed over 100 fixed-seed evaluation episodes.

## 5. Dynamic environment details

Dynamic environment implementation:

- `src/r1_uav_nav/envs/dynamic_grid_uav_env.py`

The dynamic environment is separate from the static `GridUAVEnv`.

Core features:

- 2D grid world
- same discrete UAV actions as the static environment
- moving dynamic obstacles
- each dynamic obstacle has:
  - position: `(x, y)`
  - velocity: `(vx, vy)`
- obstacles move every step
- obstacles bounce off walls
- UAV collision with moving obstacles terminates the episode
- obstacles moving into the UAV also terminate the episode
- Gymnasium modern API

Dynamic collision types:

- `"uav_into_obstacle"`
- `"obstacle_into_uav"`
- `None`

Dynamic step `info` flags:

```python
{
    "is_success": bool,
    "is_collision": bool,
    "collision_type": str | None,
}
```

Dynamic observation shape: `(9,)`

Dynamic observation:

```text
[
  uav_x,
  uav_y,
  goal_x,
  goal_y,
  nearest_dynamic_obstacle_x,
  nearest_dynamic_obstacle_y,
  nearest_dynamic_obstacle_vx,
  nearest_dynamic_obstacle_vy,
  nearest_dynamic_obstacle_distance
]
```

Dynamic environment config:

- `configs/env/dynamic_grid_2d.yaml`

Current dynamic environment values:

```yaml
env_name: DynamicGridUAVEnv
grid_size: 10
max_steps: 100
num_dynamic_obstacles: 5
random_start: true
random_goal: true
step_penalty: -0.02
hover_penalty: -0.08
boundary_penalty: -0.50
collision_penalty: -8.0
goal_reward: 10.0
timeout_penalty: -3.0
progress_reward_scale: 0.3
```

## 6. Dynamic DQN result

Latest dynamic full DQN baseline result:

- Success rate: `0.83`
- Collision rate: `0.17`
- Average reward: `7.91`
- Average steps: `7.40`
- Average path length: `7.38`
- Timeout rate: `0.00`

Dynamic model path:

- `results/trained_models/dqn_dynamic_full.zip`

Dynamic logs:

- `results/logs/dqn_dynamic_full/`

Dynamic trained DQN plots:

- `results/plots/dynamic/`

## 7. Important files

Environment configs:

- `configs/env/grid_2d.yaml`
- `configs/env/grid_2d_static_full.yaml`
- `configs/env/dynamic_grid_2d.yaml`

Training configs:

- `configs/training/dqn_static_full.yaml`
- `configs/training/dqn_dynamic_full.yaml`

Training/evaluation scripts:

- `scripts/train_static_dqn.py`
- `scripts/evaluate_static_dqn.py`
- `scripts/visualize_dynamic_env.py`
- `scripts/train_dynamic_dqn.py`
- `scripts/evaluate_dynamic_dqn.py`

Environment implementations:

- `src/r1_uav_nav/envs/grid_uav_env.py`
- `src/r1_uav_nav/envs/dynamic_grid_uav_env.py`

Evaluation helpers:

- `src/r1_uav_nav/evaluation/metrics.py`
- `src/r1_uav_nav/evaluation/plots.py`
- `src/r1_uav_nav/evaluation/rollout_selection.py`

Config and agent helpers:

- `src/r1_uav_nav/utils/config_loader.py`
- `src/r1_uav_nav/agents/dqn_agent.py`

Relevant tests:

- `tests/test_grid_uav_env.py`
- `tests/test_dynamic_grid_uav_env.py`
- `tests/test_config_loader.py`
- `tests/test_metrics.py`
- `tests/test_plots.py`
- `tests/test_dqn_agent.py`
- `tests/test_static_dqn_config.py`
- `tests/test_dynamic_dqn_config.py`

## 8. Generated artifacts

Generated model, log, and plot artifacts are ignored and should not normally be committed.

Important generated artifacts:

- `results/trained_models/dqn_static_full.zip`
- `results/trained_models/dqn_dynamic_full.zip`
- `results/logs/dqn_static_full/`
- `results/logs/dqn_dynamic_full/`
- `results/plots/static/`
- `results/plots/dynamic/`

If a future milestone needs portfolio images committed, make that an explicit decision rather than accidentally committing generated outputs.

## 9. Standard checks

Run these before handing off or committing:

```powershell
pytest
ruff check .
black --check --no-cache .
```

On Windows, the project commonly uses:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black --check --no-cache .
```

## 10. Current repo status assumption

Assume M7.8 is complete and merged into `main` before starting M8.

If this assumption is false, first verify that the dynamic DQN files, tests, and generated baseline results are present and stable before beginning planner work.

## 11. M8 next milestone — Classical planner comparison

M8 should add classical planner comparison while keeping the DQN pipelines intact.

Recommended M8 direction:

1. Start with the static environment.
2. Implement an A* planner for the static grid.
3. Consider adding Dijkstra after A*.
4. Compare classical planner performance against the trained static DQN.
5. Compare at least:
   - path length
   - success rate
   - collision rate, expected to be zero for valid static plans
   - average steps or planned path length
6. Add a script to run static planner comparison.
7. Add tests for planner correctness.
8. Consider dynamic planner comparison later as a separate milestone.

Suggested package location:

- `src/r1_uav_nav/planners/`

Possible M8 files:

- `src/r1_uav_nav/planners/__init__.py`
- `src/r1_uav_nav/planners/astar.py`
- `src/r1_uav_nav/planners/dijkstra.py`
- `scripts/compare_static_planners.py`
- `tests/test_astar_planner.py`
- `tests/test_static_planner_comparison.py`

Keep this milestone small and clear. A well-tested static A* baseline is more valuable than a broad, fragile planner framework.

## 12. M8 constraints

For M8:

- Do not modify trained model paths.
- Do not break static or dynamic DQN scripts.
- Do not retrain DQN unless explicitly needed.
- Keep classical planners in separate modules, preferably under `src/r1_uav_nav/planners/`.
- Add tests for planners.
- Keep generated comparison plots ignored unless the project owner explicitly decides otherwise.
- Do not modify `GridUAVEnv` or `DynamicGridUAVEnv` unless a planner integration bug makes it necessary.
- Do not remove or overwrite existing baseline artifacts.
- Do not implement TD3, AirSim, ROS, Unreal, or dynamic planner training as part of M8.

Good first M8 acceptance criteria:

- A* finds a valid path on a known obstacle layout.
- A* returns `None` or an empty result when no path exists.
- A planner comparison script can evaluate several seeded static environments.
- The script reports planner metrics in the same spirit as the DQN evaluation summaries.
- Existing DQN tests and scripts remain unaffected.
