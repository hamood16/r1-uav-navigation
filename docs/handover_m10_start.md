# Handover for M10 — Difficulty Configs, Repeated Evaluation, and Reporting

This document is for a fresh Codex chat starting M10 of `r1-UAV-navigation`. It summarises the current project state, completed baselines, important files, known results, and the recommended next steps.

## 1. Project overview

- Repo: `r1-UAV-navigation`
- Python package: `src/r1_uav_nav`
- Project goal: portfolio-grade UAV navigation project comparing reinforcement learning and classical planning.
- Python version: 3.11
- Main tools:
  - Gymnasium
  - Stable-Baselines3
  - PyTorch/CUDA
  - TensorBoard
  - Matplotlib
  - pytest
  - Ruff
  - Black

The project now has discrete static, discrete dynamic, and continuous dynamic environments, plus DQN, TD3, and A* baselines.

## 2. Current branch/state expectation

- Current handover branch: `docs/m10-handover`
- Assumption before starting M10 work:
  - M9.5 has been completed and merged into `main`.
  - The TD3 continuous dynamic baseline exists.
  - Baseline configs, scripts, tests, and plotting helpers are stable.

If this assumption is false, first verify that all M9.5 files exist and tests pass before beginning M10.

## 3. Completed milestone summary

### M1 — Static `GridUAVEnv`

Implemented `GridUAVEnv` in `src/r1_uav_nav/envs/grid_uav_env.py`.

It is a Gymnasium-compatible static 2D grid environment with:

- discrete actions
- static obstacles
- grid boundaries
- collision detection
- goal termination
- max-step truncation
- optional reward shaping
- optional LiDAR observations

### M2 — Static environment tests

Added detailed unit tests for static placement, boundaries, movement, collision, goal reaching, truncation, invalid actions, observations, reward shaping, and LiDAR.

Relevant test:

- `tests/test_grid_uav_env.py`

### M3 — Config loader

Implemented YAML loading and environment factories in `src/r1_uav_nav/utils/config_loader.py`.

Current factories include:

- `create_grid_uav_env_from_config`
- `create_dynamic_grid_uav_env_from_config`
- `create_continuous_dynamic_uav_env_from_config`

Relevant test:

- `tests/test_config_loader.py`

### M4 — Debug DQN pipeline

Added the quick debug DQN pipeline. This is separate from the full static/dynamic baselines.

### M5 — Evaluation metrics

Implemented reusable evaluation metrics in `src/r1_uav_nav/evaluation/metrics.py`.

Key objects/functions:

- `EpisodeResult`
- `EvaluationSummary`
- `calculate_path_length`
- `summarise_episode_results`

### M6 — Plotting helpers

Implemented plotting helpers in `src/r1_uav_nav/evaluation/plots.py`.

Current helpers cover:

- static trajectory plots
- reward curves
- success/collision/failure bar charts
- metric comparison bars
- static trajectory overlays
- discrete dynamic PNG/GIF trajectory plots
- continuous dynamic PNG/GIF trajectory plots

Relevant test:

- `tests/test_plots.py`

### M7 — Full static DQN baseline

Implemented the full static DQN pipeline using shaped `GridUAVEnv` with LiDAR.

Key files:

- `configs/env/grid_2d_static_full.yaml`
- `configs/training/dqn_static_full.yaml`
- `scripts/train_static_dqn.py`
- `scripts/evaluate_static_dqn.py`

### M7.5 — Discrete dynamic environment and visualisation

Implemented `DynamicGridUAVEnv` and a dynamic visualisation script.

Key files:

- `src/r1_uav_nav/envs/dynamic_grid_uav_env.py`
- `configs/env/dynamic_grid_2d.yaml`
- `scripts/visualize_dynamic_env.py`

### M7.8 — Full dynamic DQN baseline

Implemented the full dynamic DQN pipeline on `DynamicGridUAVEnv`.

Key files:

- `configs/training/dqn_dynamic_full.yaml`
- `scripts/train_dynamic_dqn.py`
- `scripts/evaluate_dynamic_dqn.py`
- `tests/test_dynamic_dqn_config.py`

### M8.1 — Static A* planner

Implemented deterministic A* for the static grid environment.

Key files:

- `src/r1_uav_nav/planners/astar.py`
- `scripts/evaluate_astar_static.py`
- `tests/test_astar_planner.py`

### M8.2 — Static DQN vs A* comparison

Implemented a fair comparison of static DQN and A* on the same seeded layouts.

Key files:

- `scripts/compare_static_dqn_vs_astar.py`
- `src/r1_uav_nav/evaluation/static_comparison.py`
- `tests/test_static_planner_comparison.py`

### M9 — Continuous dynamic environment for TD3 readiness

Implemented `ContinuousDynamicUAVEnv`, a continuous-action dynamic UAV environment suitable for TD3.

Key files:

- `src/r1_uav_nav/envs/continuous_dynamic_uav_env.py`
- `configs/env/continuous_dynamic_2d.yaml`
- `scripts/check_continuous_dynamic_env.py`
- `tests/test_continuous_dynamic_uav_env.py`

### M9.5 — TD3 continuous dynamic baseline

Implemented TD3 training/evaluation on `ContinuousDynamicUAVEnv`.

Key files:

- `configs/training/td3_continuous_dynamic_full.yaml`
- `src/r1_uav_nav/agents/td3_agent.py`
- `scripts/train_td3_continuous_dynamic.py`
- `scripts/evaluate_td3_continuous_dynamic.py`
- `tests/test_td3_agent.py`
- `tests/test_td3_config.py`

## 4. Key environments

### `GridUAVEnv`

File:

- `src/r1_uav_nav/envs/grid_uav_env.py`

Static 2D grid world.

Actions:

- `0 = up`
- `1 = down`
- `2 = left`
- `3 = right`
- `4 = hover`

Default observation shape:

- `(5,)`

Default observation:

```text
[uav_x, uav_y, goal_x, goal_y, nearest_obstacle_distance]
```

LiDAR-enabled observation shape:

- `(9,)`

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

Static full config:

- `configs/env/grid_2d_static_full.yaml`

Important values:

```yaml
grid_size: 10
max_steps: 100
num_obstacles: 10
use_lidar: true
step_penalty: -0.02
hover_penalty: -0.08
boundary_penalty: -0.50
collision_penalty: -8.0
goal_reward: 10.0
timeout_penalty: -3.0
progress_reward_scale: 0.3
```

### `DynamicGridUAVEnv`

File:

- `src/r1_uav_nav/envs/dynamic_grid_uav_env.py`

Discrete dynamic 2D grid world with moving obstacles.

Observation shape:

- `(9,)`

Observation:

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

Collision types:

- `"uav_into_obstacle"`
- `"obstacle_into_uav"`
- `None`

Info keys:

```python
{
    "is_success": bool,
    "is_collision": bool,
    "collision_type": str | None,
}
```

Dynamic DQN environment config:

- `configs/env/dynamic_grid_2d.yaml`

Important values:

```yaml
grid_size: 10
max_steps: 100
num_dynamic_obstacles: 5
step_penalty: -0.02
hover_penalty: -0.08
boundary_penalty: -0.50
collision_penalty: -8.0
goal_reward: 10.0
timeout_penalty: -3.0
progress_reward_scale: 0.3
```

### `ContinuousDynamicUAVEnv`

File:

- `src/r1_uav_nav/envs/continuous_dynamic_uav_env.py`

Continuous 2D dynamic environment for TD3.

Action space:

```python
Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

Action interpretation:

```text
action[0] = x velocity command
action[1] = y velocity command
```

Observation shape:

- `(9,)`

Observation:

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

Continuous dynamic config:

- `configs/env/continuous_dynamic_2d.yaml`

Important values:

```yaml
world_size: 10.0
max_steps: 200
num_dynamic_obstacles: 5
max_uav_speed: 1.0
obstacle_speed: 0.5
dt: 1.0
collision_radius: 0.35
goal_radius: 0.5
step_penalty: -0.02
boundary_penalty: -0.50
collision_penalty: -8.0
goal_reward: 10.0
timeout_penalty: -3.0
progress_reward_scale: 0.3
```

Info keys:

```python
{
    "is_success": bool,
    "is_collision": bool,
    "collision_type": "uav_obstacle_collision" | None,
}
```

## 5. Key algorithms and scripts

### Static DQN

Agent helper:

- `src/r1_uav_nav/agents/dqn_agent.py`

Training config:

- `configs/training/dqn_static_full.yaml`

Important values:

```yaml
algorithm: DQN
total_timesteps: 200000
buffer_size: 100000
learning_starts: 5000
batch_size: 64
model_output_path: results/trained_models/dqn_static_full.zip
tensorboard_log_dir: results/logs/dqn_static_full
```

Scripts:

- `scripts/train_static_dqn.py`
- `scripts/evaluate_static_dqn.py`

Static DQN plots:

- `results/plots/static/`

### Dynamic DQN

Training config:

- `configs/training/dqn_dynamic_full.yaml`

Important values:

```yaml
algorithm: DQN
total_timesteps: 300000
buffer_size: 150000
learning_starts: 10000
batch_size: 64
model_output_path: results/trained_models/dqn_dynamic_full.zip
tensorboard_log_dir: results/logs/dqn_dynamic_full
```

Scripts:

- `scripts/train_dynamic_dqn.py`
- `scripts/evaluate_dynamic_dqn.py`

Dynamic DQN plots:

- `results/plots/dynamic/`

### A*

Planner:

- `src/r1_uav_nav/planners/astar.py`

Core function:

```python
find_astar_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: set[tuple[int, int]],
    grid_size: int,
) -> list[tuple[int, int]] | None
```

Design:

- deterministic A*
- 4-connected movement only
- Manhattan heuristic
- returns path including start and goal
- returns `None` if no valid path exists

Standalone script:

- `scripts/evaluate_astar_static.py`

A* plots:

- `results/plots/planners/static_astar/`

### Static DQN vs A*

Script:

- `scripts/compare_static_dqn_vs_astar.py`

Reusable comparison helpers:

- `src/r1_uav_nav/evaluation/static_comparison.py`

Comparison outputs:

- `results/plots/comparison/static_dqn_vs_astar/`
- `results/plots/comparison/static_dqn_vs_astar/comparison_summary.json`

Fairness design:

- both methods use the same deterministic seeds
- A* plans directly on the generated static map
- DQN is run after resetting the env again with the same seed
- layout equality is checked before comparison

### TD3 continuous dynamic

Agent helper:

- `src/r1_uav_nav/agents/td3_agent.py`

Training config:

- `configs/training/td3_continuous_dynamic_full.yaml`

Important values:

```yaml
algorithm: TD3
total_timesteps: 300000
learning_rate: 0.0003
buffer_size: 200000
learning_starts: 10000
batch_size: 256
action_noise_std: 0.1
model_output_path: results/trained_models/td3_continuous_dynamic_full.zip
tensorboard_log_dir: results/logs/td3_continuous_dynamic_full
```

Scripts:

- `scripts/train_td3_continuous_dynamic.py`
- `scripts/evaluate_td3_continuous_dynamic.py`

TD3 plots:

- `results/plots/td3_continuous_dynamic/`

## 6. Important current results

### Static DQN baseline

- Success rate: `0.70`
- Collision rate: `0.00`
- Average reward: `6.20`
- Average steps: `34.59`
- Average path length: `28.90`

### Static A* baseline

- Success rate: `1.00`
- Failure rate: `0.00`
- Average successful path length: `6.63`
- Average successful steps: `6.63`

### Dynamic DQN baseline

- Success rate: `0.83`
- Collision rate: `0.17`
- Average reward: `7.91`
- Average steps: `7.40`
- Average path length: `7.38`
- Timeout rate: `0.00`

### TD3 continuous dynamic baseline

- Success rate: `0.98`
- Collision rate: `0.02`
- Average reward: `10.88`
- Average steps: `5.66`
- Average path length: `6.09`
- Timeout rate: `0.00`

## 7. Fairness note: DQN vs TD3

DQN and TD3 are not directly identical baselines.

- DQN baselines use discrete control.
- Static DQN uses `GridUAVEnv`, a static discrete grid world.
- Dynamic DQN uses `DynamicGridUAVEnv`, a discrete grid world with moving obstacles.
- TD3 uses `ContinuousDynamicUAVEnv`, a continuous-control world with continuous velocity commands.

Therefore, TD3 can produce shorter/smoother trajectories partly because it acts in a continuous control space. Comparisons are still useful for portfolio narrative, but they should be described carefully.

## 8. Current M10 plan

Recommended M10 sequence:

### M10.1 — Add easy / medium / hard configs

Start here.

Goal:

- add difficulty configs only
- do not retrain yet
- keep baseline configs untouched

Likely direction:

- create separate env configs for difficulty levels
- likely cover dynamic discrete DQN and continuous TD3 environments
- keep naming explicit, for example:
  - `configs/env/dynamic_grid_2d_easy.yaml`
  - `configs/env/dynamic_grid_2d_medium.yaml`
  - `configs/env/dynamic_grid_2d_hard.yaml`
  - `configs/env/continuous_dynamic_2d_easy.yaml`
  - `configs/env/continuous_dynamic_2d_medium.yaml`
  - `configs/env/continuous_dynamic_2d_hard.yaml`

### M10.2 — Add repeated evaluation / result summary tooling

Goal:

- evaluate trained baselines repeatedly across difficulty configs/seeds
- write machine-readable summaries
- avoid training inside tests

### M10.3 — Tune DQN and TD3 configs carefully

Goal:

- only after difficulty configs and repeated evaluation tooling exist
- tune config copies, not baseline configs
- preserve original baseline results

### M10.4 — Add documentation/report table

Goal:

- add portfolio-friendly result tables
- document baseline comparisons and fairness caveats
- possibly include committed Markdown summaries, not generated model/log files

## 9. Constraints for future Codex work

Important constraints:

- Do not overwrite baseline configs.
- Do not modify environments unless a clear bug is found.
- Do not commit generated results, models, logs, or plots.
- Preserve default train/evaluate script behaviour.
- Tests must not train models.
- Tests must not load real trained models.
- Keep new difficulty/tuning configs separate from baseline configs.
- Do not silently change existing reported results.
- Do not change trained model paths unless explicitly requested.

Baseline configs to preserve:

- `configs/env/grid_2d_static_full.yaml`
- `configs/env/dynamic_grid_2d.yaml`
- `configs/env/continuous_dynamic_2d.yaml`
- `configs/training/dqn_static_full.yaml`
- `configs/training/dqn_dynamic_full.yaml`
- `configs/training/td3_continuous_dynamic_full.yaml`

Baseline scripts to preserve:

- `scripts/train_static_dqn.py`
- `scripts/evaluate_static_dqn.py`
- `scripts/train_dynamic_dqn.py`
- `scripts/evaluate_dynamic_dqn.py`
- `scripts/train_td3_continuous_dynamic.py`
- `scripts/evaluate_td3_continuous_dynamic.py`
- `scripts/evaluate_astar_static.py`
- `scripts/compare_static_dqn_vs_astar.py`

## 10. Common commands

Use these standard checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black --check --no-cache .
```

Equivalent short forms if the venv is active:

```powershell
pytest
ruff check .
black --check --no-cache .
```

Common evaluation commands:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_dynamic_dqn.py
.\.venv\Scripts\python.exe scripts\evaluate_td3_continuous_dynamic.py
```

Other useful commands:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_static_dqn.py
.\.venv\Scripts\python.exe scripts\evaluate_astar_static.py
.\.venv\Scripts\python.exe scripts\compare_static_dqn_vs_astar.py
.\.venv\Scripts\python.exe scripts\check_continuous_dynamic_env.py
```

Training commands exist but should not be run unless explicitly requested:

```powershell
.\.venv\Scripts\python.exe scripts\train_static_dqn.py
.\.venv\Scripts\python.exe scripts\train_dynamic_dqn.py
.\.venv\Scripts\python.exe scripts\train_td3_continuous_dynamic.py
```

## 11. Generated artifacts and `.gitignore`

Generated artifacts should normally remain uncommitted.

`.gitignore` already ignores:

```text
results/logs/*
results/trained_models/*
results/videos/*
results/plots/*
runs/
```

Important generated paths:

- `results/trained_models/dqn_static_full.zip`
- `results/trained_models/dqn_dynamic_full.zip`
- `results/trained_models/td3_continuous_dynamic_full.zip`
- `results/logs/dqn_static_full/`
- `results/logs/dqn_dynamic_full/`
- `results/logs/td3_continuous_dynamic_full/`
- `results/plots/static/`
- `results/plots/dynamic/`
- `results/plots/planners/static_astar/`
- `results/plots/comparison/static_dqn_vs_astar/`
- `results/plots/td3_continuous_dynamic/`

If generated artifacts appear in `git status`, stop and inspect `.gitignore` before committing.

## 12. Recommended next step

Start M10.1 by adding difficulty configs only.

Recommended approach:

1. Inspect current baseline env configs.
2. Define easy / medium / hard variants without touching baseline configs.
3. Add config-loader tests that instantiate each difficulty config.
4. Do not retrain DQN or TD3 yet.
5. Do not change existing evaluate scripts yet unless the milestone explicitly asks for configurable evaluation over difficulty configs.

This keeps M10.1 small, reversible, and safe.

## 13. Relevant tests

Current relevant tests include:

- `tests/test_grid_uav_env.py`
- `tests/test_dynamic_grid_uav_env.py`
- `tests/test_continuous_dynamic_uav_env.py`
- `tests/test_config_loader.py`
- `tests/test_dqn_agent.py`
- `tests/test_dynamic_dqn_config.py`
- `tests/test_static_dqn_config.py`
- `tests/test_td3_agent.py`
- `tests/test_td3_config.py`
- `tests/test_astar_planner.py`
- `tests/test_static_planner_comparison.py`
- `tests/test_metrics.py`
- `tests/test_plots.py`

Future M10 tests should remain lightweight and should avoid:

- training models
- loading real trained model files
- requiring generated plots/logs/models to exist
