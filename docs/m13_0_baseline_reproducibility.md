# M13.0 Baseline Reproducibility

## Purpose

M13.0 freezes the completed M12 baseline before the project begins M13 obstacle
navigation work. The milestone preserves the known M12 behavior, records the exact
baseline commit, clarifies dependencies, and adds simulator-independent CI.

M13.0 does not add LiDAR, scene management, visible pads, obstacles, new
environments, reward changes, TD3 architecture changes, replay-buffer resume,
curriculum learning, dynamic obstacles, or M13.1 capability-probe code.

## Baseline Tag

- Baseline tag: `m12.5-complete`
- Tag type: annotated tag
- Tagged commit: `ab83487d3b125dd4072f0cbe3823900be9a10d64`
- Commit summary: completed M12.5 Colosseum TD3 baseline and pre-public hardening

This tag is the source of truth for the completed M12 behavior before M13 changes.

## Supported Python Version

The project metadata declares:

```text
Python >=3.11,<3.12
```

Local and CI validation for this baseline should use Python 3.11.

## Dependency Installation

For simulator-independent development and CI:

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

For live Colosseum work, first install the simulator-independent dependencies, then
install the validated simulator compatibility dependencies:

```powershell
python -m pip install -r requirements-colosseum.txt
```

The Colosseum/AirSim-compatible Python client is not vendored in this repository.
Install it from the matching external Colosseum checkout:

- Colosseum Blocks simulator: `v2.0.0-beta`
- Colosseum source tag: `v2.0.0-beta`
- Colosseum source commit: `7b9658a1`
- Python client source: matching external `Colosseum/PythonClient` directory
- Validated RPC compatibility pin: `msgpack==0.6.2`

Do not commit editable local client paths, virtual-environment paths, simulator
binaries, or machine-specific checkout paths.

## Simulator-Independent Validation

Run these commands before changing M13 functionality:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black --check --no-cache .
.\.venv\Scripts\python.exe -m pip check
```

The GitHub Actions workflow performs the same simulator-independent checks on
Python 3.11. It does not launch Colosseum, Unreal Engine, Blocks, training,
evaluation, or GPU-dependent jobs.

## M12.5 Baseline Result

The M12.5 fixed-goal Colosseum TD3 baseline used:

- environment: `ColosseumUAVEnv`
- action shape: `(3,)`
- observation shape: `(10,)`
- fixed goal offset: `(3.0, 0.0, 0.0)`
- no obstacles

The strongest documented live result is the TD3 Stage B run:

- training steps: 2,000
- evaluation episodes: 5
- success rate: 100%
- mean return: 12.305
- mean final distance: 0.459 m

This demonstrates that the TD3 agent learned the simple fixed, obstacle-free
forward-goal task and that the live training, checkpointing, evaluation, metrics,
and cleanup pipeline works. It does not demonstrate obstacle-aware navigation,
random-goal generalisation, LiDAR perception, camera perception, or real-world UAV
readiness.

## Known M12 Limitations

- The Colosseum TD3 baseline uses a fixed horizontal goal and no obstacles.
- The green goal is not visually rendered in the simulator.
- The policy observes numeric goal-relative state, not LiDAR, camera, or depth.
- M12 does not include static or dynamic obstacle courses in Colosseum.
- M12 does not include replay-buffer resume or long-run recovery infrastructure.
- `moveByVelocityAsync().join()` still has no safe timeout in the validated legacy
  RPC stack.
- Generated checkpoints, logs, reports, plots, and videos remain ignored by Git.

## Regression Protection

M13.0 should preserve:

- M12.2 Colosseum connection helpers and cleanup behavior.
- M12.3 waypoint-navigation route generation, correction, and cleanup behavior.
- M12.4 `ColosseumUAVEnv` observation shape, action shape, reward, reset, step, and
  close lifecycle.
- M12.5 TD3 baseline configuration, checkpointing, evaluation reports, and recorded
  live results.

Focused M12 regression tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_m12_simulator_stack_decision.py tests\test_m12_colosseum_connection.py tests\test_m12_colosseum_waypoint_navigation.py tests\test_m12_colosseum_uav_env.py tests\test_m12_colosseum_td3_baseline.py -v
```
