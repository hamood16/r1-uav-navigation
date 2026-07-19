# M12 Colosseum Gymnasium Wrapper

## Purpose

M12.4 adds a Gymnasium-compatible environment interface for goal-directed 3D
navigation in the validated Colosseum Blocks setup. This milestone is only the
environment wrapper. It does not train, evaluate, load policies, add camera or
LiDAR observations, or introduce obstacle planning.

The environment class is `ColosseumUAVEnv` in
`src/r1_uav_nav/envs/colosseum_uav_env.py`.

## Gymnasium API

`ColosseumUAVEnv` implements:

- `reset(seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `close()`
- `action_space`
- `observation_space`

Constructing the environment does not connect to Colosseum and does not make any
simulator RPC calls. The simulator client is created lazily during `reset()`, and
tests inject fake clients so automated tests do not require Unreal, Blocks, a GPU,
or the real `airsim` package.

## Action Space

The action space is a normalized continuous Box:

```text
shape: (3,)
range: [-1, 1]
dtype: float32
```

The action is mapped to AirSim-style NED world-frame velocity:

```text
vx = action_x * max_horizontal_velocity
vy = action_y * max_horizontal_velocity
vz = action_z * max_vertical_velocity
```

Positive `vz` moves downward, and negative `vz` moves upward. M12.4 keeps control in
world-frame velocity coordinates; body-frame or yaw-relative control is deferred.

## Observation Space

The observation is a finite normalized `float32` vector with shape `(10,)`:

| Index | Field |
|---:|---|
| 0-2 | current position relative to the measured airborne anchor: `x, y, z` |
| 3-5 | goal displacement from the current UAV position: `dx, dy, dz` |
| 6-8 | measured linear velocity: `vx, vy, vz` |
| 9 | normalized distance to the goal |

Normalization:

```text
rel_x / workspace_xy_limit
rel_y / workspace_xy_limit
rel_z / max(workspace_up_limit, workspace_down_limit)
goal_dx / (2 * workspace_xy_limit)
goal_dy / (2 * workspace_xy_limit)
goal_dz / (workspace_up_limit + workspace_down_limit)
vx / max_horizontal_velocity
vy / max_horizontal_velocity
vz / max_vertical_velocity
distance / full_workspace_diagonal
```

The full workspace diagonal is:

```text
sqrt((2 * workspace_xy_limit)^2
     + (2 * workspace_xy_limit)^2
     + (workspace_up_limit + workspace_down_limit)^2)
```

Measured simulator states must be finite. Non-finite state data is treated as a
simulator/client error, not as a valid observation.

## Ground Reference And Reset

The wrapper does not assume the simulator ground is `z=0`. Every `reset()` reads the
measured grounded state after `client.reset()` and stores:

```text
ground_reference_z = initial_position.z
target_anchor_z = initial_position.z - anchor_altitude
```

Ground clearance is calculated as:

```text
clearance = ground_reference_z - current_position.z
```

Every Gymnasium `reset()`, including the first one, calls `client.reset()` to
establish a known simulator state. The reset sequence is:

1. Clean up any previous control state.
2. Lazily create and connect the client if needed.
3. Call `client.reset()`.
4. Reconfirm the connection.
5. Read the grounded state.
6. Recreate the client once if reset leaves the current client unusable.
7. Validate the anchor and goal.
8. Enable API control.
9. Arm.
10. Take off.
11. Move to the measured anchor altitude.
12. Read the measured airborne anchor.
13. Return the initial observation and reset info.

Reset anchor movement uses:

```text
moveToPositionAsync(
    initial_x,
    initial_y,
    target_anchor_z,
    anchor_move_velocity,
    timeout_sec=anchor_move_timeout,
)
```

The inspected Colosseum v2.0.0-beta Python client supports `timeout_sec` on
`moveToPositionAsync`, but `moveByVelocityAsync` does not expose an equivalent
timeout argument. Step actions use a short finite `control_duration`, but the
legacy async `.join()` can still block if the RPC stack hangs. M12.4 documents this
limitation instead of adding unsafe thread cancellation or Tornado-loop handling.

## Goals

The default goal offset is `(3.0, 0.0, 0.0)` relative to the measured airborne
anchor. `reset(options={"goal_offset": (x, y, z)})` can provide an explicit safe
goal offset.

If `random_goal=True`, reset samples a seeded random goal from the safe workspace
using Gymnasium's RNG. The same seed and configuration produce the same sampled
goal.

Goals must:

- be finite;
- remain inside the configured workspace;
- satisfy minimum ground clearance;
- be at least `min_goal_distance` from the anchor;
- be farther than the goal tolerance, avoiding zero-step episodes.

## Reward Equation

The reward is intentionally simple and interpretable:

```text
progress = previous_distance_to_goal - current_distance_to_goal
action_magnitude = ||clipped_action||_2

reward =
    progress_reward_scale * progress
    + step_penalty
    - action_penalty_scale * action_magnitude
```

Terminal additions:

- success: `+ success_reward`
- collision: `+ collision_penalty`
- workspace or ground-clearance violation: `+ out_of_bounds_penalty`

This reward is not tuned in M12.4. Training belongs in M12.5.

## Termination And Truncation

`terminated=True` means the episode reached a task or safety terminal condition:

- goal reached;
- collision;
- workspace violation;
- minimum-ground-clearance violation.

`truncated=True` means the maximum episode step count was reached. Max-step
truncation is not reported as success.

Terminal priority is:

1. collision
2. ground-clearance violation
3. workspace violation
4. goal reached
5. maximum steps

On any terminal or truncated transition, the environment immediately attempts a
hover and marks the episode complete. Full landing, disarming, and API-control
release occur on the next `reset()` or `close()`.

## Cleanup

The environment reuses the M12.3 state-aware cleanup helpers:

- hover when takeoff was attempted or the drone is airborne;
- land when takeoff was attempted or the drone is airborne;
- disarm when arming succeeded;
- disable API control when API control was enabled.

`close()` is safe before reset, after repeated calls, after partial reset failure,
after step exceptions, and after terminal transitions. Cleanup failures are stored
in `last_cleanup_result` and do not replace an original simulator or environment
exception.

The live smoke-test script checks `last_cleanup_result` and returns nonzero if a
safety-critical cleanup action such as landing, disarming, or API-control release
failed.

## Live Smoke Tests

Start Colosseum Blocks first, then run one of:

```powershell
python scripts\run_colosseum_env_smoke_test.py --policy zero --steps 5
python scripts\run_colosseum_env_smoke_test.py --policy forward --steps 10
python scripts\run_colosseum_env_smoke_test.py --policy random --seed 42 --steps 10
```

You can also provide a goal offset or a fixed normalized action:

```powershell
python scripts\run_colosseum_env_smoke_test.py --goal-offset 3,0,0 --policy forward --steps 10
python scripts\run_colosseum_env_smoke_test.py --action 0.5,0,0 --steps 10
```

The script prints action, measured position, distance to goal, reward, termination
flags, and termination reason at each step. It always calls `env.close()` in a
`finally` block.

## Testing

Automated M12.4 tests use fake clients only. They check construction, reset,
observation normalization, action mapping, reward calculation, termination,
cleanup, and Gymnasium API compatibility without importing the real simulator
client or opening a network connection.

## Current Limitations

- No RL training or evaluation is included; M12.5 owns the first 3D baseline.
- No camera, image, or LiDAR observations are exposed.
- No obstacle avoidance or path planning is implemented.
- Step movement relies on `moveByVelocityAsync(...).join()`, which has no safe
  built-in timeout in the inspected legacy client.
- `simSetVehiclePose()` exists in the client but is not used as the primary reset
  path because it may not reset controller or collision state.
