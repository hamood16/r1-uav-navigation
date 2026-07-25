# M13.5 Obstacle-Aware Gymnasium Environment

## Status And Scope

M13.5 Phase A is implemented and validated offline with fake clients. It adds the
first obstacle-aware learning environment, `ColosseumObstacleUAVEnv`, without
training or evaluating a policy. No M13.5 live simulator validation has been
performed.

The implementation composes existing milestone components:

- M13.2 deterministic materialization and exact ownership cleanup;
- M13.3 deterministic solvable-course selection;
- M13.4 named LiDAR lifecycle and fixed-size sensing;
- the protected M12 world-NED action and navigation-observation concepts.

It does not replace either `ColosseumUAVEnv` or `ColosseumLidarUAVEnv`.

## Protected Contracts

The legacy M12 environment remains unchanged:

- action space `Box(-1, 1, shape=(3,), dtype=float32)`;
- observation shape `(10,)`;
- existing reset, reward, training, and checkpoint behavior.

The M13.4 LiDAR environment remains opt-in and keeps:

- its named `SimpleFlight`/`LidarSensor1` lifecycle;
- no broad simulator reset;
- the existing 83-value observation;
- strict scan, timestamp, fault, and original-ground cleanup rules.

M13.5 adds no training configuration, policy, replay-buffer, or algorithm change.

## Architecture

`ColosseumObstacleUAVEnv` owns a fresh inner `ColosseumLidarUAVEnv` for each
episode. It performs these steps:

1. Resolve and validate a deterministic accepted M13.3 course.
2. Validate runtime authorization before simulator import.
3. Finish named cleanup for any previous inner episode.
4. Materialize the course through `ColosseumSceneManager`.
5. Pass only the materialized start anchor and goal approach to a new inner
   LiDAR environment.
6. Forward normalized actions and preserve the returned observation.
7. Calculate M13.5 rewards, safety outcomes, and metrics independently.
8. Close the named UAV lifecycle before exact object and marker cleanup.

The wrapper never calls broad `client.reset()` and never derives scene ownership
from names or configuration. M13.2 remains the source of ownership truth.

## Observation And Action

The policy observation remains the M13.4 contract:

```text
indices 0-9:   existing M12 navigation values
indices 10-81: 72 normalized LiDAR sector distances
index 82:      LiDAR-valid flag
```

Its shape is `(83,)`, its dtype is `numpy.float32`, and all values must be finite
and contained by the declared observation space.

The action remains:

```text
Box(-1, 1, shape=(3,), dtype=numpy.float32)
```

The inner environment maps it to world-frame NED velocity. M13.5 adds no yaw or
body-frame action.

The policy observation excludes obstacle coordinates, scene geometry, occupancy,
A* paths, reference path lengths, raw LiDAR clouds, settings paths, and local
machine paths.

## Reward Contract

The pure reward configuration uses positive magnitudes:

| Component | Default |
| --- | ---: |
| Goal progress scale | `2.0` |
| Success bonus | `25.0` |
| Step penalty | `0.02` |
| Collision penalty | `25.0` |
| Safety violation penalty | `15.0` |
| Unsafe-clearance penalty scale | `1.0` |
| Action-magnitude penalty scale | `0.02` |
| Action-change penalty scale | `0.04` |
| Clearance safety band | `1.0 m` |
| Emergency clearance | `0.50 m` |

The signed reward is:

```text
goal progress
+ success bonus
- step penalty
- collision penalty
- safety violation penalty
- unsafe-clearance penalty
- action-magnitude penalty
- action-change penalty
```

Clearance is denormalized from the current 72 sector values. Inside the safety
band, the penalty uses the square of this clipped severity:

```text
(safety_band - clearance) / (safety_band - lidar_minimum_range)
```

Action-change cost is suppressed at or below emergency clearance so it does not
discourage an urgent turn. One transient invalid scan uses the last valid
clearance for reward evidence. Invalid all-ones fallback features are never
treated as a fresh all-clear.

Only the primary terminal outcome contributes a terminal bonus or penalty. The
priority is collision, ground-clearance violation, workspace violation, then goal
success.

## Episode Outcomes

Episodes terminate for:

- collision;
- unsafe measured ground clearance;
- leaving the exact translated M13.2 workspace;
- reaching the goal approach.

Episodes truncate for:

- persistent LiDAR failure;
- maximum episode steps;
- watchdog timeout;
- a recoverable RPC event after named cleanup.

Obstacle proximity alone does not terminate an episode. An outer terminal or
watchdog condition requests a named hover and records the evidence; it does not
claim that the UAV landed. Further `step()` calls are rejected after termination
or truncation until reset or close.

## Course Selection

The versioned configuration is
`configs/env/m13_5_obstacle_uav_env.yaml`.

Its default deterministic selection is M13.3 profile `easy`, base seed `1100`.
Seeded mode builds a sorted pool from declared easy, medium, and hard training
profile/seed pairs and samples it only through Gymnasium's episode-local NumPy
generator. Global Python and NumPy random state are not used.

Reset may instead receive an already accepted `ValidatedCourse`. Endpoint-only
fixtures require an explicit test gate and an injected fake runtime; they are
rejected by default.

Course identity may appear in `info`, but course geometry and reference-path
evidence never enter the policy observation.

## Info Contract

Reset and step `info` records schema version 1 and stable diagnostics:

- scene digest, profile ID, base seed, accepted candidate seed, attempt index;
- start and goal labels;
- current and previous goal distance, and progress;
- current or retained LiDAR clearance and its source;
- collision, workspace, ground-clearance, proximity, and sensor states;
- consecutive invalid scans;
- travelled path length;
- current and previous actions and their magnitudes;
- complete signed reward breakdown;
- success, step count, safety overrides, and termination reason;
- prior cleanup evidence where applicable.

These diagnostics are outside the policy observation.

## Cleanup

Named UAV cleanup always precedes scene cleanup. Once UAV cleanup is
conclusively safe, exact M13.2 object cleanup and global marker flushing run using
the owned runtime state. If named UAV cleanup is safety-critical or unverifiable,
scene cleanup is deliberately deferred and the reason is returned.

`close_with_result()` returns separate UAV lifecycle, object/marker cleanup, and
deferral evidence. Close is idempotent. A new reset finishes the previous named
episode before replacing its scene.

## Phase A Evidence

Automated Phase A coverage uses fake clients and includes:

- Gymnasium `check_env`;
- exact action, observation, dtype, finite-value, and bounds checks;
- fixed, seeded, externally accepted, and gated fixture course selection;
- isolated reward component and terminal-priority tests;
- collision, ground, workspace, goal, max-step, watchdog, sensor, and RPC paths;
- path-length and action-history metrics;
- no broad reset;
- ordered and idempotent cleanup;
- scene cleanup deferral after unsafe UAV cleanup;
- M12, M13.2, M13.3, and M13.4 regression coverage.

These tests establish software behavior only. They are not live Colosseum
evidence.

## Limitations And Next Work

- No M13.5 live simulator smoke validation has run.
- No obstacle-aware policy has been trained.
- Reward defaults have not been tuned from training evidence.
- M13.3 solvability excludes undocumented built-in Blocks geometry.
- Cube dimensions remain operator-confirmed nominal evidence.
- Physical simulator collision response remains unverified.
- LiDAR features retain M13.4 beam-density, angular aggregation, and vertical-FOV
  limitations.
- No camera/depth input, SLAM, live LiDAR mapping, dynamic obstacle behavior,
  yaw/body-frame control, or real-world UAV claim is added.

The next M13.5 activity is a separately authorized supervised live smoke
validation. Policy training belongs to a later phase.
