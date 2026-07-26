# M13.6 Scripted Reference Controllers And Live Course Validation

## Status And Scope

M13.6 software and fake-client/offline validation are implemented. Supervised
live course evidence is pending. No M13.6 live command has been run, and no
obstacle-aware policy has been trained.

The milestone provides three deterministic reference controllers and a
single-episode supervised runner:

- `random` is a seeded, non-privileged lower-bound baseline;
- `direct` is a non-privileged goal controller using only the protected
  navigation prefix;
- `oracle` is a privileged M13.3 A* waypoint reference controller.

The reactive LiDAR controller is deferred. M13.6 does not train TD3, SAC, or any
other policy.

## Reused Contracts

The implementation builds on the existing systems:

- M13.3 supplies accepted `ValidatedCourse` scene, grid, identity, and reference
  path evidence;
- M13.4 supplies the fixed 72-sector LiDAR representation and named sensor
  lifecycle;
- M13.5 supplies the `(83,)` observation, world-NED action mapping, episode
  outcomes, original-ground return, and UAV-before-scene cleanup;
- M13.2 remains the sole authority for exact scene ownership and cleanup.

M13.6 does not change M12 through M13.5 observations, actions, rewards,
termination rules, scene digests, course baselines, or cleanup semantics.

## Controller Contract

Every controller returns a finite `numpy.float32` action with shape `(3,)` in
`[-1, 1]`.

### Random

`RandomController` uses a private `numpy.random.Generator`. It ignores the
observation, LiDAR, course geometry, occupancy, and A* evidence. A controller seed
reproduces its action sequence without changing global Python or NumPy random
state.

### Direct

`DirectGoalController` decodes only:

- goal displacement from observation indices `3-5`;
- current velocity from indices `6-8`;
- immutable scale and action-limit metadata from `controller_state_spec()`.

It applies proportional goal steering, velocity damping, and near-goal slowdown.
It does not inspect LiDAR indices, obstacle geometry, direct-line status,
occupancy, or reference paths.

### Oracle

`OracleWaypointController` is explicitly marked
`privileged_reference_path`. It converts the accepted M13.3 reference path into a
deterministic route that:

- preserves the start, goal, and direction changes;
- limits each segment to at most `1.0 m`;
- revalidates every compressed segment against the accepted M13.3 voxel grid.

Reports contain waypoint counts and a route digest, not full route coordinates.
Oracle success demonstrates live course executability. It is not learned
intelligence and is not a deployable navigation policy.

## Safety Outcome

The M13.6 runner may record `clearance_abort` when current or retained valid
clearance is at or below `1.0 m`. The runner stops controller action generation
and immediately enters the existing named cleanup lifecycle. This does not
change M13.5 termination semantics.

A transient invalid LiDAR sample causes a zero-action hold. Persistent failure
continues to use the M13.5 sensor-failure truncation and cleanup behavior.

Named UAV cleanup always precedes exact M13.2 object and marker cleanup. Scene
cleanup is deferred when final UAV safety cannot be established. Broad simulator
reset remains prohibited.

## Required Matrix

| Controller | Course | Base seed | Expected task outcome |
| --- | --- | ---: | --- |
| Direct | Empty | `0` | Goal success |
| Oracle | Empty | `0` | Goal success |
| Direct | Easy | `1100` | Goal success |
| Oracle | Easy | `1100` | Goal success |
| Direct | Medium | `2100` | Clearance abort or collision; no goal success |
| Oracle | Medium | `2100` | Goal success |
| Random | Medium | `2100` | No success, controller seed `1360` |
| Random | Medium | `2100` | No success, controller seed `1361` |
| Random | Medium | `2100` | No success, controller seed `1362` |

Oracle on hard seed `3100` is optional and requires the additional
`--allow-optional-hard` and `--confirm-required-stages-passed` gates.

Direct-medium acceptance prefers clearance abort. Collision is accepted if it
occurs, but the run never requires deliberately flying into an obstacle. Timeout
alone does not satisfy the direct-medium baseline.

## CLI

The offline commands do not import the AirSim client:

```powershell
.\.venv\Scripts\python.exe scripts\check_m13_6_reference_controllers.py validate
.\.venv\Scripts\python.exe scripts\check_m13_6_reference_controllers.py preview
.\.venv\Scripts\python.exe scripts\check_m13_6_reference_controllers.py summarize REPORT...
```

`run` executes one supervised episode. Before any client import it validates the
configuration, exact controller/course/seed pairing, M13.3 solvability, output
directory, names, and all authorizations:

```powershell
.\.venv\Scripts\python.exe scripts\check_m13_6_reference_controllers.py run `
  --controller direct `
  --course-profile empty `
  --course-seed 0 `
  --vehicle-name SimpleFlight `
  --lidar-name LidarSensor1 `
  --allow-live-rpc `
  --allow-scene-mutation `
  --confirm-scene-area-clear `
  --confirm-no-visible-collision `
  --allow-debug-markers `
  --allow-marker-flush `
  --allow-flight `
  --allow-start-positioning `
  --confirm-clear-airspace `
  --confirm-preflight-survey-passed `
  --confirm-grounded-lidar-passed
```

This command is documented for later supervised use. It has not been run as part
of Phase A.

## Reports

Ignored episode reports are written under:

```text
results/reports/m13/reference_controllers/
```

Reports distinguish task `episode_success` from acceptance
`report_success`. Direct-medium and random reports also declare
`expected_baseline_failure`.

Evidence includes controller privilege and digest, course identity and digests,
action/observation contracts, bounded sanitized step traces, reward/progress,
clearance and collision outcomes, LiDAR status/timestamp summaries, broad-reset
guard state, cleanup evidence, acceptance checks, errors, and limitations.

Reports exclude raw point clouds, full A* routes, obstacle dumps, settings paths,
machine paths, policy weights, and training data. Generated reports remain
ignored and must not be committed.

## Limitations

- Supervised M13.6 live evidence is pending.
- M13.3 does not model undocumented built-in Blocks geometry.
- Static solvability does not prove physical simulator collision response.
- LiDAR clearance depends on beam density and angular aggregation.
- Random and direct controllers are baselines, not robust obstacle avoidance.
- Oracle is privileged and cannot be compared with deployable learned policies as
  equivalent intelligence.
- No policy training, reactive LiDAR controller, camera, SLAM, dynamic obstacle,
  live mapping, body-frame/yaw control, or real-world UAV claim is included.

