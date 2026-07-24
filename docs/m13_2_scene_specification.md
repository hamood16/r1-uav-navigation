# M13.2 Deterministic Scene Specification

## Status And Scope

M13.2 is complete. It provides a simulator-independent, deterministic
representation of a three-dimensional navigation course and a supervised
Colosseum materialization tool. Offline validation, accepted live materialization,
deterministic reset, exact cleanup, and start-anchor positioning have all been
completed.

The accepted live stack was Colosseum Blocks `v2.0.0-beta` with the matching
AirSim-compatible client. Runtime evidence is kept at three distinct levels:

- specification and fake-client evidence from automated tests;
- live RPC and read-back evidence from ignored JSON reports;
- practical behavior confirmed separately by the operator.

A successful RPC alone is not treated as practical confirmation.

M13.2 provides:

- versioned scene and asset YAML files;
- a pads-only scene for the first supervised visual materialization;
- red start-pad and green-indicated goal-pad specifications;
- pad safety volumes, a start anchor, and a goal approach point;
- explicit and locally seeded static-obstacle layouts;
- schema-only future dynamic-obstacle data;
- pure geometry and validation;
- exact runtime names, ownership manifests, and cleanup;
- runtime-spawn, prebuilt-verification, and visual-marker strategies;
- an optional supervised `SimpleFlight` start-positioning demonstration.

It does not add dynamic obstacle motion, voxel maps, 3D A*, LiDAR observations,
obstacle-aware rewards, an environment, policy changes, training, evaluation, or
collision-response testing.

## Authoritative Live Evidence

Generated reports remain local and Git-ignored. The following files are the
human-reviewed primary evidence for milestone acceptance:

| Stage | Primary report | Post-stage survey |
| --- | --- | --- |
| Cube calibration | `results/reports/m13/scenes/m13_2_calibrate-asset_20260723T134504_aa62d819.json` | `results/reports/m13/colosseum_survey_202607231347342921100000_120082f5.json` |
| Pads-only materialization | `results/reports/m13/scenes/m13_2_materialize_20260723T141008_b82af83d.json` | `results/reports/m13/colosseum_survey_202607231412423098680000_e8596fbb.json` |
| Minimal static course | `results/reports/m13/scenes/m13_2_materialize_20260723T141615_ec47bcfb.json` | `results/reports/m13/colosseum_survey_202607231416513712520000_3ddded09.json` |
| Deterministic reset | `results/reports/m13/scenes/m13_2_materialize_20260723T141807_500140c6.json` | `results/reports/m13/colosseum_survey_202607231418577571600000_c96e222d.json` |
| Start-anchor positioning | `results/reports/m13/scenes/m13_2_materialize_20260724T091908_a4444a94.json` | `results/reports/m13/colosseum_survey_202607240919331537750000_941405ea.json` |

Earlier calibration, repeated materialization, and failed start-positioning
reports are retained as local diagnostic history but are superseded by the
accepted reports above. They are not milestone acceptance evidence.

### Report Evidence

- Calibration spawned an exact scale-one Cube, read back scale
  `(1.0, 1.0, 1.0)`, and cleaned the Cube and marker resources.
- Pads-only materialization produced the two exact deterministic object names,
  read back their requested transforms and scales, and completed object and marker
  cleanup.
- The minimal course materialized start and goal pads plus `obstacle-left` and
  `obstacle-right`. Requested and measured transforms and scales agreed within
  tolerance, including the requested yaw rotation.
- The generated-course reset report contains two complete repetitions. Both use
  scene digest
  `8db07039fc4b9d7c0cca75a0afe3b73fc16d81df6f30dc8266d1624c7d77dfb5`
  and materialization digest
  `4e28ac791ec1f64f6cbc4f51ada06ca7a90b74f708e405202cc5c878fdb2ed39`.
  Exact names, ordering, requested transforms, measured positions, measured
  scales, measured yaw, and cleanup outcomes match across repetitions.
- Start positioning completed with no report errors. Physical touchdown required
  four samples (`touchdown_confirmation_attempts: 4`), including three consecutive
  stable samples. The pre-disarm landed-state value remained `Flying`, then named
  disarm and API release completed. Final `Landed` confirmation succeeded after
  two samples, final speed was `0.0 m/s`, and API control read back disabled.

The final positioning report recorded original ground
`(0.0, 0.0, 0.5611)` and final ground
`(-0.1448, 0.0, 0.5707)` approximately. The horizontal return offset was about
`0.145 m`, within the configured `0.75 m` waypoint tolerance, so
`returned_to_original_ground` and `landing_confirmed` were both true.

Every mapped post-stage survey succeeded. The final survey independently confirmed
`SimpleFlight` landed and stationary at zero speed, API control disabled, no active
collision, `safe_for_later_stages: true`, and the validated M13 LiDAR settings
profile still matched.

### Operator Evidence

Operator observations are recorded here separately from report-generated RPC
evidence:

- The scale-one Cube and one-metre marker frame were visible. The operator accepted
  nominal Cube dimensions of `1.0 x 1.0 x 1.0 m` with `0.05 m` visual
  uncertainty.
- The red start pad and green-indicated goal pad appeared in the requested
  locations, were distinguishable, and left the initial-UAV exclusion clear.
- The minimal course showed both pads and two side obstacles. One obstacle was
  visibly yaw-rotated and the centre corridor remained clear.
- Both generated-course repetitions appeared identical.
- Start positioning showed controlled takeoff, corridor transit, arrival at the
  airborne start anchor, return toward the original safe ground, and normal
  landing without visible collision or instability.
- Exact objects and marker overlays disappeared after each accepted stage.

## Coordinate And Geometry Convention

Scene files use a local, ground-relative NED frame:

- positive `x` is north;
- positive `y` is east;
- positive `z` is down;
- local ground is `z = 0`;
- values above ground have negative `z`.

Object positions are the centre of the bottom face. The simulator pose uses the
mesh centre:

```text
world_base = measured_world_origin + local_base
world_center_z = world_base_z - requested_height / 2
```

The measured grounded `SimpleFlight` position defines the local-to-world
translation. It does not locate the start pad. The minimal scene places the start
pad four metres from local origin.

M13.2 supports yaw only. It validates yaw-rotated objects with conservative
world-aligned bounds:

```text
half_x = |cos(yaw)| * width/2 + |sin(yaw)| * depth/2
half_y = |sin(yaw)| * width/2 + |cos(yaw)| * depth/2
vertical extent = [base_z - height, base_z]
```

Workspace-boundary contact is allowed within `1e-6 m`. Contact between objects,
pad safety volumes, or the initial-vehicle exclusion counts as an intersection.
Static obstacles must preserve the configured minimum separation, initially
`0.10 m`.

## Initial Vehicle Exclusion

No solid or pad safety volume may be created under or dangerously close to the
initially grounded vehicle. The default measured exclusion is:

- horizontal clearance: `2.0 m` in both local horizontal axes;
- vertical clearance: `3.0 m` above measured ground;
- below-ground tolerance: `0.25 m`.

Before its first mutation, the live manager:

1. reads finite state for exact vehicle `SimpleFlight`;
2. requires landed, stationary, collision-safe state and disabled API control;
3. records the collision timestamp baseline;
4. calculates the measured world origin and exclusion volume;
5. translates the complete scene;
6. reruns exclusion validation in world coordinates.

Any intersection stops the command before spawning an object.

The CLI resolves the scene before importing or constructing the external client.
It rejects missing marker/flush authorization, runtime-spawn mutation or
clear-area authorization, and incomplete start-positioning flight authorization
at that boundary.

## Scene Schema

The schema is version `1`. Names must match `[a-z][a-z0-9-]*`, have no surrounding
whitespace, and contain at most 32 characters. Case-folded duplicates and
post-construction runtime-name collisions are rejected.

Primary types are:

- `WorkspaceBounds`
- `SceneReferenceSpec`
- `InitialVehicleExclusionConfig`
- `StartPad`
- `GoalPad`
- `StaticObstacle`
- `DynamicObstacle`
- `ObstacleMotion`
- `SceneGenerationConfig`
- `SceneConfig`

Pads default to `1.5 x 1.5 x 0.1 m`, a `1.0 m` horizontal safety margin, a
`3.0 m` vertical safety volume, and a `2.0 m` airborne clearance. Start anchor and
goal approach use:

```text
airborne_z = pad_base_z - pad_height - airborne_clearance
```

Dynamic-obstacle paths and speeds are validated as future schema data. M13.2
rejects dynamic objects at runtime and never moves them.

## Collision Semantics

M13.1 proved spawning, transforms, appearance, segmentation, and exact cleanup. It
did not prove physical collision response.

Scene and report fields therefore remain explicit:

- `collision_intent`
- `physical_geometry_expected`
- `physics_enabled`
- `collision_response_verified`
- `collision_geometry_complete`

Runtime Cube objects use:

```text
collision_intent: solid_expected
physical_geometry_expected: true
physics_enabled: false
collision_response_verified: false
```

`collision_geometry_complete` means that each requested solid has a verified mesh
placement. It does not mean collision response was tested. A marker-only object
sets geometry completeness to false. Every M13.2 report must retain
`collision_response_verified: false`.

## Asset Calibration Gate

`configs/scenes/m13_2_assets.yaml` now records the accepted Cube calibration:

```text
catalog_version: 2
status: accepted
evidence_level: operator_confirmed_nominal
nominal_dimensions_m: 1.0 x 1.0 x 1.0
uncertainty_m: 0.05
scale_readback_verified: true
```

The evidence reference is
`m13_2_calibrate-asset_20260723T134504_aa62d819.json`, tested against Colosseum
Blocks `v2.0.0-beta` and AirSim-compatible client `1.8.1`.

Source or asset metadata is preferred and may be accepted as `source_verified`.
Where that is unavailable, the supervised calibration command places a scale-one
Cube beside a one-metre marker frame. This can establish only
`operator_confirmed_nominal` dimensions. It is not an exact mesh-bounds
measurement.

The command wrote ignored evidence and did not edit the catalog automatically. The
catalog update was a separate, deliberate acceptance step. Runtime dimensional
materialization remains blocked for any asset without an explicitly accepted
catalog entry.

## Determinism And Identity

Generation uses only `numpy.random.default_rng(seed)`. It does not seed or consume
global Python or NumPy random state.

Generated names are `obstacle-000`, `obstacle-001`, and so on. Positions and
dimensions are normalized to six decimal places. Placement uses bounded
per-object and total attempt budgets and raises a typed error when constraints are
impossible.

Two identities are intentionally separate:

- `scene_digest`: SHA-256 of the canonical resolved local scene.
- `materialization_digest`: scene digest plus backend, backend version, asset
  catalog, accepted calibration evidence, measured world origin, and requested
  world transforms.

The same seed must reproduce the same `scene_digest` independently of where the
course is materialized.

## Runtime Ownership And Cleanup

Runtime names use:

```text
r1_uav_m13s2_<scene-id>__<specification-name>__<digest-prefix>
```

Names may not exceed 96 characters. Before spawning, the exact anchored name is
queried and must be absent. Existing objects are never adopted.

Before creating the first ownership entry, the runtime backend lists assets once
and verifies that every requested asset is available. A missing asset stops the
run without an ownership claim or spawn attempt.

Every run has an ignored atomic ownership manifest under:

```text
results/reports/m13/scenes/ownership/
```

It records run, scene, materialization, backend, exact preabsence, requested and
returned names, creation status, and cleanup status. Transitions are written to a
sibling temporary file, flushed, and atomically replaced.

Cleanup-only recovery requires an explicit manifest, `--allow-scene-mutation`,
and `--allow-recovery`. A configuration, digest, deterministic name, or prefix
cannot establish ownership. Ambiguous `creating` entries require supervised
review and are not automatically deleted.

Cleanup uses exact returned names in reverse order and verifies exact
disappearance. One failure does not skip other owned objects or marker cleanup.
If flight was attempted, named UAV cleanup occurs before scene removal.

## Rendering And Fallbacks

The primary backend is exact-name runtime spawning with physics disabled.

- Start pad: Cube plus the M13.1-validated red material.
- Goal pad: Cube providing intended physical geometry plus an explicitly
  authorized green marker outline.

No green material is assumed. Marker flushing is global and therefore requires
separate authorization.

`prebuilt_verify` is a read-only fallback. It requires exact configured object
names and verifies their transforms without repositioning them. Marker preview is
visual only and cannot claim complete expected physical geometry. Repositioning
built-in objects and automatic level switching are outside M13.2.

## Commands

Offline commands do not import or construct an AirSim client:

```powershell
.\.venv\Scripts\python.exe scripts\manage_colosseum_scene.py validate

.\.venv\Scripts\python.exe scripts\manage_colosseum_scene.py `
  --scene-config configs\scenes\m13_2_generated.yaml generate
```

Supervised nominal calibration mutates the scene and uses persistent markers:

```powershell
.\.venv\Scripts\python.exe scripts\manage_colosseum_scene.py calibrate-asset `
  --vehicle-name "SimpleFlight" `
  --asset-name "Cube" `
  --allow-scene-mutation `
  --confirm-scene-area-clear `
  --confirm-no-visible-collision `
  --allow-debug-markers `
  --allow-marker-flush `
  --hold-seconds 8
```

After an accepted catalog update, supervised materialization is:

```powershell
.\.venv\Scripts\python.exe scripts\manage_colosseum_scene.py materialize `
  --vehicle-name "SimpleFlight" `
  --backend runtime_spawn `
  --allow-scene-mutation `
  --confirm-scene-area-clear `
  --confirm-no-visible-collision `
  --allow-debug-markers `
  --allow-marker-flush `
  --hold-seconds 8
```

Exact recovery requires accepted ownership evidence:

```powershell
.\.venv\Scripts\python.exe scripts\manage_colosseum_scene.py cleanup `
  --vehicle-name "SimpleFlight" `
  --ownership-source "<ignored-ownership-manifest>" `
  --allow-scene-mutation `
  --allow-recovery
```

Start positioning additionally requires `--position-start`, `--allow-flight`,
`--allow-start-positioning`, and `--confirm-clear-airspace`. It is not a navigation
or landing-on-pad test.

The first pads-only supervised materialization uses:

```powershell
.\.venv\Scripts\python.exe scripts\manage_colosseum_scene.py `
  --scene-config configs\scenes\m13_2_pads.yaml materialize `
  --vehicle-name "SimpleFlight" `
  --backend runtime_spawn `
  --allow-scene-mutation `
  --confirm-scene-area-clear `
  --confirm-no-visible-collision `
  --allow-debug-markers `
  --allow-marker-flush `
  --hold-seconds 8
```

`--repeat` reports retain each repetition independently: local and
materialization digests, exact requested and returned names, requested
transforms, measured positions, scales and yaw, and cleanup outcomes.

Generated reports and manifests stay under `results/reports/m13/scenes/`, are
Git-ignored, and must not contain local settings, simulator paths, or unrelated
scene data.

## Start Positioning Safety

The accepted optional integration:

1. records original grounded position and collision baseline;
2. materializes the validated scene outside the initial exclusion;
3. immediately before acquiring control, rereads exact `SimpleFlight` state,
   requires landed/stationary state and disabled API control, resamples collision
   evidence, verifies the vehicle remains near the recorded ground point, and
   adopts that fresh collision baseline;
4. takes off vertically from the original safe ground;
5. follows the validated clear transit corridor;
6. reaches and verifies the start anchor;
7. returns above the original ground position;
8. commands landing at that original location;
9. confirms physical touchdown from finite, near-ground, stationary samples;
10. disarms and releases API control for exact `SimpleFlight`;
11. confirms the final `Landed` state and disabled API control;
12. cleans owned objects and markers;
13. requires a post-flight read-only survey.

The UAV never lands on an unverified runtime pad. Scene resources remain present
until the UAV no longer depends on them. This integration is optional for normal
API use but required for M13.2 milestone acceptance.

The two-phase confirmation is required by the tested SimpleFlight lifecycle:
`landAsync().join()` and physical touchdown can complete while the landed enum
still reports `Flying`. M13.2 therefore requires consecutive physical-touchdown
samples before disarming, then polls for final `Landed` after disarm and API
release. The accepted run completed both phases without invoking the outer UAV
cleanup retry.

## Completed Validation Stages

1. **Offline:** explicit and generated scenes validated and local digests
   reproduced.
2. **Read-only:** Blocks, grounded state, Cube availability, initial exclusion, and
   exact-name baseline surveyed.
3. **Calibration:** nominal Cube dimensions accepted with explicit uncertainty;
   exact Cube and global markers cleaned.
4. **Pads:** initial exclusion remained clear; start and goal were materially and
   visually confirmed; exact resources cleaned.
5. **Static scene:** requested and measured pad and obstacle transforms agreed
   within tolerance; exact cleanup completed without a collision-response claim.
6. **Deterministic reset:** the same seed reproduced local and materialization
   digests, names, ordering, transforms, and cleanup evidence across two runs.
7. **Start anchor:** the UAV reached the measured-relative anchor, returned within
   tolerance of the original ground point, completed two-phase landing
   confirmation and named control release, then removed the scene.

All live commands were separately authorized. The resulting local reports preserve
machine evidence; operator observations in this document preserve practical
evidence.

## Completion Boundary

M13.2 is complete. Accepted calibration, pads, static-scene, deterministic-reset,
cleanup, and start-anchor evidence is recorded above. M13.3 is the next milestone.

The completion claim retains these boundaries:

- `collision_response_verified: false`; physical collision response was not tested.
- Cube dimensions are operator-confirmed nominal evidence, not exact measured mesh
  bounds.
- Goal green appearance uses a persistent marker overlay; marker flushing is
  global.
- Dynamic obstacles remain schema-only and never move.
- Ownership recovery is thoroughly fake-client tested but was not live-tested by
  intentionally abandoning a scene.
- Local ignored JSON reports are evidence, not tracked project artifacts.
- No LiDAR observation, reward, environment, policy, training, M13.3, or M13.4
  functionality was added.
