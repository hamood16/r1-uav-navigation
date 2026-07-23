# M13.1 Colosseum Capability Probe And Map Survey

## Purpose

M13.1 establishes which scene, marker, LiDAR, timing, and simulator-control
capabilities are available in the validated Colosseum Blocks stack before scene
design and LiDAR observation work begins. It keeps three evidence levels separate:

1. **Static client evidence:** a method exists in the Python client.
2. **Live RPC evidence:** the method returns without an RPC error.
3. **Practical behavior:** measured state or supervised operator observation
   confirms that the capability behaves correctly in Blocks.

A successful RPC is never treated as practical support by itself. Visibility,
appearance, movement, cleanup, flight behavior, and coordinate-frame conclusions
require measured or operator-confirmed evidence.

## Milestone Phases

### Phase A: repository implementation

Phase A added the typed capability module, lazy-client CLI, fake-client tests,
protected report serialization, named-vehicle safety handling, and this
documentation framework. Automated checks remain simulator-independent and do not
require the external Colosseum client, Blocks, networking, or a GPU.

### Phase B: manual live validation

Phase B ran one explicitly approved command at a time:

1. static client inspection and a read-only scene survey;
2. temporary debug markers and marker cleanup;
3. one uniquely named temporary object and exact cleanup;
4. grounded and airborne LiDAR validation, visualization, and frame testing;
5. bounded read, LiDAR, and control performance measurements.

Phase B completed successfully for the selected scope. Every flight ended with
named hover, landing, disarming, and API-control release. Temporary objects and
markers were cleaned independently. Stage 5D pause testing was intentionally
excluded from the closeout scope.

## Validated Stack

- Colosseum Blocks: `v2.0.0-beta`
- matching source tag: `v2.0.0-beta`
- matching source commit: `7b9658a1169705ca86b21b3518fac5ba83fbe183`
- Python client: AirSim-compatible `airsim==1.8.1`
- Python: `3.11.9`
- vehicle: `SimpleFlight`
- LiDAR sensor: `LidarSensor1`
- RPC port: `41451`
- `msgpack==0.6.2`
- `msgpack-rpc-python==0.4.1`

These conclusions apply to this exact Blocks, client, vehicle, and settings
combination. The external Python client and local simulator settings are installed
separately and are not part of the normal CI dependency path.

## Provisional LiDAR Profile

The following profile was used for M13.1 capability testing:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ApiServerPort": 41451,
  "Vehicles": {
    "SimpleFlight": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "DefaultVehicleState": "Inactive",
      "Sensors": {
        "LidarSensor1": {
          "SensorType": 6,
          "Enabled": true,
          "NumberOfChannels": 16,
          "Range": 20,
          "PointsPerSecond": 100000,
          "RotationsPerSecond": 10,
          "HorizontalFOVStart": 0,
          "HorizontalFOVEnd": 359,
          "VerticalFOVUpper": 10,
          "VerticalFOVLower": -30,
          "X": 0,
          "Y": 0,
          "Z": 0,
          "Roll": 0,
          "Pitch": 0,
          "Yaw": 0,
          "DrawDebugPoints": false,
          "DataFrame": "SensorLocalFrame",
          "ExternalController": false
        }
      }
    }
  }
}
```

This is a provisional M13.1 test profile, not the final M13.4 sensor mount or
observation design. It is stored in a local, user-owned settings file, supplied to
Blocks through an explicit `-settings=<absolute-user-owned-settings-path>`
argument, and is not committed. `DefaultVehicleState: Inactive` is deliberate
because Colosseum SimpleFlight otherwise starts armed by default.

The probe read the active settings before grounded and airborne validation and
compared every listed vehicle and sensor field against the expected profile.
Airborne control was blocked on any mismatch.

## Safety Model

The default command performs static inspection only. The CLI has no combined
`all` mode. Scene mutation, marker flushing, flight, coordinate motion, and control
measurement each require explicit authorization.

Airborne probes never assume ground is NED `z=0`. They read the measured grounded
position and calculate:

```text
anchor_z = measured_ground_z - anchor_altitude
ground_clearance = measured_ground_z - measured_airborne_z
```

Before control acquisition, the probe requires:

- the exact vehicle and sensor names;
- the exact active settings profile;
- finite position and velocity;
- a landed and stationary vehicle;
- API control disabled;
- collision diagnostics that are safe or operator-confirmed expected ground
  contact;
- a successful supervised grounded-LiDAR gate;
- clear-airspace and no-visible-collision confirmations.

After movement, it checks position, velocity, collision timestamp, ground
clearance, workspace, and expected displacement. A collision event newer than the
grounded baseline is unsafe.

UAV, object, and marker cleanup are independent. Failure in one domain does not
prevent the others. The named UAV cleanup order is:

1. hover when flight was attempted;
2. land;
3. disarm;
4. disable API control.

After a timeout, interruption, Tornado failure, or `IOLoop is already running`,
the operator must stop the stage, restart Blocks, and use a fresh Python process.

## Capability Matrix

| Capability | Static client | Live RPC | Practical behavior |
|---|---|---|---|
| Connection and versions | Present | Succeeded | Compatible client and server versions returned |
| Multirotor state | Present | Succeeded | Finite grounded state confirmed |
| Collision information | Present | Succeeded | Expected stationary ground contact distinguished from new collisions |
| API-control query | Present | Succeeded | Disabled before and after accepted stages |
| Reliable read-only armed state | Absent | Not applicable | Unavailable; never inferred from appearance |
| Settings and vehicle listing | Present | Succeeded | Exact `SimpleFlight` profile matched |
| Scene-object listing | Present | Succeeded | 40 scene objects returned |
| Object poses and scales | Present | Succeeded | Finite sampled values returned |
| Asset listing | Present | Succeeded | 90 assets returned; `Cube` used safely |
| Mesh buffers | Present | Not run | Deferred because payload may be very large |
| Points, line strip, and line list | Present | Succeeded | Operator confirmed visibility |
| Coordinate transform marker | Present | Succeeded | Operator confirmed visibility |
| Persistent-marker flush | Present | Succeeded | Operator confirmed complete removal |
| Temporary spawn, pose, move, and scale | Present | Succeeded | Operator confirmed expected cube behavior |
| Material assignment | Present | Succeeded | Red appearance confirmed; no reliable read-back |
| Direct arbitrary RGB object color | Absent | Not applicable | Not supported directly |
| Segmentation ID round trip | Present | Succeeded | ID set and read back |
| Temporary-object destruction | Present | Succeeded | Exact generated name disappeared |
| Raw LiDAR | Present | Succeeded | Grounded and airborne strict gates passed |
| LiDAR coordinate frame | Present | Succeeded | Conclusive `SensorLocalFrame` result |
| Pause and continue | Present | Not run | Unverified |
| Trace-line style | Present | Not run | Method presence only |
| Runtime clock speed and view mode | Absent | Not applicable | Settings-based comparison not performed |
| State, scene, LiDAR, and control rates | Present | Succeeded | Bounded rates measured |

`simGetMeshPositionVertexBuffers` was detected statically but never called in
M13.1.

## Accepted Live Results

### Stage 1A: Static Client Inspection

The installed `airsim==1.8.1` module was inspected without constructing a
`MultirotorClient`. Static inspection recorded method availability only.
`direct_object_rgb`, runtime clock-speed control, and runtime view-mode control
were absent from the inspected client interface.

### Stage 1B: Read-Only Survey

The accepted survey confirmed:

- one vehicle named `SimpleFlight`;
- finite position and velocity;
- landed and stationary state;
- API control disabled;
- expected stationary or historical ground contact;
- `safe_for_later_stages: true`;
- 40 scene objects;
- 90 available assets;
- successful bounded pose and scale sampling;
- no scene mutation or UAV control.

The collision classifier used three read-only samples. It preserved the grounded
collision timestamp as the baseline for later airborne checks.

### Stage 2: Debug Markers

The probe created a red point, green line strip, blue line list, and coordinate
transform marker at measured-state-relative positions. The operator confirmed all
four were visible during an eight-second hold. The global persistent-marker flush
succeeded, and the operator confirmed every probe marker disappeared.

### Stage 3: Temporary Scene Mutation

The accepted mutation probe:

- selected the exact `Cube` asset returned by Stage 1B;
- spawned one UUID-named object with physics disabled;
- required the returned name to match the requested name;
- assigned the red plastic material to that exact temporary object;
- moved it approximately 0.5 m;
- changed its scale to approximately `(0.4, 0.4, 0.4)`;
- completed segmentation ID `120` round-trip;
- destroyed the exact generated object;
- verified its exact disappearance.

The operator confirmed the cube appeared red, moved, resized, and disappeared.
Material assignment has no reliable read-back API, so appearance confirmation is
the practical evidence.

### Grounded LiDAR

Three supervised grounded-LiDAR gates were run for the initial Stage 4A flight,
the supplementary visualization flight, and Stage 5C. Each gate:

- matched the exact provisional profile;
- reached a valid scan during the first warm-up attempt;
- discarded warm-up data from measured statistics;
- collected 20/20 valid measured scans;
- observed 19 fresh adjacent timestamp transitions;
- observed zero repeated or regressing timestamps;
- observed zero out-of-range points;
- found no points below 0.05 m, 0.10 m, or 0.25 m;
- classified the result as `no_evident_self_hit`;
- set `ready_for_airborne_validation: true`;
- acquired no simulator resources.

The latest gate measured a global range of approximately `0.385367 m` to
`19.999804 m`. The origin mount remains provisional and is not the final physical
LiDAR-mount validation.

### Stage 4A: Airborne LiDAR

The UAV took off, moved to a measured-ground-relative airborne anchor, hovered,
collected strict LiDAR evidence, landed, disarmed, and released API control.

Results:

- measured scans: 20;
- valid scans: 20;
- empty scans: 0;
- invalid scans: 0;
- fresh adjacent transitions: 19;
- maximum repeated timestamp run: 0;
- timestamp regressions: 0;
- points beyond configured range plus 0.10 m: 0;
- global measured range: approximately `3.419776 m` to `19.999397 m`;
- UAV cleanup: succeeded.

The operator confirmed controlled takeoff, hover, and landing with no collision,
oscillation, tipping, or unexpected movement.

### Supplementary Stage 4A Visualization

The optional visualization used one scan that had already passed strict airborne
validation. It:

- transformed `SensorLocalFrame` points into world coordinates using the reported
  finite LiDAR pose;
- evenly sampled and plotted 2,000 points from a 6,544-point source scan;
- plotted 64 sparse diagnostic rays from the measured LiDAR origin to sampled hit
  points;
- held the persistent overlay for eight seconds;
- kept UAV and marker cleanup independent.

The magenta point cloud and cyan diagnostic rays were visible and broadly aligned
with the floor and surrounding structures. The overlay was a fixed world-space
snapshot, not a continuously updating sensor display. The diagnostic rays were
markers, not physical rendered lasers. The snapshot remained briefly visible as
landing began, as expected, and all markers were flushed successfully.

### Stage 4B: Coordinate-Frame Experiment

The coordinate experiment used fresh baseline and post-rotation scans:

- final main-scan timestamp: `1784797020147785472`;
- coordinate baseline timestamp: `1784797020555794176`;
- rotated timestamp: `1784797023921865984`;
- baseline scan attempts: 1;
- rotated scan attempts: 5;
- initial yaw: `0.0°`;
- target yaw: `45.0°`;
- returned yaw: `3.718°`;
- return error: `3.718°`;
- allowed tolerance: `5.0°`;
- result: conclusive `SensorLocalFrame`.

The rotated timestamp was newer than the baseline, so stale pre-rotation data was
not used as frame evidence. The original yaw was restored within tolerance in an
inner `finally` block. The operator confirmed an approximately 45-degree in-place
rotation, return toward the original heading, stable hover, and normal landing.

### Stage 5A: Read-Only Performance

| Operation | Success | Min ms | Mean ms | p95 ms | Max ms | Calls/s |
|---|---:|---:|---:|---:|---:|---:|
| Multirotor state | 20/20 | 0.279 | 0.351 | 0.419 | 0.456 | 2841.797 |
| Scene-object listing | 20/20 | 0.588 | 1.380 | 2.108 | 2.137 | 724.483 |

No control, flight, LiDAR, marker, object, or pause resource was acquired.

### Stage 5B: LiDAR Performance

| Operation | Success | Min ms | Mean ms | p95 ms | Max ms | Calls/s |
|---|---:|---:|---:|---:|---:|---:|
| Multirotor state | 20/20 | 0.344 | 0.438 | 0.545 | 0.552 | 2278.605 |
| Scene-object listing | 20/20 | 0.504 | 1.960 | 2.376 | 14.949 | 510.075 |
| LiDAR scan | 20/20 | 30.998 | 35.373 | 41.196 | 42.640 | 28.269 |

The test was read-only and acquired no simulator resources.

### Stage 5C: Bounded Control Performance

The flight-authorized benchmark issued ten named zero-velocity commands with a
configured duration of 0.1 seconds:

| Operation | Success | Min ms | Mean ms | p95 ms | Max ms | Calls/s |
|---|---:|---:|---:|---:|---:|---:|
| Multirotor state | 10/10 | 0.286 | 0.392 | 0.497 | 0.497 | 2543.817 |
| Scene-object listing | 10/10 | 0.600 | 1.350 | 2.570 | 2.570 | 740.099 |
| Zero-velocity control | 10/10 | 125.611 | 126.958 | 129.642 | 129.642 | 7.877 |

The operator confirmed normal takeoff, stable hover at the expected local anchor,
no significant translation, and normal landing. Named UAV cleanup succeeded.

The final read-only survey recorded:

- position approximately `(0.0, 0.0, 0.5704)`;
- velocity `(0.0, 0.0, 0.0)` and speed `0.0`;
- landed state;
- API control disabled;
- expected stationary ground contact;
- `safe_for_later_stages: true`;
- exact provisional LiDAR profile still matched.

Stage 5D was intentionally not run because pause testing was optional and outside
the selected closeout scope. Simulator pause behavior remains unverified.

## Generated Reports

Each invocation wrote a unique JSON report under:

```text
results/reports/m13/
```

That directory is ignored by Git. The reports remain local evidence and are not
committed. They contain sanitized settings and summaries, not raw settings, raw
point clouds, mesh buffers, local simulator paths, or editable-install paths.

Primary evidence reports:

| Evidence | Local report |
|---|---|
| Static inspection | `colosseum_inspect-client_202607221141198435140000_690f0ea7.json` |
| Accepted Stage 1B survey | `colosseum_survey_202607221809559969400000_ba5683f0.json` |
| Accepted markers | `colosseum_markers_202607221832431245580000_8e8261fe.json` |
| Accepted mutation | `colosseum_mutation_202607221934030690120000_35c7821e.json` |
| Initial profile survey | `colosseum_survey_202607230800447847020000_7166ffd1.json` |
| Initial grounded LiDAR | `colosseum_grounded-lidar_202607230804511180900000_c499691f.json` |
| Stage 4A | `colosseum_lidar_202607230808384342330000_1590c742.json` |
| Stage 4A post-flight | `colosseum_survey_202607230817226093250000_adada539.json` |
| Visualization repeat | `colosseum_lidar_202607230844169734470000_367bef66.json` |
| Visualization post-flight | `colosseum_survey_202607230848517858910000_65c40ddf.json` |
| Stage 4B | `colosseum_lidar_202607230901287649720000_7d3aad82.json` |
| Stage 4B post-flight | `colosseum_survey_202607230904088225610000_e076068a.json` |
| Stage 5A | `colosseum_performance_202607230905450762930000_5d9b7789.json` |
| Stage 5B | `colosseum_performance_202607230908197822570000_f0ab162b.json` |
| Stage 5C grounded gate | `colosseum_grounded-lidar_202607230913440298170000_06fcc816.json` |
| Stage 5C | `colosseum_performance_202607230915310791740000_ebc76731.json` |
| Final survey | `colosseum_survey_202607230917004222950000_3de9507d.json` |

Earlier collision-diagnostic surveys and first marker/mutation runs are retained
locally as superseded or exploratory evidence. The first marker and mutation RPC
runs succeeded but lacked practical visual confirmation. Two intermediate surveys
failed conservatively while collision diagnostics were being refined.

## Reproducible Commands

Every live command requires Blocks to be launched separately with the exact
user-owned profile. These commands must be run one at a time under supervision.

### Static And Read-Only

Static client inspection, no client construction:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py inspect-client
```

Read-only survey:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py survey --vehicle-name "SimpleFlight" --max-objects 100 --confirm-no-visible-collision
```

Grounded read-only LiDAR:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py grounded-lidar --vehicle-name "SimpleFlight" --lidar-name "LidarSensor1" --warm-up-attempts 10 --warm-up-interval 0.2 --scan-count 20 --scan-interval 0.2 --confirm-no-visible-collision
```

Read-only performance:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py performance --vehicle-name "SimpleFlight" --iterations 20
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py performance --vehicle-name "SimpleFlight" --iterations 20 --include-lidar --lidar-name "LidarSensor1"
```

### Scene Mutation

Marker creation and global marker flush:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py markers --allow-debug-markers --allow-marker-flush --marker-hold-seconds 8
```

Temporary object mutation:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py mutation --allow-scene-mutation --confirm-spawn-area-clear --confirm-vehicle-disarmed --asset-name "Cube" --mutation-hold-seconds 5 --material-name "/AirSim/Models/MiniQuadCopter/Prop_Red_Plastic.Prop_Red_Plastic"
```

### Flight Authorized

Basic airborne LiDAR:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py lidar --vehicle-name "SimpleFlight" --lidar-name "LidarSensor1" --allow-flight --confirm-clear-airspace --confirm-no-visible-collision --confirm-grounded-lidar-passed --scan-count 20 --scan-interval 0.2
```

Optional visual repeat:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py lidar --vehicle-name "SimpleFlight" --lidar-name "LidarSensor1" --allow-flight --confirm-clear-airspace --confirm-no-visible-collision --confirm-grounded-lidar-passed --scan-count 20 --scan-interval 0.2 --visualize-lidar --allow-marker-flush --lidar-visualization-hold-seconds 8 --lidar-visualization-max-points 2000 --lidar-visualization-max-rays 64
```

Coordinate-frame experiment:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py lidar --vehicle-name "SimpleFlight" --lidar-name "LidarSensor1" --allow-flight --confirm-clear-airspace --confirm-no-visible-collision --confirm-grounded-lidar-passed --allow-coordinate-motion --coordinate-frame-experiment --scan-count 20 --scan-interval 0.2
```

Bounded control performance:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py performance --vehicle-name "SimpleFlight" --iterations 10 --include-control --lidar-name "LidarSensor1" --allow-flight --confirm-clear-airspace --confirm-no-visible-collision --confirm-grounded-lidar-passed --control-duration 0.1
```

### Optional And Not Run

The following Stage 5D command exists but was intentionally not run:

```powershell
.\.venv\Scripts\python.exe scripts\check_colosseum_capabilities.py performance --vehicle-name "SimpleFlight" --iterations 5 --probe-pause --allow-pause --confirm-no-visible-collision
```

Its presence documents the interface only and is not evidence that pause behavior
works in Blocks.

## Implications

M13.1 establishes that:

- raw LiDAR access is viable with finite non-empty point triples, fresh timestamps,
  finite pose, and a usable bounded RPC rate;
- downstream processing must respect `SensorLocalFrame` and transform points when
  world coordinates are required;
- runtime temporary-object spawning, movement, scaling, segmentation, material
  assignment, exact destruction, and marker cleanup work in the tested stack;
- tested RPC rates are sufficient capability evidence but are not a production
  performance guarantee or a full training-throughput benchmark;
- M13.2 may proceed with scene specification and visible start/goal work using the
  validated runtime capabilities and documented fallbacks;
- later M13.4 observation design can use the resolved `SensorLocalFrame` result.

M13.1 does not implement M13.2 or M13.4 functionality.

## Limitations

- The reliable read-only armed state is unavailable. Safe startup relies on
  `DefaultVehicleState: Inactive`, explicit authorization, and named cleanup.
- Pause behavior remains unverified because Stage 5D was intentionally not run.
- Mesh-buffer retrieval, texture assignment, trace-style mutation, and alternate
  clock/view configurations were not live-tested.
- Material appearance required operator confirmation because no reliable read-back
  exists.
- The diagnostic overlay is a bounded fixed snapshot, not a production
  visualizer.
- The LiDAR origin mount remains provisional.
- Performance numbers are specific to the tested machine, profile, simulator
  session, graphics state, and legacy RPC stack.
- The legacy RPC `.join()` path still lacks a safe in-process timeout and can
  require a Blocks restart after interruption or IOLoop corruption.
- Generated JSON reports remain ignored local evidence and are not available from
  a source-only checkout.
- No LiDAR sectorisation, obstacle features, scene specification, procedural
  course generation, obstacle-aware environment, reward change, training,
  curriculum, or dynamic-obstacle behavior is included.
