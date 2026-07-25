# M13.4 LiDAR Sensing And Fixed-Size Feature Extraction

## Status And Scope

M13.4 is complete. It implements deterministic parsing, fixed-size feature
extraction, an opt-in Gymnasium environment, structured diagnostics, and
sensor-failure handling. Its grounded, known-geometry, and bounded airborne smoke
stages passed supervised live validation.

M13.4 does not add obstacle-aware training, rewards, policy changes, occupancy
mapping, SLAM, camera input, dynamic replanning, or M13.5 functionality.
M13.5 is the next milestone.

## Validated Starting Evidence

M13.1 established raw LiDAR support for:

- Colosseum Blocks `v2.0.0-beta`;
- source commit `7b9658a1169705ca86b21b3518fac5ba83fbe183`;
- AirSim-compatible client `1.8.1`;
- vehicle `SimpleFlight`;
- sensor `LidarSensor1`;
- `SensorLocalFrame`;
- finite, non-empty grounded and airborne scans with fresh timestamps.

The local, user-owned AirSim settings profile is not committed. Its validated
sensor uses a 20 m range, 24-compatible full horizontal coverage, and vertical FOV
from `-30` to `+10` degrees.

## Sensor-Local Coordinates

The tested Colosseum source uses local NED axes:

- `+x`: forward;
- `+y`: right;
- `-y`: left;
- `+z`: down;
- `-z`: up.

For one point `(x, y, z)`:

```text
range = sqrt(x^2 + y^2 + z^2)
horizontal = atan2(y, x)
elevation = atan2(-z, sqrt(x^2 + y^2))
```

Policy sectorisation uses these sensor-local values directly. Sensor pose is not
required for policy features or the LiDAR-valid flag. Pose validity is reported
separately and is required only for optional world-space evidence.

## Feature Contract

The committed profile is
`configs/sensing/m13_4_lidar_features.yaml`, schema version 1.

- 24 horizontal sectors over `[-180, 180)`, 15 degrees each.
- Lower elevation band: `[-30, -15)`.
- Middle elevation band: `[-15, 0)`.
- Upper elevation band: `[0, 10]`.
- Outer-FOV comparison tolerance: `0.0001` degrees.
- Elevation-major ordering: lower 24, middle 24, upper 24.
- Nearest range per sector.
- Empty sectors use the 20 m maximum.
- Clipped normalization maps 0.25 m to `0.0` and 20 m to `1.0`.

The obstacle-aware observation is always:

```text
indices 0-9:   existing M12 navigation values, unchanged
indices 10-81: 72 normalized LiDAR sector distances
index 82:      LiDAR-valid flag
```

Its shape is `(83,)`, dtype is `numpy.float32`, and every value is finite. The
legacy `ColosseumUAVEnv` remains unchanged with shape `(10,)`.

The policy does not receive exact obstacle coordinates, scene seeds, M13.3
occupancy, A* paths, or reference-path length.

The accepted grounded run observed elevations from `-30.000006823279076` to
`10.000001974024432` degrees. Its maximum numerical overshoots were
`0.000006823279076` degrees below the configured lower bound and
`0.000001974024432` degrees above the upper bound. The `0.0001`-degree value is
numerical comparison tolerance, not additional sensor coverage: accepted
overshoots are clamped to the corresponding outer edge before assignment, and
the three elevation bands remain exactly `[-30, -15)`, `[-15, 0)`, and
`[0, 10]`.

## Strict Scan Policy

An empty cloud is invalid. A valid-no-obstacle scan must contain usable finite
in-FOV points whose ranges all clip to maximum range; it produces 72 values of
`1.0` and a valid flag of `1.0`.

The entire scan is rejected for:

- non-numeric data;
- a coordinate count not divisible by three;
- any NaN or infinite coordinate;
- any range above `maximum_range + 0.10 m`;
- any elevation beyond the configured outer FOV by more than `0.0001` degrees;
- missing, repeated, regressing, or otherwise invalid timestamps.

Finite near points are retained and clipped to minimum range. Finite points no
more than 0.10 m beyond configured range are clipped to maximum. This deliberately
prefers a sensor fault over an incomplete false all-clear observation.

Ground filtering is disabled and is the only supported M13.4 mode. Downward and
upward returns remain available. The active sensor FOV does not provide full
nadir or steep overhead coverage, and M13.4 makes no such claim.

## Timestamp And Failure Contract

The first positive timestamp is `FIRST_VALID`; an increase is `FRESH`. Equal
timestamps are invalid for policy use. The third consecutive equal transition is
`STALE_LIMIT_EXCEEDED`.

One invalid scan substitutes 72 maximum-range values, sets the valid flag to
`0.0`, and records a structured fault. A valid fresh scan resets the fault count.
The third consecutive invalid scan:

1. sends a named bounded zero-velocity command;
2. attempts named hover;
3. returns `truncated=True`;
4. sets `sensor_failure=True`;
5. rejects further `step()` calls until named reset cleanup or close.

Hover is not described as landing. Reset and close retain responsibility for
named hover, landing, disarming, and API-control release.

## Environment Lifecycle

`ColosseumLidarUAVEnv` is separate from the legacy M12 environment. It:

- never calls broad simulator `reset()`;
- uses exact `SimpleFlight` routing for every vehicle RPC;
- uses exact `SimpleFlight` and `LidarSensor1` routing for LiDAR;
- safely cleans up control state it owns before the next episode;
- accepts externally resolved start-anchor and goal-approach positions;
- never creates, adopts, resets, or removes M13.2-owned scene objects.

The M13.2 start-positioning helper also accepts an optional bounded read-only
LiDAR callback. The callback runs only after verified anchor arrival and hover.
Callback success, exception, timeout, or interruption is followed by the existing
return-to-original-ground, landing, disarm, and API-release sequence before its
outcome is returned or propagated.

The opt-in LiDAR environment applies the same safety shape independently:

- it captures the complete original grounded position and collision baseline
  before takeoff;
- it confirms start-anchor arrival only after named hover and three consecutive
  bounded samples within `0.75 m` and `0.1 m/s`;
- cleanup returns to an airborne point above the recorded original ground
  location rather than deliberately landing on a runtime pad;
- three stable touchdown samples permit disarming before the final bounded
  `Landed`-state and API-release confirmation;
- a successful cleanup clears owned control state, while one recovery retry is
  permitted only after a genuine prior cleanup failure.

The airborne runner verifies final landed, stationary, API-disabled, and
original-ground evidence before deleting exact M13.2-owned objects or flushing
markers. If that state cannot be established, scene cleanup is deliberately
deferred and the report records why.

## Evidence And Reports

Schema versions:

- LiDAR feature configuration: 1.
- LiDAR extraction evidence: 1.
- LiDAR live-validation report: 1.

Ignored reports are written under `results/reports/m13/lidar/`. Reports include
the configuration digest, feature shape, aggregate or selected sector evidence,
timestamps, fault state, exact vehicle/sensor names, cleanup, and limitations.

Reports exclude raw clouds, local settings paths, A* paths, oracle path lengths,
and full obstacle-coordinate dumps.

### Authoritative Report Map

| Stage | Authoritative ignored report |
| --- | --- |
| Grounded feature validation | `results/reports/m13/lidar/m13_4_grounded_20260725T063131179616Z_aa53da34.json` |
| Known-geometry pre-survey | `results/reports/m13/colosseum_survey_202607250719101386670000_73e6e7a4.json` |
| Known-geometry live validation | `results/reports/m13/lidar/m13_4_known_geometry_live_20260725T071940221924Z_4d1847e5.json` |
| Known-geometry post-survey | `results/reports/m13/colosseum_survey_202607250722051013930000_c753bce6.json` |
| Airborne-smoke pre-survey | `results/reports/m13/colosseum_survey_202607250854577697480000_ffe7c5f0.json` |
| Airborne observation smoke test | `results/reports/m13/lidar/m13_4_airborne_smoke_20260725T085525896971Z_80d04938.json` |
| Airborne-smoke post-survey | `results/reports/m13/colosseum_survey_202607250859423490430000_dd3497f0.json` |

The earlier failed airborne-smoke report
`m13_4_airborne_smoke_20260725T072507331966Z_1ff81126.json` is superseded
development evidence. It is not accepted milestone evidence.

### RPC Evidence

The accepted grounded report recorded:

- 20 requested and collected scans;
- 20 policy-valid scans and zero invalid scans;
- finite bounded feature vectors with shape `(72,)`;
- `FIRST_VALID` followed by `FRESH` timestamps;
- exact settings-profile and safe grounded-collision checks;
- no control, object, or marker resources acquired;
- the accepted elevation extrema and overshoots documented above.

Its configuration digest is:

```text
aa446458d68d42364d24a05038d499b835e3d106a23bd5ec86112d2afc9f34cb
```

The accepted known-geometry report used the authoritative M13.3 `easy` profile
with base and accepted candidate seed `1100`. All 10 scans were policy-valid,
with `FIRST_VALID` followed by `FRESH`. At flattened feature index `31`, the
expected nearest-surface distance was `5.145347554864271 m`; the measured median
was `5.331569727510214 m`, for an absolute difference of
`0.186222172645943 m`, within the accepted `0.50 m` tolerance. The run returned
to the original ground area, confirmed landing, released API control, and
successfully cleaned its exact objects and markers.

The accepted airborne report used the authoritative M13.3 `medium` profile with
base and accepted candidate seed `2100`. It recorded one reset observation and
five step observations. All six were `numpy.float32`, finite, contained by the
observation space, shape `(83,)`, and carried a LiDAR-valid flag of `1.0`.
Timestamps were `FIRST_VALID` then `FRESH`; no sensor failure, truncation,
collision termination, bounds termination, or unexpected task termination
occurred. Start-anchor arrival was confirmed by three consecutive stable samples.
The environment returned to the original ground area, landed, reached zero
measured speed, and released API control. Broad simulator reset was not used.
Exact object and marker cleanup succeeded after named UAV cleanup.

Each pre/post survey in the table succeeded and recorded `SimpleFlight` landed,
speed `0.0 m/s`, API control disabled, the exact LiDAR profile matched, and
`safe_for_later_stages: true`.

### Operator Evidence

The known-geometry and airborne commands were accepted as supervised runs after
separate operator observation. Operator observation is practical-behaviour
evidence; it is kept distinct from RPC measurements and does not supply or alter
the distance, timestamp, observation-shape, or cleanup values above. The ignored
JSON reports remain the authoritative numerical and lifecycle records.

## Commands

Offline configuration and fixture validation:

```powershell
.\.venv\Scripts\python.exe scripts\check_lidar_features.py validate
```

Offline accepted easy-seed comparison evidence:

```powershell
.\.venv\Scripts\python.exe scripts\check_lidar_features.py known-geometry --course-profile easy --course-seed 1100
```

Accepted supervised grounded read-only command:

```powershell
.\.venv\Scripts\python.exe scripts\check_lidar_features.py grounded --vehicle-name "SimpleFlight" --lidar-name "LidarSensor1" --scan-count 20 --scan-interval 0.2 --allow-live-rpc --confirm-no-visible-collision
```

Accepted supervised known-geometry command:

```powershell
.\.venv\Scripts\python.exe scripts\check_lidar_features.py known-geometry-live --course-profile easy --course-seed 1100 --vehicle-name "SimpleFlight" --lidar-name "LidarSensor1" --scan-count 10 --scan-interval 0.2 --allow-live-rpc --allow-scene-mutation --confirm-scene-area-clear --confirm-no-visible-collision --allow-debug-markers --allow-marker-flush --allow-flight --allow-start-positioning --confirm-clear-airspace
```

Accepted supervised airborne observation command:

```powershell
.\.venv\Scripts\python.exe scripts\check_lidar_features.py airborne-smoke --course-profile medium --course-seed 2100 --vehicle-name "SimpleFlight" --lidar-name "LidarSensor1" --step-count 5 --allow-live-rpc --allow-scene-mutation --confirm-scene-area-clear --confirm-no-visible-collision --allow-debug-markers --allow-marker-flush --allow-flight --allow-start-positioning --confirm-clear-airspace
```

Both live commands require every authorization shown. Course profile, declared
seed, exact names, bounded counts, report path, and M13.3 solvability are checked
before simulator-client import. They materialize through the M13.2 ownership
lifecycle and require named UAV cleanup before exact scene and marker cleanup.

## Completed Phase B Validation

The easy-seed known-geometry stage accepted one attributable comparison at
feature index `31`. The medium-seed stage accepted the opt-in LiDAR environment's
fixed-size observation and named lifecycle through bounded reset and zero-action
steps. Neither stage trained or evaluated a policy.

The known-geometry check validates one attributable return, not every obstacle.
Built-in Blocks geometry is not represented by M13.3. Cube dimensions remain
operator-confirmed nominal evidence. Sector values depend on LiDAR beam density
and angular aggregation, and the configured vertical FOV remains limited. The
numerical boundary tolerance does not expand sensor coverage. Physical simulator
collision response remains unproven.

The authoritative medium template does not request a start-pad material or marker
colour, so its materialization performs no start-pad material RPC. This cosmetic
template property is independent of LiDAR smoke acceptance. M13.4 does not alter
the protected M13.3 scene digest or feasibility baseline to change its appearance.

There is still no trained obstacle-aware TD3 policy. M13.4 adds no camera input,
SLAM, dynamic-obstacle sensing or prediction, or real-world UAV claim. The
opt-in LiDAR Gymnasium environment is a validated sensing interface, not a trained
obstacle-avoidance system.

## Acceptance Boundary

M13.4 is complete. Phase A established the feature and lifecycle contracts;
grounded and Phase B evidence established finite bounded extraction, one
accepted known-distance comparison, an `(83,)` airborne observation smoke test,
safe final states, and successful cleanup. M13.5 is the next milestone.
