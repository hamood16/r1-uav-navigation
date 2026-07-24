# M13.3 Static Course Generation And 3D Solvability

## Status And Scope

M13.3 is complete as a simulator-independent static-course validation milestone.
It adds deterministic 3D voxel occupancy, conservative clearance, deterministic
26-connected A*, bounded candidate rejection, and reference-path evidence on top
of the unchanged M13.2 scene specification.

The proof covers only the configured M13.3 workspace and configured static course
obstacles. It does not include undocumented built-in Blocks level geometry, prove
physical simulator collision response, consume live LiDAR, alter observations or
rewards, or add policy training. No live M13.3 simulator validation was required
or performed.

## Validation Pipeline

Each declared course follows one fixed offline pipeline:

```text
M13.2 scene template and declared base seed
-> resolved deterministic static scene
-> yaw-aware conservative obstacle AABBs
-> 0.50 m L-infinity obstacle inflation
-> 0.50 m workspace erosion
-> 0.25 m voxel occupancy
-> deterministic 3D A*
-> continuous endpoint and segment post-validation
-> accepted course and reference-path evidence
```

For candidate attempt `i`, the generation seed is `base_seed + i`. Candidate
construction immutably replaces the template generation seed before calling the
existing M13.2 generation and validation functions. M13.2 canonicalization and
`scene_digest` semantics are unchanged.

## Clearance And Coordinates

All geometry uses the M13.2 local ground-relative NED convention: positive `x`
and `y` are horizontal, while more-negative `z` is higher. The fixed planning
clearance is calculated from:

```text
uav_collision_radius_m       = 0.35
additional_safety_margin_m   = 0.15
calculated_total_clearance_m = 0.50
```

A serialized `total_clearance_m` must match the calculated sum within `1e-6 m`.
The same calculated clearance expands every already-conservative yaw-derived
obstacle AABB on all six faces and erodes the workspace. This is conservative
axis-aligned L-infinity inflation, not exact spherical configuration-space
geometry.

The geometry tolerance is `1e-6 m`. Voxel cells are half-open:

```text
[minimum + i * resolution, minimum + (i + 1) * resolution)
```

The exact workspace maximum maps explicitly to the final voxel. Closed-volume
contact is used only for conservative obstacle rasterization and boundary safety;
adjacent free voxels are not treated as mutually overlapping.

## Endpoints And Paths

Before index mapping, the exact start anchor and goal approach must be finite,
distinct, inside the eroded workspace, and outside every inflated obstacle. An
occupied mapped endpoint is rejected; occupancy is never cleared to force a path.

The direct-line result uses deterministic 3D supercover traversal. Every voxel
touched at a face, edge, or corner must be free, and the exact segment must remain
inside the eroded workspace without intersecting an inflated obstacle.

The A* search uses deterministic 26-connectivity, Euclidean axial/edge/corner
costs, a Euclidean admissible heuristic, stable tie-breaking, and intermediate
cell checks that prevent diagonal corner cutting. Search is bounded by
`1,000,000` voxels and `250,000` expansions.

After reconstruction, validation checks the exact start-to-first-centre
connector, every centre-to-centre segment, and the final-centre-to-exact-goal
connector using both supercover occupancy and continuous segment/AABB geometry.

`reference_path_length_m` is the primary path-length field. It is an approximate
oracle length at the configured voxel resolution, not a continuous-space optimum.
`path_efficiency_ratio` is:

```text
direct_start_goal_distance_m / reference_path_length_m
```

## Course Suite

[`m13_3_voxel_astar.yaml`](../configs/planning/m13_3_voxel_astar.yaml) is the
authoritative profile registry. Each profile identifies exactly one scene
template, declared base seeds, a maximum 32-attempt budget, mandatory numeric
constraints, and accepted feasibility baselines.

| Profile | Split | Base seeds | Obstacles | Direct line | Efficiency | Vertical excursion |
| --- | --- | --- | ---: | --- | --- | --- |
| `empty` | validation | `0` | 0 | clear | 0.95-1.00 | 0-0.25 m |
| `easy` | training | `1100,1200,1300,1400` | 2-4 | clear | 0.80-1.00 | 0-0.50 m |
| `medium` | training | `2100,2200,2300,2400` | 5-8 | blocked | 0.55-0.95 | 0.50-3.00 m |
| `hard` | training | `3100,3200,3300,3400` | 10-14 | blocked | 0.30-0.8333 | at least 1.00 m |
| `held-out-reverse` | held out | `9100,9200,9300` | 5-10 | blocked | 0.50-0.95 | 0-4.00 m |
| `held-out-elevated` | held out | `10100,10200,10300` | 5-10 | blocked | 0.50-0.95 | 1.00-5.00 m |

`medium`, `hard`, and held-out templates also require named fixed structures.
Held-out profiles use disjoint seed ranges and different start-to-goal direction
contracts. Descriptions such as wide passages or dead ends are template-design
intentions; only the numeric and named-structure constraints are validated. No
general dead-end detector exists.

## Feasibility Checkpoint

All 19 shipped base seeds were recomputed with the fixed safety rules and accepted
at attempt zero, so each accepted candidate seed equals its base seed. The tracked
YAML baselines preserve exact scene, occupancy, and solvability digests plus:

| Profile | Reference length | Efficiency | Vertical excursion | Expanded nodes |
| --- | ---: | ---: | ---: | ---: |
| `empty` | 9.357071 m | 0.961839 | 0.025 m | 37 |
| `easy` | 9.357071 m | 0.961839 | 0.025 m | 37 |
| `medium` | 11.635246 m | 0.773512 | 2.775 m | 3,868-4,472 |
| `hard` | 13.903013-14.195906 m | 0.633986-0.647342 | 4.275 m | 7,800-10,173 |
| `held-out-reverse` | 11.221032 m | 0.802065 | 2.275 m | 3,188-3,969 |
| `held-out-elevated` | 13.612670 m | 0.834356 | 2.775 m | 5,011-5,376 |

Tests recompute every declared seed and compare the accepted candidate, attempt,
digests, path metrics, expanded-node count, and direct-line result with the tracked
baseline. A profile that exhausts its budget is a failure.

## Evidence And Identity

M13.3 adds separate versioned evidence without changing M13.2 scene identity:

- voxel planner configuration schema: `1`
- course-suite schema: `1`
- occupancy evidence schema: `1`
- solvability evidence schema: `1`
- saved course report schema: `1`

`scene_digest` identifies the resolved M13.2 physical scene.
`occupancy_digest` identifies its planner configuration, inflated occupancy, and
occupancy schema. `solvability_digest` additionally identifies endpoint, path, and
solvability evidence. Reference waypoints and path length are evidence, not inputs
to M13.2 scene identity.

Generated JSON reports default to `results/reports/m13/courses/`, remain ignored
by Git, and contain no simulator settings or machine-specific paths.

## Commands

These commands are offline and do not import or construct an AirSim client:

```powershell
.\.venv\Scripts\python.exe scripts\validate_static_course.py validate --profile empty
.\.venv\Scripts\python.exe scripts\validate_static_course.py validate --profile easy --seed 1100
.\.venv\Scripts\python.exe scripts\validate_static_course.py validate-all
```

The existing M13.2 manager also accepts an optional pre-client course gate:

```powershell
.\.venv\Scripts\python.exe scripts\manage_colosseum_scene.py --course-profile easy --course-seed 1100 validate
```

For a supplied profile, its scene-template path is authoritative. An explicitly
supplied conflicting `--scene-config` is rejected. Arbitrary seeds are rejected.
An unsolvable course fails before simulator client import, API control, or flight.
Existing M13.2 commands without course arguments retain their prior behavior.

The manager can apply the same gate before an explicitly authorized M13.2 live
materialization, but M13.3 itself requires no live run and adds no flight command.

## Limitations And Next Step

- The occupancy proof omits undocumented built-in Blocks geometry.
- Conservative nominal configured geometry is not verified simulator collision
  response.
- Voxel resolution and L-infinity inflation can reject routes that finer or exact
  geometry might permit.
- Static obstacle solvability does not imply dynamic-obstacle solvability.
- The reference path is planning evidence, not a learned-policy trajectory.

M13.3 establishes that every shipped static course has a deterministic safe
reference route under its declared model. M13.4 may build obstacle-aware
observations on that foundation without changing the completed M12 or M13.2
interfaces.
