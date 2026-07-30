# M13.8 Static-Obstacle Training Curriculum

## Status

M13.8 Phase A implements simulator-independent curriculum infrastructure. It does
not run Blocks, perform long TD3 training, or demonstrate learned obstacle
avoidance.

The implementation provides:

- A strict six-stage curriculum configuration.
- M13.8-only deterministic course templates validated by the unchanged M13.3
  voxel and A* pipeline.
- Separate training, curriculum-validation, and final-test seed roles.
- Deterministic local-RNG course and robustness samplers.
- Serializable curriculum state and pure promotion decisions.
- Sanitized route-shape and controller-baseline evidence.
- Additive M13.7 checkpoint and full-resume metadata.
- Offline CLI validation and a bounded fake TD3 checkpoint/resume smoke.

Phase B live pilots remain future supervised work. No live command is exposed by
the M13.8 script.

## Protected Contracts

M13.8 does not change:

- The M12 10-value observation, action, reward, or TD3 baseline.
- M13.3 accepted profiles, digests, or solvability rules.
- The M13.4 83-value LiDAR observation.
- M13.5 reward, termination, sensor safety, or cleanup behavior.
- M13.6 controller matrices or report semantics.
- M13.7 behavior or digests when curriculum mode is disabled.

Curriculum-enabled M13.7 runs add a curriculum ID and configuration digest.
Ordinary M13.7 snapshots omit both fields exactly as before.

## Curriculum Stages

| Stage | Purpose | Minimum / maximum stage steps |
| --- | --- | ---: |
| 0 | Confirm basic learning with the LiDAR-compatible empty observation | 500 / 5,000 |
| 1 | Learn a deviation around one fixed direct-route blocker | 5,000 / 20,000 |
| 2 | Learn across easy and sparse blocked static courses | 10,000 / 50,000 |
| 3 | Generalize across forward, reverse, lateral, and elevated layouts | 25,000 / 100,000 |
| 4 | Retain earlier performance while mixing easy, medium, and hard courses | 50,000 / 200,000 |
| 5 | Prepare moderate policy-view robustness perturbations offline | 50,000 / 200,000 |

Promotion depends on deterministic validation rather than training episodes.
Every gate must pass in three consecutive complete windows. Cleanup failure is a
hard gate. Reaching a maximum budget without passing records a blocked stage and
does not promote it.

Stage 4 increases hard-course probability only after validation:

1. Easy/medium/hard `0.60/0.30/0.10`.
2. `0.45/0.35/0.20`.
3. `0.30/0.40/0.30`.

Advancing a probability level resets the consecutive-promotion count.

## Course Evidence

The authoritative curriculum course suite is
`configs/planning/m13_8_curriculum_courses.yaml`. It uses the same `0.25 m`
voxels, `0.35 m` UAV radius, `0.15 m` margin, conservative AABB inflation, and
26-connected A* as M13.3.

The proposed Stage 1 obstacle was tested offline at:

- base centre `(8.5, 0.0, 0.0)`;
- dimensions `(1.0, 4.0, 3.0) m`;
- yaw `0 degrees`.

The unchanged M13.3 validator accepted it without tuning. The direct line is
blocked, one obstacle is present, and the reference path remains solvable after
the existing `0.50 m` clearance. This is offline configured-geometry evidence,
not physical collision-response evidence.

Every declared M13.8 curriculum course is regenerated during strict validation.
Validation records the accepted candidate, scene, occupancy, and solvability
digests. The locked aggregate feasibility digest is
`7ab64205ccb7fd055d2ed0b5f42528d1e7be0772ed46224197427b96c6baa1b6`.
Generated geometry and reference paths are not policy observations.

## Seed Separation

M13.8 divides existing M13.3 training seeds into curriculum roles:

- Easy training `1100/1200`; validation `1300/1400`.
- Medium training `2100/2200`; validation `2300/2400`.
- Hard training `3100/3200`; validation `3300/3400`.

M13.8-owned seeds begin at `20000`; candidate blocks are separated by at least
100.

The following M13.3 seeds are reserved exclusively for M13.9 final testing:

- reverse `9100/9200/9300`;
- elevated `10100/10200/10300`.

The M13.8 loader rejects any training or promotion configuration containing one
of those final-test seeds. M13.7's existing default validation configuration is
unchanged; curriculum mode uses its own validation pools.

## Route-Shape Evidence

M13.5 already reports travelled path length, goal progress, and LiDAR clearance.
It does not report a trajectory. The M13.8 evaluation-only collector uses the
confirmed normalized relative-position prefix and immutable controller scales to
maintain streaming summaries:

- travelled/direct ratio;
- maximum perpendicular, lateral, and vertical deviation;
- minimum valid or retained clearance;
- point count;
- initial/final relative positions;
- deterministic digest of quantized samples.

It does not retain or report a raw trajectory. Stage 1 qualifies as non-straight
only when a successful, collision-free episode has a travelled/direct ratio of
at least `1.05` and a maximum perpendicular deviation of at least `1.0 m`.

## Baseline Evidence

Random and direct controllers are non-privileged. The oracle A* controller is
privileged and is not a deployable or learned policy.

Phase A accepts only explicitly labelled fake baseline evidence for exercising
gate logic. Baseline evidence is tied to controller, scene, and solvability
digests. Missing, stale, duplicate, or wrong-scope evidence is rejected.

Future live promotion claims require separately accepted supervised-live
baseline reports. Fake evidence can never authorize such a claim. A reactive
LiDAR baseline remains deferred.

## Checkpoint And Resume

M13.8 stores versioned curriculum metadata inside the existing M13.7 run state:

- current stage and stage steps;
- sampling level;
- up to three validation-window summaries;
- completed stages;
- best validation evidence by stage;
- sampler states;
- latest promotion decision digest.

Full resume restores model, replay, exact curriculum state, and sampler history.
Replay is preserved across stage promotions. Model-only warm start creates a new
lineage, clears replay and validation history, and starts from Stage 0.

Curriculum digest mismatches remain strict resume errors. Simulator physics and
an in-progress episode are never claimed as restorable.

## Stage 5 Scope

Phase A Stage 5 implements pure deterministic transforms only:

- bounded noise on policy-view LiDAR sectors;
- bounded policy-view sector dropout;
- sampled velocity-response and control-duration scale evidence.

The transforms copy their inputs and do not modify M13.5 safety evidence. No live
settings, physics, sensor, or control duration is changed.

## Offline Commands

```powershell
python scripts/train_m13_8_static_curriculum.py validate
python scripts/train_m13_8_static_curriculum.py preview-stages
python scripts/train_m13_8_static_curriculum.py fake-smoke --timesteps 100
python scripts/train_m13_8_static_curriculum.py inspect-curriculum-state <path>
python scripts/train_m13_8_static_curriculum.py summarize <report.json>
```

The command surface contains no `run`, `pilot`, or hidden live mode. Help,
validation, preview, smoke, inspection, and summarization do not import AirSim or
`cosysairsim`.

## Limitations

- No obstacle-aware policy has been trained.
- No live curriculum stage has been executed.
- M13.6 supervised controller evidence remains pending.
- Stage budgets are limits, not performance forecasts.
- Stage 5 does not validate live robustness.
- Built-in Blocks geometry is not represented by M13.3.
- Physical collision response remains unverified.
- There is no dynamic-obstacle, camera, SLAM, mapping, or real-world UAV claim.
- Final held-out generalization belongs to M13.9.
