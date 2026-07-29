# M13.7 Long-Run Training Infrastructure

## Status

M13.7 Phase A provides offline-tested software for resumable TD3 runs,
deterministic validation ranking, heartbeat supervision decisions, and throughput
evidence. It does not run Blocks, perform long training, or validate a live
recovery. It does not demonstrate an obstacle-aware policy.

The M12 trainer and its simple `(10,)` fixed-goal environment remain unchanged.
M13.7 is a separate path intended for the M13.5 obstacle environment after
supervised M13.6 reference-course evidence is accepted.

## Checkpoint Bundle

Each safe checkpoint is an immutable directory containing:

```text
model.zip
replay_buffer.pkl
run_state.json
rng_state.json
resolved_config.json
manifest.json
```

The manifest is written last and records every artifact's size and SHA-256
digest. The completed temporary directory is atomically renamed, then
`latest.json` is atomically replaced. Discovery scans and validates complete
manifests instead of trusting a filename or pointer.

A safe checkpoint means the files are internally complete and compatible. It
does not prove that the simulator or UAV is physically safe.

The replay buffer must be the SB3 `ReplayBuffer`, retain timeout-termination
handling, and match the saved capacity and observation/action contracts. Replay
buffers are pickle files and therefore must be treated as trusted local ignored
artifacts. They must not be loaded from an untrusted source.

## Resume Modes

Full resume requires the model, replay buffer, run state, RNG evidence, resolved
configuration, and manifest from one bundle. M13.7 loads TD3 with a fresh
environment, restores replay with `truncate_last_traj=True`, restores practical
RNG sources after model/environment construction, and calls:

```text
learn(additional_timesteps, reset_num_timesteps=False)
```

Only operational settings may change: additional timesteps, device, output
locations, and checkpoint/validation frequency. Policy, optimizer, replay,
reward, sensing, course, observation, and action compatibility must remain
unchanged.

Partial resume has one explicit meaning: model-only warm start. It requires
`--allow-partial-resume`, `--reset-num-timesteps`, and a model checkpoint. It
creates a new run lineage and resets replay, timesteps, validation state, course
sampler state, and future curriculum state.

Neither mode restores an in-progress simulator episode. Resume starts a fresh
episode.

## Random-State Evidence

Phase A captures Python, global NumPy, Torch CPU, available Torch CUDA,
environment/action-space generator, and course-sampler generator states. Missing
sources are reported. Exact simulator physics and CUDA execution cannot be
guaranteed, so checkpoint evidence records
`deterministic_resume_guaranteed: false`.

Configuration snapshots contain repository-relative values and deterministic
digests. They exclude settings paths, local environment variables, credentials,
and machine-specific simulator state.

Curriculum metadata is reserved with `stage_id: none`; M13.7 Phase A does not
implement curriculum behavior.

## Validation

Training episodes never choose the best model. Two deterministic validation
tiers are configured:

- monitoring: empty `0`, easy `1100`, medium `2100`;
- promotion: empty `0`, held-out reverse `9100`, held-out elevated `10100`.

Promotion ranks complete sets by success rate, lower collision rate, safe
landing/cleanup rate, successful-path efficiency, lower final distance, then
mean return. Failed or inconclusive landing/cleanup evidence is a hard gate:
the report is retained but cannot promote a model.

Only strict promotion-rank improvement writes
`best_validation_model.zip`. Monitoring results and equal ranks never overwrite
it. Phase A uses injected fake evaluators. A live evaluator is deferred because
changing simulator scenes inside an active training callback would violate the
established ownership lifecycle.

## Supervisor

Workers atomically report a heartbeat after real progress, including worker/run
identity, sequence, phase, timestep, latest course, latest safe bundle, cleanup
status, and error evidence. An independent timer thread is not used; a blocked
RPC therefore stops progress evidence.

The Phase A supervisor can decide that a worker is healthy, inside checkpoint
grace, stale, missing a safe bundle, blocked by uncertain simulator state, ready
for a future restart, or complete. It only spawns a short fake worker that exits
normally. It does not terminate processes.

A future live restart requires confirmed worker termination and a read-only
UAV/simulator safety survey. Process exit alone is not safe-state evidence.
Broad simulator reset remains prohibited.

## Throughput Evidence

The offline benchmark schema records reset, action-step, LiDAR, episode,
checkpoint, logging, and cleanup evidence together with active and end-to-end
steps per second. Comparisons must change exactly one declared knob.

Phase A fixes simulator clock scale at `1.0`, does not edit settings, and makes no
Colosseum throughput claim. Any cleanup failure prevents benchmark acceptance.

## Offline Commands

```powershell
.\.venv\Scripts\python.exe scripts\train_colosseum_td3_long_run.py validate
.\.venv\Scripts\python.exe scripts\train_colosseum_td3_long_run.py inspect-resume --resume-latest RUN_DIR
.\.venv\Scripts\python.exe scripts\train_colosseum_td3_long_run.py fake-smoke --additional-timesteps 20
.\.venv\Scripts\python.exe scripts\train_colosseum_td3_long_run.py supervisor-smoke

.\.venv\Scripts\python.exe scripts\benchmark_colosseum_throughput.py validate
.\.venv\Scripts\python.exe scripts\benchmark_colosseum_throughput.py fake-smoke
.\.venv\Scripts\python.exe scripts\benchmark_colosseum_throughput.py summarize REPORT...
```

These commands are simulator-independent. Phase A intentionally provides no live
training or live benchmark command.

## Deferred Work And Limitations

- Supervised M13.6 evidence and separate authorization are required before live
  M13.7 work.
- No long run, live recovery, live validation callback, SAC implementation, or
  obstacle-aware policy result exists.
- `Continuous3DObstacleUAVEnv` remains an architectural option only. Its intended
  future role is surrogate pretraining followed by Colosseum fine-tuning and
  final Colosseum evaluation.
- No camera, SLAM, dynamic obstacle, live occupancy mapping, simulator speedup,
  automatic live-process termination, or real-world UAV claim is included.
