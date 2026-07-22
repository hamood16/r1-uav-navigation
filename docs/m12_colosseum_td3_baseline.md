# M12 Colosseum TD3 Baseline

## Purpose

M12.5 adds the first trainable and evaluable reinforcement-learning pipeline for
the live Colosseum Gymnasium wrapper. It uses TD3 because `ColosseumUAVEnv` exposes
a continuous normalized 3D action space for world-frame NED velocity control.

This milestone is a fixed-goal pipeline baseline. It is not a claim of strong 3D
autonomy after a short run.

## Baseline Task

The default training goal is:

```text
goal_offset = (3.0, 0.0, 0.0)
```

The goal is horizontal and at the same altitude as the measured airborne anchor.
This simple obstacle-free target is intentional:

- it validates the live RL training pipeline;
- it isolates simulator, reset, replay, checkpoint, and cleanup issues;
- it avoids mixing environment complexity with algorithm debugging;
- random-goal 3D navigation belongs in a later milestone.

## Environment And Algorithm

The environment is `ColosseumUAVEnv`:

- observation shape: `(10,)`;
- action shape: `(3,)`;
- action range: `[-1, 1]`;
- NED convention: positive z velocity moves downward, negative z velocity moves up.

The TD3 implementation is the existing Stable-Baselines3 path exposed by
`create_td3_model`. Installed Stable-Baselines3 version inspected during planning:
`2.9.0`.

M12.5 uses one shared `learning_rate`. Separate actor and critic learning rates are
future work only if needed.

## Safer Exploration Defaults

The live M12.4 random-action tests showed that unrestricted vertical exploration can
quickly trigger out-of-bounds transitions. M12.5 keeps the policy action space fully
three-dimensional but reduces physical vertical movement:

```text
max_vertical_velocity = 0.2 m/s
action_noise_std = [0.15, 0.15, 0.05]
```

SB3 random warm-up still samples the normalized action space, so reducing physical
vertical velocity also limits warm-up vertical drift without freezing or removing
the z action.

## Smoke Training Defaults

The default config is designed to prove that real replay and TD3 updates happen:

```text
total_timesteps = 100
learning_starts = 20
batch_size = 16
checkpoint_interval = 50
```

Smoke acceptance should confirm:

- replay buffer growth;
- at least one critic update;
- at least one delayed actor update;
- checkpoint saving;
- metric saving.

The lower-bound runtime is roughly `total_timesteps * control_duration`, plus reset
overhead. At `0.5` seconds per step, 100 steps is at least about 50 seconds before
resets.

## Transition Semantics

Gymnasium returns separate `terminated` and `truncated` flags. Stable-Baselines3
converts these through its VecEnv layer and records `TimeLimit.truncated` in replay
infos. Its replay buffer uses `handle_timeout_termination=True`, so time-limit
truncations remain eligible for value bootstrapping while true terminal transitions
remain terminal.

M12.5 includes a CPU-only SB3 integration test that verifies this behavior on the
installed stack.

## Checkpoints And Metrics

Generated outputs are ignored by Git:

- `results/trained_models/colosseum_td3_baseline/`
- `results/logs/colosseum_td3_baseline/`
- `results/reports/m12/`

Training saves:

- periodic checkpoints under `checkpoints/`;
- `final.zip`;
- optional `best_training_episode.zip`.

The best checkpoint is only the best observed training-episode checkpoint, using:

```text
(success, -final_distance, episode_return)
```

It is not a deterministic evaluation result.

Metrics include episode return, length, final/min distance, success, collision,
out-of-bounds, ground-clearance violations, truncation, termination reason, replay
buffer size, update count, elapsed time, checkpoint errors, and cleanup status.

Evaluation reports are policy-specific so random, scripted-forward, and TD3 runs do
not overwrite each other:

- `colosseum_td3_baseline_random_evaluation_summary.json`
- `colosseum_td3_baseline_scripted-forward_evaluation_summary.json`
- `colosseum_td3_baseline_td3_evaluation_summary.json`

## Live Commands

Run a random baseline:

```powershell
python scripts\evaluate_colosseum_td3.py --policy random --episodes 3
```

Run the scripted-forward reference:

```powershell
python scripts\evaluate_colosseum_td3.py --policy scripted-forward --episodes 3
```

Run 100-step smoke training:

```powershell
python scripts\train_colosseum_td3.py --total-timesteps 100 --learning-starts 20 --batch-size 16 --checkpoint-interval 50
```

Evaluate the smoke checkpoint:

```powershell
python scripts\evaluate_colosseum_td3.py --policy td3 --checkpoint results/trained_models/colosseum_td3_baseline/final.zip --episodes 3
```

Run a bounded Stage B baseline only after the smoke checkpoint reloads:

```powershell
python scripts\train_colosseum_td3.py --total-timesteps 2000 --learning-starts 100 --checkpoint-interval 250
```

## Safety

Training and evaluation scripts always call `env.close()` in a `finally` block and
return nonzero if safety-critical cleanup fails during reset, automatic episode
reset, interruption handling, or final close.

Training cleanup still runs if periodic, best, interrupt, or final checkpoint saving
fails. Checkpoint, cleanup, and metrics-save errors are reported separately so a
later successful cleanup or metrics write cannot hide an earlier safety-critical
failure.

If the legacy RPC client reports `IOLoop is already running` after interruption,
restart Blocks and use a fresh Python process.

## Results

Live training and evaluation were completed in Colosseum Blocks using an NVIDIA
GeForce RTX 4050 Laptop GPU with PyTorch CUDA enabled.

| Policy / checkpoint | Training steps | Evaluation episodes | Success rate | Mean return | Mean final distance |
|---|---:|---:|---:|---:|---:|
| Random policy | 0 | 3 | 0% | -1.260 | 3.081 m |
| Scripted-forward reference | 0 | 3 | 100% | 12.422 | 0.368 m |
| TD3 smoke checkpoint | 100 | 3 | 0% | -6.254 | 3.347 m |
| TD3 Stage A checkpoint | 500 | 5 | 0% | -8.366 | 5.917 m |
| TD3 Stage B checkpoint | 2,000 | 5 | 100% | 12.305 | 0.459 m |

### Observed outcome

The 100-step smoke run successfully validated the complete training pipeline,
including replay-buffer collection, TD3 updates, periodic checkpointing, final
checkpoint saving, checkpoint loading, deterministic evaluation, and simulator
cleanup. It was not long enough to learn a successful policy.

The 500-step Stage A model also failed to reach the goal during deterministic
evaluation and performed worse than the random baseline. This shows that early TD3
training was unstable and had not yet converged.

After 2,000 training steps, the Stage B TD3 policy achieved a 100% success rate
across five deterministic evaluation episodes. Its mean return of 12.305 and mean
final distance of 0.459 m were close to the scripted-forward reference, which
achieved a mean return of 12.422 and mean final distance of 0.368 m.

These results demonstrate that the TD3 agent learned the fixed, obstacle-free
forward-goal task. They do not yet demonstrate general navigation, random-goal
generalisation, obstacle avoidance, or perception-based autonomy.

### Safety validation

After all training and evaluation runs:

- API control was disabled;
- the UAV was landed;
- the UAV velocity was zero;
- no safety-critical cleanup failure was reported.

The final measured simulator state was:

```text
API control: False
Landed state: 0
Velocity: x=0.0, y=0.0, z=0.0
```

## Limitations

- Short smoke training is for pipeline validation, not policy quality.
- The fixed-goal task does not demonstrate general random-goal navigation.
- No camera, LiDAR, obstacle avoidance, curriculum, SAC, PPO, sweeps, or multi-drone
  logic is included.
- `moveByVelocityAsync().join()` still has no safe timeout in the validated legacy
  RPC stack.
