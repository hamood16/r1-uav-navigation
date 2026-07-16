# M10 Dynamic RL Results Summary

## Purpose

M10 completed the dynamic reinforcement-learning reporting pass for this project. It added easy, medium, and hard dynamic environment configs, repeated evaluation tooling, tuned dynamic DQN and TD3 training configs, and this final results summary.

This report focuses on the dynamic RL baselines and tuned models. It records the single-seed evaluation results, repeated tuned evaluation evidence, reproduction commands, and caveats needed to interpret the results responsibly.

## Headline Findings

- Tuned DQN improved from 0.83 to 0.87 single-run success and achieved 0.8920 ± 0.0232 repeated success.
- Tuned DQN reduced collision rate from 0.17 baseline to 0.13 single-run and 0.1080 ± 0.0232 repeated collision.
- Tuned TD3 maintained strong repeated performance with 0.9760 ± 0.0162 repeated success and 0.0240 ± 0.0162 repeated collision.
- Repeated evaluation is stronger evidence than a single seed-42 evaluation.

## Evaluation Setup

- Dynamic DQN uses `DynamicGridUAVEnv` with discrete actions.
- TD3 uses `ContinuousDynamicUAVEnv` with continuous velocity control.
- Both tasks involve navigation with dynamic obstacles.
- Single evaluations use 100 episodes with seed `42`.
- Repeated evaluations use 5 repeats with 100 episodes per repeat.
- Repeated seed blocks are `42`, `1042`, `2042`, `3042`, and `4042`.
- Repeated evaluation summaries use population standard deviation.

## Configs and Model Paths Used

| Method | Variant | Environment config | Training config | Model path |
|---|---|---|---|---|
| Dynamic DQN | Baseline | `configs/env/dynamic_grid_2d.yaml` | `configs/training/dqn_dynamic_full.yaml` | `results/trained_models/dqn_dynamic_full.zip` |
| Dynamic DQN | Tuned | `configs/env/dynamic_grid_2d_medium.yaml` | `configs/training/dqn_dynamic_tuned.yaml` | `results/trained_models/dqn_dynamic_tuned.zip` |
| TD3 continuous dynamic | Baseline | `configs/env/continuous_dynamic_2d.yaml` | `configs/training/td3_continuous_dynamic_full.yaml` | `results/trained_models/td3_continuous_dynamic_full.zip` |
| TD3 continuous dynamic | Tuned | `configs/env/continuous_dynamic_2d_medium.yaml` | `configs/training/td3_continuous_dynamic_tuned.yaml` | `results/trained_models/td3_continuous_dynamic_tuned.zip` |

## Baseline vs Tuned Single-Evaluation Results

| Method | Variant | Success rate | Collision rate | Timeout rate | Average reward | Average steps | Average path length |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dynamic DQN | Baseline | 0.83 | 0.17 | 0.00 | 7.91 | 7.40 | 7.38 |
| Dynamic DQN | Tuned | 0.87 | 0.13 | 0.00 | 8.63 | 6.79 | 6.72 |
| TD3 continuous dynamic | Baseline | 0.98 | 0.02 | 0.00 | 10.88 | 5.66 | 6.09 |
| TD3 continuous dynamic | Tuned | 0.96 | 0.04 | 0.00 | 10.53 | 5.83 | 6.12 |

## Tuned Repeated Evaluation Results

| Method | Repeats | Episodes per repeat | Success rate | Collision rate | Timeout rate | Average reward | Average steps | Average path length | Report path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dynamic DQN tuned | 5 | 100 | 0.8920 ± 0.0232 | 0.1080 ± 0.0232 | 0.0000 ± 0.0000 | 9.0604 ± 0.4605 | 7.1080 ± 0.1888 | 6.9740 ± 0.1648 | `results/reports/m10/dqn_dynamic_tuned_repeated_eval.json` |
| TD3 continuous dynamic tuned | 5 | 100 | 0.9760 ± 0.0162 | 0.0240 ± 0.0162 | 0.0000 ± 0.0000 | 10.7858 ± 0.3273 | 5.9600 ± 0.1071 | 6.1078 ± 0.2168 | `results/reports/m10/td3_continuous_dynamic_tuned_repeated_eval.json` |

## Interpretation

Tuned DQN improved over the dynamic DQN baseline. In the single seed-42 evaluation, success rate increased from 0.83 to 0.87, while collision rate dropped from 0.17 to 0.13. The repeated evaluation strengthens that result: tuned DQN reached 0.8920 mean success and 0.1080 mean collision rate across five 100-episode repeats.

TD3 was already very strong at baseline. The tuned TD3 single evaluation is slightly below the earlier single baseline, so it should not be described as a clear single-run improvement. However, repeated evaluation shows that tuned TD3 maintained strong performance, with 0.9760 mean success and 0.0240 mean collision rate.

Repeated evaluation is stronger evidence than a single seed-42 evaluation because it measures stability across multiple deterministic seed blocks.

## Caveats

- DQN and TD3 results are not directly apples-to-apples because DQN uses a discrete grid environment and TD3 uses a continuous dynamic environment with continuous velocity control.
- Tuned model results were generated locally and saved under ignored `results/` paths.
- Generated models, plots, and JSON reports are not committed.
- More extensive hyperparameter searches would be needed for rigorous optimisation.
- Population standard deviation is used for repeated evaluation summaries.

## Reproduction Commands

Train tuned DQN:

```powershell
.\.venv\Scripts\python.exe scripts\train_dynamic_dqn.py --env-config configs/env/dynamic_grid_2d_medium.yaml --training-config configs/training/dqn_dynamic_tuned.yaml
```

Evaluate tuned DQN once:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_dynamic_dqn.py --env-config configs/env/dynamic_grid_2d_medium.yaml --training-config configs/training/dqn_dynamic_tuned.yaml --plots-dir results/plots/dynamic_tuned
```

Repeated evaluate tuned DQN:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_dynamic_dqn_repeated.py --env-config configs/env/dynamic_grid_2d_medium.yaml --training-config configs/training/dqn_dynamic_tuned.yaml --output-path results/reports/m10/dqn_dynamic_tuned_repeated_eval.json
```

Train tuned TD3:

```powershell
.\.venv\Scripts\python.exe scripts\train_td3_continuous_dynamic.py --env-config configs/env/continuous_dynamic_2d_medium.yaml --training-config configs/training/td3_continuous_dynamic_tuned.yaml
```

Evaluate tuned TD3 once:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_td3_continuous_dynamic.py --env-config configs/env/continuous_dynamic_2d_medium.yaml --training-config configs/training/td3_continuous_dynamic_tuned.yaml --plots-dir results/plots/td3_continuous_dynamic_tuned
```

Repeated evaluate tuned TD3:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_td3_continuous_dynamic_repeated.py --env-config configs/env/continuous_dynamic_2d_medium.yaml --training-config configs/training/td3_continuous_dynamic_tuned.yaml --output-path results/reports/m10/td3_continuous_dynamic_tuned_repeated_eval.json
```

## Artifact Paths

- `results/plots/dynamic_tuned/`
- `results/plots/td3_continuous_dynamic_tuned/`
- `results/reports/m10/dqn_dynamic_tuned_repeated_eval.json`
- `results/reports/m10/td3_continuous_dynamic_tuned_repeated_eval.json`
