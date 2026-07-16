# DQN vs TD3 Dynamic Navigation Comparison

## Plain-English Summary

This comparison shows how the tuned dynamic DQN and tuned continuous TD3 models performed on dynamic obstacle navigation tasks. TD3 achieved stronger performance in the continuous-control environment, while tuned DQN still showed strong results in the discrete grid environment.

This is a project-level comparison, not a strict algorithmic benchmark. The two models operate in related but different environments with different action spaces.

## What Each Algorithm Controls

Dynamic DQN controls a UAV in `DynamicGridUAVEnv`. It chooses from discrete movement actions: move up, down, left, right, or hover. DQN fits this setup because its action space is a small set of discrete choices.

TD3 controls a UAV in `ContinuousDynamicUAVEnv`. It outputs continuous velocity commands in the x and y directions. TD3 fits this setup because it is designed for continuous-control tasks where actions are real-valued rather than chosen from a short list.

## Tuned Repeated Evaluation Results

| Method | Control type | Environment | Success rate | Collision rate | Timeout rate | Average reward | Average steps | Average path length |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Tuned DQN | Discrete actions | `DynamicGridUAVEnv` | 0.8920 +/- 0.0232 | 0.1080 +/- 0.0232 | 0.0000 +/- 0.0000 | 9.0604 +/- 0.4605 | 7.1080 +/- 0.1888 | 6.9740 +/- 0.1648 |
| Tuned TD3 | Continuous velocity | `ContinuousDynamicUAVEnv` | 0.9760 +/- 0.0162 | 0.0240 +/- 0.0162 | 0.0000 +/- 0.0000 | 10.7858 +/- 0.3273 | 5.9600 +/- 0.1071 | 6.1078 +/- 0.2168 |

## How To Read The Metrics

- Success rate is the fraction of episodes where the UAV reached the goal.
- Collision rate is the fraction of episodes that ended because the UAV hit a moving obstacle.
- Timeout rate is the fraction of episodes that reached the maximum step limit.
- Average reward combines goal rewards, collision penalties, step costs, and progress shaping.
- Average steps measures how long each episode took.
- Average path length measures how far the UAV travelled.

Lower collision, timeout, steps, and path length values are generally better when success remains high. Higher success rate and reward are generally better.

## Interpretation

Tuned TD3 achieved stronger performance in the continuous-control environment: higher repeated success, lower repeated collision, higher reward, fewer steps, and shorter average path length.

Tuned DQN performed well in the discrete dynamic grid world, reaching 0.8920 repeated success with 0.1080 repeated collision. This is a strong result for a simpler discrete-action policy.

The comparison is useful because it shows how the project handles both discrete and continuous UAV control. It should not be read as proof that TD3 is universally superior to DQN.

## Caveat

DQN and TD3 are not directly apples-to-apples because their action spaces and environments differ. DQN uses a discrete grid environment, while TD3 uses a continuous dynamic environment with continuous velocity commands.

## Reproduction

Generate the comparison plots:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dqn_vs_td3_comparison_plots.py
```

The script uses fixed M10 tuned repeated evaluation metrics. It does not train models, evaluate models, load model files, or read ignored JSON reports.

## Generated Plot Paths

- `results/plots/comparison/dqn_vs_td3/success_rate_comparison.png`
- `results/plots/comparison/dqn_vs_td3/collision_rate_comparison.png`
- `results/plots/comparison/dqn_vs_td3/average_reward_comparison.png`
- `results/plots/comparison/dqn_vs_td3/average_steps_comparison.png`
- `results/plots/comparison/dqn_vs_td3/path_length_comparison.png`
- `results/plots/comparison/dqn_vs_td3/timeout_rate_comparison.png`
