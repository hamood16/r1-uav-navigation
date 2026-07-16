# Cleaner UAV

A reinforcement learning and robotics simulation project for autonomous UAV path planning and navigation in dynamic environments.

## Project Goal

The aim of this project is to build a modular UAV autonomy system that can learn to navigate through environments using reinforcement learning algorithms such as DQN, DDPG, TD3, and SAC.

The project will start with a simple 2D Gymnasium environment and later extend into 3D simulation using Unreal Engine and an AirSim-compatible UAV simulator.

## Planned Features

- Custom 2D UAV navigation environment
- Reinforcement learning agents
- DQN baseline
- Dynamic obstacle avoidance
- Evaluation metrics
- Training and evaluation scripts
- Unreal Engine simulation extension
- Clean software engineering structure
- Unit tests and reproducible experiments

## Tech Stack

- Python
- PyTorch
- Gymnasium
- Stable-Baselines3
- NumPy
- Matplotlib
- PyYAML
- pytest
- Ruff
- Unreal Engine later
- AirSim / Colosseum / Cosys-AirSim later

## Results and Documentation

- [M10 dynamic RL results summary](docs/m10_dynamic_rl_results.md)
- [DQN vs TD3 dynamic navigation comparison](docs/results/dqn_vs_td3_dynamic.md)
- [M12 simulator stack decision](docs/m12_simulator_stack_decision.md)

## Project Structure

```text
Cleaner_UAV/
├── configs/
├── docs/
├── results/
├── scripts/
├── src/
└── tests/
