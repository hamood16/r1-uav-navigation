"""Evaluate a trained DQN agent on the GridUAVEnv environment."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN

from r1_uav_nav.evaluation import (
    EpisodeResult,
    calculate_path_length,
    plot_collision_rate_bar,
    plot_reward_curve,
    plot_success_rate_bar,
    plot_trajectory,
    summarise_episode_results,
)
from r1_uav_nav.utils import create_grid_uav_env_from_config, load_config

ENV_CONFIG_PATH = Path("configs/env/grid_2d.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/dqn_debug.yaml")
PLOTS_DIR = Path("results/plots")
NUM_EVAL_EPISODES = 20
EVAL_SEED = 42


def main() -> None:
    """Run deterministic evaluation for a trained DQN model."""
    env = create_grid_uav_env_from_config(ENV_CONFIG_PATH)
    training_config = load_config(TRAINING_CONFIG_PATH)
    model_output_path = Path(training_config["model_output_path"])
    model = DQN.load(model_output_path, env=env)

    episode_results: list[EpisodeResult] = []
    episode_rewards: list[float] = []
    trajectory_data: list[dict] = []
    for episode_index in range(NUM_EVAL_EPISODES):
        observation, _ = env.reset(seed=EVAL_SEED + episode_index)
        positions = [env.uav_position]
        start_position = env.uav_position
        goal_position = env.goal_position
        obstacles = set(env.obstacles)
        grid_size = env.grid_size
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            positions.append(env.uav_position)

        success = terminated and env.uav_position == env.goal_position
        collision = terminated and not success
        episode_rewards.append(total_reward)
        trajectory_data.append(
            {
                "positions": list(positions),
                "success": success,
                "start_position": start_position,
                "goal_position": goal_position,
                "obstacles": obstacles,
                "grid_size": grid_size,
                "steps": steps,
            }
        )
        episode_results.append(
            EpisodeResult(
                total_reward=total_reward,
                steps=steps,
                success=success,
                collision=collision,
                path_length=calculate_path_length(positions),
            )
        )

    summary = summarise_episode_results(episode_results)
    successful_trajectories = [
        episode_trajectory
        for episode_trajectory in trajectory_data
        if episode_trajectory["success"]
    ]
    selected_trajectory = (
        min(successful_trajectories, key=lambda trajectory: trajectory["steps"])
        if successful_trajectories
        else trajectory_data[0]
    )
    plot_paths = [
        plot_trajectory(
            trajectory_positions=selected_trajectory["positions"],
            obstacles=selected_trajectory["obstacles"],
            start_position=selected_trajectory["start_position"],
            goal_position=selected_trajectory["goal_position"],
            grid_size=selected_trajectory["grid_size"],
            output_path=PLOTS_DIR / "trajectory.png",
        ),
        plot_reward_curve(
            episode_rewards=episode_rewards,
            output_path=PLOTS_DIR / "reward_curve.png",
        ),
        plot_success_rate_bar(
            summary=summary,
            output_path=PLOTS_DIR / "success_rate.png",
        ),
        plot_collision_rate_bar(
            summary=summary,
            output_path=PLOTS_DIR / "collision_rate.png",
        ),
    ]

    print("DQN evaluation complete.")
    print(f"Episodes: {summary.num_episodes}")
    print(f"Success rate: {summary.success_rate:.2f}")
    print(f"Collision rate: {summary.collision_rate:.2f}")
    print(f"Average reward: {summary.average_reward:.2f}")
    print(f"Average steps: {summary.average_steps:.2f}")
    print(f"Average path length: {summary.average_path_length:.2f}")
    print(f"Model path: {model_output_path}")
    print("Saved plots:")
    for plot_path in plot_paths:
        print(f"- {plot_path}")


if __name__ == "__main__":
    main()
