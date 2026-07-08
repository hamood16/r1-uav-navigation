"""Evaluate the full static DQN baseline on the GridUAVEnv environment."""

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

ENV_CONFIG_PATH = Path("configs/env/grid_2d_static_full.yaml")
TRAINING_CONFIG_PATH = Path("configs/training/dqn_static_full.yaml")
PLOTS_DIR = Path("results/plots/static")
NUM_EVAL_EPISODES = 100
EVAL_SEED = 42
TRAINING_COMMAND = "python scripts/train_static_dqn.py"


def main() -> None:
    """Run deterministic evaluation for the full static DQN baseline."""
    env = create_grid_uav_env_from_config(ENV_CONFIG_PATH)
    training_config = load_config(TRAINING_CONFIG_PATH)
    model_output_path = Path(training_config["model_output_path"])
    if not model_output_path.exists():
        raise FileNotFoundError(
            f"Static DQN model not found at {model_output_path}. "
            f"Run `{TRAINING_COMMAND}` first."
        )

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
        path_length = calculate_path_length(positions)
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
                "path_length": path_length,
            }
        )
        episode_results.append(
            EpisodeResult(
                total_reward=total_reward,
                steps=steps,
                success=success,
                collision=collision,
                path_length=path_length,
            )
        )

    summary = summarise_episode_results(episode_results)
    successful_trajectories = [
        episode_trajectory
        for episode_trajectory in trajectory_data
        if episode_trajectory["success"]
    ]
    selected_fastest_trajectory = (
        min(successful_trajectories, key=lambda trajectory: trajectory["steps"])
        if successful_trajectories
        else trajectory_data[0]
    )
    mean_length_candidates = (
        successful_trajectories if successful_trajectories else trajectory_data
    )
    mean_path_length = sum(
        trajectory["path_length"] for trajectory in mean_length_candidates
    ) / len(mean_length_candidates)
    selected_mean_length_trajectory = min(
        mean_length_candidates,
        key=lambda trajectory: abs(trajectory["path_length"] - mean_path_length),
    )
    selected_longest_trajectory = max(
        mean_length_candidates,
        key=lambda trajectory: trajectory["path_length"],
    )
    plot_paths = [
        plot_trajectory(
            trajectory_positions=selected_fastest_trajectory["positions"],
            obstacles=selected_fastest_trajectory["obstacles"],
            start_position=selected_fastest_trajectory["start_position"],
            goal_position=selected_fastest_trajectory["goal_position"],
            grid_size=selected_fastest_trajectory["grid_size"],
            output_path=PLOTS_DIR / "trajectory.png",
        ),
        plot_trajectory(
            trajectory_positions=selected_mean_length_trajectory["positions"],
            obstacles=selected_mean_length_trajectory["obstacles"],
            start_position=selected_mean_length_trajectory["start_position"],
            goal_position=selected_mean_length_trajectory["goal_position"],
            grid_size=selected_mean_length_trajectory["grid_size"],
            output_path=PLOTS_DIR / "trajectory_mean_length.png",
        ),
        plot_trajectory(
            trajectory_positions=selected_longest_trajectory["positions"],
            obstacles=selected_longest_trajectory["obstacles"],
            start_position=selected_longest_trajectory["start_position"],
            goal_position=selected_longest_trajectory["goal_position"],
            grid_size=selected_longest_trajectory["grid_size"],
            output_path=PLOTS_DIR / "trajectory_longest.png",
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

    print("Static DQN baseline evaluation complete.")
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
