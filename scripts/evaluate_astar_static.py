"""Evaluate A* path planning on static GridUAVEnv layouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from r1_uav_nav.evaluation import (
    EvaluationSummary,
    calculate_path_length,
    plot_failure_rate_bar,
    plot_path_length_curve,
    plot_success_rate_bar,
    plot_trajectory,
)
from r1_uav_nav.planners import find_astar_path
from r1_uav_nav.utils import create_grid_uav_env_from_config

ENV_CONFIG_PATH = Path("configs/env/grid_2d_static_full.yaml")
PLOTS_DIR = Path("results/plots/planners/static_astar")
NUM_EVAL_EPISODES = 100
EVAL_SEED = 42
TrajectoryData = dict[str, Any]


def select_shortest_successful_trajectory(
    trajectories: list[TrajectoryData],
) -> TrajectoryData:
    """Select the shortest successful trajectory, or the first fallback."""
    successful_trajectories = _successful_trajectories(trajectories)
    if not successful_trajectories:
        return trajectories[0]
    return min(successful_trajectories, key=lambda trajectory: trajectory["steps"])


def select_mean_length_successful_trajectory(
    trajectories: list[TrajectoryData],
) -> TrajectoryData:
    """Select the successful trajectory closest to the successful mean length."""
    successful_trajectories = _successful_trajectories(trajectories)
    if not successful_trajectories:
        return trajectories[0]

    mean_path_length = sum(
        trajectory["path_length"] for trajectory in successful_trajectories
    ) / len(successful_trajectories)
    return min(
        successful_trajectories,
        key=lambda trajectory: abs(trajectory["path_length"] - mean_path_length),
    )


def select_longest_successful_trajectory(
    trajectories: list[TrajectoryData],
) -> TrajectoryData:
    """Select the longest successful trajectory, or the first fallback."""
    successful_trajectories = _successful_trajectories(trajectories)
    if not successful_trajectories:
        return trajectories[0]
    return max(
        successful_trajectories, key=lambda trajectory: trajectory["path_length"]
    )


def main() -> None:
    """Run deterministic static A* evaluation."""
    env = create_grid_uav_env_from_config(ENV_CONFIG_PATH)

    trajectory_data: list[TrajectoryData] = []
    path_lengths_by_episode: list[float] = []
    successful_steps: list[int] = []
    successful_path_lengths: list[float] = []

    for episode_index in range(NUM_EVAL_EPISODES):
        env.reset(seed=EVAL_SEED + episode_index)
        start_position = env.uav_position
        goal_position = env.goal_position
        obstacles = set(env.obstacles)
        grid_size = env.grid_size

        planned_path = find_astar_path(
            start=start_position,
            goal=goal_position,
            obstacles=obstacles,
            grid_size=grid_size,
        )
        success = planned_path is not None
        path = planned_path if planned_path is not None else [start_position]
        steps = len(path) - 1 if success else 0
        path_length = calculate_path_length(path) if success else 0.0

        if success:
            successful_steps.append(steps)
            successful_path_lengths.append(path_length)

        path_lengths_by_episode.append(path_length)
        trajectory_data.append(
            {
                "path": list(path),
                "success": success,
                "start_position": start_position,
                "goal_position": goal_position,
                "obstacles": obstacles,
                "grid_size": grid_size,
                "steps": steps,
                "path_length": path_length,
            }
        )

    num_episodes = len(trajectory_data)
    success_count = sum(trajectory["success"] for trajectory in trajectory_data)
    success_rate = success_count / num_episodes
    failure_rate = 1.0 - success_rate
    average_successful_steps = (
        sum(successful_steps) / len(successful_steps) if successful_steps else 0.0
    )
    average_successful_path_length = (
        sum(successful_path_lengths) / len(successful_path_lengths)
        if successful_path_lengths
        else 0.0
    )

    selected_shortest = select_shortest_successful_trajectory(trajectory_data)
    selected_mean_length = select_mean_length_successful_trajectory(trajectory_data)
    selected_longest = select_longest_successful_trajectory(trajectory_data)
    summary = EvaluationSummary(
        num_episodes=num_episodes,
        success_rate=success_rate,
        collision_rate=0.0,
        average_reward=0.0,
        average_steps=average_successful_steps,
        average_path_length=average_successful_path_length,
    )

    plot_paths = [
        _plot_static_astar_trajectory(
            selected_shortest,
            PLOTS_DIR / "trajectory.png",
        ),
        _plot_static_astar_trajectory(
            selected_mean_length,
            PLOTS_DIR / "trajectory_mean_length.png",
        ),
        _plot_static_astar_trajectory(
            selected_longest,
            PLOTS_DIR / "trajectory_longest.png",
        ),
        plot_path_length_curve(
            path_lengths=path_lengths_by_episode,
            output_path=PLOTS_DIR / "path_length_curve.png",
        ),
        plot_success_rate_bar(
            summary=summary,
            output_path=PLOTS_DIR / "success_rate.png",
        ),
        plot_failure_rate_bar(
            failure_rate=failure_rate,
            output_path=PLOTS_DIR / "failure_rate.png",
        ),
    ]

    print("A* static evaluation complete.")
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Failure rate: {failure_rate:.2f}")
    print(f"Average successful path length: {average_successful_path_length:.2f}")
    print(f"Average successful steps: {average_successful_steps:.2f}")
    print("Saved plots:")
    for plot_path in plot_paths:
        print(f"- {plot_path}")


def _successful_trajectories(
    trajectories: list[TrajectoryData],
) -> list[TrajectoryData]:
    if not trajectories:
        raise ValueError("Cannot select from empty trajectory data")
    return [trajectory for trajectory in trajectories if trajectory["success"]]


def _plot_static_astar_trajectory(
    trajectory: TrajectoryData,
    output_path: Path,
) -> Path:
    return plot_trajectory(
        trajectory_positions=trajectory["path"],
        obstacles=trajectory["obstacles"],
        start_position=trajectory["start_position"],
        goal_position=trajectory["goal_position"],
        grid_size=trajectory["grid_size"],
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
