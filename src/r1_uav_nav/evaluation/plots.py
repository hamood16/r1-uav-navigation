"""Plotting helpers for evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from r1_uav_nav.evaluation.metrics import EvaluationSummary, Position


def plot_trajectory(
    trajectory_positions: Sequence[Position],
    obstacles: Sequence[Position] | set[Position],
    start_position: Position,
    goal_position: Position,
    grid_size: int,
    output_path: str | Path,
) -> Path:
    """Plot one UAV trajectory through the grid world."""
    saved_path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 7))

    axis.set_title("UAV trajectory")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xlim(-0.5, grid_size - 0.5)
    axis.set_ylim(-0.5, grid_size - 0.5)
    axis.set_xticks(range(grid_size))
    axis.set_yticks(range(grid_size))
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axis.set_aspect("equal", adjustable="box")

    if obstacles:
        obstacle_x, obstacle_y = zip(*obstacles, strict=False)
        axis.scatter(
            obstacle_x,
            obstacle_y,
            marker="s",
            color="black",
            label="Obstacles",
        )

    if trajectory_positions:
        path_x, path_y = zip(*trajectory_positions, strict=False)
        axis.plot(path_x, path_y, color="tab:blue", marker="o", label="UAV path")

    axis.scatter(
        [start_position[0]],
        [start_position[1]],
        marker="o",
        color="tab:green",
        s=120,
        label="Start",
    )
    axis.scatter(
        [goal_position[0]],
        [goal_position[1]],
        marker="*",
        color="tab:red",
        s=180,
        label="Goal",
    )
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(saved_path)
    plt.close(figure)
    return saved_path


def plot_reward_curve(
    episode_rewards: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Plot total reward for each evaluation episode."""
    saved_path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(8, 5))
    episode_numbers = range(1, len(episode_rewards) + 1)

    axis.plot(episode_numbers, episode_rewards, marker="o", color="tab:blue")
    axis.set_title("Evaluation reward curve")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Total reward")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    figure.tight_layout()
    figure.savefig(saved_path)
    plt.close(figure)
    return saved_path


def plot_path_length_curve(
    path_lengths: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Plot planned path length for each evaluation episode."""
    saved_path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(8, 5))
    episode_numbers = range(1, len(path_lengths) + 1)

    axis.plot(episode_numbers, path_lengths, marker="o", color="tab:purple")
    axis.set_title("A* path length by episode, failed plans shown as 0")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Path length")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    figure.tight_layout()
    figure.savefig(saved_path)
    plt.close(figure)
    return saved_path


def plot_metric_comparison(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> Path:
    """Plot a simple bar comparison for named metrics."""
    saved_path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 5))

    axis.bar(labels, values, color=["tab:purple", "tab:blue"][: len(labels)])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.2f}", ha="center", va="bottom")

    figure.tight_layout()
    figure.savefig(saved_path)
    plt.close(figure)
    return saved_path


def plot_trajectory_overlay(
    astar_positions: Sequence[Position],
    dqn_positions: Sequence[Position],
    obstacles: Sequence[Position] | set[Position],
    start_position: Position,
    goal_position: Position,
    grid_size: int,
    output_path: str | Path,
) -> Path:
    """Plot A* and DQN trajectories on the same static grid layout."""
    saved_path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 7))

    axis.set_title("Static A* vs DQN trajectory overlay")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xlim(-0.5, grid_size - 0.5)
    axis.set_ylim(-0.5, grid_size - 0.5)
    axis.set_xticks(range(grid_size))
    axis.set_yticks(range(grid_size))
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axis.set_aspect("equal", adjustable="box")

    if obstacles:
        obstacle_x, obstacle_y = zip(*obstacles, strict=False)
        axis.scatter(
            obstacle_x,
            obstacle_y,
            marker="s",
            color="black",
            label="Obstacles",
        )

    if astar_positions:
        astar_x, astar_y = zip(*astar_positions, strict=False)
        axis.plot(
            astar_x,
            astar_y,
            color="tab:purple",
            marker="o",
            label="A* path",
        )

    if dqn_positions:
        dqn_x, dqn_y = zip(*dqn_positions, strict=False)
        axis.plot(
            dqn_x,
            dqn_y,
            color="tab:blue",
            marker="x",
            linestyle="--",
            label="DQN path",
        )

    axis.scatter(
        [start_position[0]],
        [start_position[1]],
        marker="o",
        color="tab:green",
        s=120,
        label="Start",
    )
    axis.scatter(
        [goal_position[0]],
        [goal_position[1]],
        marker="*",
        color="tab:red",
        s=180,
        label="Goal",
    )
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(saved_path)
    plt.close(figure)
    return saved_path


def plot_success_rate_bar(
    summary: EvaluationSummary,
    output_path: str | Path,
) -> Path:
    """Plot the evaluation success rate."""
    return _plot_rate_bar(
        rate=summary.success_rate,
        title="Evaluation success rate",
        ylabel="Success rate",
        bar_label="Success",
        output_path=output_path,
        color="tab:green",
    )


def plot_failure_rate_bar(
    failure_rate: float,
    output_path: str | Path,
) -> Path:
    """Plot a planner failure rate."""
    return _plot_rate_bar(
        rate=failure_rate,
        title="Planner failure rate",
        ylabel="Failure rate",
        bar_label="Failure",
        output_path=output_path,
        color="tab:orange",
    )


def plot_collision_rate_bar(
    summary: EvaluationSummary,
    output_path: str | Path,
) -> Path:
    """Plot the evaluation collision rate."""
    return _plot_rate_bar(
        rate=summary.collision_rate,
        title="Evaluation collision rate",
        ylabel="Collision rate",
        bar_label="Collision",
        output_path=output_path,
        color="tab:red",
    )


def plot_dynamic_trajectory_png(
    uav_positions: Sequence[Position],
    dynamic_obstacle_positions: Sequence[Sequence[Position]],
    start_position: Position,
    goal_position: Position,
    grid_size: int,
    output_path: str | Path,
    collision_step: int | None = None,
) -> Path:
    """Plot a dynamic UAV rollout with obstacle trails."""
    saved_path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 7))
    _setup_grid_axis(axis, grid_size, "Dynamic UAV trajectory")

    _plot_dynamic_rollout_frame(
        axis=axis,
        uav_positions=uav_positions,
        dynamic_obstacle_positions=dynamic_obstacle_positions,
        start_position=start_position,
        goal_position=goal_position,
        grid_size=grid_size,
        frame_index=len(uav_positions) - 1,
        collision_step=collision_step,
        show_trails=True,
    )

    figure.tight_layout()
    figure.savefig(saved_path)
    plt.close(figure)
    return saved_path


def plot_dynamic_trajectory_gif(
    uav_positions: Sequence[Position],
    dynamic_obstacle_positions: Sequence[Sequence[Position]],
    start_position: Position,
    goal_position: Position,
    grid_size: int,
    output_path: str | Path,
    collision_step: int | None = None,
    fps: int = 2,
) -> Path:
    """Animate a dynamic UAV rollout as a GIF."""
    saved_path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 7))
    frame_count = max(len(uav_positions), len(dynamic_obstacle_positions), 1)

    def update(frame_index: int) -> None:
        axis.clear()
        _setup_grid_axis(axis, grid_size, "Dynamic UAV trajectory")
        _plot_dynamic_rollout_frame(
            axis=axis,
            uav_positions=uav_positions,
            dynamic_obstacle_positions=dynamic_obstacle_positions,
            start_position=start_position,
            goal_position=goal_position,
            grid_size=grid_size,
            frame_index=frame_index,
            collision_step=collision_step,
            show_trails=False,
        )

    rollout_animation = animation.FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 // fps,
        repeat=False,
    )
    rollout_animation.save(saved_path, writer=animation.PillowWriter(fps=fps))
    plt.close(figure)
    return saved_path


def _plot_rate_bar(
    rate: float,
    title: str,
    ylabel: str,
    bar_label: str,
    output_path: str | Path,
    color: str,
) -> Path:
    saved_path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(5, 5))

    axis.bar([bar_label], [rate], color=color)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.0)
    axis.text(
        0,
        rate + 0.03 if rate < 0.95 else rate - 0.08,
        f"{rate:.2f}",
        ha="center",
    )

    figure.tight_layout()
    figure.savefig(saved_path)
    plt.close(figure)
    return saved_path


def _setup_grid_axis(axis: plt.Axes, grid_size: int, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xlim(-0.5, grid_size - 0.5)
    axis.set_ylim(-0.5, grid_size - 0.5)
    axis.set_xticks(range(grid_size))
    axis.set_yticks(range(grid_size))
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axis.set_aspect("equal", adjustable="box")


def _plot_dynamic_rollout_frame(
    axis: plt.Axes,
    uav_positions: Sequence[Position],
    dynamic_obstacle_positions: Sequence[Sequence[Position]],
    start_position: Position,
    goal_position: Position,
    grid_size: int,
    frame_index: int,
    collision_step: int | None,
    show_trails: bool,
) -> None:
    del grid_size
    max_uav_index = min(frame_index, len(uav_positions) - 1)
    uav_positions_to_plot = uav_positions[: max_uav_index + 1]
    obstacle_positions = _positions_at_frame(dynamic_obstacle_positions, frame_index)

    axis.scatter(
        [start_position[0]],
        [start_position[1]],
        marker="o",
        color="tab:green",
        s=120,
        label="Start",
    )
    axis.scatter(
        [goal_position[0]],
        [goal_position[1]],
        marker="*",
        color="tab:red",
        s=180,
        label="Goal",
    )

    if show_trails:
        _plot_dynamic_obstacle_trails(axis, dynamic_obstacle_positions)

    if obstacle_positions:
        obstacle_x, obstacle_y = zip(*obstacle_positions, strict=False)
        axis.scatter(
            obstacle_x,
            obstacle_y,
            marker="s",
            color="black",
            label="Dynamic obstacles",
        )

    if uav_positions_to_plot:
        path_x, path_y = zip(*uav_positions_to_plot, strict=False)
        axis.plot(path_x, path_y, color="tab:blue", marker="o", label="UAV path")

    if collision_step is not None and frame_index >= collision_step:
        collision_position = uav_positions[min(collision_step, len(uav_positions) - 1)]
        axis.scatter(
            [collision_position[0]],
            [collision_position[1]],
            marker="x",
            color="tab:red",
            s=200,
            linewidths=3,
            label="Collision",
        )

    axis.legend(loc="best")


def _plot_dynamic_obstacle_trails(
    axis: plt.Axes,
    dynamic_obstacle_positions: Sequence[Sequence[Position]],
) -> None:
    trails = _get_dynamic_obstacle_trails(dynamic_obstacle_positions)
    for trail in trails:
        if not trail:
            continue
        trail_x, trail_y = zip(*trail, strict=False)
        axis.plot(trail_x, trail_y, color="gray", alpha=0.5, linestyle=":")
        if len(trail) >= 2:
            previous_x, previous_y = trail[-2]
            final_x, final_y = trail[-1]
            dx = final_x - previous_x
            dy = final_y - previous_y
            if dx != 0 or dy != 0:
                axis.arrow(
                    final_x,
                    final_y,
                    dx * 0.35,
                    dy * 0.35,
                    color="gray",
                    head_width=0.12,
                    length_includes_head=True,
                    zorder=5,
                )


def _get_dynamic_obstacle_trails(
    dynamic_obstacle_positions: Sequence[Sequence[Position]],
) -> list[list[Position]]:
    max_obstacles = max(
        (len(frame_positions) for frame_positions in dynamic_obstacle_positions),
        default=0,
    )
    return [
        [
            frame_positions[obstacle_index]
            for frame_positions in dynamic_obstacle_positions
            if obstacle_index < len(frame_positions)
        ]
        for obstacle_index in range(max_obstacles)
    ]


def _positions_at_frame(
    dynamic_obstacle_positions: Sequence[Sequence[Position]],
    frame_index: int,
) -> Sequence[Position]:
    if not dynamic_obstacle_positions:
        return []

    safe_index = min(frame_index, len(dynamic_obstacle_positions) - 1)
    return dynamic_obstacle_positions[safe_index]


def _prepare_output_path(output_path: str | Path) -> Path:
    saved_path = Path(output_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    return saved_path
