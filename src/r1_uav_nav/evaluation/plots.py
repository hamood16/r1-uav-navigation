"""Plotting helpers for evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

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


def _prepare_output_path(output_path: str | Path) -> Path:
    saved_path = Path(output_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    return saved_path
