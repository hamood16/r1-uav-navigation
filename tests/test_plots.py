from pathlib import Path

from r1_uav_nav.evaluation import (
    EvaluationSummary,
    plot_collision_rate_bar,
    plot_reward_curve,
    plot_success_rate_bar,
    plot_trajectory,
)


def test_plot_trajectory_creates_non_empty_png(tmp_path: Path) -> None:
    output_path = tmp_path / "trajectory.png"

    plot_trajectory(
        trajectory_positions=[(0, 0), (1, 0), (1, 1), (2, 1)],
        obstacles={(3, 3), (4, 4)},
        start_position=(0, 0),
        goal_position=(4, 4),
        grid_size=5,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_reward_curve_creates_non_empty_png(tmp_path: Path) -> None:
    output_path = tmp_path / "reward_curve.png"

    plot_reward_curve(
        episode_rewards=[-1.0, 0.2, 0.5],
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_success_rate_bar_creates_non_empty_png(tmp_path: Path) -> None:
    output_path = tmp_path / "success_rate.png"

    plot_success_rate_bar(
        summary=_evaluation_summary(),
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_collision_rate_bar_creates_non_empty_png(tmp_path: Path) -> None:
    output_path = tmp_path / "collision_rate.png"

    plot_collision_rate_bar(
        summary=_evaluation_summary(),
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _evaluation_summary() -> EvaluationSummary:
    return EvaluationSummary(
        num_episodes=3,
        success_rate=0.67,
        collision_rate=0.33,
        average_reward=0.2,
        average_steps=12.0,
        average_path_length=8.0,
    )
