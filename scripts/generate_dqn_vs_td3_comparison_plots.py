"""Generate tuned DQN-vs-TD3 dynamic comparison plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from r1_uav_nav.evaluation.dynamic_comparison import (
    DynamicAlgorithmResult,
    get_comparison_metric_names,
    get_dynamic_comparison_results,
)

DEFAULT_OUTPUT_DIR = Path("results/plots/comparison/dqn_vs_td3")
METRIC_DISPLAY = {
    "success_rate": {
        "title": "Tuned Dynamic Navigation Success Rate",
        "ylabel": "Success rate",
        "filename": "success_rate_comparison.png",
    },
    "collision_rate": {
        "title": "Tuned Dynamic Navigation Collision Rate",
        "ylabel": "Collision rate",
        "filename": "collision_rate_comparison.png",
    },
    "average_reward": {
        "title": "Tuned Dynamic Navigation Average Reward",
        "ylabel": "Average reward",
        "filename": "average_reward_comparison.png",
    },
    "average_steps": {
        "title": "Tuned Dynamic Navigation Average Steps",
        "ylabel": "Average steps",
        "filename": "average_steps_comparison.png",
    },
    "average_path_length": {
        "title": "Tuned Dynamic Navigation Path Length",
        "ylabel": "Average path length",
        "filename": "path_length_comparison.png",
    },
    "timeout_rate": {
        "title": "Tuned Dynamic Navigation Timeout Rate",
        "ylabel": "Timeout rate",
        "filename": "timeout_rate_comparison.png",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse comparison plot generation arguments."""
    parser = argparse.ArgumentParser(
        description="Generate tuned DQN-vs-TD3 dynamic comparison plots.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def generate_comparison_plots(output_dir: Path) -> list[Path]:
    """Generate one bar-chart PNG per comparison metric."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = get_dynamic_comparison_results()
    plot_paths = []

    for metric_name in get_comparison_metric_names():
        display = METRIC_DISPLAY[metric_name]
        output_path = output_dir / display["filename"]
        _plot_metric_comparison(
            results=results,
            metric_name=metric_name,
            title=display["title"],
            ylabel=display["ylabel"],
            output_path=output_path,
        )
        plot_paths.append(output_path)

    return plot_paths


def main() -> None:
    """Generate DQN-vs-TD3 comparison plots."""
    args = parse_args()
    plot_paths = generate_comparison_plots(args.output_dir)

    print("DQN vs TD3 comparison plots generated.")
    for plot_path in plot_paths:
        print(f"- {plot_path}")


def _plot_metric_comparison(
    results: Sequence[DynamicAlgorithmResult],
    metric_name: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    labels = [result.label for result in results]
    values = [getattr(result, metric_name) for result in results]
    std_values = [getattr(result, f"{metric_name}_std") for result in results]

    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(
        labels,
        values,
        yerr=std_values,
        capsize=6,
        color=["tab:blue", "tab:green"],
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    _set_y_axis_limits(axis, metric_name, values, std_values)

    for bar, value, std_value in zip(bars, values, std_values, strict=False):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}\n+/- {std_value:.4f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _set_y_axis_limits(
    axis: plt.Axes,
    metric_name: str,
    values: Sequence[float],
    std_values: Sequence[float],
) -> None:
    upper_value = max(
        value + std_value for value, std_value in zip(values, std_values, strict=False)
    )
    if metric_name.endswith("_rate"):
        axis.set_ylim(0.0, min(1.0, upper_value + 0.12))
    else:
        axis.set_ylim(0.0, upper_value * 1.20 if upper_value > 0.0 else 1.0)


if __name__ == "__main__":
    main()
