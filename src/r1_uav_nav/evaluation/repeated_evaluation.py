"""Helpers for summarising repeated evaluation runs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class RepeatedEvaluationRun:
    """Summary metrics from one repeated evaluation run."""

    repeat_index: int
    seed: int
    num_episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    average_reward: float
    average_steps: float
    average_path_length: float


def calculate_mean_std(values: Sequence[float]) -> dict[str, float]:
    """Calculate mean and population standard deviation for numeric values."""
    if not values:
        raise ValueError("Cannot calculate mean/std for empty values")

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}


def summarise_repeated_evaluations(
    runs: Sequence[RepeatedEvaluationRun],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable summary for repeated evaluation runs."""
    if not runs:
        raise ValueError("Cannot summarise empty repeated evaluation runs")

    run_dicts = [asdict(run) for run in runs]
    return {
        "metadata": metadata or {},
        "num_repeats": len(runs),
        "episodes_per_repeat": runs[0].num_episodes,
        "seeds": [run.seed for run in runs],
        "metrics": {
            "success_rate": calculate_mean_std([run.success_rate for run in runs]),
            "collision_rate": calculate_mean_std([run.collision_rate for run in runs]),
            "timeout_rate": calculate_mean_std([run.timeout_rate for run in runs]),
            "average_reward": calculate_mean_std([run.average_reward for run in runs]),
            "average_steps": calculate_mean_std([run.average_steps for run in runs]),
            "average_path_length": calculate_mean_std(
                [run.average_path_length for run in runs]
            ),
        },
        "runs": run_dicts,
    }


def save_repeated_evaluation_summary(
    summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a repeated evaluation summary as indented JSON."""
    saved_path = Path(output_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    with saved_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return saved_path
