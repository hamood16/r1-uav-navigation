"""Evaluation metrics for UAV navigation experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Position = tuple[int, int]


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics collected from one evaluation episode."""

    total_reward: float
    steps: int
    success: bool
    collision: bool
    path_length: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics across multiple evaluation episodes."""

    num_episodes: int
    success_rate: float
    collision_rate: float
    average_reward: float
    average_steps: float
    average_path_length: float


def calculate_path_length(positions: Sequence[Position]) -> float:
    """Calculate total Euclidean distance travelled through a position sequence."""
    if len(positions) < 2:
        return 0.0

    return sum(
        math.hypot(next_x - current_x, next_y - current_y)
        for (current_x, current_y), (next_x, next_y) in zip(
            positions, positions[1:], strict=False
        )
    )


def summarise_episode_results(
    results: Sequence[EpisodeResult],
) -> EvaluationSummary:
    """Summarise per-episode evaluation results."""
    if not results:
        raise ValueError("Cannot summarise empty episode results")

    num_episodes = len(results)
    return EvaluationSummary(
        num_episodes=num_episodes,
        success_rate=sum(result.success for result in results) / num_episodes,
        collision_rate=sum(result.collision for result in results) / num_episodes,
        average_reward=sum(result.total_reward for result in results) / num_episodes,
        average_steps=sum(result.steps for result in results) / num_episodes,
        average_path_length=sum(result.path_length for result in results)
        / num_episodes,
    )
