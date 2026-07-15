"""Helpers for comparing static DQN and classical planner results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

Position = tuple[int, int]
ComparisonRecord = dict[str, Any]
ComparisonSummary = dict[str, Any]


@dataclass(frozen=True)
class StaticLayout:
    """A static grid layout shared by planners and policies."""

    start_position: Position
    goal_position: Position
    obstacles: frozenset[Position]
    grid_size: int


def layouts_match(first: StaticLayout, second: StaticLayout) -> bool:
    """Return True when two static layouts are exactly the same."""
    return (
        first.start_position == second.start_position
        and first.goal_position == second.goal_position
        and first.obstacles == second.obstacles
        and first.grid_size == second.grid_size
    )


def assert_layouts_match(first: StaticLayout, second: StaticLayout) -> None:
    """Raise ValueError if two static layouts differ."""
    if not layouts_match(first, second):
        raise ValueError(
            "A* and DQN layouts do not match after resetting with the same seed"
        )


def summarise_static_comparison(
    records: Sequence[ComparisonRecord],
    *,
    eval_seed: int,
    model_path: str,
) -> ComparisonSummary:
    """Summarise static DQN-vs-A* comparison records."""
    if not records:
        raise ValueError("Cannot summarise empty comparison records")

    num_episodes = len(records)
    astar_successes = [record for record in records if record["astar"]["success"]]
    dqn_successes = [record for record in records if record["dqn"]["success"]]
    shared_successes = [
        record
        for record in records
        if record["astar"]["success"] and record["dqn"]["success"]
    ]

    astar_success_rate = len(astar_successes) / num_episodes
    dqn_success_rate = len(dqn_successes) / num_episodes
    astar_average_successful_path_length = _average(
        record["astar"]["path_length"] for record in astar_successes
    )
    astar_average_successful_steps = _average(
        record["astar"]["steps"] for record in astar_successes
    )
    dqn_average_reward = _average(record["dqn"]["total_reward"] for record in records)
    dqn_average_path_length = _average(
        record["dqn"]["path_length"] for record in records
    )
    dqn_average_steps = _average(record["dqn"]["steps"] for record in records)

    shared_success_average_astar_path_length = _average(
        record["astar"]["path_length"] for record in shared_successes
    )
    shared_success_average_dqn_path_length = _average(
        record["dqn"]["path_length"] for record in shared_successes
    )
    shared_success_average_astar_steps = _average(
        record["astar"]["steps"] for record in shared_successes
    )
    shared_success_average_dqn_steps = _average(
        record["dqn"]["steps"] for record in shared_successes
    )
    dqn_to_astar_ratio = (
        shared_success_average_dqn_path_length
        / shared_success_average_astar_path_length
        if shared_success_average_astar_path_length > 0.0
        else 0.0
    )

    return {
        "num_episodes": num_episodes,
        "eval_seed": eval_seed,
        "model_path": model_path,
        "astar": {
            "success_rate": astar_success_rate,
            "failure_rate": 1.0 - astar_success_rate,
            "collision_rate": 0.0,
            "average_successful_path_length": astar_average_successful_path_length,
            "average_successful_steps": astar_average_successful_steps,
            "averages_note": (
                "A* path length and step averages use successful plans only."
            ),
        },
        "dqn": {
            "success_rate": dqn_success_rate,
            "collision_rate": _rate(records, "dqn", "collision"),
            "timeout_rate": _rate(records, "dqn", "timeout"),
            "average_reward": dqn_average_reward,
            "average_path_length": dqn_average_path_length,
            "average_steps": dqn_average_steps,
            "averages_note": (
                "DQN reward, path length, and step averages use all episodes."
            ),
        },
        "overall_comparison": {
            "success_rate_difference_dqn_minus_astar": dqn_success_rate
            - astar_success_rate,
            "average_path_length_difference_dqn_all_minus_astar_successful": (
                dqn_average_path_length - astar_average_successful_path_length
            ),
            "average_steps_difference_dqn_all_minus_astar_successful": (
                dqn_average_steps - astar_average_successful_steps
            ),
        },
        "shared_success_comparison": {
            "shared_success_count": len(shared_successes),
            "dqn_failed_astar_succeeded_count": sum(
                record["astar"]["success"] and not record["dqn"]["success"]
                for record in records
            ),
            "shared_success_average_astar_path_length": (
                shared_success_average_astar_path_length
            ),
            "shared_success_average_dqn_path_length": (
                shared_success_average_dqn_path_length
            ),
            "shared_success_path_length_difference_dqn_minus_astar": (
                shared_success_average_dqn_path_length
                - shared_success_average_astar_path_length
            ),
            "shared_success_average_astar_steps": shared_success_average_astar_steps,
            "shared_success_average_dqn_steps": shared_success_average_dqn_steps,
            "shared_success_steps_difference_dqn_minus_astar": (
                shared_success_average_dqn_steps - shared_success_average_astar_steps
            ),
            "dqn_to_astar_path_length_ratio_shared_success": dqn_to_astar_ratio,
        },
    }


def select_representative_shared_success_episode(
    records: Sequence[ComparisonRecord],
) -> ComparisonRecord:
    """Select a representative shared-success episode, or fall back to the first."""
    if not records:
        raise ValueError("Cannot select from empty comparison records")

    shared_successes = [
        record
        for record in records
        if record["astar"]["success"] and record["dqn"]["success"]
    ]
    if not shared_successes:
        return records[0]

    average_dqn_success_path_length = _average(
        record["dqn"]["path_length"] for record in shared_successes
    )
    return min(
        shared_successes,
        key=lambda record: abs(
            record["dqn"]["path_length"] - average_dqn_success_path_length
        ),
    )


def _average(values: Sequence[float] | Any) -> float:
    materialised_values = list(values)
    if not materialised_values:
        return 0.0
    return sum(materialised_values) / len(materialised_values)


def _rate(
    records: Sequence[ComparisonRecord],
    method_name: str,
    field_name: str,
) -> float:
    return sum(record[method_name][field_name] for record in records) / len(records)
