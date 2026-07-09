"""Helpers for selecting representative evaluation rollouts."""

from __future__ import annotations

from typing import Any

Rollout = dict[str, Any]


def select_fastest_successful_rollout(rollouts: list[Rollout]) -> Rollout:
    """Select the fastest successful rollout, or the first rollout as fallback."""
    successful_rollouts = [rollout for rollout in rollouts if rollout["success"]]
    if successful_rollouts:
        return min(successful_rollouts, key=lambda rollout: rollout["steps"])
    return rollouts[0]


def select_mean_length_rollout(rollouts: list[Rollout]) -> Rollout:
    """Select the rollout closest to mean successful path length."""
    candidates = _successful_rollouts_or_all(rollouts)
    mean_path_length = sum(rollout["path_length"] for rollout in candidates) / len(
        candidates
    )
    return min(
        candidates,
        key=lambda rollout: abs(rollout["path_length"] - mean_path_length),
    )


def select_longest_rollout(rollouts: list[Rollout]) -> Rollout:
    """Select the longest successful rollout, or longest overall as fallback."""
    candidates = _successful_rollouts_or_all(rollouts)
    return max(candidates, key=lambda rollout: rollout["path_length"])


def _successful_rollouts_or_all(rollouts: list[Rollout]) -> list[Rollout]:
    successful_rollouts = [rollout for rollout in rollouts if rollout["success"]]
    return successful_rollouts if successful_rollouts else rollouts
