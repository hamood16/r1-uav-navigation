"""Pure reward and clearance helpers for the M13.5 obstacle environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from r1_uav_nav.sim.lidar_features import LidarFeatureConfig


@dataclass(frozen=True)
class ObstacleRewardConfig:
    """Finite positive magnitudes used by the obstacle-aware reward."""

    progress_scale: float = 2.0
    success_bonus: float = 25.0
    step_penalty: float = 0.02
    collision_penalty: float = 25.0
    safety_violation_penalty: float = 15.0
    unsafe_clearance_penalty_scale: float = 1.0
    action_magnitude_penalty_scale: float = 0.02
    action_change_penalty_scale: float = 0.04
    clearance_safety_band_m: float = 1.0
    emergency_clearance_m: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "progress_scale",
            "success_bonus",
            "step_penalty",
            "collision_penalty",
            "safety_violation_penalty",
            "unsafe_clearance_penalty_scale",
            "action_magnitude_penalty_scale",
            "action_change_penalty_scale",
            "clearance_safety_band_m",
            "emergency_clearance_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.clearance_safety_band_m <= 0:
            raise ValueError("clearance_safety_band_m must be positive")
        if self.emergency_clearance_m > self.clearance_safety_band_m:
            raise ValueError(
                "emergency_clearance_m must not exceed clearance_safety_band_m"
            )


@dataclass(frozen=True)
class ClearanceEvidence:
    """Current and retained clearance evidence used for one reward step."""

    measured_clearance_m: float | None
    reward_clearance_m: float | None
    source: str
    unsafe: bool
    severity: float


@dataclass(frozen=True)
class ObstacleRewardBreakdown:
    """Signed reward components and their finite total."""

    goal_progress_reward: float
    success_bonus: float
    step_penalty: float
    collision_penalty: float
    safety_violation_penalty: float
    unsafe_clearance_penalty: float
    action_magnitude_penalty: float
    action_change_penalty: float
    total: float

    def __post_init__(self) -> None:
        values = (
            self.goal_progress_reward,
            self.success_bonus,
            self.step_penalty,
            self.collision_penalty,
            self.safety_violation_penalty,
            self.unsafe_clearance_penalty,
            self.action_magnitude_penalty,
            self.action_change_penalty,
            self.total,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("reward breakdown must contain only finite values")


def calculate_clearance_evidence(
    sector_features: Sequence[float] | np.ndarray,
    *,
    lidar_valid: bool,
    lidar_config: LidarFeatureConfig,
    reward_config: ObstacleRewardConfig,
    last_valid_clearance_m: float | None,
) -> ClearanceEvidence:
    """Resolve current or retained clearance without trusting invalid fallback data."""
    measured: float | None = None
    source = "unavailable"
    if lidar_valid:
        features = np.asarray(sector_features, dtype=np.float64)
        if features.shape != (lidar_config.feature_count,):
            raise ValueError("sector_features must match the LiDAR feature count")
        if not np.all(np.isfinite(features)):
            raise ValueError("sector_features must be finite")
        if np.any(features < 0.0) or np.any(features > 1.0):
            raise ValueError("sector_features must remain within [0, 1]")
        minimum_normalized = float(np.min(features))
        measured = lidar_config.minimum_range_m + minimum_normalized * (
            lidar_config.maximum_range_m - lidar_config.minimum_range_m
        )
        reward_clearance = measured
        source = "current"
    else:
        reward_clearance = _optional_clearance(
            last_valid_clearance_m,
            lidar_config,
        )
        if reward_clearance is not None:
            source = "last_valid"

    if reward_clearance is None:
        return ClearanceEvidence(None, None, source, False, 0.0)

    denominator = reward_config.clearance_safety_band_m - lidar_config.minimum_range_m
    if denominator <= 0:
        raise ValueError("clearance_safety_band_m must exceed the LiDAR minimum range")
    severity = float(
        np.clip(
            (reward_config.clearance_safety_band_m - reward_clearance) / denominator,
            0.0,
            1.0,
        )
    )
    return ClearanceEvidence(
        measured,
        reward_clearance,
        source,
        reward_clearance < reward_config.clearance_safety_band_m,
        severity,
    )


def calculate_obstacle_reward(
    *,
    previous_distance_to_goal: float,
    distance_to_goal: float,
    action: Sequence[float] | np.ndarray,
    previous_action: Sequence[float] | np.ndarray,
    clearance: ClearanceEvidence,
    terminal_reason: str | None,
    config: ObstacleRewardConfig,
) -> ObstacleRewardBreakdown:
    """Calculate one reward using the primary terminal outcome only."""
    previous_distance = _finite_float(
        previous_distance_to_goal, "previous_distance_to_goal"
    )
    current_distance = _finite_float(distance_to_goal, "distance_to_goal")
    current_action = _action(action, "action")
    prior_action = _action(previous_action, "previous_action")

    progress = config.progress_scale * (previous_distance - current_distance)
    success = config.success_bonus if terminal_reason == "goal_reached" else 0.0
    step = -config.step_penalty
    collision = -config.collision_penalty if terminal_reason == "collision" else 0.0
    safety = (
        -config.safety_violation_penalty
        if terminal_reason
        in {"ground_clearance_violation", "workspace_violation", "rpc_recovery"}
        else 0.0
    )
    unsafe = -config.unsafe_clearance_penalty_scale * clearance.severity**2
    action_magnitude = float(np.linalg.norm(current_action))
    action_change_magnitude = float(np.linalg.norm(current_action - prior_action))
    magnitude_penalty = -config.action_magnitude_penalty_scale * action_magnitude
    suppress_change = (
        clearance.reward_clearance_m is not None
        and clearance.reward_clearance_m <= config.emergency_clearance_m
    )
    change_penalty = (
        0.0
        if suppress_change
        else -config.action_change_penalty_scale * action_change_magnitude
    )
    components = (
        progress,
        success,
        step,
        collision,
        safety,
        unsafe,
        magnitude_penalty,
        change_penalty,
    )
    total = float(sum(components))
    return ObstacleRewardBreakdown(*components, total)


def _optional_clearance(
    value: float | None,
    config: LidarFeatureConfig,
) -> float | None:
    if value is None:
        return None
    clearance = _finite_float(value, "last_valid_clearance_m")
    if not config.minimum_range_m <= clearance <= config.maximum_range_m:
        raise ValueError("last_valid_clearance_m is outside the configured range")
    return clearance


def _action(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float64)
    if action.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(action)):
        raise ValueError(f"{name} must contain finite values")
    return action


def _finite_float(value: float, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted
