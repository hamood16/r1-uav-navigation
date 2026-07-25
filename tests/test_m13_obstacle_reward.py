from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from r1_uav_nav.envs.obstacle_reward import (
    ClearanceEvidence,
    ObstacleRewardConfig,
    calculate_clearance_evidence,
    calculate_obstacle_reward,
)
from r1_uav_nav.sim.lidar_features import load_lidar_feature_config

ROOT = Path(__file__).resolve().parents[1]
LIDAR_CONFIG = load_lidar_feature_config(
    ROOT / "configs" / "sensing" / "m13_4_lidar_features.yaml"
)


def _clearance(
    normalized: float,
    *,
    valid: bool = True,
    last_valid: float | None = None,
    reward_config: ObstacleRewardConfig | None = None,
) -> ClearanceEvidence:
    return calculate_clearance_evidence(
        np.full(72, normalized, dtype=np.float32),
        lidar_valid=valid,
        lidar_config=LIDAR_CONFIG,
        reward_config=reward_config or ObstacleRewardConfig(),
        last_valid_clearance_m=last_valid,
    )


def _reward(
    *,
    previous_distance: float = 5.0,
    distance: float = 4.0,
    action: tuple[float, float, float] = (0.0, 0.0, 0.0),
    previous_action: tuple[float, float, float] = (0.0, 0.0, 0.0),
    clearance: ClearanceEvidence | None = None,
    terminal_reason: str | None = None,
    config: ObstacleRewardConfig | None = None,
):
    return calculate_obstacle_reward(
        previous_distance_to_goal=previous_distance,
        distance_to_goal=distance,
        action=action,
        previous_action=previous_action,
        clearance=clearance or _clearance(1.0),
        terminal_reason=terminal_reason,
        config=config or ObstacleRewardConfig(),
    )


def test_progress_and_step_reward_components() -> None:
    reward = _reward(previous_distance=5.0, distance=4.5)

    assert reward.goal_progress_reward == pytest.approx(1.0)
    assert reward.step_penalty == pytest.approx(-0.02)
    assert reward.total == pytest.approx(0.98)


@pytest.mark.parametrize(
    ("reason", "field", "expected"),
    [
        ("goal_reached", "success_bonus", 25.0),
        ("collision", "collision_penalty", -25.0),
        ("ground_clearance_violation", "safety_violation_penalty", -15.0),
        ("workspace_violation", "safety_violation_penalty", -15.0),
    ],
)
def test_terminal_components_are_applied_independently(
    reason: str,
    field: str,
    expected: float,
) -> None:
    reward = _reward(terminal_reason=reason)

    assert getattr(reward, field) == pytest.approx(expected)
    if reason == "collision":
        assert reward.success_bonus == 0.0


def test_clearance_severity_is_quadratic_inside_safety_band() -> None:
    config = ObstacleRewardConfig(clearance_safety_band_m=1.0)
    target_clearance = 0.625
    normalized = (target_clearance - LIDAR_CONFIG.minimum_range_m) / (
        LIDAR_CONFIG.maximum_range_m - LIDAR_CONFIG.minimum_range_m
    )
    clearance = _clearance(normalized, reward_config=config)
    reward = _reward(clearance=clearance, config=config)

    assert clearance.severity == pytest.approx(0.5)
    assert clearance.unsafe
    assert reward.unsafe_clearance_penalty == pytest.approx(-0.25)


def test_clearance_penalty_is_zero_outside_band_and_maximum_at_minimum() -> None:
    clear = _clearance(1.0)
    danger = _clearance(0.0)

    assert clear.severity == 0.0
    assert not clear.unsafe
    assert danger.severity == 1.0
    assert _reward(clearance=danger).unsafe_clearance_penalty == pytest.approx(-1.0)


def test_action_penalties_and_emergency_change_suppression() -> None:
    normal = _reward(
        action=(1.0, 0.0, 0.0),
        previous_action=(-1.0, 0.0, 0.0),
    )
    emergency = _reward(
        action=(1.0, 0.0, 0.0),
        previous_action=(-1.0, 0.0, 0.0),
        clearance=_clearance(0.0),
    )

    assert normal.action_magnitude_penalty == pytest.approx(-0.02)
    assert normal.action_change_penalty == pytest.approx(-0.08)
    assert emergency.action_magnitude_penalty == pytest.approx(-0.02)
    assert emergency.action_change_penalty == 0.0


def test_transient_invalid_scan_uses_last_valid_clearance() -> None:
    retained = calculate_clearance_evidence(
        np.ones(72, dtype=np.float32),
        lidar_valid=False,
        lidar_config=LIDAR_CONFIG,
        reward_config=ObstacleRewardConfig(),
        last_valid_clearance_m=0.75,
    )

    assert retained.measured_clearance_m is None
    assert retained.reward_clearance_m == pytest.approx(0.75)
    assert retained.source == "last_valid"
    assert retained.unsafe


def test_invalid_fallback_is_never_interpreted_as_current_all_clear() -> None:
    unavailable = calculate_clearance_evidence(
        np.ones(72, dtype=np.float32),
        lidar_valid=False,
        lidar_config=LIDAR_CONFIG,
        reward_config=ObstacleRewardConfig(),
        last_valid_clearance_m=None,
    )

    assert unavailable == ClearanceEvidence(None, None, "unavailable", False, 0.0)


@pytest.mark.parametrize(
    "config",
    [
        ObstacleRewardConfig(progress_scale=0.0),
        ObstacleRewardConfig(action_change_penalty_scale=0.0),
    ],
)
def test_reward_remains_finite_for_valid_edge_configs(
    config: ObstacleRewardConfig,
) -> None:
    reward = _reward(config=config)
    assert math.isfinite(reward.total)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"progress_scale": float("nan")},
        {"collision_penalty": -1.0},
        {"emergency_clearance_m": 1.1},
    ],
)
def test_reward_configuration_rejects_invalid_values(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        ObstacleRewardConfig(**kwargs)


def test_reward_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="distance_to_goal"):
        _reward(distance=float("inf"))
