"""Pure M13.4 LiDAR feature-extraction tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from r1_uav_nav.sim.lidar_features import (
    LIDAR_EXTRACTION_EVIDENCE_SCHEMA_VERSION,
    LIDAR_FEATURE_CONFIG_SCHEMA_VERSION,
    LIDAR_FEATURE_COUNT,
    GroundFilterMode,
    LidarFaultTracker,
    LidarFeatureConfig,
    LidarFeatureError,
    LidarScanInput,
    LidarTimestampStatus,
    LidarTimestampTracker,
    LidarValidityStatus,
    extract_lidar_features,
    extraction_evidence,
    load_lidar_feature_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sensing/m13_4_lidar_features.yaml"


def _scan(
    points: object,
    timestamp: int = 1,
    *,
    pose: bool = True,
) -> LidarScanInput:
    return LidarScanInput(
        point_cloud=points,  # type: ignore[arg-type]
        timestamp=timestamp,
        sensor_position=(0.0, 0.0, 0.0) if pose else None,
        sensor_orientation=(1.0, 0.0, 0.0, 0.0) if pose else None,
    )


def _point(
    range_m: float, angle_degrees: float, elevation_degrees: float
) -> list[float]:
    horizontal = np.radians(angle_degrees)
    elevation = np.radians(elevation_degrees)
    horizontal_range = range_m * np.cos(elevation)
    return [
        float(horizontal_range * np.cos(horizontal)),
        float(horizontal_range * np.sin(horizontal)),
        float(-range_m * np.sin(elevation)),
    ]


def test_loads_shipped_configuration_and_digest_is_stable() -> None:
    config = load_lidar_feature_config(CONFIG_PATH)

    assert config.schema_version == LIDAR_FEATURE_CONFIG_SCHEMA_VERSION
    assert config.feature_count == LIDAR_FEATURE_COUNT
    assert config.ground_filter_mode is GroundFilterMode.NONE
    assert config.elevation_boundary_tolerance_degrees == pytest.approx(0.0001)
    assert config.configuration_digest == config.configuration_digest
    assert (
        config.configuration_digest
        != replace(
            config, elevation_boundary_tolerance_degrees=0.0
        ).configuration_digest
    )


@pytest.mark.parametrize(
    "value",
    [-0.0001, float("nan"), float("inf"), "0.0001", "not-a-number", True],
)
def test_loader_rejects_invalid_elevation_boundary_tolerance(
    tmp_path: Path, value: object
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["elevation_boundary_tolerance_degrees"] = value
    path = tmp_path / "invalid-lidar.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(
        LidarFeatureError,
        match="elevation_boundary_tolerance_degrees",
    ):
        load_lidar_feature_config(path)


def test_invalid_threshold_relationship_and_filter_are_rejected() -> None:
    with pytest.raises(LidarFeatureError, match="must be at least"):
        replace(
            LidarFeatureConfig(),
            maximum_repeated_timestamp_transitions=4,
            maximum_consecutive_invalid_scans=3,
        )
    with pytest.raises(ValueError):
        GroundFilterMode("measured_ground_plane")


@pytest.mark.parametrize(
    ("points", "status"),
    [
        ([], LidarValidityStatus.EMPTY),
        ([1.0, 2.0], LidarValidityStatus.MALFORMED),
        (["bad", 0.0, 0.0], LidarValidityStatus.MALFORMED),
        ([float("nan"), 0.0, 0.0], LidarValidityStatus.ALL_INVALID),
        (
            [1.0, 0.0, 0.0, float("inf"), 0.0, 0.0],
            LidarValidityStatus.PARTIALLY_INVALID,
        ),
    ],
)
def test_invalid_clouds_produce_bounded_fallback(
    points: object, status: LidarValidityStatus
) -> None:
    result = extract_lidar_features(_scan(points), LidarFeatureConfig())

    assert result.status is status
    assert result.lidar_valid == 0.0
    assert np.array_equal(result.features, np.ones(72, dtype=np.float32))


def test_missing_pose_does_not_invalidate_policy_features() -> None:
    result = extract_lidar_features(
        _scan(_point(2.0, 0.0, 0.0), pose=False),
        LidarFeatureConfig(),
    )

    assert result.status is LidarValidityStatus.VALID
    assert result.lidar_valid == 1.0
    assert not result.diagnostics.pose_valid
    assert result.diagnostics.pose_error


def test_rejected_partial_scan_retains_finite_point_elevation_diagnostics() -> None:
    points = _point(2.0, 0.0, 5.0) + [float("inf"), 0.0, 0.0]

    result = extract_lidar_features(_scan(points), LidarFeatureConfig())

    assert result.status is LidarValidityStatus.PARTIALLY_INVALID
    assert result.diagnostics.total_finite_point_count == 1
    assert result.diagnostics.observed_minimum_elevation_degrees == pytest.approx(5.0)
    assert result.diagnostics.observed_maximum_elevation_degrees == pytest.approx(5.0)
    assert result.diagnostics.in_fov_point_count == 1


def test_valid_no_obstacle_is_distinct_from_empty() -> None:
    config = LidarFeatureConfig()
    valid = extract_lidar_features(_scan(_point(20.0, 0.0, 0.0)), config)
    empty = extract_lidar_features(_scan([]), config)

    assert valid.status is LidarValidityStatus.VALID_NO_OBSTACLE
    assert valid.lidar_valid == 1.0
    assert np.array_equal(valid.features, np.ones(72, dtype=np.float32))
    assert empty.status is LidarValidityStatus.EMPTY
    assert empty.lidar_valid == 0.0


def test_range_clipping_and_strict_tolerance() -> None:
    config = LidarFeatureConfig()
    close = extract_lidar_features(_scan(_point(0.0, 0.0, 0.0)), config)
    tolerated = extract_lidar_features(_scan(_point(20.1, 0.0, 0.0)), config)
    rejected = extract_lidar_features(_scan(_point(20.1001, 0.0, 0.0)), config)

    assert float(np.min(close.features)) == pytest.approx(0.0)
    assert close.diagnostics.clipped_below_minimum_count == 1
    assert tolerated.status is LidarValidityStatus.VALID_NO_OBSTACLE
    assert tolerated.diagnostics.clipped_above_maximum_count == 1
    assert rejected.status is LidarValidityStatus.RANGE_VIOLATION


def test_outside_elevation_fov_rejects_whole_scan() -> None:
    result = extract_lidar_features(_scan(_point(2.0, 0.0, 10.1)), LidarFeatureConfig())

    assert result.status is LidarValidityStatus.OUTSIDE_ELEVATION_FOV
    assert result.lidar_valid == 0.0
    assert result.diagnostics.observed_maximum_elevation_degrees == pytest.approx(10.1)
    assert result.diagnostics.above_configured_fov_count == 1
    assert result.diagnostics.maximum_above_fov_overshoot_degrees == pytest.approx(0.1)
    assert result.diagnostics.in_fov_point_count == 0
    assert result.diagnostics.total_finite_point_count == 1
    assert result.diagnostics.configured_lower_elevation_degrees == -30.0
    assert result.diagnostics.configured_upper_elevation_degrees == 10.0


def test_measured_elevation_overshoots_are_accepted_with_raw_diagnostics() -> None:
    points = (
        _point(2.0, 0.0, -30.00000891919195)
        + _point(2.0, 15.0, 0.0)
        + _point(2.0, 30.0, 10.00000239043348)
    )

    result = extract_lidar_features(_scan(points), LidarFeatureConfig())

    assert result.status is LidarValidityStatus.VALID
    assert result.lidar_valid == 1.0
    diagnostics = result.diagnostics
    assert diagnostics.observed_minimum_elevation_degrees == pytest.approx(
        -30.00000891919195
    )
    assert diagnostics.observed_maximum_elevation_degrees == pytest.approx(
        10.00000239043348
    )
    assert diagnostics.below_configured_fov_count == 1
    assert diagnostics.above_configured_fov_count == 1
    assert diagnostics.maximum_below_fov_overshoot_degrees == pytest.approx(
        0.00000891919195
    )
    assert diagnostics.maximum_above_fov_overshoot_degrees == pytest.approx(
        0.00000239043348
    )
    assert diagnostics.in_fov_point_count == 1
    assert diagnostics.total_finite_point_count == 3
    assert diagnostics.configured_elevation_boundary_tolerance_degrees == pytest.approx(
        0.0001
    )
    assert diagnostics.sector_point_counts[12] == 1
    assert diagnostics.sector_point_counts[48 + 13] == 1
    assert diagnostics.sector_point_counts[48 + 14] == 1


@pytest.mark.parametrize(
    ("elevation_degrees", "expected_flat_sector"),
    [
        (-30.0001, 12),
        (10.0001, 48 + 12),
    ],
)
def test_exact_elevation_boundary_tolerance_is_accepted_and_clamped(
    elevation_degrees: float, expected_flat_sector: int
) -> None:
    result = extract_lidar_features(
        _scan(_point(2.0, 0.0, elevation_degrees)),
        LidarFeatureConfig(),
    )

    assert result.status is LidarValidityStatus.VALID
    assert result.lidar_valid == 1.0
    assert result.diagnostics.sector_point_counts[expected_flat_sector] == 1
    assert result.diagnostics.observed_minimum_elevation_degrees == pytest.approx(
        elevation_degrees
    )


@pytest.mark.parametrize("elevation_degrees", [-30.0001001, 10.0001001])
def test_elevation_beyond_configured_boundary_tolerance_rejects_whole_scan(
    elevation_degrees: float,
) -> None:
    result = extract_lidar_features(
        _scan(_point(2.0, 0.0, elevation_degrees)),
        LidarFeatureConfig(),
    )

    assert result.status is LidarValidityStatus.OUTSIDE_ELEVATION_FOV
    assert result.lidar_valid == 0.0


@pytest.mark.parametrize(
    ("elevation_degrees", "count_field", "overshoot_field"),
    [
        (
            -30.1,
            "below_configured_fov_count",
            "maximum_below_fov_overshoot_degrees",
        ),
        (
            10.1,
            "above_configured_fov_count",
            "maximum_above_fov_overshoot_degrees",
        ),
    ],
)
def test_lower_and_upper_elevation_overshoot_diagnostics_are_retained(
    elevation_degrees: float,
    count_field: str,
    overshoot_field: str,
) -> None:
    result = extract_lidar_features(
        _scan(_point(2.0, 0.0, elevation_degrees)),
        LidarFeatureConfig(),
    )

    assert result.status is LidarValidityStatus.OUTSIDE_ELEVATION_FOV
    assert getattr(result.diagnostics, count_field) == 1
    assert getattr(result.diagnostics, overshoot_field) == pytest.approx(0.1)
    assert result.diagnostics.total_finite_point_count == 1


def test_sector_boundaries_and_elevation_major_order() -> None:
    points = (
        _point(2.0, -180.0, -20.0)
        + _point(3.0, -165.0, -10.0)
        + _point(4.0, 0.0, 0.0)
        + _point(5.0, 179.999, 9.0)
    )
    result = extract_lidar_features(_scan(points), LidarFeatureConfig())

    assert result.status is LidarValidityStatus.VALID
    assert result.diagnostics.sector_point_counts[0] == 1
    assert result.diagnostics.sector_point_counts[24 + 1] == 1
    assert result.diagnostics.sector_point_counts[48 + 12] == 1
    assert result.diagnostics.sector_point_counts[48 + 23] == 1


def test_left_right_forward_diagnostics_follow_sensor_local_ned() -> None:
    points = _point(2.0, 0.0, 0.0) + _point(3.0, -90.0, 0.0) + _point(4.0, 90.0, 0.0)
    result = extract_lidar_features(_scan(points), LidarFeatureConfig())

    assert result.diagnostics.nearest_forward_m == pytest.approx(2.0)
    assert result.diagnostics.minimum_left_clearance_m == pytest.approx(3.0)
    assert result.diagnostics.minimum_right_clearance_m == pytest.approx(4.0)
    assert result.diagnostics.nearest_overall_m == pytest.approx(2.0)


def test_input_and_numpy_random_state_are_unchanged() -> None:
    points = _point(2.0, 0.0, 0.0)
    original = list(points)
    state = np.random.get_state()

    first = extract_lidar_features(_scan(points), LidarFeatureConfig())
    second = extract_lidar_features(_scan(points), LidarFeatureConfig())

    assert points == original
    assert np.array_equal(first.features, second.features)
    after = np.random.get_state()
    assert state[0] == after[0]
    assert np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_timestamp_off_by_one_contract_and_fresh_recovery() -> None:
    tracker = LidarTimestampTracker(maximum_repeated_transitions=3)

    assert tracker.observe(10).status is LidarTimestampStatus.FIRST_VALID
    assert tracker.observe(10).status is LidarTimestampStatus.REPEATED
    assert tracker.repeated_count == 1
    assert tracker.observe(10).status is LidarTimestampStatus.REPEATED
    assert tracker.repeated_count == 2
    third = tracker.observe(10)
    assert third.status is LidarTimestampStatus.STALE_LIMIT_EXCEEDED
    assert third.repeated_count == 3
    regression = tracker.observe(9)
    assert regression.status is LidarTimestampStatus.REGRESSION
    assert tracker.previous_accepted_timestamp == 10
    assert tracker.observe(11).status is LidarTimestampStatus.FRESH
    assert tracker.repeated_count == 0


def test_repeated_scan_is_policy_invalid_and_fault_threshold_is_exact() -> None:
    config = LidarFeatureConfig()
    timestamps = LidarTimestampTracker(3)
    faults = LidarFaultTracker(3)

    first = extract_lidar_features(_scan(_point(2.0, 0.0, 0.0), 10), config, timestamps)
    faults.record(first.lidar_valid == 1.0)
    assert faults.consecutive_invalid_scans == 0

    for expected in (1, 2, 3):
        repeated = extract_lidar_features(
            _scan(_point(2.0, 0.0, 0.0), 10), config, timestamps
        )
        faults.record(repeated.lidar_valid == 1.0)
        assert repeated.status is LidarValidityStatus.TIMESTAMP_INVALID
        assert faults.consecutive_invalid_scans == expected
    assert (
        repeated.diagnostics.timestamp_status
        is LidarTimestampStatus.STALE_LIMIT_EXCEEDED
    )
    assert faults.threshold_reached


def test_evidence_contains_schema_and_no_raw_cloud() -> None:
    result = extract_lidar_features(_scan(_point(2.0, 0.0, 0.0)), LidarFeatureConfig())
    evidence = extraction_evidence(result)

    assert evidence["schema_version"] == LIDAR_EXTRACTION_EVIDENCE_SCHEMA_VERSION
    assert evidence["feature_shape"] == [72]
    assert evidence["diagnostics"]["total_finite_point_count"] == 1
    assert "point_cloud" not in str(evidence)
