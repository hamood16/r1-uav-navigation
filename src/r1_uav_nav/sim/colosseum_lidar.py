"""Colosseum adapters and reports for M13.4 LiDAR features."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from r1_uav_nav.sim.colosseum_capabilities import (
    CapabilityProbeError,
    CollisionClassification,
    classify_collision_samples,
    inspect_active_lidar_settings,
    sample_collision_information,
    validate_grounded_preflight,
)
from r1_uav_nav.sim.lidar_features import (
    LIDAR_LIVE_REPORT_SCHEMA_VERSION,
    LidarFeatureConfig,
    LidarFeatureResult,
    LidarScanInput,
    LidarTimestampStatus,
    LidarTimestampTracker,
    extract_lidar_features,
    extraction_evidence,
)
from r1_uav_nav.sim.static_course import ValidatedCourse


@dataclass(frozen=True)
class GroundedLidarFeatureProbeConfig:
    """Read-only bounded grounded feature probe."""

    feature_config: LidarFeatureConfig
    scan_count: int = 20
    scan_interval_s: float = 0.2
    confirm_no_visible_collision: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.scan_count <= 100:
            raise ValueError("scan_count must be between 1 and 100")
        if (
            not math.isfinite(self.scan_interval_s)
            or not 0 <= self.scan_interval_s <= 5
        ):
            raise ValueError("scan_interval_s must be finite and between 0 and 5")
        if not self.confirm_no_visible_collision:
            raise ValueError(
                "grounded probe requires no-visible-collision confirmation"
            )


@dataclass(frozen=True)
class LidarLiveValidationReport:
    """Ignored M13.4 report without raw clouds or machine-specific paths."""

    schema_version: int
    run_id: str
    mode: str
    started_at: str
    completed_at: str
    success: bool
    interrupted: bool
    vehicle_name: str
    lidar_name: str
    configuration_digest: str
    feature_shape: tuple[int, ...]
    data: dict[str, Any]
    cleanup: dict[str, Any]
    errors: tuple[str, ...]
    limitations: tuple[str, ...]


def lidar_scan_input_from_data(lidar_data: Any) -> LidarScanInput:
    """Copy AirSim-style LiDAR data into an AirSim-independent record."""
    cloud = getattr(lidar_data, "point_cloud", None)
    point_cloud = tuple(cloud) if cloud is not None else None
    timestamp = getattr(lidar_data, "time_stamp", None)
    pose = getattr(lidar_data, "pose", None)
    position = _optional_vector(getattr(pose, "position", None))
    orientation = _optional_quaternion(getattr(pose, "orientation", None))
    return LidarScanInput(point_cloud, timestamp, position, orientation)


def read_lidar_features(
    client: Any,
    config: LidarFeatureConfig,
    timestamp_tracker: LidarTimestampTracker,
) -> LidarFeatureResult:
    """Read one exact named scan and extract fixed-size features."""
    lidar_data = client.getLidarData(config.lidar_name, config.vehicle_name)
    return extract_lidar_features(
        lidar_scan_input_from_data(lidar_data),
        config,
        timestamp_tracker,
    )


def probe_grounded_lidar_features(
    client: Any,
    client_module: ModuleType,
    config: GroundedLidarFeatureProbeConfig,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Validate fixed-size features while acquiring no simulator resources."""
    feature_config = config.feature_config
    verification = inspect_active_lidar_settings(
        client, feature_config.vehicle_name, feature_config.lidar_name
    )
    if not verification["profile_matches"]:
        raise CapabilityProbeError(
            "active settings do not match the exact M13.1 LiDAR profile"
        )
    state = client.getMultirotorState(vehicle_name=feature_config.vehicle_name)
    _, collision_samples = sample_collision_information(
        client,
        vehicle_name=feature_config.vehicle_name,
        sleep_fn=sleep_fn,
    )
    velocity = getattr(
        getattr(state, "kinematics_estimated", None),
        "linear_velocity",
        None,
    )
    speed = math.sqrt(
        float(velocity.x_val) ** 2
        + float(velocity.y_val) ** 2
        + float(velocity.z_val) ** 2
    )
    landed_type = getattr(client_module, "LandedState", None)
    landed_value = getattr(landed_type, "Landed", 0)
    assessment = classify_collision_samples(
        collision_samples,
        is_landed=getattr(state, "landed_state", None) == landed_value,
        measured_speed=speed,
        api_control_enabled=client.isApiControlEnabled(
            vehicle_name=feature_config.vehicle_name
        ),
        operator_confirmed_stable=config.confirm_no_visible_collision,
    )
    if assessment.classification not in {
        CollisionClassification.NO_COLLISION,
        CollisionClassification.EXPECTED_GROUND_CONTACT,
    }:
        raise CapabilityProbeError("grounded collision evidence is unsafe")
    position = validate_grounded_preflight(
        client,
        client_module,
        state,
        collision_assessment=assessment,
        operator_confirmed_stable=config.confirm_no_visible_collision,
        vehicle_name=feature_config.vehicle_name,
    )

    timestamps = LidarTimestampTracker(
        feature_config.maximum_repeated_timestamp_transitions
    )
    results: list[LidarFeatureResult] = []
    for index in range(config.scan_count):
        results.append(read_lidar_features(client, feature_config, timestamps))
        if index + 1 < config.scan_count and config.scan_interval_s:
            sleep_fn(config.scan_interval_s)
    valid_count = sum(result.lidar_valid == 1.0 for result in results)
    data = {
        "settings_profile_matches": True,
        "settings_comparisons": verification["comparisons"],
        "grounded_position": position,
        "collision_classification": assessment.classification.value,
        "collision_evidence_safe": True,
        "requested_scan_count": config.scan_count,
        "scan_count": len(results),
        "valid_scan_count": valid_count,
        "invalid_scan_count": len(results) - valid_count,
        "extractions": tuple(extraction_evidence(result) for result in results),
        "elevation_diagnostics_summary": _aggregate_elevation_diagnostics(
            results, feature_config
        ),
        "resources_acquired": False,
    }
    acceptance_checks = _grounded_acceptance_checks(data, results)
    data["acceptance_checks"] = acceptance_checks
    data["acceptance_failures"] = tuple(
        name for name, accepted in acceptance_checks.items() if not accepted
    )
    return data


def build_live_report(
    *,
    mode: str,
    config: LidarFeatureConfig,
    started_at: datetime,
    data: dict[str, Any],
    errors: tuple[str, ...] = (),
    interrupted: bool = False,
    cleanup: dict[str, Any] | None = None,
    feature_shape: tuple[int, ...] | None = None,
) -> LidarLiveValidationReport:
    """Create one bounded report with stable evidence schemas."""
    report_data = dict(data)
    cleanup_evidence = cleanup or {"resources_acquired": False}
    acceptance_failures = tuple(report_data.get("acceptance_failures", ()))
    if mode == "grounded":
        if not report_data:
            acceptance_failures = ("grounded_probe_evidence_available",)
        if cleanup_evidence.get("resources_acquired", False):
            acceptance_failures = (*acceptance_failures, "no_resources_acquired")
        acceptance_failures = tuple(dict.fromkeys(acceptance_failures))
        report_data["acceptance_failures"] = acceptance_failures
    return LidarLiveValidationReport(
        schema_version=LIDAR_LIVE_REPORT_SCHEMA_VERSION,
        run_id=uuid.uuid4().hex,
        mode=mode,
        started_at=started_at.astimezone(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        success=not errors and not interrupted and not acceptance_failures,
        interrupted=interrupted,
        vehicle_name=config.vehicle_name,
        lidar_name=config.lidar_name,
        configuration_digest=config.configuration_digest,
        feature_shape=feature_shape or (config.feature_count,),
        data=report_data,
        cleanup=cleanup_evidence,
        errors=errors,
        limitations=(
            "raw point clouds are intentionally omitted",
            "built-in Blocks geometry is not represented by M13.3 scene evidence",
            "physical simulator collision response is not proven",
        ),
    )


def _grounded_acceptance_checks(
    data: dict[str, Any], results: list[LidarFeatureResult]
) -> dict[str, bool]:
    allowed_timestamps = {
        LidarTimestampStatus.FIRST_VALID,
        LidarTimestampStatus.FRESH,
    }
    timestamp_sequence_valid = bool(results) and (
        results[0].diagnostics.timestamp_status is LidarTimestampStatus.FIRST_VALID
        and all(
            result.diagnostics.timestamp_status is LidarTimestampStatus.FRESH
            for result in results[1:]
        )
    )
    features_valid = all(
        result.features.shape == (72,)
        and bool(np.all(np.isfinite(result.features)))
        and bool(np.all((result.features >= 0.0) & (result.features <= 1.0)))
        for result in results
    )
    return {
        "settings_profile_matches": data.get("settings_profile_matches") is True,
        "collision_evidence_safe": data.get("collision_evidence_safe") is True,
        "requested_scan_count_collected": (
            data.get("scan_count") == data.get("requested_scan_count")
        ),
        "every_measured_scan_policy_valid": bool(results)
        and all(result.lidar_valid == 1.0 for result in results),
        "valid_scan_count_equals_scan_count": (
            data.get("valid_scan_count") == data.get("scan_count")
        ),
        "invalid_scan_count_zero": data.get("invalid_scan_count") == 0,
        "feature_shape_exactly_72": bool(results)
        and all(result.features.shape == (72,) for result in results),
        "features_finite_and_bounded": bool(results) and features_valid,
        "timestamps_first_valid_then_fresh": timestamp_sequence_valid,
        "no_timestamp_regression_or_stale_failure": all(
            result.diagnostics.timestamp_status in allowed_timestamps
            for result in results
        ),
        "no_resources_acquired": data.get("resources_acquired") is False,
    }


def _aggregate_elevation_diagnostics(
    results: list[LidarFeatureResult], config: LidarFeatureConfig
) -> dict[str, Any]:
    diagnostics = [result.diagnostics for result in results]
    minima = [
        item.observed_minimum_elevation_degrees
        for item in diagnostics
        if item.observed_minimum_elevation_degrees is not None
    ]
    maxima = [
        item.observed_maximum_elevation_degrees
        for item in diagnostics
        if item.observed_maximum_elevation_degrees is not None
    ]
    return {
        "configured_lower_elevation_degrees": float(
            config.elevation_band_edges_degrees[0]
        ),
        "configured_upper_elevation_degrees": float(
            config.elevation_band_edges_degrees[-1]
        ),
        "configured_elevation_boundary_tolerance_degrees": float(
            config.elevation_boundary_tolerance_degrees
        ),
        "observed_minimum_elevation_degrees": min(minima) if minima else None,
        "observed_maximum_elevation_degrees": max(maxima) if maxima else None,
        "below_configured_fov_count": sum(
            item.below_configured_fov_count for item in diagnostics
        ),
        "above_configured_fov_count": sum(
            item.above_configured_fov_count for item in diagnostics
        ),
        "maximum_below_fov_overshoot_degrees": max(
            (item.maximum_below_fov_overshoot_degrees for item in diagnostics),
            default=0.0,
        ),
        "maximum_above_fov_overshoot_degrees": max(
            (item.maximum_above_fov_overshoot_degrees for item in diagnostics),
            default=0.0,
        ),
        "in_fov_point_count": sum(item.in_fov_point_count for item in diagnostics),
        "total_finite_point_count": sum(
            item.total_finite_point_count for item in diagnostics
        ),
        "scans_with_below_configured_fov_points": sum(
            item.below_configured_fov_count > 0 for item in diagnostics
        ),
        "scans_with_above_configured_fov_points": sum(
            item.above_configured_fov_count > 0 for item in diagnostics
        ),
        "scan_count_with_elevation_data": len(minima),
    }


def save_lidar_live_report(
    report: LidarLiveValidationReport, output_path: str | Path
) -> Path:
    """Write one ignored JSON report."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_jsonable(asdict(report)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def build_easy_known_geometry_evidence(
    course: ValidatedCourse,
    config: LidarFeatureConfig,
) -> dict[str, Any]:
    """Lock the single attributable easy-seed comparison before live work."""
    if course.result.profile_id != "easy" or course.result.base_seed != 1100:
        raise ValueError("known-geometry check requires accepted easy seed 1100")
    obstacle = next(
        (
            item
            for item in course.scene.config.static_obstacles
            if item.name == "obstacle-000"
        ),
        None,
    )
    if obstacle is None:
        raise ValueError("easy seed 1100 does not contain obstacle-000")
    start = course.scene.start_anchor
    delta_x = obstacle.base_center.x - start.x
    delta_y = obstacle.base_center.y - start.y
    centre_z = obstacle.base_center.z - obstacle.dimensions.height / 2.0
    delta_z = centre_z - start.z
    angle = math.degrees(math.atan2(delta_y, delta_x))
    elevation = math.degrees(math.atan2(-delta_z, math.hypot(delta_x, delta_y)))
    sector = int(math.floor((((angle + 180.0) % 360.0) / 15.0)))
    edges = config.elevation_band_edges_degrees
    band = next(
        (
            index
            for index, (lower, upper) in enumerate(zip(edges, edges[1:], strict=False))
            if lower <= elevation < upper
            or (index == len(edges) - 2 and math.isclose(elevation, upper))
        ),
        None,
    )
    if band is None:
        raise ValueError("comparison obstacle centre is outside LiDAR elevation FOV")
    nearest = _distance_to_oriented_box(start, obstacle)
    if sector != 7 or band != 1 or not math.isclose(nearest, 5.14535, abs_tol=0.01):
        raise ValueError("accepted easy-seed known-geometry evidence changed")
    return {
        "profile_id": "easy",
        "base_seed": 1100,
        "comparison_object": "obstacle-000",
        "expected_horizontal_sector": 7,
        "expected_elevation_band": 1,
        "expected_flattened_feature_index": 31,
        "expected_nearest_surface_distance_m": nearest,
        "accepted_absolute_tolerance_m": 0.50,
        "clear_comparison_horizontal_sector": 14,
        "clear_comparison_elevation_band": 1,
        "clear_comparison_flattened_feature_index": 38,
        "limitations": (
            "Cube dimensions are operator-confirmed nominal",
            "built-in Blocks geometry may create nearer returns",
            "one scan does not validate every configured obstacle",
        ),
    }


def _optional_vector(value: Any) -> tuple[float, float, float] | None:
    try:
        return (float(value.x_val), float(value.y_val), float(value.z_val))
    except (AttributeError, TypeError, ValueError):
        return None


def _distance_to_oriented_box(start: Any, obstacle: Any) -> float:
    yaw = math.radians(-float(obstacle.yaw_degrees))
    delta_x = float(start.x - obstacle.base_center.x)
    delta_y = float(start.y - obstacle.base_center.y)
    local_x = math.cos(yaw) * delta_x - math.sin(yaw) * delta_y
    local_y = math.sin(yaw) * delta_x + math.cos(yaw) * delta_y
    centre_z = obstacle.base_center.z - obstacle.dimensions.height / 2.0
    local_z = float(start.z - centre_z)
    outside_x = max(abs(local_x) - obstacle.dimensions.width / 2.0, 0.0)
    outside_y = max(abs(local_y) - obstacle.dimensions.depth / 2.0, 0.0)
    outside_z = max(abs(local_z) - obstacle.dimensions.height / 2.0, 0.0)
    return math.sqrt(outside_x**2 + outside_y**2 + outside_z**2)


def _optional_quaternion(value: Any) -> tuple[float, float, float, float] | None:
    try:
        return (
            float(value.w_val),
            float(value.x_val),
            float(value.y_val),
            float(value.z_val),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value
