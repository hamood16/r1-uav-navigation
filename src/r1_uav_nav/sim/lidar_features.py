"""Pure fixed-size LiDAR feature extraction for M13.4."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

LIDAR_FEATURE_CONFIG_SCHEMA_VERSION = 1
LIDAR_EXTRACTION_EVIDENCE_SCHEMA_VERSION = 1
LIDAR_LIVE_REPORT_SCHEMA_VERSION = 1
LIDAR_FEATURE_COUNT = 72
LIDAR_OBSERVATION_SIZE = 83


class LidarFeatureError(ValueError):
    """Raised when LiDAR configuration or extraction input is invalid."""


class GroundFilterMode(str, Enum):
    """Supported point filtering policies."""

    NONE = "none"


class LidarValidityStatus(str, Enum):
    """Stable classification of one feature extraction attempt."""

    VALID = "valid"
    VALID_NO_OBSTACLE = "valid_no_obstacle"
    EMPTY = "empty"
    MALFORMED = "malformed"
    PARTIALLY_INVALID = "partially_invalid"
    ALL_INVALID = "all_invalid"
    RANGE_VIOLATION = "range_violation"
    OUTSIDE_ELEVATION_FOV = "outside_elevation_fov"
    TIMESTAMP_INVALID = "timestamp_invalid"


class LidarTimestampStatus(str, Enum):
    """One transition in the episode-local LiDAR timestamp sequence."""

    FIRST_VALID = "first_valid"
    FRESH = "fresh"
    REPEATED = "repeated"
    STALE_LIMIT_EXCEEDED = "stale_limit_exceeded"
    REGRESSION = "regression"
    STARTUP_PENDING = "startup_pending"

    @property
    def valid_for_policy(self) -> bool:
        return self in {self.FIRST_VALID, self.FRESH}


@dataclass(frozen=True)
class LidarFeatureConfig:
    """Strict, simulator-independent M13.4 feature configuration."""

    schema_version: int = LIDAR_FEATURE_CONFIG_SCHEMA_VERSION
    vehicle_name: str = "SimpleFlight"
    lidar_name: str = "LidarSensor1"
    coordinate_frame: str = "SensorLocalFrame"
    minimum_range_m: float = 0.25
    maximum_range_m: float = 20.0
    range_tolerance_m: float = 0.10
    horizontal_sector_count: int = 24
    horizontal_fov_degrees: tuple[float, float] = (-180.0, 180.0)
    elevation_band_edges_degrees: tuple[float, ...] = (-30.0, -15.0, 0.0, 10.0)
    elevation_boundary_tolerance_degrees: float = 0.0001
    ground_filter_mode: GroundFilterMode = GroundFilterMode.NONE
    self_hit_threshold_m: float = 0.25
    forward_window_degrees: tuple[float, float] = (-30.0, 30.0)
    left_window_degrees: tuple[float, float] = (-135.0, -45.0)
    right_window_degrees: tuple[float, float] = (45.0, 135.0)
    maximum_repeated_timestamp_transitions: int = 3
    maximum_consecutive_invalid_scans: int = 3
    reset_scan_attempts: int = 10
    reset_scan_interval_s: float = 0.2
    safety_zero_velocity_duration_s: float = 0.5
    danger_distance_m: float = 1.0
    caution_distance_m: float = 2.0

    def __post_init__(self) -> None:
        if self.schema_version != LIDAR_FEATURE_CONFIG_SCHEMA_VERSION:
            raise LidarFeatureError("unsupported LiDAR feature configuration schema")
        for name in ("vehicle_name", "lidar_name", "coordinate_frame"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LidarFeatureError(f"{name} must not be empty")
        if self.coordinate_frame != "SensorLocalFrame":
            raise LidarFeatureError("M13.4 requires SensorLocalFrame")
        if self.ground_filter_mode is not GroundFilterMode.NONE:
            raise LidarFeatureError("only ground_filter_mode=none is supported")
        _require_positive("minimum_range_m", self.minimum_range_m)
        _require_positive("maximum_range_m", self.maximum_range_m)
        _require_nonnegative("range_tolerance_m", self.range_tolerance_m)
        if self.minimum_range_m >= self.maximum_range_m:
            raise LidarFeatureError("minimum_range_m must be below maximum_range_m")
        if self.horizontal_sector_count != 24:
            raise LidarFeatureError("the shipped M13.4 profile requires 24 sectors")
        if tuple(self.horizontal_fov_degrees) != (-180.0, 180.0):
            raise LidarFeatureError("M13.4 requires full [-180, 180) horizontal FOV")
        if len(self.elevation_band_edges_degrees) != 4:
            raise LidarFeatureError("exactly three elevation bands are required")
        _require_strictly_increasing(
            "elevation_band_edges_degrees", self.elevation_band_edges_degrees
        )
        if tuple(self.elevation_band_edges_degrees) != (-30.0, -15.0, 0.0, 10.0):
            raise LidarFeatureError(
                "the shipped M13.4 profile requires elevation edges "
                "[-30, -15, 0, 10]"
            )
        _require_nonnegative(
            "elevation_boundary_tolerance_degrees",
            self.elevation_boundary_tolerance_degrees,
        )
        for name in (
            "self_hit_threshold_m",
            "reset_scan_interval_s",
            "safety_zero_velocity_duration_s",
            "danger_distance_m",
            "caution_distance_m",
        ):
            _require_positive(name, float(getattr(self, name)))
        if not 1 <= self.maximum_repeated_timestamp_transitions:
            raise LidarFeatureError(
                "maximum_repeated_timestamp_transitions must be positive"
            )
        if (
            self.maximum_consecutive_invalid_scans
            < self.maximum_repeated_timestamp_transitions
        ):
            raise LidarFeatureError(
                "maximum_consecutive_invalid_scans must be at least "
                "maximum_repeated_timestamp_transitions"
            )
        if not 1 <= self.reset_scan_attempts <= 100:
            raise LidarFeatureError("reset_scan_attempts must be between 1 and 100")
        for name in (
            "forward_window_degrees",
            "left_window_degrees",
            "right_window_degrees",
        ):
            bounds = getattr(self, name)
            if len(bounds) != 2:
                raise LidarFeatureError(f"{name} must contain two values")
            _require_strictly_increasing(name, bounds)
            if bounds[0] < -180.0 or bounds[1] > 180.0:
                raise LidarFeatureError(f"{name} must lie within [-180, 180]")
        if not (
            self.minimum_range_m
            <= self.danger_distance_m
            < self.caution_distance_m
            <= self.maximum_range_m
        ):
            raise LidarFeatureError("diagnostic thresholds are inconsistent")

    @property
    def elevation_band_count(self) -> int:
        return len(self.elevation_band_edges_degrees) - 1

    @property
    def feature_count(self) -> int:
        return self.horizontal_sector_count * self.elevation_band_count

    @property
    def configuration_digest(self) -> str:
        payload = json.dumps(
            _jsonable(asdict(self)), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LidarScanInput:
    """AirSim-independent raw scan data."""

    point_cloud: Sequence[Any] | None
    timestamp: int | None
    sensor_position: tuple[float, float, float] | None = None
    sensor_orientation: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class LidarTimestampEvidence:
    """Timestamp state after processing one scan."""

    status: LidarTimestampStatus
    timestamp: int | None
    previous_accepted_timestamp: int | None
    repeated_count: int


@dataclass
class LidarTimestampTracker:
    """Episode-local timestamp validator."""

    maximum_repeated_transitions: int
    previous_accepted_timestamp: int | None = None
    repeated_count: int = 0

    def __post_init__(self) -> None:
        if self.maximum_repeated_transitions <= 0:
            raise ValueError("maximum_repeated_transitions must be positive")

    def reset(self) -> None:
        self.previous_accepted_timestamp = None
        self.repeated_count = 0

    def observe(self, timestamp: int | None) -> LidarTimestampEvidence:
        try:
            value = int(timestamp) if timestamp is not None else None
        except (TypeError, ValueError, OverflowError):
            value = None
        if value is None or value <= 0:
            return self._evidence(LidarTimestampStatus.STARTUP_PENDING, value)
        if self.previous_accepted_timestamp is None:
            self.previous_accepted_timestamp = value
            self.repeated_count = 0
            return self._evidence(LidarTimestampStatus.FIRST_VALID, value)
        if value > self.previous_accepted_timestamp:
            self.previous_accepted_timestamp = value
            self.repeated_count = 0
            return self._evidence(LidarTimestampStatus.FRESH, value)
        if value < self.previous_accepted_timestamp:
            return self._evidence(LidarTimestampStatus.REGRESSION, value)
        self.repeated_count += 1
        status = (
            LidarTimestampStatus.STALE_LIMIT_EXCEEDED
            if self.repeated_count >= self.maximum_repeated_transitions
            else LidarTimestampStatus.REPEATED
        )
        return self._evidence(status, value)

    def _evidence(
        self, status: LidarTimestampStatus, timestamp: int | None
    ) -> LidarTimestampEvidence:
        return LidarTimestampEvidence(
            status,
            timestamp,
            self.previous_accepted_timestamp,
            self.repeated_count,
        )


@dataclass
class LidarFaultTracker:
    """Count consecutive policy-invalid scans."""

    maximum_consecutive_invalid_scans: int
    consecutive_invalid_scans: int = 0

    def __post_init__(self) -> None:
        if self.maximum_consecutive_invalid_scans <= 0:
            raise ValueError("maximum_consecutive_invalid_scans must be positive")

    def reset(self) -> None:
        self.consecutive_invalid_scans = 0

    def record(self, valid: bool) -> int:
        if valid:
            self.consecutive_invalid_scans = 0
        else:
            self.consecutive_invalid_scans += 1
        return self.consecutive_invalid_scans

    @property
    def threshold_reached(self) -> bool:
        return self.consecutive_invalid_scans >= self.maximum_consecutive_invalid_scans


@dataclass(frozen=True)
class LidarDiagnostics:
    """Non-policy evidence from one extraction."""

    raw_coordinate_count: int
    parsed_point_count: int
    valid_point_count: int
    rejected_point_count: int
    clipped_below_minimum_count: int
    clipped_above_maximum_count: int
    near_self_hit_count: int
    observed_minimum_elevation_degrees: float | None
    observed_maximum_elevation_degrees: float | None
    below_configured_fov_count: int
    above_configured_fov_count: int
    maximum_below_fov_overshoot_degrees: float
    maximum_above_fov_overshoot_degrees: float
    in_fov_point_count: int
    total_finite_point_count: int
    configured_lower_elevation_degrees: float
    configured_upper_elevation_degrees: float
    configured_elevation_boundary_tolerance_degrees: float
    sector_point_counts: tuple[int, ...]
    band_point_counts: tuple[int, ...]
    nearest_overall_m: float | None
    nearest_forward_m: float | None
    minimum_left_clearance_m: float | None
    minimum_right_clearance_m: float | None
    timestamp: int | None
    timestamp_status: LidarTimestampStatus
    repeated_timestamp_count: int
    pose_valid: bool
    pose_error: str | None
    sensor_position: tuple[float, float, float] | None
    sensor_orientation: tuple[float, float, float, float] | None
    failure_reason: str | None


@dataclass(frozen=True)
class LidarFeatureResult:
    """Fixed-size policy features and structured diagnostics."""

    schema_version: int
    status: LidarValidityStatus
    features: np.ndarray
    lidar_valid: float
    diagnostics: LidarDiagnostics

    def __post_init__(self) -> None:
        if self.features.shape != (LIDAR_FEATURE_COUNT,):
            raise ValueError("LiDAR features must have shape (72,)")
        if self.features.dtype != np.float32:
            raise ValueError("LiDAR features must use float32")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("LiDAR features must be finite")
        if np.any(self.features < 0.0) or np.any(self.features > 1.0):
            raise ValueError("LiDAR features must be bounded by [0, 1]")


def extract_lidar_features(
    scan: LidarScanInput,
    config: LidarFeatureConfig,
    timestamp_tracker: LidarTimestampTracker | None = None,
) -> LidarFeatureResult:
    """Convert one strict SensorLocalFrame scan into 72 nearest ranges."""
    tracker = timestamp_tracker or LidarTimestampTracker(
        config.maximum_repeated_timestamp_transitions
    )
    timestamp = tracker.observe(scan.timestamp)
    pose_valid, pose_error = _validate_pose(scan)
    fallback = np.ones(config.feature_count, dtype=np.float32)

    values, point_error, raw_count, parsed_count = _coerce_points(scan.point_cloud)
    if point_error is not None:
        status, rejected = point_error
        finite_count = len(values) if values is not None else 0
        finite_elevation = (
            np.degrees(
                np.arctan2(
                    -values[:, 2],
                    np.hypot(values[:, 0], values[:, 1]),
                )
            )
            if finite_count
            else None
        )
        return _result(
            status,
            fallback,
            0.0,
            config,
            timestamp,
            raw_count,
            parsed_count,
            finite_count,
            rejected,
            pose_valid,
            pose_error,
            scan,
            failure_reason=status.value,
            elevation=finite_elevation,
        )

    assert values is not None
    ranges = np.linalg.norm(values, axis=1)
    horizontal = np.degrees(np.arctan2(values[:, 1], values[:, 0]))
    radial_xy = np.hypot(values[:, 0], values[:, 1])
    elevation = np.degrees(np.arctan2(-values[:, 2], radial_xy))
    if not np.all(np.isfinite(ranges)):
        return _result(
            LidarValidityStatus.ALL_INVALID,
            fallback,
            0.0,
            config,
            timestamp,
            raw_count,
            parsed_count,
            0,
            parsed_count,
            pose_valid,
            pose_error,
            scan,
            failure_reason="point ranges are non-finite",
            elevation=elevation,
        )
    range_limit = config.maximum_range_m + config.range_tolerance_m
    if np.any(ranges > range_limit):
        return _result(
            LidarValidityStatus.RANGE_VIOLATION,
            fallback,
            0.0,
            config,
            timestamp,
            raw_count,
            parsed_count,
            parsed_count,
            0,
            pose_valid,
            pose_error,
            scan,
            failure_reason="point exceeds maximum_range_m + range_tolerance_m",
            elevation=elevation,
        )

    lower = config.elevation_band_edges_degrees[0]
    upper = config.elevation_band_edges_degrees[-1]
    boundary_tolerance = config.elevation_boundary_tolerance_degrees
    if np.any(elevation < lower - boundary_tolerance) or np.any(
        elevation > upper + boundary_tolerance
    ):
        return _result(
            LidarValidityStatus.OUTSIDE_ELEVATION_FOV,
            fallback,
            0.0,
            config,
            timestamp,
            raw_count,
            parsed_count,
            parsed_count,
            0,
            pose_valid,
            pose_error,
            scan,
            failure_reason="point elevation is outside configured sensor FOV",
            elevation=elevation,
        )
    if not timestamp.status.valid_for_policy:
        return _result(
            LidarValidityStatus.TIMESTAMP_INVALID,
            fallback,
            0.0,
            config,
            timestamp,
            raw_count,
            parsed_count,
            parsed_count,
            0,
            pose_valid,
            pose_error,
            scan,
            failure_reason=timestamp.status.value,
            elevation=elevation,
        )

    clipped = np.clip(ranges, config.minimum_range_m, config.maximum_range_m)
    band_elevation = np.clip(elevation, lower, upper)
    wrapped = ((horizontal + 180.0) % 360.0) - 180.0
    horizontal_indices = np.floor(
        (wrapped + 180.0) / (360.0 / config.horizontal_sector_count)
    ).astype(np.int64)
    horizontal_indices = np.clip(
        horizontal_indices, 0, config.horizontal_sector_count - 1
    )
    edges = np.asarray(config.elevation_band_edges_degrees, dtype=np.float64)
    band_indices = np.searchsorted(edges, band_elevation, side="right") - 1
    band_indices = np.clip(band_indices, 0, config.elevation_band_count - 1)
    flat_indices = band_indices * config.horizontal_sector_count + horizontal_indices

    nearest = np.full(config.feature_count, config.maximum_range_m, dtype=np.float64)
    np.minimum.at(nearest, flat_indices, clipped)
    sector_counts = np.bincount(flat_indices, minlength=config.feature_count).astype(
        np.int64
    )
    normalized = (
        (nearest - config.minimum_range_m)
        / (config.maximum_range_m - config.minimum_range_m)
    ).astype(np.float32)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    status = (
        LidarValidityStatus.VALID_NO_OBSTACLE
        if bool(np.all(clipped >= config.maximum_range_m))
        else LidarValidityStatus.VALID
    )
    return _result(
        status,
        normalized,
        1.0,
        config,
        timestamp,
        raw_count,
        parsed_count,
        parsed_count,
        0,
        pose_valid,
        pose_error,
        scan,
        ranges=ranges,
        clipped=clipped,
        horizontal=wrapped,
        elevation=elevation,
        sector_counts=tuple(int(value) for value in sector_counts),
    )


def load_lidar_feature_config(path: str | Path) -> LidarFeatureConfig:
    """Load the strict versioned M13.4 YAML configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise LidarFeatureError("LiDAR feature config must contain a mapping")
    allowed = {
        "schema_version",
        "vehicle_name",
        "lidar_name",
        "coordinate_frame",
        "minimum_range_m",
        "maximum_range_m",
        "range_tolerance_m",
        "horizontal_sector_count",
        "horizontal_fov_degrees",
        "elevation_band_edges_degrees",
        "elevation_boundary_tolerance_degrees",
        "ground_filter",
        "self_hit_threshold_m",
        "forward_window_degrees",
        "left_window_degrees",
        "right_window_degrees",
        "maximum_repeated_timestamp_transitions",
        "maximum_consecutive_invalid_scans",
        "reset_scan_attempts",
        "reset_scan_interval_s",
        "safety_zero_velocity_duration_s",
        "diagnostics",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise LidarFeatureError(f"unknown LiDAR config keys: {sorted(unknown)}")
    ground = _strict_mapping(raw.get("ground_filter", {}), "ground_filter")
    _reject_unknown(ground, {"mode"}, "ground_filter")
    diagnostics = _strict_mapping(raw.get("diagnostics", {}), "diagnostics")
    _reject_unknown(
        diagnostics, {"danger_distance_m", "caution_distance_m"}, "diagnostics"
    )
    return LidarFeatureConfig(
        schema_version=_integer(raw, "schema_version"),
        vehicle_name=_string(raw, "vehicle_name"),
        lidar_name=_string(raw, "lidar_name"),
        coordinate_frame=_string(raw, "coordinate_frame"),
        minimum_range_m=_number(raw, "minimum_range_m"),
        maximum_range_m=_number(raw, "maximum_range_m"),
        range_tolerance_m=_number(raw, "range_tolerance_m"),
        horizontal_sector_count=_integer(raw, "horizontal_sector_count"),
        horizontal_fov_degrees=_number_tuple(raw, "horizontal_fov_degrees", length=2),
        elevation_band_edges_degrees=_number_tuple(
            raw, "elevation_band_edges_degrees", length=4
        ),
        elevation_boundary_tolerance_degrees=_strict_number(
            raw, "elevation_boundary_tolerance_degrees"
        ),
        ground_filter_mode=GroundFilterMode(
            str(ground.get("mode", GroundFilterMode.NONE.value))
        ),
        self_hit_threshold_m=_number(raw, "self_hit_threshold_m"),
        forward_window_degrees=_number_tuple(raw, "forward_window_degrees", length=2),
        left_window_degrees=_number_tuple(raw, "left_window_degrees", length=2),
        right_window_degrees=_number_tuple(raw, "right_window_degrees", length=2),
        maximum_repeated_timestamp_transitions=_integer(
            raw, "maximum_repeated_timestamp_transitions"
        ),
        maximum_consecutive_invalid_scans=_integer(
            raw, "maximum_consecutive_invalid_scans"
        ),
        reset_scan_attempts=_integer(raw, "reset_scan_attempts"),
        reset_scan_interval_s=_number(raw, "reset_scan_interval_s"),
        safety_zero_velocity_duration_s=_number(raw, "safety_zero_velocity_duration_s"),
        danger_distance_m=float(diagnostics.get("danger_distance_m", 1.0)),
        caution_distance_m=float(diagnostics.get("caution_distance_m", 2.0)),
    )


def extraction_evidence(result: LidarFeatureResult) -> dict[str, Any]:
    """Build JSON-safe evidence without retaining the raw point cloud."""
    return {
        "schema_version": LIDAR_EXTRACTION_EVIDENCE_SCHEMA_VERSION,
        "status": result.status.value,
        "feature_shape": list(result.features.shape),
        "lidar_valid": result.lidar_valid,
        "feature_minimum": float(np.min(result.features)),
        "feature_maximum": float(np.max(result.features)),
        "feature_mean": float(np.mean(result.features)),
        "sector_values": [float(value) for value in result.features],
        "diagnostics": _jsonable(asdict(result.diagnostics)),
    }


def _coerce_points(
    point_cloud: Sequence[Any] | None,
) -> tuple[
    np.ndarray | None,
    tuple[LidarValidityStatus, int] | None,
    int,
    int,
]:
    if point_cloud is None:
        return None, (LidarValidityStatus.EMPTY, 0), 0, 0
    try:
        raw = tuple(point_cloud)
    except TypeError:
        return None, (LidarValidityStatus.MALFORMED, 0), 0, 0
    raw_count = len(raw)
    if raw_count == 0:
        return None, (LidarValidityStatus.EMPTY, 0), 0, 0
    if raw_count % 3:
        return None, (LidarValidityStatus.MALFORMED, 0), raw_count, raw_count // 3
    try:
        values = np.asarray(raw, dtype=np.float64).reshape((-1, 3)).copy()
    except (TypeError, ValueError, OverflowError):
        return (
            None,
            (LidarValidityStatus.MALFORMED, raw_count // 3),
            raw_count,
            raw_count // 3,
        )
    finite_rows = np.all(np.isfinite(values), axis=1)
    if not np.all(finite_rows):
        rejected = int(np.count_nonzero(~finite_rows))
        status = (
            LidarValidityStatus.ALL_INVALID
            if rejected == len(values)
            else LidarValidityStatus.PARTIALLY_INVALID
        )
        return values[finite_rows], (status, rejected), raw_count, len(values)
    return values, None, raw_count, len(values)


def _result(
    status: LidarValidityStatus,
    features: np.ndarray,
    lidar_valid: float,
    config: LidarFeatureConfig,
    timestamp: LidarTimestampEvidence,
    raw_count: int,
    parsed_count: int,
    valid_count: int,
    rejected_count: int,
    pose_valid: bool,
    pose_error: str | None,
    scan: LidarScanInput,
    *,
    failure_reason: str | None = None,
    ranges: np.ndarray | None = None,
    clipped: np.ndarray | None = None,
    horizontal: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    sector_counts: tuple[int, ...] | None = None,
) -> LidarFeatureResult:
    ranges = np.asarray((), dtype=np.float64) if ranges is None else ranges
    clipped = np.asarray((), dtype=np.float64) if clipped is None else clipped
    horizontal = np.asarray((), dtype=np.float64) if horizontal is None else horizontal
    elevation = np.asarray((), dtype=np.float64) if elevation is None else elevation
    counts = sector_counts or (0,) * config.feature_count
    band_counts = tuple(
        sum(
            counts[
                band
                * config.horizontal_sector_count : (band + 1)
                * config.horizontal_sector_count
            ]
        )
        for band in range(config.elevation_band_count)
    )
    diagnostics = LidarDiagnostics(
        raw_coordinate_count=raw_count,
        parsed_point_count=parsed_count,
        valid_point_count=valid_count,
        rejected_point_count=rejected_count,
        clipped_below_minimum_count=int(
            np.count_nonzero(ranges < config.minimum_range_m)
        ),
        clipped_above_maximum_count=int(
            np.count_nonzero(ranges > config.maximum_range_m)
        ),
        near_self_hit_count=int(np.count_nonzero(ranges < config.self_hit_threshold_m)),
        observed_minimum_elevation_degrees=(
            float(np.min(elevation)) if elevation.size else None
        ),
        observed_maximum_elevation_degrees=(
            float(np.max(elevation)) if elevation.size else None
        ),
        below_configured_fov_count=int(
            np.count_nonzero(elevation < config.elevation_band_edges_degrees[0])
        ),
        above_configured_fov_count=int(
            np.count_nonzero(elevation > config.elevation_band_edges_degrees[-1])
        ),
        maximum_below_fov_overshoot_degrees=(
            float(
                np.max(
                    config.elevation_band_edges_degrees[0]
                    - elevation[elevation < config.elevation_band_edges_degrees[0]]
                )
            )
            if np.any(elevation < config.elevation_band_edges_degrees[0])
            else 0.0
        ),
        maximum_above_fov_overshoot_degrees=(
            float(
                np.max(
                    elevation[elevation > config.elevation_band_edges_degrees[-1]]
                    - config.elevation_band_edges_degrees[-1]
                )
            )
            if np.any(elevation > config.elevation_band_edges_degrees[-1])
            else 0.0
        ),
        in_fov_point_count=int(
            np.count_nonzero(
                (elevation >= config.elevation_band_edges_degrees[0])
                & (elevation <= config.elevation_band_edges_degrees[-1])
            )
        ),
        total_finite_point_count=int(elevation.size),
        configured_lower_elevation_degrees=float(
            config.elevation_band_edges_degrees[0]
        ),
        configured_upper_elevation_degrees=float(
            config.elevation_band_edges_degrees[-1]
        ),
        configured_elevation_boundary_tolerance_degrees=float(
            config.elevation_boundary_tolerance_degrees
        ),
        sector_point_counts=counts,
        band_point_counts=band_counts,
        nearest_overall_m=float(np.min(clipped)) if clipped.size else None,
        nearest_forward_m=_minimum_in_window(
            clipped, horizontal, config.forward_window_degrees
        ),
        minimum_left_clearance_m=_minimum_in_window(
            clipped, horizontal, config.left_window_degrees
        ),
        minimum_right_clearance_m=_minimum_in_window(
            clipped, horizontal, config.right_window_degrees
        ),
        timestamp=timestamp.timestamp,
        timestamp_status=timestamp.status,
        repeated_timestamp_count=timestamp.repeated_count,
        pose_valid=pose_valid,
        pose_error=pose_error,
        sensor_position=scan.sensor_position if pose_valid else None,
        sensor_orientation=scan.sensor_orientation if pose_valid else None,
        failure_reason=failure_reason,
    )
    return LidarFeatureResult(
        LIDAR_EXTRACTION_EVIDENCE_SCHEMA_VERSION,
        status,
        features.astype(np.float32, copy=False),
        lidar_valid,
        diagnostics,
    )


def _minimum_in_window(
    ranges: np.ndarray,
    horizontal: np.ndarray,
    bounds: tuple[float, float],
) -> float | None:
    if not ranges.size:
        return None
    selected = ranges[(horizontal >= bounds[0]) & (horizontal <= bounds[1])]
    return float(np.min(selected)) if selected.size else None


def _validate_pose(scan: LidarScanInput) -> tuple[bool, str | None]:
    if scan.sensor_position is None or scan.sensor_orientation is None:
        return False, "sensor pose is unavailable"
    if len(scan.sensor_position) != 3 or len(scan.sensor_orientation) != 4:
        return False, "sensor pose has an invalid shape"
    values = (*scan.sensor_position, *scan.sensor_orientation)
    try:
        finite = all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        finite = False
    return (True, None) if finite else (False, "sensor pose is non-finite")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_positive(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LidarFeatureError(f"{name} must be a finite number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise LidarFeatureError(f"{name} must be finite and positive")


def _require_nonnegative(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LidarFeatureError(f"{name} must be a finite number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise LidarFeatureError(f"{name} must be finite and non-negative")


def _require_strictly_increasing(name: str, values: Sequence[float]) -> None:
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise LidarFeatureError(f"{name} must contain finite values")
    if any(
        current <= previous
        for previous, current in zip(converted, converted[1:], strict=False)
    ):
        raise LidarFeatureError(f"{name} must be strictly increasing")


def _strict_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LidarFeatureError(f"{name} must contain a mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise LidarFeatureError(f"unknown {name} keys: {sorted(unknown)}")


def _number(mapping: Mapping[str, Any], name: str) -> float:
    try:
        value = float(mapping[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise LidarFeatureError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise LidarFeatureError(f"{name} must be finite")
    return value


def _strict_number(mapping: Mapping[str, Any], name: str) -> float:
    value = mapping.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LidarFeatureError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise LidarFeatureError(f"{name} must be finite")
    return converted


def _integer(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LidarFeatureError(f"{name} must be an integer")
    return value


def _string(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise LidarFeatureError(f"{name} must be a non-empty string")
    return value


def _number_tuple(
    mapping: Mapping[str, Any], name: str, *, length: int
) -> tuple[float, ...]:
    value = mapping.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise LidarFeatureError(f"{name} must contain {length} numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise LidarFeatureError(f"{name} must contain numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise LidarFeatureError(f"{name} must contain finite numbers")
    return result
