"""Typed, simulator-independent Colosseum capability probe helpers."""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

from r1_uav_nav.sim.colosseum_client import (
    CleanupState,
    ColosseumClientError,
    import_colosseum_client_module,
)
from r1_uav_nav.sim.waypoint_navigation import (
    Position3D,
    calculate_position_error,
    extract_position_from_state,
)

DEFAULT_CLIENT_MODULE = "airsim"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 41451
DEFAULT_RPC_TIMEOUT = 30.0
DEFAULT_REPORTS_DIR = Path("results/reports/m13")
MAX_MARKER_HOLD_SECONDS = 15.0
MAX_MUTATION_HOLD_SECONDS = 10.0
PROBE_OBJECT_PREFIX = "r1_uav_m13_probe_"
GROUNDED_SPEED_TOLERANCE = 0.1
COLLISION_SAMPLE_COUNT = 3
COLLISION_SAMPLE_INTERVAL = 0.1
GROUND_CONTACT_MAX_PENETRATION = 0.15
GROUND_CONTACT_HORIZONTAL_TOLERANCE = 1.0
GROUND_CONTACT_VERTICAL_TOLERANCE = 0.25
GROUND_NORMAL_MIN_UPWARD_COMPONENT = 0.5
MAX_LIDAR_WARM_UP_ATTEMPTS = 10
MAX_LIDAR_WARM_UP_SECONDS = 2.0
LIDAR_RANGE_TOLERANCE = 0.10
NEAR_FIELD_THRESHOLDS = (0.05, 0.10, 0.25)
MAX_LIDAR_VISUALIZATION_HOLD_SECONDS = 15.0
MAX_LIDAR_VISUALIZATION_POINTS = 2000
MAX_LIDAR_VISUALIZATION_RAYS = 64


class CapabilityProbeError(ColosseumClientError):
    """Raised when a capability probe cannot continue safely."""


class CapabilityStatus(str, Enum):
    """Status vocabulary shared by static and live capability evidence."""

    CLIENT_METHOD_ABSENT = "client_method_absent"
    CLIENT_METHOD_PRESENT_NOT_LIVE_TESTED = "client_method_present_not_live_tested"
    RPC_SUCCEEDED = "rpc_succeeded"
    RPC_FAILED = "rpc_failed"
    RPC_TIMED_OUT = "rpc_timed_out"
    UNSUPPORTED_BY_BLOCKS = "unsupported_by_blocks"
    SUPPORTED_WITH_LIMITATIONS = "supported_with_limitations"
    REQUIRES_LOCAL_CONFIGURATION = "requires_local_configuration"
    SKIPPED_NOT_AUTHORIZED = "skipped_not_authorized"
    INCONCLUSIVE = "inconclusive"


class EvidenceLevel(str, Enum):
    """Strength of evidence associated with a capability observation."""

    STATIC_CLIENT = "static_client"
    LIVE_RPC = "live_rpc"
    PRACTICAL_BEHAVIOR = "practical_behavior"


class CollisionClassification(str, Enum):
    """Conservative interpretation of repeated read-only collision samples."""

    NO_COLLISION = "no_collision"
    EXPECTED_GROUND_CONTACT = "expected_ground_contact"
    ACTIVE_OR_UNSAFE_COLLISION = "active_or_unsafe_collision"
    INCONCLUSIVE_COLLISION = "inconclusive_collision"


class SelfHitClassification(str, Enum):
    """Conservative interpretation of repeated near-field LiDAR returns."""

    NO_EVIDENT_SELF_HIT = "no_evident_self_hit"
    POSSIBLE_SELF_HIT = "possible_self_hit"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CapabilityObservation:
    """One capability result at one evidence level."""

    capability: str
    client_method: str | None
    evidence_level: EvidenceLevel
    status: CapabilityStatus
    detail: str
    duration_seconds: float | None = None
    operator_confirmation: str = "not_required"


@dataclass(frozen=True)
class ConnectionProbeConfig:
    """Connection values for explicitly invoked live probes."""

    client_module: str = DEFAULT_CLIENT_MODULE
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    rpc_timeout: float = DEFAULT_RPC_TIMEOUT
    vehicle_name: str = ""

    def __post_init__(self) -> None:
        if not self.client_module.strip():
            raise ValueError("client_module must not be empty")
        if not self.host.strip():
            raise ValueError("host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        _require_positive_finite("rpc_timeout", self.rpc_timeout)


@dataclass(frozen=True)
class SceneSurveyConfig:
    """Bounded read-only scene survey configuration."""

    connection: ConnectionProbeConfig = field(default_factory=ConnectionProbeConfig)
    object_regex: str = ".*"
    max_objects: int = 100
    confirm_no_visible_collision: bool = False

    def __post_init__(self) -> None:
        if not self.object_regex:
            raise ValueError("object_regex must not be empty")
        if not 1 <= self.max_objects <= 500:
            raise ValueError("max_objects must be between 1 and 500")


@dataclass(frozen=True)
class SceneMutationConfig:
    """Settings for one opt-in temporary-object lifecycle."""

    asset_name: str
    allow_scene_mutation: bool = False
    confirm_spawn_area_clear: bool = False
    confirm_vehicle_disarmed: bool = False
    spawn_offset: tuple[float, float, float] = (3.0, 3.0, -1.0)
    initial_scale: float = 0.25
    moved_scale: float = 0.4
    move_distance: float = 0.5
    segmentation_id: int = 120
    material_name: str | None = None
    mutation_hold_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.asset_name.strip():
            raise ValueError("asset_name must not be empty")
        _validate_vector("spawn_offset", self.spawn_offset)
        if math.dist((0.0, 0.0, 0.0), self.spawn_offset) > 10.0:
            raise ValueError("spawn_offset must remain within 10 metres")
        _require_positive_finite("initial_scale", self.initial_scale)
        _require_positive_finite("moved_scale", self.moved_scale)
        _require_positive_finite("move_distance", self.move_distance)
        if self.initial_scale > 5.0 or self.moved_scale > 5.0:
            raise ValueError("probe object scales must not exceed 5")
        if self.move_distance > 2.0:
            raise ValueError("move_distance must not exceed 2 metres")
        if not 0 <= self.segmentation_id <= 255:
            raise ValueError("segmentation_id must be between 0 and 255")
        validate_mutation_hold_seconds(self.mutation_hold_seconds)
        validate_material_name(self.material_name)


@dataclass(frozen=True)
class AirborneProbeConfig:
    """Measured-ground safety limits for explicitly authorized flight."""

    vehicle_name: str
    allow_flight: bool = False
    confirm_clear_airspace: bool = False
    confirm_no_visible_collision: bool = False
    confirm_grounded_lidar_passed: bool = False
    anchor_altitude: float = 2.0
    min_ground_clearance: float = 1.0
    anchor_velocity: float = 0.5
    movement_timeout: float = 20.0
    movement_tolerance: float = 0.75
    workspace_xy_limit: float = 3.0
    workspace_z_limit: float = 1.0

    def __post_init__(self) -> None:
        if not self.vehicle_name.strip():
            raise ValueError("vehicle_name must not be empty")
        for name in (
            "anchor_altitude",
            "min_ground_clearance",
            "anchor_velocity",
            "movement_timeout",
            "movement_tolerance",
            "workspace_xy_limit",
            "workspace_z_limit",
        ):
            _require_positive_finite(name, float(getattr(self, name)))
        if self.anchor_altitude < self.min_ground_clearance:
            raise ValueError("anchor_altitude must preserve minimum ground clearance")


@dataclass(frozen=True)
class LidarProbeConfig:
    """Bounded raw LiDAR validation configuration."""

    airborne: AirborneProbeConfig
    lidar_name: str
    scan_count: int = 20
    scan_interval: float = 0.2
    stale_threshold: int = 3
    warm_up_attempts: int = 10
    warm_up_interval: float = 0.2
    settle_interval: float = 0.5
    coordinate_frame_experiment: bool = False
    allow_coordinate_motion: bool = False
    yaw_delta_degrees: float = 45.0
    yaw_return_tolerance_degrees: float = 5.0
    coordinate_scan_attempts: int = 5
    visualize_lidar: bool = False
    allow_marker_flush: bool = False
    visualization_hold_seconds: float = 8.0
    visualization_max_points: int = MAX_LIDAR_VISUALIZATION_POINTS
    visualization_max_rays: int = MAX_LIDAR_VISUALIZATION_RAYS

    def __post_init__(self) -> None:
        if not self.lidar_name.strip():
            raise ValueError("lidar_name must not be empty")
        if not 2 <= self.scan_count <= 100:
            raise ValueError("scan_count must be between 2 and 100")
        _require_nonnegative_finite("scan_interval", self.scan_interval)
        if self.scan_interval > 5.0:
            raise ValueError("scan_interval must not exceed 5 seconds")
        if not 0 <= self.stale_threshold <= 3:
            raise ValueError("stale_threshold must be between 0 and 3")
        if not 1 <= self.warm_up_attempts <= MAX_LIDAR_WARM_UP_ATTEMPTS:
            raise ValueError(
                f"warm_up_attempts must be between 1 and "
                f"{MAX_LIDAR_WARM_UP_ATTEMPTS}"
            )
        _require_nonnegative_finite("warm_up_interval", self.warm_up_interval)
        if self.warm_up_attempts * self.warm_up_interval > MAX_LIDAR_WARM_UP_SECONDS:
            raise ValueError(
                "warm_up_attempts * warm_up_interval must not exceed 2 seconds"
            )
        _require_nonnegative_finite("settle_interval", self.settle_interval)
        if self.settle_interval > 2.0:
            raise ValueError("settle_interval must not exceed 2 seconds")
        _require_positive_finite("yaw_delta_degrees", self.yaw_delta_degrees)
        if self.yaw_delta_degrees > 90.0:
            raise ValueError("yaw_delta_degrees must not exceed 90 degrees")
        if self.coordinate_frame_experiment and not self.allow_coordinate_motion:
            raise ValueError(
                "coordinate-frame experiment requires allow_coordinate_motion"
            )
        _require_positive_finite(
            "yaw_return_tolerance_degrees", self.yaw_return_tolerance_degrees
        )
        if self.yaw_return_tolerance_degrees > 15.0:
            raise ValueError("yaw_return_tolerance_degrees must not exceed 15")
        if not 1 <= self.coordinate_scan_attempts <= 20:
            raise ValueError("coordinate_scan_attempts must be between 1 and 20")
        validate_lidar_visualization_hold_seconds(self.visualization_hold_seconds)
        if not 1 <= self.visualization_max_points <= MAX_LIDAR_VISUALIZATION_POINTS:
            raise ValueError(
                "visualization_max_points must be between 1 and "
                f"{MAX_LIDAR_VISUALIZATION_POINTS}"
            )
        if not 0 <= self.visualization_max_rays <= MAX_LIDAR_VISUALIZATION_RAYS:
            raise ValueError(
                "visualization_max_rays must be between 0 and "
                f"{MAX_LIDAR_VISUALIZATION_RAYS}"
            )
        if self.visualize_lidar and not self.allow_marker_flush:
            raise ValueError(
                "LiDAR visualization requires explicit marker-flush authorization"
            )


@dataclass(frozen=True)
class LidarSettingsProfile:
    """Expected M13.1 vehicle and LiDAR settings."""

    vehicle_name: str
    lidar_name: str
    vehicle_fields: tuple[tuple[str, Any], ...]
    sensor_fields: tuple[tuple[str, Any], ...]


M13_LIDAR_PROVISIONAL_PROFILE = LidarSettingsProfile(
    vehicle_name="SimpleFlight",
    lidar_name="LidarSensor1",
    vehicle_fields=(
        ("VehicleType", "SimpleFlight"),
        ("AutoCreate", True),
        ("DefaultVehicleState", "Inactive"),
    ),
    sensor_fields=(
        ("SensorType", 6),
        ("Enabled", True),
        ("NumberOfChannels", 16),
        ("Range", 20),
        ("PointsPerSecond", 100000),
        ("RotationsPerSecond", 10),
        ("HorizontalFOVStart", 0),
        ("HorizontalFOVEnd", 359),
        ("VerticalFOVUpper", 10),
        ("VerticalFOVLower", -30),
        ("X", 0),
        ("Y", 0),
        ("Z", 0),
        ("Roll", 0),
        ("Pitch", 0),
        ("Yaw", 0),
        ("DrawDebugPoints", False),
        ("DataFrame", "SensorLocalFrame"),
        ("ExternalController", False),
    ),
)


@dataclass(frozen=True)
class GroundedLidarProbeConfig:
    """Strict grounded LiDAR validation configuration."""

    vehicle_name: str
    lidar_name: str
    scan_count: int = 20
    scan_interval: float = 0.2
    stale_threshold: int = 3
    warm_up_attempts: int = 10
    warm_up_interval: float = 0.2
    confirm_no_visible_collision: bool = False

    def __post_init__(self) -> None:
        if not self.vehicle_name.strip():
            raise ValueError("vehicle_name must not be empty")
        if not self.lidar_name.strip():
            raise ValueError("lidar_name must not be empty")
        if not 2 <= self.scan_count <= 100:
            raise ValueError("scan_count must be between 2 and 100")
        _require_nonnegative_finite("scan_interval", self.scan_interval)
        if self.scan_interval > 5.0:
            raise ValueError("scan_interval must not exceed 5 seconds")
        if not 0 <= self.stale_threshold <= 3:
            raise ValueError("stale_threshold must be between 0 and 3")
        if not 1 <= self.warm_up_attempts <= MAX_LIDAR_WARM_UP_ATTEMPTS:
            raise ValueError(
                f"warm_up_attempts must be between 1 and "
                f"{MAX_LIDAR_WARM_UP_ATTEMPTS}"
            )
        _require_nonnegative_finite("warm_up_interval", self.warm_up_interval)
        if self.warm_up_attempts * self.warm_up_interval > MAX_LIDAR_WARM_UP_SECONDS:
            raise ValueError(
                "warm_up_attempts * warm_up_interval must not exceed 2 seconds"
            )


@dataclass(frozen=True)
class PerformanceProbeConfig:
    """Bounded performance probe configuration."""

    vehicle_name: str
    iterations: int = 20
    include_lidar: bool = False
    lidar_name: str = ""
    include_control: bool = False
    control_duration: float = 0.1
    probe_pause: bool = False
    allow_pause: bool = False
    airborne: AirborneProbeConfig | None = None

    def __post_init__(self) -> None:
        if not self.vehicle_name.strip():
            raise ValueError("vehicle_name must not be empty")
        if not 1 <= self.iterations <= 100:
            raise ValueError("iterations must be between 1 and 100")
        _require_positive_finite("control_duration", self.control_duration)
        if self.control_duration > 1.0:
            raise ValueError("control_duration must not exceed 1 second")
        if self.include_lidar and not self.lidar_name.strip():
            raise ValueError("LiDAR performance requires lidar_name")
        if self.include_control and self.airborne is None:
            raise ValueError("control benchmark requires airborne configuration")
        if (
            self.include_control
            and self.airborne is not None
            and not self.airborne.allow_flight
        ):
            raise ValueError("control benchmark requires allow_flight")
        if (
            self.airborne is not None
            and self.airborne.vehicle_name != self.vehicle_name
        ):
            raise ValueError("performance and airborne vehicle names must match")
        if self.probe_pause and not self.allow_pause:
            raise ValueError("pause probe requires allow_pause")


@dataclass(frozen=True)
class SceneObjectObservation:
    """Serializable pose and scale information for one scene object."""

    name: str
    position: tuple[float, float, float] | None
    orientation: tuple[float, float, float, float] | None
    scale: tuple[float, float, float] | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CollisionInfoSample:
    """Sanitized fields from one AirSim-style CollisionInfo response."""

    has_collided: bool | None
    object_name: str | None
    object_id: int | None
    time_stamp: int | None
    penetration_depth: float | None
    impact_point: tuple[float, float, float] | None
    vehicle_position: tuple[float, float, float] | None
    normal: tuple[float, float, float] | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CollisionAssessment:
    """Cross-sample collision classification and baseline evidence."""

    classification: CollisionClassification
    timestamp_changed: bool | None
    object_changed: bool | None
    object_id_changed: bool | None
    penetration_changed: bool | None
    impact_point_changed: bool | None
    vehicle_position_changed: bool | None
    normal_changed: bool | None
    persistent_or_historical: bool
    baseline_timestamp: int | None
    detail: str


@dataclass(frozen=True)
class LidarScanSummary:
    """Validated summary of one raw LiDAR scan."""

    sensor_name: str
    timestamp: int
    point_count: int
    minimum_range: float | None
    maximum_range: float | None
    sensor_position: tuple[float, float, float] | None
    valid: bool
    empty: bool
    error: str | None = None


@dataclass(frozen=True)
class SettingsFieldComparison:
    """One expected-versus-observed settings value."""

    scope: str
    field: str
    expected: Any
    actual: Any
    present: bool
    matched: bool


@dataclass(frozen=True)
class GroundedLidarScanSummary:
    """Strict summary of one grounded LiDAR scan."""

    sensor_name: str
    vehicle_name: str
    timestamp: int | None
    point_count: int
    minimum_range: float | None
    maximum_range: float | None
    beyond_configured_range_count: int
    sensor_position: tuple[float, float, float] | None
    sensor_orientation: tuple[float, float, float, float] | None
    near_field_counts: tuple[int, int, int]
    valid: bool
    empty: bool
    error: str | None = None


@dataclass(frozen=True)
class LidarTimestampSummary:
    """Measured timestamp freshness using adjacent transition semantics."""

    timestamps: tuple[int, ...]
    unique_timestamp_count: int
    fresh_transition_count: int
    repeated_transition_count: int
    regression_count: int
    maximum_repeated_timestamp_run: int


@dataclass(frozen=True)
class PerformanceSummary:
    """Latency and throughput statistics for a bounded operation."""

    operation: str
    attempted: int
    succeeded: int
    total_seconds: float
    calls_per_second: float
    minimum_seconds: float | None
    mean_seconds: float | None
    p95_seconds: float | None
    maximum_seconds: float | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CleanupDomainResult:
    """One independently attempted probe-cleanup domain."""

    domain: str
    attempted: bool
    succeeded: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityProbeReport:
    """Machine-readable output from one static or live probe invocation."""

    schema_version: str
    run_id: str
    mode: str
    started_at_utc: str
    completed_at_utc: str
    success: bool
    interrupted: bool
    observations: tuple[CapabilityObservation, ...]
    data: dict[str, Any]
    cleanup_results: tuple[CleanupDomainResult, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass
class ProbeRuntimeState:
    """Mutable safety state shared only during one explicitly invoked probe."""

    cleanup_state: CleanupState = field(default_factory=CleanupState)
    vehicle_name: str = ""
    created_objects: list[str] = field(default_factory=list)
    markers_created: bool = False
    lidar_visualization_markers_created: bool = False
    client_compromised: bool = False


@dataclass(frozen=True)
class AirborneProbeContext:
    """Measured reference values established after safe takeoff."""

    vehicle_name: str
    lidar_name: str
    ground_position: Position3D
    ground_reference_z: float
    anchor_position: Position3D
    initial_collision_timestamp: int | None = None
    configured_range: float = 0.0
    settings_comparisons: tuple[SettingsFieldComparison, ...] = ()


_CLIENT_METHODS: dict[str, str | None] = {
    "connection": "confirmConnection",
    "ping": "ping",
    "client_version": "getClientVersion",
    "server_version": "getServerVersion",
    "multirotor_state": "getMultirotorState",
    "collision_state": "simGetCollisionInfo",
    "api_control_state": "isApiControlEnabled",
    "armed_state": None,
    "vehicle_listing": "listVehicles",
    "settings": "getSettingsString",
    "scene_object_listing": "simListSceneObjects",
    "object_pose": "simGetObjectPose",
    "object_scale": "simGetObjectScale",
    "asset_listing": "simListAssets",
    "mesh_buffers": "simGetMeshPositionVertexBuffers",
    "marker_points": "simPlotPoints",
    "marker_line_strip": "simPlotLineStrip",
    "marker_line_list": "simPlotLineList",
    "marker_transform": "simPlotTransforms",
    "marker_flush": "simFlushPersistentMarkers",
    "object_spawn": "simSpawnObject",
    "object_move": "simSetObjectPose",
    "object_resize": "simSetObjectScale",
    "object_material": "simSetObjectMaterial",
    "object_texture": "simSetObjectMaterialFromTexture",
    "segmentation_set": "simSetSegmentationObjectID",
    "segmentation_get": "simGetSegmentationObjectID",
    "object_destroy": "simDestroyObject",
    "lidar": "getLidarData",
    "api_control": "enableApiControl",
    "arming": "armDisarm",
    "takeoff": "takeoffAsync",
    "hover": "hoverAsync",
    "landing": "landAsync",
    "anchor_move": "moveToPositionAsync",
    "yaw_rotation": "rotateToYawAsync",
    "pause": "simPause",
    "pause_state": "simIsPause",
    "continue_time": "simContinueForTime",
    "continue_frames": "simContinueForFrames",
    "trace_style": "simSetTraceLine",
    "direct_object_rgb": None,
    "runtime_clock_speed": None,
    "runtime_view_mode": None,
}


def inspect_client_capabilities(
    client_module: ModuleType,
) -> tuple[CapabilityObservation, ...]:
    """Inspect client class methods without constructing a simulator client."""
    client_class = getattr(client_module, "MultirotorClient", None)
    observations: list[CapabilityObservation] = []
    for capability, method_name in _CLIENT_METHODS.items():
        present = (
            method_name is not None
            and client_class is not None
            and hasattr(client_class, method_name)
        )
        observations.append(
            CapabilityObservation(
                capability=capability,
                client_method=method_name,
                evidence_level=EvidenceLevel.STATIC_CLIENT,
                status=(
                    CapabilityStatus.CLIENT_METHOD_PRESENT_NOT_LIVE_TESTED
                    if present
                    else CapabilityStatus.CLIENT_METHOD_ABSENT
                ),
                detail=(
                    f"Client declares {method_name}."
                    if present
                    else "No direct client method is declared."
                ),
                operator_confirmation=(
                    "pending"
                    if capability
                    in {
                        "marker_points",
                        "marker_line_strip",
                        "marker_line_list",
                        "marker_transform",
                        "object_material",
                    }
                    else "not_required"
                ),
            )
        )
    return tuple(observations)


def load_client_module(module_name: str = DEFAULT_CLIENT_MODULE) -> ModuleType:
    """Lazily import the external client for an explicitly invoked probe."""
    return import_colosseum_client_module(module_name)


def create_probe_client(
    client_module: ModuleType, config: ConnectionProbeConfig
) -> Any:
    """Construct a bounded-timeout client only for a requested live operation."""
    client_class = getattr(client_module, "MultirotorClient", None)
    if client_class is None:
        raise CapabilityProbeError("Client module does not provide MultirotorClient.")
    try:
        return client_class(
            ip=config.host,
            port=config.port,
            timeout_value=config.rpc_timeout,
        )
    except Exception as exc:
        raise CapabilityProbeError("Could not construct the simulator client.") from exc


def invoke_capability(
    client: Any,
    capability: str,
    method_name: str,
    *args: Any,
    clock: Callable[[], float] = time.perf_counter,
    **kwargs: Any,
) -> tuple[CapabilityObservation, Any | None]:
    """Invoke one RPC while keeping successful RPC and practical support distinct."""
    method = getattr(client, method_name, None)
    if method is None:
        return (
            CapabilityObservation(
                capability,
                method_name,
                EvidenceLevel.STATIC_CLIENT,
                CapabilityStatus.CLIENT_METHOD_ABSENT,
                f"Client does not provide {method_name}.",
            ),
            None,
        )
    started = clock()
    try:
        value = method(*args, **kwargs)
    except Exception as exc:
        status = classify_rpc_exception(exc)
        return (
            CapabilityObservation(
                capability,
                method_name,
                EvidenceLevel.LIVE_RPC,
                status,
                f"{method_name} failed: {type(exc).__name__}",
                max(0.0, clock() - started),
            ),
            None,
        )
    return (
        CapabilityObservation(
            capability,
            method_name,
            EvidenceLevel.LIVE_RPC,
            CapabilityStatus.RPC_SUCCEEDED,
            f"{method_name} returned without an RPC exception.",
            max(0.0, clock() - started),
        ),
        value,
    )


def classify_rpc_exception(exc: BaseException) -> CapabilityStatus:
    """Classify an RPC exception without importing legacy RPC exception classes."""
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timed out" in message or "timeout" in message:
        return CapabilityStatus.RPC_TIMED_OUT
    unsupported_markers = (
        "method not found",
        "function not found",
        "unknown method",
        "not implemented",
        "unimplemented",
    )
    if any(marker in message for marker in unsupported_markers):
        return CapabilityStatus.UNSUPPORTED_BY_BLOCKS
    return CapabilityStatus.RPC_FAILED


def sanitize_settings(settings_text: str) -> dict[str, Any]:
    """Extract non-sensitive simulator and sensor fields from settings JSON."""
    try:
        raw = json.loads(settings_text) if settings_text.strip() else {}
    except json.JSONDecodeError as exc:
        raise CapabilityProbeError("Simulator settings are not valid JSON.") from exc
    if not isinstance(raw, dict):
        raise CapabilityProbeError("Simulator settings must contain a JSON object.")

    sanitized: dict[str, Any] = {}
    for key in ("SettingsVersion", "SimMode", "ClockSpeed", "ViewMode", "EnableTrace"):
        if key in raw and isinstance(raw[key], (str, int, float, bool, type(None))):
            sanitized[key] = raw[key]

    vehicles = raw.get("Vehicles", {})
    sanitized_vehicles: dict[str, Any] = {}
    if isinstance(vehicles, dict):
        for vehicle_name, vehicle in vehicles.items():
            if not isinstance(vehicle_name, str) or not isinstance(vehicle, dict):
                continue
            vehicle_result: dict[str, Any] = {}
            for key in (
                "VehicleType",
                "AutoCreate",
                "DefaultVehicleState",
                "EnableTrace",
            ):
                if key in vehicle and isinstance(
                    vehicle[key], (str, int, float, bool, type(None))
                ):
                    vehicle_result[key] = vehicle[key]
            sensors = vehicle.get("Sensors", {})
            sensor_result: dict[str, Any] = {}
            if isinstance(sensors, dict):
                for sensor_name, sensor in sensors.items():
                    if not isinstance(sensor_name, str) or not isinstance(sensor, dict):
                        continue
                    allowed_sensor = {
                        key: value
                        for key, value in sensor.items()
                        if key
                        in {
                            "SensorType",
                            "Enabled",
                            "NumberOfChannels",
                            "RotationsPerSecond",
                            "Range",
                            "PointsPerSecond",
                            "DataFrame",
                            "DrawDebugPoints",
                            "ExternalController",
                            "X",
                            "Y",
                            "Z",
                            "Roll",
                            "Pitch",
                            "Yaw",
                            "VerticalFOVUpper",
                            "VerticalFOVLower",
                            "HorizontalFOVStart",
                            "HorizontalFOVEnd",
                        }
                        and isinstance(value, (str, int, float, bool, type(None)))
                    }
                    sensor_result[sensor_name] = allowed_sensor
            if sensor_result:
                vehicle_result["Sensors"] = sensor_result
            sanitized_vehicles[vehicle_name] = vehicle_result
    if sanitized_vehicles:
        sanitized["Vehicles"] = sanitized_vehicles
    return sanitized


def compare_lidar_settings_profile(
    settings: dict[str, Any],
    profile: LidarSettingsProfile = M13_LIDAR_PROVISIONAL_PROFILE,
) -> tuple[SettingsFieldComparison, ...]:
    """Compare sanitized settings with the exact provisional M13.1 profile."""
    vehicles = settings.get("Vehicles", {})
    vehicle = (
        vehicles.get(profile.vehicle_name, {}) if isinstance(vehicles, dict) else {}
    )
    sensors = vehicle.get("Sensors", {}) if isinstance(vehicle, dict) else {}
    sensor = sensors.get(profile.lidar_name, {}) if isinstance(sensors, dict) else {}

    comparisons: list[SettingsFieldComparison] = []
    for scope, actual_values, expected_values in (
        ("vehicle", vehicle, profile.vehicle_fields),
        ("sensor", sensor, profile.sensor_fields),
    ):
        mapping = actual_values if isinstance(actual_values, dict) else {}
        for field_name, expected in expected_values:
            present = field_name in mapping
            actual = mapping.get(field_name)
            comparisons.append(
                SettingsFieldComparison(
                    scope=scope,
                    field=field_name,
                    expected=expected,
                    actual=actual,
                    present=present,
                    matched=present and _settings_values_match(expected, actual),
                )
            )
    return tuple(comparisons)


def _selected_vehicle_settings(
    settings: dict[str, Any], vehicle_name: str
) -> dict[str, Any] | None:
    vehicles = settings.get("Vehicles", {})
    if not vehicle_name or not isinstance(vehicles, dict):
        return None
    vehicle = vehicles.get(vehicle_name)
    return vehicle if isinstance(vehicle, dict) else None


def _settings_values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == float(expected)
        )
    return type(actual) is type(expected) and actual == expected


def sample_collision_information(
    client: Any,
    *,
    vehicle_name: str = "",
    sample_count: int = COLLISION_SAMPLE_COUNT,
    sample_interval: float = COLLISION_SAMPLE_INTERVAL,
    clock: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[tuple[CapabilityObservation, ...], tuple[CollisionInfoSample, ...]]:
    """Collect bounded, sanitized read-only collision samples."""
    if sample_count != COLLISION_SAMPLE_COUNT:
        raise ValueError(f"sample_count must be exactly {COLLISION_SAMPLE_COUNT}")
    _require_nonnegative_finite("sample_interval", sample_interval)
    if sample_interval > 1.0:
        raise ValueError("sample_interval must not exceed 1 second")

    observations: list[CapabilityObservation] = []
    samples: list[CollisionInfoSample] = []
    for index in range(sample_count):
        observation, collision_info = invoke_capability(
            client,
            "collision_state",
            "simGetCollisionInfo",
            vehicle_name=vehicle_name,
            clock=clock,
        )
        observations.append(
            replace(
                observation,
                detail=f"Collision sample {index + 1}: {observation.detail}",
            )
        )
        _require_rpc_success(observation)
        samples.append(sanitize_collision_info(collision_info))
        if index + 1 < sample_count and sample_interval:
            sleep_fn(sample_interval)
    return tuple(observations), tuple(samples)


def sanitize_collision_info(collision_info: Any) -> CollisionInfoSample:
    """Extract only bounded non-sensitive CollisionInfo fields."""
    errors: list[str] = []
    has_collided_raw = getattr(collision_info, "has_collided", None)
    has_collided = has_collided_raw if isinstance(has_collided_raw, bool) else None
    if has_collided is None:
        errors.append("has_collided is unavailable or non-boolean")

    object_name_raw = getattr(collision_info, "object_name", None)
    object_name = _sanitize_collision_object_name(object_name_raw, errors)
    object_id = _optional_collision_int(
        getattr(collision_info, "object_id", None), "object_id", errors
    )
    time_stamp = _optional_collision_int(
        getattr(collision_info, "time_stamp", None), "time_stamp", errors
    )
    penetration_depth = _optional_collision_float(
        getattr(collision_info, "penetration_depth", None),
        "penetration_depth",
        errors,
    )
    impact_point = _optional_collision_vector(
        getattr(collision_info, "impact_point", None), "impact_point", errors
    )
    vehicle_position = _optional_collision_vector(
        getattr(collision_info, "position", None), "position", errors
    )
    normal = _optional_collision_vector(
        getattr(collision_info, "normal", None), "normal", errors
    )
    return CollisionInfoSample(
        has_collided=has_collided,
        object_name=object_name,
        object_id=object_id,
        time_stamp=time_stamp,
        penetration_depth=penetration_depth,
        impact_point=impact_point,
        vehicle_position=vehicle_position,
        normal=normal,
        errors=tuple(errors),
    )


def classify_collision_samples(
    samples: Sequence[CollisionInfoSample],
    *,
    is_landed: bool | None,
    measured_speed: float,
    api_control_enabled: bool | None = None,
    operator_confirmed_stable: bool = False,
) -> CollisionAssessment:
    """Classify repeated collision evidence without assuming landed means safe."""
    if len(samples) != COLLISION_SAMPLE_COUNT or not math.isfinite(measured_speed):
        return _inconclusive_collision_assessment(
            samples, "Collision evidence is incomplete or measured speed is invalid."
        )

    collided_values = [sample.has_collided for sample in samples]
    if all(value is False for value in collided_values):
        return _build_collision_assessment(
            samples,
            CollisionClassification.NO_COLLISION,
            persistent_or_historical=False,
            detail="All three read-only samples reported no collision.",
        )
    if any(value is None for value in collided_values):
        return _inconclusive_collision_assessment(
            samples, "At least one sample omitted a reliable collision flag."
        )
    if any(
        previous is False and current is True
        for previous, current in zip(collided_values, collided_values[1:], strict=False)
    ):
        return _unsafe_collision_assessment(
            samples, "A new false-to-true collision transition was observed."
        )
    accepted_ground_patterns = {
        (True, True, True),
        (True, False, False),
        (True, True, False),
    }
    if tuple(collided_values) not in accepted_ground_patterns:
        return _unsafe_collision_assessment(
            samples, "Collision state changed in an unsupported pattern."
        )

    timestamps = [sample.time_stamp for sample in samples]
    object_names = [sample.object_name for sample in samples]
    object_ids = [sample.object_id for sample in samples]
    penetrations = [sample.penetration_depth for sample in samples]
    impact_points = [sample.impact_point for sample in samples]
    vehicle_positions = [sample.vehicle_position for sample in samples]
    normals = [sample.normal for sample in samples]
    timestamp_changed = _values_changed(timestamps)
    object_changed = _values_changed(object_names)
    object_id_changed = _values_changed(object_ids)
    penetration_changed = _float_values_changed(penetrations)
    impact_point_changed = _vector_values_changed(impact_points)
    vehicle_position_changed = _vector_values_changed(vehicle_positions)
    normal_changed = _vector_values_changed(normals)
    required_values = all(
        sample.object_name
        and sample.object_id is not None
        and sample.time_stamp is not None
        and sample.time_stamp > 0
        and sample.penetration_depth is not None
        and sample.impact_point is not None
        and sample.vehicle_position is not None
        and sample.normal is not None
        for sample in samples
    )
    if not required_values:
        return _inconclusive_collision_assessment(
            samples,
            "Collided samples lack geometry, identity, penetration, or timestamp data.",
        )
    if any(
        changed is True
        for changed in (
            timestamp_changed,
            object_changed,
            object_id_changed,
            penetration_changed,
            impact_point_changed,
            vehicle_position_changed,
            normal_changed,
        )
    ):
        return _unsafe_collision_assessment(
            samples,
            "Collision identity, timestamp, penetration, or geometry changed "
            "between stationary samples.",
        )
    if is_landed is None:
        return _inconclusive_collision_assessment(
            samples, "Landed-state evidence is unavailable or unrecognized."
        )
    if is_landed is False or measured_speed > GROUNDED_SPEED_TOLERANCE:
        return _unsafe_collision_assessment(
            samples, "Collision occurred while the vehicle was not landed and still."
        )
    if api_control_enabled is None:
        return _inconclusive_collision_assessment(
            samples, "API-control state is unavailable for ground-contact review."
        )
    if api_control_enabled:
        return _unsafe_collision_assessment(
            samples, "API control was enabled during ground-contact sampling."
        )
    if not operator_confirmed_stable:
        return _inconclusive_collision_assessment(
            samples,
            "Operator confirmation of no visible instability or impact is required.",
        )

    for sample in samples:
        penetration = float(sample.penetration_depth)
        impact = sample.impact_point
        position = sample.vehicle_position
        normal = sample.normal
        if penetration < 0.0 or penetration > GROUND_CONTACT_MAX_PENETRATION:
            return _unsafe_collision_assessment(
                samples, "Collision penetration is outside the ground-contact limit."
            )
        horizontal_distance = math.hypot(
            impact[0] - position[0], impact[1] - position[1]
        )
        vertical_distance = impact[2] - position[2]
        if (
            horizontal_distance > GROUND_CONTACT_HORIZONTAL_TOLERANCE
            or vertical_distance < -1e-3
            or vertical_distance > GROUND_CONTACT_VERTICAL_TOLERANCE
            or normal[2] > -GROUND_NORMAL_MIN_UPWARD_COMPONENT
        ):
            return _unsafe_collision_assessment(
                samples,
                "Collision geometry is inconsistent with a supporting ground surface.",
            )

    return _build_collision_assessment(
        samples,
        CollisionClassification.EXPECTED_GROUND_CONTACT,
        persistent_or_historical=True,
        detail=(
            "Stable timestamp, object, penetration, impact geometry, and upward "
            "surface normal, together with operator confirmation, are consistent "
            "with stationary historical or one-shot ground contact."
        ),
    )


def survey_scene(
    client: Any,
    config: SceneSurveyConfig,
    *,
    client_module: ModuleType | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[tuple[CapabilityObservation, ...], dict[str, Any]]:
    """Perform the bounded, read-only scene survey."""
    observations: list[CapabilityObservation] = []
    state_obs, state = invoke_capability(
        client,
        "multirotor_state",
        "getMultirotorState",
        vehicle_name=config.connection.vehicle_name,
        clock=clock,
    )
    observations.append(state_obs)
    if state is None:
        raise CapabilityProbeError("Read-only survey could not read multirotor state.")
    position = extract_finite_position(state)
    velocity = _extract_finite_velocity(state)
    landed_state = _summarize_landed_state(state, client_module)

    collision_observations, collision_samples = sample_collision_information(
        client,
        vehicle_name=config.connection.vehicle_name,
        sample_count=COLLISION_SAMPLE_COUNT,
        sample_interval=COLLISION_SAMPLE_INTERVAL,
        clock=clock,
        sleep_fn=sleep_fn,
    )
    observations.extend(collision_observations)

    api_control_obs, api_control_enabled = invoke_capability(
        client,
        "api_control_state",
        "isApiControlEnabled",
        vehicle_name=config.connection.vehicle_name,
        clock=clock,
    )
    observations.append(api_control_obs)
    _require_rpc_success(api_control_obs)
    if not isinstance(api_control_enabled, bool):
        raise CapabilityProbeError("API-control query returned a non-boolean value.")

    speed = math.sqrt(sum(component**2 for component in velocity))
    collision_assessment = classify_collision_samples(
        collision_samples,
        is_landed=landed_state["is_landed"],
        measured_speed=speed,
        api_control_enabled=api_control_enabled,
        operator_confirmed_stable=config.confirm_no_visible_collision,
    )
    collision = any(sample.has_collided is True for sample in collision_samples)
    safe_for_later_stages = (
        landed_state["is_landed"] is True
        and speed <= GROUNDED_SPEED_TOLERANCE
        and collision_assessment.classification
        in {
            CollisionClassification.NO_COLLISION,
            CollisionClassification.EXPECTED_GROUND_CONTACT,
        }
        and not api_control_enabled
    )

    observations.append(
        CapabilityObservation(
            "collision_assessment",
            "simGetCollisionInfo",
            EvidenceLevel.PRACTICAL_BEHAVIOR,
            (
                CapabilityStatus.INCONCLUSIVE
                if collision_assessment.classification
                is CollisionClassification.INCONCLUSIVE_COLLISION
                else CapabilityStatus.SUPPORTED_WITH_LIMITATIONS
            ),
            collision_assessment.detail,
        )
    )

    observations.append(
        CapabilityObservation(
            "measured_grounded_state",
            "getMultirotorState",
            EvidenceLevel.PRACTICAL_BEHAVIOR,
            CapabilityStatus.SUPPORTED_WITH_LIMITATIONS,
            (
                "Measured finite position and velocity with landed-state evidence; "
                "armed state is unavailable from this client."
            ),
        )
    )

    settings_obs, settings_text = invoke_capability(
        client, "settings", "getSettingsString", clock=clock
    )
    observations.append(settings_obs)
    _raise_if_rpc_timeout(settings_obs)
    settings = (
        sanitize_settings(settings_text) if isinstance(settings_text, str) else {}
    )
    profile_comparisons: tuple[SettingsFieldComparison, ...] = ()
    profile_matches: bool | None = None
    if config.connection.vehicle_name == M13_LIDAR_PROVISIONAL_PROFILE.vehicle_name:
        profile_comparisons = compare_lidar_settings_profile(
            settings,
            M13_LIDAR_PROVISIONAL_PROFILE,
        )
        profile_matches = all(item.matched for item in profile_comparisons)
        observations.append(
            CapabilityObservation(
                "m13_lidar_settings_profile",
                "getSettingsString",
                EvidenceLevel.LIVE_RPC,
                (
                    CapabilityStatus.RPC_SUCCEEDED
                    if profile_matches
                    else CapabilityStatus.REQUIRES_LOCAL_CONFIGURATION
                ),
                (
                    "The selected vehicle and LiDAR settings match the provisional "
                    "M13.1 profile."
                    if profile_matches
                    else "The selected vehicle or LiDAR settings do not match the "
                    "provisional M13.1 profile."
                ),
            )
        )
        safe_for_later_stages = safe_for_later_stages and profile_matches

    vehicles_obs, vehicles = invoke_capability(
        client, "vehicle_listing", "listVehicles", clock=clock
    )
    observations.append(vehicles_obs)
    _raise_if_rpc_timeout(vehicles_obs)

    objects_obs, object_names = invoke_capability(
        client,
        "scene_object_listing",
        "simListSceneObjects",
        config.object_regex,
        clock=clock,
    )
    observations.append(objects_obs)
    _raise_if_rpc_timeout(objects_obs)
    if object_names is None:
        object_names = []
    sampled_names = sorted(str(name) for name in object_names)[: config.max_objects]
    object_results = tuple(_survey_object(client, name) for name in sampled_names)

    assets_obs, assets = invoke_capability(
        client, "asset_listing", "simListAssets", clock=clock
    )
    observations.append(assets_obs)
    _raise_if_rpc_timeout(assets_obs)
    asset_names = sorted(str(name) for name in (assets or []))[:500]

    return tuple(observations), {
        "measured_position": _position_tuple(position),
        "measured_linear_velocity": velocity,
        "measured_state": {
            "position": _position_tuple(position),
            "linear_velocity": velocity,
            "landed_state_value": landed_state["value"],
            "landed_state_label": landed_state["label"],
            "is_landed": landed_state["is_landed"],
            "collision": collision,
            "collision_samples": collision_samples,
            "collision_assessment": collision_assessment,
            "api_control_enabled": api_control_enabled,
            "operator_confirmed_no_visible_collision": (
                config.confirm_no_visible_collision
            ),
            "speed": speed,
            "grounded_speed_tolerance": GROUNDED_SPEED_TOLERANCE,
            "safe_for_later_stages": safe_for_later_stages,
            "armed_state": {
                "availability": "unavailable",
                "value": None,
                "detail": (
                    "The validated client exposes armDisarm but no reliable "
                    "read-only armed-state getter."
                ),
            },
        },
        "settings": settings,
        "selected_vehicle_name": config.connection.vehicle_name,
        "selected_vehicle_settings": _selected_vehicle_settings(
            settings, config.connection.vehicle_name
        ),
        "m13_lidar_profile_matches": profile_matches,
        "m13_lidar_profile_comparisons": profile_comparisons,
        "vehicles": sorted(str(name) for name in (vehicles or [])),
        "scene_object_count": len(object_names),
        "scene_objects_truncated": len(object_names) > len(sampled_names),
        "scene_objects": object_results,
        "asset_count": len(assets or []),
        "assets": asset_names,
        "mesh_buffer_probe": "deferred_not_invoked",
    }


def derive_safe_probe_position(
    measured_position: Position3D,
    offset: tuple[float, float, float],
) -> Position3D:
    """Resolve a bounded probe position relative to measured simulator state."""
    _validate_vector("offset", offset)
    return Position3D(
        measured_position.x + offset[0],
        measured_position.y + offset[1],
        measured_position.z + offset[2],
    )


def probe_debug_markers(
    client: Any,
    client_module: ModuleType,
    measured_position: Position3D,
    runtime: ProbeRuntimeState,
    *,
    allow_debug_markers: bool,
    allow_marker_flush: bool,
    marker_hold_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    message_fn: Callable[[str], None] = print,
) -> tuple[CapabilityObservation, ...]:
    """Create persistent debug markers after explicit dual authorization."""
    if not allow_debug_markers or not allow_marker_flush:
        raise ValueError(
            "marker probe requires allow_debug_markers and allow_marker_flush"
        )
    validate_marker_hold_seconds(marker_hold_seconds)
    vector = _require_client_type(client_module, "Vector3r")
    pose_type = _require_client_type(client_module, "Pose")
    origin = derive_safe_probe_position(measured_position, (1.5, 0.0, -0.5))
    points = [vector(origin.x, origin.y, origin.z)]
    line_strip = [
        vector(origin.x, origin.y, origin.z),
        vector(origin.x + 0.5, origin.y, origin.z),
        vector(origin.x + 0.5, origin.y + 0.5, origin.z),
    ]
    line_list = [
        vector(origin.x, origin.y, origin.z),
        vector(origin.x + 0.5, origin.y, origin.z),
        vector(origin.x, origin.y, origin.z),
        vector(origin.x, origin.y + 0.5, origin.z),
    ]
    coordinate_pose = pose_type(position_val=vector(origin.x, origin.y, origin.z))
    observations: list[CapabilityObservation] = []
    runtime.markers_created = True
    for capability, method_name, args in (
        (
            "marker_points",
            "simPlotPoints",
            (points, [1.0, 0.0, 0.0, 1.0], 15.0, -1.0, True),
        ),
        (
            "marker_line_strip",
            "simPlotLineStrip",
            (line_strip, [0.0, 1.0, 0.0, 1.0], 4.0, -1.0, True),
        ),
        (
            "marker_line_list",
            "simPlotLineList",
            (line_list, [0.0, 0.0, 1.0, 1.0], 4.0, -1.0, True),
        ),
        (
            "marker_transform",
            "simPlotTransforms",
            ([coordinate_pose], 0.5, 3.0, -1.0, True),
        ),
    ):
        observation, _ = invoke_capability(client, capability, method_name, *args)
        observations.append(replace(observation, operator_confirmation="pending"))
        _require_rpc_success(observation)
    message_fn(
        "Debug markers are currently visible for "
        f"{marker_hold_seconds:g} seconds before cleanup."
    )
    if marker_hold_seconds:
        sleep_fn(marker_hold_seconds)
    return tuple(observations)


def validate_marker_hold_seconds(value: float) -> None:
    """Require a finite marker hold duration within the supervised safety bound."""
    _require_nonnegative_finite("marker_hold_seconds", value)
    if value > MAX_MARKER_HOLD_SECONDS:
        raise ValueError(
            f"marker_hold_seconds must not exceed {MAX_MARKER_HOLD_SECONDS:g} seconds"
        )


def validate_lidar_visualization_hold_seconds(value: float) -> None:
    """Require a finite LiDAR-overlay hold within the supervised safety bound."""
    _require_nonnegative_finite("lidar_visualization_hold_seconds", value)
    if value > MAX_LIDAR_VISUALIZATION_HOLD_SECONDS:
        raise ValueError(
            "lidar_visualization_hold_seconds must not exceed "
            f"{MAX_LIDAR_VISUALIZATION_HOLD_SECONDS:g} seconds"
        )


def validate_mutation_hold_seconds(value: float) -> None:
    """Require a finite staged hold within the mutation safety bound."""
    _require_nonnegative_finite("mutation_hold_seconds", value)
    if value > MAX_MUTATION_HOLD_SECONDS:
        raise ValueError(
            "mutation_hold_seconds must not exceed "
            f"{MAX_MUTATION_HOLD_SECONDS:g} seconds"
        )


def validate_material_name(value: str | None) -> None:
    """Validate an optional Unreal material object path without guessing assets."""
    if value is None:
        return
    if not value or value != value.strip():
        raise ValueError("material_name must be a non-empty trimmed value")
    if len(value) > 256:
        raise ValueError("material_name must not exceed 256 characters")
    if not re.fullmatch(r"/[A-Za-z0-9_/-]+\.[A-Za-z0-9_]+", value):
        raise ValueError(
            "material_name must be a full Unreal object path such as "
            "'/Mount/Path/Material.Material'"
        )
    package_name, object_name = value.rsplit("/", 1)[-1].split(".", 1)
    if package_name != object_name:
        raise ValueError("material_name package and object names must match")


def probe_scene_mutation(
    client: Any,
    client_module: ModuleType,
    config: SceneMutationConfig,
    runtime: ProbeRuntimeState,
    *,
    uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    sleep_fn: Callable[[float], None] = time.sleep,
    message_fn: Callable[[str], None] = print,
) -> tuple[tuple[CapabilityObservation, ...], dict[str, Any]]:
    """Exercise one temporary object without touching built-in scene geometry."""
    if not (
        config.allow_scene_mutation
        and config.confirm_spawn_area_clear
        and config.confirm_vehicle_disarmed
    ):
        raise ValueError(
            "scene mutation requires authorization, clear-area confirmation, and "
            "disarm confirmation"
        )
    state = _require_method(client, "getMultirotorState")()
    ground_position = validate_grounded_preflight(
        client,
        client_module,
        state,
        operator_confirmed_stable=(
            config.confirm_spawn_area_clear and config.confirm_vehicle_disarmed
        ),
    )
    assets = _require_method(client, "simListAssets")()
    if config.asset_name not in {str(asset) for asset in assets}:
        raise CapabilityProbeError(
            f"Requested asset {config.asset_name!r} was not returned by simListAssets."
        )
    spawn_position = derive_safe_probe_position(ground_position, config.spawn_offset)
    vector = _require_client_type(client_module, "Vector3r")
    pose_type = _require_client_type(client_module, "Pose")
    requested_name = f"{PROBE_OBJECT_PREFIX}{uuid_factory().hex}"
    pose = pose_type(position_val=vector(*_position_tuple(spawn_position)))
    scale = vector(config.initial_scale, config.initial_scale, config.initial_scale)
    observations: list[CapabilityObservation] = []

    spawn_obs, returned_name = invoke_capability(
        client,
        "object_spawn",
        "simSpawnObject",
        requested_name,
        config.asset_name,
        pose,
        scale,
        False,
        False,
    )
    observations.append(spawn_obs)
    _require_rpc_success(spawn_obs)
    if not isinstance(returned_name, str) or not returned_name.strip():
        raise CapabilityProbeError("simSpawnObject returned an empty object name.")
    runtime.created_objects.append(returned_name)
    if returned_name != requested_name:
        raise CapabilityProbeError(
            "simSpawnObject returned an unexpected object name; cleanup is required."
        )

    listed = _require_method(client, "simListSceneObjects")(
        _exact_object_regex(returned_name)
    )
    if returned_name not in {str(name) for name in listed}:
        raise CapabilityProbeError("Spawned object was not found by exact scene query.")

    material_data: dict[str, Any] = {
        "requested_name": config.material_name,
        "assignment_attempted": False,
        "assignment_rpc_succeeded": None,
        "assignment_result": None,
        "readback_availability": "not_applicable",
        "readback_detail": "No material was requested.",
    }
    if config.material_name:
        material_obs, material_result = invoke_capability(
            client,
            "object_material",
            "simSetObjectMaterial",
            returned_name,
            config.material_name,
            0,
        )
        observations.append(replace(material_obs, operator_confirmation="pending"))
        _require_rpc_success(material_obs)
        material_data["assignment_attempted"] = True
        material_data["assignment_rpc_succeeded"] = True
        material_data["assignment_result"] = material_result
        material_data["readback_availability"] = "unavailable"
        material_data["readback_detail"] = (
            "The validated client exposes no reliable material read-back API; "
            "appearance requires operator confirmation."
        )
        if material_result is not True:
            raise CapabilityProbeError("Material assignment reported failure.")

    original_pose = _require_method(client, "simGetObjectPose")(returned_name)
    _hold_mutation_stage(
        "spawned",
        config.mutation_hold_seconds,
        sleep_fn=sleep_fn,
        message_fn=message_fn,
    )
    moved_position = Position3D(
        spawn_position.x + config.move_distance,
        spawn_position.y,
        spawn_position.z,
    )
    moved_pose = pose_type(position_val=vector(*_position_tuple(moved_position)))
    move_obs, move_result = invoke_capability(
        client, "object_move", "simSetObjectPose", returned_name, moved_pose, True
    )
    observations.append(move_obs)
    _require_rpc_success(move_obs)
    if move_result is False:
        raise CapabilityProbeError("simSetObjectPose reported failure.")
    measured_pose = _require_method(client, "simGetObjectPose")(returned_name)
    measured_position = _extract_pose_position(measured_pose)
    if calculate_position_error(measured_position, moved_position) > 0.1:
        raise CapabilityProbeError(
            "Temporary object pose did not match requested move."
        )
    _hold_mutation_stage(
        "moved",
        config.mutation_hold_seconds,
        sleep_fn=sleep_fn,
        message_fn=message_fn,
    )

    moved_scale = vector(config.moved_scale, config.moved_scale, config.moved_scale)
    scale_obs, scale_result = invoke_capability(
        client, "object_resize", "simSetObjectScale", returned_name, moved_scale
    )
    observations.append(scale_obs)
    _require_rpc_success(scale_obs)
    if scale_result is False:
        raise CapabilityProbeError("simSetObjectScale reported failure.")
    measured_scale = _extract_vector(
        _require_method(client, "simGetObjectScale")(returned_name), "object scale"
    )
    if any(abs(value - config.moved_scale) > 0.01 for value in measured_scale):
        raise CapabilityProbeError(
            "Temporary object scale did not match requested scale."
        )
    _hold_mutation_stage(
        "resized",
        config.mutation_hold_seconds,
        sleep_fn=sleep_fn,
        message_fn=message_fn,
    )

    segmentation_obs, segmentation_result = invoke_capability(
        client,
        "segmentation_set",
        "simSetSegmentationObjectID",
        returned_name,
        config.segmentation_id,
        False,
    )
    observations.append(segmentation_obs)
    _require_rpc_success(segmentation_obs)
    if segmentation_result is False:
        raise CapabilityProbeError("Segmentation assignment reported failure.")
    read_id = _require_method(client, "simGetSegmentationObjectID")(returned_name)
    if int(read_id) != config.segmentation_id:
        raise CapabilityProbeError("Segmentation ID did not round trip.")

    return tuple(observations), {
        "requested_name": requested_name,
        "returned_name": returned_name,
        "asset_name": config.asset_name,
        "physics_enabled": False,
        "spawn_position": _position_tuple(spawn_position),
        "original_position": _position_tuple(_extract_pose_position(original_pose)),
        "moved_position": _position_tuple(measured_position),
        "measured_scale": measured_scale,
        "segmentation_id": int(read_id),
        "material": material_data,
    }


def _hold_mutation_stage(
    stage: str,
    duration: float,
    *,
    sleep_fn: Callable[[float], None],
    message_fn: Callable[[str], None],
) -> None:
    message_fn(
        f"Temporary cube {stage}; visible for {duration:g} seconds before "
        "the next mutation step."
    )
    if duration:
        sleep_fn(duration)


def validate_lidar_scan(lidar_data: Any, sensor_name: str = "") -> LidarScanSummary:
    """Validate raw point triples without performing feature extraction."""
    cloud = getattr(lidar_data, "point_cloud", None)
    timestamp = int(getattr(lidar_data, "time_stamp", 0))
    sensor_position = _optional_pose_position(getattr(lidar_data, "pose", None))
    if cloud is None:
        return LidarScanSummary(
            sensor_name,
            timestamp,
            0,
            None,
            None,
            sensor_position,
            False,
            True,
            "point_cloud is missing",
        )
    try:
        values = [float(value) for value in cloud]
    except (TypeError, ValueError):
        return LidarScanSummary(
            sensor_name,
            timestamp,
            0,
            None,
            None,
            sensor_position,
            False,
            False,
            "point_cloud contains non-numeric values",
        )
    if not values:
        return LidarScanSummary(
            sensor_name,
            timestamp,
            0,
            None,
            None,
            sensor_position,
            False,
            True,
            "point_cloud is empty",
        )
    if len(values) % 3 != 0:
        return LidarScanSummary(
            sensor_name,
            timestamp,
            len(values) // 3,
            None,
            None,
            sensor_position,
            False,
            False,
            "point_cloud length is not divisible by three",
        )
    if not all(math.isfinite(value) for value in values):
        return LidarScanSummary(
            sensor_name,
            timestamp,
            len(values) // 3,
            None,
            None,
            sensor_position,
            False,
            False,
            "point_cloud contains non-finite values",
        )
    ranges = [
        math.sqrt(values[index] ** 2 + values[index + 1] ** 2 + values[index + 2] ** 2)
        for index in range(0, len(values), 3)
    ]
    return LidarScanSummary(
        sensor_name,
        timestamp,
        len(ranges),
        min(ranges),
        max(ranges),
        sensor_position,
        True,
        False,
    )


def validate_grounded_lidar_scan(
    lidar_data: Any,
    *,
    sensor_name: str,
    vehicle_name: str,
    configured_range: float,
) -> GroundedLidarScanSummary:
    """Validate one grounded scan, including timestamp, full pose, and range."""
    _require_positive_finite("configured_range", configured_range)
    errors: list[str] = []
    empty = False

    timestamp: int | None
    try:
        raw_timestamp = lidar_data.time_stamp
        timestamp = int(raw_timestamp)
        if timestamp <= 0:
            errors.append("timestamp must be positive")
    except (AttributeError, TypeError, ValueError, OverflowError):
        timestamp = None
        errors.append("timestamp is unavailable or malformed")

    sensor_position: tuple[float, float, float] | None = None
    sensor_orientation: tuple[float, float, float, float] | None = None
    pose = getattr(lidar_data, "pose", None)
    try:
        sensor_position = _extract_vector(
            getattr(pose, "position", None), "LiDAR pose position"
        )
        sensor_orientation = _extract_quaternion(
            getattr(pose, "orientation", None), "LiDAR pose orientation"
        )
    except (CapabilityProbeError, ValueError) as exc:
        errors.append(str(exc))

    cloud = getattr(lidar_data, "point_cloud", None)
    values: list[float] = []
    if cloud is None:
        empty = True
        errors.append("point_cloud is missing")
    else:
        try:
            values = [float(value) for value in cloud]
        except (TypeError, ValueError):
            errors.append("point_cloud contains non-numeric values")
        if not values and not errors:
            empty = True
            errors.append("point_cloud is empty")
        elif values and len(values) % 3:
            errors.append("point_cloud length is not divisible by three")
        elif values and not all(math.isfinite(value) for value in values):
            errors.append("point_cloud contains non-finite values")

    ranges: list[float] = []
    if (
        values
        and len(values) % 3 == 0
        and all(math.isfinite(value) for value in values)
    ):
        ranges = [
            math.hypot(values[index], values[index + 1], values[index + 2])
            for index in range(0, len(values), 3)
        ]
        if not all(math.isfinite(value) and value >= 0.0 for value in ranges):
            errors.append("point_cloud produced an invalid Euclidean range")

    range_limit = configured_range + LIDAR_RANGE_TOLERANCE
    beyond_range_count = sum(value > range_limit for value in ranges)
    if beyond_range_count:
        errors.append(f"{beyond_range_count} point(s) exceed configured Range + 0.10 m")
    near_counts = tuple(
        sum(value < threshold for value in ranges)
        for threshold in NEAR_FIELD_THRESHOLDS
    )
    return GroundedLidarScanSummary(
        sensor_name=sensor_name,
        vehicle_name=vehicle_name,
        timestamp=timestamp,
        point_count=len(ranges),
        minimum_range=min(ranges) if ranges else None,
        maximum_range=max(ranges) if ranges else None,
        beyond_configured_range_count=beyond_range_count,
        sensor_position=sensor_position,
        sensor_orientation=sensor_orientation,
        near_field_counts=near_counts,
        valid=not errors,
        empty=empty,
        error="; ".join(errors) if errors else None,
    )


def analyze_lidar_timestamps(
    timestamps: Sequence[int],
) -> LidarTimestampSummary:
    """Summarize measured timestamps using adjacent transition semantics."""
    fresh_transitions = 0
    repeated_transitions = 0
    regressions = 0
    current_repeated_run = 0
    maximum_repeated_run = 0

    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        if current > previous:
            fresh_transitions += 1
            current_repeated_run = 0
        elif current == previous:
            repeated_transitions += 1
            current_repeated_run += 1
            maximum_repeated_run = max(maximum_repeated_run, current_repeated_run)
        else:
            regressions += 1
            current_repeated_run = 0

    return LidarTimestampSummary(
        timestamps=tuple(timestamps),
        unique_timestamp_count=len(set(timestamps)),
        fresh_transition_count=fresh_transitions,
        repeated_transition_count=repeated_transitions,
        regression_count=regressions,
        maximum_repeated_timestamp_run=maximum_repeated_run,
    )


def _matched_setting_value(
    comparisons: Sequence[SettingsFieldComparison],
    scope: str,
    field_name: str,
) -> Any:
    for comparison in comparisons:
        if comparison.scope == scope and comparison.field == field_name:
            return comparison.actual
    return None


def _summarize_near_field(
    scans: Sequence[GroundedLidarScanSummary],
) -> tuple[dict[str, Any], ...]:
    total_points = sum(scan.point_count for scan in scans)
    summaries: list[dict[str, Any]] = []
    for index, threshold in enumerate(NEAR_FIELD_THRESHOLDS):
        point_count = sum(scan.near_field_counts[index] for scan in scans)
        scans_with_points = sum(scan.near_field_counts[index] > 0 for scan in scans)
        summaries.append(
            {
                "threshold_metres": threshold,
                "point_count": point_count,
                "proportion": point_count / total_points if total_points else 0.0,
                "scans_with_points": scans_with_points,
                "recurs_across_scans": scans_with_points >= 2,
            }
        )
    return tuple(summaries)


def _classify_self_hits(
    scans: Sequence[GroundedLidarScanSummary],
    near_field: Sequence[dict[str, Any]],
) -> SelfHitClassification:
    if not scans or not all(scan.valid for scan in scans):
        return SelfHitClassification.INCONCLUSIVE
    below_005 = near_field[0]
    below_010 = near_field[1]
    if below_005["scans_with_points"] >= 2 or below_010[
        "scans_with_points"
    ] >= math.ceil(len(scans) / 2):
        return SelfHitClassification.POSSIBLE_SELF_HIT
    if below_010["point_count"] == 0:
        return SelfHitClassification.NO_EVIDENT_SELF_HIT
    return SelfHitClassification.INCONCLUSIVE


def probe_grounded_lidar(
    client: Any,
    client_module: ModuleType,
    config: GroundedLidarProbeConfig,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[tuple[CapabilityObservation, ...], dict[str, Any]]:
    """Validate configured LiDAR while keeping the vehicle grounded and inactive."""
    state = _require_method(client, "getMultirotorState")(
        vehicle_name=config.vehicle_name
    )
    position = extract_finite_position(state)
    velocity = _extract_finite_velocity(state)
    landed_state = _summarize_landed_state(state, client_module)
    api_control_enabled = _read_api_control_enabled(client, config.vehicle_name)
    collision_observations, collision_samples = sample_collision_information(
        client,
        vehicle_name=config.vehicle_name,
        sleep_fn=sleep_fn,
    )
    speed = math.sqrt(sum(component**2 for component in velocity))
    collision_assessment = classify_collision_samples(
        collision_samples,
        is_landed=landed_state["is_landed"],
        measured_speed=speed,
        api_control_enabled=api_control_enabled,
        operator_confirmed_stable=config.confirm_no_visible_collision,
    )
    validate_grounded_preflight(
        client,
        client_module,
        state,
        collision_assessment=collision_assessment,
        operator_confirmed_stable=config.confirm_no_visible_collision,
        vehicle_name=config.vehicle_name,
    )

    settings_text = _require_method(client, "getSettingsString")()
    if not isinstance(settings_text, str):
        raise CapabilityProbeError("Simulator settings response is not text.")
    settings = sanitize_settings(settings_text)
    comparisons = compare_lidar_settings_profile(settings)
    profile_matches = (
        config.vehicle_name == M13_LIDAR_PROVISIONAL_PROFILE.vehicle_name
        and config.lidar_name == M13_LIDAR_PROVISIONAL_PROFILE.lidar_name
        and all(item.matched for item in comparisons)
    )
    observations = list(collision_observations)
    observations.append(
        CapabilityObservation(
            "grounded_lidar_settings",
            "getSettingsString",
            EvidenceLevel.LIVE_RPC,
            (
                CapabilityStatus.RPC_SUCCEEDED
                if profile_matches
                else CapabilityStatus.REQUIRES_LOCAL_CONFIGURATION
            ),
            (
                "The exact provisional M13.1 LiDAR profile is active."
                if profile_matches
                else "The active vehicle or LiDAR settings do not match the exact "
                "provisional M13.1 profile."
            ),
        )
    )
    base_data: dict[str, Any] = {
        "vehicle_name": config.vehicle_name,
        "lidar_name": config.lidar_name,
        "grounded_preflight": {
            "position": _position_tuple(position),
            "linear_velocity": velocity,
            "speed": speed,
            "landed_state": landed_state,
            "api_control_enabled": api_control_enabled,
            "collision_assessment": collision_assessment,
            "armed_state": {
                "availability": "unavailable",
                "value": None,
            },
        },
        "settings_profile_matches": profile_matches,
        "settings_comparisons": comparisons,
        "configured_data_frame": _matched_setting_value(
            comparisons, "sensor", "DataFrame"
        ),
        "empirical_frame": "not_tested_grounded_only",
        "configured_range": _matched_setting_value(comparisons, "sensor", "Range"),
        "range_tolerance": LIDAR_RANGE_TOLERANCE,
        "warm_up": {
            "attempts": 0,
            "empty_count": 0,
            "invalid_count": 0,
            "time_to_first_valid_scan": None,
            "first_valid_timestamp": None,
            "succeeded": False,
            "excluded_from_measured_statistics": True,
        },
        "measured_scans": (),
        "ready_for_airborne_validation": False,
        "raw_point_clouds_persisted": False,
        "mount_validation_status": "provisional",
    }
    if not profile_matches:
        return tuple(observations), base_data

    configured_range = float(_matched_setting_value(comparisons, "sensor", "Range"))
    warm_up_start = clock()
    warm_up_empty = 0
    warm_up_invalid = 0
    first_valid: GroundedLidarScanSummary | None = None
    warm_up_attempts = 0
    for attempt in range(config.warm_up_attempts):
        warm_up_attempts = attempt + 1
        summary = validate_grounded_lidar_scan(
            _require_method(client, "getLidarData")(
                config.lidar_name, config.vehicle_name
            ),
            sensor_name=config.lidar_name,
            vehicle_name=config.vehicle_name,
            configured_range=configured_range,
        )
        if summary.valid:
            first_valid = summary
            break
        if summary.empty:
            warm_up_empty += 1
        else:
            warm_up_invalid += 1
        if attempt + 1 < config.warm_up_attempts and config.warm_up_interval:
            sleep_fn(config.warm_up_interval)

    elapsed_to_valid = clock() - warm_up_start if first_valid is not None else None
    base_data["warm_up"] = {
        "attempts": warm_up_attempts,
        "empty_count": warm_up_empty,
        "invalid_count": warm_up_invalid,
        "time_to_first_valid_scan": elapsed_to_valid,
        "first_valid_timestamp": (
            first_valid.timestamp if first_valid is not None else None
        ),
        "succeeded": first_valid is not None,
        "excluded_from_measured_statistics": True,
    }
    if first_valid is None:
        observations.append(
            CapabilityObservation(
                "grounded_lidar_warm_up",
                "getLidarData",
                EvidenceLevel.LIVE_RPC,
                CapabilityStatus.REQUIRES_LOCAL_CONFIGURATION,
                "No valid LiDAR scan appeared during bounded warm-up.",
            )
        )
        return tuple(observations), base_data

    measured_scans: list[GroundedLidarScanSummary] = []
    for index in range(config.scan_count):
        measured_scans.append(
            validate_grounded_lidar_scan(
                _require_method(client, "getLidarData")(
                    config.lidar_name, config.vehicle_name
                ),
                sensor_name=config.lidar_name,
                vehicle_name=config.vehicle_name,
                configured_range=configured_range,
            )
        )
        if index + 1 < config.scan_count and config.scan_interval:
            sleep_fn(config.scan_interval)

    timestamp_summary = analyze_lidar_timestamps(
        tuple(
            scan.timestamp if scan.timestamp is not None else 0
            for scan in measured_scans
        )
    )
    measured_valid = all(scan.valid for scan in measured_scans)
    timestamp_valid = (
        all(
            scan.timestamp is not None and scan.timestamp > 0 for scan in measured_scans
        )
        and timestamp_summary.regression_count == 0
        and timestamp_summary.maximum_repeated_timestamp_run <= config.stale_threshold
    )
    near_field = _summarize_near_field(measured_scans)
    self_hit = _classify_self_hits(measured_scans, near_field)
    ready = (
        measured_valid
        and timestamp_valid
        and self_hit is SelfHitClassification.NO_EVIDENT_SELF_HIT
    )
    ranges = [
        value
        for scan in measured_scans
        for value in (scan.minimum_range, scan.maximum_range)
        if value is not None
    ]
    base_data.update(
        {
            "measured_scans": tuple(measured_scans),
            "measured_scan_count": len(measured_scans),
            "all_measured_scans_valid": measured_valid,
            "timestamp_summary": timestamp_summary,
            "global_minimum_range": min(ranges) if ranges else None,
            "global_maximum_range": max(ranges) if ranges else None,
            "beyond_configured_range_count": sum(
                scan.beyond_configured_range_count for scan in measured_scans
            ),
            "near_field": near_field,
            "self_hit_classification": self_hit,
            "near_returns_recur": any(
                item["recurs_across_scans"] for item in near_field
            ),
            "ready_for_airborne_validation": ready,
        }
    )
    observations.append(
        CapabilityObservation(
            "grounded_lidar",
            "getLidarData",
            EvidenceLevel.LIVE_RPC,
            (
                CapabilityStatus.RPC_SUCCEEDED
                if ready
                else CapabilityStatus.SUPPORTED_WITH_LIMITATIONS
            ),
            (
                "Warm-up succeeded and all measured scans passed the airborne gate."
                if ready
                else "Grounded LiDAR evidence did not pass the airborne gate."
            ),
        )
    )
    return tuple(observations), base_data


def _collect_strict_lidar_warm_up(
    client: Any,
    *,
    vehicle_name: str,
    lidar_name: str,
    configured_range: float,
    attempts: int,
    interval: float,
    sleep_fn: Callable[[float], None],
) -> tuple[dict[str, Any], GroundedLidarScanSummary | None]:
    empty_count = 0
    invalid_count = 0
    first_valid: GroundedLidarScanSummary | None = None
    attempts_used = 0
    started = time.perf_counter()
    for index in range(attempts):
        attempts_used = index + 1
        summary = validate_grounded_lidar_scan(
            _require_method(client, "getLidarData")(lidar_name, vehicle_name),
            sensor_name=lidar_name,
            vehicle_name=vehicle_name,
            configured_range=configured_range,
        )
        if summary.valid:
            first_valid = summary
            break
        if summary.empty:
            empty_count += 1
        else:
            invalid_count += 1
        if index + 1 < attempts and interval:
            sleep_fn(interval)
    return (
        {
            "attempts": attempts_used,
            "empty_count": empty_count,
            "invalid_count": invalid_count,
            "time_to_first_valid_scan": (
                time.perf_counter() - started if first_valid is not None else None
            ),
            "first_valid_timestamp": (
                first_valid.timestamp if first_valid is not None else None
            ),
            "succeeded": first_valid is not None,
            "excluded_from_measured_statistics": True,
        },
        first_valid,
    )


def _collect_strict_lidar_scans(
    client: Any,
    *,
    vehicle_name: str,
    lidar_name: str,
    configured_range: float,
    count: int,
    interval: float,
    sleep_fn: Callable[[float], None],
) -> tuple[tuple[GroundedLidarScanSummary, ...], tuple[Any, ...]]:
    summaries: list[GroundedLidarScanSummary] = []
    raw_scans: list[Any] = []
    for index in range(count):
        raw_scan = _require_method(client, "getLidarData")(lidar_name, vehicle_name)
        raw_scans.append(raw_scan)
        summaries.append(
            validate_grounded_lidar_scan(
                raw_scan,
                sensor_name=lidar_name,
                vehicle_name=vehicle_name,
                configured_range=configured_range,
            )
        )
        if index + 1 < count and interval:
            sleep_fn(interval)
    return tuple(summaries), tuple(raw_scans)


def _run_coordinate_frame_experiment(
    client: Any,
    config: LidarProbeConfig,
    context: AirborneProbeContext,
    *,
    last_measured_timestamp: int,
    sleep_fn: Callable[[float], None],
) -> tuple[str, dict[str, Any]]:
    vehicle_name = context.vehicle_name
    _hover_and_monitor(client, context, config.airborne)
    if config.settle_interval:
        sleep_fn(config.settle_interval)
    initial_state = _require_method(client, "getMultirotorState")(
        vehicle_name=vehicle_name
    )
    monitor_airborne_safety(
        client,
        initial_state,
        context.ground_reference_z,
        context.anchor_position,
        config.airborne,
        collision_baseline_timestamp=context.initial_collision_timestamp,
    )
    initial_yaw = _extract_yaw_degrees(initial_state)
    baseline_scan, baseline_summary, baseline_attempts = _collect_newer_lidar_scan(
        client,
        config,
        context,
        newer_than=last_measured_timestamp,
        sleep_fn=sleep_fn,
    )
    if baseline_scan is None or baseline_summary is None:
        raise CapabilityProbeError(
            "Coordinate-frame baseline did not produce a fresh valid scan."
        )

    target_yaw = initial_yaw + config.yaw_delta_degrees
    rotated_scan: Any | None = None
    rotated_summary: GroundedLidarScanSummary | None = None
    rotated_attempts = 0
    returned_yaw: float | None = None
    yaw_error: float | None = None
    operation_error: BaseException | None = None
    restoration_error: BaseException | None = None
    try:
        _join_async(
            _require_method(client, "rotateToYawAsync")(
                target_yaw,
                timeout_sec=config.airborne.movement_timeout,
                margin=5.0,
                vehicle_name=vehicle_name,
            )
        )
        rotated_state = _require_method(client, "getMultirotorState")(
            vehicle_name=vehicle_name
        )
        monitor_airborne_safety(
            client,
            rotated_state,
            context.ground_reference_z,
            context.anchor_position,
            config.airborne,
            collision_baseline_timestamp=context.initial_collision_timestamp,
        )
        _hover_and_monitor(client, context, config.airborne)
        if config.settle_interval:
            sleep_fn(config.settle_interval)
        rotated_scan, rotated_summary, rotated_attempts = _collect_newer_lidar_scan(
            client,
            config,
            context,
            newer_than=baseline_summary.timestamp or 0,
            sleep_fn=sleep_fn,
            collect_all=True,
        )
        if rotated_scan is None or rotated_summary is None:
            raise CapabilityProbeError(
                "Coordinate-frame rotation did not produce a newer valid scan."
            )
    except BaseException as exc:
        operation_error = exc
    finally:
        try:
            _join_async(
                _require_method(client, "rotateToYawAsync")(
                    initial_yaw,
                    timeout_sec=config.airborne.movement_timeout,
                    margin=5.0,
                    vehicle_name=vehicle_name,
                )
            )
            _hover_and_monitor(client, context, config.airborne)
            returned_state = _require_method(client, "getMultirotorState")(
                vehicle_name=vehicle_name
            )
            monitor_airborne_safety(
                client,
                returned_state,
                context.ground_reference_z,
                context.anchor_position,
                config.airborne,
                collision_baseline_timestamp=context.initial_collision_timestamp,
            )
            returned_yaw = _extract_yaw_degrees(returned_state)
            yaw_error = _yaw_error_degrees(returned_yaw, initial_yaw)
            if yaw_error > config.yaw_return_tolerance_degrees:
                raise CapabilityProbeError(
                    "Coordinate-frame experiment did not restore the original yaw."
                )
        except BaseException as exc:
            restoration_error = exc

    if operation_error is not None:
        if restoration_error is not None:
            raise CapabilityProbeError(
                "Coordinate-frame operation failed and yaw restoration also failed."
            ) from operation_error
        raise operation_error
    if restoration_error is not None:
        raise restoration_error
    if rotated_scan is None or rotated_summary is None:
        raise CapabilityProbeError("Coordinate-frame rotated scan is unavailable.")

    frame_result = _classify_lidar_frame(baseline_scan, rotated_scan)
    return frame_result, {
        "requested": True,
        "initial_yaw": initial_yaw,
        "target_yaw": target_yaw,
        "returned_yaw": returned_yaw,
        "yaw_error": yaw_error,
        "yaw_return_tolerance": config.yaw_return_tolerance_degrees,
        "baseline_timestamp": baseline_summary.timestamp,
        "rotated_timestamp": rotated_summary.timestamp,
        "baseline_scan_attempts": baseline_attempts,
        "rotated_scan_attempts": rotated_attempts,
        "frame_result": frame_result,
        "conclusive": frame_result != "inconclusive",
    }


def _collect_newer_lidar_scan(
    client: Any,
    config: LidarProbeConfig,
    context: AirborneProbeContext,
    *,
    newer_than: int,
    sleep_fn: Callable[[float], None],
    collect_all: bool = False,
) -> tuple[Any | None, GroundedLidarScanSummary | None, int]:
    selected_scan: Any | None = None
    selected_summary: GroundedLidarScanSummary | None = None
    for index in range(config.coordinate_scan_attempts):
        raw_scan = _require_method(client, "getLidarData")(
            context.lidar_name, context.vehicle_name
        )
        summary = validate_grounded_lidar_scan(
            raw_scan,
            sensor_name=context.lidar_name,
            vehicle_name=context.vehicle_name,
            configured_range=context.configured_range,
        )
        if (
            summary.valid
            and summary.timestamp is not None
            and summary.timestamp > newer_than
        ):
            selected_scan = raw_scan
            selected_summary = summary
            if not collect_all:
                return selected_scan, selected_summary, index + 1
        if index + 1 < config.coordinate_scan_attempts and config.scan_interval:
            sleep_fn(config.scan_interval)
    return selected_scan, selected_summary, config.coordinate_scan_attempts


def _hover_and_monitor(
    client: Any,
    context: AirborneProbeContext,
    config: AirborneProbeConfig,
) -> Position3D:
    _join_async(
        _require_method(client, "hoverAsync")(vehicle_name=context.vehicle_name)
    )
    state = _require_method(client, "getMultirotorState")(
        vehicle_name=context.vehicle_name
    )
    return monitor_airborne_safety(
        client,
        state,
        context.ground_reference_z,
        context.anchor_position,
        config,
        collision_baseline_timestamp=context.initial_collision_timestamp,
    )


def transform_sensor_local_points_to_world(
    point_cloud: Sequence[float],
    sensor_position: tuple[float, float, float],
    sensor_orientation: tuple[float, float, float, float],
) -> tuple[Position3D, ...]:
    """Transform finite SensorLocalFrame XYZ triples through a finite world pose."""
    values = tuple(float(value) for value in point_cloud)
    if not values or len(values) % 3:
        raise ValueError("point_cloud must contain non-empty XYZ triples")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("point_cloud must contain only finite values")
    _validate_vector("sensor_position", sensor_position)
    if len(sensor_orientation) != 4 or not all(
        math.isfinite(value) for value in sensor_orientation
    ):
        raise ValueError("sensor_orientation must contain four finite values")

    w_value, x_value, y_value, z_value = sensor_orientation
    norm = math.sqrt(w_value**2 + x_value**2 + y_value**2 + z_value**2)
    if norm <= 0.0:
        raise ValueError("sensor_orientation quaternion must have positive norm")
    w_value /= norm
    x_value /= norm
    y_value /= norm
    z_value /= norm

    rotation = (
        (
            1.0 - 2.0 * (y_value**2 + z_value**2),
            2.0 * (x_value * y_value - z_value * w_value),
            2.0 * (x_value * z_value + y_value * w_value),
        ),
        (
            2.0 * (x_value * y_value + z_value * w_value),
            1.0 - 2.0 * (x_value**2 + z_value**2),
            2.0 * (y_value * z_value - x_value * w_value),
        ),
        (
            2.0 * (x_value * z_value - y_value * w_value),
            2.0 * (y_value * z_value + x_value * w_value),
            1.0 - 2.0 * (x_value**2 + y_value**2),
        ),
    )
    transformed: list[Position3D] = []
    for index in range(0, len(values), 3):
        local_x, local_y, local_z = values[index : index + 3]
        transformed.append(
            Position3D(
                sensor_position[0]
                + rotation[0][0] * local_x
                + rotation[0][1] * local_y
                + rotation[0][2] * local_z,
                sensor_position[1]
                + rotation[1][0] * local_x
                + rotation[1][1] * local_y
                + rotation[1][2] * local_z,
                sensor_position[2]
                + rotation[2][0] * local_x
                + rotation[2][1] * local_y
                + rotation[2][2] * local_z,
            )
        )
    return tuple(transformed)


def _evenly_sample_values(
    values: Sequence[Position3D],
    maximum_count: int,
) -> tuple[Position3D, ...]:
    if maximum_count <= 0 or not values:
        return ()
    selected_count = min(len(values), maximum_count)
    if selected_count == len(values):
        return tuple(values)
    if selected_count == 1:
        return (values[0],)
    return tuple(
        values[round(index * (len(values) - 1) / (selected_count - 1))]
        for index in range(selected_count)
    )


def visualize_validated_lidar_scan(
    client: Any,
    client_module: ModuleType,
    raw_scan: Any,
    scan_summary: GroundedLidarScanSummary,
    config: LidarProbeConfig,
    context: AirborneProbeContext,
    runtime: ProbeRuntimeState,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    message_fn: Callable[[str], None] = print,
) -> tuple[tuple[CapabilityObservation, ...], dict[str, Any]]:
    """Plot a bounded diagnostic overlay from one strictly validated scan."""
    if not config.visualize_lidar:
        return (), {
            "requested": False,
            "markers_created": False,
        }
    if not config.allow_marker_flush:
        raise ValueError(
            "LiDAR visualization requires explicit marker-flush authorization"
        )
    if not scan_summary.valid:
        raise CapabilityProbeError(
            "LiDAR visualization requires a strictly validated scan."
        )
    if (
        scan_summary.sensor_name != context.lidar_name
        or scan_summary.vehicle_name != context.vehicle_name
    ):
        raise CapabilityProbeError(
            "Validated scan names do not match the airborne context."
        )
    if scan_summary.sensor_position is None or scan_summary.sensor_orientation is None:
        raise CapabilityProbeError(
            "Validated scan is missing a finite complete LiDAR pose."
        )

    cloud = getattr(raw_scan, "point_cloud", None)
    if cloud is None:
        raise CapabilityProbeError("Validated scan has no point cloud.")
    world_points = transform_sensor_local_points_to_world(
        cloud,
        scan_summary.sensor_position,
        scan_summary.sensor_orientation,
    )
    if len(world_points) != scan_summary.point_count:
        raise CapabilityProbeError(
            "Validated scan point count changed before visualization."
        )
    plotted_points = _evenly_sample_values(
        world_points,
        config.visualization_max_points,
    )
    ray_hits = _evenly_sample_values(
        world_points,
        config.visualization_max_rays,
    )
    vector = _require_client_type(client_module, "Vector3r")
    point_vectors = [vector(point.x, point.y, point.z) for point in plotted_points]
    sensor_origin = vector(*scan_summary.sensor_position)
    ray_vectors = [
        vector_value
        for point in ray_hits
        for vector_value in (
            sensor_origin,
            vector(point.x, point.y, point.z),
        )
    ]

    _hover_and_monitor(client, context, config.airborne)
    runtime.lidar_visualization_markers_created = True
    observations: list[CapabilityObservation] = []
    point_observation, _ = invoke_capability(
        client,
        "lidar_visualization_points",
        "simPlotPoints",
        point_vectors,
        [1.0, 0.0, 1.0, 1.0],
        5.0,
        -1.0,
        True,
    )
    observations.append(replace(point_observation, operator_confirmation="pending"))
    _require_rpc_success(point_observation)
    if ray_vectors:
        ray_observation, _ = invoke_capability(
            client,
            "lidar_visualization_rays",
            "simPlotLineList",
            ray_vectors,
            [0.0, 1.0, 1.0, 1.0],
            1.5,
            -1.0,
            True,
        )
        observations.append(replace(ray_observation, operator_confirmation="pending"))
        _require_rpc_success(ray_observation)

    message_fn(
        "The diagnostic LiDAR point/ray overlay is currently visible for "
        f"{config.visualization_hold_seconds:g} seconds before cleanup."
    )
    if config.visualization_hold_seconds:
        sleep_fn(config.visualization_hold_seconds)
    _hover_and_monitor(client, context, config.airborne)
    return tuple(observations), {
        "requested": True,
        "markers_created": True,
        "source_scan_validated": True,
        "source_scan_timestamp": scan_summary.timestamp,
        "source_frame": "SensorLocalFrame",
        "world_transform_applied": True,
        "source_point_count": scan_summary.point_count,
        "plotted_point_count": len(plotted_points),
        "diagnostic_ray_count": len(ray_hits),
        "hold_seconds": config.visualization_hold_seconds,
        "point_limit": config.visualization_max_points,
        "ray_limit": config.visualization_max_rays,
        "diagnostic_overlay_not_physical_lasers": True,
        "operator_visibility_confirmation": "pending",
    }


def probe_lidar(
    client: Any,
    config: LidarProbeConfig,
    context: AirborneProbeContext,
    *,
    client_module: ModuleType | None = None,
    runtime: ProbeRuntimeState | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    message_fn: Callable[[str], None] = print,
) -> tuple[tuple[CapabilityObservation, ...], dict[str, Any]]:
    """Collect strict airborne scans and optionally perform a yaw-frame experiment."""
    if context.vehicle_name != config.airborne.vehicle_name:
        raise ValueError("Airborne context and configuration vehicle names differ")
    if context.lidar_name != config.lidar_name:
        raise ValueError("Airborne context and configuration LiDAR names differ")

    if config.settle_interval:
        sleep_fn(config.settle_interval)
    warm_up, first_valid = _collect_strict_lidar_warm_up(
        client,
        vehicle_name=context.vehicle_name,
        lidar_name=context.lidar_name,
        configured_range=context.configured_range,
        attempts=config.warm_up_attempts,
        interval=config.warm_up_interval,
        sleep_fn=sleep_fn,
    )
    observations: list[CapabilityObservation] = []
    if first_valid is None:
        observations.append(
            CapabilityObservation(
                "airborne_lidar_warm_up",
                "getLidarData",
                EvidenceLevel.LIVE_RPC,
                CapabilityStatus.SUPPORTED_WITH_LIMITATIONS,
                "No valid airborne LiDAR scan appeared during bounded warm-up.",
            )
        )
        return tuple(observations), {
            "vehicle_name": context.vehicle_name,
            "lidar_name": context.lidar_name,
            "warm_up": warm_up,
            "scans": (),
            "valid_scan_count": 0,
            "invalid_scan_count": 0,
            "empty_scan_count": 0,
            "configured_range": context.configured_range,
            "range_tolerance": LIDAR_RANGE_TOLERANCE,
            "airborne_scan_gate_passed": False,
            "coordinate_frame_result": "not_requested",
            "visualization": {
                "requested": config.visualize_lidar,
                "performed": False,
                "reason": "airborne warm-up did not produce a valid scan",
            },
            "raw_point_clouds_persisted": False,
        }

    scans, raw_scans = _collect_strict_lidar_scans(
        client,
        vehicle_name=context.vehicle_name,
        lidar_name=context.lidar_name,
        configured_range=context.configured_range,
        count=config.scan_count,
        interval=config.scan_interval,
        sleep_fn=sleep_fn,
    )
    timestamp_summary = analyze_lidar_timestamps(
        tuple(scan.timestamp or 0 for scan in scans)
    )
    valid_scan_count = sum(scan.valid for scan in scans)
    empty_scan_count = sum(scan.empty for scan in scans)
    invalid_scan_count = len(scans) - valid_scan_count
    timestamps_pass = (
        all(scan.timestamp is not None and scan.timestamp > 0 for scan in scans)
        and timestamp_summary.regression_count == 0
        and timestamp_summary.maximum_repeated_timestamp_run <= config.stale_threshold
    )
    scan_gate_passed = valid_scan_count == config.scan_count and timestamps_pass
    near_field = _summarize_near_field(scans)
    all_minimums = [
        scan.minimum_range for scan in scans if scan.minimum_range is not None
    ]
    all_maximums = [
        scan.maximum_range for scan in scans if scan.maximum_range is not None
    ]
    observations.append(
        CapabilityObservation(
            "lidar",
            "getLidarData",
            EvidenceLevel.LIVE_RPC,
            (
                CapabilityStatus.RPC_SUCCEEDED
                if scan_gate_passed
                else CapabilityStatus.SUPPORTED_WITH_LIMITATIONS
            ),
            (
                f"Collected {valid_scan_count}/{len(scans)} strict valid scans; "
                f"maximum repeated timestamp run "
                f"{timestamp_summary.maximum_repeated_timestamp_run}."
            ),
        )
    )

    frame_result = "not_requested"
    coordinate_data: dict[str, Any] = {
        "requested": config.coordinate_frame_experiment,
        "conclusive": False,
    }
    if config.coordinate_frame_experiment and scan_gate_passed:
        frame_result, coordinate_data = _run_coordinate_frame_experiment(
            client,
            config,
            context,
            last_measured_timestamp=timestamp_summary.timestamps[-1],
            sleep_fn=sleep_fn,
        )
        observations.append(
            CapabilityObservation(
                "lidar_frame",
                "getLidarData",
                EvidenceLevel.LIVE_RPC,
                (
                    CapabilityStatus.INCONCLUSIVE
                    if frame_result == "inconclusive"
                    else CapabilityStatus.SUPPORTED_WITH_LIMITATIONS
                ),
                (
                    f"Numeric frame evidence suggests {frame_result}; "
                    "operator review required."
                ),
                operator_confirmation="pending",
            )
        )

    visualization_data: dict[str, Any] = {
        "requested": config.visualize_lidar,
        "performed": False,
    }
    if config.visualize_lidar:
        if not scan_gate_passed:
            visualization_data["reason"] = (
                "strict airborne scan validation did not pass"
            )
        else:
            if client_module is None or runtime is None:
                raise ValueError(
                    "LiDAR visualization requires client_module and runtime"
                )
            visualization_observations, visualization_data = (
                visualize_validated_lidar_scan(
                    client,
                    client_module,
                    raw_scans[-1],
                    scans[-1],
                    config,
                    context,
                    runtime,
                    sleep_fn=sleep_fn,
                    message_fn=message_fn,
                )
            )
            visualization_data["performed"] = True
            observations.extend(visualization_observations)

    return tuple(observations), {
        "vehicle_name": context.vehicle_name,
        "lidar_name": context.lidar_name,
        "settings_comparisons": context.settings_comparisons,
        "warm_up": warm_up,
        "scans": scans,
        "valid_scan_count": valid_scan_count,
        "invalid_scan_count": invalid_scan_count,
        "empty_scan_count": empty_scan_count,
        "point_counts": tuple(scan.point_count for scan in scans),
        "global_minimum_range": min(all_minimums) if all_minimums else None,
        "global_maximum_range": max(all_maximums) if all_maximums else None,
        "configured_range": context.configured_range,
        "range_tolerance": LIDAR_RANGE_TOLERANCE,
        "beyond_configured_range_count": sum(
            scan.beyond_configured_range_count for scan in scans
        ),
        "timestamp_summary": timestamp_summary,
        "near_field": near_field,
        "airborne_scan_gate_passed": scan_gate_passed,
        "coordinate_frame_result": frame_result,
        "coordinate_frame": coordinate_data,
        "visualization": visualization_data,
        "raw_point_clouds_persisted": False,
        "raw_cloud_count_used_for_summary": len(raw_scans),
    }


def probe_performance(
    client: Any,
    config: PerformanceProbeConfig,
    *,
    airborne_context: AirborneProbeContext | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[PerformanceSummary, ...]:
    """Measure bounded RPC operations without changing simulator settings."""
    results = [
        _benchmark_calls(
            "multirotor_state",
            config.iterations,
            lambda: _require_method(client, "getMultirotorState")(
                vehicle_name=config.vehicle_name
            ),
            clock,
        ),
        _benchmark_calls(
            "scene_object_listing",
            config.iterations,
            lambda: _require_method(client, "simListSceneObjects")(".*"),
            clock,
        ),
    ]
    if config.include_lidar:
        results.append(
            _benchmark_calls(
                "lidar_scan",
                config.iterations,
                lambda: _require_method(client, "getLidarData")(
                    config.lidar_name, config.vehicle_name
                ),
                clock,
            )
        )
    if config.include_control:
        if airborne_context is None:
            raise ValueError("control benchmark requires an airborne context")
        results.append(
            _benchmark_calls(
                "zero_velocity_control",
                config.iterations,
                lambda: _run_zero_velocity_control_step(
                    client,
                    config,
                    airborne_context,
                ),
                clock,
            )
        )
    return tuple(results)


def inspect_active_lidar_settings(
    client: Any,
    vehicle_name: str,
    lidar_name: str,
) -> dict[str, Any]:
    """Read and compare the active settings without acquiring vehicle control."""
    if not vehicle_name.strip():
        raise ValueError("vehicle_name must not be empty")
    if not lidar_name.strip():
        raise ValueError("lidar_name must not be empty")
    settings_text = _require_method(client, "getSettingsString")()
    if not isinstance(settings_text, str):
        raise CapabilityProbeError("Simulator settings response is not text.")
    settings = sanitize_settings(settings_text)
    comparisons = compare_lidar_settings_profile(settings)
    names_match = (
        vehicle_name == M13_LIDAR_PROVISIONAL_PROFILE.vehicle_name
        and lidar_name == M13_LIDAR_PROVISIONAL_PROFILE.lidar_name
    )
    profile_matches = names_match and all(
        comparison.matched for comparison in comparisons
    )
    configured_range = _matched_setting_value(comparisons, "sensor", "Range")
    return {
        "vehicle_name": vehicle_name,
        "lidar_name": lidar_name,
        "profile_matches": profile_matches,
        "comparisons": comparisons,
        "configured_range": configured_range,
    }


def prepare_airborne_probe(
    client: Any,
    client_module: ModuleType,
    config: AirborneProbeConfig,
    runtime: ProbeRuntimeState,
    lidar_name: str,
    *,
    settings_verification: dict[str, Any] | None = None,
) -> AirborneProbeContext:
    """Establish a measured-relative airborne anchor with M12 cleanup tracking."""
    if not (
        config.allow_flight
        and config.confirm_clear_airspace
        and config.confirm_no_visible_collision
        and config.confirm_grounded_lidar_passed
    ):
        raise ValueError(
            "flight requires authorization, clear-airspace confirmation, "
            "no-visible-collision confirmation, and grounded-LiDAR confirmation"
        )
    if not lidar_name.strip():
        raise ValueError("lidar_name must not be empty")
    verification = settings_verification or inspect_active_lidar_settings(
        client, config.vehicle_name, lidar_name
    )
    if not verification["profile_matches"]:
        raise CapabilityProbeError(
            "Active settings do not match the exact provisional M13.1 profile."
        )
    configured_range = verification["configured_range"]
    if not isinstance(configured_range, (int, float)) or isinstance(
        configured_range, bool
    ):
        raise CapabilityProbeError("Configured LiDAR range is unavailable.")

    runtime.vehicle_name = config.vehicle_name
    state = _require_method(client, "getMultirotorState")(
        vehicle_name=config.vehicle_name
    )
    _, collision_samples = sample_collision_information(
        client, vehicle_name=config.vehicle_name
    )
    ground_velocity = _extract_finite_velocity(state)
    landed_state = _summarize_landed_state(state, client_module)
    api_control_enabled = _read_api_control_enabled(client, config.vehicle_name)
    collision_assessment = classify_collision_samples(
        collision_samples,
        is_landed=landed_state["is_landed"],
        measured_speed=math.sqrt(sum(value**2 for value in ground_velocity)),
        api_control_enabled=api_control_enabled,
        operator_confirmed_stable=config.confirm_no_visible_collision,
    )
    ground_position = validate_grounded_preflight(
        client,
        client_module,
        state,
        collision_assessment=collision_assessment,
        operator_confirmed_stable=config.confirm_no_visible_collision,
        vehicle_name=config.vehicle_name,
    )
    target = Position3D(
        ground_position.x,
        ground_position.y,
        ground_position.z - config.anchor_altitude,
    )
    clearance = ground_position.z - target.z
    if clearance < config.min_ground_clearance:
        raise CapabilityProbeError("Planned anchor violates minimum ground clearance.")

    _require_method(client, "enableApiControl")(True, vehicle_name=config.vehicle_name)
    runtime.cleanup_state = replace(runtime.cleanup_state, api_control_enabled=True)
    _require_method(client, "armDisarm")(True, vehicle_name=config.vehicle_name)
    runtime.cleanup_state = replace(runtime.cleanup_state, armed=True)
    runtime.cleanup_state = replace(runtime.cleanup_state, takeoff_attempted=True)
    _join_async(
        _require_method(client, "takeoffAsync")(
            timeout_sec=config.movement_timeout,
            vehicle_name=config.vehicle_name,
        )
    )
    runtime.cleanup_state = replace(runtime.cleanup_state, airborne=True)
    _join_async(
        _require_method(client, "moveToPositionAsync")(
            target.x,
            target.y,
            target.z,
            config.anchor_velocity,
            timeout_sec=config.movement_timeout,
            vehicle_name=config.vehicle_name,
        )
    )
    measured_state = _require_method(client, "getMultirotorState")(
        vehicle_name=config.vehicle_name
    )
    anchor = monitor_airborne_safety(
        client,
        measured_state,
        ground_position.z,
        target,
        config,
        collision_baseline_timestamp=collision_assessment.baseline_timestamp,
    )
    if calculate_position_error(anchor, target) > config.movement_tolerance:
        raise CapabilityProbeError("Measured anchor exceeds movement tolerance.")
    _join_async(_require_method(client, "hoverAsync")(vehicle_name=config.vehicle_name))
    hovered_state = _require_method(client, "getMultirotorState")(
        vehicle_name=config.vehicle_name
    )
    hovered_position = monitor_airborne_safety(
        client,
        hovered_state,
        ground_position.z,
        target,
        config,
        collision_baseline_timestamp=collision_assessment.baseline_timestamp,
    )
    if calculate_position_error(hovered_position, target) > config.movement_tolerance:
        raise CapabilityProbeError("Hovered anchor exceeds movement tolerance.")
    return AirborneProbeContext(
        config.vehicle_name,
        lidar_name,
        ground_position,
        ground_position.z,
        hovered_position,
        collision_assessment.baseline_timestamp,
        float(configured_range),
        tuple(verification["comparisons"]),
    )


def validate_grounded_preflight(
    client: Any,
    client_module: ModuleType,
    state: Any,
    *,
    collision_assessment: CollisionAssessment | None = None,
    operator_confirmed_stable: bool = False,
    vehicle_name: str = "",
) -> Position3D:
    """Reject unsafe grounded states before scene mutation or flight."""
    position = extract_finite_position(state)
    velocity = _extract_finite_velocity(state)
    landed_value = getattr(state, "landed_state", None)
    landed_type = getattr(client_module, "LandedState", None)
    expected_landed = getattr(landed_type, "Landed", 0)
    if landed_value != expected_landed:
        raise CapabilityProbeError("Vehicle must report a grounded/landed state.")
    api_control_enabled = _read_api_control_enabled(client, vehicle_name)
    if api_control_enabled:
        raise CapabilityProbeError(
            "API control must be disabled during grounded preflight."
        )
    if collision_assessment is None:
        _, collision_samples = sample_collision_information(
            client, vehicle_name=vehicle_name
        )
        collision_assessment = classify_collision_samples(
            collision_samples,
            is_landed=True,
            measured_speed=math.sqrt(sum(value**2 for value in velocity)),
            api_control_enabled=api_control_enabled,
            operator_confirmed_stable=operator_confirmed_stable,
        )
    if collision_assessment.classification not in {
        CollisionClassification.NO_COLLISION,
        CollisionClassification.EXPECTED_GROUND_CONTACT,
    }:
        raise CapabilityProbeError(
            "Grounded collision evidence is unsafe or inconclusive."
        )
    return position


def _read_api_control_enabled(client: Any, vehicle_name: str = "") -> bool:
    method = getattr(client, "isApiControlEnabled", None)
    if method is None:
        raise CapabilityProbeError(
            "API-control state is unavailable during grounded preflight."
        )
    value = method(vehicle_name=vehicle_name)
    if not isinstance(value, bool):
        raise CapabilityProbeError("API-control query returned a non-boolean value.")
    return value


def monitor_airborne_safety(
    client: Any,
    state: Any,
    ground_reference_z: float,
    anchor: Position3D,
    config: AirborneProbeConfig,
    *,
    collision_baseline_timestamp: int | None = None,
) -> Position3D:
    """Validate measured state after every airborne probe movement."""
    position = extract_finite_position(state)
    _extract_finite_velocity(state)
    collision_sample = sanitize_collision_info(
        _require_method(client, "simGetCollisionInfo")(vehicle_name=config.vehicle_name)
    )
    if collision_sample.has_collided is None:
        raise CapabilityProbeError("Airborne collision state is inconclusive.")
    if collision_sample.has_collided:
        if (
            collision_baseline_timestamp is None
            or collision_sample.time_stamp is None
            or collision_sample.time_stamp != collision_baseline_timestamp
        ):
            raise CapabilityProbeError(
                "A new or ambiguous collision was detected during airborne probe."
            )
    clearance = ground_reference_z - position.z
    if clearance < config.min_ground_clearance:
        raise CapabilityProbeError("Airborne probe violated minimum ground clearance.")
    if abs(position.x - anchor.x) > config.workspace_xy_limit:
        raise CapabilityProbeError("Airborne probe exceeded x workspace limit.")
    if abs(position.y - anchor.y) > config.workspace_xy_limit:
        raise CapabilityProbeError("Airborne probe exceeded y workspace limit.")
    if abs(position.z - anchor.z) > config.workspace_z_limit:
        raise CapabilityProbeError("Airborne probe exceeded z workspace limit.")
    return position


def cleanup_probe_domains(
    client: Any | None,
    runtime: ProbeRuntimeState,
) -> tuple[CleanupDomainResult, ...]:
    """Attempt UAV, object, and marker cleanup independently."""
    results: list[CleanupDomainResult] = []

    uav_errors: list[str] = []
    uav_succeeded = True
    uav_attempted = bool(client is not None and runtime.cleanup_state != CleanupState())
    if uav_attempted and client is not None:
        try:
            if not runtime.vehicle_name.strip():
                raise CapabilityProbeError(
                    "Named UAV cleanup requires a non-empty vehicle name."
                )
            uav_errors.extend(
                cleanup_named_probe_vehicle(
                    client,
                    runtime.cleanup_state,
                    runtime.vehicle_name,
                )
            )
            uav_succeeded = not uav_errors
        except BaseException as exc:
            uav_succeeded = False
            uav_errors.append(f"UAV cleanup raised {type(exc).__name__}")
    results.append(
        CleanupDomainResult(
            "uav",
            uav_attempted,
            uav_succeeded,
            tuple(uav_errors),
        )
    )

    object_errors: list[str] = []
    object_attempted = bool(client is not None and runtime.created_objects)
    if client is not None:
        for object_name in reversed(runtime.created_objects):
            try:
                destroyed = _require_method(client, "simDestroyObject")(object_name)
                if destroyed is False:
                    raise CapabilityProbeError("simDestroyObject reported failure")
                remaining = _require_method(client, "simListSceneObjects")(
                    _exact_object_regex(object_name)
                )
                if object_name in {str(name) for name in remaining}:
                    raise CapabilityProbeError("temporary object still exists")
            except BaseException as exc:
                object_errors.append(
                    f"Object cleanup for {object_name!r} raised {type(exc).__name__}"
                )
    results.append(
        CleanupDomainResult(
            "objects", object_attempted, not object_errors, tuple(object_errors)
        )
    )

    marker_errors: list[str] = []
    marker_attempted = bool(
        client is not None
        and (runtime.markers_created or runtime.lidar_visualization_markers_created)
    )
    if marker_attempted and client is not None:
        try:
            _require_method(client, "simFlushPersistentMarkers")()
        except BaseException as exc:
            marker_errors.append(f"Marker cleanup raised {type(exc).__name__}")
    results.append(
        CleanupDomainResult(
            "markers", marker_attempted, not marker_errors, tuple(marker_errors)
        )
    )
    return tuple(results)


def cleanup_named_probe_vehicle(
    client: Any,
    cleanup_state: CleanupState,
    vehicle_name: str,
) -> tuple[str, ...]:
    """Clean up one explicitly named probe vehicle without using M12 defaults."""
    if not vehicle_name.strip():
        raise ValueError("vehicle_name must not be empty")
    errors: list[str] = []

    def attempt(
        action: str,
        operation: Callable[[], Any],
        *,
        join: bool = False,
    ) -> None:
        try:
            result = operation()
            if join:
                _join_async(result)
        except BaseException as exc:
            errors.append(f"{action} cleanup raised {type(exc).__name__}")

    flight_attempted = cleanup_state.takeoff_attempted or cleanup_state.airborne
    if flight_attempted:
        attempt(
            "hover",
            lambda: _require_method(client, "hoverAsync")(vehicle_name=vehicle_name),
            join=True,
        )
        attempt(
            "land",
            lambda: _require_method(client, "landAsync")(vehicle_name=vehicle_name),
            join=True,
        )
    if cleanup_state.armed:
        attempt(
            "disarm",
            lambda: _require_method(client, "armDisarm")(
                False, vehicle_name=vehicle_name
            ),
        )
    if cleanup_state.api_control_enabled:
        attempt(
            "disable API control",
            lambda: _require_method(client, "enableApiControl")(
                False, vehicle_name=vehicle_name
            ),
        )
    return tuple(errors)


def generate_report_path(
    mode: str,
    output_dir: Path = DEFAULT_REPORTS_DIR,
    *,
    timestamp: str,
    run_id: str,
) -> Path:
    """Build a unique report filename without writing it."""
    safe_mode = re.sub(r"[^a-z0-9_-]+", "-", mode.lower()).strip("-")
    safe_timestamp = re.sub(r"[^0-9]+", "", timestamp)
    return output_dir / f"colosseum_{safe_mode}_{safe_timestamp}_{run_id[:8]}.json"


def validate_report_output_path(path: Path, repository_root: Path) -> None:
    """Refuse tracked or non-ignored report paths inside the repository."""
    resolved_path = path.resolve()
    resolved_root = repository_root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        return

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative.as_posix()],
        cwd=resolved_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if tracked.returncode == 0:
        raise ValueError("Report output path is already tracked by Git.")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", relative.as_posix()],
        cwd=resolved_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if ignored.returncode != 0:
        raise ValueError("Report output path inside the repository must be ignored.")


def save_capability_report(report: CapabilityProbeReport, output_path: Path) -> Path:
    """Write an indented JSON report after the CLI validates its destination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _sanitize_collision_object_name(value: Any, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        errors.append("object_name is unavailable")
        return None
    name = "".join(character for character in value.strip() if character.isprintable())
    if re.match(r"^[A-Za-z]:[\\/]", name) or name.startswith(("\\\\", "//")):
        errors.append("object_name resembled a local path and was redacted")
        return "[redacted_path_like_value]"
    return name[:256]


def _optional_collision_int(value: Any, label: str, errors: list[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{label} is unavailable or malformed")
        return None


def _optional_collision_float(
    value: Any, label: str, errors: list[str]
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{label} is unavailable or malformed")
        return None
    if not math.isfinite(result):
        errors.append(f"{label} is non-finite")
        return None
    return result


def _optional_collision_vector(
    value: Any,
    label: str,
    errors: list[str],
) -> tuple[float, float, float] | None:
    try:
        return _extract_vector(value, f"collision {label}")
    except CapabilityProbeError:
        errors.append(f"{label} is unavailable, malformed, or non-finite")
        return None


def _values_changed(values: Sequence[Any]) -> bool | None:
    if any(value is None for value in values):
        return None
    return len(set(values)) > 1


def _float_values_changed(values: Sequence[float | None]) -> bool | None:
    if not values or any(value is None for value in values):
        return None
    first = float(values[0])
    return any(not math.isclose(first, float(value), abs_tol=1e-6) for value in values)


def _vector_values_changed(
    values: Sequence[tuple[float, float, float] | None],
) -> bool | None:
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return any(
        any(
            not math.isclose(reference, current, abs_tol=1e-6)
            for reference, current in zip(first, value, strict=True)
        )
        for value in values[1:]
    )


def _latest_timestamp(samples: Sequence[CollisionInfoSample]) -> int | None:
    timestamps = [
        sample.time_stamp
        for sample in samples
        if sample.time_stamp is not None and sample.time_stamp > 0
    ]
    return max(timestamps) if timestamps else None


def _build_collision_assessment(
    samples: Sequence[CollisionInfoSample],
    classification: CollisionClassification,
    *,
    persistent_or_historical: bool,
    detail: str,
) -> CollisionAssessment:
    return CollisionAssessment(
        classification=classification,
        timestamp_changed=_values_changed([sample.time_stamp for sample in samples]),
        object_changed=_values_changed([sample.object_name for sample in samples]),
        object_id_changed=_values_changed([sample.object_id for sample in samples]),
        penetration_changed=_float_values_changed(
            [sample.penetration_depth for sample in samples]
        ),
        impact_point_changed=_vector_values_changed(
            [sample.impact_point for sample in samples]
        ),
        vehicle_position_changed=_vector_values_changed(
            [sample.vehicle_position for sample in samples]
        ),
        normal_changed=_vector_values_changed([sample.normal for sample in samples]),
        persistent_or_historical=persistent_or_historical,
        baseline_timestamp=_latest_timestamp(samples),
        detail=detail,
    )


def _inconclusive_collision_assessment(
    samples: Sequence[CollisionInfoSample], detail: str
) -> CollisionAssessment:
    return _build_collision_assessment(
        samples,
        CollisionClassification.INCONCLUSIVE_COLLISION,
        persistent_or_historical=False,
        detail=detail,
    )


def _unsafe_collision_assessment(
    samples: Sequence[CollisionInfoSample], detail: str
) -> CollisionAssessment:
    return _build_collision_assessment(
        samples,
        CollisionClassification.ACTIVE_OR_UNSAFE_COLLISION,
        persistent_or_historical=False,
        detail=detail,
    )


def _survey_object(client: Any, name: str) -> SceneObjectObservation:
    errors: list[str] = []
    position: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float, float] | None = None
    scale: tuple[float, float, float] | None = None
    try:
        pose = _require_method(client, "simGetObjectPose")(name)
        position = _extract_vector(pose.position, "object position")
        orientation = _extract_quaternion(pose.orientation, "object orientation")
    except Exception as exc:
        errors.append(f"pose read failed: {type(exc).__name__}")
    try:
        scale = _extract_vector(
            _require_method(client, "simGetObjectScale")(name), "object scale"
        )
    except Exception as exc:
        errors.append(f"scale read failed: {type(exc).__name__}")
    return SceneObjectObservation(name, position, orientation, scale, tuple(errors))


def _benchmark_calls(
    operation: str,
    iterations: int,
    operation_fn: Callable[[], Any],
    clock: Callable[[], float],
) -> PerformanceSummary:
    durations: list[float] = []
    errors: list[str] = []
    total_started = clock()
    for _ in range(iterations):
        started = clock()
        try:
            operation_fn()
        except Exception as exc:
            errors.append(type(exc).__name__)
            if classify_rpc_exception(exc) is CapabilityStatus.RPC_TIMED_OUT:
                raise CapabilityProbeError(
                    f"{operation} timed out; stop this process before further probes."
                ) from exc
        else:
            durations.append(max(0.0, clock() - started))
    total = max(0.0, clock() - total_started)
    succeeded = len(durations)
    return PerformanceSummary(
        operation=operation,
        attempted=iterations,
        succeeded=succeeded,
        total_seconds=total,
        calls_per_second=succeeded / total if total > 0.0 else 0.0,
        minimum_seconds=min(durations) if durations else None,
        mean_seconds=statistics.fmean(durations) if durations else None,
        p95_seconds=_percentile(durations, 0.95) if durations else None,
        maximum_seconds=max(durations) if durations else None,
        errors=tuple(errors),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _maximum_repeated_timestamp_run(timestamps: Sequence[int] | Any) -> int:
    return analyze_lidar_timestamps(tuple(timestamps)).maximum_repeated_timestamp_run


def _run_zero_velocity_control_step(
    client: Any,
    config: PerformanceProbeConfig,
    context: AirborneProbeContext,
) -> None:
    if config.airborne is None:
        raise ValueError("control benchmark requires airborne configuration")
    _join_async(
        _require_method(client, "moveByVelocityAsync")(
            0.0,
            0.0,
            0.0,
            config.control_duration,
            vehicle_name=config.vehicle_name,
        )
    )
    state = _require_method(client, "getMultirotorState")(
        vehicle_name=config.vehicle_name
    )
    monitor_airborne_safety(
        client,
        state,
        context.ground_reference_z,
        context.anchor_position,
        config.airborne,
        collision_baseline_timestamp=context.initial_collision_timestamp,
    )


def _extract_yaw_degrees(state: Any) -> float:
    orientation = getattr(
        getattr(state, "kinematics_estimated", None), "orientation", None
    )
    w_value, x_value, y_value, z_value = _extract_quaternion(
        orientation, "measured orientation"
    )
    sin_yaw = 2.0 * (w_value * z_value + x_value * y_value)
    cos_yaw = 1.0 - 2.0 * (y_value**2 + z_value**2)
    return math.degrees(math.atan2(sin_yaw, cos_yaw))


def _yaw_error_degrees(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def _classify_lidar_frame(first_scan: Any, second_scan: Any) -> str:
    first_raw = _cloud_centroid(first_scan.point_cloud)
    second_raw = _cloud_centroid(second_scan.point_cloud)
    raw_delta = math.dist(first_raw, second_raw)
    first_world = _transform_centroid(first_raw, first_scan.pose)
    second_world = _transform_centroid(second_raw, second_scan.pose)
    transformed_delta = math.dist(first_world, second_world)
    margin = 0.1
    if transformed_delta + margin < raw_delta:
        return "SensorLocalFrame"
    if raw_delta + margin < transformed_delta:
        return "VehicleInertialFrame"
    return "inconclusive"


def _cloud_centroid(point_cloud: Sequence[float]) -> tuple[float, float, float]:
    values = [float(value) for value in point_cloud]
    if not values or len(values) % 3:
        raise CapabilityProbeError("Cannot calculate centroid from malformed cloud.")
    count = len(values) // 3
    return (
        sum(values[0::3]) / count,
        sum(values[1::3]) / count,
        sum(values[2::3]) / count,
    )


def _transform_centroid(
    centroid: tuple[float, float, float], pose: Any
) -> tuple[float, float, float]:
    w_value, x_value, y_value, z_value = _extract_quaternion(
        pose.orientation, "sensor orientation"
    )
    x_coord, y_coord, z_coord = centroid
    rotation = (
        (
            1 - 2 * (y_value**2 + z_value**2),
            2 * (x_value * y_value - z_value * w_value),
            2 * (x_value * z_value + y_value * w_value),
        ),
        (
            2 * (x_value * y_value + z_value * w_value),
            1 - 2 * (x_value**2 + z_value**2),
            2 * (y_value * z_value - x_value * w_value),
        ),
        (
            2 * (x_value * z_value - y_value * w_value),
            2 * (y_value * z_value + x_value * w_value),
            1 - 2 * (x_value**2 + y_value**2),
        ),
    )
    position = _extract_vector(pose.position, "sensor position")
    return (
        sum(rotation[0][index] * centroid[index] for index in range(3)) + position[0],
        sum(rotation[1][index] * centroid[index] for index in range(3)) + position[1],
        sum(rotation[2][index] * centroid[index] for index in range(3)) + position[2],
    )


def _raise_if_rpc_timeout(observation: CapabilityObservation) -> None:
    if observation.status is CapabilityStatus.RPC_TIMED_OUT:
        raise CapabilityProbeError(
            f"{observation.client_method} timed out; no later live calls are safe."
        )


def extract_finite_position(state: Any) -> Position3D:
    """Extract and validate the measured NED position from a state object."""
    position = extract_position_from_state(state)
    _validate_vector("measured position", _position_tuple(position))
    return position


def _extract_finite_velocity(state: Any) -> tuple[float, float, float]:
    kinematics = getattr(state, "kinematics_estimated", None)
    velocity = getattr(kinematics, "linear_velocity", None)
    return _extract_vector(velocity, "measured velocity")


def _summarize_landed_state(
    state: Any,
    client_module: ModuleType | None,
) -> dict[str, Any]:
    value = getattr(state, "landed_state", None)
    try:
        numeric_value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CapabilityProbeError("Measured landed state is unavailable.") from exc

    landed_type = getattr(client_module, "LandedState", None)
    landed_value = getattr(landed_type, "Landed", None)
    flying_value = getattr(landed_type, "Flying", None)
    if landed_value is not None and numeric_value == int(landed_value):
        label = "Landed"
        is_landed: bool | None = True
    elif flying_value is not None and numeric_value == int(flying_value):
        label = "Flying"
        is_landed = False
    else:
        label = "Unknown"
        is_landed = None
    return {"value": numeric_value, "label": label, "is_landed": is_landed}


def _extract_pose_position(pose: Any) -> Position3D:
    values = _extract_vector(getattr(pose, "position", None), "pose position")
    return Position3D(*values)


def _optional_pose_position(pose: Any) -> tuple[float, float, float] | None:
    if pose is None:
        return None
    try:
        return _extract_vector(getattr(pose, "position", None), "sensor position")
    except CapabilityProbeError:
        return None


def _extract_vector(vector: Any, label: str) -> tuple[float, float, float]:
    try:
        values = (float(vector.x_val), float(vector.y_val), float(vector.z_val))
    except (AttributeError, TypeError, ValueError) as exc:
        raise CapabilityProbeError(f"{label} is unavailable or malformed.") from exc
    _validate_vector(label, values)
    return values


def _extract_quaternion(
    quaternion: Any, label: str
) -> tuple[float, float, float, float]:
    try:
        values = (
            float(quaternion.w_val),
            float(quaternion.x_val),
            float(quaternion.y_val),
            float(quaternion.z_val),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CapabilityProbeError(f"{label} is unavailable or malformed.") from exc
    if not all(math.isfinite(value) for value in values):
        raise CapabilityProbeError(f"{label} contains non-finite values.")
    return values


def _position_tuple(position: Position3D) -> tuple[float, float, float]:
    return (position.x, position.y, position.z)


def _exact_object_regex(name: str) -> str:
    return f"^{re.escape(name)}$"


def _require_client_type(client_module: ModuleType, name: str) -> Any:
    value = getattr(client_module, name, None)
    if value is None:
        raise CapabilityProbeError(f"Client module does not provide {name}.")
    return value


def _require_method(client: Any, name: str) -> Callable[..., Any]:
    method = getattr(client, name, None)
    if method is None:
        raise CapabilityProbeError(f"Client does not provide {name}.")
    return method


def _require_rpc_success(observation: CapabilityObservation) -> None:
    if observation.status is not CapabilityStatus.RPC_SUCCEEDED:
        raise CapabilityProbeError(observation.detail)


def _join_async(async_result: Any) -> None:
    join = getattr(async_result, "join", None)
    if join is None:
        raise CapabilityProbeError("Asynchronous result does not provide join().")
    try:
        join()
    except Exception as exc:
        raise CapabilityProbeError("Asynchronous simulator command failed.") from exc


def _validate_vector(name: str, values: Sequence[float]) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} values must be finite")


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    return value
