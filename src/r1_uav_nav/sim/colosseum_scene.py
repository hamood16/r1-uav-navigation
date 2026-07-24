"""Safe Colosseum materialization for validated M13.2 scene specifications."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Protocol, Sequence

from r1_uav_nav.sim.colosseum_capabilities import (
    CollisionClassification,
    classify_collision_samples,
    cleanup_named_probe_vehicle,
    sample_collision_information,
    validate_grounded_preflight,
    validate_report_output_path,
)
from r1_uav_nav.sim.colosseum_client import CleanupState
from r1_uav_nav.sim.scene_specification import (
    GEOMETRY_TOLERANCE_M,
    AssetCalibration,
    AssetCatalog,
    Bounds3D,
    CollisionIntent,
    GoalPad,
    SceneObjectSpec,
    SceneValidationError,
    ValidatedScene,
    Vector3,
    asset_catalog_digest,
    build_initial_vehicle_exclusion,
    conservative_bounds,
    materialization_digest,
    runtime_object_name,
    translate_bounds,
    translate_vector,
    validate_world_vehicle_exclusion,
)
from r1_uav_nav.sim.waypoint_navigation import extract_position_from_state

SCENE_BACKEND_VERSION = "1"
OWNERSHIP_MANIFEST_SCHEMA_VERSION = "1.0"
SCENE_REPORT_SCHEMA_VERSION = "1.0"
DEFAULT_SCENE_REPORTS_DIR = Path("results/reports/m13/scenes")
DEFAULT_OWNERSHIP_DIR = DEFAULT_SCENE_REPORTS_DIR / "ownership"
POSITION_TOLERANCE_M = 0.05
SCALE_TOLERANCE = 0.01
YAW_TOLERANCE_DEGREES = 1.0
GROUNDED_SPEED_TOLERANCE_M_S = 0.1


class SceneLifecycleError(RuntimeError):
    """Raised when live scene work cannot continue safely."""


class OwnershipManifestError(SceneLifecycleError):
    """Raised when ownership evidence is missing, malformed, or mismatched."""


class SceneBackend(Protocol):
    """Simulator-facing materialization strategy."""

    name: str
    version: str

    def materialize(
        self,
        scene: ValidatedScene,
        context: MaterializationContext,
        runtime: SceneRuntimeState,
    ) -> tuple[MaterializedObject, ...]:
        """Create or verify the selected scene backend."""


class OwnershipCreationStatus(str, Enum):
    """Atomic ownership transition for one exact simulator object name."""

    CREATING = "creating"
    CREATED = "created"
    CLEANUP_FAILED = "cleanup_failed"
    CLEANED = "cleaned"


@dataclass(frozen=True)
class MaterializationConfig:
    """Explicit authorizations and live placement tolerances."""

    vehicle_name: str = "SimpleFlight"
    allow_scene_mutation: bool = False
    confirm_scene_area_clear: bool = False
    confirm_no_visible_collision: bool = False
    allow_debug_markers: bool = False
    allow_marker_flush: bool = False
    position_tolerance_m: float = POSITION_TOLERANCE_M
    scale_tolerance: float = SCALE_TOLERANCE
    yaw_tolerance_degrees: float = YAW_TOLERANCE_DEGREES

    def __post_init__(self) -> None:
        if not self.vehicle_name.strip():
            raise ValueError("vehicle_name must not be empty")
        for name in (
            "position_tolerance_m",
            "scale_tolerance",
            "yaw_tolerance_degrees",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.allow_debug_markers and not self.allow_marker_flush:
            raise ValueError(
                "debug markers require explicit marker-flush authorization"
            )


@dataclass(frozen=True)
class VehiclePositioningConfig:
    """Separately authorized start-anchor demonstration settings."""

    vehicle_name: str = "SimpleFlight"
    allow_flight: bool = False
    allow_start_positioning: bool = False
    confirm_clear_airspace: bool = False
    confirm_no_visible_collision: bool = False
    velocity_m_s: float = 0.5
    movement_timeout_s: float = 20.0
    waypoint_tolerance_m: float = 0.75
    transit_altitude_m: float = 2.5
    minimum_ground_clearance_m: float = 1.0
    transit_clearance_m: float = 0.5
    grounded_position_tolerance_m: float = 0.25
    landing_confirmation_timeout_s: float = 5.0
    landing_poll_interval_s: float = 0.2
    touchdown_consecutive_samples: int = 3
    final_state_confirmation_timeout_s: float = 5.0
    final_state_poll_interval_s: float = 0.2

    def __post_init__(self) -> None:
        if not self.vehicle_name.strip():
            raise ValueError("vehicle_name must not be empty")
        for name in (
            "velocity_m_s",
            "movement_timeout_s",
            "waypoint_tolerance_m",
            "transit_altitude_m",
            "minimum_ground_clearance_m",
            "transit_clearance_m",
            "grounded_position_tolerance_m",
            "landing_confirmation_timeout_s",
            "landing_poll_interval_s",
            "final_state_confirmation_timeout_s",
            "final_state_poll_interval_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.transit_altitude_m < self.minimum_ground_clearance_m:
            raise ValueError("transit altitude must preserve minimum clearance")
        if (
            isinstance(self.touchdown_consecutive_samples, bool)
            or not isinstance(self.touchdown_consecutive_samples, int)
            or self.touchdown_consecutive_samples <= 0
        ):
            raise ValueError("touchdown_consecutive_samples must be a positive integer")


@dataclass(frozen=True)
class RequestedTransform:
    """Requested world pose, dimensions, and scale for one object."""

    base_center: Vector3
    center_position: Vector3
    yaw_degrees: float
    dimensions_m: tuple[float, float, float]
    scale: tuple[float, float, float]
    conservative_world_bounds: Bounds3D


@dataclass(frozen=True)
class MaterializedObject:
    """Verified runtime result for one exact owned or prebuilt object."""

    specification_name: str
    requested_name: str
    returned_name: str
    asset_name: str
    requested_transform: RequestedTransform
    measured_center_position: Vector3 | None
    measured_scale: tuple[float, float, float] | None
    measured_yaw_degrees: float | None
    material_assignment_succeeded: bool | None
    segmentation_assignment_succeeded: bool | None
    collision_intent: CollisionIntent
    physical_geometry_expected: bool
    physics_enabled: bool
    collision_response_verified: bool = False


@dataclass(frozen=True)
class OwnershipEntry:
    """One exact ownership transition persisted for recovery."""

    specification_name: str
    requested_name: str
    returned_name: str | None
    proven_absent_before_creation: bool
    creation_status: OwnershipCreationStatus
    cleanup_error: str | None = None


@dataclass(frozen=True)
class OwnershipManifest:
    """Atomic ownership evidence; configuration alone never establishes ownership."""

    schema_version: str
    run_id: str
    scene_id: str
    scene_digest: str
    materialization_digest: str
    backend: str
    backend_version: str
    entries: tuple[OwnershipEntry, ...]


@dataclass(frozen=True)
class SceneCleanupResult:
    """Independent cleanup result for one resource domain."""

    domain: str
    attempted: bool
    succeeded: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterializationContext:
    """Measured world reference and accepted runtime evidence."""

    run_id: str
    world_origin: Vector3
    measured_initial_vehicle_position: Vector3
    initial_vehicle_exclusion: Bounds3D
    collision_baseline_timestamp: int | None
    materialization_digest: str
    ownership_manifest_path: Path
    asset_catalog: AssetCatalog
    config: MaterializationConfig


@dataclass(frozen=True)
class MaterializedScene:
    """Structured output from one materialization attempt."""

    run_id: str
    scene_id: str
    scene_digest: str
    materialization_digest: str
    backend: str
    world_origin: Vector3
    workspace_world: Bounds3D
    initial_vehicle_exclusion: Bounds3D
    start_anchor_world: Vector3
    goal_approach_world: Vector3
    objects: tuple[MaterializedObject, ...]
    ownership_manifest_path: str
    collision_geometry_complete: bool
    collision_response_verified: bool
    markers_created: bool


@dataclass(frozen=True)
class SceneLifecycleReport:
    """JSON-safe scene operation and cleanup evidence."""

    schema_version: str
    run_id: str
    mode: str
    success: bool
    interrupted: bool
    scene_id: str | None
    scene_digest: str | None
    materialization_digest: str | None
    selected_vehicle_name: str | None
    materialized_scene: MaterializedScene | None
    data: dict[str, Any]
    cleanup_results: tuple[SceneCleanupResult, ...]
    errors: tuple[str, ...]


@dataclass
class SceneRuntimeState:
    """Mutable ownership and safety state for exactly one live process."""

    manifest: OwnershipManifest
    manifest_path: Path
    owned_names: list[str] = field(default_factory=list)
    markers_created: bool = False
    cleanup_state: CleanupState = field(default_factory=CleanupState)
    vehicle_name: str = ""
    original_ground_position: Vector3 | None = None
    collision_baseline_timestamp: int | None = None
    vehicle_positioning_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetCalibrationProbeResult:
    """Nominal scale-one comparison evidence requiring operator acceptance."""

    run_id: str
    asset_name: str
    requested_name: str
    returned_name: str
    requested_scale: tuple[float, float, float]
    measured_scale: tuple[float, float, float]
    measured_center_position: Vector3
    nominal_reference_dimensions_m: tuple[float, float, float]
    evidence_level: str
    operator_confirmation: str
    uncertainty_note: str
    collision_response_verified: bool


class AssetCalibrationProbe:
    """Supervised scale-one Cube and marker comparison without catalog mutation."""

    def __init__(
        self,
        client: Any,
        client_module: ModuleType,
        *,
        ownership_dir: Path = DEFAULT_OWNERSHIP_DIR,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.client_module = client_module
        self.ownership_dir = ownership_dir
        self.sleep_fn = sleep_fn
        self.last_runtime: SceneRuntimeState | None = None

    def run(
        self,
        *,
        asset_name: str,
        vehicle_name: str,
        allow_scene_mutation: bool,
        confirm_scene_area_clear: bool,
        confirm_no_visible_collision: bool,
        allow_debug_markers: bool,
        allow_marker_flush: bool,
        hold_seconds: float,
        run_id: str | None = None,
    ) -> tuple[AssetCalibrationProbeResult, SceneRuntimeState]:
        if not asset_name.strip():
            raise ValueError("asset_name must not be empty")
        if not vehicle_name.strip():
            raise ValueError("vehicle_name must not be empty")
        if not math.isfinite(hold_seconds) or not 0.0 <= hold_seconds <= 15.0:
            raise ValueError("hold_seconds must be finite and between 0 and 15")
        if not all(
            (
                allow_scene_mutation,
                confirm_scene_area_clear,
                confirm_no_visible_collision,
                allow_debug_markers,
                allow_marker_flush,
            )
        ):
            raise SceneLifecycleError(
                "asset calibration lacks mutation, marker, cleanup, or "
                "operator authorization"
            )

        helper = ColosseumSceneManager(
            self.client,
            self.client_module,
            AssetCatalog(1, 1, ()),
            ownership_dir=self.ownership_dir,
            sleep_fn=self.sleep_fn,
        )
        config = MaterializationConfig(
            vehicle_name=vehicle_name,
            allow_scene_mutation=True,
            confirm_scene_area_clear=True,
            confirm_no_visible_collision=True,
            allow_debug_markers=True,
            allow_marker_flush=True,
        )
        measured, baseline = helper._read_grounded_reference(config)
        selected_run_id = run_id or uuid.uuid4().hex
        requested_name = f"r1_uav_m13s2_calibration__{selected_run_id[:12]}"
        if _exact_object_exists(self.client, requested_name):
            raise SceneLifecycleError("calibration exact name already exists")
        digest = hashlib.sha256(
            f"calibration:{asset_name}:{selected_run_id}".encode("utf-8")
        ).hexdigest()
        manifest = OwnershipManifest(
            OWNERSHIP_MANIFEST_SCHEMA_VERSION,
            selected_run_id,
            "asset-calibration",
            digest,
            digest,
            "asset_calibration",
            SCENE_BACKEND_VERSION,
            (),
        )
        manifest_path = (
            self.ownership_dir / f"{selected_run_id}.calibration.ownership.json"
        )
        validate_report_output_path(manifest_path, Path.cwd())
        save_ownership_manifest_atomic(manifest, manifest_path)
        runtime = SceneRuntimeState(
            manifest,
            manifest_path,
            vehicle_name=vehicle_name,
            original_ground_position=measured,
            collision_baseline_timestamp=baseline,
        )
        self.last_runtime = runtime
        runtime.manifest = _upsert_manifest_entry(
            runtime.manifest,
            OwnershipEntry(
                "calibration-cube",
                requested_name,
                None,
                True,
                OwnershipCreationStatus.CREATING,
            ),
        )
        save_ownership_manifest_atomic(runtime.manifest, runtime.manifest_path)

        center = Vector3(measured.x + 4.0, measured.y, measured.z - 0.5)
        assets = self.client.simListAssets()
        if asset_name not in {str(item) for item in assets}:
            raise SceneLifecycleError(
                f"calibration asset {asset_name!r} is not available"
            )
        returned = self.client.simSpawnObject(
            requested_name,
            asset_name,
            _make_pose(self.client_module, center, 0.0),
            _make_vector(self.client_module, (1.0, 1.0, 1.0)),
            False,
            False,
        )
        if not isinstance(returned, str) or not returned.strip():
            raise SceneLifecycleError("calibration spawn returned an empty name")
        runtime.owned_names.append(returned)
        runtime.manifest = _upsert_manifest_entry(
            runtime.manifest,
            OwnershipEntry(
                "calibration-cube",
                requested_name,
                returned,
                True,
                OwnershipCreationStatus.CREATED,
            ),
        )
        save_ownership_manifest_atomic(runtime.manifest, runtime.manifest_path)
        if returned != requested_name or not _exact_object_exists(
            self.client, returned
        ):
            raise SceneLifecycleError(
                "calibration object exact-name verification failed"
            )

        runtime.markers_created = True
        _plot_nominal_calibration_box(self.client, self.client_module, center)
        print(
            "Nominal one-metre calibration cube and marker frame are visible; "
            "operator confirmation is required."
        )
        self.sleep_fn(hold_seconds)
        measured_pose = self.client.simGetObjectPose(returned)
        measured_scale = _tuple_from_air_sim(
            self.client.simGetObjectScale(returned), "calibration scale"
        )
        measured_center = _vector_from_air_sim(
            measured_pose.position, "calibration pose"
        )
        return (
            AssetCalibrationProbeResult(
                selected_run_id,
                asset_name,
                requested_name,
                returned,
                (1.0, 1.0, 1.0),
                measured_scale,
                measured_center,
                (1.0, 1.0, 1.0),
                "operator_confirmed_nominal",
                "pending",
                "Marker comparison is nominal and does not measure exact mesh bounds.",
                False,
            ),
            runtime,
        )


class RuntimeSpawnSceneBackend:
    """Primary exact-name runtime Cube materialization backend."""

    name = "runtime_spawn"
    version = SCENE_BACKEND_VERSION

    def __init__(
        self,
        client: Any,
        client_module: ModuleType,
        *,
        manifest_writer: Callable[[OwnershipManifest, Path], None] | None = None,
    ) -> None:
        self.client = client
        self.client_module = client_module
        self.manifest_writer = manifest_writer or save_ownership_manifest_atomic

    def materialize(
        self,
        scene: ValidatedScene,
        context: MaterializationContext,
        runtime: SceneRuntimeState,
    ) -> tuple[MaterializedObject, ...]:
        requested_assets = {
            specification.runtime_asset_name
            for specification in _ordered_scene_objects(scene)
        }
        available_assets = {str(asset) for asset in self.client.simListAssets()}
        missing_assets = sorted(requested_assets - available_assets)
        if missing_assets:
            joined = ", ".join(repr(asset) for asset in missing_assets)
            raise SceneLifecycleError(
                f"requested runtime assets are unavailable: {joined}"
            )

        objects: list[MaterializedObject] = []
        for specification in _ordered_scene_objects(scene):
            objects.append(
                self._materialize_object(specification, scene, context, runtime)
            )
        return tuple(objects)

    def _materialize_object(
        self,
        specification: SceneObjectSpec,
        scene: ValidatedScene,
        context: MaterializationContext,
        runtime: SceneRuntimeState,
    ) -> MaterializedObject:
        calibration = _require_accepted_calibration(
            context.asset_catalog, specification.runtime_asset_name
        )
        assert calibration.nominal_dimensions_m is not None
        name = runtime_object_name(
            scene.config.scene_id, specification.name, scene.scene_digest
        )
        if _exact_object_exists(self.client, name):
            raise SceneLifecycleError(
                f"refusing to adopt pre-existing exact object {name!r}"
            )

        runtime.manifest = _upsert_manifest_entry(
            runtime.manifest,
            OwnershipEntry(
                specification.name,
                name,
                None,
                True,
                OwnershipCreationStatus.CREATING,
            ),
        )
        self.manifest_writer(runtime.manifest, runtime.manifest_path)

        requested = _requested_transform(
            specification, context.world_origin, calibration
        )
        pose = _make_pose(
            self.client_module, requested.center_position, requested.yaw_degrees
        )
        scale = _make_vector(self.client_module, requested.scale)
        returned = self.client.simSpawnObject(
            name,
            specification.runtime_asset_name,
            pose,
            scale,
            specification.physics_enabled,
            False,
        )
        if not isinstance(returned, str) or not returned.strip():
            raise SceneLifecycleError("simSpawnObject returned an empty object name")

        runtime.owned_names.append(returned)
        runtime.manifest = _upsert_manifest_entry(
            runtime.manifest,
            OwnershipEntry(
                specification.name,
                name,
                returned,
                True,
                OwnershipCreationStatus.CREATED,
            ),
        )
        self.manifest_writer(runtime.manifest, runtime.manifest_path)
        if returned != name:
            raise SceneLifecycleError(
                "simSpawnObject returned an unexpected exact name; cleanup required"
            )
        if not _exact_object_exists(self.client, returned):
            raise SceneLifecycleError("spawned object was not found by exact query")

        material_succeeded: bool | None = None
        if specification.appearance.material_name:
            material_succeeded = bool(
                self.client.simSetObjectMaterial(
                    returned, specification.appearance.material_name
                )
            )
            if not material_succeeded:
                raise SceneLifecycleError("material assignment reported failure")

        segmentation_succeeded: bool | None = None
        if specification.appearance.segmentation_id is not None:
            segmentation_succeeded = bool(
                self.client.simSetSegmentationObjectID(
                    returned,
                    specification.appearance.segmentation_id,
                    False,
                )
            )
            if not segmentation_succeeded:
                raise SceneLifecycleError("segmentation assignment reported failure")

        measured_pose = self.client.simGetObjectPose(returned)
        measured_scale_raw = self.client.simGetObjectScale(returned)
        measured_position = _vector_from_air_sim(
            getattr(measured_pose, "position", None), "object pose"
        )
        measured_scale = _tuple_from_air_sim(measured_scale_raw, "object scale")
        measured_yaw = _yaw_degrees(getattr(measured_pose, "orientation", None))
        _verify_transform(
            requested,
            measured_position,
            measured_scale,
            measured_yaw,
            context.config,
        )
        return MaterializedObject(
            specification.name,
            name,
            returned,
            specification.runtime_asset_name,
            requested,
            measured_position,
            measured_scale,
            measured_yaw,
            material_succeeded,
            segmentation_succeeded,
            specification.collision_intent,
            specification.physical_geometry_expected,
            specification.physics_enabled,
            False,
        )


class PrebuiltVerifySceneBackend:
    """Read-only fallback that verifies exact configured prebuilt objects."""

    name = "prebuilt_verify"
    version = SCENE_BACKEND_VERSION

    def __init__(self, client: Any) -> None:
        self.client = client

    def materialize(
        self,
        scene: ValidatedScene,
        context: MaterializationContext,
        runtime: SceneRuntimeState,
    ) -> tuple[MaterializedObject, ...]:
        results: list[MaterializedObject] = []
        for specification in _ordered_scene_objects(scene):
            if not specification.prebuilt_name:
                raise SceneLifecycleError(
                    f"{specification.name} has no exact prebuilt_name"
                )
            if not _exact_object_exists(self.client, specification.prebuilt_name):
                raise SceneLifecycleError(
                    f"prebuilt object {specification.prebuilt_name!r} is absent"
                )
            calibration = _require_accepted_calibration(
                context.asset_catalog, specification.runtime_asset_name
            )
            requested = _requested_transform(
                specification, context.world_origin, calibration
            )
            pose = self.client.simGetObjectPose(specification.prebuilt_name)
            scale = self.client.simGetObjectScale(specification.prebuilt_name)
            measured_position = _vector_from_air_sim(pose.position, "prebuilt pose")
            measured_scale = _tuple_from_air_sim(scale, "prebuilt scale")
            measured_yaw = _yaw_degrees(pose.orientation)
            _verify_transform(
                requested,
                measured_position,
                measured_scale,
                measured_yaw,
                context.config,
            )
            results.append(
                MaterializedObject(
                    specification.name,
                    specification.prebuilt_name,
                    specification.prebuilt_name,
                    specification.runtime_asset_name,
                    requested,
                    measured_position,
                    measured_scale,
                    measured_yaw,
                    None,
                    None,
                    specification.collision_intent,
                    specification.physical_geometry_expected,
                    specification.physics_enabled,
                    False,
                )
            )
        return tuple(results)


class MarkerPreviewRenderer:
    """Visual-only green goal overlay using validated persistent markers."""

    def __init__(self, client: Any, client_module: ModuleType) -> None:
        self.client = client
        self.client_module = client_module

    def render_goal(self, goal: GoalPad, world_origin: Vector3) -> None:
        color_rgba = goal.appearance.marker_color_rgba
        if color_rgba is None:
            raise SceneLifecycleError("goal marker overlay has no configured color")
        color = _validated_rgba_list(color_rgba, "goal marker color")
        bounds = translate_bounds(conservative_bounds(goal), world_origin)
        z = bounds.min_z - 0.02
        points = [
            _make_vector(self.client_module, (bounds.min_x, bounds.min_y, z)),
            _make_vector(self.client_module, (bounds.max_x, bounds.min_y, z)),
            _make_vector(self.client_module, (bounds.max_x, bounds.max_y, z)),
            _make_vector(self.client_module, (bounds.min_x, bounds.max_y, z)),
            _make_vector(self.client_module, (bounds.min_x, bounds.min_y, z)),
        ]
        self.client.simPlotLineStrip(points, color, 8.0, -1.0, True)


class ColosseumSceneManager:
    """Coordinate preflight, materialization, ownership, and cleanup."""

    def __init__(
        self,
        client: Any,
        client_module: ModuleType,
        asset_catalog: AssetCatalog,
        *,
        ownership_dir: Path = DEFAULT_OWNERSHIP_DIR,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.client_module = client_module
        self.asset_catalog = asset_catalog
        self.ownership_dir = ownership_dir
        self.sleep_fn = sleep_fn
        self.last_runtime: SceneRuntimeState | None = None
        self.active_runtime: SceneRuntimeState | None = None
        self.last_reset_cleanup_results: tuple[SceneCleanupResult, ...] = ()

    def reset_scene(
        self,
        scene: ValidatedScene,
        config: MaterializationConfig,
        *,
        backend: SceneBackend | None = None,
        run_id: str | None = None,
    ) -> tuple[MaterializedScene, SceneRuntimeState]:
        """Clean only the previous owned scene, then deterministically rebuild."""
        self.last_reset_cleanup_results = ()
        if self.active_runtime is not None:
            cleanup = cleanup_scene_resources(self.client, self.active_runtime)
            self.last_reset_cleanup_results = cleanup
            failures = [
                error
                for result in cleanup
                if not result.succeeded
                for error in result.errors
            ]
            if failures:
                raise SceneLifecycleError(
                    "previous owned scene cleanup failed: " + "; ".join(failures)
                )
            self.active_runtime = None
        materialized, runtime = self.materialize(
            scene, config, backend=backend, run_id=run_id
        )
        self.active_runtime = runtime
        return materialized, runtime

    def materialize(
        self,
        scene: ValidatedScene,
        config: MaterializationConfig,
        *,
        backend: SceneBackend | None = None,
        run_id: str | None = None,
    ) -> tuple[MaterializedScene, SceneRuntimeState]:
        selected_backend = backend or RuntimeSpawnSceneBackend(
            self.client, self.client_module
        )
        if selected_backend.name == "runtime_spawn" and not config.allow_scene_mutation:
            raise SceneLifecycleError(
                "runtime scene materialization requires explicit mutation authorization"
            )
        if (
            selected_backend.name == "runtime_spawn"
            and not config.confirm_scene_area_clear
        ):
            raise SceneLifecycleError(
                "runtime scene materialization requires clear-area confirmation"
            )
        if not config.confirm_no_visible_collision:
            raise SceneLifecycleError(
                "scene materialization requires no-visible-collision confirmation"
            )
        if scene.config.dynamic_obstacles:
            raise SceneLifecycleError(
                "M13.2 records dynamic-obstacle schema but does not materialize motion"
            )

        measured, collision_timestamp = self._read_grounded_reference(config)
        local_initial = scene.config.reference.initial_vehicle_local_position
        world_origin = Vector3(
            measured.x - local_initial.x,
            measured.y - local_initial.y,
            measured.z - local_initial.z,
        )
        exclusion_result = validate_world_vehicle_exclusion(
            scene, world_origin, measured
        )
        if not exclusion_result.valid:
            raise SceneValidationError(exclusion_result)

        _require_scene_calibrations(scene, self.asset_catalog)
        requested_transforms = _requested_transform_mapping(
            scene, world_origin, self.asset_catalog
        )
        digest = materialization_digest(
            local_scene_digest=scene.scene_digest,
            backend=selected_backend.name,
            backend_version=selected_backend.version,
            asset_catalog_digest=asset_catalog_digest(self.asset_catalog),
            calibration_evidence=_calibration_evidence_mapping(
                scene, self.asset_catalog
            ),
            world_origin=world_origin,
            requested_world_transforms=requested_transforms,
        )
        selected_run_id = run_id or uuid.uuid4().hex
        manifest_path = self.ownership_dir / f"{selected_run_id}.ownership.json"
        manifest = OwnershipManifest(
            OWNERSHIP_MANIFEST_SCHEMA_VERSION,
            selected_run_id,
            scene.config.scene_id,
            scene.scene_digest,
            digest,
            selected_backend.name,
            selected_backend.version,
            (),
        )
        validate_report_output_path(manifest_path, Path.cwd())
        save_ownership_manifest_atomic(manifest, manifest_path)
        runtime = SceneRuntimeState(
            manifest,
            manifest_path,
            vehicle_name=config.vehicle_name,
            original_ground_position=measured,
            collision_baseline_timestamp=collision_timestamp,
        )
        self.last_runtime = runtime

        objects = selected_backend.materialize(
            scene,
            _context(
                selected_run_id,
                world_origin,
                measured,
                collision_timestamp,
                digest,
                manifest_path,
                self.asset_catalog,
                config,
                scene,
            ),
            runtime,
        )
        markers_created = False
        if scene.config.goal_pad.appearance.marker_color_rgba is not None:
            if not config.allow_debug_markers or not config.allow_marker_flush:
                raise SceneLifecycleError(
                    "goal marker overlay requires marker and flush authorization"
                )
            runtime.markers_created = True
            MarkerPreviewRenderer(self.client, self.client_module).render_goal(
                scene.config.goal_pad, world_origin
            )
            markers_created = True
        complete = all(
            item.physical_geometry_expected
            and item.collision_intent is CollisionIntent.SOLID_EXPECTED
            for item in objects
        )
        materialized = MaterializedScene(
            selected_run_id,
            scene.config.scene_id,
            scene.scene_digest,
            digest,
            selected_backend.name,
            world_origin,
            _workspace_world(scene, world_origin),
            build_initial_vehicle_exclusion(
                measured, scene.config.reference.initial_vehicle_exclusion
            ),
            translate_vector(scene.start_anchor, world_origin),
            translate_vector(scene.goal_approach, world_origin),
            objects,
            str(manifest_path),
            complete,
            False,
            markers_created,
        )
        return materialized, runtime

    def _read_grounded_reference(
        self, config: MaterializationConfig
    ) -> tuple[Vector3, int | None]:
        state = self.client.getMultirotorState(vehicle_name=config.vehicle_name)
        speed = _require_stationary_state(
            state, context="scene materialization preflight"
        )
        api_enabled = self.client.isApiControlEnabled(vehicle_name=config.vehicle_name)
        _, samples = sample_collision_information(
            self.client,
            vehicle_name=config.vehicle_name,
            sleep_fn=self.sleep_fn,
        )
        landed = _is_landed(self.client_module, state)
        assessment = classify_collision_samples(
            samples,
            is_landed=landed,
            measured_speed=speed,
            api_control_enabled=api_enabled,
            operator_confirmed_stable=config.confirm_no_visible_collision,
        )
        position = validate_grounded_preflight(
            self.client,
            self.client_module,
            state,
            collision_assessment=assessment,
            operator_confirmed_stable=config.confirm_no_visible_collision,
            vehicle_name=config.vehicle_name,
        )
        return (
            Vector3(position.x, position.y, position.z),
            assessment.baseline_timestamp,
        )


def cleanup_scene_resources(
    client: Any,
    runtime: SceneRuntimeState,
) -> tuple[SceneCleanupResult, ...]:
    """Attempt UAV, exact-object, and marker cleanup without cross-domain skipping."""
    results: list[SceneCleanupResult] = []
    uav_errors: tuple[str, ...] = ()
    uav_attempted = runtime.cleanup_state != CleanupState()
    if uav_attempted:
        try:
            uav_errors = cleanup_named_probe_vehicle(
                client, runtime.cleanup_state, runtime.vehicle_name
            )
        except BaseException as exc:
            uav_errors = (f"UAV cleanup raised {type(exc).__name__}",)
    results.append(SceneCleanupResult("uav", uav_attempted, not uav_errors, uav_errors))

    object_errors: list[str] = []
    object_attempted = bool(runtime.owned_names)
    for name in reversed(runtime.owned_names):
        try:
            if _exact_object_exists(client, name):
                if client.simDestroyObject(name) is False:
                    raise SceneLifecycleError("simDestroyObject reported failure")
            if _exact_object_exists(client, name):
                raise SceneLifecycleError("exact owned object remains")
            runtime.manifest = _set_manifest_cleanup_status(
                runtime.manifest, name, OwnershipCreationStatus.CLEANED, None
            )
            save_ownership_manifest_atomic(runtime.manifest, runtime.manifest_path)
        except BaseException as exc:
            message = f"{name!r}: {type(exc).__name__}"
            object_errors.append(message)
            runtime.manifest = _set_manifest_cleanup_status(
                runtime.manifest,
                name,
                OwnershipCreationStatus.CLEANUP_FAILED,
                message,
            )
            try:
                save_ownership_manifest_atomic(runtime.manifest, runtime.manifest_path)
            except BaseException as manifest_exc:
                object_errors.append(f"manifest update: {type(manifest_exc).__name__}")
    results.append(
        SceneCleanupResult(
            "objects", object_attempted, not object_errors, tuple(object_errors)
        )
    )

    marker_errors: list[str] = []
    marker_attempted = runtime.markers_created
    if marker_attempted:
        try:
            client.simFlushPersistentMarkers()
        except BaseException as exc:
            marker_errors.append(f"marker cleanup raised {type(exc).__name__}")
    results.append(
        SceneCleanupResult(
            "markers", marker_attempted, not marker_errors, tuple(marker_errors)
        )
    )
    return tuple(results)


def recover_owned_scene(
    client: Any,
    ownership_source: str | Path,
    *,
    allow_scene_mutation: bool,
    allow_recovery: bool,
    expected_backend: str = "runtime_spawn",
) -> tuple[OwnershipManifest, SceneCleanupResult]:
    """Recover exact created names from accepted ownership evidence only."""
    if not allow_scene_mutation or not allow_recovery:
        raise OwnershipManifestError(
            "recovery requires explicit scene-mutation and recovery authorization"
        )
    manifest, update_path = _load_ownership_source(ownership_source)
    if manifest.schema_version != OWNERSHIP_MANIFEST_SCHEMA_VERSION:
        raise OwnershipManifestError("ownership manifest schema does not match")
    if manifest.backend != expected_backend:
        raise OwnershipManifestError("ownership manifest backend does not match")
    exact_names = [
        entry.returned_name
        for entry in manifest.entries
        if entry.returned_name
        and entry.proven_absent_before_creation
        and entry.creation_status
        in {
            OwnershipCreationStatus.CREATED,
            OwnershipCreationStatus.CLEANUP_FAILED,
        }
    ]
    ambiguous = [
        entry
        for entry in manifest.entries
        if entry.creation_status is OwnershipCreationStatus.CREATING
    ]
    if ambiguous:
        raise OwnershipManifestError(
            "manifest contains ambiguous creating entries; supervised review required"
        )
    errors: list[str] = []
    updated = manifest
    for name in reversed(exact_names):
        try:
            if _exact_object_exists(client, name):
                if client.simDestroyObject(name) is False:
                    raise SceneLifecycleError("simDestroyObject reported failure")
            if _exact_object_exists(client, name):
                raise SceneLifecycleError("exact object remains after recovery")
            updated = _set_manifest_cleanup_status(
                updated, name, OwnershipCreationStatus.CLEANED, None
            )
            save_ownership_manifest_atomic(updated, update_path)
        except BaseException as exc:
            errors.append(f"{name!r}: {type(exc).__name__}")
    return updated, SceneCleanupResult(
        "objects", bool(exact_names), not errors, tuple(errors)
    )


def position_vehicle_at_start_and_return(
    client: Any,
    client_module: ModuleType,
    materialized: MaterializedScene,
    runtime: SceneRuntimeState,
    config: VehiclePositioningConfig,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Demonstrate the start anchor, then return and land at original safe ground."""
    if not all(
        (
            config.allow_flight,
            config.allow_start_positioning,
            config.confirm_clear_airspace,
            config.confirm_no_visible_collision,
        )
    ):
        raise SceneLifecycleError("start positioning lacks required authorization")
    if config.vehicle_name != runtime.vehicle_name:
        raise SceneLifecycleError(
            "positioning vehicle does not match runtime ownership"
        )
    if runtime.original_ground_position is None:
        raise SceneLifecycleError("original grounded position was not recorded")

    original = runtime.original_ground_position
    transit = Vector3(
        original.x,
        original.y,
        original.z - config.transit_altitude_m,
    )
    return_airborne = transit
    anchor = materialized.start_anchor_world
    evidence: dict[str, Any] = {
        "original_ground_position": asdict(original),
        "transit_point": asdict(transit),
        "start_anchor": asdict(anchor),
        "touchdown_confirmation_attempts": 0,
        "touchdown_consecutive_samples": 0,
        "touchdown_position": None,
        "touchdown_speed_m_s": None,
        "touchdown_rejection_reason": None,
        "landed_state_before_disarm": None,
        "final_confirmation_attempts": 0,
        "final_landed_state": None,
        "final_position": None,
        "final_speed_m_s": None,
        "final_api_control_enabled": None,
        "final_rejection_reason": None,
        "api_control_released": False,
        "returned_to_original_ground": False,
        "landing_confirmed": False,
    }
    runtime.vehicle_positioning_evidence = evidence
    for point in (transit, anchor, return_airborne):
        _require_point_outside_scene_geometry(point, materialized)
    validate_transit_corridor(
        materialized,
        (transit, anchor, return_airborne),
        clearance_m=config.transit_clearance_m,
    )

    fresh_state = client.getMultirotorState(vehicle_name=config.vehicle_name)
    fresh_position = _state_position(fresh_state)
    fresh_speed = _require_stationary_state(
        fresh_state, context="start-positioning preflight"
    )
    if not _is_landed(client_module, fresh_state):
        raise SceneLifecycleError(
            "vehicle must remain landed immediately before control acquisition"
        )
    api_enabled = client.isApiControlEnabled(vehicle_name=config.vehicle_name)
    if not isinstance(api_enabled, bool) or api_enabled:
        raise SceneLifecycleError(
            "API control must remain disabled immediately before positioning"
        )
    _, collision_samples = sample_collision_information(
        client,
        vehicle_name=config.vehicle_name,
        sleep_fn=sleep_fn,
    )
    collision_assessment = classify_collision_samples(
        collision_samples,
        is_landed=True,
        measured_speed=fresh_speed,
        api_control_enabled=api_enabled,
        operator_confirmed_stable=config.confirm_no_visible_collision,
    )
    if collision_assessment.classification not in {
        CollisionClassification.NO_COLLISION,
        CollisionClassification.EXPECTED_GROUND_CONTACT,
    }:
        raise SceneLifecycleError(
            "fresh collision evidence is unsafe or inconclusive before positioning"
        )
    if _distance(fresh_position, original) > config.grounded_position_tolerance_m:
        raise SceneLifecycleError(
            "vehicle moved away from the recorded original ground position"
        )
    runtime.collision_baseline_timestamp = collision_assessment.baseline_timestamp

    runtime.cleanup_state = replace(runtime.cleanup_state, api_control_enabled=True)
    client.enableApiControl(True, vehicle_name=config.vehicle_name)
    runtime.cleanup_state = replace(runtime.cleanup_state, armed=True)
    client.armDisarm(True, vehicle_name=config.vehicle_name)
    runtime.cleanup_state = replace(runtime.cleanup_state, takeoff_attempted=True)
    client.takeoffAsync(vehicle_name=config.vehicle_name).join()
    runtime.cleanup_state = replace(runtime.cleanup_state, airborne=True)
    takeoff_state = client.getMultirotorState(vehicle_name=config.vehicle_name)
    _verify_airborne_state(
        client,
        takeoff_state,
        original.z,
        materialized,
        runtime,
        config,
    )
    _move_and_verify(
        client, client_module, transit, original.z, materialized, runtime, config
    )
    _move_and_verify(
        client, client_module, anchor, original.z, materialized, runtime, config
    )
    _move_and_verify(
        client,
        client_module,
        return_airborne,
        original.z,
        materialized,
        runtime,
        config,
    )
    client.hoverAsync(vehicle_name=config.vehicle_name).join()
    client.landAsync(vehicle_name=config.vehicle_name).join()
    (
        touchdown_state,
        _touchdown_position,
        _touchdown_speed,
    ) = _confirm_physical_touchdown(
        client,
        config,
        original,
        runtime,
        evidence,
        sleep_fn=sleep_fn,
    )
    evidence["landed_state_before_disarm"] = getattr(
        touchdown_state, "landed_state", None
    )
    runtime.cleanup_state = replace(runtime.cleanup_state, airborne=False)
    client.armDisarm(False, vehicle_name=config.vehicle_name)
    runtime.cleanup_state = replace(runtime.cleanup_state, armed=False)
    client.enableApiControl(False, vehicle_name=config.vehicle_name)
    _final_state, final_position, final_speed = _confirm_final_landed_state(
        client,
        client_module,
        config,
        original,
        evidence,
        sleep_fn=sleep_fn,
    )
    runtime.cleanup_state = replace(runtime.cleanup_state, api_control_enabled=False)
    evidence["api_control_released"] = True
    collision = client.simGetCollisionInfo(vehicle_name=config.vehicle_name)
    if bool(getattr(collision, "has_collided", False)):
        timestamp = getattr(collision, "time_stamp", None)
        if timestamp != runtime.collision_baseline_timestamp:
            raise SceneLifecycleError("new collision detected after landing")
    evidence["returned_ground_position"] = asdict(final_position)
    evidence["returned_to_original_ground"] = True
    evidence["landing_confirmed"] = True
    evidence["landed_confirmation"] = True
    evidence["landing_confirmation_attempts"] = evidence[
        "touchdown_confirmation_attempts"
    ]
    runtime.cleanup_state = CleanupState()
    return evidence


def save_ownership_manifest_atomic(
    manifest: OwnershipManifest, output_path: Path
) -> None:
    """Atomically preserve the last complete ownership transition."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_ownership_manifest(path: str | Path) -> OwnershipManifest:
    """Load strict ownership evidence without deriving names from configuration."""
    manifest, _ = _load_ownership_source(path)
    return manifest


def _load_ownership_source(
    path: str | Path,
) -> tuple[OwnershipManifest, Path]:
    source_path = Path(path)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        if (
            isinstance(raw, dict)
            and isinstance(raw.get("data"), dict)
            and raw["data"].get("ownership_evidence_complete") is True
        ):
            raw = raw["data"]["ownership_manifest"]
            update_path = (
                source_path.parent
                / "ownership"
                / f"{raw['run_id']}.recovery.ownership.json"
            )
        else:
            update_path = source_path
        entries = tuple(
            OwnershipEntry(
                specification_name=str(entry["specification_name"]),
                requested_name=str(entry["requested_name"]),
                returned_name=(
                    str(entry["returned_name"])
                    if entry.get("returned_name") is not None
                    else None
                ),
                proven_absent_before_creation=(
                    entry["proven_absent_before_creation"] is True
                ),
                creation_status=OwnershipCreationStatus(entry["creation_status"]),
                cleanup_error=entry.get("cleanup_error"),
            )
            for entry in raw["entries"]
        )
        manifest = OwnershipManifest(
            schema_version=str(raw["schema_version"]),
            run_id=str(raw["run_id"]),
            scene_id=str(raw["scene_id"]),
            scene_digest=str(raw["scene_digest"]),
            materialization_digest=str(raw["materialization_digest"]),
            backend=str(raw["backend"]),
            backend_version=str(raw["backend_version"]),
            entries=entries,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        raise OwnershipManifestError("ownership manifest is malformed") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.scene_digest):
        raise OwnershipManifestError("manifest scene digest is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.materialization_digest):
        raise OwnershipManifestError("manifest materialization digest is invalid")
    if not manifest.run_id.strip() or not manifest.backend.strip():
        raise OwnershipManifestError("manifest identity is incomplete")
    return manifest, update_path


def save_scene_report(report: SceneLifecycleReport, path: Path) -> None:
    """Write an ignored, human-reviewable scene report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _context(
    run_id: str,
    world_origin: Vector3,
    measured: Vector3,
    collision_timestamp: int | None,
    digest: str,
    manifest_path: Path,
    catalog: AssetCatalog,
    config: MaterializationConfig,
    scene: ValidatedScene,
) -> MaterializationContext:
    return MaterializationContext(
        run_id,
        world_origin,
        measured,
        build_initial_vehicle_exclusion(
            measured, scene.config.reference.initial_vehicle_exclusion
        ),
        collision_timestamp,
        digest,
        manifest_path,
        catalog,
        config,
    )


def _ordered_scene_objects(scene: ValidatedScene) -> tuple[SceneObjectSpec, ...]:
    return (
        scene.config.start_pad,
        scene.config.goal_pad,
        *tuple(sorted(scene.config.static_obstacles, key=lambda item: item.name)),
    )


def _require_accepted_calibration(
    catalog: AssetCatalog, asset_name: str
) -> AssetCalibration:
    calibration = catalog.calibration_for(asset_name)
    if calibration is None:
        raise SceneLifecycleError(f"asset {asset_name!r} is absent from catalog")
    if not calibration.accepted_for_materialization:
        raise SceneLifecycleError(
            f"asset {asset_name!r} lacks accepted dimensional calibration"
        )
    return calibration


def _require_scene_calibrations(scene: ValidatedScene, catalog: AssetCatalog) -> None:
    for item in _ordered_scene_objects(scene):
        if item.physical_geometry_expected:
            _require_accepted_calibration(catalog, item.runtime_asset_name)


def _requested_transform(
    specification: SceneObjectSpec,
    world_origin: Vector3,
    calibration: AssetCalibration,
) -> RequestedTransform:
    native = calibration.nominal_dimensions_m
    if native is None:
        raise SceneLifecycleError("accepted calibration omitted dimensions")
    base = translate_vector(specification.base_center, world_origin)
    center = Vector3(
        base.x,
        base.y,
        base.z - specification.dimensions.height / 2.0,
    )
    scale = (
        specification.dimensions.width / native.width,
        specification.dimensions.depth / native.depth,
        specification.dimensions.height / native.height,
    )
    return RequestedTransform(
        base,
        center,
        specification.yaw_degrees,
        (
            specification.dimensions.width,
            specification.dimensions.depth,
            specification.dimensions.height,
        ),
        scale,
        translate_bounds(conservative_bounds(specification), world_origin),
    )


def _requested_transform_mapping(
    scene: ValidatedScene, origin: Vector3, catalog: AssetCatalog
) -> dict[str, Any]:
    return {
        item.name: _jsonable(
            _requested_transform(
                item,
                origin,
                _require_accepted_calibration(catalog, item.runtime_asset_name),
            )
        )
        for item in _ordered_scene_objects(scene)
    }


def _calibration_evidence_mapping(
    scene: ValidatedScene, catalog: AssetCatalog
) -> dict[str, Any]:
    assets = sorted({item.runtime_asset_name for item in _ordered_scene_objects(scene)})
    return {
        asset: _jsonable(_require_accepted_calibration(catalog, asset))
        for asset in assets
    }


def _make_vector(client_module: ModuleType, values: Sequence[float]) -> Any:
    vector_type = getattr(client_module, "Vector3r", None)
    if vector_type is None:
        raise SceneLifecycleError("client module has no Vector3r")
    return vector_type(*(float(value) for value in values))


def _make_pose(client_module: ModuleType, center: Vector3, yaw_degrees: float) -> Any:
    pose_type = getattr(client_module, "Pose", None)
    quaternion_fn = getattr(client_module, "to_quaternion", None)
    if pose_type is None or quaternion_fn is None:
        raise SceneLifecycleError("client module lacks Pose or to_quaternion")
    orientation = quaternion_fn(0.0, 0.0, math.radians(yaw_degrees))
    return pose_type(
        position_val=_make_vector(client_module, center.values()),
        orientation_val=orientation,
    )


def _plot_nominal_calibration_box(
    client: Any, client_module: ModuleType, center: Vector3
) -> None:
    corners = [
        Vector3(center.x + dx, center.y + dy, center.z + dz)
        for dz in (-0.5, 0.5)
        for dy in (-0.5, 0.5)
        for dx in (-0.5, 0.5)
    ]
    edge_indices = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    line_points = [
        _make_vector(client_module, corners[index].values())
        for edge in edge_indices
        for index in edge
    ]
    client.simPlotLineList(
        line_points,
        _validated_rgba_list((1.0, 1.0, 0.0, 1.0), "calibration marker color"),
        5.0,
        -1.0,
        True,
    )


def _validated_rgba_list(values: Sequence[float], description: str) -> list[float]:
    try:
        color = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneLifecycleError(f"{description} is malformed") from exc
    if len(color) != 4:
        raise SceneLifecycleError(f"{description} must contain four values")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in color):
        raise SceneLifecycleError(
            f"{description} values must be finite and between 0.0 and 1.0"
        )
    return color


def _vector_from_air_sim(value: Any, description: str) -> Vector3:
    try:
        result = Vector3(float(value.x_val), float(value.y_val), float(value.z_val))
    except (AttributeError, TypeError, ValueError) as exc:
        raise SceneLifecycleError(f"{description} is malformed") from exc
    if not all(math.isfinite(item) for item in result.values()):
        raise SceneLifecycleError(f"{description} is non-finite")
    return result


def _tuple_from_air_sim(value: Any, description: str) -> tuple[float, float, float]:
    vector = _vector_from_air_sim(value, description)
    return vector.values()


def _yaw_degrees(orientation: Any) -> float:
    try:
        x = float(orientation.x_val)
        y = float(orientation.y_val)
        z = float(orientation.z_val)
        w = float(orientation.w_val)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SceneLifecycleError("object orientation is malformed") from exc
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise SceneLifecycleError("object orientation is non-finite")
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw)


def _verify_transform(
    requested: RequestedTransform,
    measured_position: Vector3,
    measured_scale: tuple[float, float, float],
    measured_yaw: float,
    config: MaterializationConfig,
) -> None:
    if (
        _distance(requested.center_position, measured_position)
        > config.position_tolerance_m
    ):
        raise SceneLifecycleError("object pose read-back exceeded position tolerance")
    if any(
        abs(expected - actual) > config.scale_tolerance
        for expected, actual in zip(requested.scale, measured_scale, strict=True)
    ):
        raise SceneLifecycleError("object scale read-back exceeded tolerance")
    yaw_error = abs((measured_yaw - requested.yaw_degrees + 180.0) % 360.0 - 180.0)
    if yaw_error > config.yaw_tolerance_degrees:
        raise SceneLifecycleError("object yaw read-back exceeded tolerance")


def _exact_object_exists(client: Any, name: str) -> bool:
    values = client.simListSceneObjects(f"^{re.escape(name)}$")
    return name in {str(value) for value in values}


def _upsert_manifest_entry(
    manifest: OwnershipManifest, entry: OwnershipEntry
) -> OwnershipManifest:
    entries = [
        existing
        for existing in manifest.entries
        if existing.requested_name != entry.requested_name
    ]
    entries.append(entry)
    return replace(manifest, entries=tuple(entries))


def _set_manifest_cleanup_status(
    manifest: OwnershipManifest,
    returned_name: str,
    status: OwnershipCreationStatus,
    cleanup_error: str | None,
) -> OwnershipManifest:
    found = False
    entries: list[OwnershipEntry] = []
    for entry in manifest.entries:
        if entry.returned_name == returned_name:
            found = True
            entries.append(
                replace(
                    entry,
                    creation_status=status,
                    cleanup_error=cleanup_error,
                )
            )
        else:
            entries.append(entry)
    if not found:
        raise OwnershipManifestError(
            f"exact returned name {returned_name!r} is absent from manifest"
        )
    return replace(manifest, entries=tuple(entries))


def _state_position(state: Any) -> Vector3:
    position = extract_position_from_state(state)
    result = Vector3(position.x, position.y, position.z)
    if not all(math.isfinite(value) for value in result.values()):
        raise SceneLifecycleError("vehicle position is non-finite")
    return result


def _require_stationary_state(state: Any, *, context: str) -> float:
    speed = _extract_state_speed(state)
    if speed > GROUNDED_SPEED_TOLERANCE_M_S:
        raise SceneLifecycleError(
            f"{context} requires a stationary vehicle; "
            f"measured_speed_m_s={speed:.6f}"
        )
    return speed


def _extract_state_speed(state: Any) -> float:
    velocity = getattr(
        getattr(state, "kinematics_estimated", None), "linear_velocity", None
    )
    vector = _vector_from_air_sim(velocity, "vehicle velocity")
    speed = math.sqrt(sum(value * value for value in vector.values()))
    return speed


def _confirmation_attempt_budget(timeout_s: float, poll_interval_s: float) -> int:
    maximum_waits = int(math.floor((timeout_s + 1e-12) / poll_interval_s))
    return maximum_waits + 1


def _touchdown_collision_rejection_reason(
    client: Any,
    config: VehiclePositioningConfig,
    runtime: SceneRuntimeState,
) -> str | None:
    collision = client.simGetCollisionInfo(vehicle_name=config.vehicle_name)
    has_collided = getattr(collision, "has_collided", None)
    if not isinstance(has_collided, bool):
        return "collision evidence is unavailable or malformed"
    if has_collided:
        timestamp = getattr(collision, "time_stamp", None)
        if (
            runtime.collision_baseline_timestamp is None
            or timestamp is None
            or timestamp != runtime.collision_baseline_timestamp
        ):
            return "new or ambiguous collision evidence"
    return None


def _confirm_physical_touchdown(
    client: Any,
    config: VehiclePositioningConfig,
    original_ground_position: Vector3,
    runtime: SceneRuntimeState,
    evidence: dict[str, Any],
    *,
    sleep_fn: Callable[[float], None],
) -> tuple[Any, Vector3, float]:
    maximum_attempts = _confirmation_attempt_budget(
        config.landing_confirmation_timeout_s,
        config.landing_poll_interval_s,
    )
    last_state: Any | None = None
    last_position: Vector3 | None = None
    last_speed: float | None = None
    consecutive_samples = 0

    for attempt in range(1, maximum_attempts + 1):
        evidence["touchdown_confirmation_attempts"] = attempt
        evidence["touchdown_rejection_reason"] = "sample was not evaluated"
        last_state = client.getMultirotorState(vehicle_name=config.vehicle_name)
        evidence["landed_state_before_disarm"] = getattr(
            last_state, "landed_state", None
        )
        try:
            last_position = _state_position(last_state)
            last_speed = _extract_state_speed(last_state)
        except SceneLifecycleError as exc:
            consecutive_samples = 0
            evidence["touchdown_consecutive_samples"] = consecutive_samples
            evidence["touchdown_rejection_reason"] = str(exc)
            if attempt < maximum_attempts:
                sleep_fn(config.landing_poll_interval_s)
                continue
            break

        rejection_reasons: list[str] = []
        if last_speed > GROUNDED_SPEED_TOLERANCE_M_S:
            rejection_reasons.append("speed above grounded tolerance")
        if (
            _distance(last_position, original_ground_position)
            > config.waypoint_tolerance_m
        ):
            rejection_reasons.append("position outside landing tolerance")
        if (
            abs(last_position.z - original_ground_position.z)
            > config.waypoint_tolerance_m
        ):
            rejection_reasons.append("vertical position outside landing tolerance")
        collision_rejection = _touchdown_collision_rejection_reason(
            client, config, runtime
        )
        if collision_rejection:
            rejection_reasons.append(collision_rejection)

        if rejection_reasons:
            consecutive_samples = 0
        else:
            consecutive_samples += 1
        evidence.update(
            {
                "touchdown_consecutive_samples": consecutive_samples,
                "touchdown_position": asdict(last_position),
                "touchdown_speed_m_s": last_speed,
                "touchdown_rejection_reason": (
                    "; ".join(rejection_reasons) if rejection_reasons else None
                ),
            }
        )
        if consecutive_samples >= config.touchdown_consecutive_samples:
            return last_state, last_position, last_speed
        if attempt < maximum_attempts:
            sleep_fn(config.landing_poll_interval_s)

    assert last_state is not None
    landed_value = getattr(last_state, "landed_state", None)
    raise SceneLifecycleError(
        "physical touchdown confirmation timed out after "
        f"{maximum_attempts} attempts; "
        f"consecutive_stable_samples={consecutive_samples}; "
        f"last_rejection={evidence['touchdown_rejection_reason']!r}; "
        f"last_landed_state={landed_value!r}; "
        f"last_position={(last_position.values() if last_position else None)!r}; "
        f"last_speed_m_s={last_speed!r}"
    )


def _confirm_final_landed_state(
    client: Any,
    client_module: ModuleType,
    config: VehiclePositioningConfig,
    original_ground_position: Vector3,
    evidence: dict[str, Any],
    *,
    sleep_fn: Callable[[float], None],
) -> tuple[Any, Vector3, float]:
    maximum_attempts = _confirmation_attempt_budget(
        config.final_state_confirmation_timeout_s,
        config.final_state_poll_interval_s,
    )
    last_state: Any | None = None
    last_position: Vector3 | None = None
    last_speed: float | None = None

    for attempt in range(1, maximum_attempts + 1):
        evidence["final_confirmation_attempts"] = attempt
        evidence["final_rejection_reason"] = "sample was not evaluated"
        last_state = client.getMultirotorState(vehicle_name=config.vehicle_name)
        evidence["final_landed_state"] = getattr(last_state, "landed_state", None)
        try:
            last_position = _state_position(last_state)
            last_speed = _extract_state_speed(last_state)
        except SceneLifecycleError as exc:
            evidence["final_rejection_reason"] = str(exc)
            if attempt < maximum_attempts:
                sleep_fn(config.final_state_poll_interval_s)
                continue
            break

        api_enabled = client.isApiControlEnabled(vehicle_name=config.vehicle_name)
        rejection_reasons: list[str] = []
        if not _is_landed(client_module, last_state):
            rejection_reasons.append("landed state has not converged")
        if last_speed > GROUNDED_SPEED_TOLERANCE_M_S:
            rejection_reasons.append("speed above grounded tolerance")
        if (
            _distance(last_position, original_ground_position)
            > config.waypoint_tolerance_m
        ):
            rejection_reasons.append("position outside landing tolerance")
        if not isinstance(api_enabled, bool):
            rejection_reasons.append("API-control state is unavailable or malformed")
        elif api_enabled:
            rejection_reasons.append("API control remains enabled")
        evidence.update(
            {
                "final_position": asdict(last_position),
                "final_speed_m_s": last_speed,
                "final_api_control_enabled": api_enabled,
                "api_control_released": (
                    isinstance(api_enabled, bool) and not api_enabled
                ),
                "final_rejection_reason": (
                    "; ".join(rejection_reasons) if rejection_reasons else None
                ),
            }
        )
        if not rejection_reasons:
            return last_state, last_position, last_speed
        if attempt < maximum_attempts:
            sleep_fn(config.final_state_poll_interval_s)

    assert last_state is not None
    landed_value = getattr(last_state, "landed_state", None)
    raise SceneLifecycleError(
        "final landed-state confirmation timed out after "
        f"{maximum_attempts} attempts; "
        f"last_rejection={evidence['final_rejection_reason']!r}; "
        f"last_landed_state={landed_value!r}; "
        f"last_position={(last_position.values() if last_position else None)!r}; "
        f"last_speed_m_s={last_speed!r}"
    )


def _is_landed(client_module: ModuleType, state: Any) -> bool:
    landed_type = getattr(client_module, "LandedState", None)
    return getattr(state, "landed_state", None) == getattr(landed_type, "Landed", 0)


def _move_and_verify(
    client: Any,
    client_module: ModuleType,
    target: Vector3,
    ground_z: float,
    materialized: MaterializedScene,
    runtime: SceneRuntimeState,
    config: VehiclePositioningConfig,
) -> None:
    client.moveToPositionAsync(
        target.x,
        target.y,
        target.z,
        config.velocity_m_s,
        timeout_sec=config.movement_timeout_s,
        vehicle_name=config.vehicle_name,
    ).join()
    client.hoverAsync(vehicle_name=config.vehicle_name).join()
    state = client.getMultirotorState(vehicle_name=config.vehicle_name)
    position = _state_position(state)
    _extract_state_speed(state)
    if _distance(position, target) > config.waypoint_tolerance_m:
        raise SceneLifecycleError("vehicle exceeded waypoint tolerance")
    if ground_z - position.z < config.minimum_ground_clearance_m:
        raise SceneLifecycleError("vehicle violated minimum ground clearance")
    collision = client.simGetCollisionInfo(vehicle_name=config.vehicle_name)
    if bool(getattr(collision, "has_collided", False)):
        timestamp = getattr(collision, "time_stamp", None)
        if timestamp != runtime.collision_baseline_timestamp:
            raise SceneLifecycleError("new or ambiguous collision during positioning")
    if not all(math.isfinite(value) for value in position.values()):
        raise SceneLifecycleError("vehicle state became non-finite")
    workspace = materialized.workspace_world
    if not (
        workspace.min_x - GEOMETRY_TOLERANCE_M
        <= position.x
        <= workspace.max_x + GEOMETRY_TOLERANCE_M
        and workspace.min_y - GEOMETRY_TOLERANCE_M
        <= position.y
        <= workspace.max_y + GEOMETRY_TOLERANCE_M
        and workspace.min_z - GEOMETRY_TOLERANCE_M
        <= position.z
        <= workspace.max_z + GEOMETRY_TOLERANCE_M
    ):
        raise SceneLifecycleError("vehicle left the translated workspace")


def _verify_airborne_state(
    client: Any,
    state: Any,
    ground_z: float,
    materialized: MaterializedScene,
    runtime: SceneRuntimeState,
    config: VehiclePositioningConfig,
) -> Vector3:
    position = _state_position(state)
    if ground_z - position.z < config.minimum_ground_clearance_m:
        raise SceneLifecycleError("takeoff did not establish minimum ground clearance")
    collision = client.simGetCollisionInfo(vehicle_name=config.vehicle_name)
    if bool(getattr(collision, "has_collided", False)):
        timestamp = getattr(collision, "time_stamp", None)
        if timestamp != runtime.collision_baseline_timestamp:
            raise SceneLifecycleError("new or ambiguous collision after takeoff")
    workspace = materialized.workspace_world
    if not (
        workspace.min_x - GEOMETRY_TOLERANCE_M
        <= position.x
        <= workspace.max_x + GEOMETRY_TOLERANCE_M
        and workspace.min_y - GEOMETRY_TOLERANCE_M
        <= position.y
        <= workspace.max_y + GEOMETRY_TOLERANCE_M
        and workspace.min_z - GEOMETRY_TOLERANCE_M
        <= position.z
        <= workspace.max_z + GEOMETRY_TOLERANCE_M
    ):
        raise SceneLifecycleError("takeoff state left the translated workspace")
    return position


def validate_transit_corridor(
    materialized: MaterializedScene,
    waypoints: Sequence[Vector3],
    *,
    clearance_m: float,
) -> None:
    """Reject a Stage 6 corridor that crosses expected physical geometry."""
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("clearance_m must be finite and non-negative")
    if len(waypoints) < 2:
        raise ValueError("transit corridor requires at least two waypoints")
    for first, second in zip(waypoints, waypoints[1:], strict=False):
        for item in materialized.objects:
            if not item.physical_geometry_expected:
                continue
            bounds = item.requested_transform.conservative_world_bounds
            expanded = Bounds3D(
                bounds.min_x - clearance_m,
                bounds.max_x + clearance_m,
                bounds.min_y - clearance_m,
                bounds.max_y + clearance_m,
                bounds.min_z - clearance_m,
                bounds.max_z + clearance_m,
            )
            if _segment_intersects_bounds(first, second, expanded):
                raise SceneLifecycleError(
                    f"transit corridor intersects {item.specification_name!r}"
                )


def _segment_intersects_bounds(
    first: Vector3, second: Vector3, bounds: Bounds3D
) -> bool:
    minimum_t = 0.0
    maximum_t = 1.0
    for start, end, lower, upper in (
        (first.x, second.x, bounds.min_x, bounds.max_x),
        (first.y, second.y, bounds.min_y, bounds.max_y),
        (first.z, second.z, bounds.min_z, bounds.max_z),
    ):
        delta = end - start
        if abs(delta) <= GEOMETRY_TOLERANCE_M:
            if start < lower or start > upper:
                return False
            continue
        near = (lower - start) / delta
        far = (upper - start) / delta
        if near > far:
            near, far = far, near
        minimum_t = max(minimum_t, near)
        maximum_t = min(maximum_t, far)
        if minimum_t > maximum_t:
            return False
    return True


def _require_point_outside_scene_geometry(
    point: Vector3, materialized: MaterializedScene
) -> None:
    for item in materialized.objects:
        requested = item.requested_transform
        bounds = requested.conservative_world_bounds
        if (
            bounds.min_x - GEOMETRY_TOLERANCE_M
            <= point.x
            <= bounds.max_x + GEOMETRY_TOLERANCE_M
            and bounds.min_y - GEOMETRY_TOLERANCE_M
            <= point.y
            <= bounds.max_y + GEOMETRY_TOLERANCE_M
            and bounds.min_z - GEOMETRY_TOLERANCE_M
            <= point.z
            <= bounds.max_z + GEOMETRY_TOLERANCE_M
        ):
            raise SceneLifecycleError("vehicle transit point intersects scene geometry")


def _distance(first: Vector3, second: Vector3) -> float:
    return math.dist(first.values(), second.values())


def _workspace_world(scene: ValidatedScene, origin: Vector3) -> Bounds3D:
    workspace = scene.config.workspace
    return Bounds3D(
        workspace.min_x + origin.x,
        workspace.max_x + origin.x,
        workspace.min_y + origin.y,
        workspace.max_y + origin.y,
        workspace.min_z + origin.z,
        workspace.max_z + origin.z,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
