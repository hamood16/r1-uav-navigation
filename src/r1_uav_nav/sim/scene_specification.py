"""Pure deterministic scene specifications for M13.2 Colosseum courses."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

SCENE_SCHEMA_VERSION = 1
ASSET_CATALOG_SCHEMA_VERSION = 1
GEOMETRY_TOLERANCE_M = 1e-6
DEFAULT_MIN_OBJECT_SEPARATION_M = 0.10
RUNTIME_NAME_PREFIX = "r1_uav_m13s2_"
MAX_SPECIFICATION_NAME_LENGTH = 32
MAX_RUNTIME_NAME_LENGTH = 96
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


class SceneSpecificationError(ValueError):
    """Base error for invalid M13.2 scene data."""


class SceneValidationError(SceneSpecificationError):
    """Raised when strict validation rejects a scene."""

    def __init__(self, result: SceneValidationResult):
        self.result = result
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in result.errors)
        super().__init__(detail or "scene validation failed")


class SceneGenerationError(SceneSpecificationError):
    """Raised when a deterministic generator exhausts its placement budget."""


class CollisionIntent(str, Enum):
    """Intended semantics without claiming verified simulator collision response."""

    SOLID_EXPECTED = "solid_expected"
    VISUAL_ONLY = "visual_only"
    NONE = "none"


class AssetCalibrationStatus(str, Enum):
    """Whether dimensional materialization may use an asset."""

    REQUIRES_LIVE_VALIDATION = "requires_live_validation"
    ACCEPTED = "accepted"


class CalibrationEvidenceLevel(str, Enum):
    """Strength of physical-size evidence recorded in the asset catalog."""

    UNVALIDATED = "unvalidated"
    SOURCE_VERIFIED = "source_verified"
    OPERATOR_CONFIRMED_NOMINAL = "operator_confirmed_nominal"


class MotionMode(str, Enum):
    """Schema-only future dynamic-obstacle motion modes."""

    WAYPOINT_LOOP = "waypoint_loop"
    WAYPOINT_PING_PONG = "waypoint_ping_pong"


@dataclass(frozen=True)
class Vector3:
    """Three-dimensional value in metres using local NED axes."""

    x: float
    y: float
    z: float

    def values(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Dimensions3D:
    """Requested width (x), depth (y), and height."""

    width: float
    depth: float
    height: float


@dataclass(frozen=True)
class WorkspaceBounds:
    """Inclusive local NED workspace limits."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float


@dataclass(frozen=True)
class Bounds3D:
    """World-aligned bounds used for conservative validation."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float


@dataclass(frozen=True)
class InitialVehicleExclusionConfig:
    """Clear volume around the measured initially grounded vehicle."""

    horizontal_clearance_m: float = 2.0
    vertical_clearance_m: float = 3.0
    below_ground_tolerance_m: float = 0.25


@dataclass(frozen=True)
class SceneReferenceSpec:
    """How a local course is translated into the live world."""

    origin_policy: str = "measured_grounded_vehicle"
    initial_vehicle_local_position: Vector3 = field(
        default_factory=lambda: Vector3(0.0, 0.0, 0.0)
    )
    initial_vehicle_exclusion: InitialVehicleExclusionConfig = field(
        default_factory=InitialVehicleExclusionConfig
    )


@dataclass(frozen=True)
class AppearanceSpec:
    """Optional visual configuration, separate from physical-geometry intent."""

    material_name: str | None = None
    marker_color_rgba: tuple[float, float, float, float] | None = None
    segmentation_id: int | None = None


@dataclass(frozen=True)
class PadClearance:
    """Obstacle exclusion and airborne-point clearance around a pad."""

    horizontal_margin_m: float = 1.0
    vertical_clearance_m: float = 3.0
    anchor_clearance_m: float = 2.0
    minimum_anchor_clearance_m: float = 1.0


@dataclass(frozen=True)
class AssetCalibration:
    """Accepted or pending nominal physical-size evidence for one runtime asset."""

    asset_name: str
    nominal_dimensions_m: Dimensions3D | None
    status: AssetCalibrationStatus
    evidence_level: CalibrationEvidenceLevel
    uncertainty_m: float | None
    tested_stack: str | None
    evidence_reference: str | None
    scale_readback_verified: bool

    @property
    def accepted_for_materialization(self) -> bool:
        return (
            self.status is AssetCalibrationStatus.ACCEPTED
            and self.evidence_level
            in {
                CalibrationEvidenceLevel.SOURCE_VERIFIED,
                CalibrationEvidenceLevel.OPERATOR_CONFIRMED_NOMINAL,
            }
            and self.nominal_dimensions_m is not None
        )


@dataclass(frozen=True)
class AssetCatalog:
    """Versioned set of runtime asset calibration records."""

    schema_version: int
    catalog_version: int
    assets: tuple[AssetCalibration, ...]

    def calibration_for(self, asset_name: str) -> AssetCalibration | None:
        return next(
            (item for item in self.assets if item.asset_name == asset_name), None
        )


@dataclass(frozen=True)
class SceneObjectSpec:
    """Fields shared by pads and obstacles."""

    name: str
    base_center: Vector3
    dimensions: Dimensions3D
    yaw_degrees: float = 0.0
    runtime_asset_name: str = "Cube"
    collision_intent: CollisionIntent = CollisionIntent.SOLID_EXPECTED
    physical_geometry_expected: bool = True
    physics_enabled: bool = False
    collision_response_verified: bool = False
    appearance: AppearanceSpec = field(default_factory=AppearanceSpec)
    prebuilt_name: str | None = None


@dataclass(frozen=True)
class StartPad(SceneObjectSpec):
    """Red start-pad geometry and its safety-clearance volume."""

    clearance: PadClearance = field(default_factory=PadClearance)


@dataclass(frozen=True)
class GoalPad(SceneObjectSpec):
    """Goal-pad geometry and its approach-point clearance."""

    clearance: PadClearance = field(default_factory=PadClearance)


@dataclass(frozen=True)
class StaticObstacle(SceneObjectSpec):
    """One deterministic static obstacle."""


@dataclass(frozen=True)
class ObstacleMotion:
    """Future dynamic motion schema; no M13.2 runtime behavior."""

    mode: MotionMode
    waypoints: tuple[Vector3, ...]
    speed_m_s: float


@dataclass(frozen=True)
class DynamicObstacle(SceneObjectSpec):
    """Schema-only dynamic obstacle reserved for a later milestone."""

    motion: ObstacleMotion | None = None


@dataclass(frozen=True)
class SceneGenerationConfig:
    """Bounded deterministic static-obstacle generation."""

    enabled: bool = False
    seed: int = 0
    obstacle_count: int = 0
    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = (0.0, 0.0)
    width_range: tuple[float, float] = (1.0, 1.0)
    depth_range: tuple[float, float] = (1.0, 1.0)
    height_range: tuple[float, float] = (1.0, 1.0)
    yaw_choices_degrees: tuple[float, ...] = (0.0,)
    max_attempts_per_object: int = 100
    max_total_attempts: int = 1000
    runtime_asset_name: str = "Cube"


@dataclass(frozen=True)
class SceneConfig:
    """Complete local scene specification before live-world translation."""

    schema_version: int
    scene_id: str
    workspace: WorkspaceBounds
    reference: SceneReferenceSpec
    start_pad: StartPad
    goal_pad: GoalPad
    static_obstacles: tuple[StaticObstacle, ...] = ()
    dynamic_obstacles: tuple[DynamicObstacle, ...] = ()
    generation: SceneGenerationConfig = field(default_factory=SceneGenerationConfig)
    minimum_object_separation_m: float = DEFAULT_MIN_OBJECT_SEPARATION_M


@dataclass(frozen=True)
class SceneValidationIssue:
    """One stable validation result suitable for CLI and tests."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class SceneValidationResult:
    """Structured validation result; strict callers may raise it."""

    errors: tuple[SceneValidationIssue, ...]
    warnings: tuple[SceneValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ValidatedScene:
    """Resolved local scene and its derived deterministic evidence."""

    config: SceneConfig
    start_anchor: Vector3
    goal_approach: Vector3
    canonical_json: str
    scene_digest: str


def normalize_yaw_degrees(value: float) -> float:
    """Normalize yaw to the deterministic half-open range [-180, 180)."""
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return 0.0 if normalized == -0.0 else normalized


def conservative_bounds(spec: SceneObjectSpec) -> Bounds3D:
    """Return safe world-aligned bounds for a yaw-rotated box."""
    radians = math.radians(normalize_yaw_degrees(spec.yaw_degrees))
    half_x = (
        abs(math.cos(radians)) * spec.dimensions.width / 2.0
        + abs(math.sin(radians)) * spec.dimensions.depth / 2.0
    )
    half_y = (
        abs(math.sin(radians)) * spec.dimensions.width / 2.0
        + abs(math.cos(radians)) * spec.dimensions.depth / 2.0
    )
    return Bounds3D(
        spec.base_center.x - half_x,
        spec.base_center.x + half_x,
        spec.base_center.y - half_y,
        spec.base_center.y + half_y,
        spec.base_center.z - spec.dimensions.height,
        spec.base_center.z,
    )


def pad_safety_bounds(pad: StartPad | GoalPad) -> Bounds3D:
    """Expand a pad footprint and airspace using its configured margins."""
    bounds = conservative_bounds(pad)
    return Bounds3D(
        bounds.min_x - pad.clearance.horizontal_margin_m,
        bounds.max_x + pad.clearance.horizontal_margin_m,
        bounds.min_y - pad.clearance.horizontal_margin_m,
        bounds.max_y + pad.clearance.horizontal_margin_m,
        bounds.min_z - pad.clearance.vertical_clearance_m,
        bounds.max_z,
    )


def derive_start_anchor(start_pad: StartPad) -> Vector3:
    """Calculate an airborne point above the start pad in NED."""
    return Vector3(
        start_pad.base_center.x,
        start_pad.base_center.y,
        start_pad.base_center.z
        - start_pad.dimensions.height
        - start_pad.clearance.anchor_clearance_m,
    )


def derive_goal_approach(goal_pad: GoalPad) -> Vector3:
    """Calculate an airborne approach point above the goal pad in NED."""
    return Vector3(
        goal_pad.base_center.x,
        goal_pad.base_center.y,
        goal_pad.base_center.z
        - goal_pad.dimensions.height
        - goal_pad.clearance.anchor_clearance_m,
    )


def build_initial_vehicle_exclusion(
    position: Vector3, config: InitialVehicleExclusionConfig
) -> Bounds3D:
    """Build the measured world-space exclusion prism around the initial UAV."""
    return Bounds3D(
        position.x - config.horizontal_clearance_m,
        position.x + config.horizontal_clearance_m,
        position.y - config.horizontal_clearance_m,
        position.y + config.horizontal_clearance_m,
        position.z - config.vertical_clearance_m,
        position.z + config.below_ground_tolerance_m,
    )


def translate_vector(vector: Vector3, origin: Vector3) -> Vector3:
    """Translate a local NED vector into live world coordinates."""
    return Vector3(vector.x + origin.x, vector.y + origin.y, vector.z + origin.z)


def translate_bounds(bounds: Bounds3D, origin: Vector3) -> Bounds3D:
    """Translate local bounds into live world coordinates."""
    return Bounds3D(
        bounds.min_x + origin.x,
        bounds.max_x + origin.x,
        bounds.min_y + origin.y,
        bounds.max_y + origin.y,
        bounds.min_z + origin.z,
        bounds.max_z + origin.z,
    )


def bounds_intersect(
    first: Bounds3D,
    second: Bounds3D,
    *,
    tolerance: float = GEOMETRY_TOLERANCE_M,
    separation: float = 0.0,
) -> bool:
    """Treat boundary contact and configured separation as intersection."""
    margin = tolerance + separation
    return not (
        first.max_x < second.min_x - margin
        or second.max_x < first.min_x - margin
        or first.max_y < second.min_y - margin
        or second.max_y < first.min_y - margin
        or first.max_z < second.min_z - margin
        or second.max_z < first.min_z - margin
    )


def bounds_inside_workspace(
    bounds: Bounds3D,
    workspace: WorkspaceBounds,
    *,
    tolerance: float = GEOMETRY_TOLERANCE_M,
) -> bool:
    """Allow workspace-boundary contact within the numeric tolerance."""
    return (
        bounds.min_x >= workspace.min_x - tolerance
        and bounds.max_x <= workspace.max_x + tolerance
        and bounds.min_y >= workspace.min_y - tolerance
        and bounds.max_y <= workspace.max_y + tolerance
        and bounds.min_z >= workspace.min_z - tolerance
        and bounds.max_z <= workspace.max_z + tolerance
    )


def runtime_object_name(scene_id: str, specification_name: str, digest: str) -> str:
    """Build one deterministic bounded runtime name."""
    _validate_name_value(scene_id, "scene_id")
    _validate_name_value(specification_name, "specification_name")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("digest must be a lowercase SHA-256 hex value")
    name = f"{RUNTIME_NAME_PREFIX}{scene_id}__{specification_name}__{digest[:12]}"
    if len(name) > MAX_RUNTIME_NAME_LENGTH:
        raise ValueError("generated runtime object name exceeds 96 characters")
    return name


def validate_scene(scene: SceneConfig) -> SceneValidationResult:
    """Validate a resolved scene without importing simulator code."""
    errors: list[SceneValidationIssue] = []

    def error(code: str, path: str, message: str) -> None:
        errors.append(SceneValidationIssue(code, path, message))

    if scene.schema_version != SCENE_SCHEMA_VERSION:
        error(
            "schema_version",
            "schema_version",
            f"expected {SCENE_SCHEMA_VERSION}, got {scene.schema_version}",
        )
    _collect_name_error(scene.scene_id, "scene_id", error)
    _validate_workspace(scene.workspace, error)
    _validate_reference(scene.reference, error)
    _validate_nonnegative(
        scene.minimum_object_separation_m, "minimum_object_separation_m", error
    )
    _validate_generation(scene.generation, error)

    objects: list[tuple[str, SceneObjectSpec]] = [
        ("start_pad", scene.start_pad),
        ("goal_pad", scene.goal_pad),
        *[
            (f"static_obstacles[{index}]", obstacle)
            for index, obstacle in enumerate(scene.static_obstacles)
        ],
        *[
            (f"dynamic_obstacles[{index}]", obstacle)
            for index, obstacle in enumerate(scene.dynamic_obstacles)
        ],
    ]
    seen_names: dict[str, str] = {}
    runtime_names: dict[str, str] = {}
    placeholder_digest = "0" * 64
    for path, spec in objects:
        _validate_object(spec, path, error)
        key = spec.name.casefold()
        if key in seen_names:
            error(
                "duplicate_name",
                f"{path}.name",
                f"name collides with {seen_names[key]} after normalization",
            )
        else:
            seen_names[key] = path
        try:
            generated = runtime_object_name(
                scene.scene_id, spec.name, placeholder_digest
            )
        except ValueError as exc:
            error("runtime_name", f"{path}.name", str(exc))
        else:
            key = generated.casefold()
            if key in runtime_names:
                error(
                    "runtime_name_collision",
                    f"{path}.name",
                    f"runtime name collides with {runtime_names[key]}",
                )
            runtime_names[key] = path

        bounds = conservative_bounds(spec)
        if not bounds_inside_workspace(bounds, scene.workspace):
            error("outside_workspace", path, "object bounds leave the workspace")
        if isinstance(spec, (StartPad, GoalPad)):
            _validate_pad(spec, path, scene.workspace, error)
        if isinstance(spec, DynamicObstacle):
            _validate_motion(
                spec,
                f"{path}.motion",
                scene.workspace,
                (pad_safety_bounds(scene.start_pad), pad_safety_bounds(scene.goal_pad)),
                error,
            )

    start_safe = pad_safety_bounds(scene.start_pad)
    goal_safe = pad_safety_bounds(scene.goal_pad)
    if bounds_intersect(start_safe, goal_safe):
        error(
            "pad_safety_overlap",
            "start_pad/goal_pad",
            "start and goal safety volumes overlap or touch",
        )

    initial_local = scene.reference.initial_vehicle_local_position
    initial_exclusion = build_initial_vehicle_exclusion(
        initial_local, scene.reference.initial_vehicle_exclusion
    )
    for path, spec in objects:
        bounds_to_check = (
            pad_safety_bounds(spec)
            if isinstance(spec, (StartPad, GoalPad))
            else conservative_bounds(spec)
        )
        if bounds_intersect(bounds_to_check, initial_exclusion):
            error(
                "initial_vehicle_exclusion",
                path,
                "object or safety volume intersects the initial vehicle exclusion",
            )

    obstacles = [
        (path, spec)
        for path, spec in objects
        if isinstance(spec, (StaticObstacle, DynamicObstacle))
    ]
    for path, obstacle in obstacles:
        obstacle_bounds = conservative_bounds(obstacle)
        for pad_path, safety in (
            ("start_pad", start_safe),
            ("goal_pad", goal_safe),
        ):
            if bounds_intersect(obstacle_bounds, safety):
                error(
                    "obstacle_pad_safety_overlap",
                    path,
                    f"obstacle intersects {pad_path} safety volume",
                )

    for index, (first_path, first) in enumerate(obstacles):
        for second_path, second in obstacles[index + 1 :]:
            if bounds_intersect(
                conservative_bounds(first),
                conservative_bounds(second),
                separation=scene.minimum_object_separation_m,
            ):
                error(
                    "obstacle_overlap",
                    f"{first_path}/{second_path}",
                    "obstacles overlap, touch, or violate minimum separation",
                )

    return SceneValidationResult(tuple(errors))


def require_valid_scene(scene: SceneConfig) -> ValidatedScene:
    """Validate and return derived canonical scene evidence."""
    result = validate_scene(scene)
    if not result.valid:
        raise SceneValidationError(result)
    canonical = canonical_scene_json(scene)
    return ValidatedScene(
        config=scene,
        start_anchor=derive_start_anchor(scene.start_pad),
        goal_approach=derive_goal_approach(scene.goal_pad),
        canonical_json=canonical,
        scene_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def validate_world_vehicle_exclusion(
    scene: ValidatedScene, world_origin: Vector3, measured_vehicle_position: Vector3
) -> SceneValidationResult:
    """Validate translated scene geometry against the measured initial UAV volume."""
    errors: list[SceneValidationIssue] = []
    exclusion = build_initial_vehicle_exclusion(
        measured_vehicle_position,
        scene.config.reference.initial_vehicle_exclusion,
    )
    objects: Iterable[tuple[str, SceneObjectSpec]] = (
        ("start_pad", scene.config.start_pad),
        ("goal_pad", scene.config.goal_pad),
        *[
            (f"static_obstacles[{index}]", obstacle)
            for index, obstacle in enumerate(scene.config.static_obstacles)
        ],
        *[
            (f"dynamic_obstacles[{index}]", obstacle)
            for index, obstacle in enumerate(scene.config.dynamic_obstacles)
        ],
    )
    for path, spec in objects:
        local = (
            pad_safety_bounds(spec)
            if isinstance(spec, (StartPad, GoalPad))
            else conservative_bounds(spec)
        )
        if bounds_intersect(translate_bounds(local, world_origin), exclusion):
            errors.append(
                SceneValidationIssue(
                    "initial_vehicle_exclusion",
                    path,
                    "translated object or safety volume intersects the measured "
                    "initial vehicle exclusion",
                )
            )
    return SceneValidationResult(tuple(errors))


def generate_scene(scene: SceneConfig) -> SceneConfig:
    """Resolve deterministic generated obstacles with bounded local RNG use."""
    generation = scene.generation
    if not generation.enabled or generation.obstacle_count == 0:
        return replace(scene, generation=replace(generation, enabled=False))
    rng = np.random.default_rng(generation.seed)
    generated: list[StaticObstacle] = []
    attempts = 0
    for index in range(generation.obstacle_count):
        placed = False
        for _ in range(generation.max_attempts_per_object):
            attempts += 1
            if attempts > generation.max_total_attempts:
                break
            obstacle = StaticObstacle(
                name=f"obstacle-{index:03d}",
                base_center=Vector3(
                    _rounded_uniform(rng, generation.x_range),
                    _rounded_uniform(rng, generation.y_range),
                    0.0,
                ),
                dimensions=Dimensions3D(
                    _rounded_uniform(rng, generation.width_range),
                    _rounded_uniform(rng, generation.depth_range),
                    _rounded_uniform(rng, generation.height_range),
                ),
                yaw_degrees=float(rng.choice(generation.yaw_choices_degrees)),
                runtime_asset_name=generation.runtime_asset_name,
            )
            candidate = replace(
                scene,
                static_obstacles=scene.static_obstacles
                + tuple(generated)
                + (obstacle,),
                generation=replace(generation, enabled=False),
            )
            if validate_scene(candidate).valid:
                generated.append(obstacle)
                placed = True
                break
        if not placed:
            raise SceneGenerationError(
                f"could not place obstacle-{index:03d} after {attempts} total attempts"
            )
    return replace(
        scene,
        static_obstacles=scene.static_obstacles + tuple(generated),
        generation=replace(generation, enabled=False),
    )


def resolve_scene(scene: SceneConfig) -> ValidatedScene:
    """Generate any requested obstacles and validate the resolved scene."""
    return require_valid_scene(generate_scene(scene))


def canonical_scene_dict(scene: SceneConfig) -> dict[str, Any]:
    """Return a normalized local scene representation independent of world origin."""
    return _normalize_json(asdict(scene))


def canonical_scene_json(scene: SceneConfig) -> str:
    """Serialize a canonical local scene with stable ordering."""
    return json.dumps(
        canonical_scene_dict(scene),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def scene_digest(scene: SceneConfig) -> str:
    """Hash only the canonical local resolved scene."""
    return hashlib.sha256(canonical_scene_json(scene).encode("utf-8")).hexdigest()


def materialization_digest(
    *,
    local_scene_digest: str,
    backend: str,
    backend_version: str,
    asset_catalog_digest: str,
    calibration_evidence: Mapping[str, Any],
    world_origin: Vector3,
    requested_world_transforms: Mapping[str, Any],
) -> str:
    """Hash world/backend evidence separately from the local scene identity."""
    payload = _normalize_json(
        {
            "scene_digest": local_scene_digest,
            "backend": backend,
            "backend_version": backend_version,
            "asset_catalog_digest": asset_catalog_digest,
            "calibration_evidence": dict(calibration_evidence),
            "world_origin": asdict(world_origin),
            "requested_world_transforms": dict(requested_world_transforms),
        }
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def asset_catalog_digest(catalog: AssetCatalog) -> str:
    """Hash the versioned accepted calibration catalog."""
    payload = json.dumps(
        _normalize_json(asdict(catalog)), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_scene_config(path: str | Path) -> SceneConfig:
    """Load a versioned scene YAML file into immutable typed data."""
    raw = _load_yaml_mapping(path)
    return _scene_from_mapping(raw)


def load_asset_catalog(path: str | Path) -> AssetCatalog:
    """Load the versioned asset calibration catalog."""
    raw = _load_yaml_mapping(path)
    assets_raw = raw.get("assets")
    if not isinstance(assets_raw, list):
        raise SceneSpecificationError("asset catalog assets must be a list")
    assets = tuple(_asset_calibration_from_mapping(item) for item in assets_raw)
    catalog = AssetCatalog(
        schema_version=_required_int(raw, "schema_version"),
        catalog_version=_required_int(raw, "catalog_version"),
        assets=assets,
    )
    if catalog.schema_version != ASSET_CATALOG_SCHEMA_VERSION:
        raise SceneSpecificationError(
            f"expected asset schema {ASSET_CATALOG_SCHEMA_VERSION}"
        )
    names = [item.asset_name.casefold() for item in catalog.assets]
    if len(names) != len(set(names)):
        raise SceneSpecificationError("asset catalog names must be unique")
    for item in catalog.assets:
        _validate_asset_calibration(item)
    return catalog


def _scene_from_mapping(raw: Mapping[str, Any]) -> SceneConfig:
    return SceneConfig(
        schema_version=_required_int(raw, "schema_version"),
        scene_id=_required_str(raw, "scene_id"),
        workspace=_workspace_from_mapping(_required_mapping(raw, "workspace")),
        reference=_reference_from_mapping(_required_mapping(raw, "reference")),
        start_pad=_pad_from_mapping(_required_mapping(raw, "start_pad"), StartPad),
        goal_pad=_pad_from_mapping(_required_mapping(raw, "goal_pad"), GoalPad),
        static_obstacles=tuple(
            _object_from_mapping(item, StaticObstacle)
            for item in _mapping_list(raw.get("static_obstacles", []))
        ),
        dynamic_obstacles=tuple(
            _dynamic_from_mapping(item)
            for item in _mapping_list(raw.get("dynamic_obstacles", []))
        ),
        generation=_generation_from_mapping(
            _optional_mapping(raw.get("generation", {}), "generation")
        ),
        minimum_object_separation_m=float(
            raw.get(
                "minimum_object_separation_m",
                DEFAULT_MIN_OBJECT_SEPARATION_M,
            )
        ),
    )


def _object_from_mapping(
    raw: Mapping[str, Any], object_type: type[SceneObjectSpec]
) -> SceneObjectSpec:
    appearance = _appearance_from_mapping(
        _optional_mapping(raw.get("appearance", {}), "appearance")
    )
    return object_type(
        name=_required_str(raw, "name"),
        base_center=_vector_from_mapping(_required_mapping(raw, "base_center")),
        dimensions=_dimensions_from_mapping(_required_mapping(raw, "dimensions")),
        yaw_degrees=float(raw.get("yaw_degrees", 0.0)),
        runtime_asset_name=str(raw.get("runtime_asset_name", "Cube")),
        collision_intent=CollisionIntent(
            raw.get("collision_intent", CollisionIntent.SOLID_EXPECTED.value)
        ),
        physical_geometry_expected=bool(raw.get("physical_geometry_expected", True)),
        physics_enabled=bool(raw.get("physics_enabled", False)),
        collision_response_verified=bool(raw.get("collision_response_verified", False)),
        appearance=appearance,
        prebuilt_name=raw.get("prebuilt_name"),
    )


def _pad_from_mapping(
    raw: Mapping[str, Any], pad_type: type[StartPad] | type[GoalPad]
) -> StartPad | GoalPad:
    base = _object_from_mapping(raw, SceneObjectSpec)
    clearance_raw = _optional_mapping(raw.get("clearance", {}), "clearance")
    clearance = PadClearance(
        horizontal_margin_m=float(clearance_raw.get("horizontal_margin_m", 1.0)),
        vertical_clearance_m=float(clearance_raw.get("vertical_clearance_m", 3.0)),
        anchor_clearance_m=float(clearance_raw.get("anchor_clearance_m", 2.0)),
        minimum_anchor_clearance_m=float(
            clearance_raw.get("minimum_anchor_clearance_m", 1.0)
        ),
    )
    return pad_type(**_object_kwargs(base), clearance=clearance)


def _dynamic_from_mapping(raw: Mapping[str, Any]) -> DynamicObstacle:
    base = _object_from_mapping(raw, SceneObjectSpec)
    motion_raw = raw.get("motion")
    motion = None
    if motion_raw is not None:
        mapping = _optional_mapping(motion_raw, "motion")
        motion = ObstacleMotion(
            mode=MotionMode(_required_str(mapping, "mode")),
            waypoints=tuple(
                _vector_from_mapping(item)
                for item in _mapping_list(mapping.get("waypoints", []))
            ),
            speed_m_s=float(mapping.get("speed_m_s", 0.0)),
        )
    return DynamicObstacle(**_object_kwargs(base), motion=motion)


def _object_kwargs(spec: SceneObjectSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "base_center": spec.base_center,
        "dimensions": spec.dimensions,
        "yaw_degrees": spec.yaw_degrees,
        "runtime_asset_name": spec.runtime_asset_name,
        "collision_intent": spec.collision_intent,
        "physical_geometry_expected": spec.physical_geometry_expected,
        "physics_enabled": spec.physics_enabled,
        "collision_response_verified": spec.collision_response_verified,
        "appearance": spec.appearance,
        "prebuilt_name": spec.prebuilt_name,
    }


def _generation_from_mapping(raw: Mapping[str, Any]) -> SceneGenerationConfig:
    return SceneGenerationConfig(
        enabled=bool(raw.get("enabled", False)),
        seed=int(raw.get("seed", 0)),
        obstacle_count=int(raw.get("obstacle_count", 0)),
        x_range=_pair(raw.get("x_range", (0.0, 0.0)), "x_range"),
        y_range=_pair(raw.get("y_range", (0.0, 0.0)), "y_range"),
        width_range=_pair(raw.get("width_range", (1.0, 1.0)), "width_range"),
        depth_range=_pair(raw.get("depth_range", (1.0, 1.0)), "depth_range"),
        height_range=_pair(raw.get("height_range", (1.0, 1.0)), "height_range"),
        yaw_choices_degrees=tuple(
            float(value) for value in raw.get("yaw_choices_degrees", [0.0])
        ),
        max_attempts_per_object=int(raw.get("max_attempts_per_object", 100)),
        max_total_attempts=int(raw.get("max_total_attempts", 1000)),
        runtime_asset_name=str(raw.get("runtime_asset_name", "Cube")),
    )


def _reference_from_mapping(raw: Mapping[str, Any]) -> SceneReferenceSpec:
    exclusion = _optional_mapping(
        raw.get("initial_vehicle_exclusion", {}), "initial_vehicle_exclusion"
    )
    return SceneReferenceSpec(
        origin_policy=str(raw.get("origin_policy", "measured_grounded_vehicle")),
        initial_vehicle_local_position=_vector_from_mapping(
            _optional_mapping(
                raw.get(
                    "initial_vehicle_local_position",
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                ),
                "initial_vehicle_local_position",
            )
        ),
        initial_vehicle_exclusion=InitialVehicleExclusionConfig(
            horizontal_clearance_m=float(exclusion.get("horizontal_clearance_m", 2.0)),
            vertical_clearance_m=float(exclusion.get("vertical_clearance_m", 3.0)),
            below_ground_tolerance_m=float(
                exclusion.get("below_ground_tolerance_m", 0.25)
            ),
        ),
    )


def _appearance_from_mapping(raw: Mapping[str, Any]) -> AppearanceSpec:
    color = raw.get("marker_color_rgba")
    return AppearanceSpec(
        material_name=raw.get("material_name"),
        marker_color_rgba=(
            tuple(float(value) for value in color) if color is not None else None
        ),
        segmentation_id=(
            int(raw["segmentation_id"])
            if raw.get("segmentation_id") is not None
            else None
        ),
    )


def _asset_calibration_from_mapping(raw: Mapping[str, Any]) -> AssetCalibration:
    dimensions_raw = raw.get("nominal_dimensions_m")
    dimensions = (
        _dimensions_from_mapping(_optional_mapping(dimensions_raw, "dimensions"))
        if dimensions_raw is not None
        else None
    )
    return AssetCalibration(
        asset_name=_required_str(raw, "asset_name"),
        nominal_dimensions_m=dimensions,
        status=AssetCalibrationStatus(_required_str(raw, "status")),
        evidence_level=CalibrationEvidenceLevel(_required_str(raw, "evidence_level")),
        uncertainty_m=(
            float(raw["uncertainty_m"])
            if raw.get("uncertainty_m") is not None
            else None
        ),
        tested_stack=raw.get("tested_stack"),
        evidence_reference=raw.get("evidence_reference"),
        scale_readback_verified=bool(raw.get("scale_readback_verified", False)),
    )


def _workspace_from_mapping(raw: Mapping[str, Any]) -> WorkspaceBounds:
    return WorkspaceBounds(
        *(
            float(raw[key])
            for key in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
        )
    )


def _vector_from_mapping(raw: Mapping[str, Any]) -> Vector3:
    return Vector3(*(float(raw[key]) for key in ("x", "y", "z")))


def _dimensions_from_mapping(raw: Mapping[str, Any]) -> Dimensions3D:
    return Dimensions3D(*(float(raw[key]) for key in ("width", "depth", "height")))


def _validate_name_value(value: str, field_name: str) -> None:
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    if len(value) > MAX_SPECIFICATION_NAME_LENGTH:
        raise ValueError(
            f"{field_name} must not exceed {MAX_SPECIFICATION_NAME_LENGTH} characters"
        )
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match {_NAME_PATTERN.pattern!r}")


def _collect_name_error(
    value: str,
    path: str,
    error: Any,
) -> None:
    try:
        _validate_name_value(value, path)
    except (TypeError, ValueError) as exc:
        error("invalid_name", path, str(exc))


def _validate_workspace(workspace: WorkspaceBounds, error: Any) -> None:
    values = asdict(workspace)
    for name, value in values.items():
        if not math.isfinite(value):
            error("non_finite", f"workspace.{name}", "value must be finite")
    for lower, upper in (("min_x", "max_x"), ("min_y", "max_y"), ("min_z", "max_z")):
        if getattr(workspace, lower) >= getattr(workspace, upper):
            error(
                "invalid_bounds",
                f"workspace.{lower}/{upper}",
                "minimum must be less than maximum",
            )


def _validate_reference(reference: SceneReferenceSpec, error: Any) -> None:
    if reference.origin_policy != "measured_grounded_vehicle":
        error(
            "unsupported_origin_policy",
            "reference.origin_policy",
            "M13.2 supports measured_grounded_vehicle only",
        )
    _validate_vector_values(
        reference.initial_vehicle_local_position,
        "reference.initial_vehicle_local_position",
        error,
    )
    exclusion = reference.initial_vehicle_exclusion
    _validate_positive(
        exclusion.horizontal_clearance_m,
        "reference.initial_vehicle_exclusion.horizontal_clearance_m",
        error,
    )
    _validate_positive(
        exclusion.vertical_clearance_m,
        "reference.initial_vehicle_exclusion.vertical_clearance_m",
        error,
    )
    _validate_nonnegative(
        exclusion.below_ground_tolerance_m,
        "reference.initial_vehicle_exclusion.below_ground_tolerance_m",
        error,
    )


def _validate_object(spec: SceneObjectSpec, path: str, error: Any) -> None:
    _collect_name_error(spec.name, f"{path}.name", error)
    _validate_vector_values(spec.base_center, f"{path}.base_center", error)
    for name, value in asdict(spec.dimensions).items():
        _validate_positive(value, f"{path}.dimensions.{name}", error)
    if not math.isfinite(spec.yaw_degrees):
        error("invalid_orientation", f"{path}.yaw_degrees", "yaw must be finite")
    if not spec.runtime_asset_name.strip():
        error("invalid_asset", f"{path}.runtime_asset_name", "asset must not be empty")
    if spec.collision_response_verified:
        error(
            "unsupported_collision_claim",
            f"{path}.collision_response_verified",
            "M13.2 has no verified collision-response evidence",
        )
    if spec.physics_enabled:
        error(
            "unsupported_physics",
            f"{path}.physics_enabled",
            "M13.2 runtime solids must keep physics disabled",
        )
    if spec.collision_intent is CollisionIntent.VISUAL_ONLY:
        if spec.physical_geometry_expected:
            error(
                "appearance_collision_mismatch",
                path,
                "visual-only objects cannot expect physical geometry",
            )
    elif spec.collision_intent is CollisionIntent.SOLID_EXPECTED:
        if not spec.physical_geometry_expected:
            error(
                "appearance_collision_mismatch",
                path,
                "solid-expected objects require physical_geometry_expected",
            )
    if spec.appearance.marker_color_rgba is not None:
        if len(spec.appearance.marker_color_rgba) != 4 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in spec.appearance.marker_color_rgba
        ):
            error(
                "invalid_marker_color",
                f"{path}.appearance.marker_color_rgba",
                "RGBA values must contain four finite values in [0, 1]",
            )
    if spec.appearance.segmentation_id is not None and not (
        0 <= spec.appearance.segmentation_id <= 255
    ):
        error(
            "invalid_segmentation_id",
            f"{path}.appearance.segmentation_id",
            "segmentation ID must be between 0 and 255",
        )


def _validate_pad(
    pad: StartPad | GoalPad,
    path: str,
    workspace: WorkspaceBounds,
    error: Any,
) -> None:
    clearance = pad.clearance
    for name, value in asdict(clearance).items():
        _validate_positive(value, f"{path}.clearance.{name}", error)
    if clearance.anchor_clearance_m < clearance.minimum_anchor_clearance_m:
        error(
            "insufficient_anchor_clearance",
            f"{path}.clearance.anchor_clearance_m",
            "anchor clearance is below the configured minimum",
        )
    if not bounds_inside_workspace(pad_safety_bounds(pad), workspace):
        error(
            "pad_safety_outside_workspace",
            path,
            "pad safety volume leaves the workspace",
        )
    anchor = (
        derive_start_anchor(pad)
        if isinstance(pad, StartPad)
        else derive_goal_approach(pad)
    )
    if not (
        workspace.min_x - GEOMETRY_TOLERANCE_M
        <= anchor.x
        <= workspace.max_x + GEOMETRY_TOLERANCE_M
        and workspace.min_y - GEOMETRY_TOLERANCE_M
        <= anchor.y
        <= workspace.max_y + GEOMETRY_TOLERANCE_M
        and workspace.min_z - GEOMETRY_TOLERANCE_M
        <= anchor.z
        <= workspace.max_z + GEOMETRY_TOLERANCE_M
    ):
        error(
            "anchor_outside_workspace",
            path,
            "derived airborne point is outside workspace",
        )


def _validate_motion(
    obstacle: DynamicObstacle,
    path: str,
    workspace: WorkspaceBounds,
    pad_safety_volumes: tuple[Bounds3D, Bounds3D],
    error: Any,
) -> None:
    motion = obstacle.motion
    if motion is None:
        error("missing_motion", path, "dynamic obstacle requires a motion definition")
        return
    if not isinstance(motion.mode, MotionMode):
        error("invalid_motion_mode", f"{path}.mode", "unsupported motion mode")
    _validate_positive(motion.speed_m_s, f"{path}.speed_m_s", error)
    if len(motion.waypoints) < 2:
        error("malformed_motion_path", path, "motion path requires at least two points")
    for index, waypoint in enumerate(motion.waypoints):
        _validate_vector_values(waypoint, f"{path}.waypoints[{index}]", error)
    if len({point.values() for point in motion.waypoints}) < 2:
        error("malformed_motion_path", path, "motion path requires distinct points")
    for index, waypoint in enumerate(motion.waypoints):
        candidate = replace(obstacle, base_center=waypoint)
        candidate_bounds = conservative_bounds(candidate)
        if not bounds_inside_workspace(candidate_bounds, workspace):
            error(
                "motion_path_outside_workspace",
                f"{path}.waypoints[{index}]",
                "dynamic obstacle leaves the workspace",
            )
        if any(
            bounds_intersect(candidate_bounds, safety) for safety in pad_safety_volumes
        ):
            error(
                "motion_path_pad_overlap",
                f"{path}.waypoints[{index}]",
                "dynamic obstacle path intersects a pad safety volume",
            )


def _validate_generation(generation: SceneGenerationConfig, error: Any) -> None:
    if generation.obstacle_count < 0:
        error("invalid_generation", "generation.obstacle_count", "must be non-negative")
    if generation.max_attempts_per_object <= 0 or generation.max_total_attempts <= 0:
        error("invalid_generation", "generation", "attempt budgets must be positive")
    if not generation.yaw_choices_degrees:
        error(
            "invalid_generation", "generation.yaw_choices_degrees", "must not be empty"
        )
    for name in (
        "x_range",
        "y_range",
        "width_range",
        "depth_range",
        "height_range",
    ):
        lower, upper = getattr(generation, name)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            error("invalid_generation_range", f"generation.{name}", "invalid range")
        if name in {"width_range", "depth_range", "height_range"} and lower <= 0:
            error(
                "invalid_generation_range",
                f"generation.{name}",
                "dimension ranges must be positive",
            )


def _validate_asset_calibration(calibration: AssetCalibration) -> None:
    if not calibration.asset_name.strip():
        raise SceneSpecificationError("asset_name must not be empty")
    if calibration.nominal_dimensions_m is not None:
        for value in asdict(calibration.nominal_dimensions_m).values():
            if not math.isfinite(value) or value <= 0:
                raise SceneSpecificationError(
                    "nominal asset dimensions must be finite and positive"
                )
    if calibration.uncertainty_m is not None and (
        not math.isfinite(calibration.uncertainty_m) or calibration.uncertainty_m < 0
    ):
        raise SceneSpecificationError(
            "asset uncertainty must be finite and non-negative"
        )
    if calibration.status is AssetCalibrationStatus.ACCEPTED:
        if not calibration.accepted_for_materialization:
            raise SceneSpecificationError(
                "accepted asset calibration requires dimensions and accepted evidence"
            )
        if not calibration.evidence_reference:
            raise SceneSpecificationError(
                "accepted asset calibration requires an evidence reference"
            )
        if not calibration.tested_stack:
            raise SceneSpecificationError(
                "accepted asset calibration requires tested stack evidence"
            )


def _validate_vector_values(vector: Vector3, path: str, error: Any) -> None:
    for name, value in asdict(vector).items():
        if not math.isfinite(value):
            error("non_finite", f"{path}.{name}", "value must be finite")


def _validate_positive(value: float, path: str, error: Any) -> None:
    if not math.isfinite(value) or value <= 0:
        error("invalid_positive_value", path, "value must be finite and positive")


def _validate_nonnegative(value: float, path: str, error: Any) -> None:
    if not math.isfinite(value) or value < 0:
        error(
            "invalid_nonnegative_value", path, "value must be finite and non-negative"
        )


def _rounded_uniform(
    rng: np.random.Generator, value_range: tuple[float, float]
) -> float:
    lower, upper = value_range
    return round(float(rng.uniform(lower, upper)), 6)


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SceneSpecificationError("canonical scene values must be finite")
        rounded = round(value, 6)
        return 0.0 if rounded == -0.0 else rounded
    return value


def _load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise SceneSpecificationError("configuration root must be a mapping")
    return raw


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _optional_mapping(raw.get(key), key)


def _optional_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SceneSpecificationError(f"{path} must be a mapping")
    return value


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise SceneSpecificationError("expected a list of mappings")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise SceneSpecificationError("expected a list of mappings")
        result.append(item)
    return result


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise SceneSpecificationError(f"{key} must be a string")
    return value


def _required_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SceneSpecificationError(f"{key} must be an integer")
    return value


def _pair(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SceneSpecificationError(f"{path} must contain two values")
    return (float(value[0]), float(value[1]))
