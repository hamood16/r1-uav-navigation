"""Deterministic M13.3 static-course generation and solvability validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

from r1_uav_nav.planners.voxel_astar import (
    OCCUPANCY_EVIDENCE_SCHEMA_VERSION,
    SOLVABILITY_EVIDENCE_SCHEMA_VERSION,
    VOXEL_PLANNER_CONFIG_SCHEMA_VERSION,
    ContinuousBounds3D,
    VoxelGrid,
    VoxelGridConfig,
    VoxelPathResult,
    build_voxel_grid,
    find_voxel_astar_path,
)
from r1_uav_nav.sim.scene_specification import (
    SceneGenerationError,
    SceneValidationError,
    ValidatedScene,
    Vector3,
    conservative_bounds,
    load_scene_config,
    resolve_scene,
)

COURSE_SUITE_SCHEMA_VERSION = 1
COURSE_REPORT_SCHEMA_VERSION = 1


class CourseDifficulty(str, Enum):
    """Declared static-course difficulty."""

    EMPTY = "empty"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    HELD_OUT = "held_out"


class CourseSplit(str, Enum):
    """Intended use of a deterministic course profile."""

    VALIDATION = "validation"
    TRAINING = "training"
    HELD_OUT = "held_out"


class XDirectionRequirement(str, Enum):
    """Required sign of the local start-to-goal x displacement."""

    ANY = "any"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class CourseSolvabilityError(ValueError):
    """Raised when strict validation rejects one resolved course."""

    def __init__(self, result: CourseValidationResult):
        self.result = result
        detail = "; ".join(result.constraint_errors)
        super().__init__(
            detail or result.path_result.failure_reason or "course is unsolvable"
        )


class CourseGenerationExhaustedError(ValueError):
    """Raised after all deterministic course candidates are rejected."""

    def __init__(self, profile_id: str, rejections: tuple[CandidateRejection, ...]):
        self.profile_id = profile_id
        self.rejections = rejections
        super().__init__(
            f"profile {profile_id!r} exhausted {len(rejections)} candidate attempts"
        )


@dataclass(frozen=True)
class DifficultyConstraints:
    """Mandatory numeric and structural profile requirements."""

    obstacle_count_min: int
    obstacle_count_max: int
    direct_line_clear: bool
    path_efficiency_min: float
    path_efficiency_max: float
    vertical_excursion_min_m: float
    vertical_excursion_max_m: float | None
    required_structure_prefixes: tuple[str, ...] = ()
    required_x_direction: XDirectionRequirement = XDirectionRequirement.ANY
    require_lateral_direction: bool = False
    minimum_goal_above_start_m: float = 0.0


@dataclass(frozen=True)
class FeasibilityBaseline:
    """Tracked deterministic acceptance evidence for one declared base seed."""

    base_seed: int
    accepted_candidate_seed: int
    attempt_index: int
    scene_digest: str
    occupancy_digest: str
    solvability_digest: str
    reference_path_length_m: float
    path_efficiency_ratio: float
    vertical_excursion_m: float
    expanded_nodes: int
    direct_line_clear: bool


@dataclass(frozen=True)
class CourseProfile:
    """One authoritative scene template, split, seeds, and constraints."""

    profile_id: str
    difficulty: CourseDifficulty
    split: CourseSplit
    scene_config: str
    base_seeds: tuple[int, ...]
    max_candidate_attempts: int
    constraints: DifficultyConstraints
    accepted_baselines: tuple[FeasibilityBaseline, ...] = ()

    def baseline_for(self, base_seed: int) -> FeasibilityBaseline | None:
        return next(
            (item for item in self.accepted_baselines if item.base_seed == base_seed),
            None,
        )


@dataclass(frozen=True)
class CourseSuiteConfig:
    """Authoritative M13.3 planner and profile registry."""

    schema_version: int
    planner: VoxelGridConfig
    profiles: tuple[CourseProfile, ...]

    def profile(self, profile_id: str) -> CourseProfile:
        matches = [item for item in self.profiles if item.profile_id == profile_id]
        if len(matches) != 1:
            raise ValueError(f"unknown course profile {profile_id!r}")
        return matches[0]


@dataclass(frozen=True)
class CandidateRejection:
    """Stable evidence explaining why one deterministic candidate was rejected."""

    attempt_index: int
    candidate_seed: int
    stage: str
    reason: str
    scene_digest: str | None = None
    occupancy_digest: str | None = None


@dataclass(frozen=True)
class CourseValidationResult:
    """One resolved scene's complete static-solvability evidence."""

    schema_version: int
    profile_id: str
    difficulty: CourseDifficulty
    split: CourseSplit
    base_seed: int
    accepted_candidate_seed: int
    attempt_index: int
    scene_digest: str
    occupancy_digest: str
    solvability_digest: str
    path_result: VoxelPathResult
    obstacle_count: int
    constraint_errors: tuple[str, ...]
    rejected_candidates: tuple[CandidateRejection, ...]

    @property
    def accepted(self) -> bool:
        return self.path_result.solvable and not self.constraint_errors


@dataclass(frozen=True)
class ValidatedCourse:
    """Accepted M13.2 scene plus M13.3 reference-path evidence."""

    scene: ValidatedScene
    grid: VoxelGrid
    result: CourseValidationResult


def load_course_suite_config(path: str | Path) -> CourseSuiteConfig:
    """Load and strictly validate the authoritative course-suite YAML."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("course suite must contain a mapping")
    schema_version = _required_int(raw, "schema_version")
    if schema_version != COURSE_SUITE_SCHEMA_VERSION:
        raise ValueError("unsupported course-suite schema")
    planner_raw = _required_mapping(raw, "planner")
    planner = VoxelGridConfig(
        schema_version=_required_int(planner_raw, "schema_version"),
        resolution_m=_required_float(planner_raw, "resolution_m"),
        uav_collision_radius_m=_required_float(planner_raw, "uav_collision_radius_m"),
        additional_safety_margin_m=_required_float(
            planner_raw, "additional_safety_margin_m"
        ),
        total_clearance_m=_required_float(planner_raw, "total_clearance_m"),
        connectivity=_required_int(planner_raw, "connectivity"),
        max_voxels=_required_int(planner_raw, "max_voxels"),
        max_expansions=_required_int(planner_raw, "max_expansions"),
    )
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ValueError("course suite profiles must be a non-empty list")
    profiles = tuple(_profile_from_mapping(item) for item in profiles_raw)
    identifiers = [item.profile_id for item in profiles]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("course profile IDs must be unique")
    _validate_seed_partitions(profiles)
    return CourseSuiteConfig(schema_version, planner, profiles)


def generate_solvable_course(
    suite: CourseSuiteConfig,
    profile_id: str,
    base_seed: int,
    *,
    repository_root: str | Path,
) -> ValidatedCourse:
    """Generate candidates until all geometry and difficulty gates pass."""
    profile = suite.profile(profile_id)
    if base_seed not in profile.base_seeds:
        raise ValueError(
            f"seed {base_seed} is not declared for profile {profile.profile_id!r}"
        )
    scene_path = resolve_profile_scene_path(profile, Path(repository_root))
    source = load_scene_config(scene_path)
    rejections: list[CandidateRejection] = []

    for attempt_index in range(profile.max_candidate_attempts):
        candidate_seed = base_seed + attempt_index
        candidate_source = replace(
            source,
            generation=replace(source.generation, seed=candidate_seed),
        )
        try:
            scene = resolve_scene(candidate_source)
        except (SceneGenerationError, SceneValidationError, ValueError) as exc:
            rejections.append(
                CandidateRejection(
                    attempt_index,
                    candidate_seed,
                    "scene_generation",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        try:
            validated = validate_static_course(
                scene,
                suite.planner,
                profile,
                base_seed=base_seed,
                candidate_seed=candidate_seed,
                attempt_index=attempt_index,
                rejected_candidates=tuple(rejections),
            )
        except ValueError as exc:
            rejections.append(
                CandidateRejection(
                    attempt_index,
                    candidate_seed,
                    "occupancy",
                    f"{type(exc).__name__}: {exc}",
                    scene_digest=scene.scene_digest,
                )
            )
            continue
        if validated.result.accepted:
            return validated
        rejections.append(
            CandidateRejection(
                attempt_index,
                candidate_seed,
                "solvability_or_difficulty",
                "; ".join(validated.result.constraint_errors)
                or validated.result.path_result.failure_reason
                or "course rejected",
                scene_digest=scene.scene_digest,
                occupancy_digest=validated.grid.occupancy_digest,
            )
        )
    raise CourseGenerationExhaustedError(profile.profile_id, tuple(rejections))


def validate_static_course(
    scene: ValidatedScene,
    planner: VoxelGridConfig,
    profile: CourseProfile,
    *,
    base_seed: int,
    candidate_seed: int,
    attempt_index: int,
    rejected_candidates: tuple[CandidateRejection, ...] = (),
) -> ValidatedCourse:
    """Validate one resolved static scene without a simulator dependency."""
    if scene.config.dynamic_obstacles:
        raise ValueError("M13.3 does not support dynamic obstacles")
    workspace = _workspace_bounds(scene)
    obstacle_bounds = tuple(_obstacle_bounds(scene))
    grid = build_voxel_grid(workspace, obstacle_bounds, planner)
    start = _point(scene.start_anchor)
    goal = _point(scene.goal_approach)
    path = find_voxel_astar_path(grid, start, goal)
    errors = _difficulty_errors(scene, path, profile.constraints)
    if not path.solvable:
        errors = errors + (
            path.failure_reason or f"path status is {path.status.value}",
        )
    digest = _solvability_digest(
        scene.scene_digest,
        planner,
        grid,
        path,
    )
    result = CourseValidationResult(
        schema_version=SOLVABILITY_EVIDENCE_SCHEMA_VERSION,
        profile_id=profile.profile_id,
        difficulty=profile.difficulty,
        split=profile.split,
        base_seed=base_seed,
        accepted_candidate_seed=candidate_seed,
        attempt_index=attempt_index,
        scene_digest=scene.scene_digest,
        occupancy_digest=grid.occupancy_digest,
        solvability_digest=digest,
        path_result=path,
        obstacle_count=len(scene.config.static_obstacles),
        constraint_errors=errors,
        rejected_candidates=rejected_candidates,
    )
    return ValidatedCourse(scene, grid, result)


def require_solvable_course(course: ValidatedCourse) -> ValidatedCourse:
    """Raise with structured evidence unless every M13.3 gate passed."""
    if not course.result.accepted:
        raise CourseSolvabilityError(course.result)
    return course


def course_report_dict(course: ValidatedCourse) -> dict[str, Any]:
    """Build one machine-readable report without changing scene identity."""
    return {
        "report_schema_version": COURSE_REPORT_SCHEMA_VERSION,
        "course_suite_schema_version": COURSE_SUITE_SCHEMA_VERSION,
        "voxel_planner_config_schema_version": VOXEL_PLANNER_CONFIG_SCHEMA_VERSION,
        "occupancy_evidence_schema_version": OCCUPANCY_EVIDENCE_SCHEMA_VERSION,
        "solvability_evidence_schema_version": SOLVABILITY_EVIDENCE_SCHEMA_VERSION,
        "success": course.result.accepted,
        "profile_id": course.result.profile_id,
        "difficulty": course.result.difficulty.value,
        "split": course.result.split.value,
        "base_seed": course.result.base_seed,
        "accepted_candidate_seed": course.result.accepted_candidate_seed,
        "attempt_index": course.result.attempt_index,
        "scene_digest": course.result.scene_digest,
        "occupancy_digest": course.result.occupancy_digest,
        "solvability_digest": course.result.solvability_digest,
        "reference_path_length_m": course.result.path_result.reference_path_length_m,
        "reference_path": course.result.path_result.reference_path,
        "voxel_path": course.result.path_result.voxel_path,
        "path_efficiency_ratio": course.result.path_result.path_efficiency_ratio,
        "vertical_excursion_m": course.result.path_result.vertical_excursion_m,
        "expanded_nodes": course.result.path_result.expanded_nodes,
        "direct_line_clear": course.result.path_result.direct_line_clear,
        "direct_line_voxel_indices": (
            course.result.path_result.direct_line_voxel_indices
        ),
        "direct_line_blocking_voxel": (
            course.result.path_result.direct_line_blocking_voxel
        ),
        "direct_line_blocking_obstacle": (
            course.result.path_result.direct_line_blocking_obstacle
        ),
        "constraint_errors": course.result.constraint_errors,
        "rejected_candidates": course.result.rejected_candidates,
        "limitations": {
            "built_in_blocks_geometry_included": False,
            "physical_collision_response_verified": False,
            "continuous_space_optimality_claimed": False,
        },
    }


def save_course_report(course: ValidatedCourse, path: str | Path) -> None:
    """Save deterministic course evidence as newline-terminated JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _json_value(course_report_dict(course)),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def baseline_from_course(course: ValidatedCourse) -> FeasibilityBaseline:
    """Extract tracked deterministic acceptance metrics."""
    path = course.result.path_result
    assert path.reference_path_length_m is not None
    assert path.path_efficiency_ratio is not None
    assert path.vertical_excursion_m is not None
    return FeasibilityBaseline(
        base_seed=course.result.base_seed,
        accepted_candidate_seed=course.result.accepted_candidate_seed,
        attempt_index=course.result.attempt_index,
        scene_digest=course.result.scene_digest,
        occupancy_digest=course.result.occupancy_digest,
        solvability_digest=course.result.solvability_digest,
        reference_path_length_m=path.reference_path_length_m,
        path_efficiency_ratio=path.path_efficiency_ratio,
        vertical_excursion_m=path.vertical_excursion_m,
        expanded_nodes=path.expanded_nodes,
        direct_line_clear=path.direct_line_clear,
    )


def _difficulty_errors(
    scene: ValidatedScene,
    path: VoxelPathResult,
    constraints: DifficultyConstraints,
) -> tuple[str, ...]:
    errors: list[str] = []
    obstacle_count = len(scene.config.static_obstacles)
    if (
        not constraints.obstacle_count_min
        <= obstacle_count
        <= (constraints.obstacle_count_max)
    ):
        errors.append(
            f"obstacle count {obstacle_count} is outside "
            f"[{constraints.obstacle_count_min}, {constraints.obstacle_count_max}]"
        )
    if path.direct_line_clear is not constraints.direct_line_clear:
        errors.append(
            f"direct_line_clear expected {constraints.direct_line_clear}, "
            f"got {path.direct_line_clear}"
        )
    if path.path_efficiency_ratio is not None and not (
        constraints.path_efficiency_min
        <= path.path_efficiency_ratio
        <= constraints.path_efficiency_max
    ):
        errors.append(
            f"path efficiency {path.path_efficiency_ratio:.6f} is outside "
            f"[{constraints.path_efficiency_min}, "
            f"{constraints.path_efficiency_max}]"
        )
    if path.vertical_excursion_m is not None:
        if path.vertical_excursion_m < constraints.vertical_excursion_min_m:
            errors.append("vertical excursion is below the profile minimum")
        if (
            constraints.vertical_excursion_max_m is not None
            and path.vertical_excursion_m > constraints.vertical_excursion_max_m
        ):
            errors.append("vertical excursion exceeds the profile maximum")
    obstacle_names = tuple(item.name for item in scene.config.static_obstacles)
    for prefix in constraints.required_structure_prefixes:
        if not any(name.startswith(prefix) for name in obstacle_names):
            errors.append(f"required structure prefix {prefix!r} is absent")

    dx = scene.goal_approach.x - scene.start_anchor.x
    dy = scene.goal_approach.y - scene.start_anchor.y
    goal_above = scene.start_anchor.z - scene.goal_approach.z
    if constraints.required_x_direction is XDirectionRequirement.POSITIVE and dx <= 0.0:
        errors.append("goal must have positive x displacement")
    if constraints.required_x_direction is XDirectionRequirement.NEGATIVE and dx >= 0.0:
        errors.append("goal must have negative x displacement")
    if constraints.require_lateral_direction and math.isclose(dy, 0.0, abs_tol=1e-6):
        errors.append("goal must include lateral displacement")
    if goal_above < constraints.minimum_goal_above_start_m - 1e-6:
        errors.append("goal approach is not sufficiently above the start anchor")
    return tuple(errors)


def _workspace_bounds(scene: ValidatedScene) -> ContinuousBounds3D:
    workspace = scene.config.workspace
    return ContinuousBounds3D(
        workspace.min_x,
        workspace.max_x,
        workspace.min_y,
        workspace.max_y,
        workspace.min_z,
        workspace.max_z,
        "workspace",
    )


def _obstacle_bounds(scene: ValidatedScene) -> list[ContinuousBounds3D]:
    result = []
    for obstacle in scene.config.static_obstacles:
        bounds = conservative_bounds(obstacle)
        result.append(
            ContinuousBounds3D(
                bounds.min_x,
                bounds.max_x,
                bounds.min_y,
                bounds.max_y,
                bounds.min_z,
                bounds.max_z,
                obstacle.name,
            )
        )
    return result


def _point(value: Vector3) -> tuple[float, float, float]:
    return value.x, value.y, value.z


def _solvability_digest(
    scene_digest: str,
    planner: VoxelGridConfig,
    grid: VoxelGrid,
    path: VoxelPathResult,
) -> str:
    payload = {
        "solvability_schema_version": SOLVABILITY_EVIDENCE_SCHEMA_VERSION,
        "occupancy_schema_version": OCCUPANCY_EVIDENCE_SCHEMA_VERSION,
        "planner_schema_version": VOXEL_PLANNER_CONFIG_SCHEMA_VERSION,
        "scene_digest": scene_digest,
        "planner": asdict(planner),
        "occupancy_digest": grid.occupancy_digest,
        "status": path.status.value,
        "start_index": path.start_index,
        "goal_index": path.goal_index,
        "voxel_path": path.voxel_path,
        "reference_path_length_m": path.reference_path_length_m,
        "path_efficiency_ratio": path.path_efficiency_ratio,
        "vertical_excursion_m": path.vertical_excursion_m,
    }
    encoded = json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _profile_from_mapping(raw_value: object) -> CourseProfile:
    raw = _mapping(raw_value, "profile")
    constraints_raw = _required_mapping(raw, "constraints")
    vertical_max = constraints_raw.get("vertical_excursion_max_m")
    constraints = DifficultyConstraints(
        obstacle_count_min=_required_int(constraints_raw, "obstacle_count_min"),
        obstacle_count_max=_required_int(constraints_raw, "obstacle_count_max"),
        direct_line_clear=_required_bool(constraints_raw, "direct_line_clear"),
        path_efficiency_min=_required_float(constraints_raw, "path_efficiency_min"),
        path_efficiency_max=_required_float(constraints_raw, "path_efficiency_max"),
        vertical_excursion_min_m=_required_float(
            constraints_raw, "vertical_excursion_min_m"
        ),
        vertical_excursion_max_m=(
            float(vertical_max) if vertical_max is not None else None
        ),
        required_structure_prefixes=tuple(
            str(value)
            for value in constraints_raw.get("required_structure_prefixes", [])
        ),
        required_x_direction=XDirectionRequirement(
            str(constraints_raw.get("required_x_direction", "any"))
        ),
        require_lateral_direction=bool(
            constraints_raw.get("require_lateral_direction", False)
        ),
        minimum_goal_above_start_m=float(
            constraints_raw.get("minimum_goal_above_start_m", 0.0)
        ),
    )
    baselines = tuple(
        _baseline_from_mapping(item) for item in raw.get("accepted_baselines", [])
    )
    profile = CourseProfile(
        profile_id=_required_str(raw, "profile_id"),
        difficulty=CourseDifficulty(_required_str(raw, "difficulty")),
        split=CourseSplit(_required_str(raw, "split")),
        scene_config=_required_str(raw, "scene_config"),
        base_seeds=tuple(_int_list(raw.get("base_seeds"), "base_seeds")),
        max_candidate_attempts=_required_int(raw, "max_candidate_attempts"),
        constraints=constraints,
        accepted_baselines=baselines,
    )
    _validate_profile(profile)
    return profile


def _baseline_from_mapping(raw_value: object) -> FeasibilityBaseline:
    raw = _mapping(raw_value, "accepted baseline")
    return FeasibilityBaseline(
        base_seed=_required_int(raw, "base_seed"),
        accepted_candidate_seed=_required_int(raw, "accepted_candidate_seed"),
        attempt_index=_required_int(raw, "attempt_index"),
        scene_digest=_required_digest(raw, "scene_digest"),
        occupancy_digest=_required_digest(raw, "occupancy_digest"),
        solvability_digest=_required_digest(raw, "solvability_digest"),
        reference_path_length_m=_required_float(raw, "reference_path_length_m"),
        path_efficiency_ratio=_required_float(raw, "path_efficiency_ratio"),
        vertical_excursion_m=_required_float(raw, "vertical_excursion_m"),
        expanded_nodes=_required_int(raw, "expanded_nodes"),
        direct_line_clear=_required_bool(raw, "direct_line_clear"),
    )


def _validate_profile(profile: CourseProfile) -> None:
    if not profile.profile_id or profile.profile_id != profile.profile_id.strip():
        raise ValueError("course profile ID must be non-empty and trimmed")
    if (
        Path(profile.scene_config).is_absolute()
        or ".." in Path(profile.scene_config).parts
    ):
        raise ValueError("scene_config must be a repository-relative path")
    if not profile.base_seeds or len(set(profile.base_seeds)) != len(
        profile.base_seeds
    ):
        raise ValueError("base_seeds must be non-empty and unique")
    if not 1 <= profile.max_candidate_attempts <= 32:
        raise ValueError("max_candidate_attempts must be between 1 and 32")
    constraints = profile.constraints
    if (
        constraints.obstacle_count_min < 0
        or constraints.obstacle_count_min > constraints.obstacle_count_max
    ):
        raise ValueError("invalid obstacle-count constraint")
    for value in (
        constraints.path_efficiency_min,
        constraints.path_efficiency_max,
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("path-efficiency constraints must be in [0, 1]")
    if constraints.path_efficiency_min > constraints.path_efficiency_max:
        raise ValueError("path-efficiency minimum exceeds maximum")
    if len(profile.accepted_baselines) not in {0, len(profile.base_seeds)}:
        raise ValueError("accepted baselines must be empty or cover every base seed")
    if {item.base_seed for item in profile.accepted_baselines} not in (
        set(),
        set(profile.base_seeds),
    ):
        raise ValueError("accepted baselines do not match declared base seeds")


def _validate_seed_partitions(profiles: tuple[CourseProfile, ...]) -> None:
    occupied: dict[int, str] = {}
    declared_base_seeds = sorted(
        base_seed for profile in profiles for base_seed in profile.base_seeds
    )
    if any(
        second - first < 100
        for first, second in zip(
            declared_base_seeds,
            declared_base_seeds[1:],
            strict=False,
        )
    ):
        raise ValueError("declared base seeds must be at least 100 values apart")
    for profile in profiles:
        for base_seed in profile.base_seeds:
            for candidate_seed in range(
                base_seed, base_seed + profile.max_candidate_attempts
            ):
                previous = occupied.get(candidate_seed)
                if previous is not None:
                    raise ValueError(
                        f"candidate seed {candidate_seed} overlaps profiles "
                        f"{previous!r} and {profile.profile_id!r}"
                    )
                occupied[candidate_seed] = profile.profile_id


def resolve_profile_scene_path(profile: CourseProfile, root: Path) -> Path:
    """Resolve one authoritative profile scene path within the repository."""
    root = root.resolve()
    path = (root / profile.scene_config).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("profile scene path leaves repository root") from exc
    return path


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in raw:
        raise ValueError(f"missing required field {key!r}")
    return _mapping(raw[key], key)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(raw: Mapping[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _required_bool(raw: Mapping[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _required_digest(raw: Mapping[str, Any], key: str) -> str:
    value = _required_str(raw, key)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _int_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{label} must be a list of integers")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value
