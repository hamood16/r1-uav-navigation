"""Deterministic voxel occupancy and 3D A* planning."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import count, product
from typing import Iterable, Sequence

VOXEL_PLANNER_CONFIG_SCHEMA_VERSION = 1
OCCUPANCY_EVIDENCE_SCHEMA_VERSION = 1
SOLVABILITY_EVIDENCE_SCHEMA_VERSION = 1
VOXEL_GEOMETRY_TOLERANCE_M = 1e-6

VoxelIndex = tuple[int, int, int]
Point3D = tuple[float, float, float]


class VoxelConfigurationError(ValueError):
    """Raised when voxel geometry or planner configuration is invalid."""


class VoxelPathStatus(str, Enum):
    """Typed outcomes from one bounded voxel search."""

    SOLVABLE = "solvable"
    NO_PATH = "no_path"
    START_OCCUPIED = "start_occupied"
    GOAL_OCCUPIED = "goal_occupied"
    INVALID_ENDPOINT = "invalid_endpoint"
    EXPANSION_LIMIT_REACHED = "expansion_limit_reached"
    PATH_VALIDATION_FAILED = "path_validation_failed"


@dataclass(frozen=True)
class ContinuousBounds3D:
    """Continuous axis-aligned bounds in local NED coordinates."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    identifier: str | None = None

    def values(self) -> tuple[float, ...]:
        return (
            self.min_x,
            self.max_x,
            self.min_y,
            self.max_y,
            self.min_z,
            self.max_z,
        )


@dataclass(frozen=True)
class VoxelGridConfig:
    """Versioned planner configuration with one calculated clearance."""

    schema_version: int = VOXEL_PLANNER_CONFIG_SCHEMA_VERSION
    resolution_m: float = 0.25
    uav_collision_radius_m: float = 0.35
    additional_safety_margin_m: float = 0.15
    total_clearance_m: float = 0.50
    connectivity: int = 26
    max_voxels: int = 1_000_000
    max_expansions: int = 250_000

    def __post_init__(self) -> None:
        if self.schema_version != VOXEL_PLANNER_CONFIG_SCHEMA_VERSION:
            raise VoxelConfigurationError(
                "unsupported voxel planner configuration schema"
            )
        for field_name in (
            "resolution_m",
            "uav_collision_radius_m",
            "additional_safety_margin_m",
            "total_clearance_m",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise VoxelConfigurationError(
                    f"{field_name} must be finite and positive"
                )
        if not math.isclose(
            self.total_clearance_m,
            self.calculated_total_clearance_m,
            abs_tol=VOXEL_GEOMETRY_TOLERANCE_M,
            rel_tol=0.0,
        ):
            raise VoxelConfigurationError(
                "total_clearance_m must equal uav_collision_radius_m plus "
                "additional_safety_margin_m"
            )
        if self.connectivity != 26:
            raise VoxelConfigurationError("M13.3 requires 26-connectivity")
        if (
            isinstance(self.max_voxels, bool)
            or not isinstance(self.max_voxels, int)
            or self.max_voxels <= 0
        ):
            raise VoxelConfigurationError("max_voxels must be a positive integer")
        if (
            isinstance(self.max_expansions, bool)
            or not isinstance(self.max_expansions, int)
            or self.max_expansions <= 0
        ):
            raise VoxelConfigurationError("max_expansions must be a positive integer")

    @property
    def calculated_total_clearance_m(self) -> float:
        """Return the only clearance used by geometry operations."""
        return self.uav_collision_radius_m + self.additional_safety_margin_m


@dataclass(frozen=True)
class SegmentValidation:
    """Deterministic continuous and voxel evidence for one segment."""

    clear: bool
    voxel_indices: tuple[VoxelIndex, ...]
    blocking_voxel: VoxelIndex | None = None
    blocking_obstacle: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class VoxelGrid:
    """Immutable occupancy evidence for one local course."""

    config: VoxelGridConfig
    workspace: ContinuousBounds3D
    navigable_workspace: ContinuousBounds3D
    shape: VoxelIndex
    inflated_obstacles: tuple[ContinuousBounds3D, ...]
    occupied_indices: frozenset[VoxelIndex]
    occupancy_digest: str

    @property
    def voxel_count(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def contains_index(self, index: VoxelIndex) -> bool:
        return all(
            0 <= value < size for value, size in zip(index, self.shape, strict=True)
        )

    def is_occupied(self, index: VoxelIndex) -> bool:
        return index in self.occupied_indices

    def point_to_index(self, point: Point3D) -> VoxelIndex:
        """Map a point through half-open cells with an inclusive workspace maximum."""
        _require_finite_point(point, "point")
        result = []
        for coordinate, lower, upper, size in zip(
            point,
            (self.workspace.min_x, self.workspace.min_y, self.workspace.min_z),
            (self.workspace.max_x, self.workspace.max_y, self.workspace.max_z),
            self.shape,
            strict=True,
        ):
            if coordinate < lower - VOXEL_GEOMETRY_TOLERANCE_M or coordinate > (
                upper + VOXEL_GEOMETRY_TOLERANCE_M
            ):
                raise VoxelConfigurationError("point is outside the voxel workspace")
            if math.isclose(
                coordinate,
                upper,
                abs_tol=VOXEL_GEOMETRY_TOLERANCE_M,
                rel_tol=0.0,
            ):
                result.append(size - 1)
                continue
            adjusted = max(coordinate, lower)
            index = math.floor((adjusted - lower) / self.config.resolution_m)
            if not 0 <= index < size:
                raise VoxelConfigurationError("point did not map to a valid voxel")
            result.append(index)
        return tuple(result)  # type: ignore[return-value]

    def voxel_center(self, index: VoxelIndex) -> Point3D:
        if not self.contains_index(index):
            raise VoxelConfigurationError("voxel index is outside the grid")
        resolution = self.config.resolution_m
        return (
            self.workspace.min_x + (index[0] + 0.5) * resolution,
            self.workspace.min_y + (index[1] + 0.5) * resolution,
            self.workspace.min_z + (index[2] + 0.5) * resolution,
        )

    def voxel_bounds(self, index: VoxelIndex) -> ContinuousBounds3D:
        if not self.contains_index(index):
            raise VoxelConfigurationError("voxel index is outside the grid")
        resolution = self.config.resolution_m
        min_x = self.workspace.min_x + index[0] * resolution
        min_y = self.workspace.min_y + index[1] * resolution
        min_z = self.workspace.min_z + index[2] * resolution
        return ContinuousBounds3D(
            min_x,
            min_x + resolution,
            min_y,
            min_y + resolution,
            min_z,
            min_z + resolution,
        )


@dataclass(frozen=True)
class VoxelPathResult:
    """Complete path, direct-line, and bounded-search evidence."""

    schema_version: int
    status: VoxelPathStatus
    start: Point3D
    goal: Point3D
    start_index: VoxelIndex | None
    goal_index: VoxelIndex | None
    voxel_path: tuple[VoxelIndex, ...]
    voxel_center_path: tuple[Point3D, ...]
    reference_path: tuple[Point3D, ...]
    reference_path_length_m: float | None
    direct_distance_m: float | None
    path_efficiency_ratio: float | None
    vertical_excursion_m: float | None
    expanded_nodes: int
    direct_line_clear: bool
    direct_line_voxel_indices: tuple[VoxelIndex, ...]
    direct_line_blocking_voxel: VoxelIndex | None
    direct_line_blocking_obstacle: str | None
    direct_line_rejection_reason: str | None
    occupancy_digest: str
    failure_reason: str | None = None

    @property
    def solvable(self) -> bool:
        return self.status is VoxelPathStatus.SOLVABLE


def inflate_bounds(
    bounds: ContinuousBounds3D, clearance_m: float
) -> ContinuousBounds3D:
    """Apply conservative L-infinity inflation to an existing AABB."""
    if not math.isfinite(clearance_m) or clearance_m < 0.0:
        raise VoxelConfigurationError("clearance must be finite and nonnegative")
    return ContinuousBounds3D(
        bounds.min_x - clearance_m,
        bounds.max_x + clearance_m,
        bounds.min_y - clearance_m,
        bounds.max_y + clearance_m,
        bounds.min_z - clearance_m,
        bounds.max_z + clearance_m,
        identifier=bounds.identifier,
    )


def erode_bounds(bounds: ContinuousBounds3D, clearance_m: float) -> ContinuousBounds3D:
    """Inset a workspace by the calculated vehicle clearance."""
    eroded = ContinuousBounds3D(
        bounds.min_x + clearance_m,
        bounds.max_x - clearance_m,
        bounds.min_y + clearance_m,
        bounds.max_y - clearance_m,
        bounds.min_z + clearance_m,
        bounds.max_z - clearance_m,
        identifier=bounds.identifier,
    )
    _validate_bounds(eroded, "eroded workspace")
    return eroded


def build_voxel_grid(
    workspace: ContinuousBounds3D,
    obstacles: Sequence[ContinuousBounds3D],
    config: VoxelGridConfig,
) -> VoxelGrid:
    """Build conservative occupancy from static continuous geometry."""
    _validate_bounds(workspace, "workspace")
    for obstacle in obstacles:
        _validate_bounds(obstacle, f"obstacle {obstacle.identifier!r}")
    shape = _grid_shape(workspace, config)
    voxel_count = shape[0] * shape[1] * shape[2]
    if voxel_count > config.max_voxels:
        raise VoxelConfigurationError(
            f"voxel count {voxel_count} exceeds limit {config.max_voxels}"
        )

    clearance = config.calculated_total_clearance_m
    navigable = erode_bounds(workspace, clearance)
    inflated = tuple(inflate_bounds(item, clearance) for item in obstacles)
    occupied: set[VoxelIndex] = set()

    provisional = VoxelGrid(
        config=config,
        workspace=workspace,
        navigable_workspace=navigable,
        shape=shape,
        inflated_obstacles=inflated,
        occupied_indices=frozenset(),
        occupancy_digest="",
    )
    for index in product(*(range(size) for size in shape)):
        typed_index: VoxelIndex = index
        cell = provisional.voxel_bounds(typed_index)
        if not _bounds_contains_bounds(navigable, cell):
            occupied.add(typed_index)

    for obstacle in inflated:
        for index in _candidate_indices_for_bounds(provisional, obstacle):
            if _closed_bounds_intersect(
                provisional.voxel_bounds(index),
                obstacle,
            ):
                occupied.add(index)

    digest = _occupancy_digest(config, workspace, shape, occupied)
    return VoxelGrid(
        config=config,
        workspace=workspace,
        navigable_workspace=navigable,
        shape=shape,
        inflated_obstacles=inflated,
        occupied_indices=frozenset(occupied),
        occupancy_digest=digest,
    )


def supercover_voxels(
    grid: VoxelGrid, start: Point3D, end: Point3D
) -> tuple[VoxelIndex, ...]:
    """Return every voxel whose closed volume is touched by a segment."""
    _require_finite_point(start, "segment start")
    _require_finite_point(end, "segment end")
    start_index = grid.point_to_index(start)
    end_index = grid.point_to_index(end)
    ranges = []
    for start_value, end_value, size in zip(
        start_index, end_index, grid.shape, strict=True
    ):
        lower = max(0, min(start_value, end_value) - 1)
        upper = min(size - 1, max(start_value, end_value) + 1)
        ranges.append(range(lower, upper + 1))

    touched: list[tuple[float, VoxelIndex]] = []
    direction = tuple(
        end_value - start_value
        for start_value, end_value in zip(start, end, strict=True)
    )
    magnitude_squared = sum(value * value for value in direction)
    for raw_index in product(*ranges):
        index: VoxelIndex = raw_index
        if not _segment_intersects_bounds(start, end, grid.voxel_bounds(index)):
            continue
        center = grid.voxel_center(index)
        parameter = (
            0.0
            if magnitude_squared <= VOXEL_GEOMETRY_TOLERANCE_M**2
            else sum(
                (center_value - start_value) * direction_value
                for center_value, start_value, direction_value in zip(
                    center, start, direction, strict=True
                )
            )
            / magnitude_squared
        )
        touched.append((parameter, index))
    touched.sort(key=lambda item: (item[0], item[1]))
    return tuple(index for _, index in touched)


def validate_segment(
    grid: VoxelGrid, start: Point3D, end: Point3D
) -> SegmentValidation:
    """Validate one exact segment against workspace, obstacles, and occupancy."""
    try:
        _require_point_in_bounds(start, grid.navigable_workspace, "segment start")
        _require_point_in_bounds(end, grid.navigable_workspace, "segment end")
        traversed = supercover_voxels(grid, start, end)
    except VoxelConfigurationError as exc:
        return SegmentValidation(False, (), reason=str(exc))

    for obstacle in grid.inflated_obstacles:
        if _segment_intersects_bounds(start, end, obstacle):
            return SegmentValidation(
                False,
                traversed,
                blocking_obstacle=obstacle.identifier,
                reason="segment intersects inflated obstacle geometry",
            )
    for index in traversed:
        if grid.is_occupied(index):
            return SegmentValidation(
                False,
                traversed,
                blocking_voxel=index,
                reason="segment traverses an occupied voxel",
            )
    return SegmentValidation(True, traversed)


def find_voxel_astar_path(
    grid: VoxelGrid,
    start: Point3D,
    goal: Point3D,
) -> VoxelPathResult:
    """Find one deterministic, post-validated 26-connected voxel path."""
    try:
        _require_finite_point(start, "start anchor")
        _require_finite_point(goal, "goal approach")
    except VoxelConfigurationError as exc:
        return _empty_result(
            grid,
            start,
            goal,
            VoxelPathStatus.INVALID_ENDPOINT,
            None,
            str(exc),
        )
    direct_distance = _distance(start, goal)
    invalid_reason = _endpoint_validation_error(grid, start, goal)
    if invalid_reason is not None:
        return _empty_result(
            grid,
            start,
            goal,
            VoxelPathStatus.INVALID_ENDPOINT,
            direct_distance,
            invalid_reason,
        )

    start_index = grid.point_to_index(start)
    goal_index = grid.point_to_index(goal)
    direct = validate_segment(grid, start, goal)
    if grid.is_occupied(start_index):
        return _empty_result(
            grid,
            start,
            goal,
            VoxelPathStatus.START_OCCUPIED,
            direct_distance,
            "start anchor maps to an occupied voxel",
            start_index=start_index,
            goal_index=goal_index,
            direct=direct,
        )
    if grid.is_occupied(goal_index):
        return _empty_result(
            grid,
            start,
            goal,
            VoxelPathStatus.GOAL_OCCUPIED,
            direct_distance,
            "goal approach maps to an occupied voxel",
            start_index=start_index,
            goal_index=goal_index,
            direct=direct,
        )

    sequence = count()
    open_set: list[tuple[float, float, int, int, int, int, int]] = []
    start_h = _index_distance(start_index, goal_index, grid.config.resolution_m)
    heapq.heappush(
        open_set,
        (
            start_h,
            start_h,
            abs(start_index[2] - goal_index[2]),
            *start_index,
            next(sequence),
        ),
    )
    came_from: dict[VoxelIndex, VoxelIndex] = {}
    g_score: dict[VoxelIndex, float] = {start_index: 0.0}
    closed: set[VoxelIndex] = set()
    expanded = 0

    while open_set:
        _, _, _, ix, iy, iz, _ = heapq.heappop(open_set)
        current = (ix, iy, iz)
        if current in closed:
            continue
        if expanded >= grid.config.max_expansions:
            return _empty_result(
                grid,
                start,
                goal,
                VoxelPathStatus.EXPANSION_LIMIT_REACHED,
                direct_distance,
                "A* expansion limit reached",
                start_index=start_index,
                goal_index=goal_index,
                direct=direct,
                expanded_nodes=expanded,
            )
        expanded += 1
        if current == goal_index:
            index_path = _reconstruct_path(came_from, current)
            return _build_success_result(
                grid,
                start,
                goal,
                index_path,
                expanded,
                direct,
            )
        closed.add(current)

        for delta in _NEIGHBOUR_DELTAS:
            neighbour = tuple(
                value + offset for value, offset in zip(current, delta, strict=True)
            )
            if (
                not grid.contains_index(neighbour)
                or grid.is_occupied(neighbour)
                or neighbour in closed
                or not _move_intermediates_are_clear(grid, current, delta)
            ):
                continue
            step_cost = grid.config.resolution_m * math.sqrt(
                sum(value * value for value in delta)
            )
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbour, math.inf):
                continue
            came_from[neighbour] = current
            g_score[neighbour] = tentative
            heuristic = _index_distance(neighbour, goal_index, grid.config.resolution_m)
            heapq.heappush(
                open_set,
                (
                    tentative + heuristic,
                    heuristic,
                    abs(neighbour[2] - goal_index[2]),
                    *neighbour,
                    next(sequence),
                ),
            )

    return _empty_result(
        grid,
        start,
        goal,
        VoxelPathStatus.NO_PATH,
        direct_distance,
        "no voxel path exists",
        start_index=start_index,
        goal_index=goal_index,
        direct=direct,
        expanded_nodes=expanded,
    )


def validate_voxel_path(
    grid: VoxelGrid,
    start: Point3D,
    goal: Point3D,
    voxel_path: Sequence[VoxelIndex],
) -> tuple[bool, str | None]:
    """Post-validate endpoints, moves, and every exact path segment."""
    if not voxel_path:
        return False, "voxel path is empty"
    try:
        start_index = grid.point_to_index(start)
        goal_index = grid.point_to_index(goal)
    except VoxelConfigurationError as exc:
        return False, str(exc)
    if voxel_path[0] != start_index:
        return False, "path does not begin in the exact start voxel"
    if voxel_path[-1] != goal_index:
        return False, "path does not end in the exact goal voxel"
    if any(
        not grid.contains_index(index) or grid.is_occupied(index)
        for index in voxel_path
    ):
        return False, "path contains an invalid or occupied voxel"

    centers = tuple(grid.voxel_center(index) for index in voxel_path)
    segments = (
        (start, centers[0]),
        *zip(centers, centers[1:], strict=False),
        (centers[-1], goal),
    )
    for first, second in segments:
        validation = validate_segment(grid, first, second)
        if not validation.clear:
            return False, validation.reason
    for first, second in zip(voxel_path, voxel_path[1:], strict=False):
        delta = tuple(
            next_value - value for value, next_value in zip(first, second, strict=True)
        )
        if delta not in _NEIGHBOUR_DELTA_SET:
            return False, "path contains a non-neighbour move"
        if not _move_intermediates_are_clear(grid, first, delta):
            return False, "path cuts through an occupied corner"
    return True, None


def _build_success_result(
    grid: VoxelGrid,
    start: Point3D,
    goal: Point3D,
    voxel_path: tuple[VoxelIndex, ...],
    expanded_nodes: int,
    direct: SegmentValidation,
) -> VoxelPathResult:
    valid, reason = validate_voxel_path(grid, start, goal, voxel_path)
    if not valid:
        return _empty_result(
            grid,
            start,
            goal,
            VoxelPathStatus.PATH_VALIDATION_FAILED,
            _distance(start, goal),
            reason or "path post-validation failed",
            start_index=voxel_path[0],
            goal_index=voxel_path[-1],
            direct=direct,
            expanded_nodes=expanded_nodes,
        )
    centers = tuple(grid.voxel_center(index) for index in voxel_path)
    reference = _deduplicate_points((start, *centers, goal))
    length = sum(
        _distance(first, second)
        for first, second in zip(reference, reference[1:], strict=False)
    )
    direct_distance = _distance(start, goal)
    efficiency = direct_distance / length if length > 0.0 else None
    z_values = [point[2] for point in reference]
    return VoxelPathResult(
        schema_version=SOLVABILITY_EVIDENCE_SCHEMA_VERSION,
        status=VoxelPathStatus.SOLVABLE,
        start=start,
        goal=goal,
        start_index=voxel_path[0],
        goal_index=voxel_path[-1],
        voxel_path=voxel_path,
        voxel_center_path=centers,
        reference_path=reference,
        reference_path_length_m=length,
        direct_distance_m=direct_distance,
        path_efficiency_ratio=efficiency,
        vertical_excursion_m=max(z_values) - min(z_values),
        expanded_nodes=expanded_nodes,
        direct_line_clear=direct.clear,
        direct_line_voxel_indices=direct.voxel_indices,
        direct_line_blocking_voxel=direct.blocking_voxel,
        direct_line_blocking_obstacle=direct.blocking_obstacle,
        direct_line_rejection_reason=direct.reason,
        occupancy_digest=grid.occupancy_digest,
    )


def _empty_result(
    grid: VoxelGrid,
    start: Point3D,
    goal: Point3D,
    status: VoxelPathStatus,
    direct_distance: float | None,
    reason: str,
    *,
    start_index: VoxelIndex | None = None,
    goal_index: VoxelIndex | None = None,
    direct: SegmentValidation | None = None,
    expanded_nodes: int = 0,
) -> VoxelPathResult:
    direct = direct or SegmentValidation(False, (), reason=reason)
    return VoxelPathResult(
        schema_version=SOLVABILITY_EVIDENCE_SCHEMA_VERSION,
        status=status,
        start=start,
        goal=goal,
        start_index=start_index,
        goal_index=goal_index,
        voxel_path=(),
        voxel_center_path=(),
        reference_path=(),
        reference_path_length_m=None,
        direct_distance_m=direct_distance,
        path_efficiency_ratio=None,
        vertical_excursion_m=None,
        expanded_nodes=expanded_nodes,
        direct_line_clear=direct.clear,
        direct_line_voxel_indices=direct.voxel_indices,
        direct_line_blocking_voxel=direct.blocking_voxel,
        direct_line_blocking_obstacle=direct.blocking_obstacle,
        direct_line_rejection_reason=direct.reason,
        occupancy_digest=grid.occupancy_digest,
        failure_reason=reason,
    )


def _endpoint_validation_error(
    grid: VoxelGrid, start: Point3D, goal: Point3D
) -> str | None:
    try:
        _require_finite_point(start, "start anchor")
        _require_finite_point(goal, "goal approach")
        if _distance(start, goal) <= VOXEL_GEOMETRY_TOLERANCE_M:
            return "start anchor and goal approach must be distinct"
        _require_point_in_bounds(start, grid.navigable_workspace, "start anchor")
        _require_point_in_bounds(goal, grid.navigable_workspace, "goal approach")
        for label, point in (("start anchor", start), ("goal approach", goal)):
            for obstacle in grid.inflated_obstacles:
                if _point_in_closed_bounds(point, obstacle):
                    return (
                        f"{label} lies inside inflated obstacle "
                        f"{obstacle.identifier!r}"
                    )
        grid.point_to_index(start)
        grid.point_to_index(goal)
    except VoxelConfigurationError as exc:
        return str(exc)
    return None


def _grid_shape(workspace: ContinuousBounds3D, config: VoxelGridConfig) -> VoxelIndex:
    values = []
    for lower, upper in (
        (workspace.min_x, workspace.max_x),
        (workspace.min_y, workspace.max_y),
        (workspace.min_z, workspace.max_z),
    ):
        span = upper - lower
        cells = round(span / config.resolution_m)
        if cells <= 0 or not math.isclose(
            span,
            cells * config.resolution_m,
            abs_tol=VOXEL_GEOMETRY_TOLERANCE_M,
            rel_tol=0.0,
        ):
            raise VoxelConfigurationError(
                "workspace spans must be divisible by voxel resolution"
            )
        values.append(cells)
    return tuple(values)  # type: ignore[return-value]


def _candidate_indices_for_bounds(
    grid: VoxelGrid, bounds: ContinuousBounds3D
) -> Iterable[VoxelIndex]:
    ranges = []
    for lower, upper, workspace_min, size in zip(
        (bounds.min_x, bounds.min_y, bounds.min_z),
        (bounds.max_x, bounds.max_y, bounds.max_z),
        (grid.workspace.min_x, grid.workspace.min_y, grid.workspace.min_z),
        grid.shape,
        strict=True,
    ):
        first = math.floor((lower - workspace_min) / grid.config.resolution_m) - 1
        last = math.floor((upper - workspace_min) / grid.config.resolution_m) + 1
        ranges.append(range(max(0, first), min(size - 1, last) + 1))
    return (index for index in product(*ranges))


def _move_intermediates_are_clear(
    grid: VoxelGrid, current: VoxelIndex, delta: VoxelIndex
) -> bool:
    changed_axes = [axis for axis, value in enumerate(delta) if value]
    if len(changed_axes) <= 1:
        return True
    for mask in range(1, (1 << len(changed_axes)) - 1):
        offset = [0, 0, 0]
        for bit, axis in enumerate(changed_axes):
            if mask & (1 << bit):
                offset[axis] = delta[axis]
        intermediate = tuple(
            value + change for value, change in zip(current, offset, strict=True)
        )
        if not grid.contains_index(intermediate) or grid.is_occupied(intermediate):
            return False
    return True


def _segment_intersects_bounds(
    start: Point3D, end: Point3D, bounds: ContinuousBounds3D
) -> bool:
    minimums = (bounds.min_x, bounds.min_y, bounds.min_z)
    maximums = (bounds.max_x, bounds.max_y, bounds.max_z)
    t_min = 0.0
    t_max = 1.0
    for start_value, end_value, lower, upper in zip(
        start, end, minimums, maximums, strict=True
    ):
        direction = end_value - start_value
        lower -= VOXEL_GEOMETRY_TOLERANCE_M
        upper += VOXEL_GEOMETRY_TOLERANCE_M
        if abs(direction) <= VOXEL_GEOMETRY_TOLERANCE_M:
            if start_value < lower or start_value > upper:
                return False
            continue
        first = (lower - start_value) / direction
        second = (upper - start_value) / direction
        entry, exit_ = sorted((first, second))
        t_min = max(t_min, entry)
        t_max = min(t_max, exit_)
        if t_min > t_max:
            return False
    return True


def _closed_bounds_intersect(
    first: ContinuousBounds3D, second: ContinuousBounds3D
) -> bool:
    tolerance = VOXEL_GEOMETRY_TOLERANCE_M
    return not (
        first.max_x < second.min_x - tolerance
        or second.max_x < first.min_x - tolerance
        or first.max_y < second.min_y - tolerance
        or second.max_y < first.min_y - tolerance
        or first.max_z < second.min_z - tolerance
        or second.max_z < first.min_z - tolerance
    )


def _bounds_contains_bounds(
    outer: ContinuousBounds3D, inner: ContinuousBounds3D
) -> bool:
    tolerance = VOXEL_GEOMETRY_TOLERANCE_M
    return (
        inner.min_x >= outer.min_x - tolerance
        and inner.max_x <= outer.max_x + tolerance
        and inner.min_y >= outer.min_y - tolerance
        and inner.max_y <= outer.max_y + tolerance
        and inner.min_z >= outer.min_z - tolerance
        and inner.max_z <= outer.max_z + tolerance
    )


def _require_point_in_bounds(
    point: Point3D, bounds: ContinuousBounds3D, label: str
) -> None:
    tolerance = VOXEL_GEOMETRY_TOLERANCE_M
    if not (
        bounds.min_x - tolerance <= point[0] <= bounds.max_x + tolerance
        and bounds.min_y - tolerance <= point[1] <= bounds.max_y + tolerance
        and bounds.min_z - tolerance <= point[2] <= bounds.max_z + tolerance
    ):
        raise VoxelConfigurationError(f"{label} is outside the eroded workspace")


def _point_in_closed_bounds(point: Point3D, bounds: ContinuousBounds3D) -> bool:
    tolerance = VOXEL_GEOMETRY_TOLERANCE_M
    return (
        bounds.min_x - tolerance <= point[0] <= bounds.max_x + tolerance
        and bounds.min_y - tolerance <= point[1] <= bounds.max_y + tolerance
        and bounds.min_z - tolerance <= point[2] <= bounds.max_z + tolerance
    )


def _validate_bounds(bounds: ContinuousBounds3D, label: str) -> None:
    if any(not math.isfinite(value) for value in bounds.values()):
        raise VoxelConfigurationError(f"{label} values must be finite")
    if (
        bounds.min_x >= bounds.max_x
        or bounds.min_y >= bounds.max_y
        or bounds.min_z >= bounds.max_z
    ):
        raise VoxelConfigurationError(f"{label} minima must be less than maxima")


def _require_finite_point(point: Point3D, label: str) -> None:
    if len(point) != 3 or any(not math.isfinite(value) for value in point):
        raise VoxelConfigurationError(f"{label} must contain three finite values")


def _occupancy_digest(
    config: VoxelGridConfig,
    workspace: ContinuousBounds3D,
    shape: VoxelIndex,
    occupied: Iterable[VoxelIndex],
) -> str:
    payload = {
        "occupancy_schema_version": OCCUPANCY_EVIDENCE_SCHEMA_VERSION,
        "planner_schema_version": config.schema_version,
        "config": asdict(config),
        "workspace": asdict(workspace),
        "shape": shape,
        "occupied_indices": sorted(occupied),
    }
    encoded = json.dumps(
        _normalize_json(payload), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_json(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize_json(item) for item in value]
    return value


def _reconstruct_path(
    came_from: dict[VoxelIndex, VoxelIndex], current: VoxelIndex
) -> tuple[VoxelIndex, ...]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return tuple(path)


def _deduplicate_points(points: Sequence[Point3D]) -> tuple[Point3D, ...]:
    result: list[Point3D] = []
    for point in points:
        if not result or _distance(result[-1], point) > VOXEL_GEOMETRY_TOLERANCE_M:
            result.append(point)
    return tuple(result)


def _distance(first: Point3D, second: Point3D) -> float:
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(first, second, strict=True))
    )


def _index_distance(first: VoxelIndex, second: VoxelIndex, resolution: float) -> float:
    return resolution * math.sqrt(
        sum((left - right) ** 2 for left, right in zip(first, second, strict=True))
    )


def _neighbour_sort_key(delta: VoxelIndex) -> tuple[int, int, int, int, int]:
    squared = sum(value * value for value in delta)
    return (squared, abs(delta[2]), delta[2], delta[1], delta[0])


_NEIGHBOUR_DELTAS: tuple[VoxelIndex, ...] = tuple(
    sorted(
        (delta for delta in product((-1, 0, 1), repeat=3) if delta != (0, 0, 0)),
        key=_neighbour_sort_key,
    )
)
_NEIGHBOUR_DELTA_SET = frozenset(_NEIGHBOUR_DELTAS)
