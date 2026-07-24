"""Deterministic voxel geometry and 3D A* tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from r1_uav_nav.planners.voxel_astar import (
    ContinuousBounds3D,
    VoxelConfigurationError,
    VoxelGridConfig,
    VoxelPathStatus,
    build_voxel_grid,
    find_voxel_astar_path,
    inflate_bounds,
    supercover_voxels,
    validate_segment,
    validate_voxel_path,
)


def _config(**changes: object) -> VoxelGridConfig:
    return replace(VoxelGridConfig(), **changes)


def _workspace(size: float = 4.0) -> ContinuousBounds3D:
    return ContinuousBounds3D(0.0, size, 0.0, size, 0.0, size)


def test_clearance_is_calculated_and_mismatch_is_rejected() -> None:
    config = VoxelGridConfig()
    assert config.uav_collision_radius_m == 0.35
    assert config.additional_safety_margin_m == 0.15
    assert config.calculated_total_clearance_m == pytest.approx(0.50)

    with pytest.raises(VoxelConfigurationError, match="total_clearance_m"):
        _config(total_clearance_m=0.49)


def test_inflation_is_axis_aligned_on_all_faces() -> None:
    inflated = inflate_bounds(
        ContinuousBounds3D(1.0, 2.0, 3.0, 4.0, -2.0, 0.0, "box"),
        0.5,
    )
    assert inflated == ContinuousBounds3D(0.5, 2.5, 2.5, 4.5, -2.5, 0.5, "box")


def test_half_open_mapping_and_workspace_maximum() -> None:
    grid = build_voxel_grid(_workspace(), (), _config())
    assert grid.point_to_index((0.0, 0.0, 0.0)) == (0, 0, 0)
    assert grid.point_to_index((0.249999, 0.249999, 0.249999)) == (0, 0, 0)
    assert grid.point_to_index((0.25, 0.25, 0.25)) == (1, 1, 1)
    assert grid.point_to_index((4.0, 4.0, 4.0)) == (15, 15, 15)


def test_non_divisible_workspace_and_voxel_limit_are_rejected() -> None:
    with pytest.raises(VoxelConfigurationError, match="divisible"):
        build_voxel_grid(
            ContinuousBounds3D(0.0, 4.1, 0.0, 4.0, 0.0, 4.0),
            (),
            _config(),
        )
    with pytest.raises(VoxelConfigurationError, match="exceeds"):
        build_voxel_grid(_workspace(), (), _config(max_voxels=100))
    with pytest.raises(VoxelConfigurationError, match="positive integer"):
        _config(max_expansions=1.5)


def test_empty_workspace_produces_a_post_validated_path() -> None:
    grid = build_voxel_grid(_workspace(), (), _config())
    result = find_voxel_astar_path(grid, (1.0, 1.0, 1.0), (3.0, 3.0, 3.0))

    assert result.status is VoxelPathStatus.SOLVABLE
    assert result.reference_path_length_m is not None
    assert result.path_efficiency_ratio is not None
    assert result.direct_line_clear
    assert validate_voxel_path(grid, result.start, result.goal, result.voxel_path) == (
        True,
        None,
    )


def test_complete_inflated_wall_produces_no_path() -> None:
    wall = ContinuousBounds3D(1.75, 2.25, 0.0, 4.0, 0.0, 4.0, "wall")
    grid = build_voxel_grid(_workspace(), (wall,), _config())
    result = find_voxel_astar_path(grid, (0.75, 2.0, 2.0), (3.25, 2.0, 2.0))

    assert result.status is VoxelPathStatus.NO_PATH
    assert not result.direct_line_clear
    assert result.direct_line_blocking_obstacle == "wall"


def test_short_obstacle_permits_route_above_it() -> None:
    obstacle = ContinuousBounds3D(1.75, 2.25, 0.5, 3.5, 0.0, 1.0, "short")
    grid = build_voxel_grid(_workspace(), (obstacle,), _config())
    result = find_voxel_astar_path(grid, (0.75, 2.0, 1.0), (3.25, 2.0, 1.0))

    assert result.solvable
    assert not result.direct_line_clear
    assert result.vertical_excursion_m is not None
    assert result.vertical_excursion_m > 0.5


def test_narrow_passage_closes_but_wide_passage_remains() -> None:
    narrow = (
        ContinuousBounds3D(1.5, 2.5, 0.0, 1.6, 0.0, 4.0, "lower"),
        ContinuousBounds3D(1.5, 2.5, 2.4, 4.0, 0.0, 4.0, "upper"),
    )
    narrow_grid = build_voxel_grid(_workspace(), narrow, _config())
    narrow_result = find_voxel_astar_path(narrow_grid, (1.0, 2.0, 2.0), (3.0, 2.0, 2.0))
    assert not narrow_result.solvable

    wide = (
        ContinuousBounds3D(1.5, 2.5, 0.0, 1.0, 0.0, 4.0, "lower"),
        ContinuousBounds3D(1.5, 2.5, 3.0, 4.0, 0.0, 4.0, "upper"),
    )
    wide_grid = build_voxel_grid(_workspace(), wide, _config())
    wide_result = find_voxel_astar_path(wide_grid, (1.0, 2.0, 2.0), (3.0, 2.0, 2.0))
    assert wide_result.solvable


def test_diagonal_corner_cutting_is_rejected() -> None:
    grid = build_voxel_grid(_workspace(2.0), (), _config())
    start = grid.point_to_index((0.75, 0.75, 0.75))
    goal = (start[0] + 1, start[1] + 1, start[2])
    occupied = frozenset(
        {
            (start[0] + 1, start[1], start[2]),
            (start[0], start[1] + 1, start[2]),
        }
    )
    blocked = replace(grid, occupied_indices=grid.occupied_indices | occupied)

    valid, reason = validate_voxel_path(
        blocked,
        blocked.voxel_center(start),
        blocked.voxel_center(goal),
        (start, goal),
    )
    assert not valid
    assert reason is not None


def test_start_goal_and_exact_endpoint_failures_are_typed() -> None:
    obstacle = ContinuousBounds3D(1.0, 1.1, 1.0, 1.1, 1.0, 1.1, "tiny")
    grid = build_voxel_grid(_workspace(), (obstacle,), _config())

    coincident = find_voxel_astar_path(grid, (2.0, 2.0, 2.0), (2.0, 2.0, 2.0))
    assert coincident.status is VoxelPathStatus.INVALID_ENDPOINT

    inside = find_voxel_astar_path(grid, (1.0, 1.0, 1.0), (3.0, 3.0, 3.0))
    assert inside.status is VoxelPathStatus.INVALID_ENDPOINT

    outside = find_voxel_astar_path(grid, (0.1, 2.0, 2.0), (3.0, 2.0, 2.0))
    assert outside.status is VoxelPathStatus.INVALID_ENDPOINT

    nonfinite = find_voxel_astar_path(
        grid,
        (float("nan"), 2.0, 2.0),
        (3.0, 2.0, 2.0),
    )
    assert nonfinite.status is VoxelPathStatus.INVALID_ENDPOINT
    assert nonfinite.direct_distance_m is None


def test_supercover_includes_face_and_corner_touching_voxels() -> None:
    grid = build_voxel_grid(_workspace(2.0), (), _config())
    touched = supercover_voxels(grid, (0.75, 0.75, 0.75), (1.25, 1.25, 0.75))

    assert (3, 2, 3) in touched
    assert (2, 3, 3) in touched
    assert (3, 3, 3) in touched


def test_endpoint_connector_is_checked_against_exact_geometry() -> None:
    grid = build_voxel_grid(_workspace(), (), _config())
    obstacle = ContinuousBounds3D(1.0, 1.1, 1.0, 1.1, 1.0, 1.1, "connector")
    modified = replace(grid, inflated_obstacles=(obstacle,))
    validation = validate_segment(modified, (0.9, 0.9, 0.9), (1.2, 1.2, 1.2))

    assert not validation.clear
    assert validation.blocking_obstacle == "connector"


def test_expansion_limit_is_a_typed_result() -> None:
    grid = build_voxel_grid(_workspace(), (), _config(max_expansions=1))
    result = find_voxel_astar_path(grid, (1.0, 1.0, 1.0), (3.0, 3.0, 3.0))
    assert result.status is VoxelPathStatus.EXPANSION_LIMIT_REACHED


def test_occupancy_and_path_are_deterministic() -> None:
    obstacle = ContinuousBounds3D(1.75, 2.25, 1.0, 3.0, 0.0, 1.0, "box")
    first_grid = build_voxel_grid(_workspace(), (obstacle,), _config())
    second_grid = build_voxel_grid(_workspace(), (obstacle,), _config())
    first = find_voxel_astar_path(first_grid, (1.0, 2.0, 1.0), (3.0, 2.0, 1.0))
    second = find_voxel_astar_path(second_grid, (1.0, 2.0, 1.0), (3.0, 2.0, 1.0))

    assert first_grid.occupancy_digest == second_grid.occupancy_digest
    assert first.voxel_path == second.voxel_path
    assert first.reference_path_length_m == second.reference_path_length_m
