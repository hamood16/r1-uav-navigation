from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

from r1_uav_nav.planners import find_astar_path

Position = tuple[int, int]


def test_astar_finds_path_in_empty_grid() -> None:
    path = find_astar_path(
        start=(0, 0),
        goal=(2, 0),
        obstacles=set(),
        grid_size=3,
    )

    assert path == [(0, 0), (1, 0), (2, 0)]


def test_astar_returns_start_only_path_when_start_equals_goal() -> None:
    path = find_astar_path(
        start=(1, 1),
        goal=(1, 1),
        obstacles=set(),
        grid_size=3,
    )

    assert path == [(1, 1)]


def test_astar_avoids_obstacles() -> None:
    obstacles = {(1, 0)}

    path = find_astar_path(
        start=(0, 0),
        goal=(2, 0),
        obstacles=obstacles,
        grid_size=3,
    )

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 0)
    assert all(position not in obstacles for position in path)
    _assert_path_moves_one_cell_at_a_time(path)


def test_astar_returns_none_when_goal_is_unreachable() -> None:
    path = find_astar_path(
        start=(0, 0),
        goal=(2, 2),
        obstacles={(1, 0), (0, 1)},
        grid_size=3,
    )

    assert path is None


def test_astar_path_starts_at_start_and_ends_at_goal() -> None:
    start = (0, 0)
    goal = (4, 4)

    path = find_astar_path(
        start=start,
        goal=goal,
        obstacles={(1, 1), (2, 1), (3, 1)},
        grid_size=5,
    )

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal


def test_astar_path_never_enters_obstacle_cells() -> None:
    obstacles = {(1, 0), (1, 1), (1, 2)}

    path = find_astar_path(
        start=(0, 0),
        goal=(2, 0),
        obstacles=obstacles,
        grid_size=4,
    )

    assert path is not None
    assert set(path).isdisjoint(obstacles)


def test_astar_path_only_moves_one_grid_cell_at_a_time() -> None:
    path = find_astar_path(
        start=(0, 0),
        goal=(3, 3),
        obstacles={(1, 1), (2, 1)},
        grid_size=4,
    )

    assert path is not None
    _assert_path_moves_one_cell_at_a_time(path)


def test_astar_rejects_invalid_grid_size() -> None:
    with pytest.raises(ValueError, match="grid_size must be at least 2"):
        find_astar_path(
            start=(0, 0),
            goal=(1, 1),
            obstacles=set(),
            grid_size=1,
        )


@pytest.mark.parametrize("start", [(-1, 0), (0, -1), (3, 0), (0, 3)])
def test_astar_rejects_invalid_start(start: Position) -> None:
    with pytest.raises(ValueError, match="start position"):
        find_astar_path(
            start=start,
            goal=(1, 1),
            obstacles=set(),
            grid_size=3,
        )


@pytest.mark.parametrize("goal", [(-1, 0), (0, -1), (3, 0), (0, 3)])
def test_astar_rejects_invalid_goal(goal: Position) -> None:
    with pytest.raises(ValueError, match="goal position"):
        find_astar_path(
            start=(0, 0),
            goal=goal,
            obstacles=set(),
            grid_size=3,
        )


def test_astar_rejects_start_inside_obstacle() -> None:
    with pytest.raises(ValueError, match="start position cannot be inside"):
        find_astar_path(
            start=(0, 0),
            goal=(2, 2),
            obstacles={(0, 0)},
            grid_size=3,
        )


def test_astar_rejects_goal_inside_obstacle() -> None:
    with pytest.raises(ValueError, match="goal position cannot be inside"):
        find_astar_path(
            start=(0, 0),
            goal=(2, 2),
            obstacles={(2, 2)},
            grid_size=3,
        )


def test_astar_rejects_out_of_grid_obstacle() -> None:
    with pytest.raises(ValueError, match="obstacles cannot be outside"):
        find_astar_path(
            start=(0, 0),
            goal=(2, 2),
            obstacles={(3, 3)},
            grid_size=3,
        )


def test_select_shortest_successful_trajectory() -> None:
    module = _load_astar_evaluator_module()
    trajectories = _trajectory_examples()

    selected = module.select_shortest_successful_trajectory(trajectories)

    assert selected["name"] == "short"


def test_select_mean_length_successful_trajectory() -> None:
    module = _load_astar_evaluator_module()
    trajectories = _trajectory_examples()

    selected = module.select_mean_length_successful_trajectory(trajectories)

    assert selected["name"] == "medium"


def test_select_longest_successful_trajectory() -> None:
    module = _load_astar_evaluator_module()
    trajectories = _trajectory_examples()

    selected = module.select_longest_successful_trajectory(trajectories)

    assert selected["name"] == "long"


@pytest.mark.parametrize(
    "selector_name",
    [
        "select_shortest_successful_trajectory",
        "select_mean_length_successful_trajectory",
        "select_longest_successful_trajectory",
    ],
)
def test_trajectory_selectors_fallback_to_first_rollout_when_none_succeed(
    selector_name: str,
) -> None:
    module = _load_astar_evaluator_module()
    trajectories = [
        {"name": "first", "success": False, "steps": 0, "path_length": 0.0},
        {"name": "second", "success": False, "steps": 0, "path_length": 0.0},
    ]

    selected = getattr(module, selector_name)(trajectories)

    assert selected["name"] == "first"


def _assert_path_moves_one_cell_at_a_time(path: list[Position]) -> None:
    for current_position, next_position in zip(path, path[1:], strict=False):
        dx = abs(next_position[0] - current_position[0])
        dy = abs(next_position[1] - current_position[1])
        assert dx + dy == 1


def _load_astar_evaluator_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / ("evaluate_astar_static.py")
    )
    spec = spec_from_file_location("evaluate_astar_static", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trajectory_examples() -> list[dict[str, object]]:
    return [
        {"name": "failed", "success": False, "steps": 0, "path_length": 0.0},
        {"name": "short", "success": True, "steps": 4, "path_length": 4.0},
        {"name": "medium", "success": True, "steps": 7, "path_length": 7.0},
        {"name": "long", "success": True, "steps": 10, "path_length": 10.0},
    ]
