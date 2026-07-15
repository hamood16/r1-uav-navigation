"""A* path planning for static two-dimensional grid maps."""

from __future__ import annotations

import heapq
from itertools import count

Position = tuple[int, int]

_NEIGHBOUR_DELTAS: tuple[Position, ...] = (
    (0, 1),  # Up
    (0, -1),  # Down
    (-1, 0),  # Left
    (1, 0),  # Right
)


def find_astar_path(
    start: Position,
    goal: Position,
    obstacles: set[Position],
    grid_size: int,
) -> list[Position] | None:
    """Find a shortest 4-connected path from start to goal with A*."""
    _validate_inputs(
        start=start,
        goal=goal,
        obstacles=obstacles,
        grid_size=grid_size,
    )

    if start == goal:
        return [start]

    tie_breaker = count()
    open_set: list[tuple[int, int, int, Position]] = []
    heapq.heappush(
        open_set,
        (_manhattan_distance(start, goal), 0, next(tie_breaker), start),
    )
    came_from: dict[Position, Position] = {}
    g_score: dict[Position, int] = {start: 0}
    closed_set: set[Position] = set()

    while open_set:
        _, _, _, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
        if current == goal:
            return _reconstruct_path(came_from, current)

        closed_set.add(current)

        for neighbour in _get_valid_neighbours(current, obstacles, grid_size):
            if neighbour in closed_set:
                continue

            tentative_g_score = g_score[current] + 1
            if tentative_g_score >= g_score.get(neighbour, float("inf")):
                continue

            came_from[neighbour] = current
            g_score[neighbour] = tentative_g_score
            heuristic = _manhattan_distance(neighbour, goal)
            f_score = tentative_g_score + heuristic
            heapq.heappush(
                open_set,
                (
                    f_score,
                    heuristic,
                    next(tie_breaker),
                    neighbour,
                ),
            )

    return None


def _validate_inputs(
    start: Position,
    goal: Position,
    obstacles: set[Position],
    grid_size: int,
) -> None:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if not _is_within_grid(start, grid_size):
        raise ValueError(f"start position {start!r} is outside the grid")
    if not _is_within_grid(goal, grid_size):
        raise ValueError(f"goal position {goal!r} is outside the grid")
    if start in obstacles:
        raise ValueError("start position cannot be inside an obstacle")
    if goal in obstacles:
        raise ValueError("goal position cannot be inside an obstacle")

    out_of_grid_obstacles = [
        obstacle for obstacle in obstacles if not _is_within_grid(obstacle, grid_size)
    ]
    if out_of_grid_obstacles:
        raise ValueError("obstacles cannot be outside the grid")


def _get_valid_neighbours(
    position: Position,
    obstacles: set[Position],
    grid_size: int,
) -> list[Position]:
    neighbours = []
    for dx, dy in _NEIGHBOUR_DELTAS:
        candidate = (position[0] + dx, position[1] + dy)
        if _is_within_grid(candidate, grid_size) and candidate not in obstacles:
            neighbours.append(candidate)
    return neighbours


def _is_within_grid(position: Position, grid_size: int) -> bool:
    x, y = position
    return 0 <= x < grid_size and 0 <= y < grid_size


def _manhattan_distance(position: Position, goal: Position) -> int:
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def _reconstruct_path(
    came_from: dict[Position, Position],
    current: Position,
) -> list[Position]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
