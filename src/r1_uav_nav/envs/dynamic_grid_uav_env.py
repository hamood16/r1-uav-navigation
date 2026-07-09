"""A two-dimensional grid environment with moving obstacles."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Position = tuple[int, int]


@dataclass
class DynamicObstacle:
    """A moving obstacle in the grid world."""

    position: Position
    velocity: Position


class DynamicGridUAVEnv(gym.Env[np.ndarray, int]):
    """A discrete grid world with dynamic obstacles."""

    metadata = {"render_modes": []}

    _ACTION_DELTAS: dict[int, Position] = {
        0: (0, 1),  # Up
        1: (0, -1),  # Down
        2: (-1, 0),  # Left
        3: (1, 0),  # Right
        4: (0, 0),  # Hover
    }
    _OBSTACLE_VELOCITIES: tuple[Position, ...] = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    )

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 100,
        num_dynamic_obstacles: int = 5,
        random_start: bool = True,
        random_goal: bool = True,
        step_penalty: float = -0.02,
        hover_penalty: float | None = None,
        boundary_penalty: float = -0.50,
        collision_penalty: float = -8.0,
        goal_reward: float = 10.0,
        timeout_penalty: float = -3.0,
        progress_reward_scale: float = 0.3,
    ) -> None:
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if num_dynamic_obstacles < 0:
            raise ValueError("num_dynamic_obstacles cannot be negative")

        max_obstacles = grid_size**2 - 2
        if num_dynamic_obstacles > max_obstacles:
            raise ValueError(
                f"num_dynamic_obstacles cannot exceed {max_obstacles} for this grid"
            )

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.random_start = random_start
        self.random_goal = random_goal
        self.step_penalty = step_penalty
        self.hover_penalty = step_penalty if hover_penalty is None else hover_penalty
        self.boundary_penalty = boundary_penalty
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.timeout_penalty = timeout_penalty
        self.progress_reward_scale = progress_reward_scale

        self.action_space = spaces.Discrete(5)
        self._max_obstacle_distance = math.hypot(grid_size - 1, grid_size - 1)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, -1, -1, 0], dtype=np.float32),
            high=np.array(
                [
                    grid_size - 1,
                    grid_size - 1,
                    grid_size - 1,
                    grid_size - 1,
                    grid_size - 1,
                    grid_size - 1,
                    1,
                    1,
                    self._max_obstacle_distance,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.uav_position: Position = (0, 0)
        self.goal_position: Position = (grid_size - 1, grid_size - 1)
        self.dynamic_obstacles: list[DynamicObstacle] = []
        self.current_step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode with non-overlapping placements."""
        super().reset(seed=seed)
        del options

        fixed_start = (0, 0)
        fixed_goal = (self.grid_size - 1, self.grid_size - 1)
        reserved: set[Position] = set()
        if not self.random_start:
            reserved.add(fixed_start)
        if not self.random_goal:
            reserved.add(fixed_goal)

        available = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in reserved
        ]

        if self.random_start:
            self.uav_position = self._take_random_position(available)
        else:
            self.uav_position = fixed_start

        if self.random_goal:
            self.goal_position = self._take_random_position(available)
        else:
            self.goal_position = fixed_goal

        self.dynamic_obstacles = [
            DynamicObstacle(
                position=self._take_random_position(available),
                velocity=self._take_random_velocity(),
            )
            for _ in range(self.num_dynamic_obstacles)
        ]
        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply a UAV action, then move dynamic obstacles."""
        if not self.action_space.contains(action):
            raise ValueError(
                f"invalid action {action!r}; expected an integer from 0 to 4"
            )

        dx, dy = self._ACTION_DELTAS[int(action)]
        candidate = (self.uav_position[0] + dx, self.uav_position[1] + dy)
        previous_goal_distance = self._distance_to_goal(self.uav_position)
        terminated = False
        collision_type: str | None = None
        is_success = False
        valid_uav_movement = False

        if not self._is_within_grid(candidate):
            reward = self.boundary_penalty
        else:
            self.uav_position = candidate
            valid_uav_movement = dx != 0 or dy != 0
            if self.uav_position in self._dynamic_obstacle_positions():
                reward = self.collision_penalty
                terminated = True
                collision_type = "uav_into_obstacle"
            else:
                reward = (
                    self.hover_penalty if dx == 0 and dy == 0 else self.step_penalty
                )

        if not terminated:
            self._move_dynamic_obstacles()
            if self.uav_position in self._dynamic_obstacle_positions():
                reward = self.collision_penalty
                terminated = True
                collision_type = "obstacle_into_uav"

        if not terminated and self.uav_position == self.goal_position:
            reward = self.goal_reward
            terminated = True
            is_success = True

        if not terminated and valid_uav_movement:
            new_goal_distance = self._distance_to_goal(self.uav_position)
            progress = previous_goal_distance - new_goal_distance
            reward += self.progress_reward_scale * progress

        self.current_step += 1
        truncated = self.current_step >= self.max_steps and not terminated
        if truncated:
            reward += self.timeout_penalty

        info = {
            "is_success": is_success,
            "is_collision": collision_type is not None,
            "collision_type": collision_type,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _move_dynamic_obstacles(self) -> None:
        for obstacle in self.dynamic_obstacles:
            obstacle.position, obstacle.velocity = self._get_bounced_move(
                obstacle.position,
                obstacle.velocity,
            )

    def _get_bounced_move(
        self,
        position: Position,
        velocity: Position,
    ) -> tuple[Position, Position]:
        x, y = position
        vx, vy = velocity
        next_position = (x + vx, y + vy)

        if self._is_within_grid(next_position):
            return next_position, velocity

        if not 0 <= next_position[0] < self.grid_size:
            vx *= -1
        if not 0 <= next_position[1] < self.grid_size:
            vy *= -1

        return (x + vx, y + vy), (vx, vy)

    def _dynamic_obstacle_positions(self) -> set[Position]:
        return {obstacle.position for obstacle in self.dynamic_obstacles}

    def _take_random_position(self, positions: list[Position]) -> Position:
        index = int(self.np_random.integers(len(positions)))
        return positions.pop(index)

    def _take_random_velocity(self) -> Position:
        index = int(self.np_random.integers(len(self._OBSTACLE_VELOCITIES)))
        return self._OBSTACLE_VELOCITIES[index]

    def _is_within_grid(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _distance_to_goal(self, position: Position) -> float:
        return math.hypot(
            position[0] - self.goal_position[0],
            position[1] - self.goal_position[1],
        )

    def _get_nearest_dynamic_obstacle(self) -> tuple[Position, Position, float]:
        if not self.dynamic_obstacles:
            return (0, 0), (0, 0), self._max_obstacle_distance

        nearest_obstacle = min(
            self.dynamic_obstacles,
            key=lambda obstacle: math.hypot(
                self.uav_position[0] - obstacle.position[0],
                self.uav_position[1] - obstacle.position[1],
            ),
        )
        distance = math.hypot(
            self.uav_position[0] - nearest_obstacle.position[0],
            self.uav_position[1] - nearest_obstacle.position[1],
        )
        return nearest_obstacle.position, nearest_obstacle.velocity, distance

    def _get_observation(self) -> np.ndarray:
        obstacle_position, obstacle_velocity, obstacle_distance = (
            self._get_nearest_dynamic_obstacle()
        )

        return np.array(
            [
                self.uav_position[0],
                self.uav_position[1],
                self.goal_position[0],
                self.goal_position[1],
                obstacle_position[0],
                obstacle_position[1],
                obstacle_velocity[0],
                obstacle_velocity[1],
                obstacle_distance,
            ],
            dtype=np.float32,
        )
