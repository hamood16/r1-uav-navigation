"""A minimal two-dimensional grid environment for UAV navigation."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Position = tuple[int, int]


class GridUAVEnv(gym.Env[np.ndarray, int]):
    """A discrete grid world with a goal and static obstacles."""

    metadata = {"render_modes": []}

    _ACTION_DELTAS: dict[int, Position] = {
        0: (0, 1),  # Up
        1: (0, -1),  # Down
        2: (-1, 0),  # Left
        3: (1, 0),  # Right
        4: (0, 0),  # Hover
    }

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 100,
        num_obstacles: int = 10,
        random_start: bool = True,
        random_goal: bool = True,
        step_penalty: float = -0.01,
        hover_penalty: float | None = None,
        boundary_penalty: float = -0.10,
        collision_penalty: float = -1.0,
        goal_reward: float = 1.0,
        timeout_penalty: float = 0.0,
        progress_reward_scale: float = 0.0,
        use_lidar: bool = False,
    ) -> None:
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if num_obstacles < 0:
            raise ValueError("num_obstacles cannot be negative")

        max_obstacles = grid_size**2 - 2
        if num_obstacles > max_obstacles:
            raise ValueError(
                f"num_obstacles cannot exceed {max_obstacles} for this grid"
            )

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_obstacles = num_obstacles
        self.random_start = random_start
        self.random_goal = random_goal
        self.step_penalty = step_penalty
        self.hover_penalty = step_penalty if hover_penalty is None else hover_penalty
        self.boundary_penalty = boundary_penalty
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.timeout_penalty = timeout_penalty
        self.progress_reward_scale = progress_reward_scale
        self.use_lidar = use_lidar

        self.action_space = spaces.Discrete(5)
        max_distance = math.hypot(grid_size - 1, grid_size - 1)
        observation_high = [
            grid_size - 1,
            grid_size - 1,
            grid_size - 1,
            grid_size - 1,
            max_distance,
        ]
        if self.use_lidar:
            observation_high.extend([grid_size - 1] * 4)

        self.observation_space = spaces.Box(
            low=np.zeros(len(observation_high), dtype=np.float32),
            high=np.array(observation_high, dtype=np.float32),
            dtype=np.float32,
        )

        self.uav_position: Position = (0, 0)
        self.goal_position: Position = (grid_size - 1, grid_size - 1)
        self.obstacles: set[Position] = set()
        self.current_step = 0
        self._max_obstacle_distance = max_distance

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

        self.obstacles = {
            self._take_random_position(available) for _ in range(self.num_obstacles)
        }
        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one discrete movement action."""
        if not self.action_space.contains(action):
            raise ValueError(
                f"invalid action {action!r}; expected an integer from 0 to 4"
            )

        dx, dy = self._ACTION_DELTAS[int(action)]
        candidate = (self.uav_position[0] + dx, self.uav_position[1] + dy)
        previous_goal_distance = self._distance_to_goal(self.uav_position)
        terminated = False

        if not self._is_within_grid(candidate):
            reward = self.boundary_penalty
        elif candidate in self.obstacles:
            reward = self.collision_penalty
            terminated = True
        else:
            self.uav_position = candidate
            if self.uav_position == self.goal_position:
                reward = self.goal_reward
                terminated = True
            else:
                reward = (
                    self.hover_penalty if dx == 0 and dy == 0 else self.step_penalty
                )
                if dx != 0 or dy != 0:
                    new_goal_distance = self._distance_to_goal(self.uav_position)
                    progress = previous_goal_distance - new_goal_distance
                    reward += self.progress_reward_scale * progress

        self.current_step += 1
        truncated = self.current_step >= self.max_steps and not terminated
        if truncated:
            reward += self.timeout_penalty

        return self._get_observation(), reward, terminated, truncated, {}

    def _take_random_position(self, positions: list[Position]) -> Position:
        index = int(self.np_random.integers(len(positions)))
        return positions.pop(index)

    def _is_within_grid(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _distance_to_goal(self, position: Position) -> float:
        return math.hypot(
            position[0] - self.goal_position[0],
            position[1] - self.goal_position[1],
        )

    def _get_directional_clearance(self, dx: int, dy: int) -> int:
        clearance = 0
        x = self.uav_position[0] + dx
        y = self.uav_position[1] + dy

        while self._is_within_grid((x, y)):
            if (x, y) in self.obstacles:
                break
            clearance += 1
            x += dx
            y += dy

        return clearance

    def _get_observation(self) -> np.ndarray:
        if self.obstacles:
            nearest_obstacle_distance = min(
                math.hypot(
                    self.uav_position[0] - obstacle[0],
                    self.uav_position[1] - obstacle[1],
                )
                for obstacle in self.obstacles
            )
        else:
            nearest_obstacle_distance = self._max_obstacle_distance

        observation = [
            self.uav_position[0],
            self.uav_position[1],
            self.goal_position[0],
            self.goal_position[1],
            nearest_obstacle_distance,
        ]
        if self.use_lidar:
            observation.extend(
                [
                    self._get_directional_clearance(0, 1),
                    self._get_directional_clearance(0, -1),
                    self._get_directional_clearance(-1, 0),
                    self._get_directional_clearance(1, 0),
                ]
            )

        return np.array(observation, dtype=np.float32)
