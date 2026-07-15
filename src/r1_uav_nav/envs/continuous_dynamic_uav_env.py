"""A continuous-control dynamic UAV environment for future TD3 training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ContinuousPosition = tuple[float, float]


@dataclass
class ContinuousDynamicObstacle:
    """A moving obstacle in a continuous two-dimensional world."""

    position: ContinuousPosition
    velocity: ContinuousPosition


class ContinuousDynamicUAVEnv(gym.Env[np.ndarray, np.ndarray]):
    """A continuous 2D navigation environment with moving obstacles."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_size: float = 10.0,
        max_steps: int = 200,
        num_dynamic_obstacles: int = 5,
        max_uav_speed: float = 1.0,
        obstacle_speed: float = 0.5,
        dt: float = 1.0,
        collision_radius: float = 0.35,
        goal_radius: float = 0.5,
        random_start: bool = True,
        random_goal: bool = True,
        step_penalty: float = -0.02,
        boundary_penalty: float = -0.50,
        collision_penalty: float = -8.0,
        goal_reward: float = 10.0,
        timeout_penalty: float = -3.0,
        progress_reward_scale: float = 0.3,
    ) -> None:
        super().__init__()

        if world_size <= 0.0:
            raise ValueError("world_size must be positive")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if num_dynamic_obstacles < 0:
            raise ValueError("num_dynamic_obstacles cannot be negative")
        if max_uav_speed <= 0.0:
            raise ValueError("max_uav_speed must be positive")
        if obstacle_speed < 0.0:
            raise ValueError("obstacle_speed cannot be negative")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if collision_radius <= 0.0:
            raise ValueError("collision_radius must be positive")
        if goal_radius <= 0.0:
            raise ValueError("goal_radius must be positive")

        self.world_size = float(world_size)
        self.max_steps = max_steps
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.max_uav_speed = float(max_uav_speed)
        self.obstacle_speed = float(obstacle_speed)
        self.dt = float(dt)
        self.collision_radius = float(collision_radius)
        self.goal_radius = float(goal_radius)
        self.random_start = random_start
        self.random_goal = random_goal
        self.step_penalty = step_penalty
        self.boundary_penalty = boundary_penalty
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.timeout_penalty = timeout_penalty
        self.progress_reward_scale = progress_reward_scale

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self._max_obstacle_distance = math.sqrt(2.0) * self.world_size
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -self.obstacle_speed,
                    -self.obstacle_speed,
                    0.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.world_size,
                    self.world_size,
                    self.world_size,
                    self.world_size,
                    self.world_size,
                    self.world_size,
                    self.obstacle_speed,
                    self.obstacle_speed,
                    self._max_obstacle_distance,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.uav_position: ContinuousPosition = (0.0, 0.0)
        self.goal_position: ContinuousPosition = (self.world_size, self.world_size)
        self.dynamic_obstacles: list[ContinuousDynamicObstacle] = []
        self.current_step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Start a new continuous-control episode."""
        super().reset(seed=seed)
        del options

        fixed_start = (0.0, 0.0)
        fixed_goal = (self.world_size, self.world_size)

        self.uav_position = (
            self._sample_position_away_from([fixed_goal], self.goal_radius * 2.0)
            if self.random_start
            else fixed_start
        )
        self.goal_position = (
            self._sample_position_away_from(
                [self.uav_position],
                self.goal_radius * 2.0,
            )
            if self.random_goal
            else fixed_goal
        )
        if (
            self._distance(self.uav_position, self.goal_position)
            < self.goal_radius * 2.0
        ):
            self.goal_position = self._sample_position_away_from(
                [self.uav_position],
                self.goal_radius * 2.0,
            )

        self.dynamic_obstacles = [
            ContinuousDynamicObstacle(
                position=self._sample_position_away_from(
                    [self.uav_position, self.goal_position],
                    max(self.collision_radius, self.goal_radius),
                ),
                velocity=self._sample_obstacle_velocity(),
            )
            for _ in range(self.num_dynamic_obstacles)
        ]
        self.current_step = 0

        return self._get_observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one continuous UAV velocity command."""
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (2,):
            raise ValueError("continuous action must have shape (2,)")

        clipped_action = np.clip(action_array, -1.0, 1.0)
        previous_goal_distance = self._distance_to_goal(self.uav_position)
        candidate = np.array(self.uav_position, dtype=np.float32) + (
            clipped_action * self.max_uav_speed * self.dt
        )
        clipped_position = np.clip(candidate, 0.0, self.world_size)
        hit_boundary = not np.allclose(candidate, clipped_position)
        self.uav_position = (float(clipped_position[0]), float(clipped_position[1]))

        self._move_dynamic_obstacles()

        is_collision = self._is_uav_colliding_with_obstacle()
        is_success = False
        collision_type = "uav_obstacle_collision" if is_collision else None
        terminated = False

        if is_collision:
            reward = self.collision_penalty
            terminated = True
        elif self._distance_to_goal(self.uav_position) <= self.goal_radius:
            reward = self.goal_reward
            is_success = True
            terminated = True
        else:
            new_goal_distance = self._distance_to_goal(self.uav_position)
            progress = previous_goal_distance - new_goal_distance
            reward = self.step_penalty
            if hit_boundary:
                reward += self.boundary_penalty
            reward += self.progress_reward_scale * progress

        self.current_step += 1
        truncated = self.current_step >= self.max_steps and not terminated
        if truncated:
            reward += self.timeout_penalty

        info = {
            "is_success": is_success,
            "is_collision": is_collision,
            "collision_type": collision_type,
        }
        return self._get_observation(), float(reward), terminated, truncated, info

    def _move_dynamic_obstacles(self) -> None:
        for obstacle in self.dynamic_obstacles:
            obstacle.position, obstacle.velocity = self._get_bounced_obstacle_move(
                obstacle.position,
                obstacle.velocity,
            )

    def _get_bounced_obstacle_move(
        self,
        position: ContinuousPosition,
        velocity: ContinuousPosition,
    ) -> tuple[ContinuousPosition, ContinuousPosition]:
        x, y = position
        vx, vy = velocity
        next_x = x + vx * self.dt
        next_y = y + vy * self.dt

        if not 0.0 <= next_x <= self.world_size:
            vx *= -1.0
            next_x = x + vx * self.dt
        if not 0.0 <= next_y <= self.world_size:
            vy *= -1.0
            next_y = y + vy * self.dt

        next_x = float(np.clip(next_x, 0.0, self.world_size))
        next_y = float(np.clip(next_y, 0.0, self.world_size))
        return (next_x, next_y), (vx, vy)

    def _sample_position_away_from(
        self,
        blocked_positions: list[ContinuousPosition],
        minimum_distance: float,
    ) -> ContinuousPosition:
        for _ in range(1_000):
            position = (
                float(self.np_random.uniform(0.0, self.world_size)),
                float(self.np_random.uniform(0.0, self.world_size)),
            )
            if all(
                self._distance(position, blocked_position) >= minimum_distance
                for blocked_position in blocked_positions
            ):
                return position

        return (
            float(self.np_random.uniform(0.0, self.world_size)),
            float(self.np_random.uniform(0.0, self.world_size)),
        )

    def _sample_obstacle_velocity(self) -> ContinuousPosition:
        if self.obstacle_speed == 0.0:
            return (0.0, 0.0)

        angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        return (
            self.obstacle_speed * math.cos(angle),
            self.obstacle_speed * math.sin(angle),
        )

    def _is_uav_colliding_with_obstacle(self) -> bool:
        return any(
            self._distance(self.uav_position, obstacle.position)
            <= self.collision_radius
            for obstacle in self.dynamic_obstacles
        )

    def _distance_to_goal(self, position: ContinuousPosition) -> float:
        return self._distance(position, self.goal_position)

    def _get_nearest_dynamic_obstacle(
        self,
    ) -> tuple[ContinuousPosition, ContinuousPosition, float]:
        if not self.dynamic_obstacles:
            return (0.0, 0.0), (0.0, 0.0), self._max_obstacle_distance

        nearest_obstacle = min(
            self.dynamic_obstacles,
            key=lambda obstacle: self._distance(
                self.uav_position,
                obstacle.position,
            ),
        )
        distance = self._distance(self.uav_position, nearest_obstacle.position)
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

    @staticmethod
    def _distance(
        first_position: ContinuousPosition,
        second_position: ContinuousPosition,
    ) -> float:
        return math.hypot(
            first_position[0] - second_position[0],
            first_position[1] - second_position[1],
        )
