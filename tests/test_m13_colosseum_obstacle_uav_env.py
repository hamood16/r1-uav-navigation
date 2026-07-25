from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import r1_uav_nav.envs.colosseum_obstacle_uav_env as obstacle_module
from r1_uav_nav.envs.colosseum_obstacle_uav_env import (
    ColosseumObstacleUAVEnv,
    ColosseumObstacleUAVEnvConfig,
    CourseSelectionMode,
    ObstacleRuntimeAuthorization,
    load_colosseum_obstacle_uav_env_config,
)
from r1_uav_nav.sim.colosseum_client import CleanupResult
from r1_uav_nav.sim.colosseum_scene import (
    MaterializedScene,
    SceneCleanupResult,
)
from r1_uav_nav.sim.scene_specification import Bounds3D, Vector3
from r1_uav_nav.sim.static_course import (
    generate_solvable_course,
    load_course_suite_config,
)
from r1_uav_nav.sim.waypoint_navigation import Position3D

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "env" / "m13_5_obstacle_uav_env.yaml"
COURSE_PATH = ROOT / "configs" / "planning" / "m13_3_voxel_astar.yaml"


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def confirmConnection(self) -> None:
        self.calls.append(("confirm",))

    def reset(self) -> None:
        self.calls.append(("forbidden-reset",))


class FakeInnerEnvironment:
    def __init__(
        self,
        config: Any,
        client: FakeClient,
        events: list[str],
    ) -> None:
        self.config = config
        self.client = client
        self.events = events
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    np.full(10, -1.0, dtype=np.float32),
                    np.zeros(73, dtype=np.float32),
                )
            ),
            high=np.ones(83, dtype=np.float32),
            dtype=np.float32,
        )
        self.position = Position3D(0.0, 0.0, -2.0)
        self.goal = Position3D(3.0, 0.0, -2.0)
        self.next_position: Position3D | None = None
        self.next_distance: float | None = None
        self.next_collision = False
        self.next_ground_clearance = 2.0
        self.next_terminated = False
        self.next_truncated = False
        self.next_sensor_failure = False
        self.next_lidar_valid = True
        self.next_error: Exception | None = None
        self.close_result = CleanupResult((), (), (), False)
        self.close_count = 0
        self.hover_count = 0
        self.actions: list[np.ndarray] = []
        self.lifecycle_evidence: dict[str, Any] = {
            "cleanup_attempts": ({"landing_confirmed": True},)
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        assert options is not None
        self.position = Position3D(*options["start_anchor"])
        self.goal = Position3D(*options["goal_approach"])
        return self._observation(True), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.next_error is not None:
            error = self.next_error
            self.next_error = None
            raise error
        self.actions.append(action.copy())
        if self.next_position is None:
            self.position = Position3D(
                self.position.x + 0.1,
                self.position.y,
                self.position.z,
            )
        else:
            self.position = self.next_position
        info = self._info()
        info["collision"] = self.next_collision
        info["ground_clearance"] = self.next_ground_clearance
        info["sensor_failure"] = self.next_sensor_failure
        if self.next_distance is not None:
            info["distance_to_goal"] = self.next_distance
        if self.next_sensor_failure:
            info["lidar"]["consecutive_invalid_scans"] = 3
        return (
            self._observation(self.next_lidar_valid),
            999.0,
            self.next_terminated,
            self.next_truncated,
            info,
        )

    def request_named_terminal_hover(self) -> tuple[str, ...]:
        self.hover_count += 1
        self.events.append("terminal-hover")
        return ()

    def close_with_result(self) -> CleanupResult:
        self.close_count += 1
        self.events.append("uav-cleanup")
        return self.close_result

    def _observation(self, lidar_valid: bool) -> np.ndarray:
        observation = np.zeros(83, dtype=np.float32)
        observation[10:82] = 0.5 if lidar_valid else 1.0
        observation[82] = 1.0 if lidar_valid else 0.0
        return observation

    def _info(self) -> dict[str, Any]:
        distance = float(
            np.linalg.norm(
                np.asarray(
                    (
                        self.goal.x - self.position.x,
                        self.goal.y - self.position.y,
                        self.goal.z - self.position.z,
                    )
                )
            )
        )
        return {
            "measured_position": (
                self.position.x,
                self.position.y,
                self.position.z,
            ),
            "distance_to_goal": distance,
            "collision": False,
            "ground_clearance": 2.0,
            "sensor_failure": False,
            "lidar": {"consecutive_invalid_scans": 0},
            "termination_reason": None,
        }


class InnerFactory:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.instances: list[FakeInnerEnvironment] = []

    def __call__(self, config: Any, client: FakeClient) -> FakeInnerEnvironment:
        assert config.lidar.vehicle_name == "SimpleFlight"
        assert config.lidar.lidar_name == "LidarSensor1"
        instance = FakeInnerEnvironment(config, client, self.events)
        self.instances.append(instance)
        return instance


class FakeSceneManager:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.active_runtime: Any | None = None
        self.last_runtime: Any | None = None
        self.reset_calls: list[str] = []

    def reset_scene(
        self,
        scene: Any,
        config: Any,
    ) -> tuple[MaterializedScene, object]:
        assert config.vehicle_name == "SimpleFlight"
        self.events.append("scene-reset")
        self.reset_calls.append(scene.config.scene_id)
        workspace = scene.config.workspace
        runtime = object()
        materialized = MaterializedScene(
            run_id="fake-run",
            scene_id=scene.config.scene_id,
            scene_digest=scene.scene_digest,
            materialization_digest="fake-materialization",
            backend="fake",
            world_origin=Vector3(0.0, 0.0, 0.0),
            workspace_world=Bounds3D(
                workspace.min_x,
                workspace.max_x,
                workspace.min_y,
                workspace.max_y,
                workspace.min_z,
                workspace.max_z,
            ),
            initial_vehicle_exclusion=Bounds3D(-1.0, 1.0, -1.0, 1.0, -3.0, 1.0),
            start_anchor_world=scene.start_anchor,
            goal_approach_world=scene.goal_approach,
            objects=(),
            ownership_manifest_path="ignored",
            collision_geometry_complete=True,
            collision_response_verified=False,
            markers_created=True,
        )
        self.active_runtime = runtime
        self.last_runtime = runtime
        return materialized, runtime


def _all_authorized() -> ObstacleRuntimeAuthorization:
    return ObstacleRuntimeAuthorization(
        allow_live_rpc=True,
        allow_scene_mutation=True,
        confirm_scene_area_clear=True,
        confirm_no_visible_collision=True,
        allow_debug_markers=True,
        allow_marker_flush=True,
        allow_flight=True,
        allow_start_positioning=True,
        confirm_clear_airspace=True,
    )


def _external_options() -> dict[str, Any]:
    return {
        "start_anchor": (0.0, 0.0, -2.0),
        "goal_approach": (3.0, 0.0, -2.0),
        "workspace_bounds": {
            "min_x": -5.0,
            "max_x": 5.0,
            "min_y": -5.0,
            "max_y": 5.0,
            "min_z": -5.0,
            "max_z": 0.0,
        },
        "scene_id": "fake-scene",
        "scene_digest": "fake-digest",
        "profile_id": "fake-profile",
        "base_seed": 7,
        "accepted_candidate_seed": 8,
        "attempt_index": 1,
    }


def _external_environment(
    *,
    max_episode_steps: int = 200,
    watchdog_timeout_s: float = 120.0,
    monotonic_fn=lambda: 0.0,
) -> tuple[
    ColosseumObstacleUAVEnv,
    FakeClient,
    InnerFactory,
    list[str],
]:
    events: list[str] = []
    client = FakeClient()
    factory = InnerFactory(events)
    base = ColosseumObstacleUAVEnvConfig()
    config = replace(
        base,
        course=replace(
            base.course,
            allow_external_test_endpoints=True,
        ),
        max_episode_steps=max_episode_steps,
        watchdog_timeout_s=watchdog_timeout_s,
    )
    env = ColosseumObstacleUAVEnv(
        config,
        client_factory=lambda: client,
        client_module=ModuleType("fake_airsim"),
        inner_environment_factory=factory,
        repository_root=ROOT,
        monotonic_fn=monotonic_fn,
        sleep_fn=lambda _seconds: None,
    )
    return env, client, factory, events


def _fixed_environment() -> tuple[
    ColosseumObstacleUAVEnv,
    FakeClient,
    InnerFactory,
    FakeSceneManager,
    list[str],
]:
    events: list[str] = []
    client = FakeClient()
    inner_factory = InnerFactory(events)
    manager = FakeSceneManager(events)
    config = replace(
        ColosseumObstacleUAVEnvConfig(),
        authorization=_all_authorized(),
    )
    env = ColosseumObstacleUAVEnv(
        config,
        client_factory=lambda: client,
        client_module=ModuleType("fake_airsim"),
        scene_manager_factory=lambda _client, _module, _catalog: manager,
        inner_environment_factory=inner_factory,
        repository_root=ROOT,
        sleep_fn=lambda _seconds: None,
    )
    return env, client, inner_factory, manager, events


def test_import_constructor_and_config_loading_are_offline() -> None:
    config = load_colosseum_obstacle_uav_env_config(CONFIG_PATH)
    env = ColosseumObstacleUAVEnv(config, repository_root=ROOT)

    assert config.schema_version == 1
    assert config.course.fixed_profile_id == "easy"
    assert config.course.fixed_base_seed == 1100
    assert not config.authorization.complete
    assert env.client is None
    assert env.action_space == gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    assert env.observation_space.shape == (83,)


def test_external_endpoint_mode_requires_explicit_test_gate() -> None:
    env = ColosseumObstacleUAVEnv(
        client_factory=FakeClient,
        repository_root=ROOT,
    )

    with pytest.raises(ValueError, match="test-only gate"):
        env.reset(options=_external_options())
    assert env.client is None


def test_reset_and_step_preserve_observation_and_info_contracts() -> None:
    env, client, factory, _ = _external_environment()

    observation, reset_info = env.reset(seed=17, options=_external_options())
    action = np.asarray((0.5, -0.25, 0.0), dtype=np.float32)
    stepped, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == stepped.shape == (83,)
    assert observation.dtype == stepped.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert np.all(np.isfinite(stepped))
    assert math_is_finite(reward)
    assert not terminated and not truncated
    assert reset_info["scene_digest"] == "fake-digest"
    assert info["profile_id"] == "fake-profile"
    assert info["previous_action"] == [0.0, 0.0, 0.0]
    assert info["action"] == pytest.approx(action.tolist())
    assert info["path_length_m"] == pytest.approx(0.1)
    assert info["reward_breakdown"]["total"] == pytest.approx(reward)
    assert factory.instances[-1].actions[0].dtype == np.float32
    assert ("forbidden-reset",) not in client.calls


class _CheckableEndpointEnvironment(ColosseumObstacleUAVEnv):
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return super().reset(
            seed=seed,
            options=options or _external_options(),
        )


def test_gymnasium_environment_checker_passes_with_fake_client() -> None:
    env, _, factory, _ = _external_environment()
    checked = _CheckableEndpointEnvironment(
        env.obstacle_config,
        client_factory=env.client_factory,
        client_module=ModuleType("fake_airsim"),
        inner_environment_factory=factory,
        repository_root=ROOT,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    check_env(checked, skip_render_check=True)
    checked.close()


def test_fixed_course_and_external_validated_course_are_deterministic() -> None:
    suite = load_course_suite_config(COURSE_PATH)
    accepted = generate_solvable_course(
        suite,
        "easy",
        1100,
        repository_root=ROOT,
    )
    env, _, _, manager, _ = _fixed_environment()

    _, fixed_info = env.reset(seed=1)
    _, external_info = env.reset(
        seed=99,
        options={"validated_course": accepted},
    )

    assert fixed_info["profile_id"] == "easy"
    assert fixed_info["base_seed"] == 1100
    assert fixed_info["scene_digest"] == external_info["scene_digest"]
    assert len(manager.reset_calls) == 2


def test_seeded_selection_uses_episode_local_rng() -> None:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    base = ColosseumObstacleUAVEnvConfig()
    seeded_course = replace(
        base.course,
        mode=CourseSelectionMode.SEEDED,
        seeded_profile_ids=("easy",),
    )
    selected: list[tuple[str, int]] = []
    for _ in range(2):
        env, _, _, _, _ = _fixed_environment()
        env.obstacle_config = replace(
            base,
            course=seeded_course,
            authorization=_all_authorized(),
        )
        _, info = env.reset(seed=42)
        selected.append((info["profile_id"], info["base_seed"]))
        env.close()

    assert selected[0] == selected[1]
    assert selected[0][0] == "easy"
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("collision", "collision"),
        ("ground", "ground_clearance_violation"),
        ("workspace", "workspace_violation"),
        ("goal", "goal_reached"),
    ],
)
def test_primary_terminal_outcomes_and_reward_priority(
    outcome: str,
    reason: str,
) -> None:
    env, _, factory, _ = _external_environment()
    env.reset(options=_external_options())
    inner = factory.instances[-1]
    if outcome == "collision":
        inner.next_collision = True
        inner.next_distance = 0.0
    elif outcome == "ground":
        inner.next_ground_clearance = 0.5
    elif outcome == "workspace":
        inner.next_position = Position3D(6.0, 0.0, -2.0)
    else:
        inner.next_distance = 0.1

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated and not truncated
    assert info["termination_reason"] == reason
    assert info["success"] is (outcome == "goal")
    if outcome == "collision":
        assert info["reward_breakdown"]["success_bonus"] == 0.0
        assert info["reward_breakdown"]["collision_penalty"] == -25.0
    with pytest.raises(RuntimeError, match="Episode is complete"):
        env.step(np.zeros(3, dtype=np.float32))


def test_transient_invalid_scan_uses_retained_clearance() -> None:
    env, _, factory, _ = _external_environment()
    env.reset(options=_external_options())
    inner = factory.instances[-1]
    inner.next_lidar_valid = False

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert not terminated and not truncated
    assert info["minimum_lidar_clearance_m"] is None
    assert info["reward_clearance_m"] is not None
    assert info["clearance_source"] == "last_valid"


def test_sensor_failure_max_step_and_watchdog_are_truncations() -> None:
    sensor_env, _, sensor_factory, _ = _external_environment()
    sensor_env.reset(options=_external_options())
    sensor_inner = sensor_factory.instances[-1]
    sensor_inner.next_lidar_valid = False
    sensor_inner.next_sensor_failure = True
    sensor_inner.next_truncated = True
    _, _, terminated, truncated, info = sensor_env.step(np.zeros(3, dtype=np.float32))
    assert not terminated and truncated
    assert info["termination_reason"] == "sensor_failure"

    max_env, _, _, _ = _external_environment(max_episode_steps=1)
    max_env.reset(options=_external_options())
    _, _, _, truncated, info = max_env.step(np.zeros(3, dtype=np.float32))
    assert truncated
    assert info["termination_reason"] == "max_steps"

    times = iter((0.0, 2.0))
    watchdog_env, _, watchdog_factory, _ = _external_environment(
        watchdog_timeout_s=1.0,
        monotonic_fn=lambda: next(times),
    )
    watchdog_env.reset(options=_external_options())
    _, _, _, truncated, info = watchdog_env.step(np.zeros(3, dtype=np.float32))
    assert truncated
    assert info["termination_reason"] == "watchdog_timeout"
    assert watchdog_factory.instances[-1].hover_count == 1


def test_recoverable_rpc_error_truncates_after_named_cleanup() -> None:
    env, _, factory, events = _external_environment()
    observation, _ = env.reset(options=_external_options())
    inner = factory.instances[-1]
    inner.next_error = OSError("RPC unavailable")

    recovered, reward, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32)
    )

    assert np.array_equal(recovered, observation)
    assert math_is_finite(reward)
    assert not terminated and truncated
    assert info["termination_reason"] == "rpc_recovery"
    assert "OSError: RPC unavailable" in info["rpc_error"]
    assert events == ["uav-cleanup"]


def test_reset_cleans_previous_inner_before_scene_replacement() -> None:
    env, _, factory, _, events = _fixed_environment()
    env.reset(seed=1)
    first = factory.instances[-1]
    env.reset(seed=2)

    assert first.close_count == 1
    assert events[:3] == ["scene-reset", "uav-cleanup", "scene-reset"]


def test_close_orders_safe_uav_then_exact_scene_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _, factory, manager, events = _fixed_environment()
    env.reset()

    def fake_cleanup(_client: Any, _runtime: Any):
        events.extend(("object-cleanup", "marker-cleanup"))
        return (
            SceneCleanupResult("objects", True, True),
            SceneCleanupResult("markers", True, True),
        )

    monkeypatch.setattr(obstacle_module, "cleanup_scene_resources", fake_cleanup)
    result = env.close_with_result()
    repeated = env.close_with_result()

    assert result is repeated
    assert result is not None and result.succeeded
    assert factory.instances[-1].close_count == 1
    assert events[-3:] == ["uav-cleanup", "object-cleanup", "marker-cleanup"]


def test_unsafe_uav_cleanup_defers_scene_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _, factory, _, events = _fixed_environment()
    env.reset()
    factory.instances[-1].close_result = CleanupResult(
        ("land",),
        (),
        ("landing unverified",),
        True,
    )
    scene_calls: list[str] = []
    monkeypatch.setattr(
        obstacle_module,
        "cleanup_scene_resources",
        lambda _client, _runtime: scene_calls.append("scene"),
    )

    result = env.close_with_result()

    assert result is not None
    assert result.scene_cleanup_deferred
    assert "not conclusively safe" in str(result.scene_cleanup_deferred_reason)
    assert scene_calls == []
    assert events[-1] == "uav-cleanup"


def test_reset_after_terminal_episode_is_safe() -> None:
    env, _, factory, _ = _external_environment()
    env.reset(options=_external_options())
    factory.instances[-1].next_distance = 0.0
    env.step(np.zeros(3, dtype=np.float32))

    observation, info = env.reset(seed=4, options=_external_options())

    assert observation.shape == (83,)
    assert info["prior_episode_cleanup"]["succeeded"]


def test_config_loader_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\n"
        "max_episode_steps: 10\n"
        "watchdog_timeout_s: 2\n"
        "unknown: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown root keys"):
        load_colosseum_obstacle_uav_env_config(path)


def test_invalid_navigation_and_authorization_are_rejected_offline() -> None:
    base = ColosseumObstacleUAVEnvConfig()

    with pytest.raises(ValueError, match="workspace_xy_limit"):
        replace(
            base,
            navigation=replace(base.navigation, workspace_xy_limit=0.0),
        )
    with pytest.raises(ValueError, match="authorization values"):
        ObstacleRuntimeAuthorization(allow_live_rpc="yes")  # type: ignore[arg-type]


def test_observation_contains_no_course_or_oracle_payload() -> None:
    env, _, _, _ = _external_environment()
    observation, info = env.reset(options=_external_options())

    assert observation.dtype == np.float32
    assert observation.shape == (83,)
    assert not {
        "obstacles",
        "occupancy",
        "reference_path",
        "reference_path_length_m",
        "oracle_path_length",
        "raw_point_cloud",
    }.intersection(info)


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))
