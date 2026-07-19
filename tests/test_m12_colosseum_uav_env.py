from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from r1_uav_nav.envs import ColosseumUAVEnv, ColosseumUAVEnvConfig
from r1_uav_nav.sim import ColosseumClientError
from r1_uav_nav.sim.waypoint_navigation import Position3D

DOC_PATH = Path("docs/m12_colosseum_gym_wrapper.md")


class FakeAsyncResult:
    def __init__(self, client: "FakeColosseumEnvClient", action: str) -> None:
        self.client = client
        self.action = action

    def join(self) -> None:
        self.client.calls.append(f"{self.action}.join")
        if self.client.fail_join_action == self.action:
            raise RuntimeError(f"{self.action} join failed")


class FakeColosseumEnvClient:
    def __init__(
        self,
        *,
        initial_position: Position3D | None = None,
        fail_reset: bool = False,
        fail_anchor_move: bool = False,
        fail_velocity_command: bool = False,
        fail_state_read: bool = False,
        fail_land: bool = False,
        collision: bool = False,
    ) -> None:
        self.initial_position = initial_position or Position3D(0.0, 0.0, 0.560)
        self.position = self.initial_position
        self.velocity = Position3D(0.0, 0.0, 0.0)
        self.fail_reset = fail_reset
        self.fail_anchor_move = fail_anchor_move
        self.fail_velocity_command = fail_velocity_command
        self.fail_state_read = fail_state_read
        self.fail_land = fail_land
        self.initial_collision = collision
        self.collision = collision
        self.fail_join_action: str | None = None
        self.calls: list[object] = []

    def confirmConnection(self) -> None:
        self.calls.append("confirmConnection")

    def reset(self) -> None:
        self.calls.append("reset")
        if self.fail_reset:
            raise RuntimeError("reset failed")
        self.position = self.initial_position
        self.velocity = Position3D(0.0, 0.0, 0.0)
        self.collision = self.initial_collision

    def getMultirotorState(self) -> object:
        self.calls.append("getMultirotorState")
        if self.fail_state_read:
            raise RuntimeError("state read failed")
        return _state(self.position, self.velocity, self.collision)

    def enableApiControl(self, enabled: bool) -> None:
        self.calls.append(("enableApiControl", enabled))

    def armDisarm(self, armed: bool) -> None:
        self.calls.append(("armDisarm", armed))

    def takeoffAsync(self) -> FakeAsyncResult:
        self.calls.append("takeoffAsync")
        return FakeAsyncResult(self, "takeoffAsync")

    def moveToPositionAsync(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float,
        timeout_sec: float | None = None,
    ) -> FakeAsyncResult:
        self.calls.append(("moveToPositionAsync", x, y, z, velocity, timeout_sec))
        if self.fail_anchor_move:
            raise RuntimeError("anchor move failed")
        self.position = Position3D(x, y, z)
        self.velocity = Position3D(0.0, 0.0, 0.0)
        return FakeAsyncResult(self, "moveToPositionAsync")

    def moveByVelocityAsync(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
    ) -> FakeAsyncResult:
        self.calls.append(("moveByVelocityAsync", vx, vy, vz, duration))
        if self.fail_velocity_command:
            raise RuntimeError("velocity command failed")
        self.velocity = Position3D(vx, vy, vz)
        self.position = Position3D(
            self.position.x + vx * duration,
            self.position.y + vy * duration,
            self.position.z + vz * duration,
        )
        return FakeAsyncResult(self, "moveByVelocityAsync")

    def hoverAsync(self) -> FakeAsyncResult:
        self.calls.append("hoverAsync")
        self.velocity = Position3D(0.0, 0.0, 0.0)
        return FakeAsyncResult(self, "hoverAsync")

    def landAsync(self) -> FakeAsyncResult:
        self.calls.append("landAsync")
        if self.fail_land:
            raise RuntimeError("landing failed")
        self.position = self.initial_position
        return FakeAsyncResult(self, "landAsync")

    def simGetCollisionInfo(self) -> object:
        self.calls.append("simGetCollisionInfo")
        return SimpleNamespace(has_collided=self.collision)


def test_doc_mentions_wrapper_topics() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert DOC_PATH.exists()
    assert "Gymnasium" in doc_text
    assert "NED" in doc_text
    assert "moveByVelocityAsync" in doc_text
    assert "anchor_move_timeout" in doc_text
    assert "M12.5" in doc_text


def test_constructor_does_not_call_simulator_and_spaces_are_declared() -> None:
    client = FakeColosseumEnvClient()

    env = ColosseumUAVEnv(client_factory=lambda: client)

    assert client.calls == []
    assert env.action_space.shape == (3,)
    assert env.action_space.dtype == np.float32
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32


@pytest.mark.parametrize(
    "config",
    [
        ColosseumUAVEnvConfig(anchor_altitude=0.0),
        ColosseumUAVEnvConfig(max_horizontal_velocity=0.0),
        ColosseumUAVEnvConfig(control_duration=0.0),
        ColosseumUAVEnvConfig(anchor_move_timeout=0.0),
        ColosseumUAVEnvConfig(goal_tolerance=1.0, min_goal_distance=1.0),
        ColosseumUAVEnvConfig(default_goal_offset=(0.25, 0.0, 0.0)),
        ColosseumUAVEnvConfig(max_episode_steps=0),
    ],
)
def test_invalid_config_is_rejected(config: ColosseumUAVEnvConfig) -> None:
    with pytest.raises(ValueError):
        ColosseumUAVEnv(config=config)


@pytest.mark.parametrize(
    "config",
    [
        ColosseumUAVEnvConfig(anchor_altitude=float("nan")),
        ColosseumUAVEnvConfig(min_ground_clearance=float("inf")),
        ColosseumUAVEnvConfig(workspace_xy_limit=float("-inf")),
        ColosseumUAVEnvConfig(max_vertical_velocity=float("nan")),
        ColosseumUAVEnvConfig(control_duration=float("inf")),
        ColosseumUAVEnvConfig(anchor_move_timeout=float("nan")),
        ColosseumUAVEnvConfig(goal_tolerance=float("inf")),
        ColosseumUAVEnvConfig(progress_reward_scale=float("nan")),
        ColosseumUAVEnvConfig(step_penalty=float("-inf")),
        ColosseumUAVEnvConfig(success_reward=float("inf")),
    ],
)
def test_non_finite_config_values_are_rejected(
    config: ColosseumUAVEnvConfig,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        ColosseumUAVEnv(config=config)


@pytest.mark.parametrize(
    "config",
    [
        ColosseumUAVEnvConfig(default_goal_offset=(6.0, 0.0, 0.0)),
        ColosseumUAVEnvConfig(default_goal_offset=(1.0, 6.0, 0.0)),
        ColosseumUAVEnvConfig(default_goal_offset=(1.0, 0.0, -4.0)),
        ColosseumUAVEnvConfig(default_goal_offset=(1.0, 0.0, 1.0)),
        ColosseumUAVEnvConfig(
            anchor_altitude=1.0,
            min_ground_clearance=0.9,
            default_goal_offset=(1.1, 0.0, 0.25),
        ),
        ColosseumUAVEnvConfig(
            workspace_xy_limit=0.5,
            workspace_up_limit=0.5,
            workspace_down_limit=0.25,
            min_goal_distance=2.0,
            goal_tolerance=0.5,
            default_goal_offset=(2.0, 0.0, 0.0),
        ),
    ],
)
def test_invalid_default_goal_config_is_rejected(
    config: ColosseumUAVEnvConfig,
) -> None:
    with pytest.raises(ValueError):
        ColosseumUAVEnv(config=config)


def test_reset_calls_client_reset_on_first_and_second_reset() -> None:
    client = FakeColosseumEnvClient()
    env = ColosseumUAVEnv(client_factory=lambda: client)

    _observation, info = env.reset(seed=123)
    env.reset(seed=123)

    assert client.calls.count("reset") == 2
    assert client.calls.index("reset") < client.calls.index("getMultirotorState")
    assert info["ground_reference_z"] == pytest.approx(0.560)
    assert info["anchor_position"] == pytest.approx((0.0, 0.0, -1.440))


def test_reset_uses_measured_initial_z_anchor_velocity_and_timeout() -> None:
    client = FakeColosseumEnvClient(
        initial_position=Position3D(4.0, -2.0, 0.560),
    )
    config = ColosseumUAVEnvConfig(
        anchor_altitude=2.5,
        anchor_move_velocity=0.4,
        anchor_move_timeout=12.0,
    )
    env = ColosseumUAVEnv(config=config, client_factory=lambda: client)

    _observation, info = env.reset()

    assert ("moveToPositionAsync", 4.0, -2.0, -1.940, 0.4, 12.0) in client.calls
    assert info["ground_reference_z"] == pytest.approx(0.560)
    assert info["anchor_position"] == pytest.approx((4.0, -2.0, -1.940))


def test_reset_recreates_client_once_after_reset_failure() -> None:
    first_client = FakeColosseumEnvClient(fail_reset=True)
    second_client = FakeColosseumEnvClient()
    clients = iter((first_client, second_client))
    env = ColosseumUAVEnv(client_factory=lambda: next(clients))

    env.reset()

    assert first_client.calls == ["confirmConnection", "reset"]
    assert "reset" in second_client.calls
    assert env.client is second_client


def test_reset_rejects_explicit_near_goal_and_accepts_explicit_goal() -> None:
    env = ColosseumUAVEnv(client_factory=FakeColosseumEnvClient)

    with pytest.raises(ValueError, match="min_goal_distance"):
        env.reset(options={"goal_offset": (0.5, 0.0, 0.0)})

    observation, info = env.reset(options={"goal_offset": (2.0, 0.0, 0.0)})

    assert env.observation_space.contains(observation)
    assert info["goal_position"] == pytest.approx((2.0, 0.0, -1.440))


def test_invalid_explicit_goal_is_rejected_before_control_or_takeoff() -> None:
    client = FakeColosseumEnvClient()
    env = ColosseumUAVEnv(client_factory=lambda: client)

    with pytest.raises(ValueError, match="workspace_xy_limit"):
        env.reset(options={"goal_offset": (6.0, 0.0, 0.0)})

    assert ("enableApiControl", True) not in client.calls
    assert ("armDisarm", True) not in client.calls
    assert "takeoffAsync" not in client.calls
    assert not any(_is_call(call, "moveToPositionAsync") for call in client.calls)


def test_seeded_random_goal_is_deterministic_and_not_zero_step() -> None:
    config = ColosseumUAVEnvConfig(random_goal=True)
    env_a = ColosseumUAVEnv(config=config, client_factory=FakeColosseumEnvClient)
    env_b = ColosseumUAVEnv(config=config, client_factory=FakeColosseumEnvClient)

    _obs_a, info_a = env_a.reset(seed=7)
    _obs_b, info_b = env_b.reset(seed=7)

    assert info_a["goal_position"] == pytest.approx(info_b["goal_position"])
    assert float(info_a["distance_to_goal"]) >= config.min_goal_distance


def test_observation_normalizes_boundary_goal_displacement_without_clipping() -> None:
    env = ColosseumUAVEnv()
    anchor = Position3D(0.0, 0.0, -2.0)
    state = _env_state(
        position=Position3D(-5.0, -5.0, -3.0),
        velocity=Position3D(1.0, -1.0, 0.5),
        anchor=anchor,
        goal=Position3D(5.0, 5.0, 0.25),
        ground_reference_z=1.25,
    )

    observation = env._build_observation(state)

    assert observation[:9] == pytest.approx(
        np.asarray([-1.0, -1.0, -1.0 / 3.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0])
    )
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)
    assert observation.dtype == np.float32


def test_non_finite_state_is_rejected() -> None:
    client = FakeColosseumEnvClient()
    env = ColosseumUAVEnv(client_factory=lambda: client)
    env.reset()
    client.position = Position3D(float("nan"), 0.0, -1.0)

    with pytest.raises(ColosseumClientError, match="non-finite"):
        env.step(np.zeros(3, dtype=np.float32))


def test_step_clips_action_maps_ned_velocity_and_joins() -> None:
    client = FakeColosseumEnvClient()
    env = ColosseumUAVEnv(client_factory=lambda: client)
    env.reset()

    env.step(np.asarray([2.0, -2.0, 1.0], dtype=np.float32))

    assert ("moveByVelocityAsync", 1.0, -1.0, 0.5, 0.5) in client.calls
    assert "moveByVelocityAsync.join" in client.calls


def test_reward_equation_for_progress_step_and_action_penalty() -> None:
    client = FakeColosseumEnvClient()
    env = ColosseumUAVEnv(client_factory=lambda: client)
    env.reset()

    _observation, reward, terminated, truncated, _info = env.step(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )

    assert reward == pytest.approx(0.5 - 0.02 - 0.01)
    assert not terminated
    assert not truncated


def test_success_termination_attempts_immediate_hover() -> None:
    config = ColosseumUAVEnvConfig(
        default_goal_offset=(1.0, 0.0, 0.0),
        min_goal_distance=0.75,
    )
    client = FakeColosseumEnvClient()
    env = ColosseumUAVEnv(config=config, client_factory=lambda: client)
    env.reset()

    _observation, reward, terminated, truncated, info = env.step(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )

    assert terminated
    assert not truncated
    assert info["success"] is True
    assert info["termination_reason"] == "goal_reached"
    assert reward > config.success_reward
    assert "hoverAsync" in client.calls
    assert env.episode_complete


def test_collision_out_of_bounds_ground_clearance_and_max_steps() -> None:
    collision_client = FakeColosseumEnvClient(collision=True)
    collision_env = ColosseumUAVEnv(client_factory=lambda: collision_client)
    collision_env.reset()
    assert (
        collision_env.step(np.zeros(3, dtype=np.float32))[4]["termination_reason"]
        == "collision"
    )

    out_config = ColosseumUAVEnvConfig(
        workspace_xy_limit=1.0,
        max_horizontal_velocity=3.0,
        default_goal_offset=(1.0, 0.0, 0.0),
        min_goal_distance=0.75,
    )
    out_client = FakeColosseumEnvClient()
    out_env = ColosseumUAVEnv(config=out_config, client_factory=lambda: out_client)
    out_env.reset()
    assert (
        out_env.step(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))[4][
            "termination_reason"
        ]
        == "out_of_bounds"
    )

    ground_config = ColosseumUAVEnvConfig(max_vertical_velocity=3.0)
    ground_client = FakeColosseumEnvClient()
    ground_env = ColosseumUAVEnv(
        config=ground_config,
        client_factory=lambda: ground_client,
    )
    ground_env.reset()
    assert (
        ground_env.step(np.asarray([0.0, 0.0, 1.0], dtype=np.float32))[4][
            "termination_reason"
        ]
        == "ground_clearance_violation"
    )

    trunc_env = ColosseumUAVEnv(
        config=ColosseumUAVEnvConfig(max_episode_steps=1),
        client_factory=FakeColosseumEnvClient,
    )
    trunc_env.reset()
    _obs, _reward, terminated, truncated, info = trunc_env.step(
        np.zeros(3, dtype=np.float32),
    )
    assert not terminated
    assert truncated
    assert info["termination_reason"] == "max_steps"


def test_step_lifecycle_rejections() -> None:
    env = ColosseumUAVEnv(client_factory=FakeColosseumEnvClient)

    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3, dtype=np.float32))

    env.reset()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros(3, dtype=np.float32))

    complete_env = ColosseumUAVEnv(
        config=ColosseumUAVEnvConfig(max_episode_steps=1),
        client_factory=FakeColosseumEnvClient,
    )
    complete_env.reset()
    complete_env.step(np.zeros(3, dtype=np.float32))
    with pytest.raises(RuntimeError, match="Episode is complete"):
        complete_env.step(np.zeros(3, dtype=np.float32))


def test_cleanup_close_paths_and_safety_critical_failure_capture() -> None:
    env = ColosseumUAVEnv(client_factory=FakeColosseumEnvClient)
    assert env.close_with_result() is None
    assert env.close_with_result() is None

    client = FakeColosseumEnvClient(fail_land=True)
    cleanup_env = ColosseumUAVEnv(client_factory=lambda: client)
    cleanup_env.reset()
    cleanup_env.close()

    assert cleanup_env.last_cleanup_result is not None
    assert cleanup_env.last_cleanup_result.safety_critical_failure
    assert ("armDisarm", False) in client.calls
    assert ("enableApiControl", False) in client.calls


def test_reset_failure_after_control_preserves_error_and_cleans_up() -> None:
    client = FakeColosseumEnvClient(fail_anchor_move=True)
    env = ColosseumUAVEnv(client_factory=lambda: client)

    with pytest.raises(ColosseumClientError, match="moveToPositionAsync"):
        env.reset()

    assert env.last_cleanup_result is not None
    assert "completed landing" in env.last_cleanup_result.actions_succeeded
    assert ("armDisarm", False) in client.calls
    assert ("enableApiControl", False) in client.calls


def test_step_command_failure_can_be_cleaned_up() -> None:
    client = FakeColosseumEnvClient(fail_velocity_command=True)
    env = ColosseumUAVEnv(client_factory=lambda: client)
    env.reset()

    with pytest.raises(ColosseumClientError, match="moveByVelocityAsync"):
        env.step(np.zeros(3, dtype=np.float32))

    env.close()
    assert env.last_cleanup_result is not None
    assert ("armDisarm", False) in client.calls
    assert ("enableApiControl", False) in client.calls


def test_smoke_script_parse_args_and_cleanup_failure_exit() -> None:
    script = _load_smoke_script_module()
    args = script.parse_args(
        [
            "--steps",
            "3",
            "--seed",
            "9",
            "--policy",
            "forward",
            "--goal-offset",
            "2,0,0",
            "--action",
            "0.5,0,0",
            "--client-module",
            "custom_client",
        ]
    )

    assert args.steps == 3
    assert args.seed == 9
    assert args.policy == "forward"
    assert args.goal_offset == "2,0,0"
    assert args.action == "0.5,0,0"
    assert args.client_module == "custom_client"

    client = FakeColosseumEnvClient(fail_land=True)
    env = ColosseumUAVEnv(client_factory=lambda: client)
    args = script.parse_args(["--steps", "0"])
    script.ColosseumUAVEnv = lambda _config: env

    assert script.run_smoke_test(args) == 1


def test_smoke_script_preserves_reset_cleanup_failure_after_close() -> None:
    script = _load_smoke_script_module()
    client = FakeColosseumEnvClient(fail_anchor_move=True, fail_land=True)
    env = ColosseumUAVEnv(client_factory=lambda: client)
    args = script.parse_args(["--steps", "0"])
    script.ColosseumUAVEnv = lambda _config: env

    exit_code = script.run_smoke_test(args)

    assert exit_code == 1
    assert env.last_cleanup_result is not None
    assert env.last_cleanup_result.safety_critical_failure
    assert any("landAsync failed" in error for error in env.last_cleanup_result.errors)
    assert env.last_cleanup_result.actions_attempted.count("landAsync") == 1


def test_gymnasium_checker_with_fake_client() -> None:
    env = ColosseumUAVEnv(client_factory=FakeColosseumEnvClient)

    check_env(env, skip_render_check=True)


def _state(position: Position3D, velocity: Position3D, collision: bool) -> object:
    return SimpleNamespace(
        collision=SimpleNamespace(has_collided=collision),
        kinematics_estimated=SimpleNamespace(
            position=SimpleNamespace(
                x_val=position.x,
                y_val=position.y,
                z_val=position.z,
            ),
            linear_velocity=SimpleNamespace(
                x_val=velocity.x,
                y_val=velocity.y,
                z_val=velocity.z,
            ),
        ),
    )


def _env_state(
    *,
    position: Position3D,
    velocity: Position3D,
    anchor: Position3D,
    goal: Position3D,
    ground_reference_z: float,
) -> object:
    return SimpleNamespace(
        position=position,
        linear_velocity=velocity,
        collision=False,
        ground_reference_z=ground_reference_z,
        anchor_position=anchor,
        goal_position=goal,
    )


def _load_smoke_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_colosseum_env_smoke_test.py"
    )
    spec = spec_from_file_location("run_colosseum_env_smoke_test", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_call(call: object, method_name: str) -> bool:
    return isinstance(call, tuple) and call[0] == method_name
