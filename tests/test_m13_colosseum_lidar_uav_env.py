"""Named-lifecycle tests for the opt-in M13.4 environment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from r1_uav_nav.envs.colosseum_lidar_uav_env import (
    ColosseumLidarUAVEnv,
    ColosseumLidarUAVEnvConfig,
    LidarResetError,
)
from r1_uav_nav.envs.colosseum_uav_env import ColosseumUAVEnvConfig
from r1_uav_nav.sim.lidar_features import LidarFeatureConfig


class FakeAsync:
    def __init__(self, callback=None) -> None:
        self.callback = callback

    def join(self) -> None:
        if self.callback:
            self.callback()


class FakeNamedLidarClient:
    def __init__(self, timestamps: list[int] | None = None) -> None:
        self.calls: list[tuple] = []
        self.position = [0.0, 0.0, 0.57]
        self.velocity = [0.0, 0.0, 0.0]
        self.landed_state = 0
        self.api_enabled = False
        self.timestamps = list(timestamps or [1, 2, 3, 4, 5])
        self.phase = "ground"
        self.anchor_samples: list[
            tuple[tuple[float, float, float], tuple[float, float, float], int]
        ] = []
        self.touchdown_samples: list[
            tuple[tuple[float, float, float], tuple[float, float, float], int]
        ] = []
        self.final_samples: list[
            tuple[tuple[float, float, float], tuple[float, float, float], int]
        ] = []

    def confirmConnection(self) -> None:
        self.calls.append(("confirm",))

    def reset(self) -> None:
        self.calls.append(("forbidden-reset",))

    def getMultirotorState(self, *, vehicle_name: str) -> object:
        self.calls.append(("state", vehicle_name))
        samples = {
            "anchor": self.anchor_samples,
            "touchdown": self.touchdown_samples,
            "final": self.final_samples,
        }.get(self.phase)
        if samples:
            position, velocity, landed_state = samples.pop(0)
            self.position = list(position)
            self.velocity = list(velocity)
            self.landed_state = landed_state
        return SimpleNamespace(
            landed_state=self.landed_state,
            kinematics_estimated=SimpleNamespace(
                position=_vector(*self.position),
                linear_velocity=_vector(*self.velocity),
            ),
        )

    def isApiControlEnabled(self, *, vehicle_name: str) -> bool:
        self.calls.append(("api-query", vehicle_name))
        return self.api_enabled

    def simGetCollisionInfo(self, *, vehicle_name: str) -> object:
        self.calls.append(("collision", vehicle_name))
        return SimpleNamespace(has_collided=False)

    def enableApiControl(self, enabled: bool, *, vehicle_name: str) -> None:
        self.calls.append(("api", enabled, vehicle_name))
        self.api_enabled = enabled

    def armDisarm(self, armed: bool, *, vehicle_name: str) -> None:
        self.calls.append(("arm", armed, vehicle_name))
        if not armed:
            self.phase = "final"

    def takeoffAsync(self, *, vehicle_name: str) -> FakeAsync:
        self.calls.append(("takeoff", vehicle_name))

        def update() -> None:
            self.landed_state = 1
            self.phase = "airborne"

        return FakeAsync(update)

    def moveToPositionAsync(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float,
        *,
        timeout_sec: float,
        vehicle_name: str,
    ) -> FakeAsync:
        self.calls.append(
            ("move-position", x, y, z, velocity, timeout_sec, vehicle_name)
        )

        def update() -> None:
            self.position = [x, y, z]
            self.velocity = [0.0, 0.0, 0.0]
            self.phase = "return" if x == 0.0 and y == 0.0 else "anchor"

        return FakeAsync(update)

    def moveByVelocityAsync(
        self,
        x: float,
        y: float,
        z: float,
        duration: float,
        *,
        vehicle_name: str,
    ) -> FakeAsync:
        self.calls.append(("move-velocity", x, y, z, duration, vehicle_name))
        return FakeAsync()

    def hoverAsync(self, *, vehicle_name: str) -> FakeAsync:
        self.calls.append(("hover", vehicle_name))
        self.velocity = [0.0, 0.0, 0.0]
        return FakeAsync()

    def landAsync(self, *, vehicle_name: str) -> FakeAsync:
        self.calls.append(("land", vehicle_name))

        def update() -> None:
            self.position = [0.0, 0.0, 0.57]
            self.velocity = [0.0, 0.0, 0.0]
            self.landed_state = 0
            self.phase = "touchdown"

        return FakeAsync(update)

    def getLidarData(self, lidar_name: str, vehicle_name: str) -> object:
        self.calls.append(("lidar", lidar_name, vehicle_name))
        timestamp = self.timestamps.pop(0) if self.timestamps else 1
        return SimpleNamespace(
            point_cloud=[2.0, 0.0, 0.0],
            time_stamp=timestamp,
            pose=SimpleNamespace(
                position=_vector(0.0, 0.0, 0.0),
                orientation=SimpleNamespace(w_val=1.0, x_val=0.0, y_val=0.0, z_val=0.0),
            ),
        )


def _vector(x: float, y: float, z: float) -> object:
    return SimpleNamespace(x_val=x, y_val=y, z_val=z)


def _environment(
    client: FakeNamedLidarClient, **config_overrides: object
) -> ColosseumLidarUAVEnv:
    return ColosseumLidarUAVEnv(
        ColosseumLidarUAVEnvConfig(
            navigation=ColosseumUAVEnvConfig(workspace_xy_limit=12.0),
            lidar=LidarFeatureConfig(),
            confirm_no_visible_collision=True,
            **config_overrides,
        ),
        client_factory=lambda: client,
        sleep_fn=lambda _seconds: None,
    )


def test_constructor_is_offline_and_observation_shape_is_83() -> None:
    client = FakeNamedLidarClient()
    env = _environment(client)

    assert client.calls == []
    assert env.observation_space.shape == (83,)
    assert env.observation_space.dtype == np.float32


def test_reset_never_calls_global_reset_and_routes_exact_names() -> None:
    client = FakeNamedLidarClient()
    env = _environment(client)

    observation, info = env.reset()

    assert observation.shape == (83,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["sensor_failure"] is False
    assert ("forbidden-reset",) not in client.calls
    assert ("state", "SimpleFlight") in client.calls
    assert ("lidar", "LidarSensor1", "SimpleFlight") in client.calls
    vehicle_calls = [
        call
        for call in client.calls
        if call[0] in {"state", "api-query", "collision", "api", "arm", "takeoff"}
    ]
    assert all(call[-1] == "SimpleFlight" for call in vehicle_calls)


def test_external_scene_points_are_internal_only_and_scene_is_never_mutated() -> None:
    client = FakeNamedLidarClient()
    env = _environment(client)

    observation, info = env.reset(
        options={
            "start_anchor": (4.0, 0.0, -2.1),
            "goal_approach": (13.0, 0.0, -2.1),
        }
    )

    assert observation.shape == (83,)
    assert info["anchor_position"] == pytest.approx((4.0, 0.0, -2.1))
    assert not any(call[0].startswith("sim") for call in client.calls)


def test_third_repeated_scan_truncates_and_locks_steps() -> None:
    client = FakeNamedLidarClient([10, 10, 10, 10])
    env = _environment(client)
    env.reset()

    for expected in (1, 2):
        _obs, _reward, terminated, truncated, info = env.step(
            np.zeros(3, dtype=np.float32)
        )
        assert not terminated
        assert not truncated
        assert info["lidar"]["consecutive_invalid_scans"] == expected

    _obs, _reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert not terminated
    assert truncated
    assert info["sensor_failure"]
    assert info["lidar"]["diagnostics"]["timestamp_status"] == ("stale_limit_exceeded")
    assert ("move-velocity", 0.0, 0.0, 0.0, 0.5, "SimpleFlight") in client.calls
    assert ("hover", "SimpleFlight") in client.calls
    with pytest.raises(RuntimeError, match="requires reset or close"):
        env.step(np.zeros(3, dtype=np.float32))


def test_reset_after_sensor_failure_cleans_up_before_reacquiring_control() -> None:
    client = FakeNamedLidarClient([10, 10, 10, 10, 20])
    env = _environment(client)
    env.reset()
    for _ in range(3):
        env.step(np.zeros(3, dtype=np.float32))

    start = len(client.calls)
    env.reset()
    reset_calls = client.calls[start:]
    landing_index = reset_calls.index(("land", "SimpleFlight"))
    disable_index = reset_calls.index(("api", False, "SimpleFlight"))
    enable_index = reset_calls.index(("api", True, "SimpleFlight"))

    assert landing_index < disable_index < enable_index
    assert ("forbidden-reset",) not in reset_calls


def test_close_uses_named_cleanup_and_no_scene_calls() -> None:
    client = FakeNamedLidarClient()
    env = _environment(client)
    env.reset()

    result = env.close_with_result()

    assert result is not None
    assert not result.safety_critical_failure
    assert ("land", "SimpleFlight") in client.calls
    assert ("arm", False, "SimpleFlight") in client.calls
    assert ("api", False, "SimpleFlight") in client.calls
    assert not any(call[0] in {"destroy", "flush"} for call in client.calls)


def test_reset_scan_exhaustion_raises_after_named_cleanup() -> None:
    client = FakeNamedLidarClient([0] * 10)
    env = _environment(client)

    with pytest.raises(LidarResetError, match="bounded reset retries"):
        env.reset()

    assert ("land", "SimpleFlight") in client.calls
    assert ("arm", False, "SimpleFlight") in client.calls
    assert ("api", False, "SimpleFlight") in client.calls
    assert ("forbidden-reset",) not in client.calls


def test_moving_grounded_vehicle_is_rejected_before_control() -> None:
    client = FakeNamedLidarClient()
    client.velocity = [0.2, 0.0, 0.0]
    env = _environment(client)

    with pytest.raises(RuntimeError, match="stationary"):
        env.reset()

    assert ("api", True, "SimpleFlight") not in client.calls
    assert ("takeoff", "SimpleFlight") not in client.calls


ANCHOR_OPTIONS = {
    "start_anchor": (4.0, 0.0, -2.1),
    "goal_approach": (13.0, 0.0, -2.1),
}


def test_delayed_start_anchor_convergence_requires_stable_hover_samples() -> None:
    client = FakeNamedLidarClient()
    client.anchor_samples = [
        ((3.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
        ((4.0, 0.0, -2.1), (0.2, 0.0, 0.0), 1),
        ((4.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
        ((4.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
        ((4.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
    ]
    env = _environment(client)

    observation, _info = env.reset(options=ANCHOR_OPTIONS)

    evidence = env.lifecycle_evidence["start_anchor_confirmation"]
    assert observation.shape == (83,)
    assert evidence["confirmation_success"]
    assert evidence["confirmation_attempts"] == 5
    assert evidence["consecutive_accepted_samples"] == 3
    assert evidence["position_tolerance_m"] == pytest.approx(0.75)
    assert evidence["measured_speed_m_s"] == 0.0


def test_transient_stable_anchor_sample_is_insufficient() -> None:
    client = FakeNamedLidarClient()
    client.anchor_samples = [
        ((4.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
        ((4.0, 0.0, -2.1), (0.3, 0.0, 0.0), 1),
        ((4.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
        ((4.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
        ((4.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
    ]
    env = _environment(client)

    env.reset(options=ANCHOR_OPTIONS)

    evidence = env.lifecycle_evidence["start_anchor_confirmation"]
    assert evidence["confirmation_attempts"] == 5
    assert evidence["consecutive_accepted_samples"] == 3


def test_start_anchor_timeout_preserves_error_tolerance_and_attempts() -> None:
    client = FakeNamedLidarClient()
    client.anchor_samples = [
        ((2.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
        ((2.5, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
        ((3.0, 0.0, -2.1), (0.0, 0.0, 0.0), 1),
    ]
    env = _environment(
        client,
        start_anchor_confirmation_timeout_s=0.4,
        start_anchor_poll_interval_s=0.2,
    )

    with pytest.raises(RuntimeError, match=r"position_tolerance_m=0\.750000"):
        env.reset(options=ANCHOR_OPTIONS)

    evidence = env.lifecycle_evidence["start_anchor_confirmation"]
    assert evidence["confirmation_attempts"] == 3
    assert evidence["position_error_m"] == pytest.approx(1.0)
    assert not evidence["confirmation_success"]
    assert evidence["rejection_reason"] == ("position outside start-anchor tolerance")


def test_reset_lidar_is_read_only_after_anchor_confirmation() -> None:
    client = FakeNamedLidarClient()
    env = _environment(client)

    env.reset(options=ANCHOR_OPTIONS)

    lidar_index = next(
        index for index, call in enumerate(client.calls) if call[0] == "lidar"
    )
    anchor_hover_index = next(
        index
        for index, call in enumerate(client.calls)
        if call == ("hover", "SimpleFlight")
    )
    state_reads_before_lidar = sum(
        call[0] == "state" for call in client.calls[:lidar_index]
    )
    assert anchor_hover_index < lidar_index
    assert state_reads_before_lidar >= 4
    assert env.lifecycle_evidence["start_anchor_confirmation"]["confirmation_success"]


def test_original_ground_is_captured_before_takeoff_and_used_on_close() -> None:
    client = FakeNamedLidarClient()
    env = _environment(client)
    env.reset(options=ANCHOR_OPTIONS)

    result = env.close_with_result()

    assert result is not None and not result.safety_critical_failure
    assert env.lifecycle_evidence["original_ground_position"] == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.57,
    }
    takeoff_index = client.calls.index(("takeoff", "SimpleFlight"))
    initial_state_index = client.calls.index(("state", "SimpleFlight"))
    return_move_index = next(
        index
        for index, call in enumerate(client.calls)
        if call[0] == "move-position"
        and call[1] == 0.0
        and call[2] == 0.0
        and index > takeoff_index
    )
    land_index = client.calls.index(("land", "SimpleFlight"))
    assert initial_state_index < takeoff_index
    assert return_move_index < land_index
    cleanup = env.lifecycle_evidence["cleanup_attempts"][-1]
    assert cleanup["returned_to_original_ground"]
    assert cleanup["landing_confirmed"]


def test_reset_lidar_failure_returns_to_original_ground() -> None:
    client = FakeNamedLidarClient([0] * 10)
    env = _environment(client)

    with pytest.raises(LidarResetError):
        env.reset(options=ANCHOR_OPTIONS)

    cleanup = env.lifecycle_evidence["cleanup_attempts"][-1]
    assert cleanup["returned_to_original_ground"]
    assert cleanup["landing_confirmed"]
    assert cleanup["api_control_released"]


def test_sensor_failure_cleanup_returns_to_original_ground() -> None:
    client = FakeNamedLidarClient([10, 10, 10, 10])
    env = _environment(client)
    env.reset(options=ANCHOR_OPTIONS)
    for _ in range(3):
        env.step(np.zeros(3, dtype=np.float32))

    result = env.close_with_result()

    assert result is not None and not result.safety_critical_failure
    cleanup = env.lifecycle_evidence["cleanup_attempts"][-1]
    assert cleanup["returned_to_original_ground"]
    assert cleanup["landing_confirmed"]


def test_delayed_landed_enum_and_touchdown_velocity_are_bounded() -> None:
    client = FakeNamedLidarClient()
    client.touchdown_samples = [
        ((0.0, 0.0, 0.57), (0.2, 0.0, 0.0), 1),
        ((0.0, 0.0, 0.57), (0.0, 0.0, 0.0), 1),
        ((0.0, 0.0, 0.57), (0.0, 0.0, 0.0), 1),
        ((0.0, 0.0, 0.57), (0.0, 0.0, 0.0), 1),
    ]
    client.final_samples = [
        ((0.0, 0.0, 0.57), (0.0, 0.0, 0.0), 1),
        ((0.0, 0.0, 0.57), (0.0, 0.0, 0.0), 0),
    ]
    env = _environment(client)
    env.reset(options=ANCHOR_OPTIONS)

    result = env.close_with_result()

    assert result is not None and not result.safety_critical_failure
    cleanup = env.lifecycle_evidence["cleanup_attempts"][-1]
    assert cleanup["touchdown_confirmation_attempts"] == 4
    assert cleanup["touchdown_consecutive_samples"] == 3
    assert cleanup["landed_state_before_disarm"] == 1
    assert cleanup["final_landed_state_attempts"] == 2
    assert cleanup["final_landed_state"] == 0


def test_successful_cleanup_is_idempotent() -> None:
    client = FakeNamedLidarClient()
    env = _environment(client)
    env.reset(options=ANCHOR_OPTIONS)

    first = env.close_with_result()
    second = env.close_with_result()

    assert first is second
    assert sum(call == ("land", "SimpleFlight") for call in client.calls) == 1
    assert len(env.lifecycle_evidence["cleanup_attempts"]) == 1
    assert not env.lifecycle_evidence["recovery_retry_required"]


def test_failed_final_confirmation_may_retry_without_duplicate_flight_actions() -> None:
    client = FakeNamedLidarClient([0] * 10)
    client.final_samples = [
        *((((0.0, 0.0, 0.57), (0.0, 0.0, 0.0), 1),) * 26),
        ((0.0, 0.0, 0.57), (0.0, 0.0, 0.0), 0),
    ]
    env = _environment(client)

    with pytest.raises(LidarResetError):
        env.reset(options=ANCHOR_OPTIONS)
    assert env.last_cleanup_result is not None
    assert env.last_cleanup_result.safety_critical_failure

    recovered = env.close_with_result()

    assert recovered is not None and not recovered.safety_critical_failure
    assert len(env.lifecycle_evidence["cleanup_attempts"]) == 2
    assert env.lifecycle_evidence["recovery_retry_required"]
    assert sum(call == ("land", "SimpleFlight") for call in client.calls) == 1
    assert env.lifecycle_evidence["cleanup_attempts"][1]["actions_attempted"] == ()
