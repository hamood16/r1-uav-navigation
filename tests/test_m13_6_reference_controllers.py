from __future__ import annotations

import random
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pytest

from r1_uav_nav.evaluation.reference_controllers import (
    ControllerError,
    ControllerPrivilege,
    ControllerStateSpec,
    ControllerStepInput,
    DirectGoalController,
    DirectGoalControllerConfig,
    OracleWaypointController,
    OracleWaypointControllerConfig,
    RandomController,
    RandomControllerConfig,
    ReferenceController,
    compress_reference_path,
    validate_controller_action,
)
from r1_uav_nav.sim.static_course import (
    generate_solvable_course,
    load_course_suite_config,
)

ROOT = Path(__file__).resolve().parents[1]
COURSE_CONFIG = ROOT / "configs" / "planning" / "m13_3_voxel_astar.yaml"


class ExplodingInfo(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise AssertionError(f"privileged info was accessed: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("privileged info was iterated")

    def __len__(self) -> int:
        raise AssertionError("privileged info length was requested")


@pytest.fixture(scope="module")
def medium_course():
    suite = load_course_suite_config(COURSE_CONFIG)
    return generate_solvable_course(suite, "medium", 2100, repository_root=ROOT)


@pytest.fixture
def state_spec() -> ControllerStateSpec:
    return ControllerStateSpec(
        position_scales_m=(20.0, 20.0, 6.0),
        goal_displacement_scales_m=(40.0, 40.0, 12.0),
        velocity_scales_m_s=(0.5, 0.5, 0.4),
    )


def _observation() -> np.ndarray:
    observation = np.zeros(83, dtype=np.float32)
    observation[82] = 1.0
    return observation


def test_random_controller_protocol_bounds_and_reproducibility() -> None:
    first = RandomController(RandomControllerConfig(seed=1360))
    second = RandomController(RandomControllerConfig(seed=1360))
    step = ControllerStepInput(_observation(), 0)
    assert isinstance(first, ReferenceController)
    first_actions = [first.act(step).action for _ in range(4)]
    second_actions = [second.act(step).action for _ in range(4)]
    for actual, expected in zip(first_actions, second_actions, strict=True):
        assert actual.shape == (3,)
        assert actual.dtype == np.float32
        assert np.all(np.isfinite(actual))
        assert np.all(actual >= -1.0)
        assert np.all(actual <= 1.0)
        np.testing.assert_array_equal(actual, expected)


def test_random_controller_does_not_mutate_global_rng_state() -> None:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    controller = RandomController(RandomControllerConfig(seed=4))
    controller.act(ControllerStepInput(_observation(), 0))
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]


def test_direct_controller_uses_navigation_prefix_only(
    state_spec: ControllerStateSpec,
) -> None:
    observation = _observation()
    observation[3] = 0.25
    controller = DirectGoalController(DirectGoalControllerConfig(), state_spec)
    decision = controller.act(ControllerStepInput(observation, 0, ExplodingInfo()))
    assert decision.action[0] > 0.0
    assert decision.action[1] == pytest.approx(0.0)
    assert decision.action[2] == pytest.approx(0.0)
    assert controller.privilege is ControllerPrivilege.NONE


def test_direct_controller_damps_velocity_and_slows_near_goal(
    state_spec: ControllerStateSpec,
) -> None:
    controller = DirectGoalController(DirectGoalControllerConfig(), state_spec)
    far = _observation()
    far[3] = 0.25
    near = _observation()
    near[3] = 0.01
    near[6] = 0.5
    far_action = controller.act(ControllerStepInput(far, 0)).action
    near_action = controller.act(ControllerStepInput(near, 1)).action
    assert abs(float(near_action[0])) < abs(float(far_action[0]))
    zero = _observation()
    np.testing.assert_array_equal(
        controller.act(ControllerStepInput(zero, 2)).action,
        np.zeros(3, dtype=np.float32),
    )


def test_action_validation_rejects_invalid_shape_bounds_and_values() -> None:
    with pytest.raises(ControllerError, match="shape"):
        validate_controller_action(np.zeros(2))
    with pytest.raises(ControllerError, match="finite"):
        validate_controller_action(np.asarray([0.0, np.nan, 0.0]))
    with pytest.raises(ControllerError, match="within"):
        validate_controller_action(np.asarray([1.1, 0.0, 0.0]))


def test_oracle_rejects_empty_or_coincident_routes() -> None:
    with pytest.raises(ControllerError, match="at least two"):
        compress_reference_path((), maximum_segment_length_m=1.0)
    with pytest.raises(ControllerError, match="distinct"):
        compress_reference_path(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            maximum_segment_length_m=1.0,
        )


def test_oracle_compression_preserves_endpoints_turns_and_segment_limit() -> None:
    route = compress_reference_path(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
        ),
        maximum_segment_length_m=1.0,
    )
    assert route[0] == (0.0, 0.0, 0.0)
    assert (2.0, 0.0, 0.0) in route
    assert route[-1] == (2.0, 2.0, 0.0)
    for first, second in zip(route, route[1:], strict=False):
        assert np.linalg.norm(np.asarray(second) - np.asarray(first)) <= 1.0 + 1e-9


def test_oracle_revalidates_real_medium_path_and_is_deterministic(
    medium_course: Any,
    state_spec: ControllerStateSpec,
) -> None:
    config = OracleWaypointControllerConfig()
    path = medium_course.result.path_result.reference_path
    first = OracleWaypointController(
        config,
        state_spec,
        reference_path=path,
        grid=medium_course.grid,
    )
    second = OracleWaypointController(
        config,
        state_spec,
        reference_path=path,
        grid=medium_course.grid,
    )
    assert first.route == second.route
    assert first.route_digest == second.route_digest
    assert first.privilege is ControllerPrivilege.REFERENCE_PATH
    assert first.waypoint_count >= 2


def test_oracle_waypoint_advancement(
    medium_course: Any,
    state_spec: ControllerStateSpec,
) -> None:
    controller = OracleWaypointController(
        OracleWaypointControllerConfig(),
        state_spec,
        reference_path=medium_course.result.path_result.reference_path,
        grid=medium_course.grid,
    )
    first_index = controller.waypoint_index
    first_offset = controller.route_offsets[first_index]
    observation = _observation()
    observation[0:3] = np.asarray(first_offset) / np.asarray(
        state_spec.position_scales_m
    )
    decision = controller.act(ControllerStepInput(observation, 0))
    assert controller.waypoint_index > first_index
    assert controller.completed_waypoints >= 1
    assert decision.waypoint_index == controller.waypoint_index


def test_privilege_labels_and_configuration_are_immutable(
    state_spec: ControllerStateSpec,
) -> None:
    config = DirectGoalControllerConfig()
    controller = DirectGoalController(config, state_spec)
    assert controller.privilege.value == "non_privileged"
    with pytest.raises(FrozenInstanceError):
        config.cruise_speed_m_s = 9.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        controller.privilege = ControllerPrivilege.REFERENCE_PATH  # type: ignore[misc]
