from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from r1_uav_nav.sim import CleanupState, ColosseumClientError, cleanup_after_control
from r1_uav_nav.sim.waypoint_navigation import (
    DEFAULT_FIGURE_EIGHT_SAMPLES,
    FIGURE_EIGHT_ROUTE,
    HORIZONTAL_SQUARE_ROUTE,
    VERTICAL_SQUARE_ROUTE,
    NavigationConfig,
    Position3D,
    RouteParameters,
    WaypointOffset,
    calculate_position_error,
    execute_route,
    execute_route_suite,
    expand_route_selection,
    generate_figure_eight,
    generate_horizontal_square,
    generate_vertical_square,
    read_collision_status,
    resolve_waypoints,
    validate_navigation_config,
    validate_route_clearance,
    validate_selected_routes_for_anchor,
)

DOC_PATH = Path("docs/m12_colosseum_navigation_demo.md")


class FakeAsyncResult:
    def __init__(self, client: "FakeWaypointClient", action: str) -> None:
        self.client = client
        self.action = action

    def join(self) -> None:
        self.client.calls.append(f"{self.action}.join")


class FakeWaypointClient:
    def __init__(
        self,
        position: Position3D | None = None,
        *,
        error_offset: Position3D | None = None,
        state_collision: bool = False,
        fallback_collision: bool | None = None,
        fail_move_index: int | None = None,
        fail_state_read: bool = False,
        fail_land: bool = False,
        no_correction_api: bool = False,
        collision_after_correction: bool = False,
    ) -> None:
        self.position = position or Position3D(0.0, 0.0, -2.0)
        self.error_offset = error_offset or Position3D(0.0, 0.0, 0.0)
        self.state_collision = state_collision
        self.fallback_collision = fallback_collision
        self.fail_move_index = fail_move_index
        self.fail_state_read = fail_state_read
        self.fail_land = fail_land
        self.collision_after_correction = collision_after_correction
        self.calls: list[object] = []
        self.move_count = 0
        if no_correction_api:
            self.moveByVelocityAsync = None  # type: ignore[method-assign]

    def confirmConnection(self) -> None:
        self.calls.append("confirmConnection")

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
        self.move_count += 1
        self.calls.append(("moveToPositionAsync", x, y, z, velocity, timeout_sec))
        if self.fail_move_index == self.move_count:
            raise RuntimeError("movement failed")
        self.position = Position3D(
            x + self.error_offset.x,
            y + self.error_offset.y,
            z + self.error_offset.z,
        )
        return FakeAsyncResult(self, "moveToPositionAsync")

    def moveByVelocityAsync(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
    ) -> FakeAsyncResult:
        self.calls.append(("moveByVelocityAsync", vx, vy, vz, duration))
        self.position = Position3D(
            self.position.x + vx * duration,
            self.position.y + vy * duration,
            self.position.z + vz * duration,
        )
        if self.collision_after_correction:
            self.state_collision = True
        return FakeAsyncResult(self, "moveByVelocityAsync")

    def hoverAsync(self) -> FakeAsyncResult:
        self.calls.append("hoverAsync")
        return FakeAsyncResult(self, "hoverAsync")

    def landAsync(self) -> FakeAsyncResult:
        self.calls.append("landAsync")
        if self.fail_land:
            raise RuntimeError("landing failed")
        return FakeAsyncResult(self, "landAsync")

    def getMultirotorState(self) -> object:
        self.calls.append("getMultirotorState")
        if self.fail_state_read:
            raise RuntimeError("state read failed")
        collision = SimpleNamespace(has_collided=self.state_collision)
        return _state(self.position, collision=collision)

    def simGetCollisionInfo(self) -> object:
        self.calls.append("simGetCollisionInfo")
        return SimpleNamespace(has_collided=bool(self.fallback_collision))


class FakeFallbackCollisionClient(FakeWaypointClient):
    def getMultirotorState(self) -> object:
        self.calls.append("getMultirotorState")
        return _state(self.position, collision=None)


def test_horizontal_square_offsets_are_constant_altitude_and_return_to_anchor() -> None:
    offsets = generate_horizontal_square(2.0)

    assert offsets == (
        WaypointOffset(0.0, 0.0, 0.0),
        WaypointOffset(2.0, 0.0, 0.0),
        WaypointOffset(2.0, 2.0, 0.0),
        WaypointOffset(0.0, 2.0, 0.0),
        WaypointOffset(0.0, 0.0, 0.0),
    )
    assert {offset.dz for offset in offsets} == {0.0}


def test_figure_eight_has_exact_default_count_center_crossing_and_endpoints() -> None:
    offsets = generate_figure_eight(3.0, 2.0, DEFAULT_FIGURE_EIGHT_SAMPLES)

    assert len(offsets) == 13
    assert offsets[0] == WaypointOffset(0.0, 0.0, 0.0)
    assert offsets[-1] == WaypointOffset(0.0, 0.0, 0.0)
    assert offsets[6].dx == pytest.approx(0.0)
    assert offsets[6].dy == pytest.approx(0.0)
    assert {offset.dz for offset in offsets} == {0.0}
    assert max(offset.dy for offset in offsets) > 0.0
    assert min(offset.dy for offset in offsets) < 0.0


def test_vertical_square_keeps_y_constant_changes_x_and_z_and_returns() -> None:
    offsets = generate_vertical_square(2.0, 1.0)

    assert offsets == (
        WaypointOffset(0.0, 0.0, 0.0),
        WaypointOffset(2.0, 0.0, 0.0),
        WaypointOffset(2.0, 0.0, -1.0),
        WaypointOffset(0.0, 0.0, -1.0),
        WaypointOffset(0.0, 0.0, 0.0),
    )
    assert {offset.dy for offset in offsets} == {0.0}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: generate_horizontal_square(0.0),
        lambda: generate_figure_eight(0.0, 1.0, 21),
        lambda: generate_figure_eight(1.0, 0.0, 21),
        lambda: generate_figure_eight(1.0, 1.0, 7),
        lambda: generate_figure_eight(1.0, 1.0, 65),
        lambda: generate_vertical_square(0.0, 1.0),
        lambda: generate_vertical_square(1.0, 0.0),
    ],
)
def test_invalid_route_generation_values_raise(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()


def test_resolve_waypoints_uses_arbitrary_nonzero_anchor_without_mutation() -> None:
    anchor = Position3D(10.0, -4.0, 0.560)
    offsets = (WaypointOffset(1.0, 2.0, -3.0),)

    resolved = resolve_waypoints(anchor, offsets)

    assert resolved == (Position3D(11.0, -2.0, -2.44),)
    assert anchor == Position3D(10.0, -4.0, 0.560)
    assert offsets == (WaypointOffset(1.0, 2.0, -3.0),)


def test_ground_clearance_uses_measured_ground_reference_z() -> None:
    ground_reference_z = 0.560
    waypoint = Position3D(0.0, 0.0, -0.940)

    validate_route_clearance([waypoint], ground_reference_z, 1.5)

    with pytest.raises(ValueError, match="minimum ground clearance"):
        validate_route_clearance([waypoint], ground_reference_z, 1.6)


def test_selected_route_validation_uses_initial_z_relative_anchor() -> None:
    initial_position = Position3D(0.0, 0.0, 0.560)
    config = NavigationConfig(
        route=VERTICAL_SQUARE_ROUTE,
        anchor_altitude=2.0,
        min_ground_clearance=1.0,
        route_parameters=RouteParameters(vertical_square_height=1.0),
    )
    target_anchor = Position3D(
        initial_position.x,
        initial_position.y,
        initial_position.z - config.anchor_altitude,
    )

    validate_selected_routes_for_anchor(config, target_anchor, initial_position.z)


def test_calculate_position_error_is_euclidean_3d() -> None:
    assert calculate_position_error(
        Position3D(0.0, 0.0, 0.0),
        Position3D(1.0, 2.0, 2.0),
    ) == pytest.approx(3.0)


def test_collision_status_prefers_state_then_client_fallback_then_unavailable() -> None:
    state_collision_client = FakeWaypointClient(state_collision=True)
    assert read_collision_status(
        state_collision_client,
        state_collision_client.getMultirotorState(),
    )

    fallback_client = FakeFallbackCollisionClient(fallback_collision=True)
    assert read_collision_status(fallback_client, fallback_client.getMultirotorState())

    unavailable_state = _state(Position3D(0.0, 0.0, -2.0), collision=None)
    assert not read_collision_status(SimpleNamespace(), unavailable_state)


def test_execute_route_moves_in_order_joins_reads_state_and_reports_progress() -> None:
    client = FakeWaypointClient()
    progress: list[object] = []
    config = NavigationConfig(route=HORIZONTAL_SQUARE_ROUTE)

    result = execute_route(
        client,
        HORIZONTAL_SQUARE_ROUTE,
        Position3D(0.0, 0.0, -2.0),
        ground_reference_z=0.560,
        config=config,
        progress_callback=progress.append,
    )

    assert result.num_waypoints_requested == 5
    assert result.num_waypoints_completed == 5
    assert result.returned_to_anchor
    assert len(progress) == 5
    assert sum(call == "moveToPositionAsync.join" for call in client.calls) == 5
    assert sum(call == "getMultirotorState" for call in client.calls) == 5


def test_execute_route_fails_on_excessive_position_error() -> None:
    client = FakeWaypointClient(error_offset=Position3D(1.0, 0.0, 0.0))
    config = NavigationConfig(waypoint_tolerance=0.5, correction_attempts=0)

    with pytest.raises(ColosseumClientError, match="Waypoint tolerance exceeded"):
        execute_route(
            client,
            HORIZONTAL_SQUARE_ROUTE,
            Position3D(0.0, 0.0, -2.0),
            ground_reference_z=0.560,
            config=config,
        )


def test_execute_route_fails_on_collision_and_state_read_failure() -> None:
    with pytest.raises(ColosseumClientError, match="Collision detected"):
        execute_route(
            FakeWaypointClient(state_collision=True),
            HORIZONTAL_SQUARE_ROUTE,
            Position3D(0.0, 0.0, -2.0),
            ground_reference_z=0.560,
            config=NavigationConfig(),
        )

    with pytest.raises(ColosseumClientError, match="Could not read"):
        execute_route(
            FakeWaypointClient(fail_state_read=True),
            HORIZONTAL_SQUARE_ROUTE,
            Position3D(0.0, 0.0, -2.0),
            ground_reference_z=0.560,
            config=NavigationConfig(),
        )


def test_route_suite_executes_in_order_hovers_and_stops_on_failure() -> None:
    client = FakeWaypointClient(fail_move_index=6)
    sleeps: list[float] = []
    config = NavigationConfig(route="all", hover_between_routes=1.25)

    with pytest.raises(ColosseumClientError, match="moveToPositionAsync"):
        execute_route_suite(
            client,
            expand_route_selection("all"),
            Position3D(0.0, 0.0, -2.0),
            ground_reference_z=0.560,
            config=config,
            sleep_fn=sleeps.append,
        )

    assert "hoverAsync" in client.calls
    assert "hoverAsync.join" in client.calls
    assert sleeps == [1.25]


def test_route_suite_accepts_zero_hover_without_sleeping() -> None:
    client = FakeWaypointClient()
    sleeps: list[float] = []
    config = NavigationConfig(route="all", hover_between_routes=0.0)

    results = execute_route_suite(
        client,
        (HORIZONTAL_SQUARE_ROUTE, VERTICAL_SQUARE_ROUTE),
        Position3D(0.0, 0.0, -2.0),
        ground_reference_z=0.560,
        config=config,
        sleep_fn=sleeps.append,
    )

    assert [result.route_name for result in results] == [
        HORIZONTAL_SQUARE_ROUTE,
        VERTICAL_SQUARE_ROUTE,
    ]
    assert sleeps == []


def test_cleanup_is_state_aware_and_reports_safety_critical_failures() -> None:
    no_stage_client = FakeWaypointClient()
    no_stage_result = cleanup_after_control(no_stage_client, CleanupState())

    assert no_stage_result.actions_attempted == ()

    fail_land_client = FakeWaypointClient(fail_land=True)
    result = cleanup_after_control(
        fail_land_client,
        CleanupState(api_control_enabled=True, armed=True, takeoff_attempted=True),
    )

    assert result.safety_critical_failure
    assert any("landAsync failed" in error for error in result.errors)
    assert "disarmed drone" in result.actions_succeeded
    assert "disabled API control" in result.actions_succeeded


def test_navigation_config_validation_and_cli_config_conversion() -> None:
    script = _load_waypoint_script_module()
    args = script.parse_args([])

    config = script.build_navigation_config(args)

    assert config.route == HORIZONTAL_SQUARE_ROUTE
    assert config.velocity == pytest.approx(0.5)
    assert config.waypoint_tolerance == pytest.approx(0.5)
    assert config.waypoint_timeout == pytest.approx(20.0)
    assert config.route_parameters.figure_eight_x_scale == pytest.approx(3.0)
    assert config.route_parameters.figure_eight_y_scale == pytest.approx(2.0)
    assert config.route_parameters.figure_eight_samples == 13

    with pytest.raises(ValueError, match="hover_between_routes"):
        validate_navigation_config(NavigationConfig(hover_between_routes=-1.0))


def test_cli_accepts_valid_routes_and_rejects_invalid_route() -> None:
    script = _load_waypoint_script_module()

    parsed_args = script.parse_args(["--route", FIGURE_EIGHT_ROUTE])

    assert parsed_args.route == FIGURE_EIGHT_ROUTE
    with pytest.raises(SystemExit):
        script.parse_args(["--route", "spiral"])


def test_run_waypoint_demo_returns_nonzero_on_cleanup_failure(
    monkeypatch: object,
) -> None:
    script = _load_waypoint_script_module()
    client = FakeWaypointClient(position=Position3D(0.0, 0.0, 0.560), fail_land=True)

    monkeypatch.setattr(
        script,
        "import_colosseum_client_module",
        lambda _module_name: SimpleNamespace(MultirotorClient=lambda: client),
    )

    exit_code = script.run_waypoint_demo(script.parse_args([]), sleep_fn=lambda _: None)

    assert exit_code == 1


def test_waypoint_timeout_is_passed_to_move_to_position() -> None:
    client = FakeWaypointClient()
    config = NavigationConfig(waypoint_timeout=12.5)

    execute_route(
        client,
        HORIZONTAL_SQUARE_ROUTE,
        Position3D(0.0, 0.0, -2.0),
        ground_reference_z=0.560,
        config=config,
    )

    move_calls = [
        call for call in client.calls if _is_call(call, "moveToPositionAsync")
    ]
    assert move_calls[0] == ("moveToPositionAsync", 0.0, 0.0, -2.0, 0.5, 12.5)


def test_primary_movement_inside_tolerance_succeeds_without_correction() -> None:
    client = FakeWaypointClient(error_offset=Position3D(0.1, 0.0, 0.0))
    config = NavigationConfig(waypoint_tolerance=0.5)

    result = execute_route(
        client,
        HORIZONTAL_SQUARE_ROUTE,
        Position3D(0.0, 0.0, -2.0),
        ground_reference_z=0.560,
        config=config,
    )

    assert result.num_waypoints_completed == 5
    assert not any(_is_call(call, "moveByVelocityAsync") for call in client.calls)


def test_excessive_initial_error_triggers_bounded_correction() -> None:
    client = FakeWaypointClient(error_offset=Position3D(1.0, 1.0, 1.0))
    progress = []
    config = NavigationConfig(
        waypoint_tolerance=1.0,
        correction_speed=0.25,
        correction_max_duration=2.0,
        correction_settle_delay=0.0,
    )

    result = execute_route(
        client,
        HORIZONTAL_SQUARE_ROUTE,
        Position3D(0.0, 0.0, -2.0),
        ground_reference_z=0.560,
        config=config,
        progress_callback=progress.append,
    )

    correction_call = next(
        call for call in client.calls if _is_call(call, "moveByVelocityAsync")
    )
    _, vx, vy, vz, duration = correction_call
    assert vx < 0.0
    assert vy < 0.0
    assert vz < 0.0
    assert (vx**2 + vy**2 + vz**2) ** 0.5 <= 0.250001
    assert duration <= 2.000001
    assert "moveByVelocityAsync.join" in client.calls
    assert "hoverAsync.join" in client.calls
    assert any(item.correction_attempt == 1 for item in progress)
    assert result.max_position_error > result.final_position_error


def test_correction_exhausts_attempts_and_fails_with_attempt_count() -> None:
    client = FakeWaypointClient(error_offset=Position3D(3.0, 0.0, 0.0))
    config = NavigationConfig(
        waypoint_tolerance=0.5,
        correction_attempts=3,
        correction_settle_delay=0.0,
    )

    with pytest.raises(ColosseumClientError, match="correction_attempts=3"):
        execute_route(
            client,
            HORIZONTAL_SQUARE_ROUTE,
            Position3D(0.0, 0.0, -2.0),
            ground_reference_z=0.560,
            config=config,
        )

    assert sum(_is_call(call, "moveByVelocityAsync") for call in client.calls) == 3


def test_collision_during_correction_fails() -> None:
    client = FakeWaypointClient(
        error_offset=Position3D(1.0, 0.0, 0.0),
        collision_after_correction=True,
    )

    with pytest.raises(ColosseumClientError, match="Collision detected"):
        execute_route(
            client,
            HORIZONTAL_SQUARE_ROUTE,
            Position3D(0.0, 0.0, -2.0),
            ground_reference_z=0.560,
            config=NavigationConfig(correction_settle_delay=0.0),
        )


def test_missing_move_by_velocity_fails_only_when_correction_needed() -> None:
    inside_tolerance_client = FakeWaypointClient(
        error_offset=Position3D(0.1, 0.0, 0.0),
        no_correction_api=True,
    )
    execute_route(
        inside_tolerance_client,
        HORIZONTAL_SQUARE_ROUTE,
        Position3D(0.0, 0.0, -2.0),
        ground_reference_z=0.560,
        config=NavigationConfig(),
    )

    correction_needed_client = FakeWaypointClient(
        error_offset=Position3D(1.0, 0.0, 0.0),
        no_correction_api=True,
    )
    with pytest.raises(ColosseumClientError, match="moveByVelocityAsync"):
        execute_route(
            correction_needed_client,
            HORIZONTAL_SQUARE_ROUTE,
            Position3D(0.0, 0.0, -2.0),
            ground_reference_z=0.560,
            config=NavigationConfig(),
        )


def test_keyboard_interrupt_returns_nonzero(monkeypatch: object) -> None:
    script = _load_waypoint_script_module()
    client = FakeWaypointClient(position=Position3D(0.0, 0.0, 0.560))

    def raise_keyboard_interrupt() -> FakeAsyncResult:
        raise KeyboardInterrupt

    client.takeoffAsync = raise_keyboard_interrupt  # type: ignore[method-assign]
    monkeypatch.setattr(
        script,
        "import_colosseum_client_module",
        lambda _module_name: SimpleNamespace(MultirotorClient=lambda: client),
    )

    exit_code = script.run_waypoint_demo(script.parse_args([]), sleep_fn=lambda _: None)

    assert exit_code == 1
    assert ("enableApiControl", False) in client.calls


def test_m12_colosseum_navigation_demo_doc_mentions_required_topics() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "horizontal-square" in doc_text
    assert "figure-eight" in doc_text
    assert "vertical-square" in doc_text
    assert "scripted" in doc_text
    assert "not reinforcement-learning" in doc_text
    assert "target_anchor_z = initial_position.z - anchor_altitude" in doc_text
    assert "python scripts\\run_colosseum_waypoint_demo.py --route all" in doc_text


def _state(position: Position3D, collision: object | None) -> object:
    kinematics = SimpleNamespace(
        position=SimpleNamespace(
            x_val=position.x,
            y_val=position.y,
            z_val=position.z,
        )
    )
    state = SimpleNamespace(kinematics_estimated=kinematics)
    if collision is not None:
        state.collision = collision
    return state


def _is_call(call: object, method_name: str) -> bool:
    return isinstance(call, tuple) and call[0] == method_name


def _load_waypoint_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_colosseum_waypoint_demo.py"
    )
    spec = spec_from_file_location("run_colosseum_waypoint_demo", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
