"""Colosseum adapter and offline-first CLI tests for M13.4."""

from __future__ import annotations

import builtins
import importlib.util
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r1_uav_nav.sim.colosseum_client import CleanupResult
from r1_uav_nav.sim.colosseum_lidar import (
    GroundedLidarFeatureProbeConfig,
    build_easy_known_geometry_evidence,
    build_live_report,
    lidar_scan_input_from_data,
    probe_grounded_lidar_features,
)
from r1_uav_nav.sim.colosseum_scene import (
    SceneCleanupResult,
    StartAnchorCallbackTimeout,
    StartAnchorReadOnlyContext,
)
from r1_uav_nav.sim.lidar_features import LidarFeatureConfig
from r1_uav_nav.sim.lidar_live_validation import (
    LiveAuthorizationEvidence,
    LiveExecutionResult,
    collect_known_geometry_samples,
    execute_airborne_smoke,
    execute_known_geometry_live,
    prepare_lidar_live_run,
)
from r1_uav_nav.sim.static_course import (
    generate_solvable_course,
    load_course_suite_config,
)
from r1_uav_nav.sim.waypoint_navigation import Position3D

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/check_lidar_features.py"
COURSE_CONFIG = ROOT / "configs/planning/m13_3_voxel_astar.yaml"
FEATURE_CONFIG = ROOT / "configs/sensing/m13_4_lidar_features.yaml"
ASSET_CATALOG = ROOT / "configs/scenes/m13_2_assets.yaml"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_lidar_features", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vector(x: float, y: float, z: float) -> object:
    return SimpleNamespace(x_val=x, y_val=y, z_val=z)


class FakeLandedState:
    Landed = 0
    Flying = 1


FAKE_MODULE = SimpleNamespace(LandedState=FakeLandedState)


class FakeGroundedClient:
    def __init__(self, point_clouds: list[list[float]] | None = None) -> None:
        self.calls: list[tuple] = []
        self.timestamp = 0
        self.point_clouds = point_clouds or [[2.0, 0.0, 0.0]]

    def confirmConnection(self) -> None:
        self.calls.append(("confirm",))

    def getSettingsString(self) -> str:
        self.calls.append(("settings",))
        return json.dumps(
            {
                "SettingsVersion": 1.2,
                "SimMode": "Multirotor",
                "ApiServerPort": 41451,
                "Vehicles": {
                    "SimpleFlight": {
                        "VehicleType": "SimpleFlight",
                        "AutoCreate": True,
                        "DefaultVehicleState": "Inactive",
                        "Sensors": {
                            "LidarSensor1": {
                                "SensorType": 6,
                                "Enabled": True,
                                "NumberOfChannels": 16,
                                "Range": 20,
                                "PointsPerSecond": 100000,
                                "RotationsPerSecond": 10,
                                "HorizontalFOVStart": 0,
                                "HorizontalFOVEnd": 359,
                                "VerticalFOVUpper": 10,
                                "VerticalFOVLower": -30,
                                "X": 0,
                                "Y": 0,
                                "Z": 0,
                                "Roll": 0,
                                "Pitch": 0,
                                "Yaw": 0,
                                "DrawDebugPoints": False,
                                "DataFrame": "SensorLocalFrame",
                                "ExternalController": False,
                            }
                        },
                    }
                },
            }
        )

    def getMultirotorState(self, *, vehicle_name: str) -> object:
        self.calls.append(("state", vehicle_name))
        return SimpleNamespace(
            landed_state=FakeLandedState.Landed,
            kinematics_estimated=SimpleNamespace(
                position=_vector(0.0, 0.0, 0.57),
                linear_velocity=_vector(0.0, 0.0, 0.0),
            ),
        )

    def isApiControlEnabled(self, *, vehicle_name: str) -> bool:
        self.calls.append(("api-query", vehicle_name))
        return False

    def simGetCollisionInfo(self, *, vehicle_name: str) -> object:
        self.calls.append(("collision", vehicle_name))
        return SimpleNamespace(
            has_collided=False,
            object_name="",
            object_id=-1,
            time_stamp=0,
            penetration_depth=0.0,
            impact_point=_vector(0.0, 0.0, 0.0),
            position=_vector(0.0, 0.0, 0.57),
            normal=_vector(0.0, 0.0, -1.0),
        )

    def getLidarData(self, lidar_name: str, vehicle_name: str) -> object:
        self.calls.append(("lidar", lidar_name, vehicle_name))
        self.timestamp += 1
        index = min(self.timestamp - 1, len(self.point_clouds) - 1)
        return SimpleNamespace(
            point_cloud=self.point_clouds[index],
            time_stamp=self.timestamp,
            pose=SimpleNamespace(
                position=_vector(0.0, 0.0, 0.0),
                orientation=SimpleNamespace(
                    w_val=1.0,
                    x_val=0.0,
                    y_val=0.0,
                    z_val=0.0,
                ),
            ),
        )


def _all_live_authorizations() -> LiveAuthorizationEvidence:
    return LiveAuthorizationEvidence(
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


def _live_cli_arguments(mode: str) -> list[str]:
    profile, seed = (
        ("easy", "1100") if mode == "known-geometry-live" else ("medium", "2100")
    )
    count_argument = (
        ["--scan-count", "3", "--scan-interval", "0"]
        if mode == "known-geometry-live"
        else ["--step-count", "2"]
    )
    return [
        mode,
        "--course-profile",
        profile,
        "--course-seed",
        seed,
        "--vehicle-name",
        "SimpleFlight",
        "--lidar-name",
        "LidarSensor1",
        *count_argument,
        "--allow-live-rpc",
        "--allow-scene-mutation",
        "--confirm-scene-area-clear",
        "--confirm-no-visible-collision",
        "--allow-debug-markers",
        "--allow-marker-flush",
        "--allow-flight",
        "--allow-start-positioning",
        "--confirm-clear-airspace",
    ]


def _scan_point(
    range_m: float = 5.145347554864271,
    *,
    timestamp: int,
    horizontal_degrees: float = -67.5,
    elevation_degrees: float = -7.5,
) -> object:
    horizontal = math.radians(horizontal_degrees)
    elevation = math.radians(elevation_degrees)
    horizontal_range = range_m * math.cos(elevation)
    point = [
        horizontal_range * math.cos(horizontal),
        horizontal_range * math.sin(horizontal),
        -range_m * math.sin(elevation),
    ]
    return SimpleNamespace(
        point_cloud=point,
        time_stamp=timestamp,
        pose=SimpleNamespace(
            position=_vector(0.0, 0.0, 0.0),
            orientation=SimpleNamespace(
                w_val=1.0,
                x_val=0.0,
                y_val=0.0,
                z_val=0.0,
            ),
        ),
    )


def _context(scans: list[object]) -> StartAnchorReadOnlyContext:
    queue = iter(scans)
    return StartAnchorReadOnlyContext(
        "SimpleFlight",
        "LidarSensor1",
        max(20, len(scans)),
        100.0,
        lambda: next(queue),
        lambda: 0.0,
    )


@pytest.fixture(scope="module")
def prepared_known_geometry():
    return prepare_lidar_live_run(
        mode="known-geometry-live",
        repository_root=ROOT,
        feature_config=LidarFeatureConfig(),
        course_config_path=COURSE_CONFIG,
        asset_catalog_path=ASSET_CATALOG,
        output_dir=ROOT / "results/reports/m13/lidar",
        profile_id="easy",
        base_seed=1100,
        vehicle_name="SimpleFlight",
        lidar_name="LidarSensor1",
        authorizations=_all_live_authorizations(),
        sample_count=3,
        sample_interval_s=0.0,
    )


@pytest.fixture(scope="module")
def prepared_airborne_smoke():
    return prepare_lidar_live_run(
        mode="airborne-smoke",
        repository_root=ROOT,
        feature_config=LidarFeatureConfig(),
        course_config_path=COURSE_CONFIG,
        asset_catalog_path=ASSET_CATALOG,
        output_dir=ROOT / "results/reports/m13/lidar",
        profile_id="medium",
        base_seed=2100,
        vehicle_name="SimpleFlight",
        lidar_name="LidarSensor1",
        authorizations=_all_live_authorizations(),
        sample_count=2,
    )


def test_adapter_copies_cloud_and_keeps_invalid_pose_separate() -> None:
    cloud = [1.0, 0.0, 0.0]
    data = SimpleNamespace(point_cloud=cloud, time_stamp=7, pose=None)

    scan = lidar_scan_input_from_data(data)
    cloud[0] = 9.0

    assert scan.point_cloud == (1.0, 0.0, 0.0)
    assert scan.timestamp == 7
    assert scan.sensor_position is None


def test_easy_known_geometry_evidence_is_locked() -> None:
    suite = load_course_suite_config(COURSE_CONFIG)
    course = generate_solvable_course(suite, "easy", 1100, repository_root=ROOT)

    evidence = build_easy_known_geometry_evidence(course, LidarFeatureConfig())

    assert evidence["comparison_object"] == "obstacle-000"
    assert evidence["expected_horizontal_sector"] == 7
    assert evidence["expected_elevation_band"] == 1
    assert evidence["expected_nearest_surface_distance_m"] == pytest.approx(
        5.14535, abs=0.01
    )
    assert evidence["clear_comparison_horizontal_sector"] == 14


def test_live_authorization_is_rejected_before_client_import(tmp_path: Path) -> None:
    script = _load_script()
    args = script.parse_args(
        [
            "--config",
            str(FEATURE_CONFIG),
            "--output-dir",
            str(tmp_path),
            "grounded",
            "--vehicle-name",
            "SimpleFlight",
            "--lidar-name",
            "LidarSensor1",
            "--confirm-no-visible-collision",
        ]
    )
    imported = False

    def loader(_name: str) -> object:
        nonlocal imported
        imported = True
        raise AssertionError("client import must not occur")

    with pytest.raises(ValueError, match="allow-live-rpc"):
        script.run(args, repository_root=ROOT, client_module_loader=loader)
    assert not imported


def test_grounded_cli_writes_ignored_shape_evidence_without_raw_cloud(
    tmp_path: Path,
) -> None:
    script = _load_script()
    client = FakeGroundedClient()
    output = ROOT / "results/reports/m13/lidar/test-grounded"
    before = set(output.glob("m13_4_grounded_*.json"))
    args = script.parse_args(
        [
            "--config",
            str(FEATURE_CONFIG),
            "--output-dir",
            str(output),
            "grounded",
            "--vehicle-name",
            "SimpleFlight",
            "--lidar-name",
            "LidarSensor1",
            "--scan-count",
            "3",
            "--scan-interval",
            "0",
            "--allow-live-rpc",
            "--confirm-no-visible-collision",
        ]
    )

    result = script.run(
        args,
        repository_root=ROOT,
        client_module_loader=lambda _name: FAKE_MODULE,
        client_factory=lambda _module: client,
        sleep_fn=lambda _seconds: None,
    )

    assert result == 0
    created = set(output.glob("m13_4_grounded_*.json")) - before
    assert len(created) == 1
    report_path = created.pop()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["success"]
    assert report["feature_shape"] == [72]
    assert report["data"]["valid_scan_count"] == 3
    assert report["data"]["acceptance_failures"] == []
    assert all(report["data"]["acceptance_checks"].values())
    summary = report["data"]["elevation_diagnostics_summary"]
    assert summary["observed_minimum_elevation_degrees"] == 0.0
    assert summary["observed_maximum_elevation_degrees"] == 0.0
    assert summary["total_finite_point_count"] == 3
    assert summary["configured_elevation_boundary_tolerance_degrees"] == pytest.approx(
        0.0001
    )
    assert "point_cloud" not in report_path.read_text(encoding="utf-8")
    assert all(
        call[-1] == "SimpleFlight"
        for call in client.calls
        if call[0] in {"state", "api-query", "collision"}
    )
    assert all(
        call[1:] == ("LidarSensor1", "SimpleFlight")
        for call in client.calls
        if call[0] == "lidar"
    )


@pytest.mark.parametrize(
    "point_clouds",
    [
        [[2.0, 0.0, 0.0], [1.0, 0.0, -1.0], [2.0, 0.0, 0.0]],
        [[1.0, 0.0, -1.0]] * 3,
    ],
)
def test_failed_grounded_scans_set_report_failure_and_acceptance_checks(
    point_clouds: list[list[float]],
) -> None:
    config = LidarFeatureConfig()
    data = probe_grounded_lidar_features(
        FakeGroundedClient(point_clouds),
        FAKE_MODULE,  # type: ignore[arg-type]
        GroundedLidarFeatureProbeConfig(
            config,
            scan_count=3,
            scan_interval_s=0.0,
            confirm_no_visible_collision=True,
        ),
        sleep_fn=lambda _seconds: None,
    )
    report = build_live_report(
        mode="grounded",
        config=config,
        started_at=datetime.now(timezone.utc),
        data=data,
    )

    assert not report.success
    assert data["valid_scan_count"] < data["scan_count"]
    assert data["invalid_scan_count"] > 0
    assert "every_measured_scan_policy_valid" in report.data["acceptance_failures"]
    assert "valid_scan_count_equals_scan_count" in report.data["acceptance_failures"]
    assert "invalid_scan_count_zero" in report.data["acceptance_failures"]
    summary = data["elevation_diagnostics_summary"]
    assert summary["above_configured_fov_count"] > 0
    assert summary["maximum_above_fov_overshoot_degrees"] > 0.0


def test_grounded_cli_returns_one_and_still_saves_failed_report(
    tmp_path: Path,
) -> None:
    script = _load_script()
    output = ROOT / "results/reports/m13/lidar" / f"failed-{tmp_path.name}"
    before = set(output.glob("m13_4_grounded_*.json"))
    args = script.parse_args(
        [
            "--config",
            str(FEATURE_CONFIG),
            "--output-dir",
            str(output),
            "grounded",
            "--vehicle-name",
            "SimpleFlight",
            "--lidar-name",
            "LidarSensor1",
            "--scan-count",
            "3",
            "--scan-interval",
            "0",
            "--allow-live-rpc",
            "--confirm-no-visible-collision",
        ]
    )

    result = script.run(
        args,
        repository_root=ROOT,
        client_module_loader=lambda _name: FAKE_MODULE,
        client_factory=lambda _module: FakeGroundedClient([[1.0, 0.0, -1.0]] * 3),
        sleep_fn=lambda _seconds: None,
    )

    created = set(output.glob("m13_4_grounded_*.json")) - before
    assert result == 1
    assert len(created) == 1
    report_path = created.pop()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert not report["success"]
    assert report["data"]["valid_scan_count"] == 0
    assert report["data"]["invalid_scan_count"] == 3
    assert report["data"]["acceptance_failures"]
    assert "point_cloud" not in report_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "missing_flag",
    [
        "--allow-live-rpc",
        "--allow-scene-mutation",
        "--confirm-scene-area-clear",
        "--confirm-no-visible-collision",
        "--allow-debug-markers",
        "--allow-marker-flush",
        "--allow-flight",
        "--allow-start-positioning",
        "--confirm-clear-airspace",
    ],
)
def test_new_live_authorization_gaps_fail_before_client_import(
    missing_flag: str,
) -> None:
    script = _load_script()
    arguments = _live_cli_arguments("known-geometry-live")
    arguments.remove(missing_flag)
    imported = False

    def loader(_name: str) -> object:
        nonlocal imported
        imported = True
        raise AssertionError("client import must not occur")

    with pytest.raises(ValueError, match="authorization"):
        script.run(
            script.parse_args(arguments),
            repository_root=ROOT,
            client_module_loader=loader,
        )
    assert not imported


@pytest.mark.parametrize(
    ("mode", "option", "value", "match"),
    [
        ("known-geometry-live", "--vehicle-name", "WrongVehicle", "vehicle name"),
        ("airborne-smoke", "--lidar-name", "WrongLidar", "LiDAR name"),
        (
            "known-geometry-live",
            "--course-profile",
            "medium",
            "requires course profile",
        ),
        ("airborne-smoke", "--course-seed", "2200", "requires declared base seed"),
        ("known-geometry-live", "--scan-count", "0", "sample count"),
        ("known-geometry-live", "--scan-interval", "nan", "scan interval"),
        ("airborne-smoke", "--step-count", "21", "sample count"),
    ],
)
def test_new_live_argument_errors_fail_before_client_import(
    mode: str,
    option: str,
    value: str,
    match: str,
) -> None:
    script = _load_script()
    arguments = _live_cli_arguments(mode)
    arguments[arguments.index(option) + 1] = value
    imported = False

    def loader(_name: str) -> object:
        nonlocal imported
        imported = True
        raise AssertionError("client import must not occur")

    with pytest.raises(ValueError, match=match):
        script.run(
            script.parse_args(arguments),
            repository_root=ROOT,
            client_module_loader=loader,
        )
    assert not imported


def test_live_report_output_and_unsolvable_course_fail_before_client_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    imported = False

    def loader(_name: str) -> object:
        nonlocal imported
        imported = True
        raise AssertionError("client import must not occur")

    outside = ["--output-dir", str(tmp_path), *_live_cli_arguments("airborne-smoke")]
    with pytest.raises(ValueError, match="reports must remain"):
        script.run(
            script.parse_args(outside),
            repository_root=ROOT,
            client_module_loader=loader,
        )
    assert not imported

    import r1_uav_nav.sim.lidar_live_validation as live_validation

    monkeypatch.setattr(
        live_validation,
        "generate_solvable_course",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("synthetic unsolvable course")
        ),
    )
    with pytest.raises(ValueError, match="synthetic unsolvable"):
        script.run(
            script.parse_args(_live_cli_arguments("airborne-smoke")),
            repository_root=ROOT,
            client_module_loader=loader,
        )
    assert not imported


def test_known_geometry_sampling_uses_exact_indices_and_distance_conversion(
    prepared_known_geometry,
) -> None:
    scans = [_scan_point(timestamp=value) for value in (1, 2, 3)]
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    evidence = collect_known_geometry_samples(
        _context(scans),
        prepared_known_geometry.feature_config,
        prepared_known_geometry.known_geometry,
        scan_count=3,
        scan_interval_s=0.0,
    )

    assert evidence["collected_scan_count"] == 3
    assert evidence["expected_comparison_index"] == 31
    assert evidence["clear_comparison_index"] == 38
    assert evidence["median_distance_m"] == pytest.approx(5.145347554864271)
    assert evidence["distance_within_tolerance"]
    assert not evidence["acceptance_failures"]
    assert all(value == 1.0 for value in evidence["clear_sector_normalized_values"])
    assert "point_cloud" not in json.dumps(evidence)
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert numpy_state[0] == current_numpy_state[0]
    assert np.array_equal(numpy_state[1], current_numpy_state[1])
    assert numpy_state[2:] == current_numpy_state[2:]


@pytest.mark.parametrize(
    ("scans", "failed_check"),
    [
        (
            [_scan_point(7.0, timestamp=value) for value in (1, 2, 3)],
            "distance_within_tolerance",
        ),
        (
            [
                _scan_point(timestamp=1),
                _scan_point(timestamp=2, elevation_degrees=20.0),
                _scan_point(timestamp=3),
            ],
            "every_scan_policy_valid",
        ),
        (
            [_scan_point(timestamp=value) for value in (1, 1, 1)],
            "timestamps_first_valid_then_fresh",
        ),
    ],
)
def test_known_geometry_sampling_rejects_distance_scan_and_timestamp_failures(
    prepared_known_geometry,
    scans: list[object],
    failed_check: str,
) -> None:
    evidence = collect_known_geometry_samples(
        _context(scans),
        prepared_known_geometry.feature_config,
        prepared_known_geometry.known_geometry,
        scan_count=3,
        scan_interval_s=0.0,
    )

    assert failed_check in evidence["acceptance_failures"]


class _FakeLiveManager:
    def __init__(self, runtime: object, materialized: object) -> None:
        self.last_runtime = runtime
        self.runtime = runtime
        self.materialized = materialized

    def materialize(self, *_args, **_kwargs):
        return self.materialized, self.runtime


def _materialized() -> object:
    return SimpleNamespace(
        start_anchor_world=SimpleNamespace(x=4.0, y=0.0, z=-2.1),
        goal_approach_world=SimpleNamespace(x=13.0, y=0.0, z=-2.1),
    )


def _scene_cleanup(events: list[str], *, fail: bool = False):
    def cleanup(_client: object, _runtime: object):
        events.append("scene-cleanup")
        return (
            SceneCleanupResult("uav", False, True, ()),
            SceneCleanupResult(
                "objects",
                True,
                not fail,
                ("object cleanup failed",) if fail else (),
            ),
            SceneCleanupResult("markers", True, True, ()),
        )

    return cleanup


def _safe_position(
    events: list[str],
    scans: list[object],
):
    def position(
        _client: object,
        _client_module: object,
        _materialized_scene: object,
        runtime: object,
        _config: object,
        *,
        start_anchor_callback,
        **_kwargs,
    ):
        try:
            result = start_anchor_callback(_context(scans))
        except BaseException:
            events.append("uav-safe-return")
            runtime.vehicle_positioning_evidence = {
                "returned_to_original_ground": True,
                "landing_confirmed": True,
                "api_control_released": True,
            }
            raise
        events.append("uav-safe-return")
        evidence = {
            "returned_to_original_ground": True,
            "landing_confirmed": True,
            "api_control_released": True,
            "start_anchor_callback": {"result": result},
        }
        runtime.vehicle_positioning_evidence = evidence
        return evidence

    return position


def test_known_geometry_execution_cleans_after_safe_return_and_accepts(
    prepared_known_geometry,
) -> None:
    events: list[str] = []
    runtime = SimpleNamespace(vehicle_positioning_evidence={})
    manager = _FakeLiveManager(runtime, _materialized())
    client = FakeGroundedClient()

    result = execute_known_geometry_live(
        prepared_known_geometry,
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        sleep_fn=lambda _seconds: None,
        manager_factory=lambda *_args, **_kwargs: manager,
        position_fn=_safe_position(
            events, [_scan_point(timestamp=value) for value in (1, 2, 3)]
        ),
        cleanup_fn=_scene_cleanup(events),
    )

    assert not result.errors
    assert not result.data["acceptance_failures"]
    assert events == ["uav-safe-return", "scene-cleanup"]
    assert not result.data["broad_reset_attempted"]


@pytest.mark.parametrize(
    ("raised", "interrupted"),
    [
        (RuntimeError("callback failed"), False),
        (StartAnchorCallbackTimeout("callback timed out"), False),
        (KeyboardInterrupt(), True),
    ],
)
def test_known_geometry_callback_failures_preserve_return_and_scene_cleanup(
    prepared_known_geometry,
    raised: BaseException,
    interrupted: bool,
) -> None:
    events: list[str] = []
    runtime = SimpleNamespace(vehicle_positioning_evidence={})
    manager = _FakeLiveManager(runtime, _materialized())

    def failing_position(*_args, **kwargs):
        callback = kwargs["start_anchor_callback"]
        context = StartAnchorReadOnlyContext(
            "SimpleFlight",
            "LidarSensor1",
            20,
            100.0,
            lambda: (_ for _ in ()).throw(raised),
            lambda: 0.0,
        )
        try:
            callback(context)
        except BaseException:
            events.append("uav-safe-return")
            runtime.vehicle_positioning_evidence = {
                "returned_to_original_ground": True,
                "landing_confirmed": True,
                "api_control_released": True,
            }
            raise

    result = execute_known_geometry_live(
        prepared_known_geometry,
        FakeGroundedClient(),
        FAKE_MODULE,  # type: ignore[arg-type]
        sleep_fn=lambda _seconds: None,
        manager_factory=lambda *_args, **_kwargs: manager,
        position_fn=failing_position,
        cleanup_fn=_scene_cleanup(events),
    )

    assert result.interrupted is interrupted
    assert result.errors
    assert events == ["uav-safe-return", "scene-cleanup"]


class _ObservationSpace:
    def contains(self, observation: np.ndarray) -> bool:
        return (
            observation.shape == (83,)
            and observation.dtype == np.float32
            and bool(np.all(np.isfinite(observation)))
        )


class _FakeLidarEnvironment:
    def __init__(
        self,
        events: list[str],
        *,
        sensor_failure: bool = False,
        terminate: bool = False,
        raise_on_step: bool = False,
        raise_on_reset: bool = False,
        cleanup_failure: bool = False,
        returned_to_original: bool = True,
    ) -> None:
        self.events = events
        self.sensor_failure = sensor_failure
        self.terminate = terminate
        self.raise_on_step = raise_on_step
        self.raise_on_reset = raise_on_reset
        self.cleanup_failure = cleanup_failure
        self.returned_to_original = returned_to_original
        self.index = 0
        self.action_space = SimpleNamespace(shape=(3,))
        self.observation_space = _ObservationSpace()
        self.last_navigation_observation = np.zeros(10, dtype=np.float32)
        self.lifecycle_evidence = {
            "original_ground_position": {"x": 0.0, "y": 0.0, "z": 0.57},
            "landing_position_tolerance_m": 0.75,
            "cleanup_attempts": (),
            "recovery_retry_required": False,
        }

    def _result(self, timestamp_status: str):
        lidar_values = np.full(72, 0.5, dtype=np.float32)
        observation = np.concatenate(
            (self.last_navigation_observation, lidar_values, np.asarray([1.0]))
        ).astype(np.float32)
        info = {
            "lidar": {
                "status": "valid",
                "sector_values": lidar_values.tolist(),
                "diagnostics": {"timestamp_status": timestamp_status},
            },
            "sensor_failure": self.sensor_failure,
            "termination_reason": "collision" if self.terminate else None,
        }
        return observation, info

    def reset(self, *, options):
        assert options["start_anchor"] == Position3D(4.0, 0.0, -2.1)
        self.events.append("environment-reset")
        if self.raise_on_reset:
            raise RuntimeError("synthetic reset failure")
        return self._result("first_valid")

    def step(self, action: np.ndarray):
        assert action.shape == (3,)
        assert action.dtype == np.float32
        if self.raise_on_step:
            raise RuntimeError("synthetic step failure")
        self.index += 1
        observation, info = self._result("fresh")
        return observation, 0.0, self.terminate, self.sensor_failure, info

    def close_with_result(self):
        self.events.append("environment-cleanup")
        self.lifecycle_evidence["cleanup_attempts"] = (
            {
                "attempt_number": 1,
                "returned_to_original_ground": self.returned_to_original,
                "landing_confirmed": self.returned_to_original,
                "api_control_released": True,
            },
        )
        return CleanupResult(
            ("hoverAsync", "landAsync", "armDisarm", "enableApiControl"),
            (
                ()
                if self.cleanup_failure
                else (
                    "hoverAsync",
                    "landAsync",
                    "armDisarm",
                    "enableApiControl",
                )
            ),
            ("cleanup failed",) if self.cleanup_failure else (),
            self.cleanup_failure,
        )


def _execute_fake_airborne(
    prepared_airborne_smoke,
    environment: _FakeLidarEnvironment,
    events: list[str],
    *,
    scene_cleanup_failure: bool = False,
    client: FakeGroundedClient | None = None,
) -> LiveExecutionResult:
    runtime = SimpleNamespace(vehicle_positioning_evidence={})
    manager = _FakeLiveManager(runtime, _materialized())
    return execute_airborne_smoke(
        prepared_airborne_smoke,
        client or FakeGroundedClient(),
        FAKE_MODULE,  # type: ignore[arg-type]
        sleep_fn=lambda _seconds: None,
        manager_factory=lambda *_args, **_kwargs: manager,
        environment_factory=lambda *_args: environment,
        cleanup_fn=_scene_cleanup(events, fail=scene_cleanup_failure),
    )


def test_airborne_smoke_observations_and_cleanup_are_accepted(
    prepared_airborne_smoke,
) -> None:
    events: list[str] = []
    result = _execute_fake_airborne(
        prepared_airborne_smoke,
        _FakeLidarEnvironment(events),
        events,
    )

    assert not result.errors
    assert not result.data["acceptance_failures"]
    assert result.data["observation_count"] == 3
    assert all(item["shape"] == [83] for item in result.data["observations"])
    assert all(item["dtype"] == "float32" for item in result.data["observations"])
    assert all(item["lidar_valid_flag"] == 1.0 for item in result.data["observations"])
    assert events == [
        "environment-reset",
        "environment-cleanup",
        "scene-cleanup",
    ]


@pytest.mark.parametrize("mode", ["known-geometry-live", "airborne-smoke"])
def test_new_live_help_does_not_import_airsim(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "airsim" or name.startswith("airsim."):
            raise AssertionError("CLI help must not import AirSim")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as exit_info:
        script.parse_args([mode, "--help"])
    assert exit_info.value.code == 0


@pytest.mark.parametrize(
    ("environment_kwargs", "failed_check"),
    [
        ({"sensor_failure": True}, "no_sensor_failure"),
        ({"terminate": True}, "no_unexpected_termination"),
        ({"cleanup_failure": True}, "environment_cleanup_succeeded"),
    ],
)
def test_airborne_smoke_faults_and_cleanup_failure_fail_acceptance(
    prepared_airborne_smoke,
    environment_kwargs: dict[str, bool],
    failed_check: str,
) -> None:
    events: list[str] = []
    result = _execute_fake_airborne(
        prepared_airborne_smoke,
        _FakeLidarEnvironment(events, **environment_kwargs),
        events,
    )

    assert failed_check in result.data["acceptance_failures"]
    if environment_kwargs.get("cleanup_failure"):
        assert "scene-cleanup" in events


def test_airborne_step_failure_still_closes_before_scene_cleanup(
    prepared_airborne_smoke,
) -> None:
    events: list[str] = []
    result = _execute_fake_airborne(
        prepared_airborne_smoke,
        _FakeLidarEnvironment(events, raise_on_step=True),
        events,
    )

    assert result.errors
    assert events[-2:] == ["environment-cleanup", "scene-cleanup"]


def test_airborne_reset_error_is_preserved_after_safe_recovery_and_scene_cleanup(
    prepared_airborne_smoke,
) -> None:
    events: list[str] = []
    result = _execute_fake_airborne(
        prepared_airborne_smoke,
        _FakeLidarEnvironment(events, raise_on_reset=True),
        events,
    )

    assert result.errors == ("RuntimeError: synthetic reset failure",)
    assert events == [
        "environment-reset",
        "environment-cleanup",
        "scene-cleanup",
    ]
    assert result.data["scene_cleanup_performed"]
    assert not result.data["scene_cleanup_deferred"]
    assert result.data["environment_lifecycle"]["cleanup_attempts"][0][
        "returned_to_original_ground"
    ]


class _UnsafeFinalClient(FakeGroundedClient):
    def getMultirotorState(self, *, vehicle_name: str) -> object:
        self.calls.append(("state", vehicle_name))
        return SimpleNamespace(
            landed_state=FakeLandedState.Flying,
            kinematics_estimated=SimpleNamespace(
                position=_vector(4.0, 0.0, 0.12),
                linear_velocity=_vector(0.0, 0.0, 0.0),
            ),
        )


def test_airborne_unverified_final_state_defers_scene_cleanup(
    prepared_airborne_smoke,
) -> None:
    events: list[str] = []
    result = _execute_fake_airborne(
        prepared_airborne_smoke,
        _FakeLidarEnvironment(events, returned_to_original=False),
        events,
        client=_UnsafeFinalClient(),
    )

    assert "scene-cleanup" not in events
    assert result.data["scene_cleanup_deferred"]
    assert not result.data["scene_cleanup_performed"]
    assert result.data["scene_cleanup_deferred_reason"] == (
        "final named vehicle state is not confirmed landed"
    )
    assert "object_cleanup_succeeded" in result.data["acceptance_failures"]


def test_airborne_report_contains_anchor_return_and_appearance_evidence(
    prepared_airborne_smoke,
) -> None:
    events: list[str] = []
    environment = _FakeLidarEnvironment(events)
    environment.lifecycle_evidence["start_anchor_confirmation"] = {
        "requested_start_anchor": {"x": 4.0, "y": 0.0, "z": -2.1},
        "measured_start_anchor": {"x": 4.1, "y": 0.0, "z": -2.1},
        "position_error_m": 0.1,
        "position_tolerance_m": 0.75,
        "measured_speed_m_s": 0.0,
        "confirmation_attempts": 4,
        "confirmation_success": True,
    }

    result = _execute_fake_airborne(
        prepared_airborne_smoke,
        environment,
        events,
    )

    lifecycle = result.data["environment_lifecycle"]
    assert lifecycle["start_anchor_confirmation"]["confirmation_attempts"] == 4
    assert lifecycle["cleanup_attempts"][0]["attempt_number"] == 1
    assert result.data["scene_cleanup_performed"]
    appearance = result.data["start_pad_appearance"]
    assert appearance["requested_material_name"] is None
    assert not appearance["requests_expected_red_appearance"]
    assert not appearance["material_assignment_attempted"]
    assert appearance["cosmetic_only"]


def test_failed_live_cli_writes_report_and_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    output = ROOT / "results/reports/m13/lidar" / f"live-failed-{tmp_path.name}"
    arguments = [
        "--output-dir",
        str(output),
        *_live_cli_arguments("known-geometry-live"),
    ]
    before = set(output.glob("m13_4_known_geometry_live_*.json"))
    client = FakeGroundedClient()
    monkeypatch.setattr(
        script,
        "execute_known_geometry_live",
        lambda *_args, **_kwargs: LiveExecutionResult(
            {
                "acceptance_checks": {"synthetic_acceptance": False},
                "acceptance_failures": ("synthetic_acceptance",),
            },
            {"resources_acquired": False},
            (),
            False,
        ),
    )

    result = script.run(
        script.parse_args(arguments),
        repository_root=ROOT,
        client_module_loader=lambda _name: FAKE_MODULE,
        client_factory=lambda _module: client,
        sleep_fn=lambda _seconds: None,
    )

    created = set(output.glob("m13_4_known_geometry_live_*.json")) - before
    assert result == 1
    assert len(created) == 1
    payload = created.pop().read_text(encoding="utf-8")
    assert not json.loads(payload)["success"]
    assert "point_cloud" not in payload
