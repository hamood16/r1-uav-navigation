from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import r1_uav_nav.sim.colosseum_capabilities as capabilities
from r1_uav_nav.sim.colosseum_capabilities import (
    AirborneProbeConfig,
    AirborneProbeContext,
    CapabilityObservation,
    CapabilityProbeError,
    CapabilityProbeReport,
    CapabilityStatus,
    CleanupDomainResult,
    CollisionClassification,
    EvidenceLevel,
    GroundedLidarProbeConfig,
    LidarProbeConfig,
    PerformanceProbeConfig,
    ProbeRuntimeState,
    SceneMutationConfig,
    SceneSurveyConfig,
    SelfHitClassification,
    analyze_lidar_timestamps,
    classify_collision_samples,
    cleanup_probe_domains,
    derive_safe_probe_position,
    generate_report_path,
    inspect_client_capabilities,
    invoke_capability,
    prepare_airborne_probe,
    probe_debug_markers,
    probe_grounded_lidar,
    probe_lidar,
    probe_performance,
    probe_scene_mutation,
    sanitize_collision_info,
    sanitize_settings,
    save_capability_report,
    survey_scene,
    validate_grounded_lidar_scan,
    validate_grounded_preflight,
    validate_lidar_scan,
    validate_report_output_path,
)
from r1_uav_nav.sim.colosseum_client import CleanupState, ColosseumClientImportError
from r1_uav_nav.sim.waypoint_navigation import Position3D

DOC_PATH = Path("docs/m13_colosseum_capability_probe.md")
README_PATH = Path("README.md")
SCRIPT_PATH = Path("scripts/check_colosseum_capabilities.py")
TEST_RED_MATERIAL = "/AirSim/Models/MiniQuadCopter/Prop_Red_Plastic.Prop_Red_Plastic"
TEST_VEHICLE_NAME = "SimpleFlight"
TEST_LIDAR_NAME = "LidarSensor1"


class FakeVector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x_val = x
        self.y_val = y
        self.z_val = z


class FakeQuaternion:
    def __init__(
        self,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:
        self.w_val = w
        self.x_val = x
        self.y_val = y
        self.z_val = z


class FakePose:
    def __init__(
        self,
        position_val: FakeVector | None = None,
        orientation_val: FakeQuaternion | None = None,
    ) -> None:
        self.position = position_val or FakeVector()
        self.orientation = orientation_val or FakeQuaternion()


class FakeAsyncResult:
    def __init__(self, calls: list[Any], name: str) -> None:
        self.calls = calls
        self.name = name

    def join(self) -> None:
        self.calls.append(f"{self.name}.join")


def make_state(
    position: tuple[float, float, float] = (1.0, 2.0, 0.56),
    *,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    landed: int = 0,
    collision: bool = False,
    yaw_degrees: float = 0.0,
) -> Any:
    yaw_radians = math.radians(yaw_degrees)
    return SimpleNamespace(
        landed_state=landed,
        collision=SimpleNamespace(has_collided=collision),
        kinematics_estimated=SimpleNamespace(
            position=FakeVector(*position),
            linear_velocity=FakeVector(*velocity),
            orientation=FakeQuaternion(
                w=math.cos(yaw_radians / 2.0),
                z=math.sin(yaw_radians / 2.0),
            ),
        ),
    )


def make_lidar_scan(
    point_cloud: list[float] | None = None,
    *,
    timestamp: int = 1,
    pose: FakePose | None = None,
) -> Any:
    return SimpleNamespace(
        point_cloud=(
            [1.0, 0.0, 0.0, 0.0, 2.0, 0.0] if point_cloud is None else point_cloud
        ),
        time_stamp=timestamp,
        pose=pose or FakePose(),
    )


def make_collision_info(
    has_collided: bool = False,
    *,
    object_name: str = "",
    object_id: int = -1,
    time_stamp: int = 0,
    penetration_depth: float = 0.0,
    impact_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    return SimpleNamespace(
        has_collided=has_collided,
        object_name=object_name,
        object_id=object_id,
        time_stamp=time_stamp,
        penetration_depth=penetration_depth,
        impact_point=FakeVector(*impact_point),
        position=FakeVector(*position),
        normal=FakeVector(*normal),
    )


def make_ground_collision_samples(
    **overrides: Any,
) -> tuple[Any, Any, Any]:
    values: dict[str, Any] = {
        "has_collided": True,
        "object_name": "SpawnSurface",
        "object_id": 17,
        "time_stamp": 100,
        "penetration_depth": 0.02,
        "impact_point": (1.0, 2.0, 0.58),
        "position": (1.0, 2.0, 0.56),
        "normal": (0.0, 0.0, -1.0),
    }
    values.update(overrides)
    return tuple(
        sanitize_collision_info(make_collision_info(**values)) for _ in range(3)
    )  # type: ignore[return-value]


def make_ground_collision_sequence(
    collided: tuple[bool, bool, bool],
    **overrides: Any,
) -> tuple[Any, Any, Any]:
    values: dict[str, Any] = {
        "object_name": "SpawnSurface",
        "object_id": 17,
        "time_stamp": 100,
        "penetration_depth": 0.02,
        "impact_point": (1.0, 2.0, 0.58),
        "position": (1.0, 2.0, 0.56),
        "normal": (0.0, 0.0, -1.0),
    }
    values.update(overrides)
    return tuple(
        sanitize_collision_info(
            make_collision_info(has_collided=has_collided, **values)
        )
        for has_collided in collided
    )  # type: ignore[return-value]


class FakeClient:
    constructed = 0

    def __init__(self, *_: Any, **__: Any) -> None:
        type(self).constructed += 1
        self.calls: list[Any] = []
        self.state = make_state()
        self.api_enabled = False
        self.objects: dict[str, tuple[FakePose, FakeVector]] = {
            "Wall_01": (FakePose(FakeVector(4.0, 0.0, -1.0)), FakeVector(1, 1, 1))
        }
        self.assets = ["1M_Cube_Chamfer", "WallAsset"]
        self.returned_spawn_name: str | None = None
        self.destroy_raises = False
        self.flush_raises = False
        self.collision_samples: list[Any] = []
        self.collision_sample_index = 0
        self.lidar_scans: list[Any] = []
        self.lidar_scan_index = 0

    def confirmConnection(self) -> None:
        self.calls.append("confirmConnection")

    def ping(self) -> bool:
        return True

    def getClientVersion(self) -> int:
        return 1

    def getServerVersion(self) -> int:
        return 1

    def getMinRequiredServerVersion(self) -> int:
        return 1

    def getMinRequiredClientVersion(self) -> int:
        return 1

    def getMultirotorState(self, vehicle_name: str = "") -> Any:
        self.calls.append(("getMultirotorState", vehicle_name))
        return self.state

    def isApiControlEnabled(self, vehicle_name: str = "") -> bool:
        self.calls.append(("isApiControlEnabled", vehicle_name))
        return self.api_enabled

    def enableApiControl(self, enabled: bool, vehicle_name: str = "") -> None:
        self.calls.append(("enableApiControl", enabled, vehicle_name))
        self.api_enabled = enabled

    def armDisarm(self, armed: bool, vehicle_name: str = "") -> None:
        self.calls.append(("armDisarm", armed, vehicle_name))

    def takeoffAsync(
        self, timeout_sec: float = 20.0, vehicle_name: str = ""
    ) -> FakeAsyncResult:
        self.calls.append(("takeoffAsync", timeout_sec, vehicle_name))
        return FakeAsyncResult(self.calls, "takeoffAsync")

    def moveToPositionAsync(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float,
        *,
        timeout_sec: float,
        vehicle_name: str = "",
    ) -> FakeAsyncResult:
        self.calls.append(
            (
                "moveToPositionAsync",
                x,
                y,
                z,
                velocity,
                timeout_sec,
                vehicle_name,
            )
        )
        self.state = make_state((x, y, z), landed=1)
        return FakeAsyncResult(self.calls, "moveToPositionAsync")

    def moveByVelocityAsync(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
        vehicle_name: str = "",
    ) -> FakeAsyncResult:
        self.calls.append(("moveByVelocityAsync", vx, vy, vz, duration, vehicle_name))
        return FakeAsyncResult(self.calls, "moveByVelocityAsync")

    def hoverAsync(self, vehicle_name: str = "") -> FakeAsyncResult:
        self.calls.append(("hoverAsync", vehicle_name))
        return FakeAsyncResult(self.calls, "hoverAsync")

    def landAsync(self, vehicle_name: str = "") -> FakeAsyncResult:
        self.calls.append(("landAsync", vehicle_name))
        return FakeAsyncResult(self.calls, "landAsync")

    def rotateToYawAsync(
        self,
        yaw: float,
        timeout_sec: float = 20.0,
        margin: float = 5.0,
        vehicle_name: str = "",
    ) -> FakeAsyncResult:
        self.calls.append(("rotateToYawAsync", yaw, timeout_sec, margin, vehicle_name))
        position = self.state.kinematics_estimated.position
        self.state = make_state(
            (position.x_val, position.y_val, position.z_val),
            landed=self.state.landed_state,
            yaw_degrees=yaw,
        )
        return FakeAsyncResult(self.calls, "rotateToYawAsync")

    def simGetCollisionInfo(self, vehicle_name: str = "") -> Any:
        self.calls.append(("simGetCollisionInfo", vehicle_name))
        if not self.collision_samples:
            return make_collision_info()
        index = min(self.collision_sample_index, len(self.collision_samples) - 1)
        self.collision_sample_index += 1
        return self.collision_samples[index]

    def getSettingsString(self) -> str:
        self.calls.append("getSettingsString")
        return json.dumps(
            {
                "SettingsVersion": 1.2,
                "SimMode": "Multirotor",
                "ClockSpeed": 1,
                "LocalPath": "C:/private/path",
                "Vehicles": {
                    "Drone1": {
                        "VehicleType": "SimpleFlight",
                        "Sensors": {
                            "LidarSensor1": {
                                "SensorType": 6,
                                "Enabled": True,
                                "DataFrame": "SensorLocalFrame",
                                "Path": "C:/private/sensor",
                            }
                        },
                    },
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
                    },
                },
            }
        )

    def listVehicles(self) -> list[str]:
        self.calls.append("listVehicles")
        return ["Drone1", "SimpleFlight"]

    def simListSceneObjects(self, regex: str = ".*") -> list[str]:
        self.calls.append(("simListSceneObjects", regex))
        pattern = re_compile(regex)
        return [name for name in self.objects if pattern.fullmatch(name)]

    def simGetObjectPose(self, name: str) -> FakePose:
        self.calls.append(("simGetObjectPose", name))
        return self.objects[name][0]

    def simGetObjectScale(self, name: str) -> FakeVector:
        self.calls.append(("simGetObjectScale", name))
        return self.objects[name][1]

    def simListAssets(self) -> list[str]:
        self.calls.append("simListAssets")
        return self.assets

    def simGetMeshPositionVertexBuffers(self) -> None:
        raise AssertionError("mesh-buffer RPC must remain disabled")

    def simSpawnObject(
        self,
        name: str,
        asset: str,
        pose: FakePose,
        scale: FakeVector,
        physics_enabled: bool,
        is_blueprint: bool,
    ) -> str:
        self.calls.append(
            ("simSpawnObject", name, asset, physics_enabled, is_blueprint)
        )
        returned = self.returned_spawn_name
        if returned is None:
            returned = name
        if returned:
            self.objects[returned] = (pose, scale)
        return returned

    def simSetObjectPose(self, name: str, pose: FakePose, teleport: bool) -> bool:
        self.calls.append(("simSetObjectPose", name, teleport))
        self.objects[name] = (pose, self.objects[name][1])
        return True

    def simSetObjectScale(self, name: str, scale: FakeVector) -> bool:
        self.calls.append(("simSetObjectScale", name))
        self.objects[name] = (self.objects[name][0], scale)
        return True

    def simSetSegmentationObjectID(
        self, name: str, object_id: int, is_regex: bool
    ) -> bool:
        self.calls.append(("simSetSegmentationObjectID", name, object_id, is_regex))
        self.segmentation_id = object_id
        return True

    def simGetSegmentationObjectID(self, name: str) -> int:
        self.calls.append(("simGetSegmentationObjectID", name))
        return self.segmentation_id

    def simSetObjectMaterial(self, name: str, material: str, component: int) -> bool:
        self.calls.append(("simSetObjectMaterial", name, material, component))
        return True

    def simDestroyObject(self, name: str) -> bool:
        self.calls.append(("simDestroyObject", name))
        if self.destroy_raises:
            raise RuntimeError("destroy failed")
        self.objects.pop(name, None)
        return True

    def simPlotPoints(self, *args: Any) -> None:
        self.calls.append(("simPlotPoints", args))

    def simPlotLineStrip(self, *args: Any) -> None:
        self.calls.append(("simPlotLineStrip", args))

    def simPlotLineList(self, *args: Any) -> None:
        self.calls.append(("simPlotLineList", args))

    def simPlotTransforms(self, *args: Any) -> None:
        self.calls.append(("simPlotTransforms", args))

    def simFlushPersistentMarkers(self) -> None:
        self.calls.append("simFlushPersistentMarkers")
        if self.flush_raises:
            raise RuntimeError("flush failed")

    def getLidarData(self, lidar_name: str = "", vehicle_name: str = "") -> Any:
        self.calls.append(("getLidarData", lidar_name, vehicle_name))
        if self.lidar_scans:
            index = min(self.lidar_scan_index, len(self.lidar_scans) - 1)
            self.lidar_scan_index += 1
            return self.lidar_scans[index]
        return SimpleNamespace(
            point_cloud=[1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            time_stamp=len(self.calls),
            pose=FakePose(),
        )


FAKE_MODULE = SimpleNamespace(
    MultirotorClient=FakeClient,
    Vector3r=FakeVector,
    Pose=FakePose,
    LandedState=SimpleNamespace(Landed=0, Flying=1),
    __version__="fake",
)


def make_airborne_config(**overrides: Any) -> AirborneProbeConfig:
    values: dict[str, Any] = {
        "vehicle_name": TEST_VEHICLE_NAME,
        "allow_flight": True,
        "confirm_clear_airspace": True,
        "confirm_no_visible_collision": True,
        "confirm_grounded_lidar_passed": True,
    }
    values.update(overrides)
    return AirborneProbeConfig(**values)


def make_airborne_context(**overrides: Any) -> AirborneProbeContext:
    values: dict[str, Any] = {
        "vehicle_name": TEST_VEHICLE_NAME,
        "lidar_name": TEST_LIDAR_NAME,
        "ground_position": Position3D(1.0, 2.0, 0.56),
        "ground_reference_z": 0.56,
        "anchor_position": Position3D(1.0, 2.0, -1.44),
        "initial_collision_timestamp": 0,
        "configured_range": 20.0,
    }
    values.update(overrides)
    return AirborneProbeContext(**values)


def make_lidar_config(**overrides: Any) -> LidarProbeConfig:
    values: dict[str, Any] = {
        "airborne": make_airborne_config(),
        "lidar_name": TEST_LIDAR_NAME,
        "scan_count": 2,
        "scan_interval": 0.0,
        "warm_up_attempts": 1,
        "warm_up_interval": 0.0,
        "settle_interval": 0.0,
    }
    values.update(overrides)
    return LidarProbeConfig(**values)


def assert_vehicle_calls_are_named(calls: list[Any]) -> None:
    vehicle_index = {
        "getMultirotorState": 1,
        "isApiControlEnabled": 1,
        "simGetCollisionInfo": 1,
        "enableApiControl": 2,
        "armDisarm": 2,
        "takeoffAsync": 2,
        "hoverAsync": 1,
        "landAsync": 1,
        "moveToPositionAsync": 6,
        "moveByVelocityAsync": 5,
        "rotateToYawAsync": 4,
        "getLidarData": 2,
    }
    relevant = [
        call for call in calls if isinstance(call, tuple) and call[0] in vehicle_index
    ]
    assert relevant
    for call in relevant:
        assert call[vehicle_index[call[0]]] == TEST_VEHICLE_NAME, call


def test_package_import_succeeds_when_airsim_import_is_blocked() -> None:
    code = """
import importlib
real_import = importlib.import_module
def blocked(name, *args, **kwargs):
    if name == 'airsim':
        raise ImportError('blocked by test')
    return real_import(name, *args, **kwargs)
importlib.import_module = blocked
import r1_uav_nav.sim
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cli_defaults_and_help_do_not_import_or_construct_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(
        module,
        "load_client_module",
        lambda _: (_ for _ in ()).throw(AssertionError("must stay lazy")),
    )
    FakeClient.constructed = 0

    args = module.parse_args([])
    assert args.command == "inspect-client"
    with pytest.raises(SystemExit) as exc_info:
        module.parse_args(["--help"])

    assert exc_info.value.code == 0
    assert FakeClient.constructed == 0


def test_common_cli_options_work_before_and_after_subcommand() -> None:
    module = load_script_module()

    before = module.parse_args(["--port", "42000", "survey"])
    after = module.parse_args(["survey", "--port", "42001"])

    assert before.port == 42000
    assert after.port == 42001


def test_authorization_is_rejected_before_client_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script_module()
    imported = False

    def fail_if_imported(_: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("client import must follow authorization")

    monkeypatch.setattr(module, "load_client_module", fail_if_imported)
    args = module.parse_args(
        ["markers", "--output-path", str(tmp_path / "unauthorized.json")]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 1
    assert imported is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["grounded-lidar", "--lidar-name", "LidarSensor1"],
        [
            "grounded-lidar",
            "--vehicle-name",
            "SimpleFlight",
            "--lidar-name",
            "LidarSensor1",
            "--warm-up-attempts",
            "11",
        ],
        [
            "grounded-lidar",
            "--vehicle-name",
            "SimpleFlight",
            "--lidar-name",
            "LidarSensor1",
            "--warm-up-interval",
            "nan",
        ],
        [
            "grounded-lidar",
            "--vehicle-name",
            "SimpleFlight",
            "--lidar-name",
            "LidarSensor1",
            "--warm-up-interval",
            "0.3",
        ],
    ],
)
def test_grounded_lidar_arguments_are_validated_before_client_import(
    arguments: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    imported = False

    def fail_if_imported(_: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("client import must follow validation")

    monkeypatch.setattr(module, "load_client_module", fail_if_imported)
    output = tmp_path / "invalid-grounded-lidar.json"
    args = module.parse_args([*arguments, "--output-path", str(output)])

    assert module.run_probe(args, repository_root=Path.cwd()) == 1
    assert imported is False


@pytest.mark.parametrize(
    "omitted_flag",
    [
        "--allow-flight",
        "--confirm-clear-airspace",
        "--confirm-no-visible-collision",
        "--confirm-grounded-lidar-passed",
    ],
)
def test_airborne_cli_confirmations_are_distinct_and_precede_client_import(
    omitted_flag: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    imported = False

    def fail_if_imported(_: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("client import must follow authorization")

    monkeypatch.setattr(module, "load_client_module", fail_if_imported)
    flags = [
        "--allow-flight",
        "--confirm-clear-airspace",
        "--confirm-no-visible-collision",
        "--confirm-grounded-lidar-passed",
    ]
    flags.remove(omitted_flag)
    args = module.parse_args(
        [
            "lidar",
            "--vehicle-name",
            TEST_VEHICLE_NAME,
            "--lidar-name",
            TEST_LIDAR_NAME,
            *flags,
            "--output-path",
            str(tmp_path / f"missing-{omitted_flag[2:]}.json"),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 1
    assert imported is False


def test_lidar_visualization_authorization_precedes_client_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    imported = False

    def fail_if_imported(_: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("client import must follow visualization authorization")

    monkeypatch.setattr(module, "load_client_module", fail_if_imported)
    args = module.parse_args(
        [
            "lidar",
            "--vehicle-name",
            TEST_VEHICLE_NAME,
            "--lidar-name",
            TEST_LIDAR_NAME,
            "--allow-flight",
            "--confirm-clear-airspace",
            "--confirm-no-visible-collision",
            "--confirm-grounded-lidar-passed",
            "--visualize-lidar",
            "--output-path",
            str(tmp_path / "unauthorized-visualization.json"),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 1
    assert imported is False


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--lidar-visualization-hold-seconds", "-0.1"),
        ("--lidar-visualization-hold-seconds", "15.1"),
        ("--lidar-visualization-hold-seconds", "nan"),
        ("--lidar-visualization-hold-seconds", "inf"),
        ("--lidar-visualization-max-points", "0"),
        ("--lidar-visualization-max-points", "2001"),
        ("--lidar-visualization-max-rays", "-1"),
        ("--lidar-visualization-max-rays", "65"),
    ],
)
def test_lidar_visualization_limits_are_validated_before_client_import(
    flag: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    imported = False

    def fail_if_imported(_: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("client import must follow visualization validation")

    monkeypatch.setattr(module, "load_client_module", fail_if_imported)
    args = module.parse_args(
        [
            "lidar",
            "--vehicle-name",
            TEST_VEHICLE_NAME,
            "--lidar-name",
            TEST_LIDAR_NAME,
            "--allow-flight",
            "--confirm-clear-airspace",
            "--confirm-no-visible-collision",
            "--confirm-grounded-lidar-passed",
            "--visualize-lidar",
            "--allow-marker-flush",
            flag,
            value,
            "--output-path",
            str(tmp_path / f"invalid-{flag[2:]}.json"),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 1
    assert imported is False


def test_grounded_lidar_cli_saves_report_without_resource_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    client = FakeClient()
    monkeypatch.setattr(module, "load_client_module", lambda _: FAKE_MODULE)
    monkeypatch.setattr(module, "create_probe_client", lambda *_: client)
    output = tmp_path / "grounded-lidar.json"
    args = module.parse_args(
        [
            "grounded-lidar",
            "--vehicle-name",
            "SimpleFlight",
            "--lidar-name",
            "LidarSensor1",
            "--scan-count",
            "2",
            "--scan-interval",
            "0",
            "--warm-up-attempts",
            "1",
            "--warm-up-interval",
            "0",
            "--confirm-no-visible-collision",
            "--output-path",
            str(output),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "grounded-lidar"
    assert report["success"] is True
    assert report["data"]["grounded_lidar"]["ready_for_airborne_validation"] is True
    assert all(result["attempted"] is False for result in report["cleanup_results"])


def test_missing_client_is_reported_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script_module()
    monkeypatch.setattr(
        module,
        "load_client_module",
        lambda _: (_ for _ in ()).throw(
            ColosseumClientImportError("matching client is not installed")
        ),
    )
    output = tmp_path / "missing.json"
    args = module.parse_args(["inspect-client", "--output-path", str(output)])

    result = module.run_probe(args, repository_root=Path.cwd())

    assert result == 1
    assert "matching client is not installed" in output.read_text(encoding="utf-8")


def test_static_method_presence_does_not_construct_client() -> None:
    FakeClient.constructed = 0

    observations = inspect_client_capabilities(FAKE_MODULE)  # type: ignore[arg-type]

    status_by_name = {item.capability: item.status for item in observations}
    assert status_by_name["scene_object_listing"] == (
        CapabilityStatus.CLIENT_METHOD_PRESENT_NOT_LIVE_TESTED
    )
    assert status_by_name["collision_state"] == (
        CapabilityStatus.CLIENT_METHOD_PRESENT_NOT_LIVE_TESTED
    )
    assert status_by_name["api_control_state"] == (
        CapabilityStatus.CLIENT_METHOD_PRESENT_NOT_LIVE_TESTED
    )
    assert status_by_name["armed_state"] == CapabilityStatus.CLIENT_METHOD_ABSENT
    assert status_by_name["direct_object_rgb"] == CapabilityStatus.CLIENT_METHOD_ABSENT
    assert FakeClient.constructed == 0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("timed out"), CapabilityStatus.RPC_TIMED_OUT),
        (RuntimeError("method not found"), CapabilityStatus.UNSUPPORTED_BY_BLOCKS),
        (RuntimeError("other"), CapabilityStatus.RPC_FAILED),
    ],
)
def test_rpc_statuses_remain_distinct(
    error: Exception, expected: CapabilityStatus
) -> None:
    client = SimpleNamespace(call=lambda: (_ for _ in ()).throw(error))

    observation, value = invoke_capability(client, "example", "call")

    assert observation.status == expected
    assert observation.evidence_level == EvidenceLevel.LIVE_RPC
    assert value is None


def test_settings_are_allowlisted_and_machine_paths_removed() -> None:
    client = FakeClient()

    settings = sanitize_settings(client.getSettingsString())
    serialized = json.dumps(settings)

    assert settings["ClockSpeed"] == 1
    assert (
        settings["Vehicles"]["Drone1"]["Sensors"]["LidarSensor1"]["DataFrame"]
        == "SensorLocalFrame"
    )
    assert "LocalPath" not in serialized
    assert "C:/private" not in serialized


def test_provisional_settings_include_inactive_vehicle_and_all_sensor_fields() -> None:
    settings = sanitize_settings(FakeClient().getSettingsString())
    vehicle = settings["Vehicles"]["SimpleFlight"]
    sensor = vehicle["Sensors"]["LidarSensor1"]

    assert vehicle["DefaultVehicleState"] == "Inactive"
    assert sensor == dict(capabilities.M13_LIDAR_PROVISIONAL_PROFILE.sensor_fields)
    comparisons = capabilities.compare_lidar_settings_profile(settings)
    assert all(item.present and item.matched for item in comparisons)


@pytest.mark.parametrize(
    ("scope", "field", "expected"),
    [
        *[
            ("vehicle", field, expected)
            for field, expected in (
                capabilities.M13_LIDAR_PROVISIONAL_PROFILE.vehicle_fields
            )
        ],
        *[
            ("sensor", field, expected)
            for field, expected in (
                capabilities.M13_LIDAR_PROVISIONAL_PROFILE.sensor_fields
            )
        ],
    ],
)
def test_profile_comparison_reports_each_mismatch(
    scope: str, field: str, expected: Any
) -> None:
    if isinstance(expected, bool):
        replacement: Any = not expected
    elif isinstance(expected, (int, float)):
        replacement = expected + 1
    else:
        replacement = f"{expected}-mismatch"
    settings = sanitize_settings(FakeClient().getSettingsString())
    target = settings["Vehicles"]["SimpleFlight"]
    if scope == "sensor":
        target = target["Sensors"]["LidarSensor1"]
    target[field] = replacement

    comparisons = capabilities.compare_lidar_settings_profile(settings)
    mismatch = next(
        item for item in comparisons if item.scope == scope and item.field == field
    )

    assert mismatch.expected == expected
    assert mismatch.actual == replacement
    assert mismatch.present is True
    assert mismatch.matched is False


def test_grounded_lidar_settings_mismatch_blocks_scan_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    original = client.getSettingsString

    def mismatched_settings() -> str:
        settings = json.loads(original())
        settings["Vehicles"]["SimpleFlight"]["Sensors"]["LidarSensor1"]["Range"] = 19
        return json.dumps(settings)

    monkeypatch.setattr(client, "getSettingsString", mismatched_settings)

    observations, data = probe_grounded_lidar(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        GroundedLidarProbeConfig(
            vehicle_name="SimpleFlight",
            lidar_name="LidarSensor1",
            scan_count=2,
            scan_interval=0.0,
            warm_up_attempts=1,
            warm_up_interval=0.0,
            confirm_no_visible_collision=True,
        ),
        sleep_fn=lambda _: None,
    )

    mismatch = next(
        item for item in data["settings_comparisons"] if item.field == "Range"
    )
    assert mismatch.expected == 20
    assert mismatch.actual == 19
    assert mismatch.matched is False
    assert data["ready_for_airborne_validation"] is False
    assert observations[-1].status is CapabilityStatus.REQUIRES_LOCAL_CONFIGURATION
    assert not any(
        isinstance(call, tuple) and call[0] == "getLidarData" for call in client.calls
    )


def test_scene_survey_is_bounded_and_never_calls_mesh_buffers() -> None:
    client = FakeClient()

    observations, data = survey_scene(
        client,
        SceneSurveyConfig(max_objects=1),
        client_module=FAKE_MODULE,
        sleep_fn=lambda _: None,
    )

    assert data["scene_object_count"] == 1
    assert len(data["scene_objects"]) == 1
    assert data["mesh_buffer_probe"] == "deferred_not_invoked"
    assert data["measured_linear_velocity"] == (0.0, 0.0, 0.0)
    measured = data["measured_state"]
    assert measured["position"] == (1.0, 2.0, 0.56)
    assert measured["linear_velocity"] == (0.0, 0.0, 0.0)
    assert measured["landed_state_value"] == 0
    assert measured["landed_state_label"] == "Landed"
    assert measured["is_landed"] is True
    assert measured["collision"] is False
    assert measured["api_control_enabled"] is False
    assert measured["speed"] == 0.0
    assert measured["grounded_speed_tolerance"] == 0.1
    assert measured["safe_for_later_stages"] is True
    assert measured["armed_state"]["availability"] == "unavailable"
    assert len(measured["collision_samples"]) == 3
    assert (
        measured["collision_assessment"].classification
        is CollisionClassification.NO_COLLISION
    )
    assert any(
        observation.capability == "collision_state"
        and observation.evidence_level is EvidenceLevel.LIVE_RPC
        and observation.status is CapabilityStatus.RPC_SUCCEEDED
        for observation in observations
    )
    assert any(
        observation.capability == "api_control_state"
        and observation.evidence_level is EvidenceLevel.LIVE_RPC
        and observation.status is CapabilityStatus.RPC_SUCCEEDED
        for observation in observations
    )
    assert any(
        observation.capability == "measured_grounded_state"
        and observation.evidence_level is EvidenceLevel.PRACTICAL_BEHAVIOR
        for observation in observations
    )
    assert all(call != "simGetMeshPositionVertexBuffers" for call in client.calls)
    assert any(item.capability == "asset_listing" for item in observations)


def test_scene_survey_propagates_explicit_simpleflight_vehicle_name() -> None:
    client = FakeClient()

    _, data = survey_scene(
        client,
        SceneSurveyConfig(
            connection=capabilities.ConnectionProbeConfig(vehicle_name="SimpleFlight"),
            max_objects=1,
        ),
        client_module=FAKE_MODULE,
        sleep_fn=lambda _: None,
    )

    assert ("getMultirotorState", "SimpleFlight") in client.calls
    assert ("isApiControlEnabled", "SimpleFlight") in client.calls
    assert client.calls.count(("simGetCollisionInfo", "SimpleFlight")) == 3
    assert data["selected_vehicle_name"] == "SimpleFlight"
    assert data["m13_lidar_profile_matches"] is True
    assert data["measured_state"]["safe_for_later_stages"] is True


def test_scene_survey_rejects_non_finite_linear_velocity() -> None:
    client = FakeClient()
    client.state = make_state(velocity=(float("nan"), 0.0, 0.0))

    with pytest.raises(ValueError, match="measured velocity"):
        survey_scene(
            client,
            SceneSurveyConfig(max_objects=1),
            client_module=FAKE_MODULE,
            sleep_fn=lambda _: None,
        )


def test_scene_survey_accepts_confirmed_historical_ground_contact() -> None:
    client = FakeClient()
    client.collision_samples = [
        make_collision_info(
            has_collided=has_collided,
            object_name="SpawnSurface",
            object_id=17,
            time_stamp=100,
            penetration_depth=0.02,
            impact_point=(1.0, 2.0, 0.58),
            position=(1.0, 2.0, 0.56),
            normal=(0.0, 0.0, -1.0),
        )
        for has_collided in (True, False, False)
    ]

    _, data = survey_scene(
        client,
        SceneSurveyConfig(
            max_objects=1,
            confirm_no_visible_collision=True,
        ),
        client_module=FAKE_MODULE,
        sleep_fn=lambda _: None,
    )

    measured = data["measured_state"]
    assert measured["safe_for_later_stages"] is True
    assert measured["operator_confirmed_no_visible_collision"] is True
    assert measured["collision_assessment"].classification is (
        CollisionClassification.EXPECTED_GROUND_CONTACT
    )
    assert measured["collision_assessment"].baseline_timestamp == 100


def test_collision_classifier_reports_no_collision() -> None:
    samples = tuple(sanitize_collision_info(make_collision_info()) for _ in range(3))

    assessment = classify_collision_samples(
        samples,
        is_landed=True,
        measured_speed=0.0,
    )

    assert assessment.classification is CollisionClassification.NO_COLLISION
    assert assessment.persistent_or_historical is False


def test_collision_classifier_accepts_expected_stationary_ground_contact() -> None:
    samples = make_ground_collision_samples()

    assessment = classify_collision_samples(
        samples,
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.EXPECTED_GROUND_CONTACT
    )
    assert assessment.timestamp_changed is False
    assert assessment.object_changed is False
    assert assessment.object_id_changed is False
    assert assessment.penetration_changed is False
    assert assessment.impact_point_changed is False
    assert assessment.vehicle_position_changed is False
    assert assessment.normal_changed is False
    assert assessment.persistent_or_historical is True
    assert assessment.baseline_timestamp == 100
    assert samples[0].object_name == "SpawnSurface"
    assert samples[0].impact_point == (1.0, 2.0, 0.58)
    assert samples[0].vehicle_position == (1.0, 2.0, 0.56)
    assert samples[0].normal == (0.0, 0.0, -1.0)


@pytest.mark.parametrize(
    "flags",
    [
        (True, False, False),
        (True, True, False),
    ],
)
def test_collision_classifier_accepts_one_shot_ground_contact_decay(
    flags: tuple[bool, bool, bool],
) -> None:
    assessment = classify_collision_samples(
        make_ground_collision_sequence(flags),
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.EXPECTED_GROUND_CONTACT
    )
    assert assessment.persistent_or_historical is True
    assert assessment.baseline_timestamp == 100


def test_collision_classifier_rejects_new_false_to_true_transition() -> None:
    assessment = classify_collision_samples(
        make_ground_collision_sequence((False, True, True)),
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.ACTIVE_OR_UNSAFE_COLLISION
    )
    assert "false-to-true" in assessment.detail


def test_collision_classifier_rejects_changing_timestamps() -> None:
    samples = tuple(
        make_ground_collision_samples(time_stamp=timestamp)[0]
        for timestamp in (100, 101, 102)
    )

    assessment = classify_collision_samples(
        samples,
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.ACTIVE_OR_UNSAFE_COLLISION
    )
    assert assessment.timestamp_changed is True
    assert assessment.baseline_timestamp == 102


def test_collision_classifier_rejects_changing_object_identity() -> None:
    samples = tuple(
        make_ground_collision_samples(object_name=name, object_id=object_id)[0]
        for name, object_id in (
            ("SpawnSurface", 17),
            ("OtherSurface", 18),
            ("OtherSurface", 18),
        )
    )

    assessment = classify_collision_samples(
        samples,
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.ACTIVE_OR_UNSAFE_COLLISION
    )
    assert assessment.object_changed is True
    assert assessment.object_id_changed is True


def test_collision_classifier_rejects_changing_penetration() -> None:
    samples = tuple(
        make_ground_collision_samples(penetration_depth=penetration)[0]
        for penetration in (0.02, 0.03, 0.03)
    )

    assessment = classify_collision_samples(
        samples,
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.ACTIVE_OR_UNSAFE_COLLISION
    )
    assert assessment.penetration_changed is True


def test_collision_classifier_rejects_changing_collision_geometry() -> None:
    samples = tuple(
        make_ground_collision_samples(impact_point=impact)[0]
        for impact in (
            (1.0, 2.0, 0.58),
            (1.1, 2.0, 0.58),
            (1.1, 2.0, 0.58),
        )
    )

    assessment = classify_collision_samples(
        samples,
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.ACTIVE_OR_UNSAFE_COLLISION
    )
    assert assessment.impact_point_changed is True


def test_collision_classifier_rejects_non_ground_collision_geometry() -> None:
    samples = make_ground_collision_samples(
        object_name="WallActor",
        normal=(1.0, 0.0, 0.0),
    )

    assessment = classify_collision_samples(
        samples,
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.ACTIVE_OR_UNSAFE_COLLISION
    )
    assert "ground surface" in assessment.detail


def test_collision_classifier_rejects_excessive_penetration() -> None:
    assessment = classify_collision_samples(
        make_ground_collision_samples(penetration_depth=0.5),
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (
        CollisionClassification.ACTIVE_OR_UNSAFE_COLLISION
    )
    assert "penetration" in assessment.detail


def test_collision_classifier_marks_incomplete_information_inconclusive() -> None:
    incomplete = tuple(
        sanitize_collision_info(
            SimpleNamespace(has_collided=True, object_name="SpawnSurface")
        )
        for _ in range(3)
    )

    assessment = classify_collision_samples(
        incomplete,
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (CollisionClassification.INCONCLUSIVE_COLLISION)
    assert incomplete[0].errors


def test_collision_classifier_is_inconclusive_without_landed_evidence() -> None:
    assessment = classify_collision_samples(
        make_ground_collision_samples(),
        is_landed=None,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=True,
    )

    assert assessment.classification is (CollisionClassification.INCONCLUSIVE_COLLISION)


def test_collision_classifier_requires_operator_confirmation() -> None:
    assessment = classify_collision_samples(
        make_ground_collision_sequence((True, False, False)),
        is_landed=True,
        measured_speed=0.0,
        api_control_enabled=False,
        operator_confirmed_stable=False,
    )

    assert assessment.classification is (CollisionClassification.INCONCLUSIVE_COLLISION)
    assert "Operator confirmation" in assessment.detail


def test_survey_stops_after_timeout_without_later_calls() -> None:
    client = FakeClient()

    def timeout() -> str:
        client.calls.append("getSettingsString")
        raise TimeoutError("RPC timed out")

    client.getSettingsString = timeout  # type: ignore[method-assign]

    with pytest.raises(CapabilityProbeError, match="timed out"):
        survey_scene(client, SceneSurveyConfig(), sleep_fn=lambda _: None)

    assert "listVehicles" not in client.calls
    assert not any(
        isinstance(call, tuple) and call[0] == "simListSceneObjects"
        for call in client.calls
    )


def test_measured_nonzero_ground_reference_builds_relative_anchor() -> None:
    client = FakeClient()
    runtime = ProbeRuntimeState()

    context = prepare_airborne_probe(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        make_airborne_config(),
        runtime,
        TEST_LIDAR_NAME,
    )

    assert context.ground_reference_z == pytest.approx(0.56)
    assert context.anchor_position.z == pytest.approx(-1.44)
    move_call = next(call for call in client.calls if call[0] == "moveToPositionAsync")
    assert move_call[3] == pytest.approx(-1.44)
    assert runtime.vehicle_name == TEST_VEHICLE_NAME
    assert_vehicle_calls_are_named(client.calls)


@pytest.mark.parametrize(
    "disabled_confirmation",
    [
        "allow_flight",
        "confirm_clear_airspace",
        "confirm_no_visible_collision",
        "confirm_grounded_lidar_passed",
    ],
)
def test_airborne_probe_requires_each_distinct_confirmation(
    disabled_confirmation: str,
) -> None:
    client = FakeClient()
    config = make_airborne_config(**{disabled_confirmation: False})

    with pytest.raises(ValueError, match="requires authorization"):
        prepare_airborne_probe(
            client,
            FAKE_MODULE,  # type: ignore[arg-type]
            config,
            ProbeRuntimeState(),
            TEST_LIDAR_NAME,
        )

    assert not any(
        isinstance(call, tuple) and call[0] == "enableApiControl"
        for call in client.calls
    )


def test_airborne_settings_mismatch_blocks_before_api_control() -> None:
    client = FakeClient()

    with pytest.raises(CapabilityProbeError, match="settings do not match"):
        prepare_airborne_probe(
            client,
            FAKE_MODULE,  # type: ignore[arg-type]
            make_airborne_config(),
            ProbeRuntimeState(),
            TEST_LIDAR_NAME,
            settings_verification={
                "profile_matches": False,
                "configured_range": 20,
                "comparisons": (),
            },
        )

    assert not any(
        isinstance(call, tuple) and call[0] == "enableApiControl"
        for call in client.calls
    )


@pytest.mark.parametrize(
    "state",
    [
        make_state((float("nan"), 0.0, 0.56)),
        make_state(landed=1),
    ],
)
def test_grounded_preflight_rejects_invalid_state(state: Any) -> None:
    with pytest.raises((CapabilityProbeError, ValueError)):
        validate_grounded_preflight(FakeClient(), FAKE_MODULE, state)  # type: ignore[arg-type]


def test_grounded_preflight_rejects_unsafe_collision_assessment() -> None:
    client = FakeClient()
    client.collision_samples = [
        make_collision_info(
            True,
            object_name="WallActor",
            object_id=4,
            time_stamp=100,
            penetration_depth=0.02,
            impact_point=(1.0, 2.0, 0.56),
            position=(1.0, 2.0, 0.56),
            normal=(1.0, 0.0, 0.0),
        )
    ]

    with pytest.raises(CapabilityProbeError, match="unsafe or inconclusive"):
        validate_grounded_preflight(client, FAKE_MODULE, client.state)  # type: ignore[arg-type]


def test_airborne_safety_rejects_collision_newer_than_ground_baseline() -> None:
    client = FakeClient()
    client.state = make_state((1.0, 2.0, -1.44), landed=1)
    client.collision_samples = [
        make_collision_info(
            True,
            object_name="ObstacleActor",
            object_id=8,
            time_stamp=101,
            penetration_depth=0.02,
            impact_point=(1.0, 2.0, -1.44),
            position=(1.0, 2.0, -1.44),
            normal=(1.0, 0.0, 0.0),
        )
    ]

    with pytest.raises(CapabilityProbeError, match="new or ambiguous collision"):
        capabilities.monitor_airborne_safety(
            client,
            client.state,
            0.56,
            Position3D(1.0, 2.0, -1.44),
            AirborneProbeConfig(vehicle_name=TEST_VEHICLE_NAME),
            collision_baseline_timestamp=100,
        )


def test_anchor_config_rejects_insufficient_clearance() -> None:
    with pytest.raises(ValueError, match="clearance"):
        AirborneProbeConfig(
            vehicle_name=TEST_VEHICLE_NAME,
            anchor_altitude=0.5,
            min_ground_clearance=1.0,
        )


def test_safe_spawn_position_uses_arbitrary_measured_position() -> None:
    measured = Position3D(10.0, -4.0, 0.56)

    result = derive_safe_probe_position(measured, (1.0, 2.0, -0.5))

    assert result.x == pytest.approx(11.0)
    assert result.y == pytest.approx(-2.0)
    assert result.z == pytest.approx(0.06)


def test_scene_mutation_tracks_exact_name_and_disables_physics() -> None:
    client = FakeClient()
    runtime = ProbeRuntimeState()
    waits: list[float] = []
    messages: list[str] = []
    expected_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    config = SceneMutationConfig(
        asset_name="1M_Cube_Chamfer",
        allow_scene_mutation=True,
        confirm_spawn_area_clear=True,
        confirm_vehicle_disarmed=True,
        material_name=TEST_RED_MATERIAL,
        mutation_hold_seconds=5.0,
    )

    _, data = probe_scene_mutation(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        config,
        runtime,
        uuid_factory=lambda: expected_uuid,
        sleep_fn=waits.append,
        message_fn=messages.append,
    )

    expected_name = "r1_uav_m13_probe_12345678123456781234567812345678"
    assert runtime.created_objects == [expected_name]
    assert data["returned_name"] == expected_name
    spawn_call = next(call for call in client.calls if call[0] == "simSpawnObject")
    assert spawn_call[-2:] == (False, False)
    assert waits == [5.0, 5.0, 5.0]
    assert messages == [
        "Temporary cube spawned; visible for 5 seconds before the next mutation step.",
        "Temporary cube moved; visible for 5 seconds before the next mutation step.",
        "Temporary cube resized; visible for 5 seconds before the next mutation step.",
    ]
    material_calls = [
        call for call in client.calls if call[0] == "simSetObjectMaterial"
    ]
    assert material_calls == [
        ("simSetObjectMaterial", expected_name, TEST_RED_MATERIAL, 0)
    ]
    assert data["material"] == {
        "requested_name": TEST_RED_MATERIAL,
        "assignment_attempted": True,
        "assignment_rpc_succeeded": True,
        "assignment_result": True,
        "readback_availability": "unavailable",
        "readback_detail": (
            "The validated client exposes no reliable material read-back API; "
            "appearance requires operator confirmation."
        ),
    }


@pytest.mark.parametrize("hold_seconds", [-0.1, 10.1, float("nan"), float("inf")])
def test_mutation_hold_duration_is_rejected_before_client_import(
    hold_seconds: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    imported = False

    def fail_if_imported(_: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("invalid hold must be rejected before client import")

    monkeypatch.setattr(module, "load_client_module", fail_if_imported)
    args = module.parse_args(
        [
            "mutation",
            "--allow-scene-mutation",
            "--confirm-spawn-area-clear",
            "--confirm-vehicle-disarmed",
            "--asset-name",
            "Cube",
            "--mutation-hold-seconds",
            str(hold_seconds),
            "--output-path",
            str(tmp_path / "invalid-hold.json"),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 1
    assert imported is False


@pytest.mark.parametrize(
    "material_name",
    [
        "",
        " SolidRed",
        "C:/local/material",
        "/AirSim/Materials/First.Second",
    ],
)
def test_material_argument_is_rejected_before_client_import(
    material_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    imported = False

    def fail_if_imported(_: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("invalid material must be rejected before client import")

    monkeypatch.setattr(module, "load_client_module", fail_if_imported)
    args = module.parse_args(
        [
            "mutation",
            "--allow-scene-mutation",
            "--confirm-spawn-area-clear",
            "--confirm-vehicle-disarmed",
            "--asset-name",
            "Cube",
            "--material-name",
            material_name,
            "--output-path",
            str(tmp_path / "invalid-material.json"),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 1
    assert imported is False


@pytest.mark.parametrize("interrupted_hold", [1, 2, 3])
def test_mutation_interruption_during_each_hold_cleans_exact_object(
    interrupted_hold: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    client = FakeClient()
    client.assets.append("Cube")
    real_probe = capabilities.probe_scene_mutation
    hold_count = 0

    monkeypatch.setattr(module, "load_client_module", lambda _: FAKE_MODULE)
    monkeypatch.setattr(module, "create_probe_client", lambda *_: client)

    def interrupting_probe(*args: Any, **kwargs: Any) -> Any:
        def interrupt_on_selected_hold(_: float) -> None:
            nonlocal hold_count
            hold_count += 1
            if hold_count == interrupted_hold:
                raise KeyboardInterrupt

        kwargs["sleep_fn"] = interrupt_on_selected_hold
        kwargs["message_fn"] = lambda _: None
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(module, "probe_scene_mutation", interrupting_probe)
    output = tmp_path / f"mutation-interruption-{interrupted_hold}.json"
    args = module.parse_args(
        [
            "mutation",
            "--allow-scene-mutation",
            "--confirm-spawn-area-clear",
            "--confirm-vehicle-disarmed",
            "--asset-name",
            "Cube",
            "--material-name",
            TEST_RED_MATERIAL,
            "--mutation-hold-seconds",
            "5",
            "--output-path",
            str(output),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 130
    assert not any(name.startswith("r1_uav_m13_probe_") for name in client.objects)
    report = json.loads(output.read_text(encoding="utf-8"))
    object_cleanup = next(
        result for result in report["cleanup_results"] if result["domain"] == "objects"
    )
    assert object_cleanup == {
        "attempted": True,
        "domain": "objects",
        "errors": [],
        "succeeded": True,
    }
    assert any(call[0] == "simDestroyObject" for call in client.calls)


@pytest.mark.parametrize("returned_name", ["", "unexpected_name"])
def test_empty_or_unexpected_spawn_name_is_rejected(returned_name: str) -> None:
    client = FakeClient()
    client.returned_spawn_name = returned_name
    runtime = ProbeRuntimeState()
    config = SceneMutationConfig(
        asset_name="1M_Cube_Chamfer",
        allow_scene_mutation=True,
        confirm_spawn_area_clear=True,
        confirm_vehicle_disarmed=True,
    )

    with pytest.raises(CapabilityProbeError):
        probe_scene_mutation(
            client, FAKE_MODULE, config, runtime  # type: ignore[arg-type]
        )

    if returned_name:
        assert runtime.created_objects == [returned_name]


def test_exact_temporary_object_disappearance_is_confirmed() -> None:
    client = FakeClient()
    runtime = ProbeRuntimeState(created_objects=["probe+[1]"])
    client.objects["probe+[1]"] = (FakePose(), FakeVector(1, 1, 1))

    results = cleanup_probe_domains(client, runtime)

    object_result = next(result for result in results if result.domain == "objects")
    assert object_result.succeeded is True
    assert "probe+[1]" not in client.objects
    query = next(
        call[1]
        for call in client.calls
        if isinstance(call, tuple)
        and call[0] == "simListSceneObjects"
        and "probe" in call[1]
    )
    assert query == r"^probe\+\[1\]$"


def test_marker_positions_derive_from_measured_state_and_cleanup() -> None:
    client = FakeClient()
    runtime = ProbeRuntimeState()
    waits: list[float] = []
    messages: list[str] = []

    observations = probe_debug_markers(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        Position3D(10.0, 20.0, 0.56),
        runtime,
        allow_debug_markers=True,
        allow_marker_flush=True,
        marker_hold_seconds=8.0,
        sleep_fn=waits.append,
        message_fn=messages.append,
    )
    cleanup = cleanup_probe_domains(client, runtime)

    points_call = next(call for call in client.calls if call[0] == "simPlotPoints")
    point = points_call[1][0][0]
    assert (point.x_val, point.y_val, point.z_val) == pytest.approx((11.5, 20.0, 0.06))
    assert all(item.operator_confirmation == "pending" for item in observations)
    assert waits == [8.0]
    assert messages == [
        "Debug markers are currently visible for 8 seconds before cleanup."
    ]
    assert next(result for result in cleanup if result.domain == "markers").succeeded


@pytest.mark.parametrize("hold_seconds", [-0.1, 15.1, float("nan"), float("inf")])
def test_marker_hold_duration_rejects_unsafe_values(hold_seconds: float) -> None:
    client = FakeClient()

    with pytest.raises(ValueError, match="marker_hold_seconds"):
        probe_debug_markers(
            client,
            FAKE_MODULE,  # type: ignore[arg-type]
            Position3D(1.0, 2.0, 0.56),
            ProbeRuntimeState(),
            allow_debug_markers=True,
            allow_marker_flush=True,
            marker_hold_seconds=hold_seconds,
            sleep_fn=lambda _: pytest.fail("invalid duration must not sleep"),
            message_fn=lambda _: pytest.fail("invalid duration must not print"),
        )

    assert not any(
        isinstance(call, tuple) and call[0].startswith("simPlot")
        for call in client.calls
    )


def test_marker_interrupt_during_hold_still_flushes_in_finally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script_module()
    client = FakeClient()
    real_probe = capabilities.probe_debug_markers

    monkeypatch.setattr(module, "load_client_module", lambda _: FAKE_MODULE)
    monkeypatch.setattr(module, "create_probe_client", lambda *_: client)

    def interrupting_probe(*args: Any, **kwargs: Any) -> Any:
        kwargs["sleep_fn"] = lambda _: (_ for _ in ()).throw(KeyboardInterrupt)
        kwargs["message_fn"] = lambda _: None
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(module, "probe_debug_markers", interrupting_probe)
    output = tmp_path / "marker-interruption.json"
    args = module.parse_args(
        [
            "markers",
            "--allow-debug-markers",
            "--allow-marker-flush",
            "--marker-hold-seconds",
            "8",
            "--output-path",
            str(output),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 130
    assert "simFlushPersistentMarkers" in client.calls
    report = json.loads(output.read_text(encoding="utf-8"))
    marker_cleanup = next(
        result for result in report["cleanup_results"] if result["domain"] == "markers"
    )
    assert marker_cleanup == {
        "attempted": True,
        "domain": "markers",
        "errors": [],
        "succeeded": True,
    }


def test_cleanup_domains_remain_independent_when_uav_cleanup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    client.objects["probe"] = (FakePose(), FakeVector(1, 1, 1))
    runtime = ProbeRuntimeState(
        cleanup_state=CleanupState(api_control_enabled=True),
        vehicle_name=TEST_VEHICLE_NAME,
        created_objects=["probe"],
        markers_created=True,
    )
    monkeypatch.setattr(
        capabilities,
        "cleanup_named_probe_vehicle",
        lambda *_: (_ for _ in ()).throw(RuntimeError("uav cleanup failed")),
    )

    results = cleanup_probe_domains(client, runtime)

    assert (
        next(result for result in results if result.domain == "uav").succeeded is False
    )
    assert next(result for result in results if result.domain == "objects").succeeded
    assert next(result for result in results if result.domain == "markers").succeeded


def test_named_cleanup_uses_safe_order_and_never_targets_default_vehicle() -> None:
    client = FakeClient()
    runtime = ProbeRuntimeState(
        cleanup_state=CleanupState(
            api_control_enabled=True,
            armed=True,
            takeoff_attempted=True,
            airborne=True,
        ),
        vehicle_name=TEST_VEHICLE_NAME,
    )

    results = cleanup_probe_domains(client, runtime)

    assert next(result for result in results if result.domain == "uav").succeeded
    cleanup_calls = [
        call
        for call in client.calls
        if isinstance(call, tuple)
        and call[0] in {"hoverAsync", "landAsync", "armDisarm", "enableApiControl"}
    ]
    assert cleanup_calls == [
        ("hoverAsync", TEST_VEHICLE_NAME),
        ("landAsync", TEST_VEHICLE_NAME),
        ("armDisarm", False, TEST_VEHICLE_NAME),
        ("enableApiControl", False, TEST_VEHICLE_NAME),
    ]
    assert_vehicle_calls_are_named(client.calls)


def test_named_cleanup_continues_after_landing_failure() -> None:
    client = FakeClient()

    def failed_land(vehicle_name: str = "") -> FakeAsyncResult:
        client.calls.append(("landAsync", vehicle_name))
        raise RuntimeError("landing failed")

    client.landAsync = failed_land  # type: ignore[method-assign]
    runtime = ProbeRuntimeState(
        cleanup_state=CleanupState(
            api_control_enabled=True,
            armed=True,
            takeoff_attempted=True,
            airborne=True,
        ),
        vehicle_name=TEST_VEHICLE_NAME,
    )

    results = cleanup_probe_domains(client, runtime)

    uav_result = next(result for result in results if result.domain == "uav")
    assert uav_result.succeeded is False
    assert ("armDisarm", False, TEST_VEHICLE_NAME) in client.calls
    assert ("enableApiControl", False, TEST_VEHICLE_NAME) in client.calls
    assert_vehicle_calls_are_named(client.calls)


def test_marker_cleanup_runs_when_object_destruction_raises() -> None:
    client = FakeClient()
    client.destroy_raises = True
    runtime = ProbeRuntimeState(created_objects=["probe"], markers_created=True)

    results = cleanup_probe_domains(client, runtime)

    assert (
        next(result for result in results if result.domain == "objects").succeeded
        is False
    )
    assert next(result for result in results if result.domain == "markers").succeeded
    assert "simFlushPersistentMarkers" in client.calls


@pytest.mark.parametrize(
    ("cloud", "expected_error"),
    [
        ([], "empty"),
        ([1.0, 2.0], "divisible"),
        ([1.0, 0.0, float("nan")], "non-finite"),
    ],
)
def test_invalid_lidar_clouds_are_classified(
    cloud: list[float], expected_error: str
) -> None:
    summary = validate_lidar_scan(
        SimpleNamespace(point_cloud=cloud, time_stamp=1, pose=FakePose()),
        "LidarSensor1",
    )

    assert summary.valid is False
    assert expected_error in (summary.error or "")


def test_valid_lidar_point_triples_report_ranges() -> None:
    summary = validate_lidar_scan(
        SimpleNamespace(
            point_cloud=[3.0, 4.0, 0.0, 0.0, 0.0, 2.0],
            time_stamp=42,
            pose=FakePose(FakeVector(1.0, 2.0, 3.0)),
        ),
        "LidarSensor1",
    )

    assert summary.valid is True
    assert summary.point_count == 2
    assert summary.minimum_range == pytest.approx(2.0)
    assert summary.maximum_range == pytest.approx(5.0)
    assert summary.sensor_position == (1.0, 2.0, 3.0)


def test_grounded_lidar_warm_up_is_excluded_from_measured_scans() -> None:
    client = FakeClient()
    client.lidar_scans = [
        make_lidar_scan([], timestamp=1),
        make_lidar_scan([1.0, 2.0], timestamp=2),
        make_lidar_scan([1.0, 0.0, 0.0], timestamp=3),
        *[make_lidar_scan([1.0, 0.0, 0.0], timestamp=10 + index) for index in range(4)],
    ]
    ticks = iter((0.0, 0.4))

    observations, data = probe_grounded_lidar(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        GroundedLidarProbeConfig(
            vehicle_name="SimpleFlight",
            lidar_name="LidarSensor1",
            scan_count=4,
            scan_interval=0.0,
            warm_up_attempts=3,
            warm_up_interval=0.0,
            confirm_no_visible_collision=True,
        ),
        sleep_fn=lambda _: None,
        clock=lambda: next(ticks),
    )

    assert data["warm_up"] == {
        "attempts": 3,
        "empty_count": 1,
        "invalid_count": 1,
        "time_to_first_valid_scan": pytest.approx(0.4),
        "first_valid_timestamp": 3,
        "succeeded": True,
        "excluded_from_measured_statistics": True,
    }
    assert [scan.timestamp for scan in data["measured_scans"]] == [10, 11, 12, 13]
    assert data["timestamp_summary"].fresh_transition_count == 3
    assert data["ready_for_airborne_validation"] is True
    assert observations[-1].status is CapabilityStatus.RPC_SUCCEEDED
    assert client.calls.count(("getLidarData", "LidarSensor1", "SimpleFlight")) == 7


def test_grounded_lidar_warm_up_exhaustion_stops_before_measurement() -> None:
    client = FakeClient()
    client.lidar_scans = [
        make_lidar_scan([], timestamp=1),
        make_lidar_scan([1.0, 2.0], timestamp=2),
    ]

    observations, data = probe_grounded_lidar(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        GroundedLidarProbeConfig(
            vehicle_name="SimpleFlight",
            lidar_name="LidarSensor1",
            scan_count=2,
            scan_interval=0.0,
            warm_up_attempts=2,
            warm_up_interval=0.0,
            confirm_no_visible_collision=True,
        ),
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
    )

    assert data["warm_up"]["succeeded"] is False
    assert data["warm_up"]["empty_count"] == 1
    assert data["warm_up"]["invalid_count"] == 1
    assert data["measured_scans"] == ()
    assert data["ready_for_airborne_validation"] is False
    assert observations[-1].status is CapabilityStatus.REQUIRES_LOCAL_CONFIGURATION
    assert client.calls.count(("getLidarData", "LidarSensor1", "SimpleFlight")) == 2


@pytest.mark.parametrize(
    ("cloud", "error"),
    [
        ([], "empty"),
        ([1.0, 2.0], "divisible"),
        ([float("nan"), 0.0, 0.0], "non-finite"),
    ],
)
def test_grounded_lidar_rejects_invalid_clouds(cloud: list[float], error: str) -> None:
    summary = validate_grounded_lidar_scan(
        make_lidar_scan(cloud, timestamp=1),
        sensor_name="LidarSensor1",
        vehicle_name="SimpleFlight",
        configured_range=20.0,
    )

    assert summary.valid is False
    assert error in (summary.error or "")


def test_grounded_lidar_uses_only_read_only_calls_and_exact_names() -> None:
    client = FakeClient()
    client.lidar_scans = [
        make_lidar_scan(timestamp=1),
        make_lidar_scan(timestamp=2),
        make_lidar_scan(timestamp=3),
    ]

    _, data = probe_grounded_lidar(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        GroundedLidarProbeConfig(
            vehicle_name="SimpleFlight",
            lidar_name="LidarSensor1",
            scan_count=2,
            scan_interval=0.0,
            warm_up_attempts=1,
            warm_up_interval=0.0,
            confirm_no_visible_collision=True,
        ),
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
    )

    forbidden_names = {
        "enableApiControl",
        "armDisarm",
        "takeoffAsync",
        "hoverAsync",
        "moveToPositionAsync",
        "moveByVelocityAsync",
        "rotateToYawAsync",
        "reset",
        "simPause",
    }
    assert not any(
        (call if isinstance(call, str) else call[0]) in forbidden_names
        for call in client.calls
    )
    assert ("getMultirotorState", "SimpleFlight") in client.calls
    assert ("isApiControlEnabled", "SimpleFlight") in client.calls
    assert client.calls.count(("simGetCollisionInfo", "SimpleFlight")) == 3
    assert data["vehicle_name"] == "SimpleFlight"
    assert data["lidar_name"] == "LidarSensor1"


@pytest.mark.parametrize(
    ("timestamps", "expected_ready", "expected_run", "expected_regressions"),
    [
        ((10, 10, 10, 10), True, 3, 0),
        ((10, 10, 10, 10, 10), False, 4, 0),
        ((10, 11, 9, 12), False, 0, 1),
    ],
)
def test_grounded_lidar_timestamp_gate(
    timestamps: tuple[int, ...],
    expected_ready: bool,
    expected_run: int,
    expected_regressions: int,
) -> None:
    client = FakeClient()
    client.lidar_scans = [
        make_lidar_scan(timestamp=1),
        *[make_lidar_scan(timestamp=value) for value in timestamps],
    ]

    _, data = probe_grounded_lidar(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        GroundedLidarProbeConfig(
            vehicle_name="SimpleFlight",
            lidar_name="LidarSensor1",
            scan_count=len(timestamps),
            scan_interval=0.0,
            stale_threshold=3,
            warm_up_attempts=1,
            warm_up_interval=0.0,
            confirm_no_visible_collision=True,
        ),
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
    )

    assert data["ready_for_airborne_validation"] is expected_ready
    assert data["timestamp_summary"].maximum_repeated_timestamp_run == expected_run
    assert data["timestamp_summary"].regression_count == expected_regressions


@pytest.mark.parametrize(
    ("timestamps", "fresh", "repeated", "regressions", "maximum_run"),
    [
        ((1, 2, 3), 2, 0, 0, 0),
        ((1, 1, 2), 1, 1, 0, 1),
        ((2, 1), 0, 0, 1, 0),
        ((1, 1, 1, 1), 0, 3, 0, 3),
        ((1, 1, 1, 1, 1), 0, 4, 0, 4),
    ],
)
def test_lidar_timestamp_transition_semantics(
    timestamps: tuple[int, ...],
    fresh: int,
    repeated: int,
    regressions: int,
    maximum_run: int,
) -> None:
    summary = analyze_lidar_timestamps(timestamps)

    assert summary.unique_timestamp_count == len(set(timestamps))
    assert summary.fresh_transition_count == fresh
    assert summary.repeated_transition_count == repeated
    assert summary.regression_count == regressions
    assert summary.maximum_repeated_timestamp_run == maximum_run


@pytest.mark.parametrize(
    ("distance", "valid", "beyond"),
    [
        (20.0, True, 0),
        (20.1, True, 0),
        (20.1001, False, 1),
    ],
)
def test_grounded_lidar_range_gate(distance: float, valid: bool, beyond: int) -> None:
    summary = validate_grounded_lidar_scan(
        make_lidar_scan([distance, 0.0, 0.0], timestamp=1),
        sensor_name="LidarSensor1",
        vehicle_name="SimpleFlight",
        configured_range=20.0,
    )

    assert summary.valid is valid
    assert summary.beyond_configured_range_count == beyond
    assert summary.minimum_range == pytest.approx(distance)
    assert summary.maximum_range == pytest.approx(distance)


@pytest.mark.parametrize(
    "pose",
    [
        FakePose(FakeVector(float("nan"), 0.0, 0.0)),
        FakePose(orientation_val=FakeQuaternion(w=float("inf"))),
    ],
)
def test_grounded_lidar_rejects_invalid_sensor_pose(pose: FakePose) -> None:
    summary = validate_grounded_lidar_scan(
        make_lidar_scan([1.0, 0.0, 0.0], timestamp=1, pose=pose),
        sensor_name="LidarSensor1",
        vehicle_name="SimpleFlight",
        configured_range=20.0,
    )

    assert summary.valid is False
    assert "pose" in (summary.error or "")


def test_grounded_lidar_reports_near_field_self_hit_evidence() -> None:
    client = FakeClient()
    client.lidar_scans = [
        make_lidar_scan([1.0, 0.0, 0.0], timestamp=1),
        make_lidar_scan([0.04, 0.0, 0.0, 0.2, 0.0, 0.0], timestamp=2),
        make_lidar_scan([0.04, 0.0, 0.0, 1.0, 0.0, 0.0], timestamp=3),
    ]

    _, data = probe_grounded_lidar(
        client,
        FAKE_MODULE,  # type: ignore[arg-type]
        GroundedLidarProbeConfig(
            vehicle_name="SimpleFlight",
            lidar_name="LidarSensor1",
            scan_count=2,
            scan_interval=0.0,
            warm_up_attempts=1,
            warm_up_interval=0.0,
            confirm_no_visible_collision=True,
        ),
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
    )

    below_005, below_010, below_025 = data["near_field"]
    assert below_005["point_count"] == 2
    assert below_005["proportion"] == pytest.approx(0.5)
    assert below_005["scans_with_points"] == 2
    assert below_005["recurs_across_scans"] is True
    assert below_010["point_count"] == 2
    assert below_025["point_count"] == 3
    assert data["near_returns_recur"] is True
    assert data["self_hit_classification"] is SelfHitClassification.POSSIBLE_SELF_HIT
    assert data["ready_for_airborne_validation"] is False


def test_lidar_probe_distinguishes_missing_configuration_and_stale_scans() -> None:
    context = make_airborne_context()
    config = LidarProbeConfig(
        airborne=make_airborne_config(),
        lidar_name=TEST_LIDAR_NAME,
        scan_count=4,
        scan_interval=0.0,
        stale_threshold=2,
        warm_up_attempts=1,
        warm_up_interval=0.0,
        settle_interval=0.0,
    )
    empty_client = SimpleNamespace(
        getLidarData=lambda *_: SimpleNamespace(
            point_cloud=[], time_stamp=0, pose=FakePose()
        )
    )

    empty_observations, _ = probe_lidar(empty_client, config, context)

    assert empty_observations[0].status is CapabilityStatus.SUPPORTED_WITH_LIMITATIONS

    stale_client = SimpleNamespace(
        getLidarData=lambda *_: SimpleNamespace(
            point_cloud=[1.0, 0.0, 0.0], time_stamp=7, pose=FakePose()
        )
    )
    stale_observations, stale_data = probe_lidar(stale_client, config, context)

    assert stale_observations[0].status is CapabilityStatus.SUPPORTED_WITH_LIMITATIONS
    assert stale_data["timestamp_summary"].maximum_repeated_timestamp_run == 3


@pytest.mark.parametrize(
    ("measured_scans", "scan_count"),
    [
        (
            [
                make_lidar_scan([], timestamp=2),
                make_lidar_scan(timestamp=3),
            ],
            2,
        ),
        (
            [
                make_lidar_scan([1.0, 2.0], timestamp=2),
                make_lidar_scan(timestamp=3),
            ],
            2,
        ),
        (
            [
                make_lidar_scan([float("nan"), 0.0, 0.0], timestamp=2),
                make_lidar_scan(timestamp=3),
            ],
            2,
        ),
        (
            [
                make_lidar_scan(
                    timestamp=2,
                    pose=FakePose(position_val=FakeVector(float("nan"), 0.0, 0.0)),
                ),
                make_lidar_scan(timestamp=3),
            ],
            2,
        ),
        (
            [
                make_lidar_scan([20.1001, 0.0, 0.0], timestamp=2),
                make_lidar_scan(timestamp=3),
            ],
            2,
        ),
        (
            [
                make_lidar_scan(timestamp=3),
                make_lidar_scan(timestamp=2),
            ],
            2,
        ),
        (
            [make_lidar_scan(timestamp=2) for _ in range(5)],
            5,
        ),
    ],
    ids=[
        "empty",
        "malformed",
        "non-finite",
        "invalid-pose",
        "out-of-range",
        "timestamp-regression",
        "stale-run",
    ],
)
def test_every_airborne_measured_scan_must_pass_strict_gate(
    measured_scans: list[Any],
    scan_count: int,
) -> None:
    client = FakeClient()
    client.lidar_scans = [make_lidar_scan(timestamp=1), *measured_scans]

    observations, data = probe_lidar(
        client,
        make_lidar_config(scan_count=scan_count),
        make_airborne_context(),
        sleep_fn=lambda _: None,
    )

    assert observations[0].status is CapabilityStatus.SUPPORTED_WITH_LIMITATIONS
    assert data["airborne_scan_gate_passed"] is False
    assert data["valid_scan_count"] < scan_count or (
        data["timestamp_summary"].regression_count > 0
        or data["timestamp_summary"].maximum_repeated_timestamp_run > 3
    )
    assert_vehicle_calls_are_named(client.calls)


def test_airborne_warm_up_is_excluded_from_measured_statistics() -> None:
    client = FakeClient()
    client.lidar_scans = [
        make_lidar_scan([0.04, 0.0, 0.0], timestamp=1),
        make_lidar_scan([1.0, 0.0, 0.0], timestamp=2),
        make_lidar_scan([2.0, 0.0, 0.0], timestamp=3),
    ]

    _, data = probe_lidar(
        client,
        make_lidar_config(),
        make_airborne_context(),
        sleep_fn=lambda _: None,
    )

    assert data["warm_up"]["excluded_from_measured_statistics"] is True
    assert data["point_counts"] == (1, 1)
    assert data["global_minimum_range"] == pytest.approx(1.0)
    assert data["near_field"][0]["point_count"] == 0
    assert data["airborne_scan_gate_passed"] is True


def test_lidar_visualization_is_disabled_by_default() -> None:
    client = FakeClient()
    client.lidar_scans = [
        make_lidar_scan(timestamp=1),
        make_lidar_scan(timestamp=2),
        make_lidar_scan(timestamp=3),
    ]

    _, data = probe_lidar(
        client,
        make_lidar_config(),
        make_airborne_context(),
        sleep_fn=lambda _: None,
    )

    assert data["visualization"] == {
        "requested": False,
        "performed": False,
    }
    assert not any(
        isinstance(call, tuple) and call[0].startswith("simPlot")
        for call in client.calls
    )


def test_sensor_local_points_transform_to_world_coordinates() -> None:
    result = capabilities.transform_sensor_local_points_to_world(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        (10.0, 20.0, 30.0),
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
    )

    assert [(point.x, point.y, point.z) for point in result] == pytest.approx(
        [
            (10.0, 21.0, 30.0),
            (9.0, 20.0, 30.0),
        ]
    )


def test_lidar_visualization_uses_bounded_even_samples_and_independent_cleanup() -> (
    None
):
    client = FakeClient()
    client.state = make_state((1.0, 2.0, -1.44), landed=1)
    cloud = [
        coordinate
        for index in range(100)
        for coordinate in (1.0 + index * 0.01, 0.0, 0.0)
    ]
    sensor_pose = FakePose(FakeVector(1.0, 2.0, -1.44))
    client.lidar_scans = [
        make_lidar_scan(cloud, timestamp=1, pose=sensor_pose),
        make_lidar_scan(cloud, timestamp=2, pose=sensor_pose),
        make_lidar_scan(cloud, timestamp=3, pose=sensor_pose),
    ]
    runtime = ProbeRuntimeState(
        cleanup_state=CleanupState(
            api_control_enabled=True,
            armed=True,
            takeoff_attempted=True,
            airborne=True,
        ),
        vehicle_name=TEST_VEHICLE_NAME,
    )
    waits: list[float] = []
    messages: list[str] = []
    config = make_lidar_config(
        visualize_lidar=True,
        allow_marker_flush=True,
        visualization_hold_seconds=8.0,
        visualization_max_points=10,
        visualization_max_rays=3,
    )

    observations, data = probe_lidar(
        client,
        config,
        make_airborne_context(),
        client_module=FAKE_MODULE,  # type: ignore[arg-type]
        runtime=runtime,
        sleep_fn=waits.append,
        message_fn=messages.append,
    )
    cleanup = cleanup_probe_domains(client, runtime)

    point_call = next(call for call in client.calls if call[0] == "simPlotPoints")
    ray_call = next(call for call in client.calls if call[0] == "simPlotLineList")
    plotted_x = [point.x_val for point in point_call[1][0]]
    assert len(plotted_x) == 10
    assert plotted_x[0] == pytest.approx(2.0)
    assert plotted_x[-1] == pytest.approx(2.99)
    assert len(ray_call[1][0]) == 6
    assert data["airborne_scan_gate_passed"] is True
    assert data["visualization"]["source_scan_validated"] is True
    assert data["visualization"]["plotted_point_count"] == 10
    assert data["visualization"]["diagnostic_ray_count"] == 3
    assert data["visualization"]["diagnostic_overlay_not_physical_lasers"] is True
    assert runtime.lidar_visualization_markers_created is True
    assert waits == [8.0]
    assert messages == [
        "The diagnostic LiDAR point/ray overlay is currently visible for "
        "8 seconds before cleanup."
    ]
    assert {observation.capability for observation in observations} >= {
        "lidar_visualization_points",
        "lidar_visualization_rays",
    }
    assert next(result for result in cleanup if result.domain == "uav").succeeded
    assert next(result for result in cleanup if result.domain == "markers").succeeded
    assert ("hoverAsync", TEST_VEHICLE_NAME) in client.calls
    assert ("landAsync", TEST_VEHICLE_NAME) in client.calls
    assert "simFlushPersistentMarkers" in client.calls


def test_lidar_visualization_never_uses_an_invalid_measured_scan() -> None:
    client = FakeClient()
    client.lidar_scans = [
        make_lidar_scan(timestamp=1),
        make_lidar_scan([], timestamp=2),
        make_lidar_scan(timestamp=3),
    ]
    runtime = ProbeRuntimeState()

    _, data = probe_lidar(
        client,
        make_lidar_config(
            visualize_lidar=True,
            allow_marker_flush=True,
        ),
        make_airborne_context(),
        client_module=FAKE_MODULE,  # type: ignore[arg-type]
        runtime=runtime,
        sleep_fn=lambda _: None,
    )

    assert data["airborne_scan_gate_passed"] is False
    assert data["visualization"]["performed"] is False
    assert runtime.lidar_visualization_markers_created is False
    assert not any(
        isinstance(call, tuple) and call[0].startswith("simPlot")
        for call in client.calls
    )


def test_lidar_visualization_plot_failure_preserves_both_cleanup_domains() -> None:
    client = FakeClient()
    client.state = make_state((1.0, 2.0, -1.44), landed=1)
    client.lidar_scans = [
        make_lidar_scan(timestamp=1),
        make_lidar_scan(timestamp=2),
        make_lidar_scan(timestamp=3),
    ]

    def failed_line_plot(*args: Any) -> None:
        client.calls.append(("simPlotLineList", args))
        raise RuntimeError("plot failed")

    client.simPlotLineList = failed_line_plot  # type: ignore[method-assign]
    runtime = ProbeRuntimeState(
        cleanup_state=CleanupState(
            api_control_enabled=True,
            armed=True,
            takeoff_attempted=True,
            airborne=True,
        ),
        vehicle_name=TEST_VEHICLE_NAME,
    )

    with pytest.raises(CapabilityProbeError):
        probe_lidar(
            client,
            make_lidar_config(
                visualize_lidar=True,
                allow_marker_flush=True,
            ),
            make_airborne_context(),
            client_module=FAKE_MODULE,  # type: ignore[arg-type]
            runtime=runtime,
            sleep_fn=lambda _: None,
        )
    cleanup = cleanup_probe_domains(client, runtime)

    assert next(result for result in cleanup if result.domain == "uav").succeeded
    assert next(result for result in cleanup if result.domain == "markers").succeeded
    assert ("landAsync", TEST_VEHICLE_NAME) in client.calls
    assert "simFlushPersistentMarkers" in client.calls


def test_coordinate_experiment_requires_new_scan_and_restores_named_yaw() -> None:
    client = FakeClient()
    client.state = make_state((1.0, 2.0, -1.44), landed=1, yaw_degrees=12.0)
    client.lidar_scans = [
        make_lidar_scan([1.0, 0.0, 0.0], timestamp=11),
        make_lidar_scan(
            [0.0, -1.0, 0.0],
            timestamp=12,
            pose=FakePose(
                orientation_val=FakeQuaternion(
                    w=math.sqrt(0.5),
                    z=math.sqrt(0.5),
                )
            ),
        ),
    ]
    config = make_lidar_config(
        coordinate_frame_experiment=True,
        allow_coordinate_motion=True,
        coordinate_scan_attempts=2,
    )

    frame, data = capabilities._run_coordinate_frame_experiment(
        client,
        config,
        make_airborne_context(),
        last_measured_timestamp=10,
        sleep_fn=lambda _: None,
    )

    assert frame == "SensorLocalFrame"
    assert data["baseline_timestamp"] == 11
    assert data["rotated_timestamp"] == 12
    assert data["returned_yaw"] == pytest.approx(12.0)
    assert data["yaw_error"] == pytest.approx(0.0)
    rotate_calls = [
        call
        for call in client.calls
        if isinstance(call, tuple) and call[0] == "rotateToYawAsync"
    ]
    assert [call[1] for call in rotate_calls] == pytest.approx([57.0, 12.0])
    assert_vehicle_calls_are_named(client.calls)


def test_coordinate_experiment_rejects_stale_post_yaw_scan_and_restores_yaw() -> None:
    client = FakeClient()
    client.state = make_state((1.0, 2.0, -1.44), landed=1, yaw_degrees=-8.0)
    client.lidar_scans = [
        make_lidar_scan(timestamp=11),
        make_lidar_scan(timestamp=11),
        make_lidar_scan(timestamp=11),
    ]
    config = make_lidar_config(
        coordinate_frame_experiment=True,
        allow_coordinate_motion=True,
        coordinate_scan_attempts=2,
    )

    with pytest.raises(CapabilityProbeError, match="newer valid scan"):
        capabilities._run_coordinate_frame_experiment(
            client,
            config,
            make_airborne_context(),
            last_measured_timestamp=10,
            sleep_fn=lambda _: None,
        )

    rotate_calls = [
        call
        for call in client.calls
        if isinstance(call, tuple) and call[0] == "rotateToYawAsync"
    ]
    assert [call[1] for call in rotate_calls] == pytest.approx([37.0, -8.0])
    assert_vehicle_calls_are_named(client.calls)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (
            SimpleNamespace(
                point_cloud=[1.0, 0.0, 0.0],
                pose=FakePose(orientation_val=FakeQuaternion()),
            ),
            SimpleNamespace(
                point_cloud=[0.0, -1.0, 0.0],
                pose=FakePose(
                    orientation_val=FakeQuaternion(w=math.sqrt(0.5), z=math.sqrt(0.5))
                ),
            ),
            "SensorLocalFrame",
        ),
        (
            SimpleNamespace(
                point_cloud=[1.0, 0.0, 0.0],
                pose=FakePose(orientation_val=FakeQuaternion()),
            ),
            SimpleNamespace(
                point_cloud=[1.0, 0.0, 0.0],
                pose=FakePose(
                    orientation_val=FakeQuaternion(w=math.sqrt(0.5), z=math.sqrt(0.5))
                ),
            ),
            "VehicleInertialFrame",
        ),
        (
            SimpleNamespace(point_cloud=[1.0, 0.0, 0.0], pose=FakePose()),
            SimpleNamespace(point_cloud=[1.0, 0.0, 0.0], pose=FakePose()),
            "inconclusive",
        ),
    ],
)
def test_synthetic_lidar_frame_evidence(first: Any, second: Any, expected: str) -> None:
    assert capabilities._classify_lidar_frame(first, second) == expected


def test_keyboard_interrupt_attempts_cleanup_and_stops_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script_module()
    client = FakeClient()

    monkeypatch.setattr(module, "load_client_module", lambda _: FAKE_MODULE)
    monkeypatch.setattr(module, "create_probe_client", lambda *_: client)

    def interrupt_after_takeoff(
        _: Any,
        __: Any,
        ___: Any,
        runtime: ProbeRuntimeState,
        ____: str,
        **_____: Any,
    ) -> None:
        runtime.vehicle_name = TEST_VEHICLE_NAME
        runtime.cleanup_state = CleanupState(
            api_control_enabled=True,
            armed=True,
            takeoff_attempted=True,
            airborne=True,
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "prepare_airborne_probe", interrupt_after_takeoff)
    args = module.parse_args(
        [
            "lidar",
            "--vehicle-name",
            TEST_VEHICLE_NAME,
            "--lidar-name",
            TEST_LIDAR_NAME,
            "--allow-flight",
            "--confirm-clear-airspace",
            "--confirm-no-visible-collision",
            "--confirm-grounded-lidar-passed",
            "--output-path",
            str(tmp_path / "interrupted.json"),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 130
    assert ("hoverAsync", TEST_VEHICLE_NAME) in client.calls
    assert ("landAsync", TEST_VEHICLE_NAME) in client.calls
    assert ("armDisarm", False, TEST_VEHICLE_NAME) in client.calls
    assert ("enableApiControl", False, TEST_VEHICLE_NAME) in client.calls


def test_lidar_visualization_interrupt_flushes_markers_and_cleans_named_uav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    client = FakeClient()
    real_probe = capabilities.probe_lidar

    monkeypatch.setattr(module, "load_client_module", lambda _: FAKE_MODULE)
    monkeypatch.setattr(module, "create_probe_client", lambda *_: client)

    def interrupt_during_visualization(*args: Any, **kwargs: Any) -> Any:
        def bounded_sleep(seconds: float) -> None:
            if seconds == 8.0:
                raise KeyboardInterrupt

        kwargs["sleep_fn"] = bounded_sleep
        kwargs["message_fn"] = lambda _: None
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(module, "probe_lidar", interrupt_during_visualization)
    output = tmp_path / "lidar-visualization-interrupted.json"
    args = module.parse_args(
        [
            "lidar",
            "--vehicle-name",
            TEST_VEHICLE_NAME,
            "--lidar-name",
            TEST_LIDAR_NAME,
            "--allow-flight",
            "--confirm-clear-airspace",
            "--confirm-no-visible-collision",
            "--confirm-grounded-lidar-passed",
            "--scan-count",
            "2",
            "--scan-interval",
            "0",
            "--warm-up-attempts",
            "1",
            "--warm-up-interval",
            "0",
            "--visualize-lidar",
            "--allow-marker-flush",
            "--lidar-visualization-hold-seconds",
            "8",
            "--output-path",
            str(output),
        ]
    )

    assert module.run_probe(args, repository_root=Path.cwd()) == 130
    report = json.loads(output.read_text(encoding="utf-8"))
    cleanup = {item["domain"]: item for item in report["cleanup_results"]}
    assert cleanup["uav"]["attempted"] is True
    assert cleanup["uav"]["succeeded"] is True
    assert cleanup["markers"]["attempted"] is True
    assert cleanup["markers"]["succeeded"] is True
    assert ("hoverAsync", TEST_VEHICLE_NAME) in client.calls
    assert ("landAsync", TEST_VEHICLE_NAME) in client.calls
    assert ("armDisarm", False, TEST_VEHICLE_NAME) in client.calls
    assert ("enableApiControl", False, TEST_VEHICLE_NAME) in client.calls
    assert "simFlushPersistentMarkers" in client.calls


def test_performance_summary_uses_injected_clock() -> None:
    client = FakeClient()
    ticks = iter(float(value) for value in range(20))

    results = probe_performance(
        client,
        PerformanceProbeConfig(vehicle_name=TEST_VEHICLE_NAME, iterations=2),
        clock=lambda: next(ticks),
    )

    assert {result.operation for result in results} == {
        "multirotor_state",
        "scene_object_listing",
    }
    assert all(result.succeeded == 2 for result in results)
    assert_vehicle_calls_are_named(client.calls)


def test_lidar_and_control_performance_use_exact_vehicle_and_sensor_names() -> None:
    client = FakeClient()
    client.state = make_state((1.0, 2.0, -1.44), landed=1)
    ticks = iter(float(value) for value in range(40))
    airborne = make_airborne_config()
    context = make_airborne_context()
    config = PerformanceProbeConfig(
        vehicle_name=TEST_VEHICLE_NAME,
        iterations=2,
        include_lidar=True,
        lidar_name=TEST_LIDAR_NAME,
        include_control=True,
        airborne=airborne,
    )

    results = probe_performance(
        client,
        config,
        airborne_context=context,
        clock=lambda: next(ticks),
    )

    assert all(result.succeeded == 2 for result in results)
    assert ("getLidarData", TEST_LIDAR_NAME, TEST_VEHICLE_NAME) in client.calls
    assert any(
        isinstance(call, tuple)
        and call[0] == "moveByVelocityAsync"
        and call[5] == TEST_VEHICLE_NAME
        for call in client.calls
    )
    assert_vehicle_calls_are_named(client.calls)


def test_default_report_path_is_ignored_and_tracked_paths_are_rejected() -> None:
    default_path = generate_report_path(
        "survey",
        timestamp="2026-07-22T12:00:00+00:00",
        run_id="1234567890abcdef",
    )

    validate_report_output_path(default_path, Path.cwd())
    with pytest.raises(ValueError, match="tracked"):
        validate_report_output_path(Path("README.md"), Path.cwd())


def test_report_is_json_serializable_and_saved_to_tmp_path(tmp_path: Path) -> None:
    report = CapabilityProbeReport(
        schema_version="1.0",
        run_id="run",
        mode="inspect-client",
        started_at_utc="2026-07-22T00:00:00+00:00",
        completed_at_utc="2026-07-22T00:00:01+00:00",
        success=True,
        interrupted=False,
        observations=(
            CapabilityObservation(
                "connection",
                "confirmConnection",
                EvidenceLevel.STATIC_CLIENT,
                CapabilityStatus.CLIENT_METHOD_PRESENT_NOT_LIVE_TESTED,
                "present",
            ),
        ),
        data={"mesh_buffers": "not_invoked"},
        cleanup_results=(CleanupDomainResult("uav", False, True),),
    )
    output = tmp_path / "report.json"

    save_capability_report(report, output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["observations"][0]["status"] == (
        "client_method_present_not_live_tested"
    )
    assert loaded["data"]["mesh_buffers"] == "not_invoked"


def test_documentation_records_live_results_safety_and_decision_gates() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())

    assert "Phase A: repository implementation" in text
    assert "Phase B: manual live validation" in text
    assert "anchor_z = measured_ground_z - anchor_altitude" in text
    assert "simGetMeshPositionVertexBuffers" in text
    assert "never called in M13.1" in normalized_text
    assert "results/reports/m13/" in text
    assert "Phase B completed successfully" in text
    assert "SensorLocalFrame" in text
    assert "3.718" in text
    assert "28.269" in text
    assert "126.958" in text
    assert "Stage 5D was intentionally not run" in text
    assert "pause behavior remains unverified" in text
    assert "M13.2 may proceed" in text
    assert "M13.4 observation design" in text
    assert "No Phase B live validation has been run" not in text


def test_readme_distinguishes_capability_probe_from_navigation_features() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())

    assert "supervised Colosseum capability probe" in text
    assert "raw LiDAR access" in text
    assert (
        "does not integrate them into a navigation environment or policy"
        in normalized_text
    )
    assert "LiDAR observations in a Gymnasium environment or learned policy" in text
    assert "docs/m13_colosseum_capability_probe.md" in text


def load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_colosseum_capabilities", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def re_compile(pattern: str) -> Any:
    import re

    return re.compile(pattern)
