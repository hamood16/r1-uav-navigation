"""Simulator-independent tests for M13.2 scene materialization and recovery."""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r1_uav_nav.sim.colosseum_scene import (
    OWNERSHIP_MANIFEST_SCHEMA_VERSION,
    AssetCalibrationProbe,
    ColosseumSceneManager,
    MarkerPreviewRenderer,
    MaterializationConfig,
    OwnershipCreationStatus,
    OwnershipEntry,
    OwnershipManifest,
    OwnershipManifestError,
    SceneLifecycleError,
    VehiclePositioningConfig,
    cleanup_scene_resources,
    load_ownership_manifest,
    position_vehicle_at_start_and_return,
    recover_owned_scene,
    save_ownership_manifest_atomic,
    validate_transit_corridor,
)
from r1_uav_nav.sim.scene_specification import (
    AssetCalibrationStatus,
    CalibrationEvidenceLevel,
    Dimensions3D,
    SceneValidationError,
    ValidatedScene,
    Vector3,
    load_asset_catalog,
    load_scene_config,
    require_valid_scene,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "configs/scenes/m13_2_minimal.yaml"
ASSET_PATH = ROOT / "configs/scenes/m13_2_assets.yaml"
RED_MATERIAL = "/AirSim/Models/MiniQuadCopter/" "Prop_Red_Plastic.Prop_Red_Plastic"


class FakeVector:
    def __init__(self, x_val: float, y_val: float, z_val: float) -> None:
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val


class FakeQuaternion:
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x_val = x
        self.y_val = y
        self.z_val = z
        self.w_val = w


class FakePose:
    def __init__(
        self,
        position_val: FakeVector | None = None,
        orientation_val: FakeQuaternion | None = None,
    ) -> None:
        self.position = position_val or FakeVector(0.0, 0.0, 0.0)
        self.orientation = orientation_val or FakeQuaternion(0.0, 0.0, 0.0, 1.0)


class FakeLandedState:
    Landed = 0
    Flying = 1


def _quaternion(_pitch: float, _roll: float, yaw: float) -> FakeQuaternion:
    return FakeQuaternion(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


FAKE_MODULE = SimpleNamespace(
    Vector3r=FakeVector,
    Pose=FakePose,
    LandedState=FakeLandedState,
    to_quaternion=_quaternion,
)


class FakeAsync:
    def __init__(self, callback=None) -> None:
        self.callback = callback

    def join(self) -> None:
        if self.callback:
            self.callback()


class FakeClient:
    def __init__(self, ground: Vector3 | None = None) -> None:
        resolved_ground = ground or Vector3(10.0, -4.0, 0.57)
        self.ground = resolved_ground
        self.position = resolved_ground
        self.velocity = Vector3(0.0, 0.0, 0.0)
        self.landed_state = FakeLandedState.Landed
        self.api_enabled = False
        self.objects: dict[str, tuple[FakePose, FakeVector]] = {}
        self.logs: list[tuple] = []
        self.marker_calls = 0
        self.destroy_failures: set[str] = set()
        self.available_assets = ["Cube"]
        self.landing_confirmation_delay_polls = 0
        self._landing_confirmation_pending = False
        self.landed_only_after_disarm = False
        self.final_landed_delay_polls = 0
        self.touchdown_positions: list[Vector3] = []
        self.touchdown_velocities = [
            Vector3(0.2, 0.0, 0.0),
            Vector3(0.0, 0.0, 0.0),
            Vector3(0.0, 0.0, 0.0),
            Vector3(0.0, 0.0, 0.0),
        ]
        self.api_disable_failures_remaining = 0
        self._land_command_completed = False

    def getMultirotorState(self, *, vehicle_name: str) -> SimpleNamespace:
        self.logs.append(("state", vehicle_name))
        if self._land_command_completed and self.touchdown_positions:
            self.position = self.touchdown_positions.pop(0)
        if self._land_command_completed and self.touchdown_velocities:
            self.velocity = self.touchdown_velocities.pop(0)
        if self._landing_confirmation_pending:
            if self.landing_confirmation_delay_polls <= 0:
                self.landed_state = FakeLandedState.Landed
                self._landing_confirmation_pending = False
            else:
                self.landing_confirmation_delay_polls -= 1
        return SimpleNamespace(
            landed_state=self.landed_state,
            kinematics_estimated=SimpleNamespace(
                position=FakeVector(*self.position.values()),
                linear_velocity=FakeVector(*self.velocity.values()),
            ),
        )

    def isApiControlEnabled(self, *, vehicle_name: str) -> bool:
        self.logs.append(("api-query", vehicle_name))
        return self.api_enabled

    def simGetCollisionInfo(self, *, vehicle_name: str) -> SimpleNamespace:
        self.logs.append(("collision", vehicle_name))
        return SimpleNamespace(
            has_collided=False,
            object_name="",
            object_id=-1,
            time_stamp=1,
            penetration_depth=0.0,
            impact_point=FakeVector(0.0, 0.0, 0.0),
            position=FakeVector(*self.position.values()),
            normal=FakeVector(0.0, 0.0, -1.0),
        )

    def simListAssets(self) -> list[str]:
        self.logs.append(("assets",))
        return list(self.available_assets)

    def simListSceneObjects(self, pattern: str) -> list[str]:
        self.logs.append(("list", pattern))
        expression = re.compile(pattern)
        return [name for name in self.objects if expression.fullmatch(name)]

    def simSpawnObject(
        self,
        name: str,
        asset: str,
        pose: FakePose,
        scale: FakeVector,
        physics_enabled: bool,
        is_blueprint: bool,
    ) -> str:
        self.logs.append(("spawn", name, asset, physics_enabled, is_blueprint))
        self.objects[name] = (pose, scale)
        return name

    def simGetObjectPose(self, name: str) -> FakePose:
        return self.objects[name][0]

    def simGetObjectScale(self, name: str) -> FakeVector:
        return self.objects[name][1]

    def simSetObjectMaterial(self, name: str, material: str) -> bool:
        self.logs.append(("material", name, material))
        return True

    def simSetSegmentationObjectID(self, name: str, value: int, is_regex: bool) -> bool:
        self.logs.append(("segmentation", name, value, is_regex))
        return True

    def simDestroyObject(self, name: str) -> bool:
        self.logs.append(("destroy", name))
        if name in self.destroy_failures:
            return False
        self.objects.pop(name, None)
        return True

    def simPlotLineStrip(self, *args) -> None:
        self.marker_calls += 1
        self.logs.append(("marker", args[1]))

    def simPlotLineList(self, *args) -> None:
        self.marker_calls += 1
        self.logs.append(("marker-lines", args[1]))

    def simFlushPersistentMarkers(self) -> None:
        self.logs.append(("flush",))

    def enableApiControl(self, enabled: bool, *, vehicle_name: str) -> None:
        self.logs.append(("api", enabled, vehicle_name))
        if not enabled and self.api_disable_failures_remaining:
            self.api_disable_failures_remaining -= 1
            return
        self.api_enabled = enabled

    def armDisarm(self, armed: bool, *, vehicle_name: str) -> None:
        self.logs.append(("arm", armed, vehicle_name))
        if not armed and self.landed_only_after_disarm:
            if self.final_landed_delay_polls:
                self.landing_confirmation_delay_polls = self.final_landed_delay_polls
                self._landing_confirmation_pending = True
            else:
                self.landed_state = FakeLandedState.Landed
                self._landing_confirmation_pending = False

    def takeoffAsync(self, *, vehicle_name: str) -> FakeAsync:
        self.logs.append(("takeoff", vehicle_name))

        def takeoff() -> None:
            self.landed_state = FakeLandedState.Flying
            self._land_command_completed = False
            self._landing_confirmation_pending = False
            self.position = Vector3(
                self.ground.x,
                self.ground.y,
                self.ground.z - 1.5,
            )

        return FakeAsync(takeoff)

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
        self.logs.append(("move", x, y, z, velocity, timeout_sec, vehicle_name))
        return FakeAsync(lambda: setattr(self, "position", Vector3(x, y, z)))

    def hoverAsync(self, *, vehicle_name: str) -> FakeAsync:
        self.logs.append(("hover", vehicle_name))
        return FakeAsync()

    def landAsync(self, *, vehicle_name: str) -> FakeAsync:
        self.logs.append(("land", vehicle_name))

        def land() -> None:
            self.position = self.ground
            self.velocity = Vector3(0.0, 0.0, 0.0)
            self._land_command_completed = True
            if self.landed_only_after_disarm:
                self.landed_state = FakeLandedState.Flying
            elif self.landing_confirmation_delay_polls:
                self._landing_confirmation_pending = True
            else:
                self.landed_state = FakeLandedState.Landed

        return FakeAsync(land)


def _scene() -> ValidatedScene:
    return require_valid_scene(load_scene_config(SCENE_PATH))


def _accepted_catalog():
    catalog = load_asset_catalog(ASSET_PATH)
    cube = catalog.assets[0]
    accepted = replace(
        cube,
        nominal_dimensions_m=Dimensions3D(1.0, 1.0, 1.0),
        status=AssetCalibrationStatus.ACCEPTED,
        evidence_level=CalibrationEvidenceLevel.OPERATOR_CONFIRMED_NOMINAL,
        uncertainty_m=0.05,
        tested_stack="Blocks v2.0.0-beta / airsim 1.8.1",
        evidence_reference="ignored-report-id",
    )
    return replace(catalog, catalog_version=2, assets=(accepted,))


def _pending_catalog():
    catalog = load_asset_catalog(ASSET_PATH)
    cube = catalog.assets[0]
    pending = replace(
        cube,
        nominal_dimensions_m=None,
        status=AssetCalibrationStatus.REQUIRES_LIVE_VALIDATION,
        evidence_level=CalibrationEvidenceLevel.UNVALIDATED,
        uncertainty_m=None,
        tested_stack=None,
        evidence_reference=None,
    )
    return replace(catalog, assets=(pending,))


def _config() -> MaterializationConfig:
    return MaterializationConfig(
        vehicle_name="SimpleFlight",
        allow_scene_mutation=True,
        confirm_scene_area_clear=True,
        confirm_no_visible_collision=True,
        allow_debug_markers=True,
        allow_marker_flush=True,
    )


def test_package_import_remains_independent_of_airsim(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "airsim", None)
    module = importlib.reload(importlib.import_module("r1_uav_nav.sim"))
    assert hasattr(module, "SceneConfig")


def test_pending_calibration_blocks_before_spawn(tmp_path: Path) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _pending_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    with pytest.raises(SceneLifecycleError, match="calibration"):
        manager.materialize(_scene(), _config())
    assert not any(log[0] == "spawn" for log in client.logs)


def test_nominal_calibration_records_evidence_without_updating_catalog(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    probe = AssetCalibrationProbe(
        client,
        FAKE_MODULE,
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )

    result, runtime = probe.run(
        asset_name="Cube",
        vehicle_name="SimpleFlight",
        allow_scene_mutation=True,
        confirm_scene_area_clear=True,
        confirm_no_visible_collision=True,
        allow_debug_markers=True,
        allow_marker_flush=True,
        hold_seconds=0.0,
        run_id="calibration",
    )

    assert result.evidence_level == "operator_confirmed_nominal"
    assert result.operator_confirmation == "pending"
    assert "does not measure exact mesh bounds" in result.uncertainty_note
    assert result.measured_scale == (1.0, 1.0, 1.0)
    assert not result.collision_response_verified
    assert runtime.markers_created
    assert not hasattr(FAKE_MODULE, "ColorRgba")
    assert ("marker-lines", [1.0, 1.0, 0.0, 1.0]) in client.logs
    cleanup = cleanup_scene_resources(client, runtime)
    assert all(item.succeeded for item in cleanup)


def test_nominal_calibration_authorization_precedes_spawn(tmp_path: Path) -> None:
    client = FakeClient()
    probe = AssetCalibrationProbe(
        client,
        FAKE_MODULE,
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    with pytest.raises(SceneLifecycleError, match="authorization"):
        probe.run(
            asset_name="Cube",
            vehicle_name="SimpleFlight",
            allow_scene_mutation=False,
            confirm_scene_area_clear=True,
            confirm_no_visible_collision=True,
            allow_debug_markers=True,
            allow_marker_flush=True,
            hold_seconds=0.0,
        )
    assert not any(log[0] == "spawn" for log in client.logs)


def test_materialization_uses_exact_names_transforms_and_truthful_collision(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config(), run_id="run1")

    assert len(materialized.objects) == 4
    assert materialized.scene_digest == _scene().scene_digest
    assert materialized.collision_geometry_complete
    assert not materialized.collision_response_verified
    assert all(not item.physics_enabled for item in materialized.objects)
    assert all(not item.collision_response_verified for item in materialized.objects)
    assert all(name.startswith("r1_uav_m13s2_") for name in runtime.owned_names)
    assert client.marker_calls == 1
    assert any(
        log[:3] == ("material", runtime.owned_names[0], RED_MATERIAL)
        for log in client.logs
    )
    assert (
        runtime.manifest.entries[-1].creation_status is OwnershipCreationStatus.CREATED
    )
    assert runtime.manifest_path.exists()


def test_runtime_asset_inventory_is_checked_once_before_ownership_or_spawn(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    client.available_assets = []
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )

    with pytest.raises(SceneLifecycleError, match="unavailable"):
        manager.materialize(_scene(), _config(), run_id="missing-asset")

    assert client.logs.count(("assets",)) == 1
    assert not any(log[0] == "spawn" for log in client.logs)
    assert manager.last_runtime is not None
    assert manager.last_runtime.manifest.entries == ()
    assert manager.last_runtime.owned_names == []


def test_runtime_asset_inventory_is_listed_once_for_complete_scene(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )

    manager.materialize(_scene(), _config(), run_id="asset-inventory")

    assert client.logs.count(("assets",)) == 1
    assert len([log for log in client.logs if log[0] == "spawn"]) == 4


def test_goal_marker_uses_configured_non_green_color(tmp_path: Path) -> None:
    scene = _scene()
    color = (1.0, 0.25, 0.75, 0.8)
    changed_goal = replace(
        scene.config.goal_pad,
        appearance=replace(
            scene.config.goal_pad.appearance,
            marker_color_rgba=color,
        ),
    )
    changed_scene = replace(scene, config=replace(scene.config, goal_pad=changed_goal))
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )

    manager.materialize(changed_scene, _config(), run_id="marker-color")

    assert not hasattr(FAKE_MODULE, "ColorRgba")
    assert ("marker", list(color)) in client.logs


@pytest.mark.parametrize(
    "color",
    [
        (1.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 1.0),
        (-0.1, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.1),
    ],
)
def test_goal_marker_rejects_malformed_or_out_of_range_rgba(
    color: tuple[float, ...],
) -> None:
    goal = _scene().config.goal_pad
    changed_goal = replace(
        goal,
        appearance=replace(goal.appearance, marker_color_rgba=color),
    )
    client = FakeClient()

    with pytest.raises(SceneLifecycleError, match="goal marker color"):
        MarkerPreviewRenderer(client, FAKE_MODULE).render_goal(
            changed_goal,
            Vector3(0.0, 0.0, 0.0),
        )

    assert client.marker_calls == 0


def test_world_exclusion_rejects_invalid_scene_before_spawn(tmp_path: Path) -> None:
    valid = _scene()
    invalid_config = replace(
        valid.config,
        start_pad=replace(valid.config.start_pad, base_center=Vector3(0.0, 0.0, 0.0)),
    )
    bypassed = replace(valid, config=invalid_config)
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    with pytest.raises(SceneValidationError):
        manager.materialize(bypassed, _config())
    assert not any(log[0] == "spawn" for log in client.logs)


def test_preexisting_exact_name_is_never_adopted(tmp_path: Path) -> None:
    client = FakeClient()
    scene = _scene()
    first_name = (
        f"r1_uav_m13s2_{scene.config.scene_id}__"
        f"{scene.config.start_pad.name}__{scene.scene_digest[:12]}"
    )
    client.objects[first_name] = (FakePose(), FakeVector(1.0, 1.0, 1.0))
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    with pytest.raises(SceneLifecycleError, match="pre-existing"):
        manager.materialize(scene, _config())
    assert first_name not in (
        manager.last_runtime.owned_names if manager.last_runtime else []
    )


def test_cleanup_is_exact_idempotent_and_preserves_unrelated_objects(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    client.objects["unrelated"] = (FakePose(), FakeVector(1.0, 1.0, 1.0))
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    _, runtime = manager.materialize(_scene(), _config(), run_id="cleanup")

    first = cleanup_scene_resources(client, runtime)
    second = cleanup_scene_resources(client, runtime)

    assert all(result.succeeded for result in first)
    assert all(result.succeeded for result in second)
    assert "unrelated" in client.objects
    assert all(
        entry.creation_status is OwnershipCreationStatus.CLEANED
        for entry in runtime.manifest.entries
    )


def test_reset_scene_rebuilds_same_local_identity_after_exact_cleanup(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )

    first, first_runtime = manager.reset_scene(_scene(), _config(), run_id="repeat-one")
    second, second_runtime = manager.reset_scene(
        _scene(), _config(), run_id="repeat-two"
    )

    assert first.scene_digest == second.scene_digest
    assert first.materialization_digest == second.materialization_digest
    assert [item.returned_name for item in first.objects] == [
        item.returned_name for item in second.objects
    ]
    assert all(
        entry.creation_status is OwnershipCreationStatus.CLEANED
        for entry in first_runtime.manifest.entries
    )
    assert all(name in client.objects for name in second_runtime.owned_names)


def test_object_cleanup_failure_does_not_skip_other_objects_or_markers(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    _, runtime = manager.materialize(_scene(), _config())
    failed_name = runtime.owned_names[-1]
    client.destroy_failures.add(failed_name)

    results = cleanup_scene_resources(client, runtime)

    objects = next(result for result in results if result.domain == "objects")
    markers = next(result for result in results if result.domain == "markers")
    assert not objects.succeeded
    assert markers.succeeded
    assert ("flush",) in client.logs
    assert len(client.objects) == 1
    assert failed_name in client.objects


def _manifest(path: Path) -> OwnershipManifest:
    manifest = OwnershipManifest(
        OWNERSHIP_MANIFEST_SCHEMA_VERSION,
        "run",
        "minimal-course",
        "a" * 64,
        "b" * 64,
        "runtime_spawn",
        "1",
        (
            OwnershipEntry(
                "start-pad",
                "owned",
                "owned",
                True,
                OwnershipCreationStatus.CREATED,
            ),
        ),
    )
    save_ownership_manifest_atomic(manifest, path)
    return manifest


def test_atomic_manifest_failure_preserves_previous_valid_file(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "ownership.json"
    original = _manifest(path)
    original_text = path.read_text(encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("interrupted")

    monkeypatch.setattr("r1_uav_nav.sim.colosseum_scene.os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        save_ownership_manifest_atomic(
            replace(original, run_id="changed"),
            path,
        )
    assert path.read_text(encoding="utf-8") == original_text
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        '{"schema_version":"1.0","entries":"bad"}',
        "not-json",
    ],
)
def test_malformed_manifests_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(OwnershipManifestError):
        load_ownership_manifest(path)


def test_recovery_refuses_configuration_or_ambiguous_ownership(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    with pytest.raises(OwnershipManifestError):
        recover_owned_scene(
            client,
            SCENE_PATH,
            allow_scene_mutation=True,
            allow_recovery=True,
        )

    path = tmp_path / "ambiguous.json"
    manifest = replace(
        _manifest(path),
        entries=(
            OwnershipEntry(
                "start-pad",
                "maybe",
                None,
                True,
                OwnershipCreationStatus.CREATING,
            ),
        ),
    )
    save_ownership_manifest_atomic(manifest, path)
    with pytest.raises(OwnershipManifestError, match="ambiguous"):
        recover_owned_scene(
            client,
            path,
            allow_scene_mutation=True,
            allow_recovery=True,
        )


def test_recovery_deletes_only_exact_manifest_owned_name(tmp_path: Path) -> None:
    client = FakeClient()
    client.objects["owned"] = (FakePose(), FakeVector(1.0, 1.0, 1.0))
    client.objects["owned-similar"] = (FakePose(), FakeVector(1.0, 1.0, 1.0))
    client.objects["r1_uav_m13s2_unrelated"] = (
        FakePose(),
        FakeVector(1.0, 1.0, 1.0),
    )
    path = tmp_path / "ownership.json"
    _manifest(path)

    updated, result = recover_owned_scene(
        client,
        path,
        allow_scene_mutation=True,
        allow_recovery=True,
    )

    assert result.succeeded
    assert "owned" not in client.objects
    assert "owned-similar" in client.objects
    assert "r1_uav_m13s2_unrelated" in client.objects
    assert updated.entries[0].creation_status is OwnershipCreationStatus.CLEANED


def test_recovery_accepts_report_with_complete_embedded_ownership(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    client.objects["owned"] = (FakePose(), FakeVector(1.0, 1.0, 1.0))
    manifest_path = tmp_path / "ownership-source.json"
    manifest = _manifest(manifest_path)
    report_path = tmp_path / "accepted-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "data": {
                    "ownership_evidence_complete": True,
                    "ownership_manifest": {
                        "schema_version": manifest.schema_version,
                        "run_id": manifest.run_id,
                        "scene_id": manifest.scene_id,
                        "scene_digest": manifest.scene_digest,
                        "materialization_digest": manifest.materialization_digest,
                        "backend": manifest.backend,
                        "backend_version": manifest.backend_version,
                        "entries": [
                            {
                                "specification_name": "start-pad",
                                "requested_name": "owned",
                                "returned_name": "owned",
                                "proven_absent_before_creation": True,
                                "creation_status": "created",
                                "cleanup_error": None,
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    updated, result = recover_owned_scene(
        client,
        report_path,
        allow_scene_mutation=True,
        allow_recovery=True,
    )

    assert result.succeeded
    assert updated.entries[0].creation_status is OwnershipCreationStatus.CLEANED
    assert (tmp_path / "ownership" / "run.recovery.ownership.json").exists()


def test_named_positioning_returns_to_original_ground_before_scene_cleanup(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    evidence = position_vehicle_at_start_and_return(
        client,
        FAKE_MODULE,
        materialized,
        runtime,
        VehiclePositioningConfig(
            allow_flight=True,
            allow_start_positioning=True,
            confirm_clear_airspace=True,
            confirm_no_visible_collision=True,
        ),
    )
    assert evidence["returned_ground_position"] == {
        "x": client.ground.x,
        "y": client.ground.y,
        "z": client.ground.z,
    }
    assert evidence["touchdown_confirmation_attempts"] == 4
    assert evidence["touchdown_consecutive_samples"] == 3
    assert evidence["touchdown_rejection_reason"] is None
    assert evidence["landed_state_before_disarm"] == FakeLandedState.Landed
    assert evidence["final_confirmation_attempts"] == 1
    assert evidence["final_landed_state"] == FakeLandedState.Landed
    assert evidence["final_speed_m_s"] == 0.0
    assert evidence["returned_to_original_ground"]
    assert evidence["landing_confirmed"]
    assert evidence["api_control_released"]
    assert all(name in client.objects for name in runtime.owned_names)
    cleanup = cleanup_scene_resources(client, runtime)
    assert all(result.succeeded for result in cleanup)

    actions = [log[0] for log in client.logs]
    takeoff = actions.index("takeoff")
    moves = [index for index, action in enumerate(actions) if action == "move"]
    landing = actions.index("land")
    first_destroy = actions.index("destroy")
    assert len(moves) == 3
    assert takeoff < moves[0] < moves[1] < moves[2] < landing < first_destroy
    assert all(
        log[-1] == "SimpleFlight"
        for log in client.logs
        if log[0]
        in {
            "state",
            "api-query",
            "collision",
            "api",
            "arm",
            "takeoff",
            "move",
            "hover",
            "land",
        }
    )


def test_flying_enum_until_disarm_allows_stable_touchdown(tmp_path: Path) -> None:
    client = FakeClient()
    client.landed_only_after_disarm = True
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    evidence = position_vehicle_at_start_and_return(
        client,
        FAKE_MODULE,
        materialized,
        runtime,
        VehiclePositioningConfig(
            allow_flight=True,
            allow_start_positioning=True,
            confirm_clear_airspace=True,
            confirm_no_visible_collision=True,
            landing_confirmation_timeout_s=1.0,
            landing_poll_interval_s=0.1,
        ),
        sleep_fn=lambda _seconds: None,
    )

    assert evidence["touchdown_confirmation_attempts"] == 4
    assert evidence["touchdown_consecutive_samples"] == 3
    assert evidence["landed_state_before_disarm"] == FakeLandedState.Flying
    assert evidence["final_confirmation_attempts"] == 1
    assert evidence["final_landed_state"] == FakeLandedState.Landed
    assert evidence["landing_confirmed"]
    assert evidence["final_speed_m_s"] == 0.0
    assert evidence["api_control_released"]
    assert all(result.succeeded for result in cleanup_scene_resources(client, runtime))


def test_transient_touchdown_sample_does_not_satisfy_consecutive_gate(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    far = Vector3(client.ground.x + 2.0, client.ground.y, client.ground.z)
    client.touchdown_positions = [
        client.ground,
        far,
        client.ground,
        client.ground,
        client.ground,
    ]
    client.touchdown_velocities = [Vector3(0.0, 0.0, 0.0)] * 5
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    evidence = position_vehicle_at_start_and_return(
        client,
        FAKE_MODULE,
        materialized,
        runtime,
        VehiclePositioningConfig(
            allow_flight=True,
            allow_start_positioning=True,
            confirm_clear_airspace=True,
            confirm_no_visible_collision=True,
            landing_confirmation_timeout_s=1.0,
            landing_poll_interval_s=0.1,
        ),
        sleep_fn=lambda _seconds: None,
    )

    assert evidence["touchdown_confirmation_attempts"] == 5
    assert evidence["touchdown_consecutive_samples"] == 3
    assert evidence["landing_confirmed"]
    results = cleanup_scene_resources(client, runtime)
    assert all(result.succeeded for result in results)


def test_transient_high_speed_resets_touchdown_consecutive_count(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    client.touchdown_velocities = [
        Vector3(0.0, 0.0, 0.0),
        Vector3(0.0, 0.0, 0.0),
        Vector3(0.2, 0.0, 0.0),
        Vector3(0.0, 0.0, 0.0),
        Vector3(0.0, 0.0, 0.0),
        Vector3(0.0, 0.0, 0.0),
    ]
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    evidence = position_vehicle_at_start_and_return(
        client,
        FAKE_MODULE,
        materialized,
        runtime,
        VehiclePositioningConfig(
            allow_flight=True,
            allow_start_positioning=True,
            confirm_clear_airspace=True,
            confirm_no_visible_collision=True,
            landing_confirmation_timeout_s=1.0,
            landing_poll_interval_s=0.1,
        ),
        sleep_fn=lambda _seconds: None,
    )

    assert evidence["touchdown_confirmation_attempts"] == 6
    assert evidence["touchdown_consecutive_samples"] == 3
    assert evidence["touchdown_rejection_reason"] is None
    assert evidence["landing_confirmed"]
    assert all(result.succeeded for result in cleanup_scene_resources(client, runtime))


def test_touchdown_timeout_preserves_partial_evidence_and_cleanup_state(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    far = Vector3(client.ground.x + 2.0, client.ground.y, client.ground.z)
    client.touchdown_positions = [far] * 10
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())
    with pytest.raises(SceneLifecycleError) as exc:
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
                landing_confirmation_timeout_s=0.2,
                landing_poll_interval_s=0.1,
            ),
            sleep_fn=lambda _seconds: None,
        )

    assert "physical touchdown confirmation timed out after 3 attempts" in str(
        exc.value
    )
    evidence = runtime.vehicle_positioning_evidence
    assert evidence["touchdown_confirmation_attempts"] == 3
    assert evidence["touchdown_consecutive_samples"] == 0
    assert evidence["touchdown_position"] == {
        "x": far.x,
        "y": far.y,
        "z": far.z,
    }
    assert not evidence["landing_confirmed"]
    assert runtime.cleanup_state.airborne
    assert runtime.cleanup_state.armed
    assert runtime.cleanup_state.api_control_enabled


def test_sustained_touchdown_speed_times_out_without_immediate_exception(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    client.touchdown_velocities = [Vector3(0.2, 0.0, 0.0)] * 10
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    with pytest.raises(SceneLifecycleError, match="physical touchdown") as exc:
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
                landing_confirmation_timeout_s=0.2,
                landing_poll_interval_s=0.1,
            ),
            sleep_fn=lambda _seconds: None,
        )

    assert "speed above grounded tolerance" in str(exc.value)
    evidence = runtime.vehicle_positioning_evidence
    assert evidence["touchdown_confirmation_attempts"] == 3
    assert evidence["touchdown_consecutive_samples"] == 0
    assert evidence["touchdown_speed_m_s"] == pytest.approx(0.2)
    assert evidence["touchdown_rejection_reason"] == "speed above grounded tolerance"
    assert all(result.succeeded for result in cleanup_scene_resources(client, runtime))


def test_unsafe_touchdown_collision_is_recorded_until_bounded_timeout(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    original_collision = client.simGetCollisionInfo

    def collision_after_landing(*, vehicle_name: str) -> SimpleNamespace:
        if not client._land_command_completed:
            return original_collision(vehicle_name=vehicle_name)
        client.logs.append(("collision", vehicle_name))
        return SimpleNamespace(
            has_collided=True,
            object_name="UnexpectedObject",
            object_id=77,
            time_stamp=999,
            penetration_depth=0.01,
            impact_point=FakeVector(*client.position.values()),
            position=FakeVector(*client.position.values()),
            normal=FakeVector(0.0, 0.0, -1.0),
        )

    client.simGetCollisionInfo = collision_after_landing  # type: ignore[method-assign]
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    with pytest.raises(SceneLifecycleError, match="physical touchdown"):
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
                landing_confirmation_timeout_s=0.2,
                landing_poll_interval_s=0.1,
            ),
            sleep_fn=lambda _seconds: None,
        )

    evidence = runtime.vehicle_positioning_evidence
    assert evidence["touchdown_confirmation_attempts"] == 3
    assert evidence["touchdown_consecutive_samples"] == 0
    assert "collision" in evidence["touchdown_rejection_reason"]
    assert all(result.succeeded for result in cleanup_scene_resources(client, runtime))


def test_touchdown_timeout_cleanup_retries_named_landing_and_release(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    far = Vector3(client.ground.x + 2.0, client.ground.y, client.ground.z)
    client.touchdown_positions = [far] * 10
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    with pytest.raises(SceneLifecycleError, match="physical touchdown"):
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
                landing_confirmation_timeout_s=0.2,
                landing_poll_interval_s=0.1,
            ),
            sleep_fn=lambda _seconds: None,
        )
    actions_before_cleanup = len(client.logs)

    results = cleanup_scene_resources(client, runtime)

    assert all(result.succeeded for result in results)
    cleanup_logs = client.logs[actions_before_cleanup:]
    cleanup_actions = [log[0] for log in cleanup_logs]
    assert cleanup_actions.index("hover") < cleanup_actions.index("land")
    assert cleanup_actions.index("land") < cleanup_actions.index("arm")
    assert cleanup_actions.index("arm") < cleanup_actions.index("api")
    assert ("hover", "SimpleFlight") in cleanup_logs
    assert ("land", "SimpleFlight") in cleanup_logs
    assert ("arm", False, "SimpleFlight") in cleanup_logs
    assert ("api", False, "SimpleFlight") in cleanup_logs


def test_api_control_release_verification_failure_preserves_retry_state(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    client.landed_only_after_disarm = True
    client.api_disable_failures_remaining = 1
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    with pytest.raises(
        SceneLifecycleError, match="final landed-state confirmation timed out"
    ):
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
                final_state_confirmation_timeout_s=0.2,
                final_state_poll_interval_s=0.1,
            ),
            sleep_fn=lambda _seconds: None,
        )

    assert runtime.vehicle_positioning_evidence["touchdown_consecutive_samples"] == 3
    assert not runtime.vehicle_positioning_evidence["api_control_released"]
    assert (
        runtime.vehicle_positioning_evidence["final_rejection_reason"]
        == "API control remains enabled"
    )
    assert runtime.cleanup_state.api_control_enabled
    cleanup = cleanup_scene_resources(client, runtime)
    assert all(result.succeeded for result in cleanup)
    assert not client.api_enabled


def test_final_landed_state_may_be_delayed_after_disarm(tmp_path: Path) -> None:
    client = FakeClient()
    client.landed_only_after_disarm = True
    client.final_landed_delay_polls = 2
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    evidence = position_vehicle_at_start_and_return(
        client,
        FAKE_MODULE,
        materialized,
        runtime,
        VehiclePositioningConfig(
            allow_flight=True,
            allow_start_positioning=True,
            confirm_clear_airspace=True,
            confirm_no_visible_collision=True,
            final_state_confirmation_timeout_s=1.0,
            final_state_poll_interval_s=0.1,
        ),
        sleep_fn=lambda _seconds: None,
    )

    assert evidence["landed_state_before_disarm"] == FakeLandedState.Flying
    assert evidence["final_confirmation_attempts"] == 3
    assert evidence["final_landed_state"] == FakeLandedState.Landed
    assert evidence["landing_confirmed"]
    assert all(result.succeeded for result in cleanup_scene_resources(client, runtime))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("landing_confirmation_timeout_s", 0.0),
        ("landing_confirmation_timeout_s", float("nan")),
        ("landing_poll_interval_s", -0.1),
        ("landing_poll_interval_s", float("inf")),
        ("final_state_confirmation_timeout_s", 0.0),
        ("final_state_poll_interval_s", float("nan")),
    ],
)
def test_landing_confirmation_config_requires_finite_positive_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        VehiclePositioningConfig(**{field_name: value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_touchdown_consecutive_sample_count_must_be_positive_integer(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="touchdown_consecutive_samples"):
        VehiclePositioningConfig(touchdown_consecutive_samples=value)  # type: ignore[arg-type]


def test_transit_corridor_rejects_segment_through_expected_geometry(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, _ = manager.materialize(_scene(), _config())
    bounds = materialized.objects[0].requested_transform.conservative_world_bounds
    middle_y = (bounds.min_y + bounds.max_y) / 2.0
    middle_z = (bounds.min_z + bounds.max_z) / 2.0

    with pytest.raises(SceneLifecycleError, match="start-pad"):
        validate_transit_corridor(
            materialized,
            (
                Vector3(bounds.min_x - 1.0, middle_y, middle_z),
                Vector3(bounds.max_x + 1.0, middle_y, middle_z),
            ),
            clearance_m=0.0,
        )


def test_positioning_authorization_is_required_before_control(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())
    before = len(client.logs)
    with pytest.raises(SceneLifecycleError, match="authorization"):
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(),
        )
    assert not any(log[0] == "takeoff" for log in client.logs[before:])


def test_fresh_positioning_preflight_blocks_enabled_api_before_control(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())
    client.api_enabled = True
    before = len(client.logs)

    with pytest.raises(SceneLifecycleError, match="API control"):
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
            ),
            sleep_fn=lambda _seconds: None,
        )

    assert not any(log[0] == "api" for log in client.logs[before:])
    assert not any(log[0] == "takeoff" for log in client.logs[before:])


def test_fresh_positioning_preflight_rejects_moving_vehicle_before_control(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())
    client.velocity = Vector3(0.2, 0.0, 0.0)
    before = len(client.logs)

    with pytest.raises(
        SceneLifecycleError, match="start-positioning preflight requires"
    ):
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
            ),
            sleep_fn=lambda _seconds: None,
        )

    assert not any(log[0] == "api" for log in client.logs[before:])
    assert not any(log[0] == "takeoff" for log in client.logs[before:])


def test_fresh_positioning_collision_preflight_blocks_before_control(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())
    collision_timestamp = 10

    def changing_collision(*, vehicle_name: str) -> SimpleNamespace:
        nonlocal collision_timestamp
        collision_timestamp += 1
        client.logs.append(("collision", vehicle_name))
        return SimpleNamespace(
            has_collided=True,
            object_name="GroundSurface",
            object_id=1,
            time_stamp=collision_timestamp,
            penetration_depth=0.01,
            impact_point=FakeVector(
                client.position.x,
                client.position.y,
                client.position.z + 0.1,
            ),
            position=FakeVector(*client.position.values()),
            normal=FakeVector(0.0, 0.0, -1.0),
        )

    client.simGetCollisionInfo = changing_collision  # type: ignore[method-assign]
    before = len(client.logs)

    with pytest.raises(SceneLifecycleError, match="collision evidence"):
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
            ),
            sleep_fn=lambda _seconds: None,
        )

    assert not any(log[0] == "api" for log in client.logs[before:])
    assert not any(log[0] == "takeoff" for log in client.logs[before:])


def test_positioning_interruption_preserves_named_uav_and_scene_cleanup(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = ColosseumSceneManager(
        client,
        FAKE_MODULE,
        _accepted_catalog(),
        ownership_dir=tmp_path,
        sleep_fn=lambda _seconds: None,
    )
    materialized, runtime = manager.materialize(_scene(), _config())

    def interrupted_move(*args, **kwargs):
        client.logs.append(("move-interrupted", kwargs["vehicle_name"]))
        return FakeAsync(lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    client.moveToPositionAsync = interrupted_move  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        position_vehicle_at_start_and_return(
            client,
            FAKE_MODULE,
            materialized,
            runtime,
            VehiclePositioningConfig(
                allow_flight=True,
                allow_start_positioning=True,
                confirm_clear_airspace=True,
                confirm_no_visible_collision=True,
            ),
        )

    results = cleanup_scene_resources(client, runtime)
    assert all(result.succeeded for result in results)
    actions = [log[0] for log in client.logs]
    assert actions.index("hover") < actions.index("land") < actions.index("destroy")
    assert ("arm", False, "SimpleFlight") in client.logs
    assert ("api", False, "SimpleFlight") in client.logs


def test_cli_parser_and_help_do_not_import_airsim(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "airsim", None)
    script = importlib.import_module("scripts.manage_colosseum_scene")
    parser = script.build_parser()
    args = parser.parse_args(["validate"])
    assert args.command == "validate"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    ("extra_arguments", "message"),
    [
        (
            [
                "--allow-scene-mutation",
                "--confirm-scene-area-clear",
                "--confirm-no-visible-collision",
                "--allow-marker-flush",
            ],
            "allow-debug-markers",
        ),
        (
            [
                "--allow-scene-mutation",
                "--confirm-scene-area-clear",
                "--confirm-no-visible-collision",
                "--allow-debug-markers",
            ],
            "allow-marker-flush",
        ),
        (
            [
                "--confirm-scene-area-clear",
                "--confirm-no-visible-collision",
                "--allow-debug-markers",
                "--allow-marker-flush",
            ],
            "allow-scene-mutation",
        ),
        (
            [
                "--allow-scene-mutation",
                "--confirm-no-visible-collision",
                "--allow-debug-markers",
                "--allow-marker-flush",
            ],
            "confirm-scene-area-clear",
        ),
    ],
)
def test_cli_rejects_scene_authorization_gaps_before_client_import(
    extra_arguments: list[str],
    message: str,
) -> None:
    script = importlib.import_module("scripts.manage_colosseum_scene")
    imported = False

    def fail_if_imported(_module_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("client import must not run")

    args = script.parse_args(
        [
            "--scene-config",
            str(SCENE_PATH),
            "materialize",
            *extra_arguments,
        ]
    )
    with pytest.raises(ValueError, match=message):
        script.run(
            args,
            repository_root=ROOT,
            client_module_loader=fail_if_imported,
        )
    assert not imported


@pytest.mark.parametrize(
    "omitted_argument",
    [
        "--allow-flight",
        "--allow-start-positioning",
        "--confirm-clear-airspace",
        "--confirm-no-visible-collision",
    ],
)
def test_cli_rejects_incomplete_positioning_authorization_before_client_import(
    omitted_argument: str,
) -> None:
    script = importlib.import_module("scripts.manage_colosseum_scene")
    imported = False

    def fail_if_imported(_module_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("client import must not run")

    authorizations = [
        "--allow-scene-mutation",
        "--confirm-scene-area-clear",
        "--confirm-no-visible-collision",
        "--allow-debug-markers",
        "--allow-marker-flush",
        "--position-start",
        "--allow-flight",
        "--allow-start-positioning",
        "--confirm-clear-airspace",
    ]
    authorizations.remove(omitted_argument)
    args = script.parse_args(
        [
            "--scene-config",
            str(SCENE_PATH),
            "materialize",
            *authorizations,
        ]
    )
    with pytest.raises(ValueError):
        script.run(
            args,
            repository_root=ROOT,
            client_module_loader=fail_if_imported,
        )
    assert not imported


def test_cli_recovery_requires_manifest_and_both_authorizations() -> None:
    script = importlib.import_module("scripts.manage_colosseum_scene")
    args = script.parse_args(["cleanup", "--ownership-source", "owned.json"])
    with pytest.raises(OwnershipManifestError):
        script._validate_live_arguments(args)


def test_repeat_report_contains_directly_comparable_materialization_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    script = importlib.import_module("scripts.manage_colosseum_scene")
    client = FakeClient()
    reports = tmp_path / "reports"
    monkeypatch.setattr(script, "DEFAULT_OWNERSHIP_DIR", tmp_path / "ownership")
    monkeypatch.setattr(
        script,
        "load_asset_catalog",
        lambda _path: _accepted_catalog(),
    )
    monkeypatch.setattr(script, "confirm_connection", lambda _client: None)
    args = script.parse_args(
        [
            "--scene-config",
            str(SCENE_PATH),
            "--asset-catalog",
            str(ASSET_PATH),
            "--output-dir",
            str(reports),
            "materialize",
            "--repeat",
            "2",
            "--allow-scene-mutation",
            "--confirm-scene-area-clear",
            "--confirm-no-visible-collision",
            "--allow-debug-markers",
            "--allow-marker-flush",
            "--position-start",
            "--allow-flight",
            "--allow-start-positioning",
            "--confirm-clear-airspace",
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
    report_path = next(reports.glob("m13_2_materialize_*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    repetitions = report["data"]["repetitions"]
    assert len(repetitions) == 2
    for repetition in repetitions:
        assert repetition["scene_digest"]
        assert repetition["materialization_digest"]
        assert repetition["exact_names"]["requested"]
        assert repetition["exact_names"]["returned"]
        assert len(repetition["objects"]) == 4
        assert all(
            {
                "requested_transform",
                "measured_center_position",
                "measured_scale",
                "measured_yaw_degrees",
            }
            <= item.keys()
            for item in repetition["objects"]
        )
        object_cleanup = next(
            item
            for item in repetition["cleanup_results"]
            if item["domain"] == "objects"
        )
        assert object_cleanup["succeeded"]
    assert repetitions[0]["scene_digest"] == repetitions[1]["scene_digest"]
    assert (
        repetitions[0]["materialization_digest"]
        == repetitions[1]["materialization_digest"]
    )
    assert repetitions[0]["exact_names"] == repetitions[1]["exact_names"]
    assert (
        repetitions[0]["objects"][0]["requested_transform"]
        == repetitions[1]["objects"][0]["requested_transform"]
    )
    positioning = report["data"]["vehicle_positioning"]
    assert positioning["start_anchor"]
    assert positioning["returned_ground_position"]
    assert positioning["touchdown_confirmation_attempts"] == 4
    assert positioning["touchdown_consecutive_samples"] == 3
    assert positioning["touchdown_position"]
    assert positioning["touchdown_speed_m_s"] == 0.0
    assert positioning["touchdown_rejection_reason"] is None
    assert positioning["landed_state_before_disarm"] == FakeLandedState.Landed
    assert positioning["final_confirmation_attempts"] == 1
    assert positioning["final_landed_state"] == FakeLandedState.Landed
    assert positioning["final_position"]
    assert positioning["final_speed_m_s"] == 0.0
    assert positioning["final_api_control_enabled"] is False
    assert positioning["final_rejection_reason"] is None
    assert positioning["returned_to_original_ground"]
    assert positioning["landing_confirmed"]
    assert positioning["api_control_released"]


def test_manifest_serialization_records_exact_ownership(tmp_path: Path) -> None:
    path = tmp_path / "ownership.json"
    manifest = _manifest(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["run_id"] == manifest.run_id
    assert raw["entries"][0]["requested_name"] == "owned"
    assert raw["entries"][0]["returned_name"] == "owned"
    assert raw["entries"][0]["creation_status"] == "created"
