"""Pure M13.2 scene-model, geometry, and determinism tests."""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from r1_uav_nav.sim.scene_specification import (
    AssetCalibrationStatus,
    CalibrationEvidenceLevel,
    CollisionIntent,
    Dimensions3D,
    DynamicObstacle,
    MotionMode,
    ObstacleMotion,
    SceneGenerationError,
    SceneValidationError,
    StaticObstacle,
    Vector3,
    asset_catalog_digest,
    bounds_intersect,
    build_initial_vehicle_exclusion,
    canonical_scene_json,
    conservative_bounds,
    derive_goal_approach,
    derive_start_anchor,
    generate_scene,
    load_asset_catalog,
    load_scene_config,
    materialization_digest,
    pad_safety_bounds,
    require_valid_scene,
    resolve_scene,
    runtime_object_name,
    scene_digest,
    validate_scene,
    validate_world_vehicle_exclusion,
)

ROOT = Path(__file__).resolve().parents[1]
MINIMAL_CONFIG = ROOT / "configs/scenes/m13_2_minimal.yaml"
GENERATED_CONFIG = ROOT / "configs/scenes/m13_2_generated.yaml"
PADS_CONFIG = ROOT / "configs/scenes/m13_2_pads.yaml"
ASSET_CONFIG = ROOT / "configs/scenes/m13_2_assets.yaml"


def _minimal():
    return load_scene_config(MINIMAL_CONFIG)


def test_minimal_scene_is_valid_and_keeps_start_away_from_initial_vehicle() -> None:
    scene = require_valid_scene(_minimal())

    assert scene.start_anchor == Vector3(4.0, 0.0, -2.1)
    exclusion = build_initial_vehicle_exclusion(
        scene.config.reference.initial_vehicle_local_position,
        scene.config.reference.initial_vehicle_exclusion,
    )
    assert not bounds_intersect(pad_safety_bounds(scene.config.start_pad), exclusion)
    assert scene.config.start_pad.base_center.x >= 4.0


def test_pads_scene_matches_minimal_reference_and_contains_no_obstacles() -> None:
    minimal = require_valid_scene(_minimal())
    pads = require_valid_scene(load_scene_config(PADS_CONFIG))

    assert pads.config.workspace == minimal.config.workspace
    assert pads.config.reference == minimal.config.reference
    assert pads.config.start_pad == minimal.config.start_pad
    assert pads.config.goal_pad == minimal.config.goal_pad
    assert pads.config.static_obstacles == ()
    assert pads.config.dynamic_obstacles == ()


@pytest.mark.parametrize(
    "dimensions",
    [
        Dimensions3D(0.0, 1.0, 1.0),
        Dimensions3D(1.0, -1.0, 1.0),
        Dimensions3D(1.0, 1.0, float("nan")),
    ],
)
def test_invalid_dimensions_are_rejected(dimensions: Dimensions3D) -> None:
    scene = _minimal()
    result = validate_scene(
        replace(
            scene,
            static_obstacles=(
                replace(scene.static_obstacles[0], dimensions=dimensions),
            ),
        )
    )
    assert not result.valid
    assert any(
        issue.code in {"invalid_positive_value", "non_finite"}
        for issue in result.errors
    )


def test_duplicate_and_invalid_names_are_rejected() -> None:
    scene = _minimal()
    duplicate = replace(scene.static_obstacles[0], name="GOAL-PAD")
    invalid = replace(scene.static_obstacles[1], name=" bad ")
    result = validate_scene(replace(scene, static_obstacles=(duplicate, invalid)))

    assert {issue.code for issue in result.errors} >= {
        "duplicate_name",
        "invalid_name",
    }


def test_longest_valid_runtime_name_stays_within_limit() -> None:
    value = runtime_object_name("a" * 32, "b" * 32, "0" * 64)
    assert len(value) <= 96


def test_rotated_bounds_are_conservative() -> None:
    obstacle = StaticObstacle(
        name="rotated",
        base_center=Vector3(5.0, 0.0, 0.0),
        dimensions=Dimensions3D(2.0, 1.0, 2.0),
        yaw_degrees=45.0,
    )
    bounds = conservative_bounds(obstacle)
    expected_half_extent = pytest.approx(2**0.5 / 2 + 2**0.5 / 4)
    assert bounds.max_x - 5.0 == expected_half_extent
    assert bounds.max_y == expected_half_extent


def test_boundary_contact_is_valid_for_workspace_but_overlap_for_objects() -> None:
    scene = _minimal()
    obstacle = StaticObstacle(
        name="boundary",
        base_center=Vector3(11.5, -5.5, 0.0),
        dimensions=Dimensions3D(1.0, 1.0, 1.0),
    )
    result = validate_scene(replace(scene, static_obstacles=(obstacle,)))
    assert result.valid

    touching = replace(obstacle, name="touching", base_center=Vector3(10.5, -5.5, 0.0))
    result = validate_scene(
        replace(
            scene,
            static_obstacles=(obstacle, touching),
            minimum_object_separation_m=0.0,
        )
    )
    assert any(issue.code == "obstacle_overlap" for issue in result.errors)


@pytest.mark.parametrize("kind", ["pad", "pad-safety", "obstacle"])
def test_initial_vehicle_exclusion_blocks_scene_geometry(kind: str) -> None:
    scene = _minimal()
    if kind == "pad":
        changed = replace(
            scene,
            start_pad=replace(scene.start_pad, base_center=Vector3(0.0, 0.0, 0.0)),
        )
    elif kind == "pad-safety":
        changed = replace(
            scene,
            start_pad=replace(scene.start_pad, base_center=Vector3(3.5, 0.0, 0.0)),
        )
    else:
        obstacle = replace(
            scene.static_obstacles[0],
            base_center=Vector3(0.0, 0.0, 0.0),
        )
        changed = replace(scene, static_obstacles=(obstacle,))
    result = validate_scene(changed)
    assert any(issue.code == "initial_vehicle_exclusion" for issue in result.errors)


def test_nonzero_world_translation_preserves_vehicle_exclusion() -> None:
    scene = require_valid_scene(_minimal())
    origin = Vector3(100.0, -20.0, 0.57)
    measured = Vector3(100.0, -20.0, 0.57)

    result = validate_world_vehicle_exclusion(scene, origin, measured)

    assert result.valid
    assert scene.config.start_pad.base_center == Vector3(4.0, 0.0, 0.0)


def test_pad_overlap_and_obstacle_safety_overlap_are_rejected() -> None:
    scene = _minimal()
    overlapping_goal = replace(scene.goal_pad, base_center=Vector3(5.0, 0.0, 0.0))
    result = validate_scene(replace(scene, goal_pad=overlapping_goal))
    assert any(issue.code == "pad_safety_overlap" for issue in result.errors)

    obstacle = replace(
        scene.static_obstacles[0],
        base_center=Vector3(4.0, 1.5, 0.0),
    )
    result = validate_scene(replace(scene, static_obstacles=(obstacle,)))
    assert any(issue.code == "obstacle_pad_safety_overlap" for issue in result.errors)


def test_anchor_and_approach_follow_ned_clearance() -> None:
    scene = _minimal()
    assert derive_start_anchor(scene.start_pad).z < scene.start_pad.base_center.z
    assert derive_goal_approach(scene.goal_pad).z < scene.goal_pad.base_center.z
    assert derive_goal_approach(scene.goal_pad).z == pytest.approx(-2.1)


def test_anchor_outside_workspace_is_rejected() -> None:
    scene = _minimal()
    shallow = replace(scene.workspace, min_z=-1.0)
    result = validate_scene(replace(scene, workspace=shallow))
    assert any(issue.code == "anchor_outside_workspace" for issue in result.errors)


def test_dynamic_schema_is_valid_but_runtime_remains_separate() -> None:
    scene = _minimal()
    dynamic = DynamicObstacle(
        name="future-dynamic",
        base_center=Vector3(6.5, -4.0, 0.0),
        dimensions=Dimensions3D(0.5, 0.5, 1.0),
        motion=ObstacleMotion(
            mode=MotionMode.WAYPOINT_LOOP,
            waypoints=(Vector3(6.5, -4.0, 0.0), Vector3(7.0, -4.0, 0.0)),
            speed_m_s=0.5,
        ),
    )
    result = validate_scene(
        replace(scene, static_obstacles=(), dynamic_obstacles=(dynamic,))
    )
    assert result.valid

    malformed = replace(dynamic, motion=replace(dynamic.motion, speed_m_s=-1.0))
    result = validate_scene(
        replace(scene, static_obstacles=(), dynamic_obstacles=(malformed,))
    )
    assert any(issue.path.endswith("speed_m_s") for issue in result.errors)


def test_same_seed_produces_identical_scene_without_global_rng_mutation() -> None:
    source = load_scene_config(GENERATED_CONFIG)
    random.seed(917)
    np.random.seed(917)
    python_before = copy.deepcopy(random.getstate())
    numpy_before = copy.deepcopy(np.random.get_state())

    first = resolve_scene(source)
    second = resolve_scene(source)

    assert first.canonical_json == second.canonical_json
    assert first.scene_digest == second.scene_digest
    assert [item.name for item in first.config.static_obstacles] == [
        "obstacle-000",
        "obstacle-001",
        "obstacle-002",
    ]
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


def test_different_seeds_produce_different_generated_scenes() -> None:
    source = load_scene_config(GENERATED_CONFIG)
    first = resolve_scene(source)
    second = resolve_scene(
        replace(source, generation=replace(source.generation, seed=43))
    )
    assert first.scene_digest != second.scene_digest


def test_impossible_generation_fails_with_bounded_attempts() -> None:
    source = load_scene_config(GENERATED_CONFIG)
    impossible = replace(
        source,
        generation=replace(
            source.generation,
            obstacle_count=1,
            x_range=(0.0, 0.0),
            y_range=(0.0, 0.0),
            max_attempts_per_object=2,
            max_total_attempts=2,
        ),
    )
    with pytest.raises(SceneGenerationError, match="2 total attempts"):
        generate_scene(impossible)


def test_scene_digest_is_independent_of_world_origin() -> None:
    scene = resolve_scene(_minimal())
    assert scene.scene_digest == scene_digest(scene.config)
    first = materialization_digest(
        local_scene_digest=scene.scene_digest,
        backend="runtime_spawn",
        backend_version="1",
        asset_catalog_digest="a" * 64,
        calibration_evidence={"Cube": "accepted"},
        world_origin=Vector3(0.0, 0.0, 0.0),
        requested_world_transforms={"start": [4.0, 0.0, -0.05]},
    )
    second = materialization_digest(
        local_scene_digest=scene.scene_digest,
        backend="runtime_spawn",
        backend_version="1",
        asset_catalog_digest="a" * 64,
        calibration_evidence={"Cube": "accepted"},
        world_origin=Vector3(50.0, 20.0, 0.5),
        requested_world_transforms={"start": [54.0, 20.0, 0.45]},
    )
    assert first != second
    assert scene.scene_digest == scene_digest(scene.config)


def test_committed_asset_catalog_records_accepted_cube_calibration() -> None:
    catalog = load_asset_catalog(ASSET_CONFIG)
    cube = catalog.calibration_for("Cube")

    assert cube is not None
    assert catalog.catalog_version == 2
    assert cube.status is AssetCalibrationStatus.ACCEPTED
    assert cube.evidence_level is CalibrationEvidenceLevel.OPERATOR_CONFIRMED_NOMINAL
    assert cube.nominal_dimensions_m == Dimensions3D(1.0, 1.0, 1.0)
    assert cube.uncertainty_m == pytest.approx(0.05)
    assert cube.tested_stack
    assert cube.evidence_reference
    assert cube.scale_readback_verified
    assert cube.accepted_for_materialization
    assert len(asset_catalog_digest(catalog)) == 64


def test_synthetic_unvalidated_asset_is_not_accepted_for_materialization() -> None:
    catalog = load_asset_catalog(ASSET_CONFIG)
    cube = catalog.calibration_for("Cube")
    assert cube is not None

    pending = replace(
        cube,
        nominal_dimensions_m=None,
        status=AssetCalibrationStatus.REQUIRES_LIVE_VALIDATION,
        evidence_level=CalibrationEvidenceLevel.UNVALIDATED,
        uncertainty_m=None,
        tested_stack=None,
        evidence_reference=None,
    )

    assert not pending.accepted_for_materialization


def test_collision_semantics_do_not_claim_verified_response() -> None:
    scene = _minimal()
    for item in (
        scene.start_pad,
        scene.goal_pad,
        *scene.static_obstacles,
    ):
        assert item.collision_intent is CollisionIntent.SOLID_EXPECTED
        assert item.physical_geometry_expected
        assert not item.physics_enabled
        assert not item.collision_response_verified

    false_claim = replace(scene.start_pad, collision_response_verified=True)
    result = validate_scene(replace(scene, start_pad=false_claim))
    assert any(issue.code == "unsupported_collision_claim" for issue in result.errors)


def test_canonical_scene_is_stable_and_strict_validation_raises() -> None:
    scene = _minimal()
    assert canonical_scene_json(scene) == canonical_scene_json(scene)
    with pytest.raises(SceneValidationError):
        require_valid_scene(
            replace(
                scene,
                start_pad=replace(
                    scene.start_pad,
                    dimensions=Dimensions3D(-1.0, 1.0, 0.1),
                ),
            )
        )


def test_m13_2_documentation_and_readme_contracts() -> None:
    document = (ROOT / "docs/m13_2_scene_specification.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    required_document_terms = (
        "ground-relative NED",
        "initial-vehicle exclusion",
        "collision_response_verified: false",
        "operator_confirmed_nominal",
        "scene_digest",
        "materialization_digest",
        "ownership manifest",
        "original safe ground",
        "touchdown_confirmation_attempts",
        "m13_2_materialize_20260724T091908_a4444a94.json",
        "M13.2 is complete",
        "M13.3",
        "M13.4",
    )
    for term in required_document_terms:
        assert term in document
    assert "Live materialization has not yet been accepted" not in document
    assert "m13_2_scene_specification.md" in normalized_readme
    assert "live M13.2 acceptance is still pending" not in normalized_readme
    assert (
        "M13.2 now validates deterministic live scene materialization"
        in normalized_readme
    )
    assert "M13.3 is the next milestone" in readme
