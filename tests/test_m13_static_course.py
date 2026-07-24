from __future__ import annotations

import importlib
import importlib.util
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from r1_uav_nav.sim.scene_specification import (
    DynamicObstacle,
    MotionMode,
    ObstacleMotion,
    load_scene_config,
    resolve_scene,
)
from r1_uav_nav.sim.static_course import (
    COURSE_REPORT_SCHEMA_VERSION,
    COURSE_SUITE_SCHEMA_VERSION,
    CourseGenerationExhaustedError,
    CourseSplit,
    XDirectionRequirement,
    baseline_from_course,
    course_report_dict,
    generate_solvable_course,
    load_course_suite_config,
    save_course_report,
    validate_static_course,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "planning" / "m13_3_voxel_astar.yaml"
SCRIPT_PATH = ROOT / "scripts" / "validate_static_course.py"
DOC_PATH = ROOT / "docs" / "m13_3_static_course_solvability.md"
README_PATH = ROOT / "README.md"


def _suite():
    return load_course_suite_config(SUITE_PATH)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_static_course", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_course_suite_has_fixed_clearance_schemas_and_seed_partitions():
    suite = _suite()

    assert suite.schema_version == COURSE_SUITE_SCHEMA_VERSION == 1
    assert suite.planner.schema_version == 1
    assert suite.planner.uav_collision_radius_m == pytest.approx(0.35)
    assert suite.planner.additional_safety_margin_m == pytest.approx(0.15)
    assert suite.planner.calculated_total_clearance_m == pytest.approx(0.50)
    assert suite.planner.total_clearance_m == pytest.approx(0.50)
    assert suite.planner.resolution_m == pytest.approx(0.25)
    assert suite.planner.connectivity == 26

    candidate_ranges: list[set[int]] = []
    for profile in suite.profiles:
        assert (ROOT / profile.scene_config).is_file()
        for base_seed in profile.base_seeds:
            candidate_ranges.append(
                set(
                    range(
                        base_seed,
                        base_seed + profile.max_candidate_attempts,
                    )
                )
            )
    for index, candidate_range in enumerate(candidate_ranges):
        for other in candidate_ranges[index + 1 :]:
            assert candidate_range.isdisjoint(other)


@pytest.mark.parametrize(
    ("profile_id", "base_seed"),
    [
        ("empty", 0),
        ("easy", 1100),
        ("easy", 1200),
        ("easy", 1300),
        ("easy", 1400),
        ("medium", 2100),
        ("medium", 2200),
        ("medium", 2300),
        ("medium", 2400),
        ("hard", 3100),
        ("hard", 3200),
        ("hard", 3300),
        ("hard", 3400),
        ("held-out-reverse", 9100),
        ("held-out-reverse", 9200),
        ("held-out-reverse", 9300),
        ("held-out-elevated", 10100),
        ("held-out-elevated", 10200),
        ("held-out-elevated", 10300),
    ],
)
def test_every_shipped_seed_matches_its_feasibility_baseline(
    profile_id: str,
    base_seed: int,
):
    suite = _suite()
    profile = suite.profile(profile_id)
    expected = profile.baseline_for(base_seed)
    assert expected is not None

    course = generate_solvable_course(
        suite,
        profile_id,
        base_seed,
        repository_root=ROOT,
    )
    actual = baseline_from_course(course)

    assert course.result.accepted
    assert course.result.attempt_index < profile.max_candidate_attempts
    assert actual.base_seed == expected.base_seed
    assert actual.accepted_candidate_seed == expected.accepted_candidate_seed
    assert actual.attempt_index == expected.attempt_index
    assert actual.scene_digest == expected.scene_digest
    assert actual.occupancy_digest == expected.occupancy_digest
    assert actual.solvability_digest == expected.solvability_digest
    assert actual.reference_path_length_m == pytest.approx(
        expected.reference_path_length_m,
        abs=1e-9,
    )
    assert actual.path_efficiency_ratio == pytest.approx(
        expected.path_efficiency_ratio,
        abs=1e-9,
    )
    assert actual.vertical_excursion_m == pytest.approx(
        expected.vertical_excursion_m,
        abs=1e-9,
    )
    assert actual.expanded_nodes == expected.expanded_nodes
    assert actual.direct_line_clear is expected.direct_line_clear


def test_same_seed_reproduces_scene_occupancy_path_and_digests():
    suite = _suite()

    first = generate_solvable_course(suite, "medium", 2100, repository_root=ROOT)
    second = generate_solvable_course(suite, "medium", 2100, repository_root=ROOT)

    assert first.scene == second.scene
    assert first.scene.scene_digest == second.scene.scene_digest
    assert first.grid.occupied_indices == second.grid.occupied_indices
    assert first.grid.occupancy_digest == second.grid.occupancy_digest
    assert first.result.path_result.voxel_path == second.result.path_result.voxel_path
    assert first.result.solvability_digest == second.result.solvability_digest


def test_different_declared_seeds_produce_different_generated_layouts():
    suite = _suite()

    first = generate_solvable_course(suite, "easy", 1100, repository_root=ROOT)
    second = generate_solvable_course(suite, "easy", 1200, repository_root=ROOT)

    assert first.scene.scene_digest != second.scene.scene_digest
    assert first.grid.occupancy_digest != second.grid.occupancy_digest


def test_generation_does_not_mutate_global_random_state():
    suite = _suite()
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    generate_solvable_course(suite, "easy", 1100, repository_root=ROOT)

    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]


def test_candidate_seed_replacement_does_not_change_m13_2_digest_semantics():
    suite = _suite()
    profile = suite.profile("easy")
    course = generate_solvable_course(suite, "easy", 1100, repository_root=ROOT)
    source = load_scene_config(ROOT / profile.scene_config)
    expected_scene = resolve_scene(
        replace(
            source,
            generation=replace(
                source.generation,
                seed=course.result.accepted_candidate_seed,
            ),
        )
    )

    assert course.scene.scene_digest == expected_scene.scene_digest


def test_impossible_constraints_exhaust_bounded_candidate_seeds():
    suite = _suite()
    source_profile = suite.profile("easy")
    impossible_profile = replace(
        source_profile,
        base_seeds=(1500,),
        max_candidate_attempts=2,
        constraints=replace(
            source_profile.constraints,
            direct_line_clear=False,
            path_efficiency_min=0.99,
        ),
        accepted_baselines=(),
    )
    impossible_suite = replace(
        suite,
        profiles=tuple(
            impossible_profile if item.profile_id == "easy" else item
            for item in suite.profiles
        ),
    )

    with pytest.raises(CourseGenerationExhaustedError) as caught:
        generate_solvable_course(
            impossible_suite,
            "easy",
            1500,
            repository_root=ROOT,
        )

    assert [item.attempt_index for item in caught.value.rejections] == [0, 1]
    assert [item.candidate_seed for item in caught.value.rejections] == [1500, 1501]


def test_dynamic_obstacle_schema_is_rejected_by_static_course_validation():
    suite = _suite()
    course = generate_solvable_course(suite, "medium", 2100, repository_root=ROOT)
    source = course.scene.config.static_obstacles[0]
    dynamic = DynamicObstacle(
        name="schema-only-dynamic",
        base_center=source.base_center,
        dimensions=source.dimensions,
        runtime_asset_name=source.runtime_asset_name,
        motion=ObstacleMotion(
            mode=MotionMode.WAYPOINT_LOOP,
            waypoints=(source.base_center,),
            speed_m_s=1.0,
        ),
    )
    changed_scene = replace(
        course.scene,
        config=replace(
            course.scene.config,
            dynamic_obstacles=(dynamic,),
        ),
    )

    with pytest.raises(ValueError, match="does not support dynamic obstacles"):
        validate_static_course(
            changed_scene,
            suite.planner,
            suite.profile("medium"),
            base_seed=2100,
            candidate_seed=2100,
            attempt_index=0,
        )


def test_profiles_enforce_declared_difficulty_and_held_out_direction():
    suite = _suite()

    for profile in suite.profiles:
        for seed in profile.base_seeds:
            course = generate_solvable_course(
                suite,
                profile.profile_id,
                seed,
                repository_root=ROOT,
            )
            result = course.result
            path = result.path_result
            constraints = profile.constraints
            assert constraints.obstacle_count_min <= result.obstacle_count
            assert result.obstacle_count <= constraints.obstacle_count_max
            assert path.direct_line_clear is constraints.direct_line_clear
            assert path.path_efficiency_ratio is not None
            assert (
                constraints.path_efficiency_min
                <= path.path_efficiency_ratio
                <= constraints.path_efficiency_max
            )
            assert path.vertical_excursion_m is not None
            assert path.vertical_excursion_m >= constraints.vertical_excursion_min_m
            if constraints.vertical_excursion_max_m is not None:
                assert path.vertical_excursion_m <= constraints.vertical_excursion_max_m
            names = [item.name for item in course.scene.config.static_obstacles]
            for prefix in constraints.required_structure_prefixes:
                assert any(name.startswith(prefix) for name in names)

            dx = course.scene.goal_approach.x - course.scene.start_anchor.x
            if constraints.required_x_direction is XDirectionRequirement.POSITIVE:
                assert dx > 0.0
            elif constraints.required_x_direction is XDirectionRequirement.NEGATIVE:
                assert dx < 0.0
            if profile.split is CourseSplit.HELD_OUT:
                assert seed >= 9000


def test_report_serializes_reference_path_schemas_and_limitations(tmp_path: Path):
    suite = _suite()
    course = generate_solvable_course(suite, "empty", 0, repository_root=ROOT)
    report_path = tmp_path / "course.json"

    save_course_report(course, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    in_memory_report = course_report_dict(course)
    assert report["scene_digest"] == in_memory_report["scene_digest"]
    assert report["occupancy_digest"] == in_memory_report["occupancy_digest"]
    assert report["solvability_digest"] == in_memory_report["solvability_digest"]
    assert report["report_schema_version"] == COURSE_REPORT_SCHEMA_VERSION
    assert report["course_suite_schema_version"] == COURSE_SUITE_SCHEMA_VERSION
    assert report["voxel_planner_config_schema_version"] == 1
    assert report["occupancy_evidence_schema_version"] == 1
    assert report["solvability_evidence_schema_version"] == 1
    assert report["reference_path_length_m"] > 0.0
    assert report["reference_path"]
    assert report["direct_line_voxel_indices"]
    assert report["limitations"] == {
        "built_in_blocks_geometry_included": False,
        "physical_collision_response_verified": False,
        "continuous_space_optimality_claimed": False,
    }


def test_offline_cli_parser_and_help_do_not_import_airsim(capsys):
    sys.modules.pop("airsim", None)
    module = _load_script()

    parser = module.build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["--help"])

    assert caught.value.code == 0
    assert "airsim" not in sys.modules
    assert "validate-all" in capsys.readouterr().out


def test_offline_cli_requires_seed_for_multi_seed_profile(tmp_path: Path):
    module = _load_script()
    args = module.parse_args(
        [
            "--course-config",
            str(SUITE_PATH),
            "validate",
            "--profile",
            "easy",
            "--output-path",
            str(tmp_path / "course.json"),
        ]
    )

    with pytest.raises(ValueError, match="--seed is required"):
        module.run(args, repository_root=ROOT)


def test_offline_cli_writes_versioned_course_report(tmp_path: Path):
    module = _load_script()
    report_path = tmp_path / "course.json"
    args = module.parse_args(
        [
            "--course-config",
            str(SUITE_PATH),
            "validate",
            "--profile",
            "empty",
            "--output-path",
            str(report_path),
        ]
    )

    assert module.run(args, repository_root=ROOT) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["profile_id"] == "empty"
    assert report["base_seed"] == 0
    assert report["accepted_candidate_seed"] == 0


def test_scene_manager_rejects_conflicting_profile_scene_before_client_import():
    script = importlib.import_module("scripts.manage_colosseum_scene")
    imported = False

    def fail_if_imported(_module_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("client import must not run")

    args = script.parse_args(
        [
            "--scene-config",
            str(ROOT / "configs" / "scenes" / "m13_2_minimal.yaml"),
            "--course-profile",
            "easy",
            "--course-seed",
            "1100",
            "materialize",
        ]
    )

    with pytest.raises(ValueError, match="conflicts"):
        script.run(
            args,
            repository_root=ROOT,
            client_module_loader=fail_if_imported,
        )
    assert not imported


def test_scene_manager_rejects_unsolvable_course_before_client_import(monkeypatch):
    script = importlib.import_module("scripts.manage_colosseum_scene")
    imported = False

    def reject_course(*_args, **_kwargs):
        raise CourseGenerationExhaustedError("easy", ())

    def fail_if_imported(_module_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("client import must not run")

    monkeypatch.setattr(script, "generate_solvable_course", reject_course)
    args = script.parse_args(
        [
            "--course-profile",
            "easy",
            "--course-seed",
            "1100",
            "materialize",
            "--allow-scene-mutation",
            "--confirm-scene-area-clear",
            "--confirm-no-visible-collision",
            "--allow-debug-markers",
            "--allow-marker-flush",
        ]
    )

    with pytest.raises(CourseGenerationExhaustedError):
        script.run(
            args,
            repository_root=ROOT,
            client_module_loader=fail_if_imported,
        )
    assert not imported


def test_scene_manager_course_profile_requires_declared_seed_before_client_import():
    script = importlib.import_module("scripts.manage_colosseum_scene")
    imported = False

    def fail_if_imported(_module_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("client import must not run")

    args = script.parse_args(
        [
            "--course-profile",
            "easy",
            "materialize",
        ]
    )

    with pytest.raises(ValueError, match="requires --course-seed"):
        script.run(
            args,
            repository_root=ROOT,
            client_module_loader=fail_if_imported,
        )
    assert not imported


def test_scene_manager_without_course_arguments_keeps_m13_2_resolution_path():
    script = importlib.import_module("scripts.manage_colosseum_scene")
    args = script.parse_args(
        [
            "--scene-config",
            str(ROOT / "configs" / "scenes" / "m13_2_minimal.yaml"),
            "validate",
        ]
    )

    scene, course = script._resolve_scene_or_course(args, ROOT)

    assert scene.config.scene_id == "minimal-course"
    assert course is None


def test_documentation_records_proof_boundary_and_next_milestone():
    document = DOC_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    for required in (
        "M13.3 is complete",
        "0.35",
        "0.15",
        "0.50",
        "L-infinity",
        "3D supercover",
        "reference_path_length_m",
        "undocumented built-in Blocks",
        "physical simulator collision response",
        "No live M13.3 simulator validation",
        "M13.4",
    ):
        assert required in document
    assert "M13.3 deterministic static-course generation" in readme
    assert "M13.4 is the current next milestone" in readme
    assert "docs/m13_3_static_course_solvability.md" in readme
