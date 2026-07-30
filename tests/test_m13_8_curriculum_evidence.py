from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from r1_uav_nav.training.curriculum_evidence import (
    CURRICULUM_BASELINE_SCHEMA_VERSION,
    BaselineEvidenceScope,
    CurriculumBaselineEvidence,
    TrajectorySummaryCollector,
    require_baseline_set,
    write_curriculum_report,
)


def _observation(position: tuple[float, float, float]) -> np.ndarray:
    result = np.zeros(83, dtype=np.float32)
    result[:3] = np.asarray(position, dtype=np.float32)
    result[10:83] = 1.0
    return result


def test_straight_trajectory_does_not_qualify() -> None:
    collector = TrajectorySummaryCollector(
        start_relative=(0.0, 0.0, 0.0),
        goal_relative=(4.0, 0.0, 0.0),
    )
    for point in ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)):
        collector.add(
            observation=_observation(point),
            position_scales_m=(1.0, 1.0, 1.0),
            info={"reward_clearance_m": 2.0},
        )
    result = collector.result(
        successful=True,
        collision=False,
        safety_violation=False,
    )
    assert result.travelled_to_direct_ratio == pytest.approx(1.0)
    assert result.maximum_perpendicular_deviation_m == pytest.approx(0.0)
    assert not result.non_straight_qualified


def test_lateral_and_vertical_detour_produce_sanitized_summary() -> None:
    collector = TrajectorySummaryCollector(
        start_relative=(0.0, 0.0, 0.0),
        goal_relative=(4.0, 0.0, 0.0),
    )
    for point in (
        (0, 0, 0),
        (1, 1.2, -0.5),
        (2, 1.5, -1.1),
        (3, 1.2, -0.5),
        (4, 0, 0),
    ):
        collector.add(
            observation=_observation(point),
            position_scales_m=(1.0, 1.0, 1.0),
            info={
                "minimum_lidar_clearance_m": 0.8,
                "path_length_m": float(
                    {
                        (0, 0, 0): 0.0,
                        (1, 1.2, -0.5): 1.7,
                        (2, 1.5, -1.1): 3.0,
                        (3, 1.2, -0.5): 4.3,
                        (4, 0, 0): 6.0,
                    }[point]
                ),
            },
        )
    result = collector.result(
        successful=True,
        collision=False,
        safety_violation=False,
    )
    assert result.travelled_to_direct_ratio > 1.05
    assert result.maximum_perpendicular_deviation_m >= 1.0
    assert result.maximum_lateral_deviation_m >= 1.0
    assert result.maximum_vertical_deviation_m >= 1.0
    assert result.minimum_clearance_m == pytest.approx(0.8)
    assert len(result.trajectory_digest) == 64
    assert result.non_straight_qualified
    assert not hasattr(result, "trajectory")


def test_baseline_scope_and_digest_mismatch_are_rejected() -> None:
    evidence = CurriculumBaselineEvidence(
        schema_version=CURRICULUM_BASELINE_SCHEMA_VERSION,
        evidence_scope=BaselineEvidenceScope.OFFLINE_FAKE,
        controller_id="oracle",
        privileged=True,
        controller_config_digest="c" * 64,
        suite_id="m13_8",
        profile_id="curriculum-blocker-validation",
        base_seed=21100,
        accepted_candidate_seed=21100,
        scene_digest="s" * 64,
        solvability_digest="v" * 64,
        success_rate=1.0,
        collision_rate=0.0,
        cleanup_success_rate=1.0,
        report_digest="r" * 64,
    )
    evidence.require_current(
        controller_config_digest="c" * 64,
        scene_digest="s" * 64,
        solvability_digest="v" * 64,
        required_scope=BaselineEvidenceScope.OFFLINE_FAKE,
    )
    with pytest.raises(ValueError, match="scope"):
        evidence.require_current(
            controller_config_digest="c" * 64,
            scene_digest="s" * 64,
            solvability_digest="v" * 64,
            required_scope=BaselineEvidenceScope.SUPERVISED_LIVE,
        )
    with pytest.raises(ValueError, match="stale"):
        evidence.require_current(
            controller_config_digest="x" * 64,
            scene_digest="s" * 64,
            solvability_digest="v" * 64,
            required_scope=BaselineEvidenceScope.OFFLINE_FAKE,
        )
    with pytest.raises(ValueError, match="missing required baselines"):
        require_baseline_set(
            (),
            required_controller_ids=("direct", "oracle"),
            required_scope=BaselineEvidenceScope.OFFLINE_FAKE,
            expected_controller_config_digests={
                "direct": "d" * 64,
                "oracle": "c" * 64,
            },
            scene_digest="s" * 64,
            solvability_digest="v" * 64,
        )


def test_report_writer_rejects_prohibited_payload_and_external_path(
    tmp_path: Path,
) -> None:
    root = tmp_path
    accepted = write_curriculum_report(
        root / "results" / "reports" / "summary.json",
        {"schema_version": 1, "stage_id": "stage-0"},
        repository_root=root,
    )
    assert accepted.is_file()
    with pytest.raises(ValueError, match="prohibited"):
        write_curriculum_report(
            root / "results" / "reports" / "raw.json",
            {"raw_point_cloud": [1, 2, 3]},
            repository_root=root,
        )
    with pytest.raises(ValueError, match="under results"):
        write_curriculum_report(
            root / "outside.json",
            {"schema_version": 1},
            repository_root=root,
        )
