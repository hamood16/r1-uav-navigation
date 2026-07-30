from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from r1_uav_nav.training.curriculum import (
    CURRICULUM_STATE_SCHEMA_VERSION,
    DEFAULT_CURRICULUM_CONFIG_PATH,
    CurriculumState,
    CurriculumStatus,
    PromotionStatus,
    curriculum_state_for_resume,
    enable_curriculum_mode,
    evaluate_promotion,
    load_curriculum_config,
    load_curriculum_state,
    save_curriculum_state,
    validate_curriculum_configuration,
)
from r1_uav_nav.training.curriculum_evidence import (
    CURRICULUM_VALIDATION_SCHEMA_VERSION,
    CurriculumValidationSummary,
)
from r1_uav_nav.training.long_run_state import LongRunRunState, ResumeMode
from r1_uav_nav.training.long_run_training import load_long_run_td3_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / DEFAULT_CURRICULUM_CONFIG_PATH
LONG_RUN_PATH = ROOT / "configs" / "training" / "m13_7_long_run_td3.yaml"


def _summary(
    stage_id: str,
    *,
    episode_count: int = 30,
    success_rate: float = 1.0,
    collision_rate: float = 0.0,
    timeout_rate: float = 0.0,
    cleanup_success_rate: float = 1.0,
    safety_violation_rate: float = 0.0,
    sensor_failure_rate: float = 0.0,
    non_straight_success_rate: float | None = 1.0,
    direct_success_advantage: float | None = 1.0,
    subgroup_success_rates: dict[str, float] | None = None,
) -> CurriculumValidationSummary:
    return CurriculumValidationSummary(
        schema_version=CURRICULUM_VALIDATION_SCHEMA_VERSION,
        stage_id=stage_id,
        global_timesteps=100_000,
        episode_count=episode_count,
        success_rate=success_rate,
        collision_rate=collision_rate,
        timeout_rate=timeout_rate,
        cleanup_success_rate=cleanup_success_rate,
        safety_violation_rate=safety_violation_rate,
        sensor_failure_rate=sensor_failure_rate,
        subgroup_success_rates=subgroup_success_rates or {},
        non_straight_success_rate=non_straight_success_rate,
        direct_success_advantage=direct_success_advantage,
    ).with_digest()


def test_committed_curriculum_has_exact_stages_and_thresholds() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    assert [stage.stage_id for stage in config.stages] == [
        f"stage-{index}" for index in range(6)
    ]
    assert config.required_consecutive_passes == 3
    assert config.replay_on_promotion == "preserve"
    assert config.stages[0].gate.minimum_success_rate == pytest.approx(0.95)
    assert config.stages[1].gate.minimum_non_straight_success_rate == pytest.approx(
        0.90
    )
    assert config.stages[4].gate.subgroup_minimum_success_rates == {
        "easy": 0.90,
        "medium": 0.80,
        "hard": 0.65,
    }
    assert config.stages[5].robustness is not None


def test_reserved_final_test_seeds_never_appear_in_curriculum_pools() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    used = {
        course.base_seed
        for stage in config.stages
        for course in (*stage.training_courses, *stage.validation_courses)
    }
    assert not used.intersection(config.reserved_final_test_seeds)
    assert set(config.reserved_final_test_seeds) == {
        9100,
        9200,
        9300,
        10100,
        10200,
        10300,
    }


def test_strict_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_curriculum_config(path)


def test_every_declared_course_passes_offline_feasibility() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    evidence = validate_curriculum_configuration(
        config,
        repository_root=ROOT,
        verify_courses=True,
    )
    assert evidence["course_count"] == len(evidence["feasibility"])
    assert evidence["feasibility_digest"] == (
        "7ab64205ccb7fd055d2ed0b5f42528d1e7be0772ed46224197427b96c6baa1b6"
    )
    blocker = next(
        item
        for item in evidence["feasibility"]
        if item["profile_id"] == "curriculum-blocker-train"
    )
    assert blocker["direct_line_clear"] is False
    assert blocker["accepted_candidate_seed"] == 21000


def test_three_consecutive_windows_promote_stage_zero() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    state = replace(
        CurriculumState.initial(config),
        stage_completed_timesteps=config.stages[0].minimum_stage_steps,
    )
    summary = _summary("stage-0", non_straight_success_rate=None)
    first, state = evaluate_promotion(config, state, summary)
    second, state = evaluate_promotion(config, state, summary)
    third, state = evaluate_promotion(config, state, summary)
    assert first.status is PromotionStatus.PASS_RECORDED
    assert second.consecutive_pass_count == 2
    assert third.status is PromotionStatus.PROMOTED
    assert third.promoted_to_stage_id == "stage-1"
    assert state.stage_id == "stage-1"
    assert state.completed_stage_ids == ("stage-0",)


def test_cleanup_failure_resets_passes_and_blocks_promotion() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    state = replace(
        CurriculumState.initial(config),
        stage_completed_timesteps=500,
        consecutive_pass_count=2,
    )
    decision, updated = evaluate_promotion(
        config,
        state,
        _summary(
            "stage-0",
            cleanup_success_rate=0.95,
            non_straight_success_rate=None,
        ),
    )
    assert decision.status is PromotionStatus.VALIDATION_FAILED
    assert "cleanup hard gate failed" in decision.failures
    assert updated.consecutive_pass_count == 0


def test_stage_four_advances_weight_levels_before_promotion() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    stage = config.stages[4]
    state = replace(
        CurriculumState.initial(config),
        stage_id=stage.stage_id,
        stage_index=stage.index,
        stage_completed_timesteps=stage.minimum_stage_steps,
        completed_stage_ids=("stage-0", "stage-1", "stage-2", "stage-3"),
    )
    summary = _summary(
        "stage-4",
        subgroup_success_rates={"easy": 1.0, "medium": 1.0, "hard": 1.0},
    )
    _, state = evaluate_promotion(config, state, summary)
    assert state.sampling_level_index == 1
    assert state.consecutive_pass_count == 0
    _, state = evaluate_promotion(config, state, summary)
    assert state.sampling_level_index == 2
    for _ in range(2):
        decision, state = evaluate_promotion(config, state, summary)
        assert decision.status is PromotionStatus.PASS_RECORDED
    decision, state = evaluate_promotion(config, state, summary)
    assert decision.status is PromotionStatus.PROMOTED
    assert state.stage_id == "stage-5"


def test_stage_budget_exhaustion_blocks_without_promoting() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    stage = config.stages[0]
    state = replace(
        CurriculumState.initial(config),
        stage_completed_timesteps=stage.maximum_stage_steps,
    )
    decision, updated = evaluate_promotion(
        config,
        state,
        _summary(
            "stage-0",
            success_rate=0.0,
            non_straight_success_rate=None,
        ),
    )
    assert decision.status is PromotionStatus.STAGE_BUDGET_EXHAUSTED
    assert updated.status is CurriculumStatus.BLOCKED
    assert updated.stage_id == "stage-0"


def test_curriculum_state_round_trip_and_long_run_acceptance() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    state = CurriculumState.initial(config)
    restored = CurriculumState.from_mapping(state.to_dict())
    assert restored == state
    long_run = LongRunRunState(
        run_id="curriculum-test",
        created_at="2026-07-29T00:00:00+00:00",
        updated_at="2026-07-29T00:00:01+00:00",
        full_config_digest="f" * 64,
        compatibility_digest="c" * 64,
        environment_config_digest="e" * 64,
        observation_signature="o" * 64,
        action_signature="a" * 64,
        curriculum_state=state.to_dict(),
    )
    assert long_run.curriculum_state["stage_id"] == "stage-0"
    assert long_run.curriculum_state["schema_version"] == (
        CURRICULUM_STATE_SCHEMA_VERSION
    )


def test_curriculum_state_file_round_trip_stays_under_results(
    tmp_path: Path,
) -> None:
    config = load_curriculum_config(CONFIG_PATH)
    state = CurriculumState.initial(config)
    path = save_curriculum_state(
        tmp_path / "results" / "reports" / "state.json",
        state,
        repository_root=tmp_path,
    )
    assert load_curriculum_state(path) == state


def test_legacy_m13_7_digest_is_unchanged_when_curriculum_is_disabled() -> None:
    config = load_long_run_td3_config(LONG_RUN_PATH)
    snapshot = config.resolved_snapshot()
    assert "curriculum_id" not in snapshot
    assert "curriculum_config_digest" not in snapshot
    assert config.curriculum_id is None


def test_curriculum_fields_are_strictly_paired() -> None:
    base = load_long_run_td3_config(LONG_RUN_PATH)
    with pytest.raises(ValueError, match="must be set together"):
        replace(base, curriculum_id="m13-8-static-obstacle")


def test_curriculum_mode_uses_separate_pool_without_mutating_m13_7_defaults() -> None:
    curriculum = load_curriculum_config(CONFIG_PATH)
    base = load_long_run_td3_config(LONG_RUN_PATH)
    enabled = enable_curriculum_mode(
        base,
        curriculum,
        stage_id="stage-3",
    )
    pairs = {
        (course.profile_id, course.base_seed)
        for course in curriculum.stages[3].validation_courses
    }
    assert enabled.validation == base.validation
    assert enabled.curriculum_config_digest == curriculum.config_digest
    assert not {
        9100,
        9200,
        9300,
        10100,
        10200,
        10300,
    }.intersection(seed for _, seed in pairs)


def test_state_rejects_invalid_status() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    raw = CurriculumState.initial(config).to_dict()
    raw["status"] = "unknown"
    with pytest.raises(ValueError):
        CurriculumState.from_mapping(raw)


def test_curriculum_status_defaults_active() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    assert CurriculumState.initial(config).status is CurriculumStatus.ACTIVE


def test_full_resume_restores_state_but_warm_start_resets_stage_zero() -> None:
    config = load_curriculum_config(CONFIG_PATH)
    source = replace(
        CurriculumState.initial(config),
        stage_id="stage-3",
        stage_index=3,
        stage_completed_timesteps=1234,
        completed_stage_ids=("stage-0", "stage-1", "stage-2"),
    )
    assert (
        curriculum_state_for_resume(
            config,
            resume_mode=ResumeMode.FULL,
            source_state=source.to_dict(),
        )
        == source
    )
    warm = curriculum_state_for_resume(
        config,
        resume_mode=ResumeMode.MODEL_ONLY_WARM_START,
        source_state=source.to_dict(),
    )
    assert warm.stage_id == "stage-0"
    assert warm.completed_stage_ids == ()
