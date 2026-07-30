from __future__ import annotations

import json
import random
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from r1_uav_nav.training.long_run_state import (
    LongRunRunState,
    ResumeMode,
    canonical_digest,
    capture_rng_state,
    discover_latest_safe_checkpoint,
    restore_rng_state,
    save_checkpoint_bundle,
    validate_checkpoint_bundle,
    validate_resume_request,
)
from r1_uav_nav.training.long_run_training import (
    LongRunOutputConfig,
    load_long_run_td3_config,
    run_fake_td3_smoke,
    validate_long_run_configuration,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "training" / "m13_7_long_run_td3.yaml"


class ReplayBuffer:
    buffer_size = 64
    handle_timeout_termination = True
    optimize_memory_usage = False

    def __init__(self, size: int = 3) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class FakeBundleModel:
    def __init__(self) -> None:
        self.replay_buffer = ReplayBuffer()

    def save(self, path: Path) -> None:
        path.write_bytes(b"model")

    def save_replay_buffer(self, path: Path) -> None:
        path.write_bytes(b"replay")


def _state(config: dict[str, object], *, step: int, sequence: int) -> LongRunRunState:
    return LongRunRunState(
        run_id="test-run",
        created_at="2026-07-26T00:00:00+00:00",
        updated_at="2026-07-26T00:00:01+00:00",
        global_timesteps=step,
        checkpoint_sequence=sequence,
        full_config_digest=canonical_digest(config),
        compatibility_digest="c" * 64,
        environment_config_digest="e" * 64,
        observation_signature="o" * 64,
        action_signature="a" * 64,
    )


def test_committed_config_is_strict_and_uses_approved_validation_sets() -> None:
    config = load_long_run_td3_config(CONFIG_PATH)
    assert config.algorithm == "td3"
    assert config.checkpoint_interval_steps == 1000
    monitoring = [
        (item.profile_id, item.base_seed) for item in config.validation.monitoring_cases
    ]
    assert monitoring == [
        ("empty", 0),
        ("easy", 1100),
        ("medium", 2100),
    ]
    promotion = [
        (item.profile_id, item.base_seed) for item in config.validation.promotion_cases
    ]
    assert promotion == [
        ("empty", 0),
        ("held-out-reverse", 9100),
        ("held-out-elevated", 10100),
    ]
    assert config.validation.cleanup_hard_gate
    assert config.curriculum_id is None
    assert config.curriculum_config_digest is None
    assert "curriculum_id" not in config.resolved_snapshot()


def test_atomic_bundle_round_trip_and_latest_discovery(tmp_path: Path) -> None:
    config = {"schema_version": 1, "value": "stable"}
    run_root = tmp_path / "run"
    first = save_checkpoint_bundle(
        model=FakeBundleModel(),
        run_state=_state(config, step=10, sequence=1),
        resolved_config=config,
        run_root=run_root,
    )
    second = save_checkpoint_bundle(
        model=FakeBundleModel(),
        run_state=_state(config, step=20, sequence=2),
        resolved_config=config,
        run_root=run_root,
    )
    assert validate_checkpoint_bundle(first.directory) == first
    assert discover_latest_safe_checkpoint(run_root).directory == second.directory
    latest = json.loads((run_root / "latest.json").read_text(encoding="utf-8"))
    assert latest["bundle"] == second.directory.name


def test_corrupt_and_interrupted_bundles_are_not_discovered(tmp_path: Path) -> None:
    config = {"schema_version": 1}
    run_root = tmp_path / "run"
    valid = save_checkpoint_bundle(
        model=FakeBundleModel(),
        run_state=_state(config, step=5, sequence=1),
        resolved_config=config,
        run_root=run_root,
    )
    corrupt = save_checkpoint_bundle(
        model=FakeBundleModel(),
        run_state=_state(config, step=10, sequence=2),
        resolved_config=config,
        run_root=run_root,
    )
    (corrupt.directory / "model.zip").write_bytes(b"corrupt")
    interrupted = run_root / "checkpoints" / ".step.tmp-interrupted"
    interrupted.mkdir()
    (interrupted / "model.zip").write_bytes(b"partial")
    assert discover_latest_safe_checkpoint(run_root).directory == valid.directory
    with pytest.raises(ValueError, match="integrity"):
        validate_checkpoint_bundle(corrupt.directory)


def test_full_resume_and_model_only_warm_start_are_unambiguous(
    tmp_path: Path,
) -> None:
    config = {"schema_version": 1}
    safe = save_checkpoint_bundle(
        model=FakeBundleModel(),
        run_state=_state(config, step=7, sequence=1),
        resolved_config=config,
        run_root=tmp_path / "run",
    )
    full = validate_resume_request(
        resume_run_state=safe.run_state_path,
        expected_compatibility_digest="c" * 64,
    )
    assert full.mode is ResumeMode.FULL
    assert not full.reset_num_timesteps
    warm = validate_resume_request(
        resume_checkpoint=safe.model_path,
        allow_partial_resume=True,
        reset_num_timesteps=True,
    )
    assert warm.mode is ResumeMode.MODEL_ONLY_WARM_START
    assert warm.creates_new_lineage
    assert warm.source_model_digest is not None
    assert len(warm.source_model_digest) == 64
    with pytest.raises(ValueError, match="requires --resume-run-state"):
        validate_resume_request(resume_checkpoint=safe.model_path)
    with pytest.raises(ValueError, match="forbids replay"):
        validate_resume_request(
            resume_checkpoint=safe.model_path,
            resume_replay_buffer=safe.replay_buffer_path,
            allow_partial_resume=True,
            reset_num_timesteps=True,
        )


def test_rng_round_trip_restores_python_numpy_and_local_generator() -> None:
    original = capture_rng_state()
    try:
        random.seed(12)
        np.random.seed(13)
        local = np.random.default_rng(14)
        evidence = capture_rng_state(course_sampler=local)
        expected = (random.random(), float(np.random.random()), float(local.random()))
        random.seed(99)
        np.random.seed(99)
        local = np.random.default_rng(99)
        restored = restore_rng_state(evidence, course_sampler=local)
        actual = (random.random(), float(np.random.random()), float(local.random()))
        assert actual == pytest.approx(expected)
        assert restored["python_random"]
        assert restored["numpy_global"]
        assert restored["course_sampler"]
    finally:
        restore_rng_state(original)


def test_configuration_validation_is_offline_and_declares_training_pool(
    tmp_path: Path,
) -> None:
    _prepare_repository_config(tmp_path)
    config = replace(
        load_long_run_td3_config(CONFIG_PATH),
        output=LongRunOutputConfig(
            model_root="results/models",
            log_root="results/logs",
            report_root="results/reports",
        ),
    )
    evidence = validate_long_run_configuration(config, repository_root=tmp_path)
    assert ("easy", 1100) in evidence["training_pairs"]
    assert ("hard", 3400) in evidence["training_pairs"]
    assert evidence["live_training_enabled"] is False


def test_tiny_real_td3_checkpoint_replay_and_counter_resume(tmp_path: Path) -> None:
    _prepare_repository_config(tmp_path)
    config = replace(
        load_long_run_td3_config(CONFIG_PATH),
        additional_timesteps=4,
        learning_starts=1,
        batch_size=2,
        buffer_size=32,
        policy_hidden_layers=(8, 8),
        checkpoint_interval_steps=2,
        output=LongRunOutputConfig(
            model_root="results/models",
            log_root="results/logs",
            report_root="results/reports",
        ),
    )
    first = run_fake_td3_smoke(
        config,
        repository_root=tmp_path,
        run_id="resume-smoke",
    )
    assert first.final_timesteps == 4
    assert first.checkpoint.manifest.replay_buffer_size == 4
    assert {"profile_id": "easy", "base_seed": 1100} in (
        first.checkpoint.run_state.course_pool
    )
    plan = validate_resume_request(
        resume_latest=tmp_path / "results" / "models" / "resume-smoke",
        expected_compatibility_digest=config.compatibility_digest,
    )
    second = run_fake_td3_smoke(
        config,
        repository_root=tmp_path,
        resume_plan=plan,
    )
    assert second.initial_timesteps == 4
    assert second.final_timesteps == 8
    assert second.checkpoint.manifest.replay_buffer_size == 8


def test_fake_smoke_final_save_uses_one_effective_config_digest(
    tmp_path: Path,
) -> None:
    _prepare_repository_config(tmp_path)
    config = replace(
        load_long_run_td3_config(CONFIG_PATH),
        learning_starts=1,
        batch_size=2,
        buffer_size=32,
        policy_hidden_layers=(8, 8),
        checkpoint_interval_steps=1_000,
        output=LongRunOutputConfig(
            model_root="results/models",
            log_root="results/logs",
            report_root="results/reports",
        ),
    )
    result = run_fake_td3_smoke(
        config,
        repository_root=tmp_path,
        run_id="shorter-than-checkpoint-interval",
        additional_timesteps=4,
    )
    validated = validate_checkpoint_bundle(result.checkpoint.directory)
    assert validated == result.checkpoint
    assert validated.run_state.global_timesteps == 4
    assert validated.run_state.next_checkpoint_step == 1_000
    resolved = json.loads(
        (validated.directory / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert resolved["additional_timesteps"] == 4
    assert resolved["checkpoint_interval_steps"] == 1_000
    assert validated.run_state.full_config_digest == canonical_digest(resolved)


def _prepare_repository_config(root: Path) -> None:
    planning = root / "configs" / "planning"
    env = root / "configs" / "env"
    planning.mkdir(parents=True)
    env.mkdir(parents=True)
    shutil.copy(
        ROOT / "configs" / "planning" / "m13_3_voxel_astar.yaml",
        planning / "m13_3_voxel_astar.yaml",
    )
    shutil.copy(
        ROOT / "configs" / "env" / "m13_5_obstacle_uav_env.yaml",
        env / "m13_5_obstacle_uav_env.yaml",
    )
    (root / "results").mkdir()
