from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from r1_uav_nav.training.long_run_state import capture_rng_state, restore_rng_state
from r1_uav_nav.training.validation_callback import (
    DEFAULT_MONITORING_CASES,
    DEFAULT_PROMOTION_CASES,
    DeterministicValidationCallback,
    ValidationEpisodeResult,
    ValidationRank,
    ValidationTier,
    calculate_validation_rank,
    summarize_validation,
)


class FakeModel:
    def __init__(self) -> None:
        self.saved: list[Path] = []
        self.num_timesteps = 20_000

    def save(self, path: Path) -> None:
        self.saved.append(Path(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake", encoding="utf-8")


def _results(cases, *, cleanup: bool = True, collision: bool = False):
    return tuple(
        ValidationEpisodeResult(
            profile_id=case.profile_id,
            base_seed=case.base_seed,
            episode_seed=case.episode_seed,
            episode_return=10.0,
            success=not collision,
            collision=collision,
            landing_success=cleanup,
            cleanup_success=cleanup,
            final_distance_m=0.2,
            travelled_path_length_m=10.0,
            reference_path_length_m=8.0,
            termination_reason="goal_reached" if not collision else "collision",
        )
        for case in cases
    )


def test_rank_uses_approved_lexicographic_order() -> None:
    good = calculate_validation_rank(_results(DEFAULT_PROMOTION_CASES))
    collided = calculate_validation_rank(
        _results(DEFAULT_PROMOTION_CASES, collision=True)
    )
    assert good.success_rate == 1.0
    assert good.mean_path_efficiency == pytest.approx(0.8)
    assert good > collided


def test_monitoring_is_never_eligible_for_promotion() -> None:
    summary = summarize_validation(
        tier=ValidationTier.MONITORING,
        cases=DEFAULT_MONITORING_CASES,
        results=_results(DEFAULT_MONITORING_CASES),
        global_timesteps=5_000,
    )
    assert summary.complete
    assert summary.cleanup_gate_passed
    assert not summary.eligible_for_promotion
    assert not summary.promoted


def test_cleanup_failure_is_hard_promotion_gate() -> None:
    summary = summarize_validation(
        tier=ValidationTier.PROMOTION,
        cases=DEFAULT_PROMOTION_CASES,
        results=_results(DEFAULT_PROMOTION_CASES, cleanup=False),
        global_timesteps=20_000,
    )
    assert not summary.cleanup_gate_passed
    assert not summary.eligible_for_promotion
    assert not summary.promoted
    assert "cleanup" in summary.failures[0]


def test_incomplete_result_set_cannot_promote() -> None:
    summary = summarize_validation(
        tier=ValidationTier.PROMOTION,
        cases=DEFAULT_PROMOTION_CASES,
        results=_results(DEFAULT_PROMOTION_CASES[:-1]),
        global_timesteps=20_000,
    )
    assert not summary.complete
    assert not summary.promoted


def test_callback_saves_best_only_for_strict_improvement(tmp_path: Path) -> None:
    def evaluator(*, cases, **_):
        return _results(cases)

    callback = DeterministicValidationCallback(
        evaluator=evaluator,
        report_directory=tmp_path / "reports",
        model_output_directory=tmp_path / "models",
        initial_best_rank=ValidationRank(0.5, 0.0, 1.0, 0.5, -2.0, 1.0),
    )
    model = FakeModel()
    callback.model = model
    callback.num_timesteps = 20_000
    first = callback.run_now(ValidationTier.PROMOTION)
    second = callback.run_now(ValidationTier.PROMOTION)
    assert first.promoted
    assert not second.promoted
    assert model.saved == [tmp_path / "models" / "best_validation_model.zip"]
    state = json.loads(
        (tmp_path / "models" / "best_validation_state.json").read_text(encoding="utf-8")
    )
    assert state["rank"] == list(first.rank.as_tuple())
    assert (tmp_path / "reports" / "validation_summary.csv").is_file()


def test_validation_evaluator_does_not_advance_global_rng_state(
    tmp_path: Path,
) -> None:
    original = capture_rng_state()
    try:
        random.seed(31)
        np.random.seed(32)
        expected = (random.random(), float(np.random.random()))
        random.seed(31)
        np.random.seed(32)

        def evaluator(*, cases, **_):
            random.random()
            np.random.random()
            return _results(cases)

        callback = DeterministicValidationCallback(
            evaluator=evaluator,
            report_directory=tmp_path / "reports",
            model_output_directory=tmp_path / "models",
        )
        callback.model = FakeModel()
        callback.num_timesteps = 5_000
        callback.run_now(ValidationTier.MONITORING)
        assert (random.random(), float(np.random.random())) == pytest.approx(expected)
    finally:
        restore_rng_state(original)
