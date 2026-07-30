from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from r1_uav_nav.training.curriculum import load_curriculum_config
from r1_uav_nav.training.curriculum_sampling import (
    DeterministicCurriculumSampler,
    RobustnessPerturbationSampler,
    apply_policy_view_perturbation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "training" / "m13_8_static_obstacle_curriculum.yaml"


def test_course_sampling_is_reproducible_and_stateful() -> None:
    courses = load_curriculum_config(CONFIG_PATH).stages[3].training_courses
    first = DeterministicCurriculumSampler(courses, seed=1380)
    expected = [first.sample().identity for _ in range(8)]
    snapshot = first.snapshot()
    continued = [first.sample().identity for _ in range(5)]

    replay = DeterministicCurriculumSampler(courses, seed=0, state=snapshot)
    assert [replay.sample().identity for _ in range(5)] == continued
    repeated = DeterministicCurriculumSampler(courses, seed=1380)
    assert [repeated.sample().identity for _ in range(8)] == expected


def test_curriculum_sampling_does_not_mutate_global_rng() -> None:
    courses = load_curriculum_config(CONFIG_PATH).stages[2].training_courses
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    sampler = DeterministicCurriculumSampler(courses, seed=42)
    for _ in range(10):
        sampler.sample()
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    assert np.array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]


def test_stage_four_sampling_levels_are_locked() -> None:
    stage = load_curriculum_config(CONFIG_PATH).stages[4]
    levels = stage.sampling_levels
    assert [level.difficulty_weights for level in levels] == [
        {"easy": 0.60, "medium": 0.30, "hard": 0.10},
        {"easy": 0.45, "medium": 0.35, "hard": 0.20},
        {"easy": 0.30, "medium": 0.40, "hard": 0.30},
    ]
    sampler = DeterministicCurriculumSampler(
        stage.training_courses,
        seed=88,
        difficulty_weights=levels[-1].difficulty_weights,
    )
    difficulties = [sampler.sample().difficulty for _ in range(20)]
    assert set(difficulties) == {"easy", "medium", "hard"}


def test_stage_five_perturbations_are_deterministic_and_bounded() -> None:
    robustness = load_curriculum_config(CONFIG_PATH).stages[5].robustness
    assert robustness is not None
    first = RobustnessPerturbationSampler(robustness, seed=55).sample()
    second = RobustnessPerturbationSampler(robustness, seed=55).sample()
    assert first == second
    assert 0 <= first.lidar_noise_std <= 0.02
    assert 0 <= first.sector_dropout_probability <= 0.05
    assert 0.9 <= first.velocity_response_scale <= 1.1
    assert 0.9 <= first.control_duration_scale <= 1.1


def test_policy_view_transform_does_not_mutate_source_or_valid_flag() -> None:
    robustness = load_curriculum_config(CONFIG_PATH).stages[5].robustness
    assert robustness is not None
    perturbation = RobustnessPerturbationSampler(robustness, seed=70).sample()
    source = np.ones(83, dtype=np.float32)
    source[:10] = 0.25
    original = source.copy()
    result = apply_policy_view_perturbation(source, perturbation)
    assert np.array_equal(source, original)
    assert result.shape == (83,)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert np.all((0 <= result[10:82]) & (result[10:82] <= 1))
    assert result[82] == pytest.approx(1.0)
    assert np.array_equal(result[:10], source[:10])
