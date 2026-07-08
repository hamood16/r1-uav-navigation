import pytest

from r1_uav_nav.evaluation import (
    EpisodeResult,
    calculate_path_length,
    summarise_episode_results,
)


def test_calculate_path_length_returns_zero_for_empty_or_single_position() -> None:
    assert calculate_path_length([]) == pytest.approx(0.0)
    assert calculate_path_length([(1, 2)]) == pytest.approx(0.0)


def test_calculate_path_length_sums_euclidean_segments() -> None:
    positions = [(0, 0), (3, 4), (6, 8)]

    assert calculate_path_length(positions) == pytest.approx(10.0)


def test_summarise_episode_results_calculates_success_rate() -> None:
    summary = summarise_episode_results(
        [
            EpisodeResult(1.0, 10, True, False, 5.0),
            EpisodeResult(-1.0, 8, False, True, 3.0),
            EpisodeResult(-0.5, 20, False, False, 4.0),
            EpisodeResult(1.0, 12, True, False, 6.0),
        ]
    )

    assert summary.success_rate == pytest.approx(0.5)


def test_summarise_episode_results_calculates_collision_rate() -> None:
    summary = summarise_episode_results(
        [
            EpisodeResult(1.0, 10, True, False, 5.0),
            EpisodeResult(-1.0, 8, False, True, 3.0),
            EpisodeResult(-1.0, 9, False, True, 2.0),
            EpisodeResult(-0.5, 20, False, False, 4.0),
        ]
    )

    assert summary.collision_rate == pytest.approx(0.5)


def test_summarise_episode_results_calculates_average_reward() -> None:
    summary = summarise_episode_results(
        [
            EpisodeResult(1.0, 10, True, False, 5.0),
            EpisodeResult(-1.0, 8, False, True, 3.0),
            EpisodeResult(0.5, 20, False, False, 4.0),
        ]
    )

    assert summary.average_reward == pytest.approx(1.0 / 6.0)


def test_summarise_episode_results_calculates_average_steps() -> None:
    summary = summarise_episode_results(
        [
            EpisodeResult(1.0, 10, True, False, 5.0),
            EpisodeResult(-1.0, 8, False, True, 3.0),
            EpisodeResult(0.5, 18, False, False, 4.0),
        ]
    )

    assert summary.average_steps == pytest.approx(12.0)


def test_summarise_episode_results_calculates_average_path_length() -> None:
    summary = summarise_episode_results(
        [
            EpisodeResult(1.0, 10, True, False, 5.0),
            EpisodeResult(-1.0, 8, False, True, 3.0),
            EpisodeResult(0.5, 18, False, False, 4.0),
        ]
    )

    assert summary.average_path_length == pytest.approx(4.0)


def test_summarise_episode_results_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="empty episode results"):
        summarise_episode_results([])
