import json

import pytest

from r1_uav_nav.evaluation.static_comparison import (
    StaticLayout,
    assert_layouts_match,
    layouts_match,
    select_representative_shared_success_episode,
    summarise_static_comparison,
)


def test_matching_layouts_compare_equal() -> None:
    first = _layout()
    second = _layout(obstacles=frozenset({(2, 2), (1, 1)}))

    assert layouts_match(first, second)


@pytest.mark.parametrize(
    "changed_layout",
    [
        StaticLayout((1, 0), (4, 4), frozenset({(1, 1), (2, 2)}), 5),
        StaticLayout((0, 0), (3, 4), frozenset({(1, 1), (2, 2)}), 5),
        StaticLayout((0, 0), (4, 4), frozenset({(1, 1)}), 5),
        StaticLayout((0, 0), (4, 4), frozenset({(1, 1), (2, 2)}), 6),
    ],
)
def test_mismatched_layouts_are_detected(changed_layout: StaticLayout) -> None:
    assert layouts_match(_layout(), changed_layout) is False


def test_assert_layouts_match_raises_for_mismatch() -> None:
    with pytest.raises(ValueError, match="layouts do not match"):
        assert_layouts_match(
            _layout(),
            StaticLayout((1, 0), (4, 4), frozenset({(1, 1), (2, 2)}), 5),
        )


def test_comparison_summary_calculates_astar_successful_plan_averages() -> None:
    summary = summarise_static_comparison(
        _comparison_records(),
        eval_seed=42,
        model_path="results/trained_models/dqn_static_full.zip",
    )

    assert summary["astar"]["success_rate"] == pytest.approx(2 / 3)
    assert summary["astar"]["failure_rate"] == pytest.approx(1 / 3)
    assert summary["astar"]["average_successful_path_length"] == pytest.approx(5.0)
    assert summary["astar"]["average_successful_steps"] == pytest.approx(5.0)


def test_comparison_summary_calculates_dqn_all_episode_averages() -> None:
    summary = summarise_static_comparison(
        _comparison_records(),
        eval_seed=42,
        model_path="results/trained_models/dqn_static_full.zip",
    )

    assert summary["dqn"]["success_rate"] == pytest.approx(2 / 3)
    assert summary["dqn"]["collision_rate"] == pytest.approx(1 / 3)
    assert summary["dqn"]["timeout_rate"] == pytest.approx(0.0)
    assert summary["dqn"]["average_reward"] == pytest.approx(10 / 3)
    assert summary["dqn"]["average_path_length"] == pytest.approx(8.0)
    assert summary["dqn"]["average_steps"] == pytest.approx(8.0)


def test_comparison_summary_calculates_shared_success_metrics() -> None:
    summary = summarise_static_comparison(
        _comparison_records(),
        eval_seed=42,
        model_path="results/trained_models/dqn_static_full.zip",
    )
    shared_summary = summary["shared_success_comparison"]

    assert shared_summary["shared_success_count"] == 2
    assert shared_summary["dqn_failed_astar_succeeded_count"] == 0
    assert shared_summary["shared_success_average_astar_path_length"] == pytest.approx(
        5.0
    )
    assert shared_summary["shared_success_average_dqn_path_length"] == pytest.approx(
        5.5
    )
    assert shared_summary[
        "shared_success_path_length_difference_dqn_minus_astar"
    ] == pytest.approx(0.5)
    assert shared_summary[
        "dqn_to_astar_path_length_ratio_shared_success"
    ] == pytest.approx(1.1)


def test_representative_selection_prefers_shared_success_episode() -> None:
    selected = select_representative_shared_success_episode(_comparison_records())

    assert selected["episode_index"] in {0, 2}


def test_representative_selection_chooses_dqn_mean_length_shared_success() -> None:
    selected = select_representative_shared_success_episode(
        [
            _record(
                0,
                astar_success=True,
                dqn_success=True,
                astar_length=3.0,
                dqn_length=4.0,
            ),
            _record(
                1,
                astar_success=True,
                dqn_success=True,
                astar_length=8.0,
                dqn_length=10.0,
            ),
            _record(
                2,
                astar_success=True,
                dqn_success=True,
                astar_length=5.0,
                dqn_length=6.0,
            ),
        ]
    )

    assert selected["episode_index"] == 2


def test_representative_selection_falls_back_to_first_episode() -> None:
    records = [
        _record(
            0, astar_success=True, dqn_success=False, astar_length=4.0, dqn_length=8.0
        ),
        _record(
            1, astar_success=False, dqn_success=False, astar_length=0.0, dqn_length=5.0
        ),
    ]

    selected = select_representative_shared_success_episode(records)

    assert selected["episode_index"] == 0


def test_summary_dictionary_is_json_serialisable() -> None:
    summary = summarise_static_comparison(
        _comparison_records(),
        eval_seed=42,
        model_path="results/trained_models/dqn_static_full.zip",
    )

    json.dumps(summary)


def _layout(
    obstacles: frozenset[tuple[int, int]] = frozenset({(1, 1), (2, 2)}),
) -> StaticLayout:
    return StaticLayout(
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=obstacles,
        grid_size=5,
    )


def _comparison_records() -> list[dict]:
    return [
        _record(
            0, astar_success=True, dqn_success=True, astar_length=4.0, dqn_length=5.0
        ),
        _record(
            1,
            astar_success=False,
            dqn_success=False,
            astar_length=0.0,
            dqn_length=13.0,
            collision=True,
        ),
        _record(
            2, astar_success=True, dqn_success=True, astar_length=6.0, dqn_length=6.0
        ),
    ]


def _record(
    episode_index: int,
    *,
    astar_success: bool,
    dqn_success: bool,
    astar_length: float,
    dqn_length: float,
    collision: bool = False,
) -> dict:
    astar_steps = int(astar_length)
    dqn_steps = int(dqn_length)
    return {
        "episode_index": episode_index,
        "seed": 42 + episode_index,
        "layout": {
            "start_position": (0, 0),
            "goal_position": (4, 4),
            "obstacles": [(1, 1)],
            "grid_size": 5,
        },
        "astar": {
            "success": astar_success,
            "failure": not astar_success,
            "collision": False,
            "path": [(0, 0), (1, 0)] if astar_success else [(0, 0)],
            "steps": astar_steps if astar_success else 0,
            "path_length": astar_length if astar_success else 0.0,
        },
        "dqn": {
            "success": dqn_success,
            "collision": collision,
            "timeout": False,
            "total_reward": 10.0 if dqn_success else -10.0,
            "positions": [(0, 0), (0, 1)],
            "steps": dqn_steps,
            "path_length": dqn_length,
        },
    }
