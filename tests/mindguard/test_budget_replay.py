from collections import Counter

import pytest

from benchmarks.mindguard.budget_replay import family_deletions, main, paired_prefix, summarize


def test_same_realized_spend_in_each_placement_not_just_on_average():
    runs = [
        (4, {"fixed": [False, True, True], "rl": [True], "rl_fifo": [True, True]}),
        (5, {"fixed": [True, False], "rl": [False, True, True], "rl_fifo": []}),
    ]
    r = summarize(runs, 3)
    assert r["paired_mean_actual_minutes"] == 30
    assert [x["minutes_each"] for x in r["per_placement"]] == [20, 40]
    assert r["paired_catches"] == {"fixed": 1, "rl": 2}
    assert r["placements_filling_cap_in_both"] == 0
    assert r["rl_paired_recall"] == 2 / 9  # all eligible arrivals, including after exhaustion
    assert r["fifo_ablation"]["paired_mean_actual_minutes"] == 20
    assert r["standalone_at_nominal_cap"]["fixed"]["mean_actual_minutes"] == 50
    assert r["standalone_at_nominal_cap"]["fixed"]["recall"] == 3 / 9


def test_prefix_not_an_oracle_selecting_best_reviews():
    assert paired_prefix([False, True, True], [False, False, True], 1) == (1, 0, 0)
    assert paired_prefix([True], [], 4) == (0, 0, 0)
    with pytest.raises(ValueError, match="nonnegative integer"):
        paired_prefix([], [], -1)


def test_deletion_holds_common_traffic_and_denominator_weights_fixed():
    r = family_deletions(Counter(a=2, b=3), Counter(a=2, b=1), Counter(a=1, b=3))
    assert r["min_max_delta_pp"] == pytest.approx([-50.0, 200 / 3])
    assert family_deletions(Counter(a=2), Counter(a=1), Counter())["min_max_delta_pp"] is None
    with pytest.raises(ValueError, match="evaluable held-out"):
        summarize([(0, {"fixed": [], "rl": [], "rl_fifo": []})], 10)


def test_rejects_unspendable_fraction_of_review_before_opening_any_file(capsys):
    assert main(["--annotations", "a", "--responses", "b", "--trace", "c", "--minutes", "21"]) == 2
    assert "ValueError" in capsys.readouterr().err
