import json

import pytest

from benchmarks.mindguard.ensemble_overlap import ROOT, summarize
from benchmarks.mindguard.surge_replay import Nomination


def test_overlap_counts_positives_not_turns_and_never_means_no_ticket_not_wrong_ticket() -> None:
    none = Nomination(None, None, 4, False)
    early = Nomination(1, 0.8, 4, False)
    on_time = Nomination(9, 0.8, 4, True)
    late = Nomination(10, 0.8, 4, True)
    false_alarm = Nomination(2, 0.8, None, False)
    fam = {"private1": "foo", "private2": "foo", "private3": "bar", "private4": "baz", "private5": "z"}
    models = {
        "a": dict(zip(fam, (early, none, none, on_time, false_alarm), strict=True)),
        "b": dict(zip(fam, (on_time, none, late, none, false_alarm), strict=True)),
    }
    r = summarize(fam, models)
    assert (r["observed_positive_paths"], r["positive_opening_families"], r["flagged_by_any"]) == (4, 3, 3)
    assert r["joint_never_flagged_counts"] == [[2, 1], [1, 2]]
    assert r["never_flagged_by_n_models_from_0_up"] == [1, 2, 1]
    assert r["observed_negative_flagged_by_any"] == 1
    assert r["by_model"]["b"]["first_flag_after_deadline"] == 1
    assert r["families_containing_a_path_missed_by_all"] == 1
    assert r["positive_families_with_no_positive_path_flagged_by_any"] == 0
    assert r["retrospective_union_of_immediate_supported_catches"] == 2
    assert r["first_to_flag_or_immediate_supported_catches"] == 1
    assert r["by_model"]["a"]["never_flagged_here_but_flagged_elsewhere"] == 1
    assert "private" not in json.dumps(r)
    with pytest.raises(ValueError, match="coverage"):
        summarize(fam, {"a": {}, "b": models["b"]})
    with pytest.raises(ValueError, match="first evidence"):
        summarize(fam, {"a": models["a"], "b": models["b"] | {"private1": Nomination(1, 0.8, 1, True)}})
    with pytest.raises(ValueError, match="same turn"):
        summarize(fam, {"a": models["a"], "b": models["b"] | {"private4": Nomination(9, 0.8, 4, False)}})


def test_released_audit_reconciles_distinct_coverage_with_model_intersections() -> None:
    r = json.loads((ROOT / "figures" / "ensemble-overlap.json").read_text())
    m = r["models_and_matrix_order"]
    matrix = r["joint_never_flagged_counts"]
    assert len(m) == 6
    assert r["observed_positive_paths"] == 17
    assert [row[i] for i, row in enumerate(matrix)] == [r["by_model"][k]["never_flagged_positive"] for k in m]
    assert matrix == [list(col) for col in zip(*matrix, strict=True)]
    histogram = r["never_flagged_by_n_models_from_0_up"]
    assert sum(histogram) == r["observed_positive_paths"]
    assert sum(i * n for i, n in enumerate(histogram)) == 49
    assert histogram[-1] == 3
    assert sum(f["never_flagged_by_any"] for f in r["family_blocked_fold_counts"]) == histogram[-1]
    assert r["flagged_by_any"] == 14
    assert r["first_to_flag_or_immediate_supported_catches"] == 10
