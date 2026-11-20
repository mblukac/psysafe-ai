from copy import deepcopy
from typing import Any

import pytest

from benchmarks.mindguard.full_merge import merge_reports


def example(models: list[str]) -> dict[str, Any]:
    return dict(
        kind="mindguard_full_either_risk_v1",
        seed=2026,
        models=models,
        results=[dict(model=m, scenario="burst", staff=1, policy="fifo", rate=len(m) / 10) for m in models],
        training=[dict(model=m, fold=0, cut=0.1) for m in models],
        probability={m: dict(native=0.1, calibrated=0.2) for m in models},
    )


def test_deterministic_single_model_replication_can_expand_panel() -> None:
    assert merge_reports(example(["a", "b"]), example(["b", "c"])) == example(["a", "b", "c"])


def test_rejects_no_replication_wrong_settings_repeated_cells_and_nonidentical_training() -> None:
    a, b = example(["a", "b"]), example(["b", "c"])
    with pytest.raises(ValueError, match="replicated detector"):
        merge_reports(a, example(["c"]))
    bad = deepcopy(b)
    bad["training"][0]["cut"] = 0.2
    with pytest.raises(ValueError, match="does not exactly replicate"):
        merge_reports(a, bad)
    with pytest.raises(ValueError, match="incompatible study"):
        merge_reports(a, dict(b, seed=42))
    bad = deepcopy(b)
    bad["results"].append(bad["results"][0])
    with pytest.raises(ValueError, match="duplicated"):
        merge_reports(a, bad)
    with pytest.raises(ValueError, match="only independent"):
        merge_reports(
            dict(a, kind="full_mindguard_exact_across_detector_fifo_work_match_v1"),
            dict(b, kind="full_mindguard_exact_across_detector_fifo_work_match_v1"),
        )
