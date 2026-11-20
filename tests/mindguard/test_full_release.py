import json
from typing import Any

import pytest

from benchmarks.mindguard.full_release import FIELDS, export


def test_no_individual_cache_or_extra_fields_released() -> None:
    row = dict.fromkeys(FIELDS, 0) | {
        "model": "jev",
        "scenario": "steady",
        "staff": 1,
        "policy": "fifo",
        "by_period": {"peak": {"recall": 0.0}, "offpeak": {"recall": 0.0}},
        "source_text": "private fixture",
    }
    base: dict[str, Any] = dict.fromkeys(
        "gate service deadline_minutes turn_minutes placements training_episodes seed rated_turns dialogues families "
        "trace_sha256 azure_sha256 burst_sha256".split(),
        1,
    )
    base.update(
        kind="mindguard_full_either_risk_v1",
        models=["jev"],
        results=[row],
        counts={
            c: {"turns": 1, "dialogues": 1, "unreleased": "private fixture"}
            for c in ("either", "self_harm", "harm_others")
        },
        training=["private fixture"],
    )
    release = export(base)
    assert "private fixture" not in json.dumps(release)
    assert dict(zip(release["columns"], release["rows"][0], strict=True))["model"] == "jev"
    with pytest.raises(ValueError, match="duplicate"):
        export(base | {"results": [row, row]})
    with pytest.raises(ValueError, match="no operating"):
        export(base | {"results": []})
