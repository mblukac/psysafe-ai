"""Use small complete event streams, including ties, to challenge the optimized joint work match."""

import random

import pytest

from benchmarks.mindguard.loss_workmatch import ARMS, aggregate, dispatch_starts, hits, plan
from benchmarks.mindguard.surge_queue import Case, simulate


@pytest.mark.parametrize("staff", [1, 2, 3, 0])
@pytest.mark.parametrize("mode", ["fifo", "risk"])
def test_queue_starts_match_primary_implementation_including_contested_ties(staff, mode):
    for seed in range(12):
        rng = random.Random(seed)  # noqa: S311 - synthetic tie-heavy arrivals
        tickets = sorted((float(rng.randrange(20) * 2), sid, rng.randrange(4) / 4) for sid in range(40))
        ordered = sorted(tickets, key=lambda x: x[1])
        cases = [Case(t, t, q, 1, None, False, False) for t, _, q in ordered]
        ref = {}
        simulate(cases, reviewers=staff or None, policy=mode, start_times=ref)
        assert dispatch_starts(tickets, staff, mode) == list(ref.items())


def test_expiry_equality_and_later_releases_do_not_starve_fifo_or_risk():
    # At 20, nomination 9.999 has expired, nomination 10 exactly has not.
    tickets = [(0, 0, 0.99), (9.999, 1, 1.0), (10, 2, 0.5), (21, 3, 1.0)]
    assert dispatch_starts(tickets, 1, "risk") == [(0, 0), (2, 20)]
    assert dispatch_starts(tickets, 1, "fifo") == [(0, 0), (2, 20)]


def test_payload_uses_only_visible_evidence_and_cap_drops_whole_starts():
    path = ((1, 0.9, 0), (3, 0.9, 1), (4, 0.2, 0), (5, 0.99, 0))
    p = plan(path, 0.7)
    # First prefix score is tied by turn 3 and exceeded only at turn 5.
    # At onset (minute 4), the latest-on-max policy—not earliest-on-max—can show turn 3.
    starts = [(0, 3.999999), (1, 4), (2, 6), (3, 8)]
    assert hits([p], [0, 0, 0, 0], [0] * 4, starts, 2) == ([0, 1, 0, 1], [0, 2, 0, 2])
    assert hits([p], [0, 0, 0, 0], [0] * 4, starts, 0)[0] == [0, 0, 0, 0]
    assert plan(path, 1.0).flagged is None
    with pytest.raises(ValueError, match="cap"):
        hits([p], [0], [0], starts[:1], 2)


def test_zero_budget_fold_remains_in_positive_denominator_and_loo_statistics():
    rows = [
        dict(
            model="x",
            scenario="s",
            staff=1,
            fold=f,
            rep=0,
            weight=w,
            positives=n,
            cap=cap,
            catches=[cap] * len(ARMS),
            full=[cap] * len(ARMS),
            queue_starts=[cap] * 4,
        )
        for f, w, n, cap in [(0, 2, 5, 0), (1, 3, 4, 2)]
    ]
    r = aggregate(rows)
    assert r["positives"] == 22 / 5
    assert r["matched_minutes"] == 24
    assert r["zero_budget_folds"] == [0]
    assert r["recall"] == [6 / 22] * len(ARMS)
    assert sum(f["weighted_positives"] for f in r["leave_fold_sufficient"]) == 22
    with pytest.raises(ValueError, match="duplicate block"):
        aggregate([rows[0], rows[0]])
