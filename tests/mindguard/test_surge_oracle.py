from functools import cache
from random import Random

from benchmarks.mindguard.surge_oracle import _component, oracle_counts
from benchmarks.mindguard.surge_queue import Case


def _brute(jobs: list[tuple[int, int]], c: int, p: int) -> int:
    """Independent integer-time enumeration: assign or skip each job, any feasible slot."""

    @cache
    def best(i: int, load: tuple[int, ...]) -> int:
        if i == len(jobs):
            return 0
        answer = best(i + 1, load)
        r, d = jobs[i]
        for t in range(r, d + 1):
            if max(load[t : t + p]) < c:
                after = (*load[:t], *(v + 1 for v in load[t : t + p]), *load[t + p :])
                answer = max(answer, 1 + best(i + 1, after))
        return answer

    return best(0, (0,) * (max(d for _, d in jobs) + p))


def test_oracle_skips_wrong_and_late_alerts_and_reuses_boundary() -> None:
    tickets = [
        Case(0, 0, 0.8, 1, 0, False, False),
        Case(0, 11, 0.8, 2, 0, True, False),
        Case(0, 2, 0.8, 2, 0, True, False),
        Case(0, 0, 0.8, 1, 0, True, False),
        Case(0, None, None, None, 4, False, False),
    ]
    assert oracle_counts(tickets, 1, 10) == (2, 2)
    assert oracle_counts(tickets, 1, 20) == (1, 1)
    assert oracle_counts(tickets, 2, 20) == (2, 2)


def test_hindsight_reschedules_flexible_ticket() -> None:
    # Greedy release-order schedules only two; hindsight can get all three.
    jobs = [(0.0, 0.0), (2.0, 8.0), (3.0, 4.0)]
    assert _component(jobs, 1, 4) == (3, 3)


def test_agrees_with_independent_discrete_solver() -> None:
    rng = Random(283)  # noqa: S311 - deterministic test
    for _ in range(90):
        jobs = sorted((r, r + rng.randrange(3)) for r in (rng.randrange(7) for _ in range(rng.randint(2, 9))))
        staff, service = rng.randint(1, 2), rng.randint(1, 3)
        lo, hi = _component([(float(r), float(d)) for r, d in jobs], staff, service)
        exact = _brute(jobs, staff, service)
        assert lo <= exact <= hi
        if len(jobs) <= 8:
            assert lo == hi
