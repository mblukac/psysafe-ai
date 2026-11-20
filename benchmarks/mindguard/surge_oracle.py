"""Hindsight throughput bounds for fixed tickets, equal service times and a small team."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from heapq import heapreplace
from math import inf, isfinite

from .surge_queue import Case

Window = tuple[float, float]  # release and inclusive latest timely start, not completion


def _schedule(windows: Iterable[Window], c: int, service: float) -> int:
    free = [-inf] * c  # a min heap; start each accepted ticket as early as possible
    completed = 0
    for release, latest in windows:
        start = max(release, free[0])
        if start <= latest:
            heapreplace(free, start + service)
            completed += 1
    return completed


def _component(jobs: list[Window], c: int, service: float) -> tuple[int, int]:
    # jobs are sorted by release. Extending each latest start to the running
    # maximum makes releases and deadlines agree in order. In this relaxation
    # greedily taking each feasible job is optimal: for equal durations, an
    # earlier-release, earlier-deadline job can replace a later one, and starts
    # of inverted jobs can be exchanged without delaying another job. Enlarged
    # windows include every original schedule, so this count is an upper bound.
    last = -inf
    relaxed = []
    for r, d in jobs:
        last = max(last, d)
        relaxed.append((r, last))
    hi = _schedule(relaxed, c, service)
    lo = max(_schedule(jobs, c, service), _schedule(sorted(jobs, key=lambda j: (j[1], j[0])), c, service))
    if lo == hi or len(jobs) > 8:
        return lo, hi

    def visit(remaining: tuple[Window, ...], free: tuple[float, ...], n: int) -> None:
        nonlocal lo
        # Equal-duration jobs can always be left-shifted on the earliest free
        # server in order of their starts in an optimal schedule. Enumerate
        # the first job of every feasible subsequence, including deliberate idle.
        viable = tuple(j for j in remaining if j[1] >= free[0])
        if n + len(viable) <= lo or lo == hi:
            return
        for idx, (r, d) in enumerate(viable):
            if idx and viable[idx - 1] == (r, d):
                continue
            start = max(r, free[0])
            lo = max(lo, n + 1)
            visit(viable[:idx] + viable[idx + 1 :], tuple(sorted((*free[1:], start + service))), n + 1)
            if lo == hi:
                break

    visit(tuple(jobs), (-inf,) * c, 0)
    return lo, lo


def oracle_counts(
    cases: Sequence[Case], reviewers: int, service_minutes: float, deadline_minutes: float = 10.0
) -> tuple[int, int]:
    """Bracket the best attainable hindsight count; never available to a live policy.

    The oracle may ignore any ticket or idle to await a future ticket, even if
    live dispatch must be work-conserving. It uses observed support and onset.
    Uncertain labels, skill matching, shifts and unknown durations are not solved.
    """
    if (
        type(reviewers) is not int
        or reviewers < 1
        or not all(isfinite(v) and v > 0 for v in (service_minutes, deadline_minutes))
    ):
        raise ValueError("need a positive team, service duration and deadline")
    windows = sorted(
        (j.nomination, j.onset + deadline_minutes)
        for j in cases
        if j.supported
        and j.onset is not None
        and j.nomination is not None
        and j.onset <= j.nomination <= j.onset + deadline_minutes
    )
    lo = hi = 0
    jobs: list[Window] = []
    end = -inf
    for r, d in windows:
        # No feasible occupation overlaps the next component, even if a job
        # starts at its inclusive deadline. Equality permits slot reuse.
        if jobs and r >= end:
            a, b = _component(jobs, reviewers, service_minutes)
            lo += a
            hi += b
            jobs = []
        jobs.append((r, d))
        end = max(end, d + service_minutes)
    if jobs:
        a, b = _component(jobs, reviewers, service_minutes)
        lo += a
        hi += b
    return lo, hi
