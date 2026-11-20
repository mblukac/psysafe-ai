"""Dispatch fixed nominations to reviewers on an exogenous, minute-based clock.

All policies see the same incoming tickets. They cannot change the nomination,
preempt service or read labels. Equal-time events: release, observe all arrivals,
start feasible tickets, expire the rest. Deadline equality is feasible. The
stream is drained even if service or nominations continue past the stated
staffing horizon. The horizon is a scenario parameter, never learned from the
time of the last actual ticket. 'peak' is attributed by *session arrival*.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any


@dataclass(frozen=True, slots=True)
class Case:
    """One session, including non-nominated sessions. Times are absolute minutes."""

    arrival: float
    nomination: float | None
    q: float | None
    turn: int | None
    onset: float | None  # first majority endorsement, for evaluation only
    supported: bool  # whether the nominated prefix has majority support
    peak: bool  # external flag for the session's arrival bin, not a policy feature


@dataclass(frozen=True, slots=True)
class Candidate:
    q: float
    age: float
    turn: int


@dataclass(frozen=True, slots=True)
class Context:
    waiting: int
    reviewers: int
    elapsed: float
    remaining: float


Select = Callable[[Sequence[Candidate], Context], int]


def rule_index(candidates: Sequence[Candidate], mode: str, deadline_minutes: float = 10.0) -> int:
    """Return an index into FIFO-ordered feasible candidates, using stable FIFO ties."""
    if not candidates:
        raise ValueError("need at least one feasible candidate")
    if mode == "fifo":
        return 0
    if mode == "risk":
        return max(range(len(candidates)), key=lambda i: candidates[i].q)
    if mode == "aging":
        return max(range(len(candidates)), key=lambda i: candidates[i].q + 0.10 * candidates[i].age / deadline_minutes)
    raise ValueError("unknown dispatch rule")


def _valid_time(time: float | None, lower: float) -> bool:
    return time is not None and math.isfinite(time) and time >= lower


def _summary() -> dict[str, int]:
    return dict.fromkeys(
        (
            "sessions",
            "positives",
            "nominated",
            "starts",
            "catches",
            "expired",
            "unsupported_starts",
            "misses_no_alert",
            "misses_wrong_turn",
            "misses_nomination_late",
            "misses_queue_delay",
            "misses_queue_expired",
            "misses_queue_started_late",
        ),
        0,
    )


def _with_rate(counts: dict[str, int]) -> dict[str, int | float | None]:
    return {**counts, "recall": counts["catches"] / counts["positives"] if counts["positives"] else None}


def _p95(values: list[float]) -> float | None:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def simulate(
    sessions: Sequence[Case],
    *,
    reviewers: int | None,
    service_minutes: float = 20.0,
    deadline_minutes: float = 10.0,
    policy: str = "fifo",
    select: Select | None = None,
    horizon_minutes: float = 43200.0,
    contest_starts: list[int] | None = None,
    start_hits: list[bool] | None = None,
) -> dict[str, Any]:
    """One-shot dispatch (no RL reward updates or labels during rollout).

    None reviewers = unlimited, immediate service; zero reviewers = no starts.
    A contest means *more waiting tickets than currently free reviewers* and at
    least one start now. Callback selects an index into a FIFO-ordered tuple of
    only the current feasible Candidates. No label, timestamp of future events,
    case identity, case length, or realized outcome enters that API.

    Two optional local-only training audits extend caller-owned lists *after*
    rollout: contest_starts (zero-based start positions within this run) and
    start_hits (onset-deadline binary outcomes of every start, in dispatch order).
    """
    if reviewers is not None and (type(reviewers) is not int or reviewers < 0):
        raise ValueError("reviewers must be a nonnegative integer or None")
    if not all(math.isfinite(n) and n > 0 for n in (service_minutes, deadline_minutes, horizon_minutes)):
        raise ValueError("service, deadline and planning horizon must be positive finite minutes")
    if policy not in ("fifo", "risk", "aging", "learned") or (policy == "learned") != (select is not None):
        raise ValueError("expected built-in policy or learned with a selector")

    # The dispatch table intentionally carries only operational state.
    arrivals: list[tuple[float, int, float, int]] = []
    for sid, case in enumerate(sessions):
        if (
            not _valid_time(case.arrival, 0.0)
            or (case.onset is not None and not _valid_time(case.onset, case.arrival))
            or type(case.supported) is not bool
            or type(case.peak) is not bool
        ):
            raise ValueError("invalid session times or reference flags")
        if case.nomination is None:
            if case.q is not None or case.turn is not None or case.supported:
                raise ValueError("non-nomination must have no score, turn or nominated support")
            continue
        if (
            not _valid_time(case.nomination, case.arrival)
            or type(case.turn) is not int
            or case.turn < 1
            or case.q is None
            or not math.isfinite(case.q)
            or not 0 <= case.q <= 1
            or (case.supported and (case.onset is None or case.onset > case.nomination))
        ):
            raise ValueError("invalid nominated ticket or inconsistent reference")
        arrivals.append((case.nomination, sid, case.q, case.turn))
    arrivals.sort()

    cursor = 0
    waiting: dict[int, tuple[float, float, int]] = {}  # FIFO insertion order, tie by stream position
    expiry: deque[tuple[float, int]] = deque()
    busy: list[float] = []
    started: dict[int, float] = {}  # insertion order is exact dispatch order
    contests: list[int] = []
    max_waiting = 0
    while cursor < len(arrivals) or waiting:
        while expiry and expiry[0][1] not in waiting:
            expiry.popleft()
        now = min(
            arrivals[cursor][0] if cursor < len(arrivals) else math.inf,
            busy[0] if busy else math.inf,
            expiry[0][0] if expiry else math.inf,
        )
        while busy and busy[0] <= now:
            heappop(busy)
        while cursor < len(arrivals) and arrivals[cursor][0] == now:
            at, sid, q, turn = arrivals[cursor]
            waiting[sid] = (at, q, turn)
            expiry.append((at + deadline_minutes, sid))
            cursor += 1
        while waiting and (reviewers is None or len(busy) < reviewers):
            sid = next(iter(waiting))
            # If all tickets can start now, servicing in FIFO order costs no one
            # a slot; there is no scarce action, exploration or learning event.
            if reviewers is not None and len(waiting) > reviewers - len(busy):
                contests.append(len(started))
                if policy != "fifo":
                    ids = tuple(waiting)
                    cands = tuple(Candidate(q, now - at, t) for at, q, t in waiting.values())
                    if select is None:
                        index = rule_index(cands, mode=policy, deadline_minutes=deadline_minutes)
                    else:
                        index = select(cands, Context(len(waiting), reviewers, now, max(0.0, horizon_minutes - now)))
                    if type(index) is not int or not 0 <= index < len(ids):
                        raise ValueError("selector must choose an index into feasible candidates")
                    sid = ids[index]
            del waiting[sid]
            started[sid] = now
            if reviewers is not None:
                heappush(busy, now + service_minutes)
        while expiry and expiry[0][0] <= now:
            _, sid = expiry.popleft()
            waiting.pop(sid, None)
        max_waiting = max(max_waiting, len(waiting))

    # The joins below, including both caller-visible audits, run only after
    # every decision has already been made.
    periods = {"peak": _summary(), "offpeak": _summary()}
    timely: set[int] = set()
    start_wait: list[float] = []
    in_horizon_busy = 0.0
    for sid, case in enumerate(sessions):
        counts = periods["peak" if case.peak else "offpeak"]
        counts["sessions"] += 1
        counts["positives"] += int(case.onset is not None)
        if case.nomination is None:
            counts["misses_no_alert"] += int(case.onset is not None)
            continue
        counts["nominated"] += 1
        start_at = started.get(sid)
        counts["expired"] += int(start_at is None)
        if start_at is not None:
            counts["starts"] += 1
            counts["unsupported_starts"] += int(not case.supported)
            start_wait.append(start_at - case.nomination)
            in_horizon_busy += max(0.0, min(service_minutes, horizon_minutes - start_at))
        if case.onset is None:
            continue
        if not case.supported:
            counts["misses_wrong_turn"] += 1
        elif case.nomination > case.onset + deadline_minutes:
            counts["misses_nomination_late"] += 1
        elif start_at is None or start_at > case.onset + deadline_minutes:
            counts["misses_queue_delay"] += 1
            counts["misses_queue_expired" if start_at is None else "misses_queue_started_late"] += 1
        else:
            counts["catches"] += 1
            timely.add(sid)
    if start_hits is not None:
        start_hits.extend(sid in timely for sid in started)
    if contest_starts is not None:
        contest_starts.extend(contests)
    totals = {k: periods["peak"][k] + periods["offpeak"][k] for k in periods["peak"]}
    return {
        **_with_rate(totals),
        "by_period": {name: _with_rate(c) for name, c in periods.items()},
        "contested_starts": len(contests),
        "review_minutes": len(started) * service_minutes,
        "utilization": in_horizon_busy / (reviewers * horizon_minutes) if reviewers else None,
        "max_waiting": max_waiting,
        "mean_wait_started": sum(start_wait) / len(start_wait) if start_wait else None,
        "p95_wait_started": _p95(start_wait),
        # Time in queue until start or expiry, avoiding survivorship from only measuring starts.
        "p95_wait_nominated": _p95(start_wait + [deadline_minutes] * totals["expired"]),
    }
