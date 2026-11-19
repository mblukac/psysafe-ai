"""Non-preemptive human review on an exogenous stream, in minutes.

Each policy locks one nominated prefix per session, even if its ticket expires.
A reviewer may start through nominated_at + deadline (inclusive), then stays
busy for service_minutes. The end of the trace limits *arrivals*, not service.
At equal timestamps: free reviewers, observe all new scores, dispatch, expire.
Labels and future messages are never policy features. Unrated positions are
not free/negative observations: supply fully assessed trajectories only.

Fixed uses a constant cutoff and FIFO. Paced and RL use the identical controller
and differ only in the supplied metric (calibrated current probability versus
fitted stop advantage). The cutoff increases by 0.10 * min(1, w/deadline),
where w is the time to the first release plus floor(queue/staff) service times
if all staff are busy; w is zero when someone is idle. All arrivals in one
timestamp see the same pre-dispatch pressure. Among admitted tickets choose the
largest metric + 0.10 * waiting_time/deadline (ties: older, then input order).
These coefficients are explicit design assumptions, not learned staffing data.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush

Trajectory = tuple[tuple[int, float, int], ...]
_PRESSURE = 0.10
_AGING = 0.10


@dataclass(frozen=True)
class Session:
    arrival: float
    turns: Trajectory  # (one-based user turn index, raw or precomputed policy score, binary reference)


def simulate(
    sessions: Sequence[Session],
    *,
    reviewers: int,
    service_minutes: float,
    turn_minutes: float,
    deadline_minutes: float,
    policy: str,
    metric: Callable[[float, int], float],
    threshold: float,
) -> dict[str, int | float | None]:
    """Run one stream/policy. metric is a pure current-score/current-index transform.

    Nominal candidates can be pre-indexed without looking ahead in decisions:
    congestion only ever *raises* the cutoff. The clock exposes these records
    one timestamp at a time; priority uses only locked tickets already seen.
    Misses_no_alert, misses_wrong_turn, and misses_late partition all misses.
    Late means a *supported* nominated turn expired before its review began.
    """
    if type(reviewers) is not int or reviewers < 0:
        raise ValueError("reviewers must be a nonnegative integer")
    if policy not in ("fixed", "paced", "rl"):
        raise ValueError("policy must be fixed, paced or rl")
    if not all(math.isfinite(v) and v > 0 for v in (service_minutes, turn_minutes, deadline_minutes)):
        raise ValueError("service, turn and deadline minutes must be positive and finite")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    positive: list[bool] = []
    measurements: list[tuple[float, int, int, float]] = []
    prepared: dict[Trajectory, tuple[bool, list[tuple[int, float]]]] = {}
    for sid, session in enumerate(sessions):
        if not math.isfinite(session.arrival) or session.arrival < 0:
            raise ValueError("arrival minutes must be finite and nonnegative")
        if session.turns not in prepared:
            if not session.turns:
                raise ValueError("empty session")
            candidates: list[tuple[int, float]] = []
            for offset, (turn, score, reference) in enumerate(session.turns):
                if type(turn) is not int or turn != offset + 1:
                    raise ValueError("only complete, consecutive, one-based turns are assessable")
                if not math.isfinite(score) or not -1 <= score <= 1 or reference not in (0, 1):
                    raise ValueError("expected bounded finite score and binary reference")
                value = metric(score, turn)
                if not math.isfinite(value):
                    raise ValueError("metric must be finite")
                if value >= threshold:
                    candidates.append((offset, value))
            prepared[session.turns] = (any(t[2] for t in session.turns), candidates)
        has_support, candidates = prepared[session.turns]
        positive.append(has_support)
        measurements.extend(
            (session.arrival + turn_minutes * offset, sid, offset, value) for offset, value in candidates
        )
    measurements.sort()
    cursor = 0
    nominations: dict[int, int] = {}
    busy_until: list[float] = []
    pending: set[int] = set()
    priority: list[tuple[float, float, int]] = []
    expiry: deque[tuple[float, int]] = deque()  # identical TTLs: ticket admission order = deadline order
    started: dict[int, float] = {}
    waited = 0.0
    max_queue = 0
    while cursor < len(measurements) or pending:
        while expiry and expiry[0][1] not in pending:
            expiry.popleft()
        now = min(
            measurements[cursor][0] if cursor < len(measurements) else math.inf,
            busy_until[0] if busy_until else math.inf,
            expiry[0][0] if expiry else math.inf,
        )
        while busy_until and busy_until[0] <= now:
            heappop(busy_until)
        cutoff = threshold
        if policy != "fixed" and len(busy_until) == reviewers:
            if reviewers:
                wait_estimate = busy_until[0] - now + (len(pending) // reviewers) * service_minutes
                cutoff += _PRESSURE * min(1.0, wait_estimate / deadline_minutes)
            else:
                cutoff += _PRESSURE
        # Observe every simultaneous measurement before using any free capacity.
        while cursor < len(measurements) and measurements[cursor][0] == now:
            at, sid, offset, value = measurements[cursor]
            cursor += 1
            if sid in nominations or value < cutoff:
                continue
            nominations[sid] = offset
            pending.add(sid)
            expiry.append((at + deadline_minutes, sid))
            # The common +AGING * current_time/deadline cancels across candidates.
            rank = 0.0 if policy == "fixed" else -value + _AGING * at / deadline_minutes
            heappush(priority, (rank, at, sid))
        while pending and len(busy_until) < reviewers:
            _, at, sid = heappop(priority)
            if sid not in pending:
                continue
            pending.remove(sid)
            started[sid] = now
            waited += now - at
            heappush(busy_until, now + service_minutes)
        while expiry and expiry[0][0] <= now:
            _, sid = expiry.popleft()
            pending.discard(sid)
        max_queue = max(max_queue, len(pending))

    # Join to reference labels only for evaluation. No future support can
    # retrospectively redeem a ticket nominating a different, unsupported turn.
    supported = {sid for sid, i in nominations.items() if sessions[sid].turns[i][2] == 1}
    positives = sum(positive)
    timely = supported & started.keys()
    catches = len(timely)
    # Secondary, stricter clock: a newly endorsed later turn must not
    # silently restart the deadline for the first observed positive signal.
    catches_by_onset = 0
    for sid in timely:
        session = sessions[sid]
        first = next(offset for offset, (_, _, y) in enumerate(session.turns) if y)
        onset = session.arrival + first * turn_minutes
        catches_by_onset += int(started[sid] <= onset + deadline_minutes)
    no_alert = sum(y and sid not in nominations for sid, y in enumerate(positive))
    wrong_turn = sum(positive[sid] and sid not in supported for sid in nominations)
    late = len(supported - started.keys())
    return {
        "arrivals": len(sessions),
        "positives": positives,
        "alerts": len(nominations),
        "reviews": len(started),
        "catches": catches,
        "catches_by_onset_deadline": catches_by_onset,
        "misses": positives - catches,
        "misses_no_alert": no_alert,
        "misses_wrong_turn": wrong_turn,
        "misses_late": late,
        "misses_no_supported_alert": no_alert + wrong_turn,
        "misses_expired": late,
        "misses_unsupported_review": sum(positive[sid] and sid not in supported for sid in started),
        "unsupported_reviews": len(started) - catches,
        "expired": len(nominations) - len(started),
        "max_queue": max_queue,
        "review_minutes": len(started) * service_minutes,
        "recall": catches / positives if positives else None,
        "recall_by_onset_deadline": catches_by_onset / positives if positives else None,
        "mean_wait_minutes": waited / len(started) if started else None,
    }
