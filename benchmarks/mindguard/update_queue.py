"""Single-ticket, live prefix replacement with identical FIFO reviewer starts.

An update can change the prefix shown when a ticket is opened, not its queue
position, release time, expiry, service duration, or who gets a review. The
clock and end point are the original either-risk full-panel end point. A later
turn is visible at a start-time tie; nothing after service starts is visible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .staffing_queue import Trajectory
from .surge_queue import simulate
from .surge_replay import Nomination, attach, nominate

ARMS = ("frozen", "latest", "latest_qualifying", "max_score", "bayes", "paired_bayes", "visible_oracle")


class PairDecision(Protocol):
    def replace(self, original_q: float, challenger_q: float) -> bool: ...


@dataclass(frozen=True, slots=True)
class UpdatePath:
    nomination: Nomination
    paired: bool
    # One row per observed position at/after ticket release; missing turn
    # indices are never compressed. Rows contain *already fixed* selections.
    turns: tuple[int, ...]
    selected: tuple[tuple[int, ...], ...]  # Turn indices, aligned to ARMS.
    supported: tuple[tuple[bool, ...], ...]  # Evaluation labels for selections.


def prepare_path(
    path: Trajectory, bayes_scores: Sequence[float], cutoff: float, *, pair_rule: PairDecision | None = None
) -> UpdatePath:
    """Prepare forward-only updates. Bayesian scores must themselves be causal.

    Selection policy never sees any votes. Only visible_oracle is allowed to
    pick by endorsement retrospectively; it is a bound on these presentations
    at the same fixed starts, not an operational policy. Use chronological
    prefix argmax, not argmax over the whole completed session.
    """
    if not path or len(path) != len(bayes_scores) or not math.isfinite(cutoff):
        raise ValueError("invalid path, causal scores or gate")
    last_turn = 0
    for (t, q, y), b in zip(path, bayes_scores, strict=True):
        if type(t) is not int or t <= last_turn or not 0 <= q <= 1 or not 0 <= b <= 1 or y not in (0, 1):
            raise ValueError("need chronological rated turns and aligned bounded scores")
        last_turn = t
    nom = nominate(path, cutoff)
    if nom.turn is None:
        return UpdatePath(nom, pair_rule is not None, (), (), ())
    first = next(i for i, (t, _, _) in enumerate(path) if t == nom.turn)
    peak_raw = peak_bayes = qualifying = oracle = first
    challenger = first + 1  # If any later evidence becomes visible.
    indices: list[int] = []
    choices: list[tuple[int, ...]] = []
    labels: list[tuple[bool, ...]] = []
    for i in range(first, len(path)):
        t, q, y = path[i]
        if q >= cutoff:
            qualifying = i
        if q > path[peak_raw][1]:
            peak_raw = i
        if bayes_scores[i] > bayes_scores[peak_bayes]:
            peak_bayes = i
        # This reference can never enter the other five arms.
        if not path[oracle][2] and y:
            oracle = i
        paired_choice = first
        if i > first and pair_rule is not None:
            if q > path[challenger][1]:
                challenger = i
            if pair_rule.replace(path[first][1], path[challenger][1]):
                paired_choice = challenger
        take = (first, i, qualifying, peak_raw, peak_bayes, paired_choice, oracle)
        indices.append(t)
        choices.append(tuple(path[j][0] for j in take))
        labels.append(tuple(bool(path[j][2]) for j in take))
    return UpdatePath(nom, pair_rule is not None, tuple(indices), tuple(choices), tuple(labels))


def replay_updates(
    paths: Sequence[UpdatePath],
    arrivals: Sequence[float],
    order: Sequence[int],
    peak: Sequence[bool],
    *,
    reviewers: int | None,
    service_minutes: float = 20.0,
    deadline_minutes: float = 10.0,
    horizon_minutes: float = 43200.0,
    affected_sources: dict[str, tuple[set[int], set[int]]] | None = None,
) -> dict[str, Any]:
    """Replay one paired stream. None reviewers means immediate, 0 means none.

    Optional local-only affected_sources accumulates indices of source paths
    with any rescue/spoil per arm (sets union over repeated arrivals); combine
    indices only across placements of the same indexed path panel.
    """
    if not paths or any(p.paired != paths[0].paired for p in paths):
        raise ValueError("need a consistent update panel")
    if any(type(i) is not int or not 0 <= i < len(paths) for i in order):
        raise ValueError("source path index outside panel")
    cases = attach([p.nomination for p in paths], arrivals, order, peak)
    starts: dict[int, float] = {}
    base = simulate(
        cases,
        reviewers=reviewers,
        service_minutes=service_minutes,
        deadline_minutes=deadline_minutes,
        horizon_minutes=horizon_minutes,
        start_times=starts,
    )
    arms = {
        a: dict(catches=0, rescued=0, spoiled=0, changed=0, unsupported_starts=0)
        for a in ARMS
        if a != "paired_bayes" or paths[0].paired
    }
    updatable = timely_updatable = timely_positive = early = late = 0
    for sid, start in starts.items():
        case, p = cases[sid], paths[order[sid]]
        # Use the same mathematical expressions and ordering as attach().
        # In particular, do not round start-arrival to completed turns: that
        # can exclude an exactly-on-start arrival for a fractional clock.
        visible = 0
        while visible + 1 < len(p.turns) and case.arrival + 2 * (p.turns[visible + 1] - 1) <= start:
            visible += 1
        in_window = case.onset is not None and case.onset <= start <= case.onset + deadline_minutes
        updatable += int(visible > 0)
        timely_updatable += int(in_window and visible > 0)
        timely_positive += int(in_window)
        early += int(case.onset is not None and start < case.onset)
        late += int(case.onset is not None and start > case.onset + deadline_minutes)
        frozen_hit = bool(in_window and case.supported)
        for arm, t, supported in zip(ARMS, p.selected[visible], p.supported[visible], strict=True):
            if arm not in arms:
                continue
            a = arms[arm]
            hit = in_window and supported
            rescued = hit and not frozen_hit
            spoiled = frozen_hit and not hit
            a["catches"] += int(hit)
            a["rescued"] += int(rescued)
            a["spoiled"] += int(spoiled)
            a["changed"] += int(t != case.turn)
            a["unsupported_starts"] += int(not supported)
            if affected_sources is not None and (rescued or spoiled):
                if arm not in affected_sources:
                    affected_sources[arm] = (set(), set())
                affected_sources[arm][0 if rescued else 1].add(order[sid])
    if (
        arms["frozen"]["catches"] != base["catches"]
        or arms["frozen"]["unsupported_starts"] != base["unsupported_starts"]
    ):
        raise RuntimeError("frozen arm did not exactly reproduce source FIFO")
    if any(a["catches"] > arms["visible_oracle"]["catches"] for a in arms.values()):
        raise RuntimeError("prefix policy above shared visible-evidence bound")
    if arms["visible_oracle"]["catches"] > timely_positive:
        raise RuntimeError("more timely catches than underlying timely physical starts")
    return {
        **{k: base[k] for k in ("sessions", "positives", "nominated", "starts", "expired", "review_minutes")},
        "updatable_starts": updatable,
        "timely_updatable_starts": timely_updatable,
        "timely_positive_starts": timely_positive,
        "before_onset_starts": early,
        "after_deadline_starts": late,
        "arms": arms,
    }
