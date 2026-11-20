"""Public arrival clocks and paired placements for the offline surge experiment.

Clock time is in minutes. A source path is assigned as a whole: no mixing of
turns, model scores, or clinician votes. The synthetic risk-mix arm may use a
path's outcome to *construct* a stream, never to induce a decision or queue rank.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from io import BytesIO, TextIOWrapper
from itertools import pairwise
from pathlib import Path

from .staffing import placement

AZURE_REVISION = "790921015d50dd6aae7f7e47f39ba0e235ad6b08"
AZURE_URL = (
    f"https://raw.githubusercontent.com/Azure/AzurePublicDataset/{AZURE_REVISION}/data/AzureLLMInferenceTrace_conv.csv"
)
AZURE_SHA256 = "2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8"
BURST_REVISION = "5d93d820d9b6bee4e26914df003e5713538164ae"  # v1.1 git-LFS pointer
BURST_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv"
BURST_SHA256 = "4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122"
_WEEK_SECONDS = 7 * 24 * 60 * 60


def _validate_clock(arrivals: Sequence[float]) -> None:
    if len(arrivals) < 2 or any(not math.isfinite(t) or t < 0 for t in arrivals):
        raise ValueError("need two or more nonnegative finite arrival times")
    if arrivals[0] >= arrivals[-1] or any(a > b for a, b in pairwise(arrivals)):
        raise ValueError("arrival times must be ordered and span a positive duration")


@lru_cache(maxsize=2)
def _azure_offsets(raw: bytes) -> tuple[float, ...]:
    """Use invocation times only from the checksummed Splitwise conversation file.

    The actual 19,366 timestamp rows span 58.36 minutes on 2023-11-16.
    These are request arrivals, not measured starts of crisis sessions or a
    representative day. Verify bytes on every load, reuse only parsed valid input.
    The file contains only timestamps and token counts; token counts are ignored.
    """
    if hashlib.sha256(raw).hexdigest() != AZURE_SHA256:
        raise ValueError("Azure arrival trace differs from the pinned public file")
    records = csv.DictReader(raw.decode("ascii").splitlines())
    if records.fieldnames != ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]:
        raise ValueError("unexpected Azure timestamp schema")
    # Seven fractional digits in this release, last digit zero. Trim to Python
    # microseconds explicitly for Python 3.10 ISO parser compatibility.
    timestamps = [datetime.fromisoformat(row["TIMESTAMP"][:26]) for row in records]
    if len(timestamps) != 19_366:
        raise ValueError("incomplete Azure invocation trace")
    source = tuple((stamp - timestamps[0]).total_seconds() / 60 for stamp in timestamps)
    _validate_clock(source)
    return source


def load_azure_native(path: Path) -> tuple[float, ...]:
    """All native elapsed request times (minutes) for source-defined peak bins."""
    return _azure_offsets(path.read_bytes())


def _sample_episode(source: Sequence[float], *, n: int, seed: int) -> list[float]:
    if type(n) is not int or not 2 <= n <= len(source):
        raise ValueError("sample size must be between two and the trace length")
    rng = random.Random(seed)  # noqa: S311 - reproducible temporal thinning
    return [source[i] for i in sorted(rng.sample(range(len(source)), n))]


def azure_episode(path: Path, *, n: int, seed: int) -> list[float]:
    """Uniformly select n different request rows; retain actual times from original start.

    A scenario with 60 monitored requests, for example, retains 60 of 19,366
    invocations within the one real 58.36-minute episode. Monitoring probability
    is an assumption, not an observed rate. Never stretch, repeat, or select by risk.
    """
    return _sample_episode(load_azure_native(path), n=n, seed=seed)


@lru_cache(maxsize=2)
def _verified_burst_week(path: Path, _identity: tuple[int, ...]) -> tuple[float, ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BURST_SHA256:
        raise ValueError("BurstGPT trace differs from the pinned v1.1 public file")
    times: list[float] = []
    with TextIOWrapper(BytesIO(raw), encoding="ascii", newline="") as source:
        rows = csv.DictReader(source)
        if rows.fieldnames != ["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"]:
            raise ValueError("unexpected BurstGPT schema")
        for row in rows:
            timestamp = int(row["Timestamp"])
            # Verified pinned file is chronological across all 1,429,737
            # records. Do not condition on whether a response was generated.
            if timestamp >= _WEEK_SECONDS:
                break
            if row["Log Type"] == "Conversation log":
                times.append(timestamp / 60)
    if len(times) != 19_120:
        raise ValueError("unexpected first-week conversational-mode coverage")
    _validate_clock(times)
    return tuple(times)


def load_burst_week(path: Path) -> tuple[float, ...]:
    """All first-week conversational-mode request submission times, in native minutes.

    BurstGPT counts *requests*, including failures, not distinct chats or people;
    dated by seconds since midnight on the first day in the authors' local time.
    The 52 MB v1.1 file contains no request text or user/session identifiers.
    Cache only the parsed weekly timestamps; a file change invalidates verification.
    """
    path = path.resolve()
    st = path.stat()
    identity = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    return _verified_burst_week(path, identity)


def burst_week(path: Path, *, n: int, seed: int) -> list[float]:
    """Uniformly thin across the full week, never within daily/hourly quotas."""
    return _sample_episode(load_burst_week(path), n=n, seed=seed)


@dataclass(frozen=True)
class Clock:
    arrivals: tuple[float, ...]
    peak_mask: tuple[bool, ...]
    summary: dict[str, float | int | bool]


def make_clock(
    arrivals: Sequence[float],
    *,
    steady: bool = False,
    load: float = 1.0,
    peak_fraction: float = 0.10,
    bin_minutes: float = 60.0,
    peak_reference: Sequence[float] | None = None,
) -> Clock:
    """Top-volume windows are determined from the untreated reference *clock*.

    Non-overlapping bins start at zero, include empty bins and the final bin,
    and are left-closed/right-open; earliest breaks equal counts. Select the
    top ceil(fraction * bin count). Steady spaces the same number evenly between
    the same endpoints. Load doubles the rate at 2x by compressing all timestamps
    *and windows* by two; it never copies data or speeds up service/review turns.
    Thus the peak regions are paired over the entire shape-by-load comparison.
    Supply the *unthinned* peak_reference for a thinned trace to freeze bins
    across replications. These peak bins are for scoring only, never features.
    """
    _validate_clock(arrivals)
    if not math.isfinite(load) or load <= 0 or not math.isfinite(bin_minutes) or bin_minutes <= 0:
        raise ValueError("load and bin width must be positive and finite")
    if not 0 < peak_fraction <= 1:
        raise ValueError("peak fraction must be in (0,1]")
    ref = arrivals if peak_reference is None else peak_reference
    if peak_reference is not None:
        _validate_clock(ref)
        if arrivals[-1] > ref[-1]:
            raise ValueError("arrival outside the frozen peak-reference horizon")
    n_bins = math.floor(ref[-1] / bin_minutes) + 1
    counts = [0] * n_bins
    for t in ref:
        counts[math.floor(t / bin_minutes)] += 1
    n_peak = math.ceil(n_bins * peak_fraction)
    peak = set(sorted(range(n_bins), key=lambda j: (-counts[j], j))[:n_peak])
    if steady:
        gap = (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1)
        base: Sequence[float] = [
            arrivals[0],
            *(arrivals[0] + i * gap for i in range(1, len(arrivals) - 1)),
            arrivals[-1],
        ]
    else:
        base = arrivals
    mask = tuple(math.floor(t / bin_minutes) in peak for t in base)
    clock = tuple(t / load for t in base)
    return Clock(
        clock,
        mask,
        {
            "arrivals": len(base),
            "load": load,
            "steady": steady,
            "first_arrival_minutes": clock[0],
            "last_arrival_minutes": clock[-1],
            "source_bin_minutes": bin_minutes,
            "effective_bin_minutes": bin_minutes / load,
            "peak_window_fraction_setting": peak_fraction,
            "windows": n_bins,
            "peak_windows": n_peak,
            "peak_arrivals": sum(mask),
            "peak_arrival_share": sum(mask) / len(mask),
        },
    )


def assign_paths(
    positive_paths: Sequence[bool], clock: Clock, *, seed: int, risk_burst: bool = False
) -> tuple[list[int], dict[str, int | bool]]:
    """Balance and shuffle sources; optionally enrich the source-defined peak.

    Desired peak count is ceil(twice global positive prevalence * peak slots),
    bounded by physically possible counts. Swap randomly chosen positive
    off-peak and negative peak arrivals; never reduce a naturally higher peak
    count. Each source path's multiplicity (hence total positives and turns)
    is *exactly* identical to the paired untreated placement, not just in expectation.
    """
    if not positive_paths or any(type(p) is not bool for p in positive_paths):
        raise ValueError("need one explicit binary positive flag per source path")
    if not clock.arrivals or len(clock.peak_mask) != len(clock.arrivals):
        raise ValueError("unaligned peak mask and arrival times")
    indices = placement(len(positive_paths), len(clock.arrivals), seed)
    eligible = [positive_paths[i] for i in indices]
    n_positive = sum(eligible)
    peak_count = sum(clock.peak_mask)
    before = sum(p and peak for p, peak in zip(eligible, clock.peak_mask, strict=True))
    target = min(n_positive, peak_count, math.ceil(2 * n_positive * peak_count / len(indices)))
    swapped = max(0, target - before) if risk_burst else 0
    if swapped:
        rng = random.Random(f"risk-surge:{seed}")  # noqa: S311 - experimental permutation
        inside = [i for i, (p, peak) in enumerate(zip(eligible, clock.peak_mask, strict=True)) if peak and not p]
        outside = [i for i, (p, peak) in enumerate(zip(eligible, clock.peak_mask, strict=True)) if p and not peak]
        for a, b in zip(rng.sample(inside, swapped), rng.sample(outside, swapped), strict=True):
            indices[a], indices[b] = indices[b], indices[a]
    return indices, {
        "risk_burst": risk_burst,
        "positives": n_positive,
        "peak_arrivals": peak_count,
        "peak_positives_before": before,
        "peak_positives": before + swapped,
        "positives_moved_into_peak": swapped,
    }
