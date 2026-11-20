"""Offline scenario invariants; no downloads, tokens, private cases or transcripts."""

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.mindguard import surge_stream as stream
from benchmarks.mindguard.staffing import placement


def test_peak_rank_is_frozen_before_smoothing_and_compression():
    # Hour two wins (4 arrivals); smoothing must not recalculate the winning bin.
    observed = [0, 1, 61, 62, 63, 64, 121]
    raw = stream.make_clock(observed, peak_fraction=0.25)
    scaled = stream.make_clock(observed, peak_fraction=0.25, load=2.0)
    steady = stream.make_clock(observed, peak_fraction=0.25, steady=True)
    assert raw.peak_mask == (False, False, True, True, True, True, False)
    assert scaled.peak_mask == raw.peak_mask
    assert scaled.arrivals == tuple(t / 2 for t in observed)
    assert scaled.summary["effective_bin_minutes"] == 30
    assert steady.peak_mask == (False, False, False, True, True, True, False)
    assert steady.arrivals[0] == raw.arrivals[0]
    assert steady.arrivals[-1] == raw.arrivals[-1]


def test_external_volume_reference_and_ties_ignore_thinned_positions():
    # Two tied reference windows; earlier wins. Selected sample's own density
    # would wrongly place the peak in the second bin without a frozen reference.
    reference = [0, 59, 60, 119]
    clock = stream.make_clock([0, 60, 61, 119], peak_reference=reference)
    assert clock.peak_mask == (True, False, False, False)
    assert clock.summary["peak_windows"] == 1


def test_risk_enrichment_is_a_paired_swap_of_intact_paths():
    positives = [False, False, False, True, False]
    clock = stream.make_clock(list(range(1200)))
    original, control = stream.assign_paths(positives, clock, seed=33)
    changed, stress = stream.assign_paths(positives, clock, seed=33, risk_burst=True)
    assert original == placement(len(positives), len(clock.arrivals), 33)
    assert Counter(original) == Counter(changed)
    assert control["positives"] == stress["positives"] == 240
    assert stress["peak_arrivals"] == 120  # two of twenty equally sized bins
    assert stress["peak_positives"] == 48
    assert stress["peak_positives"] > control["peak_positives"]
    assert stress["peak_positives"] == sum(
        positives[i] and peak for i, peak in zip(changed, clock.peak_mask, strict=True)
    )
    assert (changed, stress) == stream.assign_paths(positives, clock, seed=33, risk_burst=True)


@pytest.mark.parametrize("positives", [[False, False], [True, True], [True, False]])
def test_enrichment_cannot_manufacture_positives_or_negative_slots(positives):
    clock = stream.make_clock([1, 2, 3, 4], peak_fraction=1)
    before, _ = stream.assign_paths(positives, clock, seed=10)
    after, summary = stream.assign_paths(positives, clock, seed=10, risk_burst=True)
    assert after == before
    assert summary["positives_moved_into_peak"] == 0


def test_native_azure_thinning_preserves_original_request_times(monkeypatch):
    curve = (0.0, 0.5, 8.0, 9.0, 9.5, 10.0)
    monkeypatch.setattr(stream, "load_azure_native", lambda _: curve)
    small = stream.azure_episode(Path("ignored"), n=3, seed=1)
    assert small == sorted(set(small))
    assert len(small) == 3
    assert set(small) <= set(curve)
    assert stream.azure_episode(Path("ignored"), n=3, seed=1) == small
    assert stream.azure_episode(Path("ignored"), n=6, seed=1) == list(curve)
    with pytest.raises(ValueError, match="sample size"):
        stream.azure_episode(Path("ignored"), n=7, seed=1)


def test_invalid_clocks_and_corrupted_metadata_rejected(tmp_path):
    with pytest.raises(ValueError, match="ordered"):
        stream.make_clock([2, 1])
    with pytest.raises(ValueError, match="bin width"):
        stream.make_clock([0, 1], load=0)
    with pytest.raises(ValueError, match="fraction"):
        stream.make_clock([0, 1], peak_fraction=0)
    with pytest.raises(ValueError, match="outside"):
        stream.make_clock([0, 1, 3], peak_reference=[0, 2])
    fake = tmp_path / "not-azure.txt"
    fake.write_bytes(b"TIMESTAMP,ContextTokens,GeneratedTokens\r\n")
    with pytest.raises(ValueError, match="pinned"):
        stream.load_azure_native(fake)
    with pytest.raises(ValueError, match="pinned"):
        stream.load_burst_week(fake)


def test_burst_week_preserves_arrival_days_and_simultaneous_requests(monkeypatch):
    requests = (0.1, 0.1, 60.0, 300.0, 8300.0, 10079.0)
    monkeypatch.setattr(stream, "load_burst_week", lambda _: requests)
    all_requests = stream.burst_week(Path("unused"), n=len(requests), seed=7)
    assert all_requests == list(requests)
    sample = stream.burst_week(Path("unused"), n=3, seed=54)
    assert len(sample) == 3
    assert Counter(sample) <= Counter(requests)
    assert sample == sorted(sample)
    assert sample == stream.burst_week(Path("unused"), n=3, seed=54)


def test_verified_burst_week_includes_failures_not_api_or_next_week(tmp_path, monkeypatch):
    header = b"Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
    # The 0-token conversational records are failed submissions. They are
    # still arrivals. Empty synthetic model/token fields are never used.
    data = (
        header
        + b"0,,,,,API log\n"
        + b"30,,,0,,Conversation log\n" * 19_119
        + b"604799,,,,,Conversation log\n"
        + b"604800,,,,,Conversation log\n"
    )
    path = tmp_path / "public-metadata-fixture.csv"
    path.write_bytes(data)
    monkeypatch.setattr(stream, "BURST_SHA256", hashlib.sha256(data).hexdigest())
    result = stream.load_burst_week(path)
    assert len(result) == 19_120
    assert result[0] == 0.5
    assert result[-1] == 604_799 / 60
    path.write_bytes(header)
    with pytest.raises(ValueError, match="pinned"):
        stream.load_burst_week(path)
