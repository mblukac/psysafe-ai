"""Deterministic, monotonic sensitivity calibration."""

from __future__ import annotations

from psysafe.core.contracts import EvidenceDirectness, Sensitivity

_DIRECTNESS_RANK = {
    EvidenceDirectness.NONE: 0,
    EvidenceDirectness.AMBIGUOUS: 1,
    EvidenceDirectness.CONTEXTUAL: 2,
    EvidenceDirectness.EXPLICIT: 3,
}

_MINIMUM_DIRECTNESS = {
    Sensitivity.PRECISE: EvidenceDirectness.EXPLICIT,
    Sensitivity.BALANCED: EvidenceDirectness.CONTEXTUAL,
    Sensitivity.PRECAUTIONARY: EvidenceDirectness.AMBIGUOUS,
}

_ORDERED_FINDING_DIRECTNESS = (
    EvidenceDirectness.AMBIGUOUS,
    EvidenceDirectness.CONTEXTUAL,
    EvidenceDirectness.EXPLICIT,
)


def minimum_directness(sensitivity: Sensitivity) -> EvidenceDirectness:
    """Return the weakest evidence included by a named sensitivity."""

    return _MINIMUM_DIRECTNESS[Sensitivity(sensitivity)]


def matches_sensitivity(
    directness: EvidenceDirectness,
    sensitivity: Sensitivity,
) -> bool:
    """Whether one finding crosses the selected deterministic boundary."""

    normalized_directness = EvidenceDirectness(directness)
    if normalized_directness is EvidenceDirectness.NONE:
        return False
    return _DIRECTNESS_RANK[normalized_directness] >= _DIRECTNESS_RANK[minimum_directness(sensitivity)]


def least_direct(directness_values: tuple[EvidenceDirectness, ...]) -> EvidenceDirectness:
    """Conservatively summarize a non-empty set of selected findings."""

    if not directness_values:
        raise ValueError("at least one directness value is required")
    return min(directness_values, key=_DIRECTNESS_RANK.__getitem__)


def strongest_directness(directness_values: tuple[EvidenceDirectness, ...]) -> EvidenceDirectness:
    """Return the strongest value in a non-empty set of findings."""

    if not directness_values:
        raise ValueError("at least one directness value is required")
    return max(directness_values, key=_DIRECTNESS_RANK.__getitem__)


def sensitivity_boundaries() -> dict[str, tuple[str, ...]]:
    """Export the complete categorical boundary table for other runtimes."""

    return {
        sensitivity.value: tuple(
            directness.value
            for directness in _ORDERED_FINDING_DIRECTNESS
            if matches_sensitivity(directness, sensitivity)
        )
        for sensitivity in Sensitivity
    }


__all__ = [
    "least_direct",
    "matches_sensitivity",
    "minimum_directness",
    "sensitivity_boundaries",
    "strongest_directness",
]
