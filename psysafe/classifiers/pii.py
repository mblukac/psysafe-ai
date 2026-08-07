"""Local, value-free detection of common personally identifiable information.

This classifier intentionally does not use an LLM.  It reports only the kind
of identifier and its location in the supplied conversation; matched text is
never copied into the result.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from psysafe.classifiers.calibration import least_direct
from psysafe.core.contracts import Assessment, Conversation, EvidenceDirectness, Outcome, Sensitivity

MAX_PII_LOCATIONS = 256


class PIIType(str, Enum):
    """Identifier kinds recognized by the local detector."""

    EMAIL_ADDRESS = "email_address"
    PAYMENT_CARD = "payment_card"
    US_SSN = "us_ssn"
    PHONE_NUMBER = "phone_number"
    IP_ADDRESS = "ip_address"


class PIILocation(BaseModel):
    """A value-free location using Python Unicode code-point offsets."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    pii_type: PIIType
    message_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> PIILocation:
        if self.end <= self.start:
            raise ValueError("end must follow start")
        return self


class PIIAssessment(Assessment):
    """An assessment with value-free locations for each local match."""

    locations: tuple[PIILocation, ...] = Field(default_factory=tuple, max_length=MAX_PII_LOCATIONS)
    locations_truncated: bool = False

    @model_validator(mode="after")
    def locations_must_match_outcome(self) -> PIIAssessment:
        if self.outcome is Outcome.MATCHED and not self.locations:
            raise ValueError("matched PII assessments require at least one location")
        if self.outcome is not Outcome.MATCHED and self.locations:
            raise ValueError("only matched PII assessments may include locations")
        if self.outcome is not Outcome.MATCHED and self.locations_truncated:
            raise ValueError("only matched PII assessments may have truncated locations")
        location_types = {location.pii_type.value for location in self.locations}
        if not location_types <= set(self.signals):
            raise ValueError("PII signals must describe every located identifier type")
        if not self.locations_truncated and location_types != set(self.signals):
            raise ValueError("complete PII locations must describe every signal type")
        return self


@dataclass(frozen=True, slots=True)
class _Candidate:
    pii_type: PIIType
    message_index: int
    start: int
    end: int
    minimum_sensitivity: Sensitivity


@dataclass(frozen=True, slots=True)
class _ScanText:
    """Compatibility-normalized text with an exact original-offset map."""

    content: str
    original_indices: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        return self.original_indices[start], self.original_indices[end - 1] + 1


_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+(?![\w.-])",
)
_CARD_PATTERN = re.compile(r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![0-9])")
_SSN_PATTERN = re.compile(r"(?<![0-9])([0-9]{3})-([0-9]{2})-([0-9]{4})(?![0-9])")
_INTERNATIONAL_PHONE_PATTERN = re.compile(r"(?<![\w+])\+[1-9](?:[\s().-]?[0-9]){8,14}(?![0-9])")
_FORMATTED_PHONE_PATTERN = re.compile(
    r"(?<![0-9])(?:\([0-9]{3}\)\s*|[0-9]{3}[ .-])[0-9]{3}[ .-][0-9]{4}(?![0-9])",
)
_COMPACT_PHONE_PATTERN = re.compile(r"(?<![0-9])[0-9]{10}(?![0-9])")
_IPV4_PATTERN = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])")
_IPV6_WITH_IPV4_PATTERN = re.compile(
    r"(?<![0-9A-Fa-f:.])(?:[0-9A-Fa-f]{0,4}:){2,7}(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9A-Fa-f:.])",
)
_IPV6_PATTERN = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])")
_DATE_SHAPED_INTERNATIONAL_PHONE_PATTERN = re.compile(
    r"^\+(?:19|20)[0-9]{2}[- .](?:0[1-9]|1[0-2])[- .](?:0[1-9]|[12][0-9]|3[01])(?:[- .][0-9]{1,4})?$",
)

_SENSITIVITY_RANK = {
    Sensitivity.PRECISE: 0,
    Sensitivity.BALANCED: 1,
    Sensitivity.PRECAUTIONARY: 2,
}
_TYPE_PRIORITY = {
    PIIType.EMAIL_ADDRESS: 0,
    PIIType.PAYMENT_CARD: 1,
    PIIType.US_SSN: 2,
    PIIType.PHONE_NUMBER: 3,
    PIIType.IP_ADDRESS: 4,
}
_DIRECTNESS_BY_MINIMUM_SENSITIVITY = {
    Sensitivity.PRECISE: EvidenceDirectness.EXPLICIT,
    Sensitivity.BALANCED: EvidenceDirectness.CONTEXTUAL,
    Sensitivity.PRECAUTIONARY: EvidenceDirectness.AMBIGUOUS,
}


def _passes_luhn(value: str) -> bool:
    """Validate a card-shaped digit sequence without retaining it."""

    if not 13 <= len(value) <= 19 or len(set(value)) == 1:
        return False
    total = 0
    parity = len(value) % 2
    for index, character in enumerate(value):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _normalized_scan_text(content: str) -> _ScanText:
    """Apply per-code-point NFKC while preserving original code-point spans.

    Per-code-point normalization captures compatibility forms used to obscure
    identifiers without losing the one-to-many mapping required to report
    locations against the caller's original string.
    """

    parts: list[str] = []
    original_indices: list[int] = []
    for original_index, character in enumerate(content):
        normalized = unicodedata.normalize("NFKC", character)
        parts.append(normalized)
        original_indices.extend((original_index,) * len(normalized))
    return _ScanText(content="".join(parts), original_indices=tuple(original_indices))


def _ascii_digits(value: str) -> str:
    return "".join(character for character in value if "0" <= character <= "9")


def _phone_digits_are_plausible(digits: str) -> bool:
    return bool(digits) and len(set(digits)) > 1


def _international_phone_minimum_sensitivity(
    value: str,
    content: str,
    start: int,
) -> Sensitivity | None:
    digits = _ascii_digits(value)
    if not 10 <= len(digits) <= 15 or digits[0] == "0" or not _phone_digits_are_plausible(digits):
        return None
    if _DATE_SHAPED_INTERNATIONAL_PHONE_PATTERN.fullmatch(value):
        if _looks_explicitly_labeled(content, start, ("phone", "mobile", "tel", "telephone")):
            return Sensitivity.BALANCED
        return Sensitivity.PRECAUTIONARY
    return Sensitivity.PRECISE


def _valid_ssn(area: str, group: str, serial: str) -> bool:
    area_number = int(area)
    return area_number not in {0, 666} and area_number < 900 and group != "00" and serial != "0000"


def _looks_explicitly_labeled(content: str, start: int, labels: tuple[str, ...]) -> bool:
    prefix = content[max(0, start - 24) : start].lower()
    return any(re.search(rf"\b{re.escape(label)}\s*(?:is|=|:)?\s*[\[(]?\s*$", prefix) for label in labels)


def _valid_email_candidate(value: str) -> bool:
    local_part, _, _ = value.partition("@")
    return not local_part.startswith(".") and not local_part.endswith(".") and ".." not in local_part


def _ip_minimum_sensitivity(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    content: str,
    start: int,
) -> Sensitivity:
    if _looks_explicitly_labeled(content, start, ("ip", "ip address")):
        return Sensitivity.PRECISE
    if address.is_global:
        return Sensitivity.BALANCED
    return Sensitivity.PRECAUTIONARY


class PIIClassifier:
    """Detect common PII locally with a monotonic named sensitivity boundary.

    ``precise`` recognizes high-specificity formats. ``balanced`` also
    recognizes common formatted phone numbers and public IP addresses.
    ``precautionary`` additionally recognizes compact phone numbers and
    non-public IP addresses. Payment-card candidates must pass Luhn validation
    at every sensitivity.
    """

    classifier_id = "pii"
    policy_version = "2026.08.1"

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> PIIAssessment:
        normalized_sensitivity = Sensitivity(sensitivity)
        candidates: list[_Candidate] = []
        for message_index, message in enumerate(conversation.messages):
            candidates.extend(self._candidates(message.content, message_index))

        allowed_rank = _SENSITIVITY_RANK[normalized_sensitivity]
        eligible = [
            candidate for candidate in candidates if _SENSITIVITY_RANK[candidate.minimum_sensitivity] <= allowed_rank
        ]
        eligible.sort(
            key=lambda candidate: (
                candidate.message_index,
                candidate.start,
                _TYPE_PRIORITY[candidate.pii_type],
                -(candidate.end - candidate.start),
            ),
        )

        accepted_all: list[_Candidate] = []
        for candidate in eligible:
            previous = accepted_all[-1] if accepted_all else None
            if (
                previous is not None
                and candidate.message_index == previous.message_index
                and candidate.start < previous.end
            ):
                continue
            accepted_all.append(candidate)

        signal_values = tuple(dict.fromkeys(candidate.pii_type.value for candidate in accepted_all))
        accepted = accepted_all[:MAX_PII_LOCATIONS]
        locations_truncated = len(accepted_all) > MAX_PII_LOCATIONS

        if not accepted:
            return PIIAssessment(
                classifier_id=self.classifier_id,
                policy_version=self.policy_version,
                sensitivity=normalized_sensitivity,
                outcome=Outcome.NOT_MATCHED,
            )

        evidence_directness = least_direct(
            tuple(_DIRECTNESS_BY_MINIMUM_SENSITIVITY[candidate.minimum_sensitivity] for candidate in accepted_all),
        )
        locations = tuple(
            PIILocation(
                pii_type=candidate.pii_type,
                message_index=candidate.message_index,
                start=candidate.start,
                end=candidate.end,
            )
            for candidate in accepted
        )
        return PIIAssessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=normalized_sensitivity,
            outcome=Outcome.MATCHED,
            evidence_directness=evidence_directness,
            signals=signal_values,
            locations=locations,
            locations_truncated=locations_truncated,
        )

    async def aclassify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> PIIAssessment:
        """Provide async parity without introducing I/O or an executor."""

        return self.classify(conversation, sensitivity=sensitivity)

    @staticmethod
    def _candidates(content: str, message_index: int) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        scan_text = _normalized_scan_text(content)
        normalized_content = scan_text.content

        for match in _EMAIL_PATTERN.finditer(normalized_content):
            if not _valid_email_candidate(match.group()):
                continue
            start, end = scan_text.original_span(match.start(), match.end())
            candidates.append(
                _Candidate(
                    PIIType.EMAIL_ADDRESS,
                    message_index,
                    start,
                    end,
                    Sensitivity.PRECISE,
                ),
            )

        for match in _CARD_PATTERN.finditer(normalized_content):
            digits = re.sub(r"[^0-9]", "", match.group())
            if _passes_luhn(digits):
                labeled = _looks_explicitly_labeled(
                    normalized_content,
                    match.start(),
                    ("card", "card number", "credit card", "debit card", "payment card"),
                )
                separated = " " in match.group() or "-" in match.group()
                minimum = (
                    Sensitivity.PRECISE if labeled else Sensitivity.BALANCED if separated else Sensitivity.PRECAUTIONARY
                )
                start, end = scan_text.original_span(match.start(), match.end())
                candidates.append(
                    _Candidate(
                        PIIType.PAYMENT_CARD,
                        message_index,
                        start,
                        end,
                        minimum,
                    ),
                )

        for match in _SSN_PATTERN.finditer(normalized_content):
            if _valid_ssn(*match.groups()):
                start, end = scan_text.original_span(match.start(), match.end())
                candidates.append(
                    _Candidate(
                        PIIType.US_SSN,
                        message_index,
                        start,
                        end,
                        Sensitivity.PRECISE,
                    ),
                )

        for match in _INTERNATIONAL_PHONE_PATTERN.finditer(normalized_content):
            international_minimum = _international_phone_minimum_sensitivity(
                match.group(),
                normalized_content,
                match.start(),
            )
            if international_minimum is None:
                continue
            start, end = scan_text.original_span(match.start(), match.end())
            candidates.append(
                _Candidate(
                    PIIType.PHONE_NUMBER,
                    message_index,
                    start,
                    end,
                    international_minimum,
                ),
            )

        for match in _FORMATTED_PHONE_PATTERN.finditer(normalized_content):
            if not _phone_digits_are_plausible(_ascii_digits(match.group())):
                continue
            minimum = (
                Sensitivity.PRECISE
                if _looks_explicitly_labeled(
                    normalized_content,
                    match.start(),
                    ("phone", "mobile", "tel", "telephone"),
                )
                else Sensitivity.BALANCED
            )
            start, end = scan_text.original_span(match.start(), match.end())
            candidates.append(
                _Candidate(PIIType.PHONE_NUMBER, message_index, start, end, minimum),
            )

        for match in _COMPACT_PHONE_PATTERN.finditer(normalized_content):
            if not _phone_digits_are_plausible(match.group()):
                continue
            minimum = (
                Sensitivity.PRECISE
                if _looks_explicitly_labeled(
                    normalized_content,
                    match.start(),
                    ("phone", "mobile", "tel", "telephone"),
                )
                else Sensitivity.PRECAUTIONARY
            )
            start, end = scan_text.original_span(match.start(), match.end())
            candidates.append(
                _Candidate(PIIType.PHONE_NUMBER, message_index, start, end, minimum),
            )

        for match in _IPV4_PATTERN.finditer(normalized_content):
            try:
                address = ipaddress.IPv4Address(match.group())
            except ipaddress.AddressValueError:
                continue
            minimum = _ip_minimum_sensitivity(address, normalized_content, match.start())
            start, end = scan_text.original_span(match.start(), match.end())
            candidates.append(_Candidate(PIIType.IP_ADDRESS, message_index, start, end, minimum))

        for pattern in (_IPV6_WITH_IPV4_PATTERN, _IPV6_PATTERN):
            for match in pattern.finditer(normalized_content):
                try:
                    address_v6 = ipaddress.IPv6Address(match.group())
                except ipaddress.AddressValueError:
                    continue
                minimum = _ip_minimum_sensitivity(address_v6, normalized_content, match.start())
                start, end = scan_text.original_span(match.start(), match.end())
                candidates.append(_Candidate(PIIType.IP_ADDRESS, message_index, start, end, minimum))

        return candidates


__all__ = ["MAX_PII_LOCATIONS", "PIIAssessment", "PIIClassifier", "PIILocation", "PIIType"]
