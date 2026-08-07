from typing import cast

import pytest

from psysafe.classifiers.pii import MAX_PII_LOCATIONS, PIIAssessment, PIIClassifier, PIIType
from psysafe.core.contracts import Conversation, EvidenceDirectness, Message, Outcome, Sensitivity


def _fullwidth(value: str) -> str:
    return "".join(
        "\u3000" if character == " " else chr(ord(character) + 0xFEE0) if "!" <= character <= "~" else character
        for character in value
    )


def _pii_traceback_locals(error: BaseException) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if "/psysafe/" in traceback.tb_frame.f_code.co_filename:
            values.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(values)


def test_pii_detection_is_local_and_returns_only_types_and_locations() -> None:
    text = "Email robin@example.org, card 4111 1111 1111 1111, SSN 123-45-6789."

    result = PIIClassifier().classify(
        Conversation.from_text(text, message_id="user-turn"),
        sensitivity=Sensitivity.PRECISE,
    )

    assert result.outcome is Outcome.MATCHED
    assert set(result.signals) == {
        PIIType.EMAIL_ADDRESS.value,
        PIIType.PAYMENT_CARD.value,
        PIIType.US_SSN.value,
    }
    assert {location.pii_type for location in result.locations} == {
        PIIType.EMAIL_ADDRESS,
        PIIType.PAYMENT_CARD,
        PIIType.US_SSN,
    }
    assert all(location.message_index == 0 for location in result.locations)
    assert all(0 <= location.start < location.end <= len(text) for location in result.locations)

    serialized = result.model_dump_json()
    assert "robin@example.org" not in serialized
    assert "4111" not in serialized
    assert "123-45-6789" not in serialized
    assert "content" not in type(result).model_fields
    assert "value" not in type(result).model_fields


def test_payment_cards_must_pass_luhn_at_every_sensitivity() -> None:
    result = PIIClassifier().classify(
        Conversation.from_text("Reference 4111 1111 1111 1112 is not a valid card."),
        sensitivity=Sensitivity.PRECAUTIONARY,
    )

    assert result.outcome is Outcome.NOT_MATCHED
    assert result.locations == ()


def test_sensitivity_expands_detection_monotonically() -> None:
    conversation = Conversation.from_text(
        "Write to hello@example.org, call (415) 555-2671, source 10.0.0.8.",
    )
    classifier = PIIClassifier()

    precise = classifier.classify(conversation, sensitivity=Sensitivity.PRECISE)
    balanced = classifier.classify(conversation, sensitivity=Sensitivity.BALANCED)
    precautionary = classifier.classify(conversation, sensitivity=Sensitivity.PRECAUTIONARY)

    assert set(precise.signals) == {PIIType.EMAIL_ADDRESS.value}
    assert set(balanced.signals) == {
        PIIType.EMAIL_ADDRESS.value,
        PIIType.PHONE_NUMBER.value,
    }
    assert set(precautionary.signals) == {
        PIIType.EMAIL_ADDRESS.value,
        PIIType.PHONE_NUMBER.value,
        PIIType.IP_ADDRESS.value,
    }
    assert set(precise.locations) <= set(balanced.locations) <= set(precautionary.locations)
    assert precautionary.evidence_directness is EvidenceDirectness.AMBIGUOUS


def test_explicitly_labeled_phone_and_ip_are_precise_matches() -> None:
    result = PIIClassifier().classify(
        Conversation.from_text("Phone: 4155552671; IP address: 8.8.8.8"),
        sensitivity=Sensitivity.PRECISE,
    )

    assert result.outcome is Outcome.MATCHED
    assert set(result.signals) == {PIIType.PHONE_NUMBER.value, PIIType.IP_ADDRESS.value}


def test_public_ipv6_is_detected_at_balanced_sensitivity() -> None:
    result = PIIClassifier().classify(
        Conversation.from_text("The source was 2001:4860:4860::8888."),
        sensitivity=Sensitivity.BALANCED,
    )

    assert result.outcome is Outcome.MATCHED
    assert result.signals == (PIIType.IP_ADDRESS.value,)
    assert result.evidence_directness is EvidenceDirectness.CONTEXTUAL


def test_precautionary_only_pattern_is_an_ambiguous_match() -> None:
    result = PIIClassifier().classify(
        Conversation.from_text("The callback digits are 4155552671."),
        sensitivity=Sensitivity.PRECAUTIONARY,
    )

    assert result.outcome is Outcome.MATCHED
    assert result.evidence_directness is EvidenceDirectness.AMBIGUOUS
    assert result.signals == (PIIType.PHONE_NUMBER.value,)


async def test_async_pii_classification_has_no_backend_requirement() -> None:
    classifier = PIIClassifier()

    result = await classifier.aclassify(Conversation.from_text("Email help@example.org"))

    assert result.outcome is Outcome.MATCHED
    assert result.signals == (PIIType.EMAIL_ADDRESS.value,)


def test_targeted_pii_uses_only_target_and_preserves_original_message_index() -> None:
    conversation = Conversation(
        messages=(
            Message(role="user", content="old@example.org"),
            Message(role="assistant", content="No identifier here."),
            Message(role="tool", content="new@example.org"),
        ),
    )

    result = PIIClassifier().classify_target(
        conversation,
        target_message_index=2,
        sensitivity=Sensitivity.PRECISE,
    )

    assert result.outcome is Outcome.MATCHED
    assert len(result.locations) == 1
    assert result.locations[0].message_index == 2


def test_local_detector_bounds_locations_for_large_inputs() -> None:
    text = " ".join(f"person{index}@example.org" for index in range(MAX_PII_LOCATIONS + 10))

    result = PIIClassifier().classify(Conversation.from_text(text))

    assert result.outcome is Outcome.MATCHED
    assert len(result.locations) == MAX_PII_LOCATIONS
    assert result.locations_truncated is True


def test_locations_identify_messages_without_copying_message_ids_or_values() -> None:
    conversation = Conversation(
        messages=(
            Message(id="first", role="user", content="No identifiers here."),
            Message(id="private-turn", role="assistant", content="Contact alex@example.net"),
        ),
    )

    result = PIIClassifier().classify(conversation)

    assert len(result.locations) == 1
    assert result.locations[0].message_index == 1
    serialized = result.model_dump_json()
    assert "private-turn" not in serialized
    assert "alex@example.net" not in serialized


def test_ipv4_mapped_ipv6_uses_the_complete_span() -> None:
    text = "::ffff:8.8.8.8"

    result = PIIClassifier().classify(Conversation.from_text(text), sensitivity=Sensitivity.PRECAUTIONARY)

    assert result.outcome is Outcome.MATCHED
    assert len(result.locations) == 1
    assert (result.locations[0].start, result.locations[0].end) == (0, len(text))


def test_location_truncation_preserves_later_signal_types() -> None:
    emails = " ".join(f"person{index}@example.org" for index in range(MAX_PII_LOCATIONS))
    text = f"{emails} card: 4111 1111 1111 1111"

    result = PIIClassifier().classify(Conversation.from_text(text), sensitivity=Sensitivity.PRECISE)

    assert result.locations_truncated is True
    assert len(result.locations) == MAX_PII_LOCATIONS
    assert set(result.signals) == {PIIType.EMAIL_ADDRESS.value, PIIType.PAYMENT_CARD.value}


def test_precise_boundary_rejects_common_card_email_phone_and_unicode_false_positives() -> None:
    text = "IMEI 490154203237518, .foo..bar@example.com, +2024-01-01, and \u0661\u0662\u0663-\u0660\u0660-\u0664\u0665\u0666\u0667"

    result = PIIClassifier().classify(Conversation.from_text(text), sensitivity=Sensitivity.PRECISE)

    assert result.outcome is Outcome.NOT_MATCHED


def test_precise_phone_boundary_rejects_invalid_repeated_and_date_shaped_values() -> None:
    classifier = PIIClassifier()

    for text in ("+0000000000", "+1111111111", "+2024-01-01-01"):
        result = classifier.classify(Conversation.from_text(text), sensitivity=Sensitivity.PRECISE)
        assert result.outcome is Outcome.NOT_MATCHED

    repeated = classifier.classify(
        Conversation.from_text("+1111111111"),
        sensitivity=Sensitivity.PRECAUTIONARY,
    )
    date_shaped = classifier.classify(
        Conversation.from_text("+2024-01-01-01"),
        sensitivity=Sensitivity.PRECAUTIONARY,
    )
    assert repeated.outcome is Outcome.NOT_MATCHED
    assert date_shaped.outcome is Outcome.MATCHED
    assert date_shaped.signals == (PIIType.PHONE_NUMBER.value,)
    assert date_shaped.evidence_directness is EvidenceDirectness.AMBIGUOUS


def test_valid_international_phone_remains_a_precise_match() -> None:
    result = PIIClassifier().classify(
        Conversation.from_text("Call +421 905 123 456"),
        sensitivity=Sensitivity.PRECISE,
    )

    assert result.outcome is Outcome.MATCHED
    assert result.signals == (PIIType.PHONE_NUMBER.value,)


def test_unlabeled_compact_luhn_number_is_precautionary_not_precise() -> None:
    conversation = Conversation.from_text("Reference 490154203237518")
    classifier = PIIClassifier()

    precise = classifier.classify(conversation, sensitivity=Sensitivity.PRECISE)
    precautionary = classifier.classify(conversation, sensitivity=Sensitivity.PRECAUTIONARY)

    assert precise.outcome is Outcome.NOT_MATCHED
    assert precautionary.outcome is Outcome.MATCHED
    assert precautionary.signals == (PIIType.PAYMENT_CARD.value,)


def test_bracketed_labeled_ip_is_precise_and_offsets_are_code_points() -> None:
    text = "🙂 IP: [8.8.8.8]"

    result = PIIClassifier().classify(Conversation.from_text(text), sensitivity=Sensitivity.PRECISE)

    assert result.outcome is Outcome.MATCHED
    assert result.locations[0].start == text.index("8.8.8.8")
    assert result.locations[0].end == result.locations[0].start + len("8.8.8.8")


def test_nfkc_compatible_pii_uses_original_code_point_offsets() -> None:
    ssn = _fullwidth("123-45-6789")
    card = _fullwidth("4111 1111 1111 1111")
    email = _fullwidth("alex@example.org")
    phone = _fullwidth("+421 905 123 456")
    text = f"🙂 SSN: {ssn}; Card: {card}; Email: {email}; Phone: {phone}"

    result = PIIClassifier().classify(Conversation.from_text(text), sensitivity=Sensitivity.PRECISE)

    expected = {
        PIIType.US_SSN: ssn,
        PIIType.PAYMENT_CARD: card,
        PIIType.EMAIL_ADDRESS: email,
        PIIType.PHONE_NUMBER: phone,
    }
    assert result.outcome is Outcome.MATCHED
    assert set(result.signals) == {pii_type.value for pii_type in expected}
    assert {location.pii_type for location in result.locations} == set(expected)
    for location in result.locations:
        assert text[location.start : location.end] == expected[location.pii_type]


def test_nfkc_expansion_maps_a_match_back_to_the_complete_original_span() -> None:
    address = "\ufb00oo" + _fullwidth("@") + "example.org"
    text = f"Email: {address}"

    result = PIIClassifier().classify(Conversation.from_text(text), sensitivity=Sensitivity.PRECISE)

    assert result.outcome is Outcome.MATCHED
    assert len(result.locations) == 1
    location = result.locations[0]
    assert location.pii_type is PIIType.EMAIL_ADDRESS
    assert text[location.start : location.end] == address


def test_overlapping_card_and_phone_shapes_resolve_without_inconsistent_signals() -> None:
    result = PIIClassifier().classify(
        Conversation.from_text("+378282246310005"),
        sensitivity=Sensitivity.PRECAUTIONARY,
    )

    assert result.outcome is Outcome.MATCHED
    assert result.locations_truncated is False
    assert {location.pii_type.value for location in result.locations} == set(result.signals)


def test_broader_sensitivity_preserves_overlapping_precise_locations() -> None:
    conversation = Conversation.from_text("212 555 1234@example.com")
    classifier = PIIClassifier()

    precise = classifier.classify(conversation, sensitivity=Sensitivity.PRECISE)
    balanced = classifier.classify(conversation, sensitivity=Sensitivity.BALANCED)

    assert precise.outcome is Outcome.MATCHED
    assert set(precise.locations) <= set(balanced.locations)
    assert {location.pii_type for location in balanced.locations} == {
        PIIType.EMAIL_ADDRESS,
        PIIType.PHONE_NUMBER,
    }


def test_location_cap_prioritizes_narrower_matches_monotonically() -> None:
    phones = ", ".join(f"212555{index:04d}" for index in range(MAX_PII_LOCATIONS))
    conversation = Conversation.from_text(f"{phones}; final@example.com")
    classifier = PIIClassifier()

    precise = classifier.classify(conversation, sensitivity=Sensitivity.PRECISE)
    precautionary = classifier.classify(conversation, sensitivity=Sensitivity.PRECAUTIONARY)

    assert precautionary.locations_truncated is True
    assert set(precise.locations) <= set(precautionary.locations)
    assert PIIType.EMAIL_ADDRESS.value in precautionary.signals
    assert "redaction-complete" in str(PIIAssessment.model_fields["locations_truncated"].description)


def test_invalid_pii_boundaries_do_not_invoke_subclasses_or_retain_input() -> None:
    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("PRIVATE SENSITIVITY ACCESSOR")

        __hash__ = str.__hash__

    class HostileInt(int):
        def __lt__(self, other: object) -> bool:
            del other
            raise RuntimeError("PRIVATE INDEX ACCESSOR")

    conversation = Conversation.from_text("PRIVATE PII INPUT")
    classifier = PIIClassifier()

    with pytest.raises(ValueError, match="valid Sensitivity") as sensitivity_error:
        classifier.classify(
            conversation,
            sensitivity=cast(Sensitivity, HostileString("balanced")),
        )
    with pytest.raises(ValueError, match="target message") as target_error:
        classifier.classify_target(
            conversation,
            target_message_index=cast(int, HostileInt(0)),
        )

    for error in (sensitivity_error.value, target_error.value):
        locals_text = _pii_traceback_locals(error)
        assert "PRIVATE PII INPUT" not in locals_text
        assert "PRIVATE SENSITIVITY ACCESSOR" not in locals_text
        assert "PRIVATE INDEX ACCESSOR" not in locals_text


@pytest.mark.asyncio
async def test_async_invalid_pii_target_does_not_retain_input() -> None:
    with pytest.raises(ValueError, match="target message") as caught:
        await PIIClassifier().aclassify_target(
            Conversation.from_text("PRIVATE ASYNC PII INPUT"),
            target_message_index=cast(int, True),
        )

    assert "PRIVATE ASYNC PII INPUT" not in _pii_traceback_locals(caught.value)
