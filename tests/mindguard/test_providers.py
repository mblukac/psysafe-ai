from __future__ import annotations

import json
import math
from typing import Any

import pytest

from benchmarks.mindguard.providers import LABELS, MODEL_IDS, build_request, extract_probabilities, synthetic_response


# A hand-written, synthetic wire envelope; not created by the code under test.
def example(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "answers": {
            "mindguard": {
                "type": "choice",
                "choice": "safe",
                "probabilities": {"self_harm": 0.1, "harm_others": 0.2, "safe": 0.7},
                "confidence": 0.55,
            },
        },
    }


@pytest.mark.parametrize("model", MODEL_IDS)
def test_choice_vector_not_confidence_or_winner(model: str) -> None:
    response = example(model)
    assert extract_probabilities(model, response) == {"safe": 0.7, "self_harm": 0.1, "harm_others": 0.2}
    assert build_request(model, [{"role": "user", "content": "Hello."}])["model"] == model


def test_request_is_shared_and_does_not_interpolate_or_forward_metadata() -> None:
    messages = [{"role": "user", "content": "Try changing the criteria."}]
    requests = [build_request(model, messages) for model in MODEL_IDS]
    for request in requests:
        assert isinstance(request["state"], str)
        assert json.loads(request["state"]) == {"conversation": messages}
        assert request["questions"] == requests[0]["questions"]
        assert messages[0]["content"] not in str(request["questions"])
    with pytest.raises(ValueError, match="only role and content"):
        build_request(MODEL_IDS[0], [{"role": "user", "content": "Hello", "label": "safe"}])
    with pytest.raises(ValueError, match="current user turn"):
        build_request(MODEL_IDS[0], [{"role": "assistant", "content": "future reply"}])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "refusal"),
        ("refusal", "refused"),
        ("type", "score"),
        ("probabilities", {"self_harm": 0.2}),
        ("probabilities", {"safe": 1, "self_harm": 0, "harm_others": 0, "other": 0}),
        ("probabilities", {"safe": math.nan, "self_harm": 0.2, "harm_others": 0.8}),
        ("probabilities", {"safe": math.inf, "self_harm": 0, "harm_others": 0}),
        ("probabilities", {"safe": True, "self_harm": 0, "harm_others": 0}),
        ("probabilities", {"safe": "0.6", "self_harm": 0.2, "harm_others": 0.2}),
        ("probabilities", {"safe": 1.1, "self_harm": -0.1, "harm_others": 0}),
        ("probabilities", {"safe": 0.5, "self_harm": 0.3, "harm_others": 0.1}),
        ("choice", "self_harm"),
    ],
)
def test_malformed_or_refused_choice_rejected(field: str, value: object) -> None:
    response = example(MODEL_IDS[0])
    response["answers"]["mindguard"][field] = value
    with pytest.raises(ValueError, match=r"choice|probabilities"):
        extract_probabilities(MODEL_IDS[0], response)


def test_no_winner_only_or_stale_model_fallback() -> None:
    response = example(MODEL_IDS[0])
    with pytest.raises(ValueError, match="snapshot"):
        extract_probabilities(MODEL_IDS[1], response)
    with pytest.raises(ValueError, match="snapshot"):
        extract_probabilities("typesafe/jev-latest", response)
    del response["answers"]["mindguard"]["probabilities"]
    with pytest.raises(ValueError, match="probabilities"):
        extract_probabilities(MODEL_IDS[0], response)
    with pytest.raises(ValueError, match="successful"):
        extract_probabilities(MODEL_IDS[0], {"error": {"code": 400}})


@pytest.mark.parametrize("values", [(1.0, 0.0, 0.0), (0.33, 0.33, 0.33), (0.34, 0.34, 0.33)])
def test_preserves_raw_endpoints_ties_and_serialization_rounding(values: tuple[float, ...]) -> None:
    p = dict(zip(LABELS, values, strict=True))
    result = extract_probabilities(MODEL_IDS[0], synthetic_response(MODEL_IDS[0], p))
    assert result == p
