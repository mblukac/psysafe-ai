"""Immutable mapping from categorical classifier results to gate actions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from psysafe.core.contracts import Outcome
from psysafe.gates.contracts import MAX_GATE_CLASSIFIERS, GateAction


class ClassifierPolicyOverride(BaseModel):
    """Optional action overrides for one configured classifier."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    classifier_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.-]*$")
    matched_action: GateAction | None = None
    indeterminate_action: GateAction | None = None
    review_signal_action: GateAction | None = None

    @field_validator("indeterminate_action", "review_signal_action")
    @classmethod
    def fail_safe_actions_must_not_allow(cls, value: GateAction | None) -> GateAction | None:
        if value is GateAction.ALLOW:
            raise ValueError("indeterminate outcomes and review signals cannot allow a gate")
        return value

    @model_validator(mode="after")
    def at_least_one_action_must_be_overridden(self) -> ClassifierPolicyOverride:
        if self.matched_action is None and self.indeterminate_action is None and self.review_signal_action is None:
            raise ValueError("a classifier policy override must set at least one action")
        return self


class GatePolicy(BaseModel):
    """Categorical action policy shared by every classifier in one gate.

    A non-match allows by default. The action for a match is explicit, while
    indeterminate results and review signals can never be configured to allow.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    matched_action: GateAction
    indeterminate_action: GateAction = GateAction.REVIEW
    review_signal_action: GateAction = GateAction.REVIEW
    overrides: tuple[ClassifierPolicyOverride, ...] = Field(
        default_factory=tuple,
        max_length=MAX_GATE_CLASSIFIERS,
    )

    @field_validator("indeterminate_action", "review_signal_action")
    @classmethod
    def fail_safe_actions_must_not_allow(cls, value: GateAction) -> GateAction:
        if value is GateAction.ALLOW:
            raise ValueError("indeterminate outcomes and review signals cannot allow a gate")
        return value

    @field_validator("overrides")
    @classmethod
    def override_classifier_ids_must_be_unique(
        cls,
        values: tuple[ClassifierPolicyOverride, ...],
    ) -> tuple[ClassifierPolicyOverride, ...]:
        classifier_ids = tuple(value.classifier_id for value in values)
        if len(set(classifier_ids)) != len(classifier_ids):
            raise ValueError("classifier policy overrides must have unique classifier IDs")
        return values

    @property
    def override_classifier_ids(self) -> tuple[str, ...]:
        """Classifier IDs with explicit action overrides."""

        return tuple(override.classifier_id for override in self.overrides)

    def _override_for(self, classifier_id: str | None) -> ClassifierPolicyOverride | None:
        if classifier_id is None:
            return None
        return next(
            (override for override in self.overrides if override.classifier_id == classifier_id),
            None,
        )

    def action_for(self, outcome: Outcome, *, classifier_id: str | None = None) -> GateAction:
        """Return the configured action for one categorical outcome."""

        normalized = Outcome(outcome)
        override = self._override_for(classifier_id)
        if normalized is Outcome.MATCHED:
            if override is not None and override.matched_action is not None:
                return override.matched_action
            return self.matched_action
        if normalized is Outcome.INDETERMINATE:
            if override is not None and override.indeterminate_action is not None:
                return override.indeterminate_action
            return self.indeterminate_action
        return GateAction.ALLOW

    def review_action_for(self, classifier_id: str) -> GateAction:
        """Return the non-ALLOW action for a classifier's review signal."""

        override = self._override_for(classifier_id)
        if override is not None and override.review_signal_action is not None:
            return override.review_signal_action
        return self.review_signal_action


__all__ = ["ClassifierPolicyOverride", "GatePolicy"]
