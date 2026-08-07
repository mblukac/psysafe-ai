## Summary

<!-- What changed, why, and at which AI-system boundary? -->

Closes #

## Change type

- [ ] Bug fix
- [ ] Feature
- [ ] Breaking change
- [ ] Documentation
- [ ] Refactor or maintenance

## Safety and calibration contract

<!-- Explain “not applicable” where needed. -->

- [ ] Outcomes remain categorical: `matched`, `not_matched`, or `indeterminate`.
- [ ] Calibration uses the named `precise`, `balanced`, and `precautionary` sensitivity boundaries.
- [ ] Detection remains separate from application actions such as `allow`, `review`, or `block`.
- [ ] `indeterminate` cannot silently become `allow`.
- [ ] This change adds no confidence score, numerical rating, clinical decision, prevention claim, or compliance claim.
- [ ] Fail-safe behavior and any agent/tool side-effect limitations are documented.

## Validation

<!-- Summarize tests and aggregate evaluation results. Never paste sensitive cases, prompts, logs, or traces. -->

- [ ] Focused tests cover changed behavior and failure paths.
- [ ] Classifier, policy, prompt, or calibration changes include representative tuning and held-out evaluation.
- [ ] Relevant slices, `indeterminate` behavior, and sensitivity monotonicity were checked.
- [ ] The full test suite, coverage threshold, formatting, linting, and type checks pass.
- [ ] Documentation and migration guidance are updated where public behavior changed.
- [ ] Fixtures and discussion use synthetic, minimized data only.

## Compatibility and risks

<!-- List breaking changes, migration steps, provider-data exposure, tracing implications, and rollback considerations. -->
