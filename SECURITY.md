# Security policy

## Supported versions

PsySafe is alpha software. Security fixes are made on the latest development or released version; older versions are not guaranteed to receive backports.

## Report a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/mblukac/psysafe-ai/security/advisories/new). Do not open a public issue for an unpatched vulnerability.

Include the affected version or commit, impact, minimal reproduction steps, and a suggested mitigation if you have one. Use synthetic data and minimized configuration. Never include API keys, access tokens, raw prompts, private conversations, provider responses, agent traces, or other personal data.

Maintainers will assess the report, coordinate a fix where appropriate, and credit reporters who want attribution. Please allow time for a patch before publishing technical details.

## Security-relevant behavior

Useful reports include:

- secret or sensitive-data exposure;
- unsafe logging, tracing, serialization, or exception behavior;
- dependency, build, or packaging compromise;
- prompt-injection boundary bypasses;
- artifact/version binding or authorization mistakes in integrations; and
- a fail-open path that treats an unavailable or malformed check as clean.

Classifier false positives, false negatives, and policy disagreements are normally safety/evaluation issues rather than software vulnerabilities. Report them through a regular issue using only synthetic or fully anonymized examples.
