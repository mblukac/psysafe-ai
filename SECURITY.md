# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities through
[GitHub's private vulnerability reporting](https://github.com/mblukac/psysafe-ai/security/advisories/new).
Do not open a public issue for an unpatched vulnerability.

Include the affected version or commit, impact, minimal reproduction steps,
and any suggested mitigation. Use synthetic data and redacted configuration;
never include API keys, access tokens, private conversations, or other personal
data in a report.

Maintainers will assess reports, coordinate a fix where appropriate, and
credit reporters who want attribution. Please allow time for a patch before
publishing technical details.

## Security-relevant behavior

Reports are especially useful when they concern secret exposure, unsafe
logging, dependency or packaging compromise, prompt-injection boundary
bypasses, authorization mistakes in integrations, or a fail-open path that
could incorrectly treat an unavailable classification as a clean result.

Model quality disagreements and classifier false positives or false negatives
are generally safety or evaluation issues rather than software
vulnerabilities. Please report those through a regular GitHub issue using only
synthetic or fully anonymized examples.
