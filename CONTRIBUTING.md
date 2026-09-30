# Contributing to AlbumentationsX

For small bug fixes, submit a pull request directly. For larger changes, open an
[issue](https://github.com/albumentations-team/AlbumentationsX/issues) describing the problem and proposed behavior
before implementation. You can discuss it in [Discord](https://discord.gg/e6zHCXTvaN).

## Prepare and submit a change

1. Fork the repository and follow the [Environment Setup Guide](docs/contributing/environment_setup.md).
2. Create a branch in your fork: `git checkout -b feature/my-new-feature`.
3. Follow the [Coding Guidelines](docs/contributing/coding_guidelines.md). Keep transform policy and dispatch in the
   transform class and image operations in the functional layer.
4. Add or update tests that demonstrate the changed behavior, then run the relevant tests and pre-commit hooks.
5. Open a pull request explaining the problem, resulting behavior, and validation. Address review feedback before merge.

Source code is in `albumentations/`, tests in `tests/`, and documentation in `docs/`.
For help, ask in the issue, pull request, or Discord discussion.

## Dependencies, third-party material, and AI assistance

Read the [dependency and contribution license policy](LICENSE_POLICY.md) before
adding a runtime dependency or copying code, data, fonts, binaries, or other
third-party material. State its source, version, license, and required notices
in the pull request. A new dependency or license change also needs its reviewed
record updated in `legal/dependency-licenses.json`.

AI assistance is allowed. Before requesting review, personally read every
change, understand it, and take responsibility for the code, tests,
documentation, and pull-request description. See [AI_USAGE.md](AI_USAGE.md)
for the complete policy. Mentioning AI assistance is encouraged but optional.

## Contributor License Agreement

Before we can accept your contribution, you must accept our
[Contributor License Agreement (CLA) Version 2.0](https://github.com/albumentations-team/AlbumentationsX/blob/main/legal/cla/archive/CLA-v2.0-2026-07-14.md).
It lets
Albumentations, LLC publish accepted contributions under AGPL-3.0-only and
offer the same contributions under separately negotiated commercial terms.
You retain ownership of your work.

CLA acceptance is version-specific. A Version 1 signature does **not** accept
Version 2.0. Contributors recorded only against Version 1 must review and
accept Version 2.0 before another contribution can be merged. The new
acceptance grants rights in qualifying contributions submitted before, on, and
after the Version 2.0 acceptance date; it does not pretend that Version 2.0 was
accepted earlier.

For an individual contribution, complete the CLA Assistant form. We offer
this form as the additional electronic acceptance method permitted by
[Section 11 of Version 2.0](CLA.md#11-agreement-versions-and-acceptance-records).
The [archive manifest](legal/cla/archive/MANIFEST.md#hosted-version-20)
identifies its exact agreement text and acceptance-record requirements.

Use the individual form only if you have reached the age of legal majority
and have legal capacity to accept the agreement. Otherwise, contact
`vladimir@albumentations.ai` for a separate capacity and parent-or-guardian
process appropriate to your circumstances.

1. Open the signing link in the CLA Assistant comment on your pull request or
   in the `license/cla` check.
2. Sign in with the GitHub account associated with your commits.
3. Read Version 2.0, enter your full legal name, and select the required
   individual acceptance checkbox.
4. Submit the form and return to the pull request. CLA Assistant records your
   acceptance and updates the `license/cla` status.

Each committer listed by the bot must complete the applicable signing process
before `license/cla` can pass. The form registers the signature used by this
check. If you have submitted the form but the check is still pending, use the
**recheck** link in the bot's comment.

If an employer or another legal entity owns or controls the contribution, use
the Entity Acceptance process in
[CLA.md](https://github.com/albumentations-team/AlbumentationsX/blob/main/legal/cla/archive/CLA-v2.0-2026-07-14.md)
instead. A corporate signer
must identify the exact legal entity, their authority, and the covered
contributors. Do not use an individual acceptance to license employer-owned
work.

Maintainers verify the applicable Version 2.0 Acceptance Record before merge.
