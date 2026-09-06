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

## Contributor License Agreement

Before we can accept your contribution, you must accept our
[Contributor License Agreement (CLA) Version 2.0](CLA.md). It lets
Albumentations, LLC publish accepted contributions under AGPL-3.0-only and
offer the same contributions under separately negotiated commercial terms.
You retain ownership of your work.

CLA acceptance is version-specific. A Version 1 signature does **not** accept
Version 2.0. Contributors recorded only against Version 1 must review and
accept Version 2.0 before another contribution can be merged. The new
acceptance grants rights in qualifying contributions submitted before, on, and
after the Version 2.0 acceptance date; it does not pretend that Version 2.0 was
accepted earlier.

For an individual contribution, comment on the pull request with this exact
Version 2.0 statement:

```text
I have read and agree to the AlbumentationsX CLA Version 2.0 (July 14, 2026) as an individual.
```

If an employer or another legal entity owns or controls the contribution, use
the Entity Acceptance process in [CLA.md](CLA.md) instead. A corporate signer
must identify the exact legal entity, their authority, and the covered
contributors. Do not use an individual acceptance to license employer-owned
work.

Maintainers verify the applicable Version 2.0 Acceptance Record before merge.
