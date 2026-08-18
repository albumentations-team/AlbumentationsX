# Contributing to AlbumentationsX

Thank you for your interest in contributing to [AlbumentationsX](https://albumentations.ai/)! This guide will help you get started with contributing to our image augmentation library.

## Quick Start

For small changes (e.g., bug fixes), feel free to submit a PR directly.

For larger changes:

1. Create an [issue](https://github.com/albumentations-team/AlbumentationsX/issues) outlining your proposed change
2. Join our [Discord community](https://discord.gg/e6zHCXTvaN) to discuss your idea

## Contribution Guides

We've organized our contribution guidelines into focused documents:

- [Environment Setup Guide](docs/contributing/environment_setup.md) - How to set up your development environment
- [Coding Guidelines](docs/contributing/coding_guidelines.md) - Code style, best practices, and technical requirements

## Contribution Process

1. **Find an Issue**: Look for open issues or propose a new one. For newcomers, look for issues labeled "good first issue"
2. **Fork & Set Up**: Fork the repository and follow our [Environment Setup Guide](docs/contributing/environment_setup.md)
3. **Create a Branch**: In your fork, create a new branch: `git checkout -b feature/my-new-feature`
4. **Make Changes**: Write code following our [Coding Guidelines](docs/contributing/coding_guidelines.md). Image
   operations must preserve their dtype range; clip the operation itself, never through a forwarding wrapper added
   only to attach a decorator. Keep `apply*` methods as short dispatchers: move image arithmetic, routing, and clipping
   into a functional helper.
5. **Test**: Add tests and ensure all tests pass
6. **Submit**: Open a Pull Request from your fork with a clear description of your changes

## Code Review Process

1. Maintainers will review your contribution
2. Address any feedback or questions
3. Once approved, your code will be merged

## Project Structure

- `albumentations/` - Main source code
- `tests/` - Test suite
- `docs/` - Documentation

## Getting Help

- Join our [Discord community](https://discord.gg/e6zHCXTvaN)
- Open a GitHub [issue](https://github.com/albumentations-team/AlbumentationsX/issues)
- Ask questions in your pull request

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
