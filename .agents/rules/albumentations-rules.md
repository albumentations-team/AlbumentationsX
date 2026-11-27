---
description: Quick reference rules for AlbumentationsX
applies_to: all files
always_apply: true
---

# AlbumentationsX Quick Rules

## Automation Boundary
- Before creating or expanding a skill, classify the rule first: deterministic source-local invariants belong in a focused
  pre-commit hook or quality-gate check, with tests.
- Keep skills for architecture, trade-offs, performance decisions, workflow, and reviewer judgment that tooling cannot
  prove reliably.
- Do not create a skill merely to repeat an automated check; make the hook diagnostic the source of the mechanical rule.

## Code Style
- Avoid unclear variable names (e.g., single-letter `k`); use descriptive names like `rot90_count`
- Prefer reusable `Annotated` validators for standard single-field checks. Use `field_validator` only when validation
  genuinely requires field context or cannot be expressed as reusable type metadata; use `model_validator` for
  cross-field constraints.
- Avoid redundant numeric container unions such as `tuple[int, int] | tuple[float, float]`. Prefer
  `tuple[int | float, int | float]` plus explicit runtime homogeneity validation when integers and floats encode
  different units or behavior.
- Keep public docstrings focused on user-facing behavior. Omit post-initialization, serialization, replay, and other
  implementation details unless they are part of the supported public contract.
- Run `pre-commit run check-ax-rules --all-files` for AX repository checks. Its `coding-guidance` diagnostics and
  `docs/contributing/coding_guidelines.md` are the mechanical source of truth for deterministic source contracts.
- Use `pytest.mark.parametrize` for parameterized tests
- Default test values should be 137, not 42
- NEVER create temporary tests - add permanent tests to test suite
- For transform configuration, replay, target coverage, runtime performance, Torch, CI, or licensing work, use the
  matching project skill and its linked design document. These rules do not restate their contracts.
- For Python, pre-commit, or CI edits, use `validate-and-fix` to select and run the focused validation.

## Channels
- Inside `Compose` every image has a channel axis. NumPy images are channel-last and grayscale is `(H, W, 1)`;
  native Tensor handlers are channel-first and grayscale is `(1, H, W)`. Do not add channel-free compatibility branches or
  hedge for "maybe grayscale" in `apply_*` methods or in functional kernels that `Compose` dispatches to. Functional helpers
  that are public for direct calls keep their documented `(H, W)` support in their owning functional layer.
  Direct calls to a leaf transform bypass `Compose` normalization and must pass the canonical layout with an explicit
  channel axis; transforms do not add `(H, W)` support for them.
- Every channel is image data. Do not treat a fourth channel as alpha or a mask, and do not exempt it from a
  transform. Masks and labels go through `mask`, `masks`, and the other targets, which image transforms do not touch.
- A transform that handles channels differently (for example RGB-only color math) declares that restriction on the
  transform and documents it in its public docstring.

## License and CLA Integrity
- Use SPDX `AGPL-3.0-only` consistently; do not silently change it to an
  `-or-later` expression.
- Keep `LICENSE` as the complete canonical AGPL text. Preserve the repository
  expression and history in `LICENSING.md`, the legacy Albumentations 2.0.8
  MIT notice, and immutable CLA archives.
- Treat CLA acceptance as version-specific. Never infer acceptance of a new CLA
  from an old signature.
- Run `python tools/verify_legal_integrity.py`; after packaging changes, build
  both distributions and pass them with `--artifacts`.
- Read `docs/maintaining/license-provenance.md` and use the
  `license-integrity` skill for any license, CLA, notice, packaging-license, or
  commercial-license wording change.
