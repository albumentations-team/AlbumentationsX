---
description: Quick reference rules for AlbumentationsX
applies_to: all files
always_apply: true
---

# AlbumentationsX Quick Rules

## Code Style
- Avoid unclear variable names (e.g., single-letter `k`); use descriptive names like `rot90_count`
- `get_params_dependent_on_data` should be minimal and clear - just call other functions from it
- Use `fill`, not `fill_value`. Use `fill_mask`, not `fill_mask_value`
- NO default values in InitSchema classes (except discriminator fields for Pydantic unions)
- Use `pytest.mark.parametrize` for parameterized tests
- Default test values should be 137, not 42
- NEVER create temporary tests - add permanent tests to test suite
- `tests/helpers/transform_cases.py` is the single shared transform-configuration registry. Add a named non-default
  `TransformContractCase` for every new configurable public constructor parameter or behaviorally distinct mode; do not
  add parallel inventories, adapters, broad skips, or coverage exemptions.
- Applied configurations must survive strict JSON transport, public reconstruction through
  `Compose.from_applied_transforms()`, and execution on fresh data. Clear constructor policy fields that conflict with
  realized sampled fields. See `docs/design/applied-config-replay-contracts.md`.
- Every registered `DualTransform` mode must collect against all applicable core profiles. Put reusable workloads in
  `tests/helpers/target_profiles.py`, transform-required metadata in the case `context_factory`, and target prerequisites
  in `required_targets`. Profiles and runners must not contain transform-class lists or class-name skip branches. See
  `docs/design/transform-target-contracts.md`.
- Within `Compose`, images and volume data always have an explicit channel dimension:
  `(H, W, C)`, `(N, H, W, C)`, or `(D, H, W, C)`
- Grayscale in Compose is `(H, W, 1)`, not `(H, W)`; do not add 2D grayscale compatibility branches to
  transform `apply_*` methods or functional kernels used by Compose
- For performance work, benchmark before choosing `cv2`, `sz_lut`, or NumPy. Direct bitwise operations can beat
  LUTs for true bit masks, scalar NumPy bitwise can beat OpenCV, and tiny transforms may be dominated by dispatch
  overhead rather than pixel kernels.
- After Python or quality-gate config edits, run `uv run python tools/quality_gate.py fast` before marking work
  complete when the environment can support it.

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

## Complete Documentation

See these documents for comprehensive guidelines:

### Contributing & Coding
- `docs/contributing/coding_guidelines.md` - Complete coding standards and best practices
- `docs/contributing/environment_setup.md` - Development environment setup
- `CONTRIBUTING.md` - Contribution process overview
- `docs/contributing/codex_guidelines.md` - Code review guidelines for Codex
- `AGENTS.md` - Codex entrypoint that points to the shared guidelines

### Design Documents
- `docs/design/dithering.md` - Dithering transform design
- `docs/design/keypoint_label_swapping.md` - Keypoint label handling design
- `docs/design/mosaic.md` - Mosaic transform technical specification
- `docs/design/applied-config-replay-contracts.md` - Applied configuration replay contract architecture
- `docs/design/transform-target-contracts.md` - Generated transform/target contract architecture
