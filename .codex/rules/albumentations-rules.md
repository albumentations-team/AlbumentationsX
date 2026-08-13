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
- `sample_parameters` should be minimal and clear - just call other functions from it
- Use `fill`, not `fill_value`. Use `fill_mask`, not `fill_mask_value`
- NO default values in InitSchema classes (except discriminator fields for Pydantic unions)
- Prefer reusable `Annotated` validators for standard single-field checks. Use `field_validator` only when validation
  genuinely requires field context or cannot be expressed as reusable type metadata; use `model_validator` for
  cross-field constraints.
- Avoid redundant numeric container unions such as `tuple[int, int] | tuple[float, float]`. Prefer
  `tuple[int | float, int | float]` plus explicit runtime homogeneity validation when integers and floats encode
  different units or behavior.
- Keep public docstrings focused on user-facing behavior. Omit post-initialization, serialization, replay, and other
  implementation details unless they are part of the supported public contract.
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
- For Torch backend work, follow `docs/design/torch-cpu-backend-migration.md`. Extend the existing `Compose`,
  `apply_*`, and functional layers; do not add a second Tensor composition API. CPU Tensor targets use `C,H,W` for
  `image`, `C,L,H,W` for `images`, and `C,D,H,W` for `volume`. The central planner may route NumPy input through
  Torch segments and Tensor input through NumPy/OpenCV/NumKong segments, but only when the full path including every
  bridge, layout conversion, and return conversion has evidence of no repeatable regression. Helpers must not perform
  ad hoc representation conversion. Backend routing preserves the input representation; existing explicit terminal
  `ToTensorV2` and `ToTensor3D` behavior remains unchanged for NumPy input, and Tensor-input pipelines reject those
  unnecessary terminal transforms. Do not add device, CUDA, MPS, stream, or autograd support in the CPU stage. Every
  accepted Tensor route must be no slower than the equivalent NumPy `Compose` in a direct pre-created-input benchmark,
  and must also pass the NumPy-Compose-plus-terminal-conversion versus Tensor-Compose model-ready-output benchmark.
  Run `DataLoader`/collation benchmarks for shared Compose, bridge, planner, batching, and milestone changes; do not
  require them in every isolated transform-family pull request.
- For Torch packaging, dependency-profile, or CI setup work, follow
  `docs/design/torch-dependency-and-ci-greenfield.md`. Package metadata must not select Torch or TorchVision. CI and
  documentation jobs that import AlbumentationsX must request the shared CPU-only runtime profile explicitly; static
  jobs remain Torch-free.
- For CI code shared across AlbumentationsX, Albucore, and albumentations.ai, follow
  `docs/design/ecosystem-ci-foundation-greenfield.md`. Shared actions own bootstrap, CPU-Torch mechanics, and generic
  review orchestration; repository dependency graphs, job policy, releases, legal checks, and deployments stay local.
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
- `docs/design/mosaic.md` - Mosaic transform technical specification
- `docs/design/applied-config-replay-contracts.md` - Applied configuration replay contract architecture
- `docs/design/transform-target-contracts.md` - Generated transform/target contract architecture
