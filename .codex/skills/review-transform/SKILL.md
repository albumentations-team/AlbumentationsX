---
name: review-transform
description: Review an AlbumentationsX transform for correctness, public API coherence, performance, documentation, and test coverage.
---

# Review a transform

Start with `pre-commit run check-ax-rules --all-files`. Report failing diagnostics with their rule and location.
The hook and [Coding Guidelines](../../../docs/contributing/coding_guidelines.md) own deterministic source contracts.
Then review the behavior the hook cannot prove.

## Correctness and API

- Check the mathematical operation, coordinate boundaries, dtype range, and annotation semantics.
- Verify that constructor fields, validators, and defaults express a coherent public policy.
- Check empty inputs, ownership, aliasing, and unsupported input handling through the public route.
- Distinguish public grayscale inputs from internal execution: Compose normalizes NumPy inputs to explicit channels
  before dispatch. Use [NumPy and Tensor routing](../../../docs/design/numpy-tensor-routing.md) for boundary changes.
- Take bbox type from `BboxParams.bbox_type`, never column count. Convert OpenCV rotated rectangles through
  `cv2.boxPoints` and `polygons_to_obb`.
- Remove unreachable code and unused helpers after checking external callers and inherited dispatch.

## Sampling and replay

Use [Applied Configuration Replay Contracts](../../../docs/design/applied-config-replay-contracts.md) when reviewing
constructor serialization, applied configuration, or `ReplayCompose`.

- Identify what is sampled per invocation and which constructor fields remain policy.
- Check target prerequisites, deterministic sampling, and representation-dependent parameters.
- Verify that applied configuration records realized constructor values and clears conflicting source policy fields.
- Claim exact replay only when the recorded configuration captures every output-changing random value.
- Reconstruct through the public API and execute on fresh equivalent data without mutating caller-owned inputs.

## Performance

Read [Performance Optimization](../performance-optimization/SKILL.md) and its required reference before inspecting
runtime code. Tie each proposed optimization to an affected route and a measurement. Consider removable work,
existing Albucore primitives, allocation and conversion costs, and setup that can be shared across a batch.

## Documentation and tests

Use [Public docstring review](../docstring-deep-dive/SKILL.md) for reader-facing quality and
[Generated Transform Target Contracts](../../../docs/design/transform-target-contracts.md) for coverage.

- Register each public constructor mode, including mutually exclusive behavior.
- Derive target-profile applicability from declared capabilities and prerequisites.
- Keep focused tests for semantic properties the generated profiles cannot express.
- Check strict JSON transport, public reconstruction, fresh-data execution, and the declared replay strength.

## Report

For each finding, give severity, file and line, the affected public behavior, and evidence that demonstrates the issue.
Distinguish confirmed defects from benchmark hypotheses or design decisions. Report checks actually run and any
unresolved validation limits. If there are no actionable findings, say so.
