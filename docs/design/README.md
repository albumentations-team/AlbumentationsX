# AlbumentationsX design documents

Use these references for architectural decisions and behavior spanning multiple transforms or execution routes.
Document small fixes in their pull requests.

## Runtime contracts

| Reference | Scope |
| --- | --- |
| [Bounding Box Processing](bounding_boxes.md) | HBB and OBB coordinates, clipping, filtering, and processors. |
| [Instance Binding](instance_binding.md) | Alignment of masks, boxes, keypoints, and labels during filtering. |
| [Mosaic](mosaic.md) | Donor data, preprocessing, label encoding, and cell assembly. |
| [Applied Configuration Replay Contracts](applied-config-replay-contracts.md) | JSON transport, reconstruction, replay strength, and constructor coverage. |
| [ElasticTransform](elastic-transform.md) | Bounded B-spline deformation, synchronized targets, keypoint inversion, and persistence. |
| [ElasticTransform3D](elastic-transform3d.md) | Cubic control planes, raster remapping, and NumPy/Tensor volume routes. |
| [Generated Transform Target Contracts](transform-target-contracts.md) | Transform cases and reusable target profiles. |
| [Compose Serialization and Execution Tracing](compose-serialization-and-tracing.md) | Composition policy, transport, snapshots, and timing. |
| [Compose Architecture](compose-greenfield-architecture.md) | Graph compilation and per-invocation execution. |
| [Target-Specific Parameter Sampling](target-specific-parameter-sampling.md) | Shared parameters, actual-target-key records, determinism, and replay. |
| [Sampling Inventory](target-specific-parameter-sampling-inventory.md) | Sampler ownership and representation-dependent behavior. |
| [NumPy and CPU Tensor Routing](numpy-tensor-routing.md) | Container preservation, layouts, and NumPy fallback. |

## CI and dependencies

| Reference | Scope |
| --- | --- |
| [Pull-Request CI](pr-ci-greenfield.md) | Hook partitions, compatibility gates, and bounded performance evidence. |
| [Torch Dependency and CI](torch-dependency-and-ci-greenfield.md) | User-selected Torch builds and explicit CI runtime profiles. |
| [Shared CI Foundation](https://github.com/albumentations-team/ci-foundation/blob/main/docs/architecture.md) | Python/uv bootstrap, CPU-only Torch setup, and trusted review orchestration. |

Keep implementation status, pending evidence, and revision pins in the owning document or workflow. The foundation
owns shared setup; AlbumentationsX owns dependency groups, test selection, release policy, legal checks, and its review
policy.

## Add or revise a design

State the problem, required behavior, constraints, decision, and validation that can reject it. Record unresolved
choices and pending evidence explicitly. Add the document to this index and update the
[Codex working guide](../contributing/codex_guidelines.md) if it introduces a new workflow boundary.

Keep references aligned with implementation as it evolves. See [Coding Guidelines](../contributing/coding_guidelines.md),
[Environment Setup](../contributing/environment_setup.md), and [Contributing](../../CONTRIBUTING.md) for routine work.
