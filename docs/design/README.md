# AlbumentationsX Design Documents

This directory contains design documents for significant features and architectural changes in AlbumentationsX.

## Purpose

Design documents are created for:

- Complex features requiring detailed planning
- Significant architectural changes
- Features with multiple implementation phases
- System-wide behavior that needs maintainer documentation

Document regular bug fixes and small improvements in commit messages and PR descriptions.

## Current Architectural References

### [Bounding Box Processing](bounding_boxes.md)

Reference for HBB and OBB coordinate formats, clipping, filtering, and processor behavior.

### [Instance Binding](instance_binding.md)

Structural contract for keeping masks, bounding boxes, keypoints, and labels aligned as instances are filtered.

### [Mosaic Transform](mosaic.md)

Technical specification for the Mosaic transform's data handling, including label encoding and preprocessing for multiple input images.

### [Applied Configuration Replay Contracts](applied-config-replay-contracts.md)

Implemented configuration-centric contract system that verifies strict JSON transport, public transform
reconstruction, replay execution, exact output where declared, and non-default coverage for every public constructor
parameter.

### [Bounded 2D ElasticTransform](elastic-transform.md)

Implemented contract for the greenfield 2D elastic transform: bounded cubic B-spline coefficient grids, synchronized
targets, certificate-bounded keypoint inversion, and separate constructor, applied-config, and `ReplayCompose` persistence rules.

### [ElasticTransform3D](elastic-transform3d.md)

Implemented true 3D elastic deformation from compact orthogonal cubic control planes, one `remap3d` call per raster target,
a native CPU Tensor route for one channel, and a measured NumPy bridge for multi-channel volumes.

### [Generated Transform Target Contracts](transform-target-contracts.md)

Coverage contract for the shared transform-case registry and reusable target profiles.

### [Compose Serialization and Execution Tracing](compose-serialization-and-tracing.md)

Canonical composition policy, JSON/YAML transport, and opt-in per-step trace paths, snapshots, and timing.

## Compose Architecture

### [Greenfield Compose Architecture](compose-greenfield-architecture.md)

Execution architecture for compiling an immutable `Compose` graph at construction time so repeated training calls perform only
sample-dependent work, with separate branch-free executors for ordinary, observed, trace, and replay routes.

## Active Design Work

### [Greenfield Pull-Request CI](pr-ci-greenfield.md)

Plan for parallel pre-commit hook jobs, the complete 3 by 5 pull-request compatibility matrix, direct required contexts,
local large-input evidence for performance-sensitive PRs, bounded release ASV evidence, and removal of Codecov and
duplicated pull-request work.

**Status**: Implemented.

### [Greenfield Target-Specific Parameter Sampling](target-specific-parameter-sampling.md)

Plan for replacing flat sampled-parameter dictionaries and special `volume_*` fields with parameters plus
actual-target-key `TargetParams` records, including aliases, mixed representations, deterministic execution, and replay.

**Status**: Implemented (greenfield cutover).

The maintained sampler inventory is in [Target-Specific Sampling Inventory](target-specific-parameter-sampling-inventory.md).

### [Shared CI Foundation](https://github.com/albumentations-team/ci-foundation/blob/main/docs/architecture.md)

The public, SHA-pinned foundation owns Python and uv bootstrap, CPU-only Torch mechanics, and trusted Antigravity
orchestration. AlbumentationsX keeps its dependency groups, test selection, release policy, legal checks, deployment,
and project-specific review policy here.

**Status**: Implemented. The local CI profile action and Antigravity caller pin ci-foundation commit
6b9045dbea58026a1e8f96b0392c411934a27199.

### [Greenfield Torch Dependency and CI Architecture](torch-dependency-and-ci-greenfield.md)

Plan for keeping Torch out of package metadata while making every import-capable CI and documentation job request the
shared CPU-only runtime profile explicitly.

**Status**: Implemented. The repository uses explicit `none` and `torch-cpu` runtime profiles.

### [NumPy and CPU Tensor Transform Routing](numpy-tensor-routing.md)

Current contract for per-target container preservation, canonical NumPy/Tensor layouts, and transform-level Tensor
fallback through the existing NumPy paths.

**Status**: Implemented.

## Creating New Design Documents

When creating a new design document:

1. Use the `.md` extension (standard Markdown)
2. Include these sections:
   - **Overview** - What is being designed
   - **Problem Statement** - What problem it solves
   - **Design Principles** - Core design decisions
   - **Implementation** - Technical details
   - **Testing Strategy** - How to validate
   - **References** - External resources

3. Add a reference to the new document in:
   - This README
   - `.codex/rules/albumentations-rules.md` (if relevant for Codex-facing guidance)

4. Keep documents up-to-date as implementation evolves

## Related Documentation

- [Coding Guidelines](../contributing/coding_guidelines.md) - Code standards and best practices
- [Environment Setup](../contributing/environment_setup.md) - Development environment
- [Contributing Guide](../../CONTRIBUTING.md) - Contribution process
- [AGENTS.md](../../AGENTS.md) - Codex guidelines
