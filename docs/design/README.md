# AlbumentationsX Design Documents

This directory contains design documents for significant features and architectural changes in AlbumentationsX.

## Purpose

Design documents are created for:

- Complex features requiring detailed planning
- Significant architectural changes
- Features with multiple implementation phases
- Systems that need comprehensive documentation for maintainers

**Note**: Regular bug fixes and small improvements should be documented in commit messages and PR descriptions, not as separate design documents.

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

### [Generated Transform Target Contracts](transform-target-contracts.md)

Coverage contract for the shared transform-case registry and reusable target profiles.

### [Compose Serialization and Execution Tracing](compose-serialization-and-tracing.md)

Canonical composition policy, JSON/YAML transport, and opt-in per-step trace paths, snapshots, and timing.

## Compose Architecture

### [Greenfield Compose Architecture](compose-greenfield-architecture.md)

Execution architecture for compiling an immutable `Compose` graph at construction time so repeated training calls perform only
sample-dependent work, with separate branch-free executors for ordinary, observed, trace, and replay routes.

## Active Design Work

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

### [Torch CPU Backend and Tensor-Native Compose](torch-cpu-backend-migration.md)

Plan for making Torch a soft-required runtime dependency and routing compatible NumPy/OpenCV/NumKong and Torch segments through
the existing `Compose`. The route planner may cross the representation boundary in either direction when the complete
measured path does not regress.

**Status**: In progress. The CPU Tensor boundary is implemented; capability routing and per-family Tensor paths remain
planned.

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
