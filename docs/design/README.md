# AlbumentationsX Design Documents

This directory contains design documents for significant features and architectural changes in AlbumentationsX.

## Purpose

Design documents are created for:

- Complex features requiring detailed planning
- Significant architectural changes
- Features with multiple implementation phases
- Systems that need comprehensive documentation for maintainers

**Note**: Regular bug fixes and small improvements should be documented in commit messages and PR descriptions, not as separate design documents.

## Available Design Documents

### [Dithering Transform](dithering.md)

Design specification for the Dithering transform, which applies various dithering algorithms (Floyd-Steinberg, Bayer matrix, etc.) to reduce color depth in images.

**Status**: Design complete, pending implementation
**Phase**: Planning

### [Keypoint Label Swapping](keypoint_label_swapping.md)

Design for semantic keypoint label swapping during geometric transforms (e.g., swapping left/right eye labels during horizontal flip).

**Status**: Implemented
**Phase**: Complete

### [Mosaic Transform](mosaic.md)

Technical specification for the Mosaic transform's data handling, including label encoding and preprocessing for multiple input images.

**Status**: Implemented
**Phase**: Complete

### [Applied Configuration Replay Contracts](applied-config-replay-contracts.md)

Implemented configuration-centric contract system that verifies strict JSON transport, public transform
reconstruction, replay execution, exact output where declared, and non-default coverage for every public constructor
parameter.

**Status**: Implemented
**Phase**: Complete

### [Torch CPU Backend and Tensor-Native Compose](torch-cpu-backend-migration.md)

Plan for making Torch a required dependency and routing compatible NumPy/OpenCV/NumKong and Torch segments through
the existing `Compose`. The route planner can cross the representation boundary in either direction when the complete
measured path does not regress.

**Status**: Proposed
**Phase**: Planning

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
