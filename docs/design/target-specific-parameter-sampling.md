# Greenfield Target-Specific Parameter Sampling

**Status**: Implemented (greenfield cutover)

**Owner**: AlbumentationsX core transform maintainers

## Overview

AlbumentationsX now samples one structured parameter plan for a transform invocation and resolves it for each active target.
The old flat dictionary model worked when all targets could consume the same realized parameters. It failed when a parameter depends
on the actual target's channel count, dtype, shape, batch layout, or volume layout.

Several transforms compensate by adding fields such as `volume_noise_map`, `volume_multiplier`, and `image_gains`. Those
fields solve individual branches, but they encode target routing in parameter names and cannot represent two image aliases
with different representations. The same defect appears whenever a transform derives an execution parameter from the first
image-like input and reuses it for every other input.

This document defines a greenfield cutover to structured, target-aware execution parameters. Each transform invocation still
represents one sampled augmentation event. The event can contain shared parameters and one or more materialized parameter
groups addressed by actual target key. Compatible targets may share a group; incompatible targets receive parameters suited
to their own representation.

The cutover is complete only when all sampled transforms use the structured contract and target-specific fields are removed.
There is no permanent compatibility path for flat sampled-parameter dictionaries.

## Decision Summary

The implementation will make these changes as one atomic repository cutover:

1. Build an ordered `TargetSet` for every invocation. Its entries are keyed by the actual input name, including aliases such
   as `image2`, and carry the canonical target type plus representation metadata.
2. Replace flat dictionaries returned by `sample_parameters` with `TransformParameterPlan`.
3. Store target-independent values in `TransformParameterPlan.shared`.
4. Store representation-dependent values in `TargetParameterGroup` objects. Each group names the actual targets that consume
   its parameters. Groups may overlap when different parameters have different sharing domains.
5. Resolve the relevant group before calling `apply`, `apply_to_images`, `apply_to_volume`, or an alias of those methods.
6. Persist the same structure in deterministic mode and `ReplayCompose` so replay performs no sampling or rematerialization.
7. Remove first-target sampling helpers from transform implementations and delete target names embedded in execution
   parameter names.

The core dispatch layer will not guess whether two targets are compatible. The transform defines its sharing key or constructs
groups directly because compatibility depends on transform semantics.

## Delivery and Dependencies

The cutover depends only on existing AlbumentationsX execution components: `SamplingContext`, `BasicTransform`, `Compose`,
`ReplayCompose`, target preprocessing, and the generated target-contract suite. It introduces no external runtime dependency.

PR #462 remains the motivating contribution. Its regression cases and valid transform-specific sampling behavior are
preserved, while target routing moves into the shared core contract defined here. The implementation removes the special
`volume_*` fields as part of the cutover.

Because the sampling override and execution-replay schemas are customization contracts, the completed cutover ships as a
documented breaking change. The implementation PR owns the migration guide, paired performance evidence, and exact-head test
results.

## Problem Statement

### Current execution model

`BasicTransform` currently performs these operations:

1. Derive shared metadata, including `shape`, from the first recognized target.
2. Call `sample_parameters` once and merge the result into one flat dictionary.
3. Save that dictionary for deterministic execution or replay.
4. Pass the same dictionary to every function in `_key2func`.

`add_targets` maps an alias to a canonical target function, but the actual input key is not part of parameter selection. A
transform therefore knows that it is applying an image function, yet cannot select parameters specifically sampled for
`image2`.

### Failure class

The defect is broader than volume noise. It occurs when all of the following are true:

- one invocation contains multiple image-like targets;
- a sampled parameter depends on a target property such as shape, channels, dtype, layout, or content;
- the transform derives that parameter from one target or from global metadata; and
- the same realized value is applied to another target with a different representation.

Observable failures include:

- broadcasting errors when a three-channel multiplier is applied to a five-channel volume;
- incorrect scaling when a value derived for `uint8` is applied to `float32`;
- invalid channel indices or permutations when channel counts differ;
- masks or dense noise maps with the wrong spatial shape;
- incomplete support for multiple aliases of the same canonical target type; and
- replay data whose field names encode only the built-in `image` and `volume` cases.

The bug can remain silent. A parameter may broadcast successfully while producing the wrong magnitude or correlation, so
shape-only tests do not provide sufficient coverage.

### Motivating regressions

[PR #462](https://github.com/albumentations-team/AlbumentationsX/pull/462) exposes both the immediate defect and the limit of
the local-field approach:

- `MultiplicativeNoise(elementwise=False, per_channel=False)` can still derive a three-value multiplier from `image` and apply
  it to a five-channel `volume`, producing a broadcasting error;
- the same representation mismatch is possible between `image`, `images`, and aliases mapped through
  `additional_targets`; and
- constant `AdditiveNoise` sampled from a `uint8` primary image can use the wrong scale for a `float32` secondary image even
  when broadcasting succeeds.

These cases require a single architectural fix because they differ only in which representation property invalidates a shared
materialization.

### Why local fields do not close the issue

A field such as `volume_noise_map` adds one second route. It cannot express all of these inputs in the same call:

```python
transform(
    image=image_rgb_uint8,
    image2=image_gray_float32,
    images=batch_rgba_float32,
    volume=volume_five_channel_uint8,
)
```

Adding `image2_noise_map`, `images_noise_map`, and similar fields would couple transform internals to caller-selected names,
duplicate application methods, and keep the first-target assumption in the sampling API. A structured actual-key mapping is
the required abstraction.

## Goals

- Support target-specific sampling for every canonical target and every `additional_targets` alias.
- Preserve the meaning of one transform invocation as one random augmentation event.
- Allow exact parameter sharing when targets are semantically compatible.
- Materialize correct values for targets with different shapes, channel counts, dtypes, layouts, or content.
- Keep geometric synchronization explicit through a shared spatial frame.
- Make target routing visible in types and replay data instead of parameter-name conventions.
- Fail before applying any target when a plan is incomplete, ambiguous, or incompatible with the current invocation.
- Preserve the fast path for the common single-target, shared-parameter case.
- Give custom transforms one documented way to express shared and target-specific parameters.

## Non-Goals

- Supporting different spatial frames for targets that a synchronized geometric transform requires to be aligned.
- Inferring transform-specific sharing semantics in `BasicTransform`.
- Changing random-number generators, seed derivation, probability gates, or transform ordering.
- Moving execution parameters into `applied_config`; constructor replay and execution replay remain separate contracts.
- Preserving old `ReplayCompose` payloads across this greenfield schema cutover.
- Adding a second runtime route that permanently accepts legacy flat dictionaries.
- Optimizing image-processing kernels unrelated to parameter sampling and dispatch.

## Design Principles

### Actual keys own target-specific state

The routing identity is the actual key present in the call: `image`, `volume`, `image2`, or another configured alias. The
canonical target type remains metadata used to select the application function and interpret layout.

This distinction is essential. Two inputs may both map to `image` while requiring different arrays, scaling, or channel
indices.

### A shared event may have multiple materializations

The transform samples one conceptual event. Shared policy values describe that event, while target parameter groups describe
how it is realized for compatible inputs.

For example, additive noise may share distribution parameters and intensity across all image-like targets. Targets with equal
shape, channel behavior, dtype scale, and sampling topology can share one dense noise map. A volume with a different channel
count or topology receives another map sampled from the same shared policy.

### Transforms define compatibility

Compatibility is semantic. Equal array shapes do not prove that a batch and a volume should share the same dense map. Dtype
may matter for one transform and be irrelevant for another. Content-derived transforms may require one group per input even
when all metadata matches.

The framework provides deterministic grouping helpers. Each transform supplies the grouping key or the groups themselves.

### Spatial synchronization is separate from representation

Geometric transforms need a validated spatial frame shared by aligned targets. Photometric and dense-value transforms need
per-target representation metadata. A single global `shape` derived from the first target conflates these concerns.

The sampling input therefore contains both `spatial_frame` and `targets`. Geometry reads the former. Representation-dependent
sampling reads the latter.

### Replay stores realized execution

Replay records the complete structured plan used by the original call. It does not rerun grouping, rescale values, or draw
random numbers. Target-specific replay validates the current target schema before any target is mutated.

### The ordinary path stays small

Shared-only plans use one shared dictionary and no per-target group lookup. Mixed-target calls pay work proportional to the
number of active targets. Dense arrays are materialized once per compatible group.

## Terminology

- **Actual target key**: The key supplied by the caller, such as `image2`.
- **Active target**: A non-`None` input whose actual key resolves through the transform's target mapping.
- **Canonical target type**: The built-in role to which a key resolves, such as `image` or `volume`.
- **Target view**: A read-only invocation-local view of one target and its representation metadata.
- **Spatial frame**: Validated spatial dimensions and coordinate conventions shared by synchronized targets.
- **Shared parameter**: A realized value consumed by every applicable target in an invocation.
- **Target parameter group**: One parameter mapping shared by a named set of compatible actual targets.
- **Sampling topology**: The semantic layout used to generate a parameter, such as one 2D map, one map per batch item, or
  one full 3D map.
- **Parameter plan**: The shared parameters and target groups for one transform invocation.

## Implemented Types

The implementation types live in a new `albumentations/core/transform_params.py` module. Custom transforms import the public
sampling contract from that module. The root package does not re-export these advanced customization types.

```python
@dataclass(frozen=True, slots=True)
class TargetDescriptor:
    name: str
    canonical_type: str
    shape: tuple[int, ...] | None
    spatial_shape: tuple[int, ...] | None
    channels: int | None
    dtype: Any
    dtype_scale: str | None
    layout: str
    sampling_topology: str


@dataclass(frozen=True, slots=True)
class TargetView:
    descriptor: TargetDescriptor
    value: Any


@dataclass(frozen=True, slots=True)
class TargetRequirement:
    shape: tuple[int, ...] | None = None
    spatial_shape: tuple[int, ...] | None = None
    spatial_shape_suffix: tuple[int, ...] | None = None
    channels: int | None = None
    dtype: Any = None
    dtype_scale: str | None = None
    layout: str | None = None
    sampling_topology: str | None = None


@dataclass(frozen=True, slots=True)
class SpatialFrame:
    rank: int
    shape: tuple[int, ...]


class TargetSet:
    ordered: tuple[TargetView, ...]

    def by_name(self, name: str) -> TargetView: ...
    def by_canonical_type(self, target_type: Targets | str) -> tuple[TargetView, ...]: ...
    def image_like(self) -> tuple[TargetView, ...]: ...
    def primary_image_like(self) -> TargetView: ...
    def group_by(self, key: Callable[[TargetView], Hashable]) -> tuple[tuple[TargetView, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class TargetParameterGroup:
    targets: tuple[str, ...]
    params: Mapping[str, Any]
    requirements: Mapping[str, TargetRequirement]


@dataclass(frozen=True, slots=True)
class TransformParameterPlan:
    shared: Mapping[str, Any]
    groups: tuple[TargetParameterGroup, ...] = ()
    target_schema: Mapping[str, str] | None = None

    @classmethod
    def shared_only(cls, params: Mapping[str, Any]) -> TransformParameterPlan: ...

    def params_for(self, target_name: str) -> Mapping[str, Any]: ...


class TransformParameterPlanError(ValueError): ...


@dataclass(frozen=True, slots=True)
class TransformSamplingInput:
    base_params: Mapping[str, Any]
    spatial_frame: SpatialFrame | None
    targets: TargetSet
    data: dict[str, Any]
```

`TargetView.value` is a borrowed reference for content-dependent sampling. Constructing a target view does not copy an array;
sampling code must not mutate the borrowed value. Metadata is derived after input preprocessing has established the
representation that the application function will receive. Layout and topology are stable strings derived from the canonical
target type plus the validated value; array rank alone never decides whether a value is a batch or a volume.

`SpatialFrame.rank` is declared by the transform family: 2D transforms consume the aligned height-width frame and 3D
transforms consume the aligned depth-height-width frame. A transform that has no shared geometric domain can opt out with a
`None` rank and derive representation-dependent sizes from target descriptors.

`TargetParameterGroup.params` contains only the group-specific delta. `params_for` builds one keyword dictionary from `shared`
and every group that contains the actual target; dense arrays are referenced, not copied.

`TargetRequirement` records only the representation properties on which a materialized parameter depends. A dense map may
require exact array shape, channels, dtype scale, and topology. A 2D spatial parameter applied slice-wise to a volume can
declare `spatial_shape_suffix` so replay checks height and width without coupling the parameter to volume depth. A channel
permutation may require only channel count. Replay
uses these declared constraints and does not reject harmless differences. `dtype` and `dtype_scale` are separate constraints:
some parameters require the exact storage type, while others require only an equivalent numeric range.

## Parameter Plan Invariants

The core validates a complete plan immediately after sampling:

1. `shared` and every group parameter mapping are treated as immutable by the execution layer for the rest of the invocation.
2. A parameter name cannot appear in both `shared` and a group. This prevents silent shadowing.
3. Every group contains at least one actual target key.
4. Every named key exists in the invocation and is supported by the transform.
5. A target key may appear in multiple groups when the groups supply different parameter names.
6. For any actual target, a parameter name appears in at most one group and never also in `shared`.
7. Group target names are stored in canonical deterministic order.
8. Every group declares requirements for each target it names, including an empty requirement when the materialization is
   representation-independent.
9. Parameters required by an application method are present in either `shared` or one of that target's groups.
10. A target that needs only shared parameters may remain outside all groups.
11. Application functions cannot access another target's groups.
12. Sampling and validation finish before the first application function is called.

The framework raises `TransformParameterPlanError`, a `ValueError` subclass containing the transform name and violated
invariant (or a `TypeError` when a sampler returns a non-plan). It must not fall through to a NumPy broadcasting error.

## Deterministic Target Ordering

Parameter sampling must not depend on keyword argument order. `TargetSet` uses this order:

1. canonical target priority defined once in core;
2. the canonical key before its aliases; and
3. aliases in lexical order within a canonical type.

Grouping preserves that order. Random draws for multiple groups occur in group order. The cutover may change seeded results
for existing mixed-target invocations; after the cutover, the new ordering is stable and covered by tests.

## Sampling API

`BasicTransform.sample_parameters` becomes a typed greenfield contract:

```python
def sample_parameters(
    self,
    inputs: TransformSamplingInput,
    sampling: SamplingContext,
) -> TransformParameterPlan: ...
```

The base implementation returns `TransformParameterPlan.shared_only({})`. Shared-only transforms return a plan explicitly:

```python
return TransformParameterPlan.shared_only({"angle": angle})
```

Target-sensitive transforms return shared policy plus groups:

```python
return TransformParameterPlan(
    shared={"distribution": distribution, "intensity": intensity},
    groups=(
        TargetParameterGroup(
            targets=("image", "image2"),
            params={"noise_map": shared_image_noise},
            requirements=image_noise_requirements,
        ),
        TargetParameterGroup(
            targets=("volume",),
            params={"noise_map": volume_noise},
            requirements=volume_noise_requirements,
        ),
    ),
)
```

Returning a plain dictionary is a type error after the cutover. This rule makes incomplete migrations fail at the sampling
boundary.

### Target-set lifetime

`TargetSet` wraps the current data immediately before one transform samples parameters. It cannot be constructed once at the
start of `Compose`, because an earlier transform may change shape, dtype, channels, or content. The stable actual-key ordering
template can be cached with the configured target mapping; values and descriptors belong to the current transform step.

Only active targets enter the set. Unknown metadata stays in `TransformSamplingInput.data`, and recognized keys with `None`
values remain inactive. Replay treats missing and `None` target values equivalently for target-schema validation.

Representation metadata is derived once from the borrowed values for the current transform step. The descriptors do not copy
arrays, and shared-only samplers take the same fast path as before; target-aware samplers use the already available descriptors.

### Shared base parameters

Core-derived values such as interpolation, fill values, and annotation processor metadata enter through
`TransformSamplingInput.base_params`. The core combines them with `plan.shared` after checking for duplicate keys.

The shared `base_params["shape"]` compatibility field remains for existing geometry and application code. It is normalized
to the canonical 2D shape for batched images, volumes, batched masks, and 3D masks. New target-sensitive sampling must not
derive representation from that first-target field: geometry reads `spatial_frame`, while target-specific sampling reads the
relevant descriptor from `inputs.targets`.

### Grouping helpers

`TargetSet.group_by` creates deterministic compatible groups without imposing a global policy. Typical keys include:

| Transform behavior | Example sharing key |
|---|---|
| Channel scalar or vector | `(channels, dtype_scale)` |
| Dense 2D value map | `(sampling_topology, spatial_shape, channels, dtype_scale)` |
| Full 3D value map | `(sampling_topology, spatial_shape, channels, dtype_scale)` |
| Channel permutation | `(channels,)` |
| Content-derived reference matching | actual target key |
| Shared geometry | no target groups; use `spatial_frame` |

`dtype_scale` is a transform-defined semantic value. Raw `dtype` is insufficient when several dtypes share the same value
range or when a transform operates in normalized floating-point space.

### Correlation rules

Every migrated target-sensitive transform documents and tests its correlation policy:

- compatible targets in one group consume the exact same realized parameters;
- different groups share policy parameters and draw separate representation-specific values;
- a transform may share a normalized latent program and materialize it per group when that preserves stronger alignment;
- batches and volumes remain different topologies unless the transform explicitly defines them as compatible.

This preserves the usual `additional_targets` expectation for compatible aligned images while supporting aliases that differ in
channels or dtype.

## Application Dispatch

`apply_with_params` receives the validated plan and keeps two execution paths.

The shared-only path performs the current loop with `plan.shared`. It checks once that `plan.groups` is empty and performs no
per-target lookup.

The target-aware path resolves by actual key:

```python
for target_name, value in data.items():
    if target_name in self._key2func and value is not None:
        target_function = self._key2func[target_name]
        target_params = plan.params_for(target_name)
        result[target_name] = target_function(value, **target_params)
```

Application methods use semantic parameter names consistently. `apply`, `apply_to_images`, and `apply_to_volume` may all
accept `noise_map`; dispatch selects the correct value. Fields whose only purpose was target routing are deleted.

Required keyword names for each canonical application function are compiled once from its signature when the transform class
is prepared. Plan validation resolves those requirements through every actual key before the application loop. Methods that
accept only shared parameters keep the shared-only path.

Annotation postprocessing that needs realized geometry receives the shared geometry parameters. If a future annotation target
requires its own materialization, it participates through its actual target key under the same plan contract.

## Replay and Deterministic Execution

The in-memory plan is normalized into this replay shape:

```python
{
    "parameter_schema": 2,
    "target_schema": {
        "image": "image",
        "image2": "image",
        "volume": "volume",
    },
    "shared": {"distribution": "gaussian", "intensity": 0.1},
    "groups": [
        {
            "targets": ["image", "image2"],
            "params": {"noise_map": image_noise},
            "requirements": {
                "image": image_requirement,
                "image2": image2_requirement,
            },
        },
        {
            "targets": ["volume"],
            "params": {"noise_map": volume_noise},
            "requirements": {"volume": volume_requirement},
        },
    ],
}
```

The execution-parameter payload retains the existing support for arrays and other runtime values. It is not part of the strict
JSON `applied_config` contract.

Replay follows these rules:

1. It reconstructs `TransformParameterPlan` without calling a sampler.
2. Shared-only plans keep current replay applicability.
3. A plan with target groups requires the exact recorded set of active actual target keys; extra keys fail as well as missing
   keys because no realized target-specific parameters exist for an added target.
4. Each current key must resolve to the canonical target type recorded in `target_schema`.
5. Each current descriptor must satisfy the properties declared by its `TargetRequirement`. Unconstrained properties may
   differ.
6. Missing, extra, renamed, duplicated, or incompatible targets fail before any transform is applied.
7. A legacy flat parameter dictionary is rejected with a schema-version error.

`get_applied_params()` returns the normalized structured payload. Existing consumers that inspect flat execution parameters
need a migration note. `applied_config()` remains constructor-valid and unchanged.

## Custom Transform Contract

This is a breaking change for custom transforms that override `sample_parameters` or inspect deterministic execution params.
The release containing the cutover must include a migration guide with these cases:

- wrap shared dictionaries with `TransformParameterPlan.shared_only`;
- replace `image_*` and `volume_*` execution fields with groups keyed by actual target names;
- use `inputs.spatial_frame` for synchronized geometry;
- use `inputs.targets` for representation metadata and content access;
- use a transform-defined `group_by` key when compatible aliases should share parameters; and
- update deterministic and replay assertions to the structured payload.

No deprecation shim will accept both return types. A clear boundary error is safer than silently routing a partially migrated
custom transform through the old semantics.

## Transform Inventory and Migration Classes

The implementation begins with an AST-assisted inventory of every `sample_parameters` override and every application method
whose required parameters differ by canonical target. The current tree has more than one hundred sampling overrides, so the
inventory is a required artifact of the implementation PR.

Known target-name workarounds include:

| Transform | Current target-specific fields | Required migration |
|---|---|---|
| `MultiplicativeNoise` | `volume_multiplier` (removed) | Group `multiplier` by channel, topology, shape, and mode |
| `AdditiveNoise` | `volume_noise_map` (removed) | Group `noise_map` by topology, shape, channels, and dtype scale |
| `GaussNoise` | `volume_noise_map` (removed) | Group `noise_map` with the same explicit correlation rules |
| `SaltAndPepper` | `volume_salt_mask`, `volume_pepper_mask` (removed) | Group `salt_mask` and `pepper_mask` |
| `FilmGrain` | `volume_grain` (removed) | Group `grain` by sampling topology and representation |
| `RicianNoise` | `volume_real_noise`, `volume_imaginary_noise` (removed) | Group both components together |
| `ExposureMatching` | `image_gains`, `volume_gains` (removed) | Group one semantic `gain` value per compatible actual target set |
| `GaussianBlur` | `volume_sigma`, `volume_kernel_size` (removed) | Keep the 3D sampling policy; route semantic `sigma` and `kernel_size` through a volume group |

The audit must also classify transforms that currently depend on first-target metadata without target-named fields:

| Class | Representative transforms | Audit question |
|---|---|---|
| Channel selection and permutation | `ChannelDropout`, `ChannelShuffle` | Does every target receive valid indices with the intended correlation? |
| Dense dropout and replacement | `PixelDropout` | Are mask shape, channel sharing, dtype scale, and replacement values target-correct? |
| Noise and color scaling | `AdditiveNoise`, `GaussNoise`, advanced color transforms | Does dtype or channel count change materialization? |
| Content-derived sampling | `ExposureMatching` and reference-based transforms | Is a sampled value tied to one target's pixels? |
| Batch and volume sampling | transforms with `apply_to_images` or `apply_to_volume` | Does sampling topology match the application method? |
| Geometry | all shape-dependent geometric transforms | Can the transform use the validated shared spatial frame exclusively? |
| Semantic 3D policy | transforms with volume-specific kernel or sigma policy | Is the difference true transform policy or only dispatch encoded in a name? |

Volume-specific constructor policy may remain when it expresses real 3D semantics. Realized execution values still use the
same semantic parameter names inside target groups.

## Implementation Plan

### Phase 0: Freeze the contract and inventory — complete

- Record every `sample_parameters` override, `get_image_data` call, `_extract_shape_from_data` dependency, and target-specific
  application signature.
- Assign each transform to shared-only, shared-spatial, grouped-by-representation, grouped-by-content, or mixed.
- Write the expected grouping and correlation policy beside every target-sensitive transform in the inventory.
- Confirm which execution-param inspection APIs are public and include them in the migration note.

The resulting [inventory](target-specific-parameter-sampling-inventory.md) is the durable source of truth for these
classifications and the review hook's coverage.

**Completion condition**: every built-in sampler has an assigned migration class; no target-sensitive transform remains under
an implicit first-target assumption.

### Phase 1: Add target and plan types — complete

- Add `TargetView`, `TargetSet`, `SpatialFrame`, `TransformSamplingInput`, `TargetParameterGroup`, and
  `TransformParameterPlan` in the core transform execution layer.
- Derive target views from the post-preprocessing values that application functions receive.
- Add deterministic target ordering and grouping helpers.
- Add plan validation and dedicated errors.
- Add the shared-only application fast path and actual-key group dispatch.

**Completion condition**: focused core tests prove routing, validation, ordering, and zero application before a validation
failure.

### Phase 2: Migrate the sampling boundary — complete

- Change the base sampling signature to accept `TransformSamplingInput` and return `TransformParameterPlan`.
- Convert every shared-only transform explicitly.
- Replace global first-target `shape` reads with `spatial_frame` or a specific `TargetView`.
- Reject dictionary returns.
- Update deterministic state and `get_applied_params()`.

**Completion condition**: all repository samplers type-check against the new signature and no built-in sampler returns a flat
dictionary.

### Phase 3: Migrate target-sensitive families — complete

- Migrate noise transforms as one family so shared distribution and dtype-scale rules stay consistent.
- Migrate channel transforms and dropout transforms.
- Migrate content-derived and reference-based transforms.
- Migrate remaining image, batch, and volume special cases from the inventory.
- Give every migrated transform explicit compatibility keys and correlation tests.

**Completion condition**: every known target-sensitive transform works with canonical targets and multiple aliases that differ
in channel count and dtype.

### Phase 4: Cut over replay — complete

- Store `parameter_schema: 2` and the normalized plan in deterministic state and `ReplayCompose`.
- Validate actual keys, canonical types, and materialized parameter compatibility before replay.
- Update replay fixtures and tests.
- Document the intentional incompatibility with flat execution-param payloads.

**Completion condition**: replay reproduces exact outputs for mixed representations without invoking random sampling or
materialization code.

### Phase 5: Delete legacy machinery — complete

- Delete target-routing fields such as `volume_noise_map`, `volume_multiplier`, `volume_sigma`, `image_gains`, and
  `volume_gains`.
- Delete application arguments that exist only to receive those fields.
- Remove `get_image_data(data)` and first-target shape extraction from transform sampling paths.
- Remove legacy replay decoding and any temporary migration adapters used on the development branch.
- Add a deterministic source check that rejects new application parameters whose only purpose is encoding a target name.
  Legitimate constructor policies such as 3D mode selection remain outside this rule.
- Update custom-transform and replay documentation.

**Completion condition**: a repository search and AST audit find no target-name execution fields or first-target sampling
dependencies.

### Phase 6: Validate and benchmark the atomic cutover — complete

- Run the focused core and transform matrices described below.
- Run the full test suite, type checks, documentation checks, and pre-commit hooks.
- Benchmark the same base commit and exact cutover head in the same environment.
- Inspect the final diff for duplicate compatibility logic, transitional branches, and obsolete tests.

**Completion condition**: all correctness gates pass, performance stays within budget, and the cutover contains one runtime
contract.

### Merge strategy

Implementation may use staged commits on one feature branch. The repository merges the work only after every built-in
transform and replay consumer has migrated. Intermediate commits are development checkpoints, not supported mixed-schema
releases.

## Testing Strategy

### Core contract tests

Add focused tests for:

- actual-key routing for canonical targets and aliases;
- deterministic target ordering independent of call keyword order;
- shared-only fast-path dispatch;
- exact sharing of one parameter object by compatible targets;
- separate groups for incompatible targets;
- overlapping target groups with disjoint parameter names;
- duplicate parameter names across overlapping groups;
- duplicate shared/group keys;
- unknown, repeated, empty, and missing group targets;
- missing required parameters;
- validation before any application function runs;
- read-only plan behavior and no caller-input mutation; and
- precise diagnostic messages containing transform and target identity.

### Representation matrix

Use a pairwise matrix that covers the failure dimensions without creating a full Cartesian test suite:

| Profile | Targets | Representation difference |
|---|---|---|
| Shared compatible aliases | `image`, `image2` | Same shape, channels, dtype, and topology |
| Channel mismatch | `image`, `image2` | Three and five channels |
| Dtype mismatch | `image`, `image2` | `uint8` and `float32` |
| Combined mismatch | `image`, `volume` | Channels, dtype, spatial rank, and topology |
| Batch mismatch | `image`, `images` | Single item and batch layout |
| Multiple aliases | `image`, `image2`, `image3` | One compatible pair and one incompatible target |
| Canonical mixture | `image`, `images`, `volume` | All built-in image-like routes in one invocation |

Use one-, three-, four-, and five-channel examples across the matrix. Include different spatial shapes only where the public
shape-check contract allows them.

### Transform-family tests

At minimum, add public `Compose` coverage for:

- `MultiplicativeNoise` across all `elementwise` and `per_channel` combinations;
- `AdditiveNoise` across constant, uniform, Gaussian, Laplace, and beta distributions;
- `GaussNoise`, `RicianNoise`, `SaltAndPepper`, and `FilmGrain`;
- `ChannelShuffle` and `ChannelDropout` with different channel counts;
- `PixelDropout` with per-channel masks, shared masks, and replacement values;
- `ExposureMatching` with canonical and aliased inputs; and
- every additional transform identified by the inventory.

Assertions must cover values and correlation, not only output shape and dtype. For a fixed seed, compatible targets should
prove exact parameter sharing where declared. Incompatible targets should prove valid target-scaled outputs and the intended
shared-policy relationship.

### Replay and determinism tests

- Same seed plus the same target set produces identical structured plans and outputs.
- Permuting call keyword order does not change the plan or outputs.
- Replay produces exact outputs for mixed channel counts and dtypes.
- Replay performs zero RNG calls and zero materialization calls.
- Missing or renamed aliases fail before any output is produced.
- Canonical-type changes fail even if the array shapes happen to match.
- Incompatible shape, channels, dtype scale, or topology fails with the recorded and current descriptors.
- Shared-only replay retains existing behavior.
- `applied_config` remains constructor-valid and strict-JSON serializable.

### Negative controls

Keep explicit regressions for the known silent and loud failures:

- a non-elementwise, non-per-channel multiplier sampled from a three-channel image and applied to a five-channel volume;
- constant additive noise with `uint8` primary data and `float32` secondary data;
- a channel permutation sampled for three channels and applied to a five-channel alias;
- a dense dropout mask whose source and destination spatial shapes differ; and
- two aliases of canonical type `image` that require separate materializations.

Each negative control must fail on the pre-cutover base or demonstrate the wrong value, then pass on the exact cutover head.

## Performance Budget

Measure base and exact head in the same process environment with identical inputs, seeds, warmup, and repetitions. Report
median and dispersion for these cells:

- cheap shared-only transform on one image;
- cheap shared-only transform inside `Compose`;
- target-sensitive transform on one image;
- compatible `image` plus alias;
- incompatible `image` plus alias;
- `image`, `images`, and `volume` together;
- deterministic capture; and
- replay.

The implementation must satisfy these budgets:

- shared-only application performs one empty-group branch per transform and no per-target group lookup;
- target view construction is allocation-light and does not copy arrays;
- target-aware dispatch is `O(T)` in active targets;
- dense parameters are materialized once per compatible group;
- ordinary single-target and shared-only representative cells regress by no more than 5%; and
- a larger regression blocks the cutover unless the implementation removes equivalent work elsewhere and provides measured
  full-route evidence.

Microbenchmarks support diagnosis. The acceptance decision uses representative `Compose` routes because parameter setup,
dispatch, and application all contribute to user-visible cost.

## Source Enforcement

The implemented `check-ax-coding-guidance` rules protect the architectural boundary.
The rule should detect:

- a `sample_parameters` override returning a plain dictionary;
- transform sampling code deriving representation from `get_image_data(data)` or another first-target helper; and
- parameters required only by one `apply_to_*` method whose names encode the target solely for routing.

The checks use AST structure and a narrow semantic allowlist. A broad text search would incorrectly reject legitimate public
policy names containing words such as `volume`.

## Files and Ownership

The implementation is expected to touch these ownership areas:

- `albumentations/core/transforms_interface.py`: sampling boundary, plan validation, actual-key dispatch, deterministic state;
- core type modules: target views, spatial frame, parameter plans, and errors;
- `albumentations/core/composition.py`: target-set construction, preprocessing boundary, and replay integration;
- transform modules: explicit plan construction and removal of target-named execution fields;
- `tests/`: core contract, family regressions, aliases, determinism, and replay;
- `tools/ax_coding_guidance/`: post-cutover deterministic enforcement; and
- public customization and replay documentation: migration guidance and structured parameter examples;
- `docs/design/target-specific-parameter-sampling-inventory.md`: durable sampler classification and review surface.

The inventory is public design documentation because it is part of the maintained source contract. Scratch benchmark
artifacts may live under `_internal/`, but durable performance conclusions belong in the PR description or public design
documentation before merge.

## Deletion Checklist

The final diff must confirm all of the following:

- [x] No built-in `sample_parameters` returns a flat dictionary.
- [x] No sampled representation is derived implicitly from the first image-like target.
- [x] No `volume_*`, `image_*`, or alias-specific execution field exists solely for dispatch.
- [x] No compatibility decoder accepts parameter schema 1.
- [x] No temporary dual dispatch path remains.
- [x] No dense parameter is duplicated across compatible groups.
- [x] No tests assert legacy target-specific field names.
- [x] No documentation teaches the legacy sampling signature.

## Acceptance Criteria

The class of bugs is resolved when:

1. Every built-in sampler returns `TransformParameterPlan`.
2. Every target-sensitive transform addresses parameters by actual target key through groups.
3. Multiple aliases of the same canonical target type can differ in channels, dtype, shape where allowed, and content without
   misrouting or silent rescaling.
4. Compatible aliases retain their declared parameter correlation.
5. Mixed `image`, `images`, and `volume` calls pass the representation matrix.
6. Replay records and restores the structured plan exactly and rejects incompatible target schemas before applying data.
7. `applied_config` behavior remains unchanged.
8. The full transform inventory has no unresolved first-target dependency.
9. Legacy target-routing fields and adapters are deleted.
10. Focused, full-suite, type, lint, documentation, and coding-guidance gates pass.
11. Paired base-to-head benchmarks satisfy the performance budget.
12. The custom-transform migration guidance is published in the coding guidelines and add-transform skill.

## References

- [Greenfield Compose Architecture](compose-greenfield-architecture.md)
- [Generated Transform Target Contracts](transform-target-contracts.md)
- [Applied Configuration Replay Contracts](applied-config-replay-contracts.md)
- [Coding Guidelines](../contributing/coding_guidelines.md)
- [Testing Conventions](../../.codex/rules/testing-conventions.md)
- [Performance and Benchmarking Rules](../../.codex/rules/benchmarking.md)
