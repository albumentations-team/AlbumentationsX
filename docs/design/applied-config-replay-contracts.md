# Applied Configuration Replay Contracts

**Status:** Implemented
**Scope:** `BasicTransform.applied_config`, `Compose(save_applied_params=True)`, constructor serialization, and
`Compose.from_applied_transforms()`
**Primary readers:** maintainers changing transform constructors, validation, sampling, serialization, or replay

## What this system guarantees

Every public serializable transform has one or more named configuration cases. Each case runs through the same public
path a user calls:

1. construct the transform with a non-default parameter mode;
2. capture its realized configuration with `Compose(save_applied_params=True)`;
3. cross a strict JSON boundary;
4. reconstruct with `Compose.from_applied_transforms()`;
5. execute on fresh equivalent data; and
6. compare all supplied outputs when the case declares exact replay.

The contract unit is a constructor configuration, not a class. This matters when one class has mutually exclusive
fields, mode selectors, custom metadata keys, scalar-or-range behavior, or another cross-field validator.

The harness catches records whose individual values are valid but whose realized combination cannot be passed back
through the public constructor.

```mermaid
flowchart LR
    C["Construct public mode"] --> A["Capture applied configuration"]
    A --> J["Strict JSON transport"]
    J --> R["Public reconstruction"]
    R --> E["Execute on fresh data"]
    E --> Q["Check declared replay strength"]
```

## Keep the persistence contracts separate

| Contract | State preserved | Required assertion |
|---|---|---|
| Constructor serialization | Original stochastic policy | Dict, JSON, and YAML round trips preserve seeded behavior |
| Applied-configuration replay | Constructor state after realized overrides | JSON-transported record reconstructs and runs |
| `ReplayCompose` | Parameters sampled for one invocation | Stored call parameters reproduce supported targets exactly |

Passing one contract does not imply either of the others. Their tests remain separate.

## Architecture

```text
tests/
├── contracts/
│   ├── test_applied_config_contract.py
│   ├── test_constructor_parameter_coverage.py
│   ├── test_contract_harness.py
│   └── test_target_cluster_contract.py
├── helpers/
│   ├── applied_config.py
│   ├── contract_assertions.py
│   ├── contract_data.py
│   ├── target_contracts.py
│   ├── target_profiles.py
│   └── transform_cases.py
├── property/
│   └── test_applied_config_replay.py
└── test_serialization.py
```

- `transform_cases.py` is the single typed source of truth for public transform configurations.
- `contract_data.py` provides deterministic fresh data for ordinary and special targets.
- `applied_config.py` owns the five-level public-path runner and failure diagnostics.
- `target_profiles.py` owns reusable target workloads without transform-specific lists.
- `target_contracts.py` derives applicable case/profile pairs from declared capabilities.
- `contract_assertions.py` provides one exact structural comparison for replay and mutation checks.
- `test_constructor_parameter_coverage.py` rejects missing classes, duplicate IDs, and uncovered public parameters.
- `test_contract_harness.py` contains one deliberately broken transform for every contract level.
- `test_serialization.py` consumes the same cases for dict, JSON, and YAML constructor round trips.
- `test_applied_config_replay.py` adds bounded generated coverage for normalization and high-risk mode interactions.

See [Generated Transform Target Contracts](transform-target-contracts.md) for the target-cluster matrix and its
all-mode versus primary-mode policy.

## The typed case registry

`TransformContractCase` contains only the information needed to construct and exercise one public mode:

```python
@dataclass(frozen=True)
class TransformContractCase:
    case_id: str
    transform_cls: type[A.BasicTransform]
    init_kwargs: Mapping[str, Any] = field(default_factory=dict)
    primary_data_factory: ContractDataFactory = make_image_data
    context_factory: ContractContextFactory = make_empty_context
    primary_compose_kwargs: Mapping[str, Any] = field(default_factory=dict)
    replay_profile: ReplayProfile = ReplayProfile.RUNNABLE
    metadata_keys: frozenset[str] = frozenset()
    required_targets: frozenset[str] = frozenset()
    seeds: tuple[int, ...] = (137,)
```

Registry invariants are enforced:

- `case_id` is stable, lowercase kebab-case, and globally unique;
- `init_kwargs` contains only public constructor fields and never harness-owned `p` or `strict`;
- mappings are copied and exposed read-only, preventing parametrized tests from mutating shared state;
- every case has at least one deterministic seed;
- every public serializable transform has a case;
- every configurable public constructor parameter except `p` and `strict` has a non-default case;
- singleton `Literal` parameters are proven to have no legal non-default value rather than exempted;
- there are no coverage exemptions;
- transform-required metadata is built by `context_factory` and merged without key collisions;
- `required_targets` prevents target-dependent modes from collecting against incomplete profiles;
- cases needing bbox or keypoint processors carry them in `primary_compose_kwargs`.

`TRANSFORM_CASES_BY_CLASS` retains the complete mode index. `ALL_DUAL_TRANSFORM_CONTRACT_CASES` makes mode-sensitive
coverage explicit. `PRIMARY_TRANSFORM_CONTRACT_CASES` and `PRIMARY_DUAL_TRANSFORM_CONTRACT_CASES` provide one
canonical mode per class for mode-independent dtype, layout, and batch workloads. No compatibility class-to-kwargs map
exists beside the registry.

## Deterministic data factories

Factories accept an independent `np.random.Generator` and return fresh objects for every invocation. Implemented
factories cover:

- RGB, grayscale, and normalized float images;
- masks;
- horizontal and oriented bounding boxes with labels;
- keypoints with labels;
- volumes and `mask3d`;
- image, mask, volume, and `masks3d` batches;
- reference-image metadata;
- mosaic, copy-and-paste, overlay, and text metadata;
- configurable metadata keys.

The harness creates original and replay inputs from separate generators with the same seed. This makes equality useful
without allowing in-place mutation of one call to contaminate the other.

## Five contract levels

### Level 1: emission

The harness applies one transform with `p=1.0`, `strict=True`, and `save_applied_params=True`. It requires:

- exactly one applied record;
- a class name present in `SERIALIZABLE_REGISTRY`;
- `p` in the final configuration;
- no field outside the replay class's public constructor;
- equality between the captured record and `transform.applied_config`;
- a fresh record on the next invocation; and
- no mutation of the first invocation's record after a later call.

### Level 2: transport

The exact record crosses:

```python
json.loads(json.dumps(record, allow_nan=False))
```

This rejects callables, sets, NumPy arrays or scalars that were not normalized, non-finite floats, and other values that
cannot be carried by strict JSON.

### Level 3: reconstruction

The transported value goes to `Compose.from_applied_transforms()`. Reconstruction performs registry lookup, annotation-
driven tuple/range normalization, Pydantic validation, cross-field validation, and public construction.

Replay normalization uses the replay class's public type annotations. It preserves valid scalar fields such as `fill`,
wraps scalars only for tuple-only fields, converts JSON lists through Pydantic `TypeAdapter`, and is idempotent.

### Level 4: execution

The reconstructed pipeline runs on a fresh case sample with warnings promoted to errors. The result must retain all
non-metadata targets, avoid object-dtype arrays, and contain only finite numeric array values. The capture, repeated
capture, and replay calls must also leave every caller-owned fixture unchanged, including arrays nested in metadata.

### Level 5: equivalence

`ReplayProfile.EXACT` cases compare every common non-metadata target with exact array equality or ordinary value
equality. `ReplayProfile.RUNNABLE` means that applied configuration guarantees reconstruction and execution but does not
claim to capture all internal random parameters.

The negative controls intentionally cover emission, transport, reconstruction, execution, and equivalence, with extra
controls for stale-record and caller-input mutation. They prevent refactors from silently weakening any harness stage.

## Production replay rules

### Public constructor fields define the boundary

`BasicTransform.get_transform_init_args_names()` reads the concrete class's public `__init__` signature. Parent-only
implementation fields do not leak into serialization or applied records.

`BasicTransform._get_valid_config_keys()` uses the same boundary. A transform cannot emit a field merely because an
internal base class happens to accept it.

### Realized state replaces stochastic policy

An applied record represents a runnable realized constructor configuration. When a sampled field conflicts with its
source policy, the transform must explicitly clear or replace the policy field. Merging a sampled override with every
original constructor value is not sufficient.

### Aliases replay through canonical classes

`BasicTransform.get_applied_replay_class()` defaults to the concrete class. Convenience or deprecated aliases declare
`_applied_replay_class` when their realized configuration belongs to a canonical implementation:

- `TimeReverse` replays as `HorizontalFlip`;
- `TimeMasking` and `FrequencyMasking` replay as `XYMasking`;
- `ShiftScaleRotate` replays as `Affine`.

Capture stores the canonical class name and validates the final fields against that class.

### Constructor serialization remains transport-safe

Constructor serialization preserves the original policy rather than the realized sample. Static NumPy configuration,
such as an `Equalize` mask, must serialize to standard Python containers and reconstruct to the runtime type in its
schema.

## Constructor-mode coverage

Coverage is checked directly from each concrete public signature:

1. inspect every constructor parameter and resolve its runtime type hints;
2. ignore only `self`, `p`, `strict`, `*args`, and `**kwargs`;
3. recognize a singleton `Literal` whose sole value equals the default as non-configurable;
4. collect registered values that differ from the signature default; and
5. fail with `TransformName.parameter` for every missing configurable mode.

There is no skip or exemption list. Adding a public parameter makes CI fail until a real non-default case can construct,
capture, cross JSON, reconstruct, and execute.

This is a coverage floor, not a substitute for focused semantics. New `Literal` branches, mutually exclusive families,
or target-specific modes still need separate named cases when one non-default value cannot exercise their behavior.

## CI policy

`python -m tools.quality_gate contracts` includes:

```text
pytest -q tests/contracts
```

The repository-contract job runs this deterministic suite on every pull request selected for contracts. Because shared
base-class changes can invalidate every transform, changed-file scoping does not narrow these tests.

Bounded Hypothesis tests run in the existing property/regression CI paths with `ci-fast` and `ci-nightly` profiles. A
real generated failure should be promoted to a stable named registry case.

Local commands:

```bash
uv run pytest -q tests/contracts
uv run pytest -q tests/property/test_applied_config_replay.py --hypothesis-profile=ci-fast
uv run pytest -q tests/test_serialization.py
uv run python -m tools.quality_gate fast
```

## Contributor workflow

When adding or changing a transform:

1. add or update its cases in `tests/helpers/transform_cases.py`;
2. give every public parameter a non-default registered value;
3. separate standard target data from transform-required context instead of adding class-name branches in runners;
4. set `ReplayProfile.EXACT` only when applied configuration resolves all relevant randomness;
5. write `self.applied_config` overrides for every realized constructor field;
6. clear source policy fields that conflict with those realized values;
7. declare `_applied_replay_class` only when the emitted state belongs to a different canonical constructor;
8. confirm every declared target collects through the generated target-cluster matrix;
9. keep mathematical, dtype, target-alignment, and distribution tests alongside the transform; and
10. run the contract, target-cluster, property, serialization, and fast quality commands above.

The review question is concrete: **which named registry case exercises the final applied configuration through strict
JSON and `Compose.from_applied_transforms()`?**

## Failure diagnostics

`AppliedConfigContractError` reports:

- contract level;
- case ID and seed;
- transform class;
- original `init_kwargs`;
- captured applied record;
- JSON-transported record;
- detailed validation or execution error; and
- the original exception chain.

This is enough to reproduce a failure without temporary tests or debug prints.

## Non-goals

- Applied-configuration replay is not a replacement for `ReplayCompose` parameter replay.
- The harness does not infer special external data from arbitrary signatures; the registry declares context factories.
- Production does not instantiate a second transform on every application merely to revalidate the record.
- The contract suite does not replace mathematical, dtype, alignment, regression-vector, or distribution tests.
- Broad class-level skips and class-wide `xfail` entries are not accepted.

## Completion criteria

The implementation remains complete only while all of these statements are true:

- every public serializable transform is registered;
- every configurable public constructor parameter has a non-default case, with no exemptions;
- all cases pass emission, strict JSON transport, reconstruction, and execution;
- exact cases compare all supplied targets;
- image, mask, HBB, OBB, keypoint, volume, `mask3d`, image-, mask-, volume-, and `masks3d`-batch replay paths are
  represented;
- negative controls cover all five contract levels and fail at their intended levels;
- dict, JSON, and YAML constructor serialization consume the same registry;
- deterministic contracts run in the required PR gate;
- bounded generative coverage runs in property CI; and
- contributor documentation and transform skills point to this registry and no removed inventory.

## References

- `albumentations/core/transforms_interface.py`: applied-record assembly and replay-class selection
- `albumentations/core/composition.py`: capture and public reconstruction
- `tests/helpers/transform_cases.py`: canonical configuration registry
- `tests/helpers/contract_data.py`: deterministic special-data factories
- `tests/helpers/applied_config.py`: five-level harness
- `tests/helpers/target_profiles.py`: reusable target workloads and shared assertions
- `tests/helpers/target_contracts.py`: capability resolver, pair generation, and public-path runner
- `tests/contracts/`: deterministic contracts and negative controls
- `tests/property/test_applied_config_replay.py`: bounded generated cases
- `tests/test_serialization.py`: constructor round trips from the same registry
