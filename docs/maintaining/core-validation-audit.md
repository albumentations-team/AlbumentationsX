# Core validation audit

The [pipeline validation contract](../validation.md) defines public error behavior. This inventory records why
core warnings and fallback paths exist and which paths required a correction.

| Area | Classification and behavior |
| --- | --- |
| Transform constructor | Compatibility warning for unknown keywords in permissive mode. Use the documented keyword or enable strict mode to catch typos. Declared-parameter errors and invalid argument binding propagate. |
| Composition setup | A single transform supplied in place of a sequence is wrapped with a warning. Use a sequence. `SomeOf` caps the count when sampling without replacement, including after graph edits. Set `n` to the available count to avoid the warning. A processor with no applicable transform warns because its preprocessing can still be useful. |
| `OneOrOther` | A one-branch configuration is retained with a compatibility warning. Use a two-branch configuration or a single-transform `Compose`. Empty configurations and extra branches raise because they cannot implement the binary choice. |
| Root input | Optional channels and accepted empty annotation representations are compatibility behavior. Bad aliases, mismatched shapes, malformed annotations, and required-field errors previously accepted or discovered after sampling now raise during input validation. |
| Tensor routing | Documented backend fallback. Supported CPU tensors use the native route or a NumPy bridge with restoration. Invalid Tensor inputs raise before route selection. |
| Keypoint label mappings | A warning distinguishes label alignment from optional semantic remapping when no mapping was provided. A mapping for a categorical label absent from the current sample is skipped with a warning so the same mapping can serve samples with different labels. |
| Replay and serialization | Range and tuple normalization restores values represented in JSON. When a type annotation cannot be adapted, construction still validates the supplied value. Replay warns when a transform depends on the original sample. Explicit `on_not_implemented_error="warn"` permits a diagnostic export with no restorable parameters. Missing fields, classes, and custom-transform registrations raise. |
| Optional dependencies and introspection | YAML and Hub support depend on installed extras. Missing functionality raises when requested. Registration and signature introspection can use documented alternatives for optional classes or unsupported annotations. |
| RNG synchronization | Missing worker context selects the normal process stream. Worker RNG errors propagate so a failed lookup cannot silently change determinism. |
| Analytics | Best effort only. Missing environment metadata, unavailable preferences or identifiers, and failed telemetry I/O never affect validation, target output, or RNG. |

The [contract tests](../../tests/contracts/test_validation_contract.py) run in the compatibility job selected by
the required PR gate. The version-bump preflight runs the same module and requires its result in the
[Correctness & Compatibility Report](correctness-and-compatibility-report.md).
