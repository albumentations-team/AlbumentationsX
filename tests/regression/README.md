# Regression Tests

The regression suite stores small, reviewable behavior vectors for selected
transform contracts. Tests verify these vectors; they never regenerate them as
a side effect.

Golden vectors are compatibility sentinels. They complement the existing
parameterized transform sweeps, functional tests, serialization tests,
annotation geometry tests, 3D tests, and property tests. They should be added
when exact or structured output stability is important to review explicitly,
not as a mechanical duplicate of every transform test.

The initial sentinel set covers exact deterministic 2D geometry, a
tolerance-based bbox/keypoint resize contract, and an exact 3D volume/mask3d
crop contract. Each manifest case records the transform, params, seed, input
recipe, environment, behavior epoch, targets, stability mode, tolerance, and
output metadata.

Follow the [behavior compatibility policy](../../docs/maintaining/behavior-compatibility.md#update-regression-behavior-epochs-deliberately)
when changing expectations. Record the rationale and old/new epoch in the pull request and release notes, and update
the registry and manifest together. The verifier rejects mismatched epochs. Do not bump unaffected sentinels or
regenerate vectors merely to accept an unexplained failure.

`tests/regression/transform_contracts.py` also tracks public transform coverage
routes. A new public transform-like API should be covered by the established
parameterized sweeps or assigned an explicit focused route; otherwise the
regression coverage route test fails.

## Commands

```bash
uv run python tools/generate_regression_vectors.py --transform HorizontalFlip
uv run python tools/generate_regression_vectors.py --all
uv run python tools/verify_regression_vectors.py --all
uv run pytest tests/regression
```

Golden vectors use seeded `Compose` calls and generated synthetic inputs.
Image/mask recipes use `TestDataFactory`; annotation and 3D recipes are kept
small and explicit in the generator for reviewability. New vectors must declare
a stability mode: `exact`, `tolerance`, `digest`, or `structural`.
