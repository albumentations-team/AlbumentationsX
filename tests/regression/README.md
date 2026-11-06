# Regression Tests

The regression suite stores small, reviewable behavior vectors for selected
transform contracts. Tests verify these vectors; they never regenerate them as
a side effect.

Golden vectors are compatibility sentinels. They complement the existing
parameterized transform sweeps, functional tests, serialization tests,
annotation geometry tests, 3D tests, and property tests. They should be added
when exact or structured output stability is important to review explicitly,
not as a mechanical duplicate of every transform test.

The sentinel set covers exact deterministic 2D geometry, including oversized
crop, tolerance-based HBB and OBB resize, and exact 3D volume/mask3d crop.
The OBB resize vector stores ordered corners in output pixels so equivalent
width, height, and angle representations compare as the same rectangle.
Each manifest case records the transform, params, seed, input
recipe, environment, behavior epoch, targets, stability mode, tolerance, and
output metadata.

`tests/regression/transform_contracts.py` also tracks public transform coverage
routes. A new public transform-like API should be covered by the established
parameterized sweeps or assigned an explicit focused route; otherwise the
regression coverage route test fails.

## Commands

```bash
uv run python tools/generate_regression_vectors.py --transform HorizontalFlip --epoch 2.4
uv run python tools/generate_regression_vectors.py --all --epoch 2.4
uv run python tools/verify_regression_vectors.py --all
uv run pytest tests/regression
```

Golden vectors use seeded `Compose` calls and generated synthetic inputs.
Image/mask recipes use `TestDataFactory`; annotation and 3D recipes are kept
small and explicit in the generator for reviewability. New vectors must declare
a stability mode: `exact`, `tolerance`, `digest`, or `structural`.
