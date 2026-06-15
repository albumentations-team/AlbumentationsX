# Regression Tests

The regression suite stores small, reviewable behavior vectors for selected
transform contracts. Tests verify these vectors; they never regenerate them as
a side effect.

## Commands

```bash
uv run python tools/generate_regression_vectors.py --transform HorizontalFlip --epoch 2.4
uv run python tools/generate_regression_vectors.py --all --epoch 2.4
uv run python tools/verify_regression_vectors.py --all
uv run pytest tests/regression
```

Golden vectors use seeded `Compose` calls and synthetic inputs from
`TestDataFactory`. New vectors must declare a stability mode: `exact`,
`tolerance`, `digest`, or `structural`.
