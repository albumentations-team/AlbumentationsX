---
name: validate-and-fix
description: After completing code changes, runs tests and pre-commit, then iteratively fixes failures until all pass. Use when finishing a coding task, validating changes, or when the user asks to run tests or fix errors.
---

# Validate and Fix

## Workflow

1. **Run tests**: `uv run pytest -m "not slow"`
2. **Run pre-commit**: `pre-commit run --all-files`
3. **If either fails**: Fix the issues, then repeat from step 1. Do not stop until both pass.

## Rules

- Iterate until clean. Do not report errors and stop—fix them.
- Exclude slow tests during iteration; run full suite once at the end if desired: `uv run pytest`
