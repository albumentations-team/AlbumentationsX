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

## Common Pre-commit Fixes

### E501 Line too long
- **Never** suppress with `# noqa: E501` or add `E501` to `pyproject.toml` ignore list.
- **Fix**: Split the line at a word or operator boundary. For docstrings, wrap to the next line — the
  Google format allows multi-line short descriptions. For code, break at `(`, `,`, or `+`.

### C901 / PLR0912 Function too complex / too many branches
- **Never** suppress with `# noqa: C901`, `# noqa: PLR0912`, or raise the limits in `pyproject.toml`.
- **Fix**: Extract private helper methods that each own a single concern. If a function exceeds the
  limit it is doing too many things and must be split.
