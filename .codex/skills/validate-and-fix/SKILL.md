---
name: validate-and-fix
description: After completing code changes, runs tests and pre-commit, then iteratively fixes failures until all pass. Use when finishing a coding task, validating changes, or when the user asks to run tests or fix errors.
---

# Validate and Fix

## Necessity Rule

Every code path, test, and CI job must justify itself with a specific decision it protects and a failure it can expose.
Do not add or run work because an adjacent file changed, because a broad command is convenient, or because a check ran in a
previous pull request. Remove duplicate coverage instead of accumulating it.

For validation, state the changed contract and select the smallest command set that can falsify it. Add a broader suite only
when the changed contract can affect its route. A version-only release uses the release preflight; a dependency or packaging
change beyond the version uses the corresponding dependency, artifact, and install checks.

## Workflow

1. **Name the decision and failure mode**: identify the changed contract and the smallest test, check, or artifact that can
   reject it.
2. **Run that focused validation**: use the matching test or quality-gate subcommand.
3. **Add only required adjacent checks**: code that can affect multiple routes, a workflow/router change, or a release protocol
   may require a broader command. Record why.
4. **If a selected command fails**: fix the issue, then repeat that command. Do not expand the validation scope unless the
   failure proves that the initial scope was incomplete.

For transform or guidance changes, the focused mechanical check is
`pre-commit run check-ax-coding-guidance --all-files`. Keep deterministic contract details in
`docs/contributing/coding_guidelines.md`; keep design and benchmark review in the applicable skill.

## Rules

- Iterate until clean. Do not report errors and stop—fix them.
- Do not run the full suite merely by habit. Run it when the changed contract, release protocol, or a focused failure requires it.
- For CI, workflow, support-policy, or dependency-profile changes, also run:
  - `uv run python -m tools.ci_matrix check`
  - `uv run python -m tools.ci_shard check`
  - `uv run pytest -q tests/test_ci_plan.py tests/test_ci_gate.py tests/test_ci_shard.py tests/test_pr_workflow.py`
  - `uv run zizmor --format=plain --min-severity=medium --min-confidence=medium .github`
- For legal or packaging workflow changes, follow the `license-integrity` skill and verify built wheel/sdist artifacts.

## Common Pre-commit Fixes

### E501 Line too long
- **Never** suppress with `# noqa: E501` or add `E501` to `pyproject.toml` ignore list.
- **Fix**: Split the line at a word or operator boundary. For docstrings, wrap to the next line — the
  Google format allows multi-line short descriptions. For code, break at `(`, `,`, or `+`.

### C901 / PLR0912 Function too complex / too many branches
- **Never** suppress with `# noqa: C901`, `# noqa: PLR0912`, or raise the limits in `pyproject.toml`.
- **Fix**: Extract private helper methods that each own a single concern. If a function exceeds the
  limit it is doing too many things and must be split.
