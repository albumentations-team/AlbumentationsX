# Codex Instructions for AlbumentationsX

Read and follow `docs/contributing/codex_guidelines.md`.
Deterministic AX source contracts are documented in `docs/contributing/coding_guidelines.md` and enforced by the single
`check-ax-coding-guidance` pre-commit hook. Do not duplicate those mechanical rules in this file.

Read the always-on Codex rule file `.codex/rules/albumentations-rules.md`. For scoped work, also read:

- `.codex/rules/benchmarking.md` for performance-sensitive changes
- `.codex/rules/testing-conventions.md` for test changes

Project-specific skills live in `.codex/skills`. Use the matching skill whenever the user asks for that workflow:

- `add-transform`
- `benchmark`
- `docstring-deep-dive`
- `internal-workspace`
- `license-integrity`
- `mixing-transforms`
- `performance-optimization`
- `release-notes`
- `review-transform`
- `validate-and-fix`

Use `performance-optimization` whenever implementing, reviewing, profiling, or optimizing runtime code. The skill
loads Albucore's canonical performance workflow, including delete-first review, vectorization, grouped reductions,
LUT and random-generator selection, backend comparison, Albucore extraction, and safe in-place operations.

## PR Review Guardrails

- Reuse Albucore dtype, conversion, and arithmetic helpers when they provide the required semantics. Do not recreate
  conversions such as uint8 normalization locally without a documented reason.
- Record the exact validation commands and results in the PR or task report. Do not claim a broader quality gate than
  was run; include both mypy and Pyrefly when a change relies on type narrowing.
- For runtime-sensitive changes, publish every before/after benchmark cell, not only ranges or aggregates. Cover the
  affected direct and Compose paths, including direct batch methods and public image/volume batch routes, and include
  all relevant modes, fallbacks, and specializations. State and justify any intentionally bounded axis.
- Rework or explicitly justify regressions above 5%, following `.codex/rules/benchmarking.md`; do not hide a common
  route's regression inside an overall average.
- Check `git status --short` before handoff. Because `pre-commit run --all-files` omits untracked files, run
  `pre-commit` with `--files` for each new file until it is tracked or staged.
