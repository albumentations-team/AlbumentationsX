# Codex Instructions for AlbumentationsX

Read and follow `docs/contributing/codex_guidelines.md`.

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
