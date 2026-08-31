# Codex Instructions for AlbumentationsX

`check-ax-rules` owns configurable AX repository checks. Its `coding-guidance` rule owns deterministic AX source
contracts; the canonical explanation is in `docs/contributing/coding_guidelines.md`. Do not duplicate those mechanics
in this file.

Read `docs/contributing/codex_guidelines.md` to choose the owning skill or design document. Load only the workflow that
matches the task; runtime-code work always uses `performance-optimization`.

Read `.codex/rules/albumentations-rules.md`. For scoped work, read:

- `.codex/rules/benchmarking.md` for performance-sensitive changes
- `.codex/rules/testing-conventions.md` for test changes
