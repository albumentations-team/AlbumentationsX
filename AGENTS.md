# Codex Instructions for AlbumentationsX

`check-ax-rules` owns configurable AX repository checks. Its `coding-guidance` rule owns deterministic AX source
contracts; the canonical explanation is in `docs/contributing/coding_guidelines.md`. Do not duplicate those mechanics
in this file.

Read `docs/contributing/codex_guidelines.md` to choose the owning skill or design document. Load only the workflow that
matches the task; runtime-code work always uses `performance-optimization`.

Read `.agents/rules/albumentations-rules.md`. For scoped work, read:

- `.agents/rules/benchmarking.md` for performance-sensitive changes
- `.agents/rules/testing-conventions.md` for test changes

Every timing, including a quick ad-hoc check, runs with exactly one CPU thread per process (env vars, `cv2.setNumThreads(0)`, Torch intra- and inter-op threads set to 1). Load the `benchmark` skill before any timing; it holds the full controls.
