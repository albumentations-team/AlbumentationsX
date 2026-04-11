---
name: internal-workspace
description: Use the repo `_internal/` directory for anything that must not be committed — scratch files, temporary outputs, local demos, agent artifacts, or one-off scripts. Use when creating temp files, debug dumps, or local-only tooling during a task.
---

# Internal workspace (`_internal/`)

## Rules

1. **Put non-repo files under `_internal/`** — not in `tools/`, project root, or `tests/` unless they are
   permanent, reviewed project assets.
2. **Applies to**: temporary scripts, screenshot/debug exports, large downloaded data, personal benchmark
   runs, WIP notebooks, Cursor/agent scratch output, anything you would otherwise `.gitignore` ad hoc at
   repo root.
3. **Do not commit** contents of `_internal/` except `_internal/.gitkeep` (the directory is in
   `.gitignore` via `_internal/*` with an exception for `.gitkeep`).
4. **If a file was useful long-term**, promote it into the proper place (`tools/` for maintained dev
   scripts, `tests/` for permanent tests, `docs/` for documentation) and follow normal review standards —
   do not leave it in `_internal/`.

## When helping the user

- Prefer writing ephemeral or user-specific artifacts to `_internal/<descriptive-name>/` rather than the
  tracked tree.
- Removing sensitive or mistaken files from **GitHub history** requires a history rewrite (e.g.
  `git filter-repo --invert-paths --path ...`) and a **force push**; that is separate from day-to-day use
  of `_internal/` for local junk prevention.
