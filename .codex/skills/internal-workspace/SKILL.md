---
name: internal-workspace
description: Use the repo `_internal/` directory for anything that must not be committed — scratch files, temporary outputs, local demos, Codex artifacts, or one-off scripts. Use when creating temp files, debug dumps, or local-only tooling during a task.
---

# Internal workspace (`_internal/`)

## Use it for

- Temporary scripts, screenshot/debug exports, large downloaded data, personal benchmark runs, WIP notebooks, and
  Codex scratch output.
- Anything that would otherwise need an ad hoc root-level `.gitignore` entry.

The directory is ignored and pre-commit rejects staged contents other than `_internal/.gitkeep`, including files added
with `git add -f`.

## Promote durable work

If a file was useful long-term, promote it into the proper place (`tools/` for maintained dev scripts, `tests/` for
permanent tests, or `docs/` for documentation) and follow normal review standards — do not leave it in `_internal/`.

## When helping the user

- Prefer writing ephemeral or user-specific artifacts to `_internal/<descriptive-name>/` rather than the
  tracked tree.
- Removing sensitive or mistaken files from **GitHub history** requires a history rewrite (e.g.
  `git filter-repo --invert-paths --path ...`) and a **force push**; that is separate from day-to-day use
  of `_internal/` for local junk prevention.
