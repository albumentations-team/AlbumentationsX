---
name: release-notes
description: Generate release notes for AlbumentationsX. Use when the user asks to prepare, draft, or write release notes for a new version (e.g. "prepare release notes for 2.x.y", "draft release X").
---

# Release notes

## Where they live

`_internal/release_notes/RELEASE_NOTES_<version>.md` (e.g. `RELEASE_NOTES_2.2.0.md`).

## Source of truth for what changed

- `git log --oneline <prev_tag>..HEAD` and `git show --stat <sha>` per commit.
- `git diff <prev_tag>..HEAD -- README.md` for the canonical added/removed transforms list.
- `git diff <prev_tag>..HEAD -- pyproject.toml` for dependency / version bumps.
- The PR descriptions in commit messages — they're written carefully, mine them for one-liners.

## Required structure

1. `## Summary` — 3–5 bullet TL;DR.
2. `## Breaking changes` — every backwards-incompatible change. Be specific (old → new). Include short before/after code snippets when the API surface changed.
3. `## New features` — new transforms, new core features, new params on existing transforms.
4. `## Bug fixes` — one entry per fix, link the PR.
5. `## Misc` — CI, build, deps, internal refactors users might notice.
6. `## Commits` — table of `commit | PR | description` for every commit since the previous tag.

Match the tone of `RELEASE_NOTES_2.1.2.md` and `RELEASE_NOTES_2.2.0.md`: terse, technical, no marketing fluff, no emoji.

## Transform name → link

**Every mention of a transform name in release notes MUST be a markdown link to its explore page**:

```
[TransformName](https://albumentations.ai/explore/transform/TransformName/)
```

This applies to:
- New transforms (in the "New features" section header and prose).
- Transforms touched by bug fixes.
- Transforms touched by breaking changes (including in the rename table).
- The `Commits` table descriptions.

The first mention in each section should be a link. Repeat mentions in the same paragraph can stay as plain backticked names if it would be visually noisy, but err on the side of always linking.

Examples (good):

```markdown
### [CopyAndPaste](https://albumentations.ai/explore/transform/CopyAndPaste/) (mixing)

[CopyAndPaste](https://albumentations.ai/explore/transform/CopyAndPaste/) was mixing positional indices and `_bbox_instance_id` values...

| `Rotate.limit`, `SafeRotate.limit` | `angle_range` |
```

Becomes:

```markdown
| [Rotate](https://albumentations.ai/explore/transform/Rotate/).limit, [SafeRotate](https://albumentations.ai/explore/transform/SafeRotate/).limit | `angle_range` |
```

Do **not** link:
- Core classes (`Compose`, `BboxParams`, `KeypointParams`, `BasicTransform`, `BaseDistortion`).
- Parameter names (`angle_range`, `blur_range`).
- Helper functions (`unpack_label_wrappers`, `to_tuple`).
- Internal modules (`core/utils.py`).

## Other conventions

- Code references use single backticks for params/files; triple-backtick fenced blocks for code samples.
- Use the actual commit short SHA (7 chars) and the real PR number in the commits table.
- "Breaking changes" section comes **before** "New features" — users scan that first.
- For mass renames, use a single 2-column table (`Old | New`) instead of a bullet list.
- If a feature spans multiple PRs (e.g. instance binding in #222, #223, #237), mention them inline in the prose, not as separate entries.
- If `pyproject.toml` version isn't bumped yet, note that in the response to the user; do not bump it as part of writing the notes.
