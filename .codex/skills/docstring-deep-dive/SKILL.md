---
name: docstring-deep-dive
description: Review public AlbumentationsX docstrings for useful descriptions, runnable examples, parameter semantics, and related transforms.
---

# Public docstring review

Read the complete docstring and implementation before editing. Follow the format in
[Coding Guidelines](../../../docs/contributing/coding_guidelines.md) and run
`pre-commit run check-ax-rules --all-files`; the hook owns mechanically checkable requirements.

## Explain the effect and when to use it

The first paragraph is the web and search preview. Describe what the operation changes, how it changes it, and the
problem it helps model. Keep parameter lists, return types, supported targets, and dtype details in their own sections.
Avoid filling the preview with generic shape-preservation claims or instructions to read the rest of the docstring.

When shortening the preview, move useful details into the relevant parameter description or note. Remove repetition.

## Make parameters and examples usable

- Explain units, ranges, sampling behavior, defaults, and interactions that change the result.
- Show runnable public calls with imports, input data, and the targets the example claims to support.
- Use sampled ranges in examples when a parameter accepts a range. Show a fixed range only when fixing that value
  helps explain the behavior.
- For base classes, demonstrate a working custom implementation; for functions, show a typical call.
- Include the key equation or reference when it explains geometry, color, normalization, or another numerical choice.

## Help readers choose related transforms

Use `See also` for two to four related transforms where useful, with one transform and a short selection hint per
bullet. Keep cross-links reciprocal. Keep `Note` bullets factual; put recommendations in `See also`.

## Final check

A reader should be able to choose the operation, configure it, and run the example. Re-read the complete docstring
following the last edit, verify each claim against the public route, and check that examples expose consequential
behavior without narrating obvious Python statements.
