---
name: docstring-deep-dive
description: Quality bar for docstrings in albumentations. Use when writing or updating docstrings in albumentations/, especially for transforms and public APIs.
---

# Docstring deep-dive quality

Apply these criteria to **every docstring you write or update** in albumentations (transforms, public functions, and any API that appears on the docs site).

**Transform apply methods:** Do **not** add docstrings to `apply`, `apply_to_image`, `apply_to_mask`, `apply_to_images`, or other `apply_to_*` methods in transform classes. The transform class docstring and the base interface in `transforms_interface` are sufficient; apply methods are implementation detail.

## 1. First paragraph: 120–160 chars, useful short description (elevator pitch, two lines)

The **first paragraph** is the useful short description: an elevator pitch that explains intuitively what the function or transform does. It appears as the web/search preview under the link.

- **Paragraphs are separated by blank lines.** There is no blank line within a paragraph. So the first elevator pitch (120–160 chars) is **one paragraph**: it occupies two lines of text **with no blank line between them** (two lines, not "line + blank line + line").
- **Length:** **120–160 characters** (under 120 loses value, over 160 gets cut off).
- **Line limit 120 chars** ⇒ **first paragraph is two lines**. No single line may exceed 120 characters (ruff E501). So the first paragraph must be split across two lines; do not use a single long line with `# noqa: E501`.
- **Content:** Intuitive, user-facing summary — a true **elevator pitch**: what the transform/function does, how it works in one sentence, and when it's useful. So someone can decide "do I need to click?" Do **not** list parameter names ("Parameters: x, y, z" or "Params: ...") in the first paragraph — that belongs in Args. Do **not** write "Used by X" or "Used in Y". Do **not** use "Preserves X" boilerplate (e.g. "Preserves channel count", "preserves dtype and channels", "preserves shape") — that wastes the short description; describe effect and when to use it instead. Do not put: "Targets: ...", "Same shape", **return type**, or **"Supports uint8 and float32"** / Image types — return type in Returns; dtype/target support in Image types / Targets. All transforms support uint8 and float32 unless noted.
- **Line wrap:** Break at a word boundary so each line stays under 120 chars.
- **When shortening:** Do not delete useful information. Move any removed content into the second paragraph, a Note, or the relevant Args/Returns section so it is still documented.

**Good example (first paragraph, elevator pitch, two lines):**
```text
    """Sharpen the image via unsharp masking: blur, subtract from original, add back with
    strength. Use when you need crisp edges without changing overall brightness.

    More detail...
```

**Do not include:** "Used by X" or "Used in Y"; "Parameters: ..." or "Params: ..."; "Preserves channel count" / "preserves dtype" / "preserves shape" (boilerplate); return type or "Supports uint8/float32". Describe what it does and when to use it. Keep return type only in Returns; "Targets"/"Same shape" in a later paragraph if needed.

**Bad example (do not use):** Filling the 120–160 chars with meta boilerplate instead of an elevator pitch:

```text
    """Apply Gaussian blur using a randomly sized kernel. Params: blur_range, sigma_range.
    Supports image, volume. See Args and Examples.
```

- Do **not** use "Params: ...", "Supports ..." (including "Supports uint8 and float32"), or "See Args and Examples" in the first paragraph — that is already in the docstring (Args, Targets, Image types, Examples sections). It wastes the short description and tells the reader nothing about what Gaussian blur *is* or when to use it.
- **Good first paragraph:** Describe the transform's effect and when it's useful in 120–160 chars, e.g. "Smooth the image with a Gaussian kernel (weighted average; reduces noise and fine detail). Kernel size and sigma are sampled randomly per call."

## 2. Well written

- Use **Google-style** sections: Args, Returns, Raises, Examples, References (where relevant).
- Every argument in Args must have a **type** in the docstring (e.g. `param (float): Description`).
- Returns section must have a **type** (e.g. `dict[str, Any]: ...` or `None`).
- Be consistent and clear; avoid jargon without a brief explanation.

## 3. Examples

- **Every transform** and important public function must have an **Examples** section.
- Follow the pattern from `docs/contributing/ai_assistant_guidelines.md` and the add-transform skill: sample
  image, mask, bboxes, keypoints, Compose with params, and a call showing the result. Use `>>>` for doctest-style
  blocks.
- For non-transform APIs, include a minimal runnable example that shows typical usage.

## 4. Math where possible

- Transforms with a clear mathematical formulation (affine, color, geometric, normalization) should include a short **Note** or inline math with the key equations (e.g. rotation matrix, normalization formula, transfer function).
- Use standard notation; keep it concise (one or two lines of math is enough when it adds clarity).

## 5. Use-cases / problems

- Include at least one sentence (or a “Use when” / “Typical use cases” line) describing **which problems or tasks** the API is for (e.g. segmentation, object detection, robustness to lighting, data augmentation for medical imaging).
- Help the reader decide “is this the right transform/function for my use case?”

## 6. Similar transforms / See also

- **Where possible**, mention related or alternative transforms so users who know basic ones discover others.
- **See also** (and **Related transforms**): Use a **bullet list** (`-` per item) with 2–4 alternatives and brief when-to-use hints (e.g. `- RandomFog: Patch-based fog; use when…`). **One transform per bullet** — do not combine multiple transforms in a single bullet.
- **Note:** Use a **bullet list** (`-` per point). Note is **pure info** only — no call-to-action (no "Explore other transforms…" or "Consider using…"). Put discoverability in See also, not in Note.
- **Reciprocal cross-links:** When you add transform X to transform Y's See also, update X's docstring to mention Y in its own See also so discoverability works both ways.
- Many users rely on a limited set (e.g. RandomResizedCrop, ColorJitter); See also helps them discover alternatives.

## 7. Deep dive (combined bar)

Together, the docstring should give a new user:

- What the API does and how it behaves.
- What the parameters mean and how to set them.
- When to pick this over alternatives.
- What similar transforms or functions exist.
- A runnable example and, when relevant, the underlying math or references.

## When to use this skill

- When writing or updating docstrings in **albumentations/** (especially transform classes and public APIs).
- When the google-docstring-parser pre-commit hook reports errors: fix the errors and at the same time bring the docstring up to this bar.
- When reviewing or adding new transforms: ensure the docstring meets all sections above (short description length, Args/Returns types, Examples, use-cases, See also where applicable).
