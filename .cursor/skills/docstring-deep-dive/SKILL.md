---
name: docstring-deep-dive
description: Quality bar for docstrings in albumentations. Use when writing or updating docstrings in albumentations/, especially for transforms and public APIs.
---

# Docstring deep-dive quality

Apply these criteria to **every docstring you write or update** in albumentations (transforms, public functions, and any API that appears on the docs site).

## 1. First paragraph: 120–160 chars, useful short description (elevator pitch, two lines)

The **first paragraph** is the useful short description: an elevator pitch that explains intuitively what the function or transform does. It appears as the web/search preview under the link.

- **Length:** **120–160 characters** (under 120 loses value, over 160 gets cut off).
- **Content:** Intuitive, user-facing summary — what it does and main parameters, so someone can decide "do I need to click?" Not part of the preview: "Targets: ...", "Same shape", "Used by X", or **return type** (e.g. "Returns np.ndarray"). Return type belongs only in the Returns section of the docstring.
- **Line wrap:** Line limit 120 chars, so the first paragraph usually spans **two lines** (break at a word boundary).

**Example (first paragraph, two lines):**
```text
    """Sharpen via unsharp masking (blur, subtract, add back). Parameters: blur_limit,
    sigma_limit, alpha.

    More detail...
```

**NOT in the preview (keep in docstring body/Returns only):** "Targets: image, volume", "Same shape", "Used by X", and return type (e.g. "Returns np.ndarray"). Return type belongs in the Returns section; the rest can go in a second paragraph or body if needed.

## 2. Well written

- Use **Google-style** sections: Args, Returns, Raises, Examples, References (where relevant).
- Every argument in Args must have a **type** in the docstring (e.g. `param (float): Description`).
- Returns section must have a **type** (e.g. `dict[str, Any]: ...` or `None`).
- Be consistent and clear; avoid jargon without a brief explanation.

## 3. Examples

- **Every transform** and important public function must have an **Examples** section.
- Follow the pattern from CLAUDE.md and add-transform skill: sample image, mask, bboxes, keypoints, Compose with params, and a call showing the result. Use `>>>` for doctest-style blocks.
- For non-transform APIs, include a minimal runnable example that shows typical usage.

## 4. Math where possible

- Transforms with a clear mathematical formulation (affine, color, geometric, normalization) should include a short **Note** or inline math with the key equations (e.g. rotation matrix, normalization formula, transfer function).
- Use standard notation; keep it concise (one or two lines of math is enough when it adds clarity).

## 5. Use-cases / problems

- Include at least one sentence (or a “Use when” / “Typical use cases” line) describing **which problems or tasks** the API is for (e.g. segmentation, object detection, robustness to lighting, data augmentation for medical imaging).
- Help the reader decide “is this the right transform/function for my use case?”

## 6. Similar transforms / See also

- **Where possible**, mention related or alternative transforms so users who know basic ones discover others.
- Add a **See also** or **Related transforms** section listing 2–4 alternatives with brief when-to-use hints (e.g. “For per-channel shifts see `RGBShift`; for full affine see `Affine`”).
- This improves discoverability: many users only use a few transforms and are unaware of better alternatives.

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
