# Codex Working Guide for AlbumentationsX

Use this guide to select the workflow for your task. The `coding-guidance` rule in `check-ax-rules` owns deterministic
AX source contracts; [Coding Guidelines](coding_guidelines.md) explains its diagnostics.

## Choose the owning workflow

| Task | Read first |
| --- | --- |
| Add or substantially change a transform | `.agents/skills/add-transform/SKILL.md` |
| Review a transform | `.agents/skills/review-transform/SKILL.md` |
| Change, inspect, or optimize runtime code | `.agents/skills/performance-optimization/SKILL.md` |
| Measure a runtime claim | `.agents/skills/benchmark/SKILL.md` |
| Write public docstrings | `.agents/skills/docstring-deep-dive/SKILL.md` |
| Change tests | `.agents/rules/testing-conventions.md` |
| Change mixing transforms | `.agents/skills/mixing-transforms/SKILL.md` |
| Change licensing, procurement/privacy copy, packaging legal metadata, or CLA text | `.agents/skills/license-integrity/SKILL.md` |
| Validate a completed change | `.agents/skills/validate-and-fix/SKILL.md` |

Read a design document only when the change enters its boundary:

- `docs/design/bounding_boxes.md` for HBB/OBB behavior;
- `docs/design/applied-config-replay-contracts.md` for constructor state, sampling, or replay;
- `docs/design/transform-target-contracts.md` for transform cases and generated target coverage;
- `docs/vector-fields.md` for vector-field meanings, component layouts, NaN interpolation, and geometric dispatch;
- `docs/design/instance_binding.md` for instance binding, frame binding, or collection target semantics;
- `docs/design/mosaic.md` for Mosaic metadata and annotation handling;
- `docs/design/numpy-tensor-routing.md` for Tensor execution; and
- `docs/design/torch-dependency-and-ci-greenfield.md` for Torch packaging or CI.

## Skill layout

Repository skills live in `.agents/skills/<name>/`, which Codex reads. Claude Code reads `.claude/skills/<name>`, a
relative symlink to that directory; commit both. The `agent-skills` rule in `check-ax-rules` rejects a missing or
wrong link and invalid frontmatter. `release-notes` is explicit-only in both agents. Each skill keeps trigger cases
in `evals/evals.json`; run them with `python tools/eval_skill_triggers.py`.

## Work in this order

1. Establish the public contract, affected routes, and the smallest validation that could disprove the change.
2. Run `pre-commit run check-ax-rules --all-files` after a transform or guidance change. Treat a failing `AXG`
   diagnostic as the mechanical finding; record a pass as evidence rather than inventing one.
3. Review what the hook cannot prove: public API coherence, mathematical and target semantics, replay strength,
   ownership and aliasing, the appropriate functional/Albucore boundary, and benchmark evidence.
4. Record the commands actually run and their results. Do not describe unrun checks as evidence.

## Review outcomes

Report a concrete failure or decision, not a generic concern. A useful review finding names the affected public route,
why it is incorrect or slower, and the focused evidence needed to confirm the fix. Keep fixed source rules in the hook;
keep architecture and trade-offs in the owning skill or design document.
