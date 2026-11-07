# Behavior compatibility and migration

Check the release notes before upgrading a training pipeline. AlbumentationsX currently permits intentional API and
behavior changes without a deprecation period, including changes within a major version. A seed does not promise the
same outputs across releases. Pin the package, dependencies, pipeline configuration, and input data when reproducing
an experiment.

This policy covers sampling and random sequences, replay, applied configurations, serialization, geometry, annotation
handling, interpolation, borders and fill, validation, warnings, defaults, and output values. Environment support and
retirement remain governed by the [support policy](support-policy.md).

## Decide how a change ships

- A correction to the documented contract may ship immediately. Examples include misaligned annotations, incomplete
  replay records, invalid inputs accepted by a fallback, or incorrect output values.
- An intentional change to a supported API, default, random sequence, or output may also ship without a warning
  period under the current policy. Its release note must distinguish the new contract from a bug fix.
- Every observable change needs the previous behavior, new behavior, affected users, minimal migration, first affected
  version, contract, and permanent tests. Documentation-only clarifications must say when behavior is unchanged.
- Do not introduce a temporary compatibility alias, dual reader, or fallback merely to defer a decision. A new migration
  adapter needs a named replacement and a testable deletion condition. Existing documented persistence formats remain
  governed by their contracts until a release explicitly changes them.

If a release explicitly announces a deprecation window, honor that window. The warning and release note must name
the replacement, the first warning version, and an unreleased removal version or an observable removal condition.
For example, a condition can require completion of a named migration and a specified minor release containing its
guide. "A future release" and dates or versions that have already passed are not removal conditions. Ordinary
compatibility diagnostics do not imply deprecation; the [validation audit](core-validation-audit.md) explains them.

A future change to this policy takes effect only through a published policy update and release note. It does not
retroactively create a warning period for earlier releases.

## Record release evidence once

The [unreleased notes](../release-notes/unreleased.md) are the behavior-change inventory for the next release. Each
level-three heading is a change, under a level-two category. Use this format:

```text
### Short descriptive title

- Previous behavior: What the old release did.
- New behavior: What callers now observe and whether this is a correction or an intentional change.
- Affected users: The concrete pipeline or input that triggers the difference.
- Migration: The smallest working replacement or cache/replay regeneration step, with an example.
- First affected version: unreleased
- Contract: A relative Markdown link to the owning contract.
- Tests: Relative Markdown links to the permanent tests.
```

Keep the `Release baseline` at the last published version. Record `none` explicitly in categories with no observable
change; do not omit a category or infer "none" from an empty section. Archive released notes under their version before
starting a new inventory. Historical migration guidance is available in the [2.6.0 notes](../release-notes/2.6.0.md).

`python -m tools.check_behavior_changes` checks the required fields, evidence links, categories, and links between this
policy and the maintenance documents. The release report reads that same inventory; there is no second hand-written
change list. Release preflight resolves `unreleased` to the version being built and rejects a version at or below the
baseline. Local report previews retain the unresolved version explicitly.

The [release checklist](release-process.md#behavior-change-checklist) requires this check. The generated
[Correctness & Compatibility Report](correctness-and-compatibility-report.md) includes every change's migration,
contract, and test links pinned to the tested revision. Publish it and the release notes with the verified package.
An unreleased inventory or a local report preview is not evidence of a published release.

## Update regression behavior epochs deliberately

A behavior epoch identifies a registered sentinel's expected semantics; it is not automatically the package version.
Keep its epoch when the registered output, tolerance, and stability promise are unchanged. Changes outside the sentinel
cases do not require a blanket epoch bump.

When intentionally changing a registered expectation:

1. Explain the old and new behavior, why the change is needed, and the affected inputs in the pull request and release
   note. Name each affected sentinel and its old and new epoch in the `Regression vectors` category.
2. Update the sentinel's `TransformContract.behavior_epoch`, regenerate only its vectors and manifest entries, and
   inspect the output diff. Do not regenerate expectations merely to silence a failing test.
3. Keep the registry and manifest epochs equal. Run the vector verifier and the focused behavior tests. Record any
   dependency-sensitive tolerance or stability change with its environment and rationale.

See the [regression suite](../../tests/regression/README.md) for the commands. The current contract work leaves the
existing sentinel outputs and epochs unchanged; its new edge cases live in focused contract tests.
