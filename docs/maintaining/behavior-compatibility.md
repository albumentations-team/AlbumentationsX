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
- Every observable change is described in the release notes with the previous behavior, new behavior, affected users,
  and a minimal migration. Add a permanent test for the new behavior.
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

## Update regression behavior epochs deliberately

A behavior epoch identifies a registered sentinel's expected semantics; it is not automatically the package version.
Keep its epoch when the registered output, tolerance, and stability promise are unchanged. Changes outside the sentinel
cases do not require a blanket epoch bump.

When intentionally changing a registered expectation:

1. Explain the old and new behavior, why the change is needed, and the affected inputs in the pull request and release
   note. Name each affected sentinel and its old and new epoch.
2. Update the sentinel's `TransformContract.behavior_epoch`, regenerate only its vectors and manifest entries, and
   inspect the output diff. Do not regenerate expectations merely to silence a failing test.
3. Keep the registry and manifest epochs equal. Run the vector verifier and the focused behavior tests. Record any
   dependency-sensitive tolerance or stability change with its environment and rationale.

See the [regression suite](../../tests/regression/README.md) for the commands. The current contract work leaves the
existing sentinel outputs and epochs unchanged; its new edge cases live in focused contract tests.
