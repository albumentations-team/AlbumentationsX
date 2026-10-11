# Dependency license review

AlbumentationsX records its reviewed base runtime dependency set in
[`legal/dependency-licenses.json`](../../legal/dependency-licenses.json).
The verifier exports the base package and its transitive dependencies from

```bash
uv export --frozen --no-dev --no-emit-project
```

The lockfile preserves NumPy, SciPy, and other versions selected by Python and
platform markers. Declared extras are excluded unless a package is also
reachable through the base package. This includes `hub` and all OpenCV extras.
Torch is selected separately as a CI runtime profile and is outside this
registry. The vulnerability audit has its own scope and includes all extras.

The initial review was completed on 2026-09-16 from the locked distribution
metadata and license files, with the named PyPI release as the source record.
The registry stores the version variants, SPDX expressions, and any binary-wheel
notice handling. `opencv-*`, SciPy, and NumPy need particular care because
their wheels can include additional binary components. Those
components remain separately installed dependencies; a distributor of a
combined environment must keep the notices supplied with the relevant wheel.

`tools/verify_dependency_licenses.py` checks the base export against reviewed
names and versions, then writes the reviewed SPDX expressions into the
CycloneDX SBOM. The release workflow compares each installed base distribution's
declared license metadata with the identifiers accepted in the registry. PR
and scheduled security workflows use the same base-only license check;
`pip-audit` separately checks both the base and optional dependency exports.
Review future base dependency or license changes through
[`LICENSE_POLICY.md`](../../LICENSE_POLICY.md).

Same-repository Dependabot PRs that change only `uv.lock` and the registry use
`.github/workflows/dependency-license-sync.yml`. It starts when the ordinary PR
workflow is requested. Trusted default-branch code reads the PR lockfile as
data. `tools/update_dependency_licenses.py` verifies every
locked distribution's SHA-256, package identity, license metadata, declared
license files, and packaged licenses and notices against reviewed base
distributions. It does not install or execute those distributions. New
dependency names, missing evidence, changed license contents or metadata, and
non-PyPI sources stop the update for maintainer review.

The workflow records its distribution evidence in the
`dependency-license-review` artifact, retained for 90 days, and commits only
the computed registry to the Dependabot branch. A concurrent branch change
rejects the write. To run the updated commit without waiting for approval of a
token-generated PR event, the updater explicitly dispatches `pr.yml` on that branch.
The dispatched run verifies the current PR head and applies the same routing
and required checks as a normal PR event. Fork PRs and broader changes do not
receive privileged automatic updates.

To review an update locally, save the reviewed base lockfile and run:

```bash
python -m tools.update_dependency_licenses \
  --base-lock /path/to/base-uv.lock --head-lock uv.lock \
  --output /path/to/updated-registry.json \
  --evidence /path/to/license-evidence.json --cache /path/to/distributions
```

The command writes its output only after every candidate passes. Copy the
verified registry into `legal/dependency-licenses.json`, then run
`pre-commit run check-dependency-license-registry --all-files`.

The current package does not copy these runtime dependencies into its wheel or
sdist. The project-level notices that are packaged are described in
[license provenance](license-provenance.md) and
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
