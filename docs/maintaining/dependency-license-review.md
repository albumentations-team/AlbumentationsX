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

The current package does not copy these runtime dependencies into its wheel or
sdist. The project-level notices that are packaged are described in
[license provenance](license-provenance.md) and
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
