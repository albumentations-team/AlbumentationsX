# Dependency license review

AlbumentationsX records its reviewed runtime dependency set in
[`legal/dependency-licenses.json`](../../legal/dependency-licenses.json). The
record covers the base package and all declared extras resolved from `uv.lock`.
It includes transitive dependencies and the NumPy and SciPy versions selected by
different Python and platform markers. Development, build, and CI tooling are
out of scope because they do not ship in the library wheel or sdist.

The initial review was completed on 2026-09-16 from the locked distribution
metadata and license files, with the named PyPI release as the source record.
The registry stores the version variants, SPDX expressions, and any binary-wheel
notice handling. `opencv-*`, SciPy, NumPy, and `pyvips-binary` need particular
care because their wheels can include additional binary components. Those
components remain separately installed dependencies; a distributor of a
combined environment must keep the notices supplied with the relevant wheel.

`tools/verify_dependency_licenses.py` rejects a dependency name absent from the
registry and writes the reviewed SPDX expressions into the CycloneDX SBOM. The
release workflow also compares the installed distributions' declared license
metadata with the identifiers accepted in the registry before publishing that
SBOM. The security workflow checks the exported dependency graphs. A future
dependency or license change is reviewed through the procedure in
[`LICENSE_POLICY.md`](../../LICENSE_POLICY.md).

The current package does not copy these runtime dependencies into its wheel or
sdist. The project-level notices that are packaged are described in
[license provenance](license-provenance.md) and
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
