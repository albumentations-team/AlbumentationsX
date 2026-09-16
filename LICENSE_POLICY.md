# Dependency and contribution license policy

This policy governs third-party material proposed for AlbumentationsX. It does
not replace the repository's [AGPL-3.0-only license](LICENSING.md), its
[third-party notices](THIRD_PARTY_NOTICES.md), or the contributor rights
required by the [CLA](CLA.md).

## What is reviewed

The reviewed runtime dependency set is recorded in
[`legal/dependency-licenses.json`](legal/dependency-licenses.json). It covers
the base installation and every declared extra, including transitive packages
and platform-specific locked versions. The registry records the SPDX expression,
evidence source, decision, and any notice handling for each package. Build,
test, and CI tools are kept outside this runtime registry because they are not
part of the distributed library.

The registry is a reviewed record, not a blanket list of allowed or forbidden
licenses. A package can be acceptable in one distribution context and require a
different decision in another. In particular, a copied file, a vendored or
minified asset, a binary wheel, a font, and a package supplied in a combined
environment each need review of their own notices and redistribution terms.

## Changing dependencies or third-party material

Open a normal pull request for a new runtime dependency, a declared license
change, or copied or bundled third-party material. Include the dependency graph
change, an update to the registry with the upstream license evidence and a short
reason for the decision, and any required notice or packaging change. The CI
check compares the locked export with the registry and the release process puts
the reviewed SPDX expressions into the published SBOM. Vladimir Iglovikov makes
the usual maintainer review and merge decision; no separate label or approval
service is used.

A version update does not need a new manual decision when its verified license
information and applicable notices remain the same as the reviewed record. If
the evidence is missing, contradictory, or changed, treat it as a license
change and update the record before merge. Never replace unresolved evidence
with a generic "allowed" label.

Runtime dependencies are installed separately from the AlbumentationsX wheel
and sdist. When distributing a combined environment or copied binary material,
preserve notices that belong to those components. The project package itself
continues to include the notices required by
[license provenance](docs/maintaining/license-provenance.md).

## Contributions

Contributors must have the rights needed to submit their changes under the CLA.
The CLA does not grant rights in third-party code, data, fonts, binaries, or
assets that the contributor does not control. State the source, version,
license, and required attribution for any such material in the pull request.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the submission checklist.
