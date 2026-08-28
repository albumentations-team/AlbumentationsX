# Release Verification

AlbumentationsX release verification is based on four things:

1. A successful version-bump pull request builds the final release bundle.
2. The bundle manifest binds every payload file to the reviewed source tree.
3. GitHub Release assets and PyPI receive the same verified wheel and sdist.
4. PyPI provenance/attestations bind the uploaded package files to the trusted
   delivery workflow.

## Official Artifacts

For each release, the following artifacts are official:

- wheel in the GitHub Release assets
- sdist in the GitHub Release assets
- `SHA256SUMS.txt` in the GitHub Release assets
- CycloneDX SBOM JSON in the GitHub Release assets
- Correctness & Compatibility Report Markdown in the GitHub Release assets
- matching wheel and sdist files published on PyPI

Each wheel and sdist also carries the repository license, license history,
third-party notices, and exact legacy Albumentations 2.0.8 MIT notice. The
inbound contributor agreement and its archive are intentionally excluded.

## Quick Verification

To verify a release as a downstream user:

1. Download the release assets from GitHub.
2. Run:

```bash
sha256sum -c SHA256SUMS.txt
```

3. Confirm the wheel and sdist checksums match the values in the checksum manifest.
4. Confirm the same version exists on PyPI.
5. Open the PyPI file details page for the wheel or sdist and confirm that:
   - the file was published with trusted publishing
   - attestation/provenance metadata is present
   - the source repository and workflow identity match `albumentations-team/AlbumentationsX`
6. Run `python tools/verify_legal_integrity.py --artifacts <wheel> <sdist>`
   from the matching source revision to confirm exact notice contents and CLA
   exclusion.

## Programmatic Verification

PyPI exposes attestation data through its integrity APIs and file details pages. Consumers who need stronger automation should:

1. Fetch the target distribution from PyPI.
2. Fetch the corresponding attestation/provenance data from PyPI.
3. Verify that the attested repository, workflow identity, and commit correspond to the expected AlbumentationsX release.
4. Verify that the distribution hash matches the downloaded file.

## Correctness Report Verification

The Correctness & Compatibility Report summarizes what CI tested for a release:

- supported Python and operating-system matrix
- lower-bound dependency coverage
- golden regression and property-test coverage
- performance benchmark evidence
- exact ASV baseline-to-candidate refs and the resolved benchmark profile
- performance-budget status for coverage contracts and regression triage;
  normal release bundles require `comparison.provided: true`
- runtime dependency audit, workflow audit, and OpenSSF Scorecard status
- pip-audit, zizmor, Scorecard JSON, and Scorecard SARIF artifacts where the
  corresponding workflows ran
- SBOM, checksum, and PyPI provenance links

The report is a transparency artifact. It describes tested guarantees and known
limitations; it does not prove absence of bugs.

## What The Trust Root Is

AlbumentationsX relies on ecosystem-standard trust roots instead of manual long-lived signing keys:

- the protected pull-request workflow that builds and verifies the release bundle
- GitHub's immutable Actions artifact storage between review and delivery
- the source, manifest, payload, and checksum digests verified by the delivery workflow
- GitHub Actions OIDC identity for the PyPI delivery job
- PyPI trusted publishing
- PyPI-hosted provenance/attestations for published distribution files

This means authenticity is anchored in the reviewed source, protected CI
workflows, immutable bundle, and publishing identity, not in a
maintainer-managed GPG key. PyPI provenance authenticates the delivery of the
verified files; the release manifest connects those files to the pull-request
build that produced them.

This posture is aligned with SLSA-style provenance principles: the source repository,
workflow identity, artifact build path, PyPI provenance, SBOM, and checksum manifest
are all published or linked for downstream verification. Provenance confirms build
origin and artifact identity; it does not guarantee source-code correctness.

## SBOM Verification

The CycloneDX SBOM attached to the GitHub Release is generated from the locked runtime dependency set used for the release. Consumers can:

1. download the SBOM JSON
2. inspect the listed runtime dependencies
3. compare them with the published package metadata and their own dependency review tooling

The SBOM is a transparency artifact. The checksum manifest and PyPI provenance are the primary authenticity artifacts.
