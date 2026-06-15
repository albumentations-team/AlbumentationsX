# Release Verification

AlbumentationsX release verification is based on three things:

1. GitHub Release assets are the canonical public release bundle.
2. PyPI publishes the same wheel and sdist via trusted publishing.
3. PyPI provenance/attestations bind the uploaded package files to the GitHub Actions workflow that produced them.

## Official Artifacts

For each release, the following artifacts are official:

- wheel in the GitHub Release assets
- sdist in the GitHub Release assets
- `SHA256SUMS.txt` in the GitHub Release assets
- CycloneDX SBOM JSON in the GitHub Release assets
- Correctness & Compatibility Report Markdown in the GitHub Release assets
- matching wheel and sdist files published on PyPI

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
- runtime dependency audit, workflow audit, and OpenSSF Scorecard status
- SBOM, checksum, and PyPI provenance links

The report is a transparency artifact. It describes tested guarantees and known
limitations; it does not prove absence of bugs.

## What The Trust Root Is

AlbumentationsX relies on ecosystem-standard trust roots instead of manual long-lived signing keys:

- GitHub Actions OIDC identity for the release workflow
- PyPI trusted publishing
- PyPI-hosted provenance/attestations for published distribution files

This means authenticity is anchored in the CI identity that built and published the release, not in a maintainer-managed GPG key.

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
