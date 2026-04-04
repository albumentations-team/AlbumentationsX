# STF Milestone 1 Evidence Bundle

This document tracks the concrete evidence package for:

`Milestone 1: Security, Transparency, Release Integrity & Operational Baseline`

## Repository Deliverables

The milestone implementation is evidenced by the following repository files:

- `SECURITY.md`
- `docs/maintaining/release-process.md`
- `docs/maintaining/release-verification.md`
- `docs/maintaining/operational-continuity.md`
- `MAINTAINERS.md`
- `.github/workflows/upload_to_pypi.yml`

## GitHub Configuration Evidence

Capture and store the following screenshots after configuration is enabled:

1. Repository settings page showing private vulnerability reporting enabled.
2. Repository Security tab showing the vulnerability reporting entry point.
3. A release page showing attached wheel, sdist, SBOM, and checksum assets.
4. PyPI file details page showing trusted publishing / provenance metadata for a released file.

Suggested storage location outside the repo:

- `grants/Sovereign Tech Fund 2025/milestone-1-evidence/`

## Verified During Implementation

The following milestone items were verified during implementation:

- GitHub private vulnerability reporting was enabled for `albumentations-team/AlbumentationsX` and verified via:

```bash
gh api repos/albumentations-team/AlbumentationsX/private-vulnerability-reporting
```

- Expected response:

```json
{"enabled":true}
```

- The release workflow logic was dry-run locally with:
  - `uv build`
  - `twine check`
  - wheel install in a clean virtual environment with `opencv-contrib-python-headless`
  - `import albumentations` from outside the repository checkout
  - `uv export --frozen --no-dev --format requirements-txt`
  - `cyclonedx-py requirements`
  - checksum generation with `shasum -a 256`

- Repository validation passed:
  - `uv run pytest -m "not slow"`
  - `pre-commit run --all-files`

## Browser-Captured Evidence

Public browser captures were collected during implementation:

- Security overview page screenshot:
  - `/var/folders/68/k137nch11m76w1plfrw320r00000gn/T/cursor/screenshots/security-overview-page.png`
- Report-a-vulnerability sign-in flow screenshot:
  - `/var/folders/68/k137nch11m76w1plfrw320r00000gn/T/cursor/screenshots/report-vulnerability-signin-required.png`
- Security advisories page screenshot:
  - `/var/folders/68/k137nch11m76w1plfrw320r00000gn/T/cursor/screenshots/security-advisories-page.png`

These show the public Security entry points. The repository settings screenshot proving the toggle itself must still be captured while logged in as a repository admin.

## Release Artifact Evidence

For the first milestone-compliant release, record:

- GitHub Release URL
- PyPI project URL for the release version
- attached wheel filename
- attached sdist filename
- attached SBOM filename
- attached checksum manifest filename

Status:

- Not yet captured in this branch-only implementation step.
- Capture this immediately after the first release cut from the merged workflow changes.

## Short Narrative For STF

Use this summary in milestone reporting:

AlbumentationsX now provides a documented vulnerability disclosure policy, a private reporting path through GitHub Security Advisories, a documented CI-gated release process, release-attached SBOM and checksum artifacts, trusted-publishing-based provenance for PyPI releases, and a lightweight operational continuity baseline covering release ownership, backup handling, and security triage routing.

## Invoice Submission Checklist

After milestone acceptance evidence is assembled:

1. Confirm the milestone due date and the final evidence links.
2. Prepare the invoice for `30,000 EUR`.
3. Include contract-identifying information on the invoice.
4. Add the reverse-charge note if required for the issuing entity.
5. Send the invoice within 14 days of milestone achievement to `invoices@sovereign.tech`.
