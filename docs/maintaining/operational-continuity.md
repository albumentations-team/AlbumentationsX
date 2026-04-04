# Operational Continuity

This document defines the minimal operational continuity baseline for AlbumentationsX release and security operations.

## Roles

- Primary maintainer and release owner: Vladimir Iglovikov
- Backup release owner for continuity: Mikhail Druzhinin

## Security Triage Routing

Security reports should enter through GitHub private vulnerability reporting.

Fallback path if GitHub private reporting is unavailable:

- `vladimir@albumentations.ai`

Public issues must not be used for unpatched security reports.

## Continuity Rules

- If the primary maintainer is unavailable for 5 business days during an active release window or security incident, the backup release owner can coordinate the response.
- If the primary maintainer is unavailable for 10 business days and a patch release is required, the backup release owner may cut or coordinate the patch release.
- If no release can be cut safely, the project publishes a status update and temporary mitigation guidance instead of rushing an unsafe release.

## Escalation

Escalation happens when any of the following are true:

- a suspected vulnerability affects supported versions
- release verification fails for a tagged release
- the release workflow fails after a public release is published
- a maintainer is unavailable during an active incident or scheduled release

Escalation actions:

1. Pause promotion of the affected release.
2. Route the issue to the primary maintainer.
3. If the primary maintainer is unavailable within the continuity window, route to the backup release owner.
4. Prefer the smallest safe fix and a patch release over an unreviewed broad change.

## Operational Expectations

- Keep release credentials out of long-lived maintainer-managed secrets whenever possible.
- Use trusted publishing and CI-based provenance for official releases.
- Keep release notes, security notes, and advisory links attached to the relevant GitHub Release.
- Maintain a public release process and a public security policy so external users know how to report issues and verify fixes.
