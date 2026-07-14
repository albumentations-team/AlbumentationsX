# AlbumentationsX License History

This file records the public license boundary between the legacy
`albumentations` project and the successor `AlbumentationsX` repository. It is
a provenance record, not a substitute for the license text that accompanied a
particular copy.

## Timeline

- **May 27, 2025 — legacy Albumentations 2.0.8.** The
  [`2.0.8`](https://github.com/albumentations-team/albumentations/tree/2.0.8)
  tag points to commit `4d2cf04b6635663275a747333754410ef255e54c` in the
  legacy repository. That release was published under the MIT License. Its
  exact license notice is preserved at
  [`LICENSES/MIT-Albumentations-2.0.8.txt`](LICENSES/MIT-Albumentations-2.0.8.txt).
- **June 19, 2025 — AlbumentationsX begins.** The first commit in this
  repository, `c1720fbab8209450328ef2e68f0ddc0c4806f7a8`, created the
  AlbumentationsX successor repository. Its `LICENSE` contained GNU AGPL
  version 3, while its README separately described a commercial-license path.
  Inherited legacy material retains the MIT notice described below.
- **April 8–May 28, 2026 — published `AGPL-3.0-or-later` metadata.** Every
  published tag from `2.1.3` through `2.3.1` declared
  `License-Expression: AGPL-3.0-or-later` in package metadata. The boundary
  starts at tag `2.1.3`, commit
  `068f0ec5a6a49e0b0f8b138ea0dcdc1d60cdcc21`, and ends at tag `2.3.1`, commit
  `337dc65588a032e2bd878462cdc7c5cdb099c6b6`. Those releases and their license
  metadata remain as published.
- **Beginning with release 2.3.2 — `AGPL-3.0-only`.** AlbumentationsX adopts
  `AGPL-3.0-only` prospectively beginning with release `2.3.2` unless a later
  notice says otherwise.
  Separately negotiated commercial licenses may grant alternative permissions
  for the scope stated in an executed agreement or order form. The already
  published `2.3.1` artifacts must never be rebuilt or republished with
  different license metadata.

## No Retroactive Relicensing

Creating AlbumentationsX did not revoke or alter the MIT permissions already
granted for legacy Albumentations releases. A person who received legacy
Albumentations 2.0.8, or an earlier MIT-licensed revision, may continue to rely
on the MIT license that accompanied that copy.

Likewise, this history file does not assert that later AlbumentationsX changes
apply to legacy releases. It explains why some AlbumentationsX files have a
lineage that includes MIT-licensed Albumentations material while the current
AlbumentationsX repository has a different default outbound license.

The prospective `AGPL-3.0-only` notice also does not withdraw, narrow, or
reinterpret permissions already granted for the published `2.1.3` through
`2.3.1` releases. Copies of those releases continue under the notices and
metadata that accompanied them.

## How To Read The Repository

- [`LICENSE`](LICENSE) contains the current repository notice and complete
  AGPL-3.0-only text.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) identifies inherited or
  third-party material that requires a separate notice.
- A file-specific copyright or license notice controls for that file when it
  expressly differs from the repository default.

When a release changes the default license, introduces material under another
license, or changes the provenance boundary described here, update this file,
the notices, package metadata, and release verification in the same pull
request.
