# License and CLA Provenance

This document defines the repository controls that keep AlbumentationsX license
metadata, inherited notices, contributor grants, and release artifacts
consistent. It is an operational record, not advice about a user's facts.

## Current Repository Position

- Repository default: `AGPL-3.0-only`.
- The AGPL permits commercial use subject to its terms.
- Albumentations, LLC may offer separately negotiated commercial terms for the
  versions and uses specified in an executed agreement or order form.
- Commercial terms do not remove the public repository's AGPL availability.
- Support, maintenance, warranties, and service levels exist only when an
  executed agreement or order form expressly includes them.

Do not describe a commercial license as required solely because a use is
commercial, proprietary, internal, or in production. Those labels do not by
themselves resolve the license analysis.

## Authoritative Files

| File | Purpose |
|---|---|
| `LICENSE` | Repository notice and full AGPL-3.0-only text |
| `LICENSE_HISTORY.md` | Boundary between legacy MIT Albumentations and AlbumentationsX |
| `THIRD_PARTY_NOTICES.md` | Notice for inherited and separately licensed material |
| `LICENSES/MIT-Albumentations-2.0.8.txt` | Exact MIT notice from legacy tag `2.0.8` |
| `LICENSES/OFL-1.1.txt` | OFL-1.1 notice for the source-only Liberation Serif Bold test font |
| `CLA.md` | CLA version currently offered to contributors |
| `legal/cla/archive/MANIFEST.md` | Immutable CLA texts, byte hashes, and acceptance-record requirements |
| `pyproject.toml` | SPDX package metadata and PEP 639 license-file declarations |
| `conda.recipe/meta.yaml` | Conda license metadata and packaged notice list |

The complete AGPL text below the repository lead-in must not be edited. If the
project changes its default outbound license, treat that as a separate,
explicitly reviewed change rather than a copy edit.

## Historical Boundary

Legacy Albumentations tag `2.0.8` points to commit
`4d2cf04b6635663275a747333754410ef255e54c` dated May 27, 2025. Its `LICENSE`
file is MIT text with SHA-256
`bea4dc8e93ae2784bccd45f1cdba53da97b99646bca390c7d725e17b72dc2180`.

AlbumentationsX began in this repository at commit
`c1720fbab8209450328ef2e68f0ddc0c4806f7a8` on June 19, 2025. The successor
repository's different default license did not revoke or rewrite permissions
already granted for legacy MIT copies.

Published tags `2.1.3` through `2.3.1` declared
`License-Expression: AGPL-3.0-or-later`. Those artifacts and permissions remain
as published. Release `2.3.2` is the first prospective
`AGPL-3.0-only` distribution; never rebuild or republish `2.3.1` with the new
metadata.

The tracked `tests/files/LiberationSerif-Bold.ttf` is a separate source-only
test asset under OFL-1.1. Its embedded copyright notices name Google
Corporation (2010) and Red Hat, Inc. (2012), and its SHA-256 is
`d754ba427cfe0bca54ae052384baa8f842da5bd6550ad4da024ac441e7a7d5ce`.
The font and `LICENSES/OFL-1.1.txt` are excluded from wheel and sdist artifacts;
their notices accompany redistribution of the Git source repository.

## CLA Versioning

CLA acceptance is version-specific:

1. Archive every operative text without normalization.
2. Record its SHA-256 digest in `legal/cla/archive/MANIFEST.md`.
3. Make the acceptance statement name the exact version and date.
4. Preserve the accepting identity, timestamp, acceptance path, covered
   identity or identities, version, and digest in a private durable record.
5. Require a new acceptance for a materially new CLA version. Do not infer it
   from a prior signature or repository activity.

The hosted Version 1 CLA Assistant offer was the public Gist
`https://gist.github.com/ternaus/df31e11d8a3180ba5520f72b72d57198`, immutable
revision `3115a7364f5ab8a58a7e7ffa51dfdf1ec8a5b006`, timestamped
`2025-06-19T21:58:29Z`. Its 7,360 bytes have SHA-256
`0318da3ff5c1d7b6e67ab0affa59e97b7e64902ae591e0c2d9e39a6299f835e9` and
are byte-identical to the base64 archive of the initial Version 1 text. Hosted
signer records therefore refer to that digest, not the later 7,365-byte
formatting revision in the repository root.

The observed Version 1 pull-request check used the public label `license/cla`.
Repository history contains at least six external authors, so the signer export
and author-by-author reconciliation below are required evidence, not an empty
checklist exercise.

Version 2.0 provides separate individual and entity paths. An individual path
does not cover employer-owned work. An entity representative must identify the
exact legal entity, their authority, and the covered contributors; unnamed
affiliates are not automatically included.

### External migration still required

The repository cannot update hosted CLA Assistant settings or its stored
signature database. Before Version 2.0 is enforced as a merge gate, the owner
must:

1. Configure the bot to display and record the exact Version 2.0 statement.
2. Export and preserve a dated snapshot of the Version 1 bot configuration,
   signature store, accepted GitHub identities, timestamps, and associated pull
   requests before changing the hosted configuration.
3. Map each Version 1 acceptance to the exact archived digest in
   `legal/cla/archive/MANIFEST.md` based on the repository text shown when the
   acceptance was recorded; do not assign a digest without evidence.
4. Inventory external authors whose contributions were merged into
   AlbumentationsX and reconcile that inventory against the preserved Version 1
   records.
5. Confirm that new records identify Version 2.0 rather than reusing a generic
   Version 1 status.
6. Treat Version 1-only signers as requiring Version 2.0 reacceptance.
7. Establish the private Entity Acceptance record path described in `CLA.md`.
8. Test the entire flow with a non-maintainer pull request and retain evidence.

Until those checks pass, maintainers must verify Version 2.0 acceptance
manually and must not treat a green legacy bot status as Version 2.0 consent.

## Release Artifact Contract

Every wheel and sdist must contain exact copies of:

- `LICENSE`;
- `LICENSE_HISTORY.md`;
- `THIRD_PARTY_NOTICES.md`; and
- `LICENSES/MIT-Albumentations-2.0.8.txt`.

These files may appear below the distribution metadata license directory, as
required by the packaging backend. Neither `CLA.md` nor `legal/cla/` belongs in
a release artifact: the CLA governs inbound contributions, not package use.
The source-only Liberation font and `LICENSES/OFL-1.1.txt` are also excluded.

Run:

```bash
uv run python tools/verify_legal_integrity.py
artifact_dir="$(mktemp -d)"
uv build --out-dir "${artifact_dir}"
uv run python tools/verify_legal_integrity.py --artifacts "${artifact_dir}"/*.whl "${artifact_dir}"/*.tar.gz
```

The verifier checks source metadata, wheel `METADATA`, sdist `PKG-INFO`,
historical and notice structure, archived CLA hashes, artifact contents, and
the absence of inbound CLA, source-only font assets, and local build-output
directories from distributions. Always build into a fresh directory outside
the checkout so stale artifacts cannot contaminate a source distribution or be
mistaken for the current release.

## Change Procedure

When changing licensing, CLA, notices, or package metadata:

1. Read every authoritative file listed above.
2. Update all affected repository surfaces in the same pull request.
3. Run the legal-integrity tests and build both artifact types.
4. Inspect the verifier output and `twine check` result.
5. Record any external configuration task explicitly; do not imply that a
   repository commit changed hosted bots, branch rules, or past signatures.
6. In release review, verify the four required files by content, not only by
   filename.

Never remove a historical notice because current files have changed
substantially. Notice retention and copyright ownership are separate questions.
