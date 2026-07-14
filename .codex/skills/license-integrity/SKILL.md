---
name: license-integrity
description: Maintain AlbumentationsX license, CLA, provenance notices, and packaged legal artifacts consistently.
---

# License Integrity

Use this skill for any change to `LICENSE`, `CLA.md`, license history,
third-party notices, package license metadata, contributor acceptance language,
or public repository license wording.

## Required Context

Read these files completely before editing:

1. `docs/maintaining/license-provenance.md`
2. `LICENSE`
3. `LICENSE_HISTORY.md`
4. `THIRD_PARTY_NOTICES.md`
5. `CLA.md`
6. `legal/cla/archive/MANIFEST.md`
7. the relevant packaging, contribution, release, and public-copy files

## Invariants

- Default repository SPDX expression is `AGPL-3.0-only`.
- The AGPL permits commercial use subject to its terms. Do not use the words
  commercial, proprietary, internal, or production as automatic license
  triggers.
- Separately negotiated commercial terms grant alternative permissions only
  for their stated scope. Do not promise support, warranties, maintenance, or
  an SLA unless an executed agreement or order form includes it.
- Do not alter the complete AGPL text below the `LICENSE` lead-in.
- Do not remove the exact legacy Albumentations 2.0.8 MIT notice or describe
  the successor license as retroactive.
- CLA acceptance is version-specific. Archive every operative byte version,
  record its SHA-256 identifier, and require explicit acceptance of a new
  version.
- An Entity Acceptance covers only exact named legal entities and contributors
  within the signer's documented authority.
- Release artifacts contain the four outbound license/provenance files and do
  not contain the inbound CLA or private acceptance records.
- Build into a fresh directory outside the checkout. A source distribution
  must not contain local build-output directories or nested release artifacts.

## Verification

Run the focused checks first:

```bash
uv run python tools/verify_legal_integrity.py
uv run pytest -q tests/test_legal_integrity.py
artifact_dir="$(mktemp -d)"
uv build --out-dir "${artifact_dir}"
uv run python tools/verify_legal_integrity.py --artifacts "${artifact_dir}"/*.whl "${artifact_dir}"/*.tar.gz
uv tool run --from twine twine check "${artifact_dir}"/*
```

Then run the repository quality gate required by the surrounding change. If a
hosted CLA bot, signature store, ruleset, or branch-protection setting must
change, report it as an unresolved external action until verified; a repository
edit cannot complete that task.
