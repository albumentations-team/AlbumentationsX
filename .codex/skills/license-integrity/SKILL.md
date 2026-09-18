---
name: license-integrity
description: Maintain AlbumentationsX license, CLA, provenance notices, and packaged legal artifacts consistently.
---

# License Integrity

Use this skill for any change to `LICENSE`, `CLA.md`, license history,
third-party notices, package license metadata, contributor acceptance language,
or public repository license wording. Also use it for library procurement and
privacy disclosures.

## Required Context

For licensing, CLA, provenance, or packaging changes, read these files completely
before editing:

1. `docs/maintaining/license-provenance.md`
2. `LICENSE`
3. `LICENSING.md`
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
- Keep `LICENSE` byte-identical to the complete canonical GNU AGPL version 3
  text; keep the repository expression and commercial-license path in
  `LICENSING.md`.
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

## Library Procurement and Privacy Copy

For prose-only changes, read the owning documents and the implementation behind
the affected claims. The full legal context and artifact checks apply when the
change also touches licensing, provenance, or package metadata.

- Describe the product as the AlbumentationsX Python library running in the
  customer's environment. Lead purchasing copy with product fit, license coverage,
  and the next step. Include limitations when they affect the buyer's decision
  or answer a specific question; do not add speculative warnings or repeat
  security/privacy caveats throughout licensing and procurement pages.
- AX owns its library assessment documents: `LICENSING.md`, `SECURITY.md`,
  `docs/privacy.md`, and `docs/maintaining/`. Link to the owning document instead
  of copying another procedure.
- `albumentations.ai` is AX's project website. Preserve relevant website links,
  including documentation, benchmarks, Explore, newsletters, licensing, and funding.
  Website sign-in, accounts, and saved user content are separate from library
  execution and the commercial license; do not present them as AX requirements
  or license features. Keep the official contact `vladimir@albumentations.ai`.
- Keep telemetry to a brief README mention linking to `docs/privacy.md`, which
  owns the collected fields and global opt-out instructions. API parameter docs
  explain the control and link to that notice. Do not repeat telemetry copy in
  licensing or general procurement pages; answer explicit buyer questions in
  their assessment documents.
- Verify telemetry disclosures against the event model, collectors, backend,
  and pre-import opt-out behavior. A persistent random UUID is not proof of
  anonymity. Telemetry remains default-on with opt-out; its use is limited to
  product analysis and development, not prospect identification or sales.
- Describe the analytics provider in `docs/privacy.md` as the current implementation.
  Keep general product wording independent of the provider. When providers change,
  update the relevant facts and account for versions still using the previous service.
- Provider retention, storage region, deletion, certifications, and operational
  practices need evidence beyond source code. Do not turn an unverified target
  or a questionnaire question into a public assurance or recurring manual duty.

## Verification

Run the focused checks first:

```bash
uv run python tools/verify_legal_integrity.py
uv run pytest -q tests/test_legal_integrity.py
artifact_dir="$(mktemp -d)"
uv build --out-dir "${artifact_dir}"
uv run python tools/verify_legal_integrity.py --artifacts "${artifact_dir}"/*.whl "${artifact_dir}"/*.tar.gz
uv run twine check "${artifact_dir}"/*
```

Then run the repository quality gate required by the surrounding change.
