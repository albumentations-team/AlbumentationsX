# License verification record

On 2026-09-16, GitHub's License API identified this repository's `LICENSE` as
GNU AGPL v3 at remote `main` commit
`b49291f34158ec1288d846fe85de3426c92e586c`. The repository metadata uses
`AGPL-3.0-only`; published version 2.4.10 is the checked release snapshot.

`tools/verify_legal_integrity.py` verifies the canonical AGPL text, package
metadata, legacy MIT notice, CLA archive, and wheel/sdist notice contract. The
dependency review and SBOM controls are documented in
[dependency-license-review.md](dependency-license-review.md). The historic
license boundary is recorded in [LICENSING.md](../../LICENSING.md) and
[contribution-history.md](contribution-history.md).

The checked source package includes its existing legacy MIT notice as described
in [license provenance](license-provenance.md). Runtime dependencies are not
copied into the wheel or sdist; their license and notice records are maintained
separately in
[`legal/dependency-licenses.json`](../../legal/dependency-licenses.json).
