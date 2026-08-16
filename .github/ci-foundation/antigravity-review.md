# AlbumentationsX Antigravity review policy

Review the supplied pull request for demonstrable correctness defects, security issues, behavioral regressions,
maintainability risks, and missing tests or documentation.

For runtime changes, inspect the public transform contracts, target support, dtype handling, and direct and Compose
routes. Performance claims require relevant benchmark evidence. Torch and CI changes must preserve the soft-required
user runtime and CPU-only automation boundary described in docs/design/torch-dependency-and-ci-greenfield.md.

For workflow, packaging, and legal changes, inspect the local policy and repository contracts. Do not propose broad
refactors outside the pull request's scope. Cite changed file paths and line numbers for each actionable finding.
