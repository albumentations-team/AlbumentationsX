# AlbumentationsX Antigravity review policy

Review the supplied pull request for demonstrable correctness defects, security issues, behavioral regressions,
maintainability risks, and missing tests or documentation.

For runtime changes, inspect the public transform contracts, target support, dtype handling, and direct and Compose
routes. Performance claims require relevant benchmark evidence. Torch and CI changes must preserve the soft-required
user runtime and CPU-only automation boundary described in docs/design/torch-dependency-and-ci-greenfield.md.

NumPy Compose inputs are channel-last. They may use explicit-channel `(H, W, C)`, `(N, H, W, C)`, and `(D, H, W, C)`
layouts or channel-less grayscale `(H, W)`, `(N, H, W)`, and `(D, H, W)` layouts. Compose temporarily adds `C=1` to
channel-less inputs before transform execution and restores their original layouts afterward. Review Compose boundary
changes against both forms, and assess internal transform execution using the normalized explicit-channel layouts.
`mask3d` is a separate target and may omit the channel axis.

Public Tensor inputs remain channel-first: `image` is `(C, H, W)`, `images` is `(C, L, H, W)`, and `volume` is
`(C, D, H, W)`. Review Tensor routing against these layouts as described in
docs/design/torch-cpu-backend-migration.md.

For workflow, packaging, and legal changes, inspect the local policy and repository contracts. Do not propose broad
refactors outside the pull request's scope. Cite changed file paths and line numbers for each actionable finding.
