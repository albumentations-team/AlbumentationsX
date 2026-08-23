# AlbumentationsX Antigravity review policy

Review the supplied pull request for demonstrable correctness defects, security issues, behavioral regressions,
maintainability risks, and missing tests or documentation.

For runtime changes, inspect the public transform contracts, target support, dtype handling, and direct and Compose
routes. Performance claims require relevant benchmark evidence. Torch and CI changes must preserve the soft-required
user runtime and CPU-only automation boundary described in docs/design/torch-dependency-and-ci-greenfield.md.

Within Compose, image targets have channel-last layouts with an explicit channel axis: `image` is `(H, W, C)`,
`images` is `(N, H, W, C)`, and `volume` is `(D, H, W, C)`. Grayscale images and volumes use `C=1`. `mask3d` is a
separate target and may omit the channel axis.

Do not report missing transform support, compatibility branches, or test coverage for channel-less `(H, W)` images or
`(D, H, W)` volumes. Those layouts are outside the Compose transform contract. Assess volume changes with explicit,
channel-last `(D, H, W, C)` inputs.

For workflow, packaging, and legal changes, inspect the local policy and repository contracts. Do not propose broad
refactors outside the pull request's scope. Cite changed file paths and line numbers for each actionable finding.
