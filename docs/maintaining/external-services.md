# External services and third-party assets

This inventory covers the AlbumentationsX source repository, its package
release process, and the Markdown rendered from this repository. It does not
cover the separate `albumentations.ai` website repository.

| Component or service | Use and boundary | Data or material | Source and notes |
| --- | --- | --- | --- |
| Mixpanel | Runtime telemetry when telemetry is enabled. The implementation and opt-out are documented in [the privacy notice](../privacy.md). | The event fields listed in the privacy notice, sent by HTTPS to `api.mixpanel.com`. | The privacy notice owns the current provider detail. |
| Hugging Face Hub | Optional `hub` extra; a caller who uses its integration may download Hub-hosted resources. | Request and repository details chosen by that caller. | `huggingface-hub` is listed in the dependency registry; remote content is governed by its own repository terms and license. |
| GitHub Actions and GitHub | Source hosting, pull requests, release automation, badges, and release assets. | Repository and workflow data. | Workflow actions are pinned by commit. |
| PyPI and the PyTorch wheel index | Resolve dependencies and publish package artifacts. | Package names, versions, hashes, and publishing metadata. | The PyTorch index is used only for the documented platform-specific Torch installation. |
| CLA Assistant | Individual CLA status for pull requests. | The contributor identity and acceptance statement described by the CLA procedure. | Acceptance records remain private. |
| README badges and images | Rendered only by a Markdown viewer: shields.io, badge.fury.io, GitHub badges, contrib.rocks, and Habr Storage images. | Browser requests from a documentation reader; no library runtime call. | The source URL appears in `README.md`; these are not vendored assets. |

The repository scan covered `albumentations/`, `docs/`, `README.md`,
`pyproject.toml`, and `.github/` for network clients and external URLs. No
vendored or minified JavaScript, CDN-loaded runtime code, or copied third-party
web asset was found in the Python package. Runtime dependencies are separately
installed and their license records are maintained in
[`dependency-licenses.json`](../../legal/dependency-licenses.json).
