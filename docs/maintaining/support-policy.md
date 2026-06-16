# Support Policy

This document defines the public compatibility policy for AlbumentationsX. The
policy covers tested Python versions, operating systems, dependency sets,
optional extras, and how support is retired.

## Python Versions

AlbumentationsX currently supports Python 3.10, 3.11, 3.12, 3.13, and 3.14.

The package metadata must keep `requires-python`, Python classifiers, CI
workflows, the release process, and the correctness report template in sync.
When one of those files changes support, the others must move in the same pull
request.

AlbumentationsX intentionally keeps Python 3.10 support for production computer
vision users that deploy on conservative stacks. The window is reviewed
quarterly. After Python 3.15 is stable and the NumPy, OpenCV, and PyTorch wheel
ecosystem supports it, maintainers should keep at least four actively tested
Python minor versions unless maintenance cost becomes unreasonable.

## Operating Systems

| Combination | Policy | CI Coverage |
| --- | --- | --- |
| `ubuntu-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Required PR gate |
| `windows-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Required PR gate |
| `macos-latest` on Python 3.10, 3.11, 3.12, 3.13, 3.14 | Guaranteed | Required PR gate |
| Non-x86 architectures | Best effort | Manual or future dedicated runners |

The current policy keeps the full all-OS/all-Python PR matrix. If CI time
becomes too high, Ubuntu all-version coverage remains required and some
Windows/macOS combinations may move to nightly only after maintainers update
this document and the matrix validator.

## Dependency Sets

| Dependency Set | Purpose | Initial Gate |
| --- | --- | --- |
| `locked-latest` | Tests the repository lockfile and normal contributor environment. | Required PR gate |
| `declared-minimum` | Tests the declared lower runtime bounds on Ubuntu and Python 3.10. | Nightly and release gate |
| `optional-extras` | Smoke-tests extras such as `pillow`, `pytorch`, `text`, `hub`, and OpenCV variants. | Advisory until stable |
| `pre-release-probe` | Probes future Python or dependency releases when wheels are available. | Scheduled advisory |

Lower-bound failures block a release unless the support policy, dependency
metadata, and release notes are updated in the same change. Minimum dependency
jobs are scoped to combinations where those lower bounds are actually
installable.

## OpenCV Policy

The default CI runtime path uses `opencv-python-headless`. The
`opencv-contrib-python-headless` extra receives a smoke test. GUI OpenCV wheels
are outside normal Linux CI unless a workflow has a real GUI reason.

## Optional Extras

The `headless` extra is the default runtime path. The `contrib-headless`,
`pillow`, `pytorch`, `text`, and `hub` extras get targeted import or small
functional smoke tests. The `pyvips` extra is scheduled-only when binary
availability is stable.

## Retiring Support

Support retirement requires:

1. A pull request updating `pyproject.toml`, CI workflows, this document, and
   the correctness report template together.
2. A release note announcing the change before or in the release that removes
   support.
3. At least one minor-release warning period for Python or OS support removals,
   except when upstream packages stop providing installable wheels or a severe
   security issue forces an emergency change.

Classifier updates, `requires-python`, CI matrix, release report content, and
public documentation must move together.
