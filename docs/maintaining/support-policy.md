# Support Policy

This document defines the public compatibility policy for AlbumentationsX. The
policy covers tested Python versions, operating systems, dependency sets,
optional extras, and how support is retired.

## Python Versions

AlbumentationsX currently supports Python 3.11, 3.12, 3.13, and 3.14.

The package metadata must keep `requires-python`, Python classifiers, CI
workflows, the release process, and the correctness report template in sync.
When one of those files changes support, the others must move in the same pull
request.

The support window is reviewed quarterly. After Python 3.15 is stable and the
NumPy, OpenCV, and PyTorch wheel ecosystem supports it, maintainers should
keep at least four actively tested Python minor versions unless maintenance
cost becomes unreasonable.

## Operating Systems

| Combination | Policy | CI Coverage |
| --- | --- | --- |
| `ubuntu-latest` on Python 3.11, 3.12, 3.13, 3.14 | Guaranteed | Runtime-change PR gate and nightly |
| `windows-latest` on Python 3.11, 3.12, 3.13, 3.14 | Guaranteed | Runtime-change PR gate and nightly |
| `macos-latest` on Python 3.11, 3.12, 3.13, 3.14 | Guaranteed | Runtime-change PR gate and nightly |
| Non-x86 architectures | Best effort | Manual or future dedicated runners |

Runtime source and shared-test-infrastructure changes keep the full
all-OS/all-Python PR matrix. Changes that cannot affect runtime compatibility
use narrower risk-based profiles. The complete matrix also runs nightly and
before a release. Moving a supported combination out of the runtime PR profile
requires an update to this document and the matrix validator.

## Dependency Sets

| Dependency Set | Purpose | Initial Gate |
| --- | --- | --- |
| `locked-latest` | Tests the repository lockfile and normal contributor environment. | Selected PR gates and full nightly/release |
| `declared-minimum` | Tests the declared lower runtime bounds on Ubuntu and Python 3.11. | Nightly and release gate |
| `optional-extras` | Smoke-tests extras such as `hub` and OpenCV variants. | Advisory until stable |
| `pre-release-probe` | Probes future Python or dependency releases when wheels are available. | Scheduled advisory |

Lower-bound failures block a release unless the support policy, dependency
metadata, and release notes are updated in the same change. Minimum dependency
jobs are scoped to combinations where those lower bounds are actually
installable.

## Dependency Updates

`pyproject.toml` owns runtime requirements, optional extras, and development
groups. `uv.lock` records the resolved versions used by CI. Contributor setup
uses those groups through uv or pip; see the
[environment setup guide](../contributing/environment_setup.md).

Dependabot monitors the uv project with `versioning-strategy: lockfile-only`:

- Direct runtime and development dependencies receive weekly version updates
  within their declared ranges, after a seven-day cooldown for new releases.
- Patch and minor updates share one pull request. Major updates use separate
  pull requests.
- Security updates cover vulnerable direct and transitive dependencies and
  receive separate pull requests without the version-update cooldown.
- Updates that require changing a declared requirement, including exact pins
  such as `albucore==...`, need a maintainer-authored pull request.

Dependabot alerts and security updates stay enabled in repository settings.
The strategy and grouping options are described in the
[Dependabot options reference](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference).

Minimum runtime versions change when the library needs a newer API, a security
fix, or a supported installation that upstream packages no longer provide.
The pull request states the reason and verifies the affected behavior with
the declared minimum versions. A new upstream release alone does not require
raising the minimum.

A lockfile security fix protects the repository environment. If a supported
runtime requirement still permits a vulnerable version, maintainers review
and update that requirement and verify the new lower bound. Dependabot's
lockfile-only strategy leaves that compatibility decision to maintainers.

Dependency updates merge after the selected compatibility checks pass and a
maintainer approves the change. Changes to base runtime versions also require the
[dependency license review](dependency-license-review.md). The existing
declared-minimum nightly and release checks continue to verify the advertised
lower bounds.

## OpenCV Policy

The default CI runtime path uses `opencv-python-headless`. The
`opencv-contrib-python-headless` extra receives a smoke test. GUI OpenCV wheels
are outside normal Linux CI unless a workflow has a real GUI reason.

## Optional Extras

The `headless` extra is the default runtime path. The `contrib-headless` and
`hub` extras get targeted import or small functional smoke tests. Packages
reachable only through extras are outside the mandatory dependency license
registry unless they are also resolved by the base package. Torch is a
soft-required runtime dependency: the base install does not select a CPU, CUDA,
or MPS build, while importing `albumentations` requires an installed Torch
runtime. CI jobs that import AlbumentationsX explicitly select the CPU-only
profile; link-only static, packaging, audit, and static-documentation jobs do
not select Torch.

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
