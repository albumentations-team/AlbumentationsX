# Setting Up Your Development Environment

This guide will help you set up your development environment for contributing to AlbumentationsX.

## Prerequisites

- Python 3.10 or higher
- Git
- [uv](https://docs.astral.sh/uv/)
- A GitHub account

## Step-by-Step Setup

### 1. Fork and Clone the Repository

1. Fork the [AlbumentationsX repository](https://github.com/albumentations-team/AlbumentationsX) on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/AlbumentationsX.git
cd AlbumentationsX
```

### 2. Install Dependencies With uv

Create a local virtual environment and install the project plus development tools:

```bash
uv sync --locked --group dev --inexact
```

This is the canonical setup path for contributors and coding agents. It installs the same toolchain used by CI,
including Ruff, mypy, Pyrefly, pytest, pre-commit, and security tooling. It does not choose a Torch build. Install
the CPU, CUDA, or MPS Torch build required by your development environment before running code that imports
AlbumentationsX.

To reproduce the CPU-only runtime used by CI, add its explicit profile:

```bash
uv sync --locked --group dev --group ci-torch-cpu --inexact
```

CI itself uses smaller locked groups so unrelated jobs do not install the full
toolchain: `ci-test`, `ci-quality`, `ci-types`, `ci-security`, `ci-package`,
`ci-benchmark`, `ci-release`, and `ci-torch-cpu`. Tool groups never select a
runtime. Jobs that import AlbumentationsX add `ci-torch-cpu`; static jobs do
not. Contributors normally should keep using `dev`; the purpose-specific
groups are useful when reproducing one CI leaf, for example:

```bash
uv sync --locked --no-default-groups --group ci-test --group ci-torch-cpu --inexact
```

#### pip fallback

If you cannot use uv, create and activate a virtual environment manually, then install the project and development
requirements:

```bash
python3 -m venv env
source env/bin/activate
pip install -e .
pip install -r requirements-dev.txt
```

The fallback requirements also leave Torch to you. Install the CPU, CUDA, or
MPS Torch build before running the test suite.

On Windows, activate the environment with `env\Scripts\activate.bat` for cmd.exe or `env\Scripts\activate.ps1` for
PowerShell.

### 3. Set Up Pre-commit Hooks

Pre-commit hooks help maintain code quality by automatically checking your changes before each commit.

1. Set up the hooks:

```bash
uv run pre-commit install
```

If you used the pip fallback with an activated virtual environment, run:

```bash
pre-commit install
```

1. (Optional) Run hooks manually on all files:

```bash
uv run pre-commit run --all-files
```

With the pip fallback:

```bash
pre-commit run --all-files
```

## Verifying Your Setup

### Run Tests

Ensure everything is set up correctly by running the test suite:

```bash
uv run pytest
```

With the pip fallback:

```bash
pytest
```

For a faster local gate before handing work off, run:

```bash
pre-commit run --all-files --show-diff-on-failure
```

With the pip fallback:

```bash
pre-commit run --all-files --show-diff-on-failure
```

### Verification Infrastructure Commands

The maintenance verification layer adds focused commands for CI, release, and
support-policy changes:

```bash
uv run python -m tools.ci_matrix check
uv run python -m tools.ci_shard check
uv run python tools/verify_regression_vectors.py --all
uv run pytest -q tests/regression tests/property --hypothesis-profile=ci-fast
uv run python tools/generate_correctness_report.py \
  --allow-missing-evidence \
  --output _internal/correctness-report-dry-run.md
```

Golden regression vectors are updated only by an explicit command:

```bash
uv run python tools/generate_regression_vectors.py --transform HorizontalFlip --epoch 2.4
```

Local dry-run reports and other one-off evidence belong under `_internal/`.

### Common Issues and Solutions

#### Permission Errors

- **Linux/macOS**: If you encounter permission errors, try using `sudo` for system-wide installations or consider using `--user` flag with pip
- **Windows**: Run your terminal as administrator if you encounter permission issues

#### Virtual Environment Not Activating

- Ensure you're in the correct directory
- Check that Python is properly installed and in your system PATH
- Try creating the virtual environment with the full Python path

#### Import Errors After Installation

- Verify that you're using the correct virtual environment
- Confirm that all dependencies were installed successfully
- Try reinstalling the package in editable mode

## Next Steps

After setting up your environment:

1. Create a new branch for your work
2. Make your changes
3. Run tests and pre-commit hooks
4. Submit a pull request

For more detailed information about contributing, please refer to [Coding Guidelines](./coding_guidelines.md)

## Getting Help

If you encounter any issues with the setup:

1. Check our [Discord community](https://discord.gg/e6zHCXTvaN)
2. Open an [issue on GitHub](https://github.com/albumentations-team/AlbumentationsX/issues)
3. Review existing issues for similar problems and solutions
