# Set up a development environment

Use Python 3.10 or higher, Git, and [uv](https://docs.astral.sh/uv/). Fork the
[repository](https://github.com/albumentations-team/AlbumentationsX), then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/AlbumentationsX.git
cd AlbumentationsX
```

## Install dependencies

Install the project and development tools from the lockfile:

```bash
uv sync --locked --group dev --inexact
```

This installs Ruff, mypy, Pyrefly, pytest, pre-commit, and security tooling. Install the PyTorch build for your CPU,
CUDA, or MPS environment before importing AlbumentationsX. The development group leaves that runtime choice to you.

To use the CPU-only runtime profile used by CI:

```bash
uv sync --locked --group dev --group ci-torch-cpu --inexact
```

CI selects smaller groups for individual jobs. Use those groups when reproducing a specific job, for example:

```bash
uv sync --locked --no-default-groups --group ci-test --group ci-torch-cpu --inexact
```

The tool groups are `ci-test`, `ci-quality`, `ci-types`, `ci-security`, `ci-package`, `ci-benchmark`, and `ci-release`.
Import-capable jobs add `ci-torch-cpu`; static jobs do not need it.

### pip fallback

If uv is unavailable, create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
pip install -e .
pip install -r requirements-dev.txt
```

On Windows, activate with `env\Scripts\activate.bat` in cmd.exe or `env\Scripts\activate.ps1` in PowerShell.
Install the appropriate PyTorch build separately. In the commands below, omit `uv run` when using this activated
pip environment.

## Enable hooks and verify the environment

```bash
uv run pre-commit install
uv run python -c "import albumentations"
uv run pytest -n 4 -q tests/test_core_utils.py
uv run pre-commit run --all-files --show-diff-on-failure
```

The test command checks one module to confirm the environment works. For a change, select tests that can detect its
failure modes. Use `uv run pytest -n auto` when the change requires the full suite.

If imports fail, check which interpreter is running with `uv run python -c "import sys; print(sys.executable)"` and
confirm that the selected environment contains PyTorch and an OpenCV variant. Re-run the appropriate sync command
above if dependencies are missing. Installation should use the virtual environment without administrator privileges.

## Maintenance checks

For CI and support-policy changes:

```bash
uv run python -m tools.ci_matrix check
uv run python -m tools.ci_shard check
```

For regression contracts:

```bash
uv run python tools/verify_regression_vectors.py --all
uv run pytest -n 4 -q tests/regression tests/property --hypothesis-profile=ci-fast
```

Golden vectors change only through an explicit regeneration command, for example:

```bash
uv run python tools/generate_regression_vectors.py --transform HorizontalFlip --epoch 2.4
```

To inspect report generation without release evidence:

```bash
uv run python tools/generate_correctness_report.py \
  --allow-missing-evidence \
  --output _internal/correctness-report-dry-run.md
```

Keep local reports and other temporary artifacts under `_internal/`.

Follow the [Coding Guidelines](coding_guidelines.md) and [Contributing Guide](../../CONTRIBUTING.md) for your change.
For setup problems, check existing [issues](https://github.com/albumentations-team/AlbumentationsX/issues) or ask in
[Discord](https://discord.gg/e6zHCXTvaN).
