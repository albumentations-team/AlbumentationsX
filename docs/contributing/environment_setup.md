# Set up a development environment

Use Python 3.11 or higher, Git, and [uv](https://docs.astral.sh/uv/). Fork the
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
Import-capable jobs and runtime API type checks add `ci-torch-cpu`. Link-only, formatting, packaging, and audit jobs
can use the tool group alone.

### pip fallback

If uv is unavailable, create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate with `env\Scripts\activate.bat` in cmd.exe or `env\Scripts\activate.ps1` in PowerShell.
Use pip 25.1 or higher for dependency-group support. Install the appropriate PyTorch build, then install the project
and development tools from the groups in `pyproject.toml`:

```bash
python -m pip install -e . --group dev
```

This uses the same dependency declarations as uv. Pip resolves allowed versions independently; use the uv commands
above when reproducing the locked CI environment. In the commands below, omit `uv run` when using this activated
pip environment.

Lockfile checks and the complete pre-commit suite require uv. A pip-only environment can run tests and the mypy and
Pyrefly hooks through its active Python interpreter.

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

## Check types against installed dependencies

Run both checkers in the project environment so they see the installed NumPy, OpenCV, Albucore, NumKong and Torch
types. Their Python target follows that environment; Ruff and the runtime matrix continue to enforce Python 3.11
compatibility. An isolated checker without backend stubs can silently treat array operations as `Any`.

```bash
uv run mypy albumentations
uv run pyrefly check
uv run python -m unittest discover -s tests/typing -p check_contracts.py
```

The last command verifies that valid container/output calls pass and incorrect return types, destinations and Tensor
layouts fail in both checkers. CI runs it in `ci-types` with `ci-torch-cpu`; it does not require the full pytest profile.

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

Follow the [Coding Guidelines](coding_guidelines.md) and
[Contributing Guide](https://github.com/albumentations-team/AlbumentationsX/blob/main/CONTRIBUTING.md) for your change.
For setup problems, check existing [issues](https://github.com/albumentations-team/AlbumentationsX/issues) or ask in
[Discord](https://discord.gg/e6zHCXTvaN).
