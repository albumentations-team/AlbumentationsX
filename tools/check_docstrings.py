"""Launcher so the check-google-docstrings hook runs the package's checker.

The pre-commit hook runs `python -m tools.check_docstrings` with pass_filenames: false.
With repo root as cwd, that would load this file and skip the package's checker.
When invoked as __main__ with no args, we run the package's checker in a subprocess:
use -I (isolate) so cwd is not on sys.path, and PYTHONPATH = site-packages + repo_root
so the package's tools is found and pyproject.toml is still in cwd.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__" and len(sys.argv) <= 1:
    repo_root = Path.cwd().resolve()
    # Paths that are not repo root (so package's tools is found first)
    rest_path = [Path(p).resolve() for p in sys.path if p and Path(p).resolve() != repo_root]
    # Put site-packages first so tools.check_docstrings loads from package; keep repo for pyproject
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(x) for x in [*rest_path, repo_root])
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-m", "tools.check_docstrings"],
        env=env,
        cwd=repo_root,
        check=False,
    )
    sys.exit(result.returncode)
