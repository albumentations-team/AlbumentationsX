"""Launcher so the check-google-docstrings hook runs the package's checker.

The pre-commit hook runs `python -m tools.check_docstrings` with pass_filenames: false.
With repo root as cwd, that would load this file and skip the package's checker.
When invoked as __main__ with no args, we run the package's checker in a subprocess.

We use -I (isolated) so the child ignores PYTHONPATH and uses default sys.path; the child
then loads tools.check_docstrings from site-packages (the package's checker), avoiding a
second fork. Without -I, the child can load this repo's launcher again and fork once more,
which can trigger BlockingIOError (EAGAIN) on some systems. Tradeoff: the child does not
see repo_root on sys.path; if the checker ever needs to import the repo's albumentations,
we would need a different approach (e.g. pass repo root via env and run checker in-process).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.stderr.write(
            "check_docstrings.py does not accept positional arguments; "
            "invoke the package-level docstring checker directly instead.\n",
        )
        sys.exit(1)
    repo_root = Path.cwd().resolve()
    result = subprocess.run(
        [sys.executable, "-I", "-m", "tools.check_docstrings"],
        cwd=repo_root,
        check=False,
    )
    sys.exit(result.returncode)
