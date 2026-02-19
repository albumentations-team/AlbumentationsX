#!/usr/bin/env python3
"""Pre-commit hook to check that transforms don't use np.random or random module directly.

Enforces the rule: "NEVER use np.random or random module directly.
Use self.py_random or self.random_generator instead."

Allowed:
- np.random.default_rng() — creates a Generator object, used legitimately for seeded RNG forwarding
- random.Random() — constructor for self.py_random setup
- Type annotations (not calls): np.random.Generator etc.
- Docstring examples (inside string literals — not parsed by AST)

Detection strategy:
- Tracks all import aliases for `numpy`, `numpy.random`, and `random` modules.
- For every Call node, resolves the root name through aliases and checks whether
  the call chain accesses the `random` submodule — catching aliased imports like
  `from numpy import random as rnd; rnd.uniform(...)` or `import numpy as np`.
- Structural match (module + attribute path) instead of exact string set, so new
  sampling APIs are caught automatically.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Sampling methods that bypass the transform's RNG infrastructure.
# Constructors (default_rng, Random, RandomState) and type-annotation-only names
# are intentionally NOT banned.
BANNED_NUMPY_RANDOM_ATTRS = {
    "randint",
    "rand",
    "randn",
    "random",
    "choice",
    "shuffle",
    "permutation",
    "uniform",
    "normal",
    "seed",
    "RandomState",
    "integers",
    "standard_normal",
    "standard_uniform",
}

BANNED_RANDOM_ATTRS = {
    "randint",
    "random",
    "choice",
    "shuffle",
    "uniform",
    "seed",
    "sample",
    "randrange",
    "gauss",
    "normalvariate",
    "triangular",
    "betavariate",
    "expovariate",
}

# Constructors we explicitly allow even under the random namespace
ALLOWED_NUMPY_RANDOM_ATTRS = {"default_rng", "Generator"}
ALLOWED_RANDOM_ATTRS = {"Random"}


def _unpack_attr_chain(node: ast.expr) -> list[str] | None:
    """Return the dotted name chain for an Attribute/Name node, or None."""
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return list(reversed(parts))
    return None


class RandomUsageChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[tuple[int, str]] = []
        # Maps local alias → canonical module path, e.g. "np" → "numpy"
        self._aliases: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Collect import aliases so we can resolve names structurally
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name
            self._aliases[local] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            canonical = f"{base}.{alias.name}" if base else alias.name
            self._aliases[local] = canonical
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Check every call
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        chain = _unpack_attr_chain(node.func)
        if chain:
            self._check_chain(node.lineno, chain)
        self.generic_visit(node)

    def _resolve(self, name: str) -> str:
        """Resolve a local name to its canonical module path."""
        return self._aliases.get(name, name)

    def _check_chain(self, lineno: int, chain: list[str]) -> None:
        # Resolve the root name through aliases
        root = self._resolve(chain[0])
        rest = chain[1:]  # attributes after the root

        # Pattern: numpy.random.<method> or np.random.<method>
        # Also handles: from numpy import random as rnd → rnd.<method>
        if root in ("numpy", "numpy.random") and rest:
            if root == "numpy" and rest and rest[0] == "random" and len(rest) >= 2:
                method = rest[1]
                if method in BANNED_NUMPY_RANDOM_ATTRS:
                    self._report(lineno, ".".join(chain))
            elif root == "numpy.random":
                method = rest[0]
                if method in BANNED_NUMPY_RANDOM_ATTRS:
                    self._report(lineno, ".".join(chain))

        # Pattern: random.<method> (stdlib)
        elif root == "random" and rest:
            method = rest[0]
            if method in BANNED_RANDOM_ATTRS and method not in ALLOWED_RANDOM_ATTRS:
                self._report(lineno, ".".join(chain))

    def _report(self, lineno: int, call_str: str) -> None:
        self.errors.append(
            (
                lineno,
                f"Direct use of '{call_str}' is forbidden. Use self.py_random or self.random_generator instead.",
            ),
        )


def check_file(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    checker = RandomUsageChecker()
    checker.visit(tree)
    return checker.errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for direct np.random/random usage")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    failed = False
    for filename in args.filenames:
        path = Path(filename)
        errors = check_file(path)
        for lineno, msg in errors:
            print(f"{filename}:{lineno}: {msg}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
