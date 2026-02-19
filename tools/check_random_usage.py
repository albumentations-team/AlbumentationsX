#!/usr/bin/env python3
"""Pre-commit hook to check that transforms don't use np.random or random module directly.

Enforces the rule: "NEVER use np.random or random module directly.
Use self.py_random or self.random_generator instead."

Allowed:
- np.random.default_rng() — creates a Generator object, used legitimately for seeded RNG forwarding
- random.Random() — constructor for self.py_random setup
- Type annotations: np.random.Generator
- Docstring examples (inside string literals — not parsed by AST)
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Module-level sampling calls that bypass the transform's RNG infrastructure.
# np.random.default_rng and random.Random are NOT banned — they're constructors
# used to set up self.random_generator / self.py_random and for seeded functional layer calls.
BANNED_CALLS = {
    "np.random.randint",
    "np.random.rand",
    "np.random.randn",
    "np.random.random",
    "np.random.choice",
    "np.random.shuffle",
    "np.random.permutation",
    "np.random.uniform",
    "np.random.normal",
    "np.random.seed",
    "np.random.RandomState",
    "random.randint",
    "random.random",
    "random.choice",
    "random.shuffle",
    "random.uniform",
    "random.seed",
    "random.sample",
    "random.randrange",
}


class RandomUsageChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        call_str = ast.unparse(node.func)
        if call_str in BANNED_CALLS:
            self.errors.append(
                (
                    node.lineno,
                    f"Direct use of '{call_str}' is forbidden. Use self.py_random or self.random_generator instead.",
                ),
            )
        self.generic_visit(node)


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
