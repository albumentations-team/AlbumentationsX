"""Keep transform ``apply*`` methods as thin dispatchers.

The hook limits code-bearing lines in class methods whose names start with
``apply``. Signatures, blank lines, standalone comments, and docstrings do not
count. Existing over-limit methods are hash-baselined; changing one requires
refactoring it below the limit instead of extending it further.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
import tokenize
from collections.abc import Iterable, Mapping
from pathlib import Path

MAX_APPLY_BODY_LINES = 15
BASELINE_PATH = Path(__file__).with_name("apply_method_length_baseline.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORED_TOKEN_TYPES = {
    tokenize.COMMENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}


def _is_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Expr | None:
    """Return the method docstring node when one exists."""
    if not node.body:
        return None

    candidate = node.body[0]
    if (
        isinstance(candidate, ast.Expr)
        and isinstance(candidate.value, ast.Constant)
        and isinstance(candidate.value.value, str)
    ):
        return candidate
    return None


def _docstring_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    """Return physical lines occupied by a method docstring."""
    docstring = _is_docstring(node)
    if docstring is None:
        return set()
    return set(range(docstring.lineno, docstring.end_lineno + 1))


def _iter_apply_methods(tree: ast.AST) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield direct class methods whose names start with ``apply``."""
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("apply"):
                yield f"{class_node.name}.{item.name}", item


def _code_tokens_by_line(source: str) -> dict[int, list[tokenize.TokenInfo]]:
    """Return non-comment Python tokens grouped by their starting line."""
    tokens_by_line: dict[int, list[tokenize.TokenInfo]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in IGNORED_TOKEN_TYPES:
            continue
        tokens_by_line.setdefault(token.start[0], []).append(token)
    return tokens_by_line


def _body_line_numbers(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    tokens_by_line: Mapping[int, list[tokenize.TokenInfo]],
) -> tuple[int, ...]:
    """Return code-bearing physical lines in a method body.

    The method signature is outside the range. A line with code followed by an
    inline comment still counts, while standalone comments and docstrings do
    not.
    """
    if not node.body:
        return ()

    first_body_line = node.body[0].lineno
    docstring_lines = _docstring_lines(node)
    end_line = node.end_lineno or first_body_line
    return tuple(
        line_number
        for line_number in range(first_body_line, end_line + 1)
        if line_number not in docstring_lines and line_number in tokens_by_line
    )


def _method_fingerprint(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    tokens_by_line: Mapping[int, list[tokenize.TokenInfo]],
) -> str:
    """Return a comment- and docstring-insensitive fingerprint for a method body."""
    docstring_lines = _docstring_lines(node)
    first_body_line = node.body[0].lineno if node.body else node.lineno
    end_line = node.end_lineno or first_body_line
    canonical_tokens = [
        f"{token.type}:{token.string}"
        for line_number in range(first_body_line, end_line + 1)
        if line_number not in docstring_lines
        for token in tokens_by_line.get(line_number, [])
    ]
    return hashlib.sha256("\n".join(canonical_tokens).encode()).hexdigest()


def _path_key(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_baseline() -> dict[str, str]:
    """Load fingerprints for legacy methods that must shrink when changed."""
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    is_string_mapping = isinstance(data, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    )
    if not is_string_mapping:
        msg = f"{BASELINE_PATH} must contain a JSON object of method fingerprints"
        raise ValueError(msg)
    return data


def collect_errors(paths: Iterable[Path], baseline: Mapping[str, str] | None = None) -> list[str]:
    """Return line-limit violations for the supplied Python files."""
    baseline = _load_baseline() if baseline is None else baseline
    errors: list[str] = []

    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue

        tokens_by_line = _code_tokens_by_line(source)
        for qualified_name, node in _iter_apply_methods(tree):
            body_lines = _body_line_numbers(node, tokens_by_line)
            if len(body_lines) <= MAX_APPLY_BODY_LINES:
                continue

            key = f"{_path_key(path)}:{qualified_name}"
            if baseline.get(key) == _method_fingerprint(node, tokens_by_line):
                continue

            errors.append(
                f"{path}:{node.lineno}: {qualified_name} has {len(body_lines)} code-bearing body lines; "
                f"limit is {MAX_APPLY_BODY_LINES}. Move image arithmetic and routing into a functional helper.",
            )

    return errors


def main() -> int:
    """Run the apply-method length pre-commit check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    errors = collect_errors(Path(filename) for filename in args.filenames)
    if errors:
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
