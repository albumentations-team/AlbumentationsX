"""Parse and resolve the production source tree without importing it."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

TRANSFORM_ROOTS = frozenset(
    {"BasicTransform", "ImageOnlyTransform", "DualTransform", "Transform3D", "VolumeOnlyTransform"}
)
COMPOSE_ROOTS = frozenset({"BaseCompose"})
FRAMEWORK_ROOTS = TRANSFORM_ROOTS | COMPOSE_ROOTS


def dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def argument_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    return (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)


def constructor_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    return (*node.args.posonlyargs, *node.args.args[1:], *node.args.kwonlyargs)


@dataclass
class FileUnit:
    key: str
    source: str
    tree: ast.Module
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class ClassInfo:
    file: FileUnit
    node: ast.ClassDef
    qname: str
    bases: tuple[str, ...]
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    nested_schemas: tuple[ast.ClassDef, ...]

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def is_framework_root(self) -> bool:
        return self.name in FRAMEWORK_ROOTS


class SourceIndex:
    """One parse and a conservative package-wide symbol index for all rules."""

    def __init__(self, units: Iterable[FileUnit], *, complete: bool = False) -> None:
        self.units = tuple(units)
        self.complete = complete
        self.classes: dict[str, ClassInfo] = {}
        self.simple_classes: dict[str, list[str]] = {}
        for unit in self.units:
            self._collect_aliases(unit)
            self._collect_classes(unit)
        for info in self.classes.values():
            info.bases = tuple(self.resolve_name(info.file, base) for base in info.bases)

    @classmethod
    def from_repo(cls, root: Path) -> SourceIndex:
        package = root / "albumentations"
        units: list[FileUnit] = []
        for path in sorted(package.rglob("*.py")):
            if path.suffix != ".py" or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            try:
                key = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                key = path.as_posix()
            units.append(FileUnit(key, text, tree))
        return cls(units, complete=True)

    @classmethod
    def from_sources(cls, sources: dict[str, str]) -> SourceIndex:
        units: list[FileUnit] = []
        for key, text in sorted(sources.items()):
            path = Path(key)
            units.append(FileUnit(path.as_posix(), text, ast.parse(text, filename=key)))
        return cls(units)

    @staticmethod
    def _module_name(key: str) -> str:
        if key.endswith("/__init__.py"):
            return key[: -len("/__init__.py")].replace("/", ".")
        if key.endswith(".py"):
            return key[:-3].replace("/", ".")
        return key.replace("/", ".")

    def _collect_aliases(self, unit: FileUnit) -> None:
        module = self._module_name(unit.key)
        for node in unit.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    unit.aliases[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    prefix = module.rsplit(".", node.level)[0] if "." in module else ""
                    base = f"{prefix}.{base}" if base else prefix
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    unit.aliases[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name

    def _collect_classes(self, unit: FileUnit) -> None:
        module = self._module_name(unit.key)

        def visit(body: list[ast.stmt], prefix: str) -> None:
            for node in body:
                if not isinstance(node, ast.ClassDef):
                    continue
                qname = f"{prefix}.{node.name}"
                methods = {
                    member.name: member
                    for member in node.body
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                schemas = tuple(
                    member for member in node.body if isinstance(member, ast.ClassDef) and member.name == "InitSchema"
                )
                bases = tuple(dotted(base) or "" for base in node.bases)
                info = ClassInfo(unit, node, qname, bases, methods, schemas)
                self.classes[qname] = info
                self.simple_classes.setdefault(node.name, []).append(qname)
                visit(node.body, qname)

        visit(unit.tree.body, module)

    def resolve_name(self, unit: FileUnit, name: str) -> str:
        if not name:
            return name
        parts = name.split(".")
        root = unit.aliases.get(parts[0], parts[0])
        candidate = ".".join([root, *parts[1:]]) if parts[1:] else root
        if candidate in self.classes:
            return candidate
        local = f"{self._module_name(unit.key)}.{name}"
        if local in self.classes:
            return local
        matches = self.simple_classes.get(parts[-1], ())
        if len(matches) == 1:
            return matches[0]
        return candidate

    def ancestors(self, info: ClassInfo) -> tuple[ClassInfo, ...]:
        result: list[ClassInfo] = []
        seen: set[str] = set()

        def visit(qname: str) -> None:
            if qname in seen:
                return
            seen.add(qname)
            parent = self.classes.get(qname)
            if parent is None:
                return
            result.append(parent)
            for base in parent.bases:
                visit(base)

        for base in info.bases:
            visit(base)
        return tuple(result)

    def is_descendant(self, info: ClassInfo, roots: frozenset[str]) -> bool:
        return any(parent.name in roots or parent.qname in roots for parent in self.ancestors(info))

    def concrete_transforms(self) -> tuple[ClassInfo, ...]:
        return tuple(
            info
            for info in self.classes.values()
            if self.is_descendant(info, TRANSFORM_ROOTS)
            and not info.is_framework_root
            and not info.name.startswith("_")
            and not info.name.startswith("Base")
            and info.file.key.startswith(("albumentations/augmentations/", "albumentations/pytorch/"))
        )
