"""Individual AX guidance rules.

The checker intentionally uses only the standard library.  It analyses the
source tree as a package so that inheritance and import aliases do not depend
on optional runtime dependencies or on import-time side effects.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Iterable

from .diagnostics import Diagnostic
from .source_index import (
    COMPOSE_ROOTS,
    FRAMEWORK_ROOTS,
    TRANSFORM_ROOTS,
    ClassInfo,
    SourceIndex,
    argument_nodes,
    constructor_parameters,
    dotted,
)

MAX_APPLY_BODY_LINES = 20
LEGACY_RANDOM_TRANSFORM_NAMES = frozenset(
    {
        "RandomBrightnessContrast",
        "RandomCrop",
        "RandomCrop3D",
        "RandomCropFromBorders",
        "RandomCropNearBBox",
        "RandomFog",
        "RandomGamma",
        "RandomGravel",
        "RandomGridShuffle",
        "RandomOrder",
        "RandomRain",
        "RandomResizedCrop",
        "RandomRotate90",
        "RandomRotate90_3D",
        "RandomScale",
        "RandomShadow",
        "RandomSizedBBoxSafeCrop",
        "RandomSizedCrop",
        "RandomSnow",
        "RandomSunFlare",
        "RandomToneCurve",
    },
)
RANGE_ALIASES = frozenset({"AxisRanges3D", "MaskLengthRange", "PixelLengthRange", "PositiveAxisRanges3D"})
NUMPY_MATH_TO_MATH = {
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctan2": "atan2",
    "arctanh": "atanh",
    "cos": "cos",
    "cosh": "cosh",
    "deg2rad": "radians",
    "degrees": "degrees",
    "exp": "exp",
    "expm1": "expm1",
    "log": "log",
    "log10": "log10",
    "log1p": "log1p",
    "radians": "radians",
    "rad2deg": "degrees",
    "sin": "sin",
    "sinh": "sinh",
    "sqrt": "sqrt",
    "tan": "tan",
    "tanh": "tanh",
}


def _d(rule: str, info: ClassInfo, node: ast.AST, message: str, symbol: str = "") -> Diagnostic:
    return Diagnostic(
        rule,
        info.file.key,
        getattr(node, "lineno", 1),
        getattr(node, "col_offset", 0) + 1,
        symbol,
        message,
    )


def _unit_d(rule: str, unit, node: ast.AST, message: str, symbol: str = "") -> Diagnostic:
    return Diagnostic(rule, unit.key, getattr(node, "lineno", 1), getattr(node, "col_offset", 0) + 1, symbol, message)


def _function_defs(info: ClassInfo) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    return info.methods.values()


def _method_targets(info: ClassInfo) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    for name, node in info.methods.items():
        if name == "apply" or name.startswith("apply_to_"):
            yield name, node


def _is_field_classvar(annotation: ast.expr) -> bool:
    text = ast.unparse(annotation)
    return text.startswith("ClassVar[") or ".ClassVar[" in text


def _is_field_call_without_default(value: ast.expr) -> bool:
    if not isinstance(value, ast.Call) or dotted(value.func) not in {"Field", "pydantic.Field"}:
        return False
    return not value.args and all(keyword.arg not in {"default", "default_factory"} for keyword in value.keywords)


def _schema_nodes(index: SourceIndex) -> list[tuple[ClassInfo, ast.ClassDef]]:
    nodes: list[tuple[ClassInfo, ast.ClassDef]] = []
    for info in index.classes.values():
        nodes.extend((info, schema) for schema in info.nested_schemas)
        if info.name.endswith("InitSchema"):
            nodes.append((info, info.node))
    return nodes


def _schema_default_diagnostics(owner: ClassInfo, schema: ast.ClassDef) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for member in schema.body:
        if isinstance(member, ast.AnnAssign):
            target = member.target.id if isinstance(member.target, ast.Name) else "field"
            if target.startswith("_") or target == "model_config" or _is_field_classvar(member.annotation):
                continue
            if member.value is not None and not _is_field_call_without_default(member.value):
                result.append(
                    _d(
                        "AXG001",
                        owner,
                        member,
                        f"InitSchema field '{target}' must not define a default",
                        target,
                    )
                )
        elif isinstance(member, ast.Assign):
            result.extend(
                _d(
                    "AXG001",
                    owner,
                    member,
                    f"InitSchema field '{target.id}' must not define a default",
                    target.id,
                )
                for target in member.targets
                if isinstance(target, ast.Name) and not target.id.startswith("_") and target.id != "model_config"
            )
    return result


def rule_init_schema_defaults(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    seen: set[tuple[str, int]] = set()
    for owner, schema in _schema_nodes(index):
        key = (owner.file.key, schema.lineno)
        if key in seen:
            continue
        seen.add(key)
        result.extend(_schema_default_diagnostics(owner, schema))
    return result


def _defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterable[tuple[ast.arg, ast.expr]]:
    positional = [*node.args.posonlyargs, *node.args.args]
    if node.args.defaults:
        yield from zip(positional[-len(node.args.defaults) :], node.args.defaults, strict=True)
    for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        if value is not None:
            yield arg, value


def rule_apply_defaults(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.concrete_transforms():
        for name, node in _method_targets(info):
            for arg, value in _defaults(node):
                result.append(
                    _d(
                        "AXG002",
                        info,
                        arg,
                        f"{name} parameter '{arg.arg}' must not have a default ({ast.unparse(value)})",
                        f"{info.name}.{name}",
                    )
                )
    return result


def _docstring_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    if (
        not node.body
        or not isinstance(node.body[0], ast.Expr)
        or not isinstance(node.body[0].value, ast.Constant)
        or not isinstance(node.body[0].value.value, str)
    ):
        return set()
    return set(range(node.body[0].lineno, node.body[0].end_lineno + 1))


def rule_apply_length(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    ignored = {
        tokenize.COMMENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.NEWLINE,
        tokenize.NL,
    }
    for info in index.concrete_transforms():
        try:
            tokens = tokenize.generate_tokens(io.StringIO(info.file.source).readline)
            lines = {token.start[0] for token in tokens if token.type not in ignored}
        except (IndentationError, tokenize.TokenError):
            continue
        for name, node in _method_targets(info):
            if not node.body:
                continue
            start = node.body[0].lineno
            end = node.end_lineno or start
            count = sum(line in lines and line not in _docstring_lines(node) for line in range(start, end + 1))
            if count > MAX_APPLY_BODY_LINES:
                result.append(
                    _d(
                        "AXG003",
                        info,
                        node,
                        (
                            f"{name} has {count} code-bearing body lines; limit is {MAX_APPLY_BODY_LINES}; "
                            "move arithmetic to a functional helper"
                        ),
                        f"{info.name}.{name}",
                    )
                )
    return result


def rule_sampling_signature(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.concrete_transforms():
        node = info.methods.get("sample_parameters")
        if node is None:
            continue
        args = node.args
        names = [arg.arg for arg in [*args.posonlyargs, *args.args]]
        if names != ["self", "params", "data", "sampling"] or args.kwonlyargs or args.vararg or args.kwarg:
            result.append(
                _d(
                    "AXG004",
                    info,
                    node,
                    "sample_parameters must have exactly self, params, data, sampling",
                    f"{info.name}.sample_parameters",
                )
            )
            continue
        annotation = args.args[-1].annotation
        if annotation is None or ast.unparse(annotation).split(".")[-1] != "SamplingContext":
            result.append(
                _d(
                    "AXG004",
                    info,
                    args.args[-1],
                    "sampling must be annotated as SamplingContext",
                    f"{info.name}.sample_parameters",
                )
            )
    return result


def _chain(node: ast.expr) -> list[str] | None:
    chain = dotted(node)
    return chain.split(".") if chain else None


def rule_random_usage(index: SourceIndex) -> list[Diagnostic]:
    banned_numpy = {
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
    banned_random = {
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
    result: list[Diagnostic] = []
    for unit in index.units:
        aliases = unit.aliases
        for node in ast.walk(unit.tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _chain(node.func)
            if not chain:
                continue
            root = aliases.get(chain[0], chain[0])
            rest = chain[1:]
            method = None
            if root == "numpy" and rest[:1] == ["random"] and len(rest) > 1:
                method = rest[1]
            elif root == "numpy.random" and rest:
                method = rest[0]
            if method in banned_numpy:
                result.append(
                    _unit_d(
                        "AXG005",
                        unit,
                        node,
                        f"direct NumPy random call '{'.'.join(chain)}' is forbidden; use SamplingContext",
                        ".".join(chain),
                    )
                )
            elif root == "random" and rest and rest[0] in banned_random:
                result.append(
                    _unit_d(
                        "AXG005",
                        unit,
                        node,
                        f"direct stdlib random call '{'.'.join(chain)}' is forbidden; use SamplingContext",
                        ".".join(chain),
                    )
                )
    return result


def rule_sampling_rng(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.concrete_transforms():
        for name, method in _method_targets(info):
            for node in ast.walk(method):
                if not isinstance(node, ast.Call):
                    continue
                chain = _chain(node.func)
                if (
                    chain
                    and len(chain) >= 3
                    and chain[:2] == ["sampling", chain[1]]
                    and chain[1] in {"py_random", "random_generator"}
                ):
                    result.append(
                        _d(
                            "AXG006",
                            info,
                            node,
                            "draws must be made in sample_parameters and passed through params",
                            f"{info.name}.{name}",
                        )
                    )
    return result


def rule_transform_names(index: SourceIndex) -> list[Diagnostic]:
    roots = TRANSFORM_ROOTS | COMPOSE_ROOTS
    return [
        _d("AXG007", info, info.node, f"new transform class '{info.name}' must not use the 'Random' prefix", info.name)
        for info in index.classes.values()
        if index.is_descendant(info, roots)
        and info.file.key.startswith(
            ("albumentations/augmentations/", "albumentations/core/", "albumentations/pytorch/")
        )
        if info.name.startswith("Random") and info.name not in LEGACY_RANDOM_TRANSFORM_NAMES
    ]


def rule_fill_naming(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    public_transforms = {info.qname for info in index.concrete_transforms()}
    for info in index.classes.values():
        for node in _function_defs(info):
            for arg in argument_nodes(node):
                if arg.arg in {"fill_value", "fill_mask_value"}:
                    replacement = "fill" if arg.arg == "fill_value" else "fill_mask"
                    result.append(
                        _d(
                            "AXG008",
                            info,
                            arg,
                            f"parameter '{arg.arg}' should be '{replacement}'",
                            f"{info.name}.{node.name}",
                        )
                    )
                if (
                    info.qname in public_transforms
                    and node.name == "__init__"
                    and info.name not in FRAMEWORK_ROOTS
                    and arg.arg
                    in {"cval", "fill_color", "pad_value", "pad_cval", "fill_mask_color", "fill_mask_cval", "pad_mode"}
                ):
                    result.append(
                        _d(
                            "AXG008",
                            info,
                            arg,
                            f"public transform constructor parameter '{arg.arg}' is not an AX name",
                            f"{info.name}.__init__",
                        )
                    )
    return result


def _slice_items(node: ast.expr) -> tuple[ast.expr, ...]:
    return tuple(node.elts) if isinstance(node, ast.Tuple) else (node,)


def _annotation_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _range_annotation(node: ast.expr | None) -> bool:
    valid = False
    if node is not None:
        if isinstance(node, ast.Subscript) and _annotation_name(node.value) == "Annotated":
            valid = _range_annotation(_slice_items(node.slice)[0])
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            valid = _range_annotation(node.left) and _range_annotation(node.right)
        elif isinstance(node, ast.Constant) and node.value is None:
            valid = True
        elif isinstance(node, ast.Name):
            valid = node.id in RANGE_ALIASES
        elif isinstance(node, ast.Subscript):
            base = _annotation_name(node.value)
            items = _slice_items(node.slice)
            if base in {"tuple", "Tuple"}:
                valid = len(items) == 2
            elif base in {"dict", "Dict"}:
                valid = len(items) == 2 and _range_annotation(items[1])
    return valid


def rule_range_annotations(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.concrete_transforms():
        node = info.methods.get("__init__")
        if node is None:
            continue
        result.extend(
            _d(
                "AXG009",
                info,
                arg,
                f"`{arg.arg}` must describe a pair or axis-to-pair map",
                f"{info.name}.__init__",
            )
            for arg in constructor_parameters(node)
            if arg.arg.endswith("_range") and not _range_annotation(arg.annotation)
        )
    return result


def rule_removed_sampling(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.classes.values():
        result.extend(
            _d(
                "AXG010",
                info,
                info.methods[name],
                f"{info.name}.{name} was removed; implement sample_parameters(params, data, sampling) instead",
                f"{info.name}.{name}",
            )
            for name in ("get_params", "get_params_dependent_on_data")
            if name in info.methods
        )
    return result


def rule_serialization_override(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.classes.values():
        node = info.methods.get("get_transform_init_args_names")
        if node is not None and not (
            info.file.key == "albumentations/core/transforms_interface.py" and info.name == "BasicTransform"
        ):
            result.append(
                _d(
                    "AXG011",
                    info,
                    node,
                    "get_transform_init_args_names() is owned by BasicTransform; do not override it",
                    f"{info.name}.get_transform_init_args_names",
                )
            )
    return result


CV2_ALLOWLIST = {
    ("albumentations/augmentations/geometric/_functional_distortion.py", "upscale_distortion_maps", "resize"): 2,
}
CV2_FORBIDDEN = frozenset({"resize", "warpAffine", "warpPerspective", "copyMakeBorder", "remap"})


def _cv2_calls(unit) -> list[tuple[ast.Call, tuple[str, str, str]]]:
    calls: list[tuple[ast.Call, tuple[str, str, str]]] = []
    for function in (node for node in ast.walk(unit.tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            chain = _chain(node.func)
            if not chain:
                continue
            canonical = unit.aliases.get(chain[0], chain[0])
            is_cv2 = chain[0] == "cv2" or canonical.endswith(".cv2") or canonical == "cv2"
            if is_cv2 and chain[-1] in CV2_FORBIDDEN:
                calls.append((node, (unit.key, function.name, chain[-1])))
    return calls


def _cv2_diagnostics(index: SourceIndex) -> tuple[list[Diagnostic], dict[tuple[str, str, str], int]]:
    diagnostics: list[Diagnostic] = []
    counts: dict[tuple[str, str, str], int] = {}
    for unit in index.units:
        for node, key in _cv2_calls(unit):
            counts[key] = counts.get(key, 0) + 1
            if key not in CV2_ALLOWLIST:
                chain = _chain(node.func) or ["cv2", key[-1]]
                diagnostics.append(
                    _unit_d(
                        "AXG012",
                        unit,
                        node,
                        f"cv2.{chain[-1]} is forbidden here; use the albucore equivalent",
                        ".".join(chain),
                    )
                )
    return diagnostics, counts


def rule_cv2(index: SourceIndex) -> list[Diagnostic]:
    result, counts = _cv2_diagnostics(index)
    for key, expected in CV2_ALLOWLIST.items():
        actual = counts.get(key, 0)
        if actual != expected:
            path, function, operation = key
            unit = next((item for item in index.units if item.key == path), None)
            node = (
                next((n for n in ast.walk(unit.tree) if isinstance(n, ast.FunctionDef) and n.name == function), None)
                if unit
                else None
            )
            if unit and node:
                result.append(
                    _unit_d(
                        "AXG019",
                        unit,
                        node,
                        f"cv2.{operation} allowlist expects {expected} call(s) in {function}, found {actual}",
                        function,
                    )
                )
    return result


def rule_bbox_defaults(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.classes.values():
        for node in _function_defs(info):
            for arg, _value in _defaults(node):
                if arg.arg != "bbox_type":
                    continue
                public_boundary = (
                    info.file.key == "albumentations/core/bbox_utils.py"
                    and info.name == "BboxParams"
                    and node.name == "__init__"
                )
                if not public_boundary:
                    result.append(
                        _d(
                            "AXG013",
                            info,
                            arg,
                            "bbox_type must be passed explicitly below the public BboxParams boundary",
                            f"{info.name}.{node.name}",
                        )
                    )
    return result


def rule_integrity(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    roots = TRANSFORM_ROOTS | COMPOSE_ROOTS
    random_classes = {
        info.name
        for info in index.classes.values()
        if index.is_descendant(info, roots) and info.name.startswith("Random")
    }
    if random_classes != LEGACY_RANDOM_TRANSFORM_NAMES:
        info = next((candidate for candidate in index.classes.values() if candidate.name.startswith("Random")), None)
        if info is not None:
            result.append(
                _d(
                    "AXG019",
                    info,
                    info.node,
                    "legacy Random* transform allowlist changed; update the explicit compatibility decision",
                    info.name,
                )
            )
    boundary = next(
        (
            info
            for info in index.classes.values()
            if info.file.key == "albumentations/core/bbox_utils.py" and info.name == "BboxParams"
        ),
        None,
    )
    boundary_init = boundary.methods.get("__init__") if boundary else None
    boundary_default = (
        next((value for arg, value in _defaults(boundary_init) if arg.arg == "bbox_type"), None)
        if boundary_init
        else None
    )
    if boundary is not None and not (isinstance(boundary_default, ast.Constant) and boundary_default.value == "hbb"):
        result.append(
            _d(
                "AXG019",
                boundary,
                boundary_init or boundary.node,
                'public BboxParams.__init__ must retain bbox_type="hbb" as the compatibility boundary',
                "BboxParams.__init__",
            )
        )
    return result


NUMPY_MATH_NAMES = frozenset(NUMPY_MATH_TO_MATH)


def _scalar_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            if ast.unparse(node.annotation) in {"float", "int"}:
                names.add(node.arg)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if ast.unparse(node.annotation) in {"float", "int"}:
                names.add(node.target.id)
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
            and not isinstance(node.value.value, bool)
        ):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return names


def _numpy_math_operation(unit, node: ast.Call) -> tuple[str, str] | None:
    chain = dotted(node.func)
    if not chain:
        return None
    parts = chain.split(".")
    canonical = unit.aliases.get(parts[0], parts[0])
    if canonical.startswith("numpy."):
        operation = canonical.rsplit(".", 1)[-1]
    elif canonical == "numpy" and len(parts) >= 2:
        operation = parts[1]
    else:
        return None
    return (operation, chain) if operation in NUMPY_MATH_NAMES else None


def _scalar_expression(node: ast.expr, names: set[str]) -> bool:
    return (
        isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    ) or (isinstance(node, ast.Name) and node.id in names)


def rule_scalar_numpy_math(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for unit in index.units:
        names = _scalar_names(unit.tree)
        for node in (candidate for candidate in ast.walk(unit.tree) if isinstance(candidate, ast.Call)):
            operation_info = _numpy_math_operation(unit, node)
            if operation_info is None or not node.args or not all(_scalar_expression(arg, names) for arg in node.args):
                continue
            operation, chain = operation_info
            result.append(
                _unit_d(
                    "AXG014",
                    unit,
                    node,
                    (
                        f"'numpy.{operation}' receives Python scalar values; use "
                        f"'math.{NUMPY_MATH_TO_MATH[operation]}' instead"
                    ),
                    chain,
                )
            )
    return result


def rule_docstring_format(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for unit in index.units:
        for node in ast.walk(unit.tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            if "---" in doc:
                result.append(
                    _unit_d(
                        "AXG015",
                        unit,
                        node,
                        "docstrings must not contain Markdown horizontal-rule syntax",
                        getattr(node, "name", ""),
                    )
                )
            if "``" in doc and "```" not in doc:
                result.append(
                    _unit_d(
                        "AXG015",
                        unit,
                        node,
                        "docstrings must not use double-backtick markup",
                        getattr(node, "name", ""),
                    )
                )
    return result


def _has_examples(doc: str | None) -> str | None:
    if not doc:
        return "missing 'Examples' section"
    if re.search(r"^\s*Example:\s*$", doc, re.MULTILINE) and not re.search(r"^\s*Examples:\s*$", doc, re.MULTILINE):
        return "use 'Examples:' (plural) instead of 'Example:'"
    if not re.search(r"^\s*Examples:\s*$", doc, re.MULTILINE):
        return "missing 'Examples:' section"
    return None


def rule_examples(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    target_roots = frozenset({"DualTransform", "ImageOnlyTransform", "Transform3D", "VolumeOnlyTransform"})
    for info in index.concrete_transforms():
        if not index.is_descendant(info, target_roots):
            continue
        message = _has_examples(ast.get_docstring(info.node, clean=False))
        if message:
            result.append(_d("AXG016", info, info.node, message, info.name))
    return result


OPTIONAL_METHOD_DOCS = frozenset(
    {
        "apply",
        "apply_to_images",
        "apply_to_volume",
        "apply_to_mask",
        "apply_to_masks",
        "apply_to_mask3d",
        "apply_to_bboxes",
        "apply_to_keypoints",
        "sample_parameters",
        "to_dict_private",
        "targets_as_params",
        "get_transform_init_args",
    }
)


def _property_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (isinstance(dec, ast.Name) and dec.id == "property")
        or (isinstance(dec, ast.Attribute) and dec.attr in {"setter", "deleter"})
        for dec in node.decorator_list
    )


def rule_method_docstrings(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    selected = list(index.concrete_transforms()) + [
        info
        for info in index.classes.values()
        if info.file.key.startswith("albumentations/core/") and info.name.startswith("Base")
    ]
    for info in selected:
        for node in _function_defs(info):
            if node.name.startswith("_") or node.name in OPTIONAL_METHOD_DOCS or _property_method(node):
                continue
            args = [*node.args.posonlyargs, *node.args.args]
            if not args or args[0].arg not in {"self", "cls"} or ast.get_docstring(node):
                continue
            result.append(
                _d(
                    "AXG017",
                    info,
                    node,
                    "public instance/class methods must have docstrings",
                    f"{info.name}.{node.name}",
                )
            )
    return result


def rule_apply_docstrings(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.concrete_transforms():
        for name, node in _method_targets(info):
            if ast.get_docstring(node):
                result.append(
                    _d(
                        "AXG018",
                        info,
                        node,
                        (
                            "concrete apply methods must not have docstrings; document the transform and "
                            "functional helper instead"
                        ),
                        f"{info.name}.{name}",
                    )
                )
    return result


def _schema_fields(index: SourceIndex, schema: ast.ClassDef, unit) -> set[str]:
    fields: set[str] = set()
    for member in schema.body:
        if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            fields.add(member.target.id)
        elif isinstance(member, ast.Assign):
            fields.update(target.id for target in member.targets if isinstance(target, ast.Name))
    for base in schema.bases:
        name = index.resolve_name(unit, dotted(base) or "")
        base_info = index.classes.get(name)
        if base_info is not None and base_info.name.endswith("InitSchema"):
            fields.update(_schema_fields(index, base_info.node, base_info.file))
    return fields


def _own_schema(index: SourceIndex, info: ClassInfo) -> ast.ClassDef | None:
    if info.nested_schemas:
        return info.nested_schemas[0]
    for member in info.node.body:
        if not isinstance(member, (ast.Assign, ast.AnnAssign)):
            continue
        targets = member.targets if isinstance(member, ast.Assign) else (member.target,)
        if not any(isinstance(target, ast.Name) and target.id == "InitSchema" for target in targets):
            continue
        value = member.value
        if isinstance(value, ast.Name):
            candidate = index.classes.get(f"{index._module_name(info.file.key)}.{value.id}")
            if candidate is not None:
                return candidate.node
    expected = f"{info.name}InitSchema"
    for candidate in index.classes.values():
        if candidate.file is info.file and candidate.name == expected:
            return candidate.node
    return None


def _effective_init(index: SourceIndex, info: ClassInfo) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if "__init__" in info.methods:
        return info.methods["__init__"]
    for parent in index.ancestors(info):
        if "__init__" in parent.methods:
            return parent.methods["__init__"]
    return None


def _normalized_annotation(annotation: ast.expr | None) -> str | None:
    return ast.dump(annotation, annotate_fields=False) if annotation is not None else None


def _forwarded_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if (
            not isinstance(child, ast.Call)
            or not isinstance(child.func, ast.Attribute)
            or child.func.attr != "__init__"
        ):
            continue
        if (
            not isinstance(child.func.value, ast.Call)
            or not isinstance(child.func.value.func, ast.Name)
            or child.func.value.func.id != "super"
        ):
            continue
        names.update(keyword.arg for keyword in child.keywords if keyword.arg is not None)
    return names


def rule_constructor_schema(index: SourceIndex) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for info in index.concrete_transforms():
        own_init = info.methods.get("__init__")
        if own_init is None:
            continue
        parent = next(
            (
                candidate
                for candidate in index.ancestors(info)
                if index.is_descendant(candidate, TRANSFORM_ROOTS) or candidate.name in TRANSFORM_ROOTS
            ),
            None,
        )
        parent_init = _effective_init(index, parent) if parent else None
        own_params = constructor_parameters(own_init)
        own_by_name = {arg.arg: arg for arg in own_params}
        parent_by_name = {arg.arg: arg for arg in constructor_parameters(parent_init)} if parent_init else {}
        changed = {
            name
            for name, arg in own_by_name.items()
            if name not in parent_by_name
            or _normalized_annotation(arg.annotation) != _normalized_annotation(parent_by_name[name].annotation)
        }
        schema = _own_schema(index, info)
        if schema is None:
            # A constructor that only repeats inherited inputs is valid when it explicitly forwards them.
            if changed - {"p", "strict"}:
                result.append(
                    _d(
                        "AXG020",
                        info,
                        own_init,
                        (
                            f"constructor adds or changes parameters ({', '.join(sorted(changed))}); "
                            "define a non-empty InitSchema"
                        ),
                        f"{info.name}.__init__",
                    )
                )
            elif parent_by_name and not set(parent_by_name).issubset(_forwarded_names(own_init)):
                missing = sorted(set(parent_by_name) - _forwarded_names(own_init))
                result.append(
                    _d(
                        "AXG020",
                        info,
                        own_init,
                        f"constructor must explicitly forward inherited parameters: {', '.join(missing)}",
                        f"{info.name}.__init__",
                    )
                )
            continue
        fields = _schema_fields(index, schema, info.file)
        missing = sorted(name for name in own_by_name if name not in {"self", "p", "strict"} and name not in fields)
        if missing:
            result.append(
                _d(
                    "AXG020",
                    info,
                    schema,
                    f"InitSchema must declare constructor parameters: {', '.join(missing)}",
                    f"{info.name}.InitSchema",
                )
            )
    return result


def run_all(index: SourceIndex) -> list[Diagnostic]:
    checks = (
        rule_init_schema_defaults,
        rule_apply_defaults,
        rule_apply_length,
        rule_sampling_signature,
        rule_random_usage,
        rule_sampling_rng,
        rule_transform_names,
        rule_fill_naming,
        rule_range_annotations,
        rule_removed_sampling,
        rule_serialization_override,
        rule_cv2,
        rule_bbox_defaults,
        rule_integrity,
        rule_scalar_numpy_math,
        rule_docstring_format,
        rule_examples,
        rule_method_docstrings,
        rule_apply_docstrings,
        rule_constructor_schema,
    )
    diagnostics: list[Diagnostic] = []
    for check in checks:
        diagnostics.extend(check(index))
    return sorted(diagnostics)
