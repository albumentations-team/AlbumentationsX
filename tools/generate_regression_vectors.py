"""Generate golden regression vectors for selected transform contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

import albumentations
from tests.helpers import TestDataFactory
from tests.regression.transform_contracts import REGRESSION_CONTRACTS, TransformContract, contract_by_name

REPO_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_ROOT = REPO_ROOT / "tests" / "files" / "regression"
MANIFEST_PATH = REGRESSION_ROOT / "manifest.json"
VECTOR_DIR = REGRESSION_ROOT / "v1"

INPUT_RECIPES: dict[str, dict[str, Any]] = {
    "image_mask": {
        "image_shape": [8, 10, 3],
        "image_dtype": "uint8",
        "mask_shape": [8, 10],
        "mask_dtype": "uint8",
    },
    "hbb_keypoints": {
        "image_shape": [8, 10, 3],
        "image_dtype": "uint8",
        "bboxes": [[1.0, 1.0, 6.0, 5.0], [3.0, 2.0, 9.0, 7.0]],
        "bbox_labels": [3, 7],
        "keypoints": [[2.0, 3.0], [8.0, 6.0]],
        "keypoint_labels": [11, 13],
    },
    "volume_mask3d": {
        "volume_shape": [6, 8, 10, 1],
        "volume_dtype": "uint8",
        "mask3d_shape": [6, 8, 10],
        "mask3d_dtype": "uint8",
    },
}


def _input_recipe_for_contract(contract: TransformContract) -> dict[str, Any]:
    recipe = dict(INPUT_RECIPES[contract.input_recipe])
    recipe["name"] = contract.input_recipe
    recipe["seed"] = contract.seed
    return recipe


def _recipe_shape(recipe: dict[str, Any], key: str) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in recipe[key])


def _recipe_dtype(recipe: dict[str, Any], key: str) -> type[np.generic]:
    return np.dtype(str(recipe[key])).type


def _as_output_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _array_metadata(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _array_digest(array),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "sum": float(array.sum()) if array.size else 0.0,
    }


def _environment_metadata() -> dict[str, str | None]:
    try:
        import cv2
    except ImportError:
        opencv_version = None
    else:
        opencv_version = cv2.__version__

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": opencv_version,
        "albumentations": albumentations.__version__,
    }


def _build_image_mask_data(recipe: dict[str, Any]) -> dict[str, Any]:
    seed = int(recipe["seed"])
    mask = TestDataFactory.create_mask(_recipe_shape(recipe, "mask_shape"), seed=seed + 1)
    return {
        "image": TestDataFactory.create_image(
            _recipe_shape(recipe, "image_shape"),
            dtype=_recipe_dtype(recipe, "image_dtype"),
            seed=seed,
        ),
        "mask": mask.astype(_recipe_dtype(recipe, "mask_dtype"), copy=False),
    }


def _build_hbb_keypoint_data(recipe: dict[str, Any]) -> dict[str, Any]:
    seed = int(recipe["seed"])
    return {
        "bbox_labels": list(recipe["bbox_labels"]),
        "bboxes": np.asarray(recipe["bboxes"], dtype=np.float32),
        "image": TestDataFactory.create_image(
            _recipe_shape(recipe, "image_shape"),
            dtype=_recipe_dtype(recipe, "image_dtype"),
            seed=seed,
        ),
        "keypoint_labels": list(recipe["keypoint_labels"]),
        "keypoints": np.asarray(recipe["keypoints"], dtype=np.float32),
    }


def _build_volume_mask3d_data(recipe: dict[str, Any]) -> dict[str, Any]:
    seed = int(recipe["seed"])
    rng = np.random.default_rng(seed)
    volume_shape = _recipe_shape(recipe, "volume_shape")
    mask_shape = _recipe_shape(recipe, "mask3d_shape")
    volume = rng.integers(0, 256, volume_shape, dtype=_recipe_dtype(recipe, "volume_dtype"))
    mask3d = rng.integers(0, 4, mask_shape, dtype=_recipe_dtype(recipe, "mask3d_dtype"))
    return {"mask3d": mask3d, "volume": volume}


def _base_data(contract: TransformContract) -> dict[str, Any]:
    recipe = _input_recipe_for_contract(contract)
    if contract.input_recipe == "image_mask":
        return _build_image_mask_data(recipe)
    if contract.input_recipe == "hbb_keypoints":
        return _build_hbb_keypoint_data(recipe)
    if contract.input_recipe == "volume_mask3d":
        return _build_volume_mask3d_data(recipe)
    msg = f"Unsupported regression input recipe: {contract.input_recipe}"
    raise ValueError(msg)


def _compose_kwargs(contract: TransformContract) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"seed": contract.seed, "strict": True}
    if contract.input_recipe == "hbb_keypoints":
        kwargs["bbox_params"] = albumentations.BboxParams(
            coord_format="pascal_voc",
            label_fields=["bbox_labels"],
            bbox_type="hbb",
        )
        kwargs["keypoint_params"] = albumentations.KeypointParams(
            coord_format="xy",
            label_fields=["keypoint_labels"],
            label_mapping={},
            remove_invisible=False,
        )
    return kwargs


def _run_contract(contract: TransformContract) -> dict[str, np.ndarray]:
    transform_cls = getattr(albumentations, contract.name)
    transform = albumentations.Compose(
        [transform_cls(**contract.params, p=1.0)],
        **_compose_kwargs(contract),
    )
    data = _base_data(contract)
    output = transform(**data)
    return {target: _as_output_array(output[target]) for target in contract.targets}


def _manifest_entry(contract: TransformContract, outputs: dict[str, np.ndarray], vector_file: str) -> dict[str, Any]:
    return {
        "id": contract.name,
        "transform": contract.name,
        "params": contract.params,
        "seed": contract.seed,
        "behavior_epoch": contract.behavior_epoch,
        "input_recipe_name": contract.input_recipe,
        "stability": contract.stability,
        "tolerance": contract.tolerance,
        "targets": list(contract.targets),
        "input_recipe": _input_recipe_for_contract(contract),
        "vector_file": vector_file,
        "outputs": {target: _array_metadata(array) for target, array in outputs.items()},
        "environment": _environment_metadata(),
    }


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"schema_version": 1, "cases": []}
    return json.loads(MANIFEST_PATH.read_text())


def _write_manifest_case(entry: dict[str, Any]) -> None:
    manifest = _load_manifest()
    cases = [case for case in manifest.get("cases", []) if case.get("id") != entry["id"]]
    cases.append(entry)
    manifest["schema_version"] = 1
    manifest["cases"] = sorted(cases, key=lambda case: str(case["id"]))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def generate_contract(contract: TransformContract, epoch: str | None = None) -> Path:
    if epoch is not None:
        contract = TransformContract(
            contract.name,
            contract.params,
            contract.targets,
            contract.stability,
            contract.seed,
            epoch,
            contract.input_recipe,
            contract.tolerance,
        )

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    outputs = _run_contract(contract)
    vector_path = VECTOR_DIR / f"{contract.name}.npz"
    np.savez_compressed(vector_path, **outputs)
    _write_manifest_case(_manifest_entry(contract, outputs, str(vector_path.relative_to(REGRESSION_ROOT))))
    return vector_path


def _selected_contracts(transform_name: str | None, generate_all: bool) -> tuple[TransformContract, ...]:
    if generate_all:
        return REGRESSION_CONTRACTS
    if transform_name is None:
        msg = "Use --transform <name> or --all."
        raise ValueError(msg)
    return (contract_by_name(transform_name),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", help="Single transform name to regenerate.")
    parser.add_argument("--all", action="store_true", help="Regenerate all registered vectors.")
    parser.add_argument("--epoch", help="Behavior epoch to record in the manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contracts = _selected_contracts(args.transform, args.all)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    for contract in contracts:
        vector_path = generate_contract(contract, args.epoch)
        print(f"Wrote {vector_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
