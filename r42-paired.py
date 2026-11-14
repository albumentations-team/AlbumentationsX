from __future__ import annotations

import ast
import csv
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import median
from timeit import Timer
from unittest.mock import patch

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import albucore
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.augmentations.pixel import noise
from augbench.adapters.sources import SourceRef, load_albumentationsx_rgb
from augbench.execution.pipeline import _collate_images
from augbench.implementations.albumentationsx_pipeline_impl import build_recipe
from augbench.recipes.load import load_recipe_catalog
from numkong import get_capabilities

cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if sys.platform == 'darwin':
    assert get_capabilities()['neon']
print(platform.platform(), sys.version.split()[0], A.__file__, albucore.__file__, flush=True)
print({p: version(p) for p in ('albucore', 'numpy', 'opencv-python-headless', 'torch', 'numkong', 'stringzilla', 'simplejpeg')}, flush=True)
print('threads', cv2.getNumThreads(), torch.get_num_threads(), torch.get_num_interop_threads(), flush=True)

repo, catalog_root, jpeg_dir, output = map(Path, sys.argv[1:])
source = subprocess.check_output(['git', '-C', str(repo), 'show', '592c92e7f9aeb090f87d0a28cdfe8e8dad126ab0:albumentations/augmentations/pixel/noise.py'], text=True)
tree = ast.parse(source)
tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SaltAndPepper']
namespace = vars(noise).copy()
exec(compile(tree, '<baseline>', 'exec'), namespace)
old_transform = namespace['SaltAndPepper']
new_transform = A.SaltAndPepper
rng = np.random.default_rng(137)
jpeg_dir.mkdir(parents=True, exist_ok=True)
paths = [jpeg_dir / f'{i:02d}.jpg' for i in range(16)]
for i, path in enumerate(paths):
    image = rng.integers(0, 256, (480 + i * 3, 640 - i * 2, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

def load_images():
    return [load_albumentationsx_rgb(SourceRef(path)) for path in paths]

def assert_bits(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    else:
        np.testing.assert_array_equal(a.view(f'u{a.itemsize}'), b.view(f'u{b.itemsize}'), strict=True)

recipe = next(r for r in load_recipe_catalog(catalog_root / 'catalog/recipes/rgb.yaml').recipes if r.augmentation_id == 'SaltAndPepper')
pipelines = []
for cls in (old_transform, new_transform):
    with patch.object(A, 'SaltAndPepper', cls):
        pipeline = build_recipe(recipe, 'albumentationsx_cpu').cpu_transform
    assert isinstance(pipeline.transforms[-1], A.ToTensorV2)
    pipelines.append(pipeline)
print(recipe.recipe_id, flush=True)
images = load_images()

def compare(calls, count, batch_size):
    timers = [Timer(fn) for fn in calls]
    for timer in timers:
        timer.timeit(3)
    timings = [[], []]
    for j in range(9):
        for i in ((0, 1) if j % 2 == 0 else (1, 0)):
            timings[i].append(timers[i].timeit(count) * 1e6 / count / batch_size)
    return [median(t) for t in timings]

with output.open('w', newline='', buffering=1) as file:
    writer = csv.writer(file)
    writer.writerow(('case', 'before_us', 'after_us'))
    for jpeg in (False, True):
        def run(pipeline, seed):
            return _collate_images([pipeline(image=image, invocation_seed=seed+i)['image'] for i, image in enumerate(load_images() if jpeg else images)])
        for seed in (0, 137, 731, 1761, 16476):
            assert_bits(*(run(pipeline, seed) for pipeline in pipelines))
        writer.writerow(('R42 JPEG' if jpeg else 'R42 decoded', *compare([lambda p=p:run(p, 137) for p in pipelines], 12, len(images))))
    cases = 0
    for shape in ((256, 256, 1), (256, 256, 3), (256, 256, 5), (512, 512, 3), (1024, 1024, 3), (4, 128, 96, 3)):
        for dtype in (np.uint8, np.float16, np.float32):
            raw = rng.integers(0, 256, shape, dtype=np.uint8)
            raw = raw if dtype is np.uint8 else (raw.astype(np.float32) / 255).astype(dtype)
            image = raw[..., ::-1, :]
            image.setflags(write=False)
            targets = {'image': image} if image.ndim == 3 else {'volume': image}
            if image.ndim == 3:
                targets['images'] = np.repeat(image[None], 2, axis=0)
            else:
                targets['volumes'] = np.repeat(image[None], 2, axis=0)
            for amount in (0.0, 0.01, 0.06, 0.16, 0.17, 0.5, 1.0):
                functions = [A.Compose([cls(amount_range=(amount, amount), p=1)], seed=137, telemetry=False) for cls in (old_transform, new_transform)]
                for seed in (0, 137, 731):
                    a, b = (fn(**targets, invocation_seed=seed) for fn in functions)
                    for target in targets:
                        assert_bits(a[target], b[target])
                        cases += 1
                if dtype is np.uint8 and shape == (256, 256, 3):
                    writer.writerow((f'RGB 256² amount {amount:.2f}', *compare([lambda fn=fn:fn(image=image, invocation_seed=137) for fn in functions], 120, 1)))
    print(f'R42 matched with and without JPEG at five seeds. Target and density matrix matched in {cases} comparisons.', flush=True)
