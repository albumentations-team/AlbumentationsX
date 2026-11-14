from __future__ import annotations

import ast
import csv
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from itertools import product
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
import numkong
import numpy as np
import torch
from albumentations.augmentations.pixel import _functional_color as color, _functional_sharpness as sharp, functional

cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if platform.system() == 'Darwin':
    assert numkong.get_capabilities()['neon']
print(platform.platform(), platform.processor(), sys.version, A.__file__, albucore.__file__, flush=True)
print({p: version(p) for p in ('albucore', 'numpy', 'opencv-python-headless', 'torch', 'numkong', 'stringzilla')}, flush=True)
print('OpenCV, torch threads:', cv2.getNumThreads(), torch.get_num_threads(), torch.get_num_interop_threads(), flush=True)


def baseline(module, name):
    file = Path(module.__file__).relative_to(Path(A.__file__).parent.parent)
    source = subprocess.check_output(['git', 'show', f'2fa2d187cee53e93a5022baab8726cd648544161:{file}'], text=True)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(tree.body) == 1
    namespace = vars(module).copy()
    exec(compile(tree, str(file), 'exec', flags=0x1000000), namespace)
    return namespace[name]


def compare(calls):
    a, b = [fn() for fn in calls]
    np.testing.assert_array_equal(a.view(f'u{a.itemsize}'), b.view(f'u{b.itemsize}'), strict=True)
    timers = [Timer(fn) for fn in calls]
    count = max(4, min(1000, int(.22 / (timers[0].timeit(5) / 5))))
    times = [[], []]
    for repeat in range(14):
        for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            times[i].append(timers[i].timeit(count) * 1e6 / count)
    print("Paired percentage changes:", [round(100 * (b / a - 1), 1) for a, b in zip(*times)], flush=True)
    return [median(values) for values in times]


def compare_compose(name, pipeline, image, old, new):
    def run(fn, seed):
        with patch.object(functional, name, fn):
            return pipeline(image=image, invocation_seed=seed)['image']

    for seed in (0, 137, 731):
        a, b = [run(fn, seed) for fn in (old, new)]
        np.testing.assert_array_equal(a.view(f'u{a.itemsize}'), b.view(f'u{b.itemsize}'), strict=True)

    # Both routes pay the same setter overhead during the timed calls.
    return compare([lambda fn=fn: run(fn, 137) for fn in (old, new)])


rng = np.random.default_rng(137)
with open(sys.argv[1], 'w', newline='', buffering=1) as output:
    rows = csv.writer(output)
    rows.writerow(('operation', 'route', 'side', 'channels', 'dtype', 'before_us', 'after_us'))
    for operation, module, shapes in (
        ('chromatic_aberration', sharp, tuple(product((256, 512, 1024), (3,), (np.uint8, np.float16, np.float32)))),
        ('image_compression', color, ((224, 4, np.uint8), (256, 9, np.uint8), *product((256, 512, 1024), (5,), (np.uint8,)), (256, 5, np.float16), (256, 5, np.float32))),
    ):
        old, new = baseline(module, operation), getattr(module, operation)
        if operation == 'chromatic_aberration':
            transform = A.ChromaticAberration(p=1)
            args = (.1, .06, -.1, -.06, cv2.INTER_LINEAR)
        else:
            transform = A.ImageCompression(quality_range=(75, 75), compression_type='jpeg', p=1)
            args = (75, '.jpg')
        if operation != "chromatic_aberration":
            continue
        for size, channels, dtype in ((1024, 3, np.uint8), (1024, 3, np.uint8), (512, 3, np.uint8), (1024, 3, np.uint8)):
            raw = rng.integers(0, 256, (size + 8, size + 12, channels), dtype=np.uint8)
            if dtype != np.uint8:
                raw = (raw.astype(np.float32) / 255).astype(dtype)
            crop = raw[2:size + 2, 3:size + 3]
            crop.flags.writeable = False
            for image in (crop.copy(), crop, crop[:, ::-1]):
                a, b = [fn(image, *args) for fn in (old, new)]
                np.testing.assert_array_equal(a.view(f'u{a.itemsize}'), b.view(f'u{b.itemsize}'), strict=True)
            rows.writerow((operation, 'kernel', size, channels, dtype.__name__, *compare([lambda fn=fn: fn(crop, *args) for fn in (old, new)])))
            pipeline = A.Compose([A.RandomCrop(size, size, p=1), transform], telemetry=False)
            rows.writerow((operation, 'Compose', size, channels, dtype.__name__, *compare_compose(operation, pipeline, raw, old, new)))
print('All comparisons bitwise identical.', flush=True)
