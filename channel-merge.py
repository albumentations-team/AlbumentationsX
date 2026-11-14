import inspect
import os
from pathlib import Path
from statistics import median
from timeit import Timer

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import cv2
import numpy as np
from albucore import get_opencv_max_channels
from albumentations.augmentations.pixel import _functional_color as color

cv2.setNumThreads(0)
maximum = get_opencv_max_channels()
source = inspect.getsource(color.image_compression)
old_line = '    result = cast("ImageUInt8", np.dstack([decoded_bgr, *extra_channels]))'
new_lines = '''    channels = [decoded_bgr, *extra_channels]
    result = cast(
        "ImageUInt8",
        cv2.merge(channels) if image_type == ".jpg" and num_channels <= _OPENCV_MAX_CHANNELS else np.dstack(channels),
    )'''
assert old_line in source or new_lines in source

def compile_function(text):
    namespace = vars(color) | {'_OPENCV_MAX_CHANNELS': maximum}
    exec('from __future__ import annotations\n' + text, namespace)
    return namespace['image_compression']

old = compile_function(source.replace(new_lines, old_line))
new = compile_function(source.replace(old_line, new_lines))

print(cv2.__version__, np.__version__, maximum, flush=True)
print('side,channels,layout,before_us,after_us,paired_delta_pct', flush=True)
rng = np.random.default_rng(137)
for size, channels in ((8, 4), (8, maximum // 2), (8, maximum), (8, maximum + 1), (32, maximum), (64, maximum)):
    base = rng.integers(0, 256, (size, size + 4, channels), dtype=np.uint8)
    for layout in ('dense', 'crop'):
        image = base[:, 2:-2]
        if layout == 'dense':
            image = image.copy()
        image.setflags(write=False)
        np.testing.assert_array_equal(old(image, 75, '.jpg'), new(image, 75, '.jpg'), strict=True)
        timers = [Timer(lambda fn=fn: fn(image, 75, '.jpg')) for fn in (old, new)]
        count = max(8, min(1600, int(.14 / (timers[0].timeit(10) / 10))))
        times = [[], []]
        for index in range(12):
            for route in ((0, 1) if index % 2 == 0 else (1, 0)):
                times[route].append(timers[route].timeit(count) * 1e6 / count)
        delta = median(after / before - 1 for before, after in zip(*times)) * 100
        print(f'{size},{channels},{layout},{median(times[0]):.2f},{median(times[1]):.2f},{delta:.2f}', flush=True)
print('Bitwise checks passed.', flush=True)
