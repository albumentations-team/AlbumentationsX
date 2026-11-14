import ast
import csv
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from pathlib import Path
from statistics import median
from timeit import Timer
import sys
from unittest.mock import patch
import albumentations as A
import cv2 as cv
import numpy as np
import torch
from albucore import get_opencv_max_channels
from albumentations.augmentations.pixel import _functional_color as color, functional
cv.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(cv.__version__, np.__version__, get_opencv_max_channels(), sys.version, A.__file__, flush=True)
old = color.image_compression
path = Path(color.__file__)
text = path.read_text()
node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "image_compression")
source = ast.get_source_segment(text, node)
old_expr = "np.dstack([decoded_bgr, *extra_channels])"
assert old_expr in source
namespace = vars(color).copy()
namespace["_OPENCV_MAX_CHANNELS"] = get_opencv_max_channels()
# Probe both formats; the final routing decision follows the cross-platform measurements.
expr = "cv2.merge([decoded_bgr, *extra_channels]) if num_channels <= _OPENCV_MAX_CHANNELS else " + old_expr
exec("from __future__ import annotations\n" + source.replace(old_expr, expr), namespace)
new = namespace["image_compression"]

def compare(calls):
    a,b = (fn() for fn in calls)
    np.testing.assert_array_equal(a.view(f"u{a.itemsize}"),b.view(f"u{b.itemsize}"),strict=True)
    timers = [Timer(fn) for fn in calls]
    counts = [max(3,min(700,int(.04/(timer.timeit(3)/3)))) for timer in timers]
    values = [[],[]]
    for rep in range(7):
        for index in ((0,1) if rep%2==0 else (1,0)):
            values[index].append(timers[index].timeit(counts[index])*1e6/counts[index])
    return [median(v) for v in values]

rng = np.random.default_rng(137)
with Path(sys.argv[1]).open("w", newline="", buffering=1) as buffer:
    writer = csv.writer(buffer)
    writer.writerow(("side","channels","format","layout","dtype","route","before_us","after_us"))
    for size,channels in ((32,4),(224,4),(256,5),(512,5),(1024,5),(256,9),(8,128),(8,129)):
        base = rng.integers(0,256,(size,size+4,channels),dtype=np.uint8)
        for fmt in (".jpg",".webp"):
            if size>=512 and fmt==".webp": continue
            for layout in ("dense","crop"):
                for dtype in ((np.uint8,np.float16,np.float32) if (size,channels)==(256,5) else (np.uint8,)):
                    pixels=base if dtype==np.uint8 else (base.astype(np.float32)/255).astype(dtype)
                    image=pixels[:,2:-2]
                    if layout=="dense": image=image.copy()
                    image.flags.writeable=False
                    functions=(old,new)
                    calls=[lambda fn=fn:fn(image,75,fmt) for fn in functions]
                    writer.writerow((size,channels,fmt,layout,dtype.__name__,"kernel",*compare(calls)))
            if size in (224,256,512,1024) and channels<=9:
                compression_type="jpeg" if fmt==".jpg" else "webp"
                pipeline=A.Compose([A.RandomCrop(size,size,p=1),A.ImageCompression(quality_range=(75,75),compression_type=compression_type,p=1)],telemetry=False)
                def invoke(fn):
                    with patch.object(functional,"image_compression",fn):
                        return pipeline(image=base,invocation_seed=137)["image"]
                writer.writerow((size,channels,fmt,"crop","uint8","Compose",*compare([lambda fn=fn:invoke(fn) for fn in (old,new)])))
    print("Bitwise checks passed.",flush=True)
