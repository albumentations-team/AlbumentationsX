import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"]="1"

import csv
import inspect
import sys
import textwrap
from statistics import median
from timeit import Timer
from unittest.mock import patch
import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.pixel import _functional_sharpness as sharp, _functional_color as color
import cv2 as cv
import numpy as np
import torch
from albucore import histogram as hist, filtering

cv.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def replace(fn,old,new):
    source=textwrap.dedent(inspect.getsource(fn))
    assert old in source
    namespace=fn.__globals__.copy()
    exec("from __future__ import annotations\n"+source.replace(old,new),namespace)
    return namespace[fn.__name__]

def run(output,label,a,b):
    expected, actual=a(),b()
    np.testing.assert_array_equal(expected.view(f"u{expected.itemsize}"),actual.view(f"u{actual.itemsize}"),strict=True)
    timers=[Timer(a),Timer(b)]
    counts=[max(12,min(1200,int(.018/(t.timeit(5)/5)))) for t in timers]
    values=[[],[]]
    for i in range(7):
        for j in ((0,1) if i%2==0 else (1,0)):
            values[j].append(timers[j].timeit(counts[j])*1e6/counts[j])
    output.writerow((*label,*(median(x) for x in values)))

fast_equalize=replace(color.equalize,"ycrcb[..., :1] = albucore.equalize_uint8(ycrcb[..., :1], mode=mode, mask=selection)",
    "plane = cv2.extractChannel(ycrcb, 0)\n        plane = albucore.equalize_uint8(plane[..., None], mode=mode, mask=selection)\n        cv2.insertChannel(plane, ycrcb, 0)")
fast_chromatic=replace(sharp.chromatic_aberration,"np.dstack([red_distorted, img[..., 1], blue_distorted])",
    "cv2.merge((red_distorted, cv2.extractChannel(img, 1), blue_distorted))")
# Build the complete candidate from the original source to exercise exactly the three substitutions.
source=inspect.getsource(sharp.chromatic_aberration)
ns=vars(sharp).copy()
source=source.replace("img[..., 0],","cv2.extractChannel(img, 0),").replace("img[..., 2],","cv2.extractChannel(img, 2),")
source=source.replace("np.dstack([red_distorted, img[..., 1], blue_distorted])","cv2.merge((red_distorted, cv2.extractChannel(img, 1), blue_distorted))")
exec("from __future__ import annotations\n"+source,ns)
fast_both=ns["chromatic_aberration"]
rng=np.random.default_rng(137)
with open(sys.argv[1],"w",newline="",buffering=1) as f:
 out=csv.writer(f)
 out.writerow(("operation","n","dtype","mode","old_us","new_us"))
 for n in (32,64,256,512,1024):
    pixels=rng.integers(0,256,(n+6,n+8,3),dtype=np.uint8)
    for dtype in (np.uint8,np.float16,np.float32):
        image=pixels if dtype==np.uint8 else (pixels.astype(np.float32)/255).astype(dtype)
        for mode in ("cv","pil"):
            pipeline=A.Compose([A.RandomCrop(n,n,p=1),A.Equalize(mode=mode,by_channels=False,p=1)],telemetry=False)
            def evaluate(candidate):
                with patch.object(fpixel,"equalize",candidate):
                    return pipeline(image=image,invocation_seed=137)["image"]
            run(out,("Equalize Compose",n,dtype.__name__,mode),lambda:evaluate(color.equalize),lambda:evaluate(fast_equalize))
        p=A.Compose([A.RandomCrop(n,n,p=1),A.ChromaticAberration(p=1)],telemetry=False)
        def aberration(candidate):
            with patch.object(fpixel,"chromatic_aberration",candidate):
                return p(image=image,invocation_seed=137)["image"]
        for mode, fn in (("all",fast_both),("merge",fast_chromatic)):
            run(out,("Chromatic Compose",n,dtype.__name__,mode),lambda:aberration(sharp.chromatic_aberration),lambda:aberration(fn))
 print("bitwise matched",flush=True)
