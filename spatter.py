import inspect
import os
from statistics import median
from timeit import Timer
from unittest.mock import patch
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import albumentations as A
import cv2 as cv
import numpy as np
from albumentations.augmentations.pixel import weather as w
cv.setNumThreads(0)
old=A.Spatter.sample_parameters
before='''            liquid_layer[liquid_layer < cutout_threshold] = 0
            if mode == "rain":
'''
assert inspect.getsource(old).count(before)==1
from textwrap import dedent
from types import SimpleNamespace
candidates={
    'skip mud':'''            if mode == "rain":
                liquid_layer[liquid_layer < cutout_threshold] = 0
''',
    'native rain':'''            if mode == "rain":
                threshold = np.nextafter(np.float32(cutout_threshold), np.float32(-np.inf))
                if cutout_threshold > 0:
                    cv2.threshold(liquid_layer, float(threshold), 0, cv2.THRESH_TOZERO, dst=liquid_layer)
                else:
                    liquid_layer[liquid_layer < cutout_threshold] = 0
''',
}
rng=np.random.default_rng(137)
for mode in ('mud','rain'):
 for size in (64,224,512):
    image=rng.integers(0,256,(size,size,3),dtype=np.uint8)
    def build(fn):
        with patch.object(A.Spatter,'sample_parameters',fn):
            return A.Compose([A.Spatter(mode=mode,p=1)],telemetry=False)
    models=[build(old)]
    for rep in candidates.values():
        ns=vars(w).copy()
        exec(compile(dedent(inspect.getsource(old).replace(before,rep)),'<probe>','exec',flags=0x1000000),ns)
        models.append(build(ns['sample_parameters']))
    for seed in (0,137,731,1761,16476):
        a,*b=[p(image=image,invocation_seed=seed)['image'] for p in models]
        for cand in b: np.testing.assert_array_equal(a,cand,strict=True)
    timers=[Timer(lambda p=p:p(image=image,invocation_seed=137)) for p in models]
    times=[[] for _ in models]
    count=80 if size<512 else 24
    for t in timers:t.timeit(4)
    for j in range(9):
        for k in range(len(timers))[::1 if j%2==0 else -1]:
            times[k].append(timers[k].timeit(count)*1e6/count)
    print(mode,size,*[round(median(t),2) for t in times],flush=True)
