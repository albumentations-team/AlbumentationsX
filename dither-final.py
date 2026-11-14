import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
from statistics import median
from timeit import Timer
from unittest.mock import patch
import albumentations as A
import numpy as np
import cv2 as cv
from albumentations.augmentations.pixel import dithering_functional as f
cv.setNumThreads(0)
import sys
from pathlib import Path
ns=vars(f).copy()
exec(compile(Path(sys.argv[1]).read_text(),sys.argv[1],'exec',flags=0x1000000),ns)
routes=[(fn,ns[fn.__name__]) for fn in (f.random_dither_uint8,f.random_dither)]
rng=np.random.default_rng(137)
for dtype,(old,new) in zip((np.uint8,np.float32),routes):
    for size in (32,64,224,512,1024):
        raw=rng.integers(0,256,(size,size+2,3),dtype=np.uint8)
        if dtype==np.float32: raw=raw.astype(dtype)/np.float32(255)
        img=raw[:,1:-1]
        if dtype==np.uint8:
            noise=rng.integers(-255,256,img.shape,dtype=np.int16)
            noise.flat[:6]=[-32768, -1,0,127,128,32767]
        else:
            noise=rng.uniform(-.5,.5,img.shape)
            noise.flat[:8]=[np.nan,np.inf,-np.inf,-0.,0.,0.5,np.nextafter(.5,0),np.nextafter(.5,1)]
        a,b=[fn(img,2,(-.5,.5),random_noise=noise) for fn in (old,new)]
        np.testing.assert_array_equal(a,b,strict=True)
        aug=A.Compose([A.Dithering(method='random',n_colors=2,color_mode='per_channel',p=1)],telemetry=False)
        def invoke(seed,fn):
            with patch.object(f,old.__name__,fn):
                return aug(image=img,invocation_seed=seed)['image']
        for seed in (0,137,731,1761,16476):
            np.testing.assert_array_equal(invoke(seed,old),invoke(seed,new),strict=True)
        for k,calls in (
            ('kernel',[lambda fn=fn:fn(img,2,(-.5,.5),random_noise=noise) for fn in (old,new)]),
            ('Compose',[lambda fn=fn:invoke(137,fn) for fn in (old,new)]),
        ):
            timers=[Timer(fn) for fn in calls]
            vals=[[],[]]
            count=75 if size<=224 else 15
            for i in range(7):
                for j in ((0,1) if not i%2 else (1,0)):
                    vals[j].append(timers[j].timeit(count)*1e6/count)
            print(dtype.__name__,size,k,*[round(median(x),2) for x in vals],flush=True)
