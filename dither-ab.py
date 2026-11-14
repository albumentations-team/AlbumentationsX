import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import inspect
from statistics import median
from timeit import Timer
from unittest.mock import patch
import albumentations as A
import numpy as np
import cv2 as cv
from albumentations.augmentations.pixel import dithering_functional as f
cv.setNumThreads(0)
def build(fn,axis):
    text=inspect.getsource(fn)
    if axis=='u8':
        replace='''    if n_colors == 2 and noise_uint8.dtype == np.int16 and 0 < img.size <= 2_147_483_647:
        noisy = np.add(img, noise_uint8, dtype=np.int16).reshape(1, -1)
        return cv2.compare(noisy, 127, cv2.CMP_GT).reshape(img.shape)

'''
        mark='    # Add noise and clip'
    else:
        replace='''    if n_colors == 2 and 0 < noisy.size <= 2_147_483_647 and noisy.dtype in (np.float32, np.float64):
        threshold = np.nextafter(noisy.dtype.type(0.5), noisy.dtype.type(-np.inf))
        destination = noisy.reshape(1, -1)
        cv2.threshold(destination, float(threshold), 1, cv2.THRESH_BINARY, dst=destination)
        return destination.reshape(noisy.shape).astype(np.float32, copy=False)

'''
        mark='    np.clip(noisy, 0, 1, out=noisy)'
        # The write must go to a view or to the returned packed buffer.
    ns=vars(f).copy();ns['cv2']=cv
    exec(compile(text.replace(mark,replace+mark),'<probe>','exec',flags=0x1000000),ns)
    return ns[fn.__name__]
routes=[(old,build(old,angle)) for old,angle in ((f.random_dither_uint8,'u8'),(f.random_dither,'float'))]
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
