import inspect
import os
from statistics import median
from timeit import Timer
from unittest.mock import patch

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'

import cv2 as cv
import numpy as np
import albumentations as A
import albucore
from albumentations.augmentations.pixel import _functional_weather as weather, functional as pixel, dithering_functional as dithering
from numkong import get_capabilities
cv.setNumThreads(0)
assert get_capabilities()['neon']
print(A.__file__,albucore.__version__,cv.__version__,np.__version__)
def binary(a,thresh,maxval=1):
    return cv.threshold(a,thresh,maxval,cv.THRESH_BINARY)[1].reshape(a.shape)
def build(fn, module, before, after):
    scope=vars(module).copy()
    scope["cv2"] = cv
    src=inspect.getsource(fn)
    assert src.count(before)==1
    exec(compile(src.replace(before,after),'<probe>','exec',flags=0x1000000),scope)
    return scope[fn.__name__]
mud=build(weather.get_mud_params, weather,'    mask = (liquid_layer > cutout_threshold).astype(np.float32)', '''    _, mask64 = cv2.threshold(liquid_layer, cutout_threshold, 1, cv2.THRESH_BINARY)
    mask = mask64.astype(np.float32, copy=False)''')
random_old=dithering.random_dither
random_new=build(random_old,dithering,'    np.clip(noisy, 0, 1, out=noisy)\n\n    # Quantize using vectorized numpy operations\n    if n_colors == 2:\n        return (noisy >= 0.5).astype(np.float32)','''    if n_colors == 2:
        threshold = np.nextafter(noisy.dtype.type(0.5), noisy.dtype.type(-np.inf))
        _, result = cv2.threshold(noisy, float(threshold), 1, cv2.THRESH_BINARY)
        return result.reshape(noisy.shape).astype(np.float32, copy=False)

    np.clip(noisy, 0, 1, out=noisy)''')
def timed(label, f, g, count=180):
    timers=[Timer(fn) for fn in (f,g)]
    for t in timers: t.timeit(8)
    values=[[],[]]
    for n in range(9):
        for i in ((0,1) if n%2==0 else (1,0)):
            values[i].append(timers[i].timeit(count)*1e6/count)
    a,b=map(median,values)
    print(f'{label}: {a:.2f} -> {b:.2f} µs ({(b/a-1)*100:+.1f}%)',flush=True)
rng=np.random.default_rng(137)
for s in (64,224,512,1024):
    layer=rng.normal(.65,.3,(s,s))
    base=np.random.default_rng(137)
    candidate=np.random.default_rng(137)
    for threshold in (0,.65,1,5):
        a=weather.get_mud_params(layer,np.array([12,35,25]),threshold,.5,.8,base)
        b=mud(layer,np.array([12,35,25]),threshold,.5,.8,candidate)
        assert all(np.array_equal(a[k].view('u4'),b[k].view('u4')) for k in a)
        assert base.bit_generator.state == candidate.bit_generator.state
    color=np.array([12,35,25])
    timed(f'Mud params {s}',lambda:weather.get_mud_params(layer,color,.65,.5,.8,base),lambda:mud(layer,color,.65,.5,.8,candidate),100 if s<512 else 24)
    img=rng.random((s,s,3),dtype=np.float32)
    for dtype in (np.float32,np.float64):
        noise=rng.uniform(-.2,.2,img.shape).astype(dtype)
        control=np.zeros_like(img)
        edge=tuple(dtype(x) for x in (0.5,-0.5,0,1,np.inf,-np.inf,np.nan))
        noise.flat[:len(edge)+2]=(*edge,np.nextafter(dtype(.5),dtype(0)),np.nextafter(dtype(.5),dtype(1)))
        np.testing.assert_array_equal(random_old(control,2,(-.2,.2),random_noise=noise),random_new(control,2,(-.2,.2),random_noise=noise),strict=True)
        timed(f'Random binary {s} {dtype.__name__}',lambda:random_old(img,2,(-.2,.2),random_noise=noise),lambda:random_new(img,2,(-.2,.2),random_noise=noise),100 if s<512 else 22)
    fp=layer.astype('f4')
    def clip_np():
        out=fp.copy()
        return np.clip(out,0,1,out=out)
    def clip_cv():
        out=fp.copy()
        cv.threshold(out,0,0,cv.THRESH_TOZERO,dst=out)
        cv.threshold(out,1,0,cv.THRESH_TRUNC,dst=out)
        return out
    assert np.array_equal(clip_np().view('u4'),clip_cv().view('u4'))
    timed(f'Clip f32 2D {s}',clip_np,clip_cv,150 if s<512 else 44)

def through(fn, image):
    with patch.object(pixel,'get_mud_params',fn):
        return pipeline(image=image,invocation_seed=137)['image']
pipeline=A.Compose([A.Spatter(mode='mud',p=1)],telemetry=False)
for s in (224,512,1024):
    img=rng.integers(0,256,(s,s,3),dtype=np.uint8)
    np.testing.assert_array_equal(through(weather.get_mud_params,img),through(mud,img),strict=True)
    timed(f'Spatter mud Compose {s}',lambda:through(weather.get_mud_params,img),lambda:through(mud,img),50 if s<512 else 15)

pipeline=A.Compose([A.Dithering(method='random',n_colors=2,color_mode='per_channel',p=1)],telemetry=False)
def run_dither(fn, image, seed):
    with patch.object(dithering,'random_dither',fn):
        return pipeline(image=image,invocation_seed=seed)['image']

for s in (64,224,512):
    img=rng.random((s,s,3),dtype=np.float32)
    for seed in (0,137,731,1761,16476):
        np.testing.assert_array_equal(run_dither(random_old,img,seed),run_dither(random_new,img,seed),strict=True)
    timed(f'Random binary float32 Compose {s}',lambda:run_dither(random_old,img,137),lambda:run_dither(random_new,img,137),80 if s<512 else 24)

clip_edge=np.array([[-0.0,0.0,np.nan,np.inf,-np.inf,1]],dtype=np.float32)
expect=np.clip(clip_edge,0,1)
actual=clip_edge.copy()
cv.threshold(actual,0,0,cv.THRESH_TOZERO,dst=actual)
cv.threshold(actual,1,0,cv.THRESH_TRUNC,dst=actual)
print('Clip edge bits',expect.view('u4').tolist(),actual.view('u4').tolist(),flush=True)
