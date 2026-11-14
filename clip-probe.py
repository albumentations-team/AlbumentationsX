import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import cv2 as cv
import numpy as np
import torch
from albucore import clip
from statistics import median
from timeit import Timer
cv.setNumThreads(0)
torch.set_num_threads(1)
rng=np.random.default_rng(137)
for kind in (np.float16,np.float32,np.uint8):
 for n in (64,224,512,1024):
  raw=rng.uniform(-.25,1.25,(n,n,3)).astype(kind) if kind!=np.uint8 else rng.integers(0,256,(n,n,3),dtype=kind)
  for mode in ('allocate','out','inplace'):
   out=None if mode=='allocate' else raw.copy() if mode=='inplace' else np.empty_like(raw)
   src=raw if mode!='inplace' else out
   def base():
    return clip(src,out=out)
   def native():
    if kind==np.uint8:
     if out is None: return src.copy()
     if out is not src: np.copyto(out,src)
     return out
    target=np.empty_like(src) if out is None else out
    cv.max(src.reshape(1,-1),0,target.reshape(1,-1))
    cv.min(target.reshape(1,-1),1,target.reshape(1,-1))
    return target
   def threshold():
    target=np.empty_like(src) if out is None else out
    cv.threshold(src.reshape(1,-1),0,0,cv.THRESH_TOZERO,dst=target.reshape(1,-1))
    cv.threshold(target.reshape(1,-1),1,0,cv.THRESH_TRUNC,dst=target.reshape(1,-1))
    return target
   def torchclip():
    s=torch.from_numpy(src)
    if out is None: return s.clamp(0,1).numpy()
    torch.clamp(s,0,1,out=torch.from_numpy(out))
    return out
   def uintbits():
    data=src.view('u4')
    signed=data.view('i4')
    largest=data.max()
    if largest>0xff800000 or (largest>0x7f800000 and signed.max()>0x7f800000):
        return base()
    target=np.empty_like(src) if out is None else out
    d=target.view('i4').reshape(1,-1)
    if largest>0x7f800000:
        cv.max(signed.reshape(1,-1),0,d)
        cv.min(d,0x3f800000,d)
    else:
        cv.min(signed.reshape(1,-1),0x3f800000,d)
    return target
   methods={'current':base}
   if kind==np.uint8: methods['copy']=native
   if kind==np.float32: methods.update(cvminmax=native,threshold=threshold,torch=torchclip,bits=uintbits)
   if kind==np.float16: methods['torch']=torchclip
   reference=np.clip(raw,0,255 if kind==np.uint8 else 1)
   def verify(data):
    if mode=='inplace': src[...] = data
    return
   for name,fn in list(methods.items()):
    verify(raw)
    a=fn()
    np.testing.assert_array_equal(a,reference,strict=True)
   if kind!=np.uint8:
    before=raw.copy(); before.reshape(-1)[:6]=[np.nan,-np.nan,np.inf,-np.inf,-0.,0.]
    expected=np.clip(before,0,1)
    for name,fn in methods.items():
     if mode=='inplace': src[...] = before
     else: raw[...] = before
     if not np.array_equal(fn(),expected,equal_nan=True): print('nonfinite mismatch',kind.__name__,name,mode,n,flush=True)
     if mode!='inplace': raw.reshape(-1)[:6]=rng.uniform(-.25,1.25,6)
   verify(raw)
   timers=[Timer(fn) for fn in methods.values()]
   count=120 if n<=224 else 25
   trials=[[] for fn in methods]
   for i in range(7):
    for j in (range(len(timers)) if not i%2 else reversed(range(len(timers)))):
     trials[j].append(timers[j].timeit(count)*1e6/count)
   print(kind.__name__,n,mode,{k:round(median(a),2) for k,a in zip(methods,trials)},flush=True)
