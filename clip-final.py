import ast
import os
import sys
from pathlib import Path
from statistics import median
from timeit import Timer
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import cv2 as cv
import numpy as np
from albucore import clip,utils
cv.setNumThreads(0)
ns=vars(utils).copy()
s=Path(sys.argv[1]).read_text()
t=ast.parse(s)
t.body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('_clip_float16_opencv','clip') and not n.decorator_list]
exec(compile(t,'<patch>','exec'),ns)
new=ns['clip']
rng=np.random.default_rng(137)
print(sys.version.split()[0], cv.__version__,np.__version__,utils.__file__,flush=True)

for dtype in (np.uint8,np.float16):
    for n,c in ((64,1),(224,1),(224,3),(224,5),(512,3),(1024,3)):
        domain = (0,256) if dtype==np.uint8 else (-.25,1.25)
        raw = rng.uniform(*domain,(n,n+2,c)).astype(dtype)
        if dtype==np.float16:
            raw[0,1,0]=-0.0
        for layout in ('dense','crop','readonly'):
            src = raw[:,1:-1].copy() if layout!='crop' else raw[:,1:-1]
            if layout=='readonly': src.flags.writeable=False
            for mode in ('allocate','out','inplace'):
                if layout=='readonly' and mode=='inplace': continue
                before=src.copy()
                if mode=='allocate':
                    calls=[lambda fn=fn:fn(src) for fn in (clip,new)]
                elif mode=='out':
                    dst=np.empty_like(src)
                    calls=[lambda fn=fn:fn(src,out=dst) for fn in (clip,new)]
                else:
                    copies=[np.array(src,copy=True) for _ in range(2)]
                    calls=[lambda fn=fn,a=arr:fn(a,out=a) for fn,arr in zip((clip,new),copies)]
                a,b=[fn().copy() for fn in calls]
                np.testing.assert_array_equal(a,b,strict=True)
                np.testing.assert_array_equal(src,before,strict=True)
                if mode=='inplace' and dtype==np.float16:
                    continue # benchmark each call with original signed inputs, below
                timers=list(map(Timer,calls))
                times=[[],[]]
                n_iters=140 if n<512 else 35
                for rep in range(7):
                    for j in ((0,1) if rep%2==0 else (1,0)):
                        times[j].append(timers[j].timeit(n_iters)*1e6/n_iters)
                print(dtype.__name__,n,c,layout,mode,*[round(median(a),2) for a in times],flush=True)
for dtype in (np.float16,np.uint8):
    src=np.arange(65536,dtype=np.uint16).view(dtype).reshape(-1,16,1)
    with np.errstate(invalid='ignore'):
        a,b=clip(src),new(src)
        np.testing.assert_array_equal(a,b,strict=True)
        if dtype==np.float16:
            mismatch=(a.view('u2')!=b.view('u2')) & ~((a==0)&(b==0))
            assert not mismatch.any()
print('Complete finite and nonfinite half domain matched except signed zero.',flush=True)
