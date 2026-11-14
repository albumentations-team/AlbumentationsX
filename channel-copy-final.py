import ast
import csv
from itertools import product
import os
from pathlib import Path
from statistics import median
import sys
from timeit import Timer
from unittest.mock import patch

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
os.environ["NO_ALBUMENTATIONS_UPDATE"]="1"

import albumentations as A
import albucore
import cv2 as cv
import numkong
import numpy as np
import torch
import platform
from albumentations.augmentations.pixel import _functional_color as color, _functional_sharpness as sharp, functional
from albucore import histogram as hist

cv.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if platform.system()=="Darwin": assert numkong.get_capabilities()["neon"]
print(platform.platform(), sys.version, albucore.__file__, A.__file__, np.__version__,cv.__version__,torch.__version__,cv.getNumThreads(),flush=True)

root=Path(sys.argv[1])
def load(module,file,names):
    text=(root/file).read_text()
    tree=ast.parse(text)
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(tree.body)==len(names)
    namespace=vars(module).copy()
    exec(compile(tree,file,"exec",flags=0x1000000),namespace)
    return namespace

old_eq=load(color,"channel-copy-baseline-color.py",["equalize"])["equalize"]
old_ca=load(sharp,"channel-copy-baseline-sharp.py",["chromatic_aberration"])["chromatic_aberration"]
old_hist=load(hist,"channel-copy-baseline-hist.py",["equalize_uint8","_equalize_pillow_rgb"])["equalize_uint8"]
rng=np.random.default_rng(137)
def measure(fn,variants):
    for seed in (0,137,731):
        a,b=[fn(v,seed) for v in variants]
        np.testing.assert_array_equal(a.view(f"u{a.itemsize}"),b.view(f"u{b.itemsize}"),strict=True)
    ts=[Timer(lambda v=v:fn(v,137)) for v in variants]
    counts=[max(50,min(1500,int(.015/(t.timeit(4)/4)))) for t in ts]
    times=[[],[]]
    for rep in range(7):
        for i in ((0,1) if rep%2==0 else (1,0)):
            times[i].append(ts[i].timeit(counts[i])*1e6/counts[i])
    return [median(x) for x in times]

with open(sys.argv[2],"w",newline="",buffering=1) as out:
    w=csv.writer(out)
    w.writerow(("operation","route","side","channels","dtype","mode","layout","baseline_us","candidate_us"))
    for side,channel,mode,layout in product((64,256,512,1024),(1,3,5),("pil","cv"),("dense","crop")):
        if channel==1 and layout=="crop":continue
        raw=rng.integers(256,size=(side,side+4,channel),dtype=np.uint8)
        im=raw[:,2:-2]; im=im.copy() if layout=="dense" else im
        mask=(im[...,0]>40).astype(np.uint8)
        if mode=="cv" and channel==5: mask=np.repeat(mask[...,None],channel,axis=-1); mask[...,1]=0
        w.writerow(("masked Equalize","Core",side,channel,"uint8",mode,layout,*measure(
            lambda fn,_:fn(im,mode=mode,mask=mask),(old_hist,albucore.equalize_uint8))))
    print("Core matrix complete",flush=True)
    for side,dtype in product((32,64,256,512,1024),(np.uint8,np.float16,np.float32)):
        raw=rng.integers(256,size=(side+8,side+12,3),dtype=np.uint8)
        if dtype!=np.uint8: raw=(raw.astype(np.float32)/255).astype(dtype)
        crop=raw[2:side+2,3:side+3]; crop.flags.writeable=False
        for operation, baseline, transform in (
            ("equalize",old_eq,A.Equalize(mode="cv",by_channels=False,p=1)),
            ("equalize",old_eq,A.Equalize(mode="pil",by_channels=False,p=1)),
            ("chromatic_aberration",old_ca,A.ChromaticAberration(p=1)),
        ):
            candidate=getattr(functional,operation)
            mode=getattr(transform,"mode","")
            pipeline=A.Compose([A.RandomCrop(side,side,p=1),transform],telemetry=False)
            if operation=="equalize":
                direct=lambda fn,im:fn(im,mode=mode,by_channels=False)
            else:
                direct=lambda fn,im:fn(im,.1,.06,-.1,-.06,cv.INTER_LINEAR)
            for layout,img in (("dense",crop.copy()),("crop",crop),("flipped",crop[:,::-1])):
                expected=direct(baseline,img)
                actual=direct(candidate,img)
                np.testing.assert_array_equal(expected.view(f"u{expected.itemsize}"),actual.view(f"u{actual.itemsize}"),strict=True)
                if layout=="crop" and side>=256:
                    w.writerow((operation,"kernel",side,3,dtype.__name__,mode,layout,*measure(lambda fn,_:direct(fn,img),(baseline,candidate))))
            # Patch only once per timed block. Measure the same Compose instance and options.
            def eval(fn,seed):
                with patch.object(functional,operation,fn):
                    return pipeline(image=raw,invocation_seed=seed)["image"]
            w.writerow((operation,"Compose",side,3,dtype.__name__,mode,"RandomCrop",*measure(eval,(baseline,candidate))))
    print("All-bitwise parity passed",flush=True)
