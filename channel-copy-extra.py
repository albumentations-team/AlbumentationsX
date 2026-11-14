import csv
import os
import sys
from statistics import median
from timeit import Timer
from unittest.mock import patch

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"]="1"
import albumentations as A
import cv2 as cv
import numpy as np
import torch
from albumentations.core.composition import SelectiveChannelTransform
from albumentations.augmentations.pixel import _functional_color as color, _functional_torchvision as tv
from albumentations.augmentations.other import annotation_artifacts_functional as ann

cv.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
rng=np.random.default_rng(137)

def prepared(fn, old, new):
    import inspect
    import textwrap
    source=textwrap.dedent(inspect.getsource(fn))
    assert old in source,old
    namespace=fn.__globals__.copy()
    exec("from __future__ import annotations\n"+source.replace(old,new),namespace)
    return namespace[fn.__name__]

def measure(pipeline,target,attribute,old,new,image):
    phases=[old,new]
    for seed in (0,137,731):
        with patch.object(target,attribute,old):
            before=pipeline(image=image,invocation_seed=seed)["image"]
        with patch.object(target,attribute,new):
            after=pipeline(image=image,invocation_seed=seed)["image"]
        np.testing.assert_array_equal(before.view(f"u{before.itemsize}"),after.view(f"u{after.itemsize}"),strict=True)
    timer=Timer(lambda:pipeline(image=image,invocation_seed=137))
    draws=[[],[]]
    counts=[]
    for fn in phases:
        with patch.object(target,attribute,fn):
            counts.append(max(40,min(1500,int(.022/(timer.timeit(5)/5)))))
    for rep in range(7):
        for idx in ((0,1) if rep%2==0 else (1,0)):
            with patch.object(target,attribute,phases[idx]):
                draws[idx].append(timer.timeit(counts[idx])*1e6/counts[idx])
    return [median(x) for x in draws]

def draw(image,color,draw_fn):
    if image.ndim==3 and 1<image.shape[2]<=512:
        for i, value in enumerate(ann._normalize_color(color,image.shape[2])):
            plane=cv.extractChannel(image,i)
            draw_fn(plane,value)
            cv.insertChannel(plane,image,i)
    else:
        original_draw(image,color,draw_fn)

original_draw=ann._draw_channelwise
hue=prepared(tv._adjust_hue_torchvision_uint8,"img[..., 0] = apply_uint8_lut(img[..., 0], lut)",
"""hue = cv2.extractChannel(img, 0)
    apply_uint8_lut(hue, lut, out=hue)
    cv2.insertChannel(hue, img, 0)""")
selective=prepared(SelectiveChannelTransform.apply_in_invocation,"output_image[:, :, channel_index] = channel",
    "cv2.insertChannel(channel, output_image, channel_index)")
with open(sys.argv[1],"w",newline="",buffering=1) as file:
    output=csv.writer(file)
    output.writerow(("operation","size","channels","dtype","base_us","new_us"))
    for n in (32,64,256,512,1024):
        for c in (1,3,5):
            image=rng.integers(256,size=(n,n,c),dtype=np.uint8)
            for dtype in (np.uint8,np.float32):
                source=image if dtype==np.uint8 else image.astype(dtype)/255
                transform=A.AnnotationArtifacts(element_types=("rectangle",),element_probabilities=(1,),count_range=(2,2),p=1)
                p=A.Compose([transform],telemetry=False)
                output.writerow(("annotation",n,c,dtype.__name__,*measure(p,ann,"_draw_channelwise",original_draw,draw,source)))
            if c==3:
                jitter=A.ColorJitter(brightness_range=(.9,1.1),contrast_range=(.9,1.1),saturation_range=(.9,1.1),hue_range=(.15,.15),p=1)
                p=A.Compose([jitter],telemetry=False)
                output.writerow(("color jitter",n,c,"uint8",*measure(p,tv,"_adjust_hue_torchvision_uint8",
                    tv._adjust_hue_torchvision_uint8,hue,image)))
            if c>=3:
                for changed in ((0,), (0,2)):
                    selective_pipeline=A.Compose([A.SelectiveChannelTransform([A.InvertImg(p=1)],channels=changed,p=1)],telemetry=False)
                    out=measure(selective_pipeline,SelectiveChannelTransform,"apply_in_invocation",
                               SelectiveChannelTransform.apply_in_invocation,selective,image)
                    output.writerow((f"selective {len(changed)}",n,c,"uint8",*out))
    print("bitwise matched all pairs",flush=True)
