import csv
import inspect
import os
import platform
import sys
from pathlib import Path
from statistics import median
from timeit import Timer
from unittest.mock import patch

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
os.environ["NO_ALBUMENTATIONS_UPDATE"]="1"

import albucore
import albumentations as A
import cv2 as cv
import numpy as np
import torch
from albumentations.augmentations.pixel import _functional_torchvision as tv
from augbench.adapters.sources import SourceRef, load_albumentationsx_rgb
from augbench.execution.pipeline import _collate_images
from augbench.implementations.albumentationsx_pipeline_impl import build_recipe
from augbench.recipes.load import load_recipe_catalog
from numkong import get_capabilities

cv.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if sys.platform == 'darwin':
    assert get_capabilities()['neon']
print(platform.platform(),sys.version.split()[0],A.__file__,albucore.__version__,albucore.__file__,cv.__version__,np.__version__,flush=True)
catalog, fixtures, output = map(Path,sys.argv[1:4])
fixtures.mkdir(exist_ok=True,parents=True)
rng = np.random.default_rng(137)
paths = [fixtures / f'{i:02d}.jpg' for i in range(16)]
for i, path in enumerate(paths):
    assert cv.imwrite(str(path),rng.integers(0,256,(480+i*3,640-i*2,3),dtype=np.uint8),[cv.IMWRITE_JPEG_QUALITY,95])
def decode():
    return [load_albumentationsx_rgb(SourceRef(p)) for p in paths]
decoded=decode()
old_hue=tv._adjust_hue_torchvision_uint8
old_sat=tv.adjust_saturation_torchvision

def transformed(original, before, after):
    scope=vars(tv).copy()
    source=inspect.getsource(original)
    assert source.count(before)==1
    exec(compile(source.replace(before,after),'<candidate>','exec',flags=0x1000000),scope)
    return scope[original.__name__]

line='    img[..., 0] = apply_uint8_lut(img[..., 0], lut)'
hue_extract=transformed(old_hue,line,'    hue = cv2.extractChannel(img, 0)\n    cv2.insertChannel(apply_uint8_lut(hue, lut, out=hue), img, 0)')
hue_cv=transformed(old_hue,line,'    hue = cv2.extractChannel(img, 0)\n    cv2.LUT(hue, lut, dst=hue)\n    cv2.insertChannel(hue, img, 0)')
sat_reuse=transformed(old_sat,'else add_weighted(img, factor, gray_expanded, 1 - factor)','else add_weighted(img, factor, gray_expanded, 1 - factor, out=gray_expanded)' )
sat_cv=transformed(old_sat,'else add_weighted(img, factor, gray_expanded, 1 - factor)','else cv2.addWeighted(img, factor, gray_expanded, 1 - factor, 0)' )
sat_broadcast=transformed(old_sat,'    gray_expanded = grayscale_to_multichannel(gray, img.shape[-1])','    gray_expanded = np.broadcast_to(gray, img.shape) if factor != 0 else grayscale_to_multichannel(gray, img.shape[-1])')

def measure(functions, count, mul, rounds=9):
    timers=[Timer(f) for f in functions]
    for timer in timers: timer.timeit(4)
    samples=[[] for f in functions]
    for rep in range(rounds):
        for i in list(range(len(functions)))[::1 if rep%2==0 else -1]:
            samples[i].append(timers[i].timeit(count)*1e6/count/mul)
    return list(map(median,samples))

def bits(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a.view(f'u{a.itemsize}'),b.view(f'u{b.itemsize}'))

routes = {'base':(old_hue,old_sat), 'hue_extract':(hue_extract,old_sat),
          'hue_cv':(hue_cv,old_sat), 'sat_reuse':(old_hue,sat_reuse),
          'sat_broadcast':(old_hue,sat_broadcast), 'hue_extract_sat_reuse':(hue_extract,sat_reuse)}
if len(sys.argv) == 5:
    funcs=vars(tv).copy()
    exec(compile(Path(sys.argv[4]).read_text(),sys.argv[4],'exec',flags=0x1000000),funcs)
    routes={'base':routes['base'],'final':(funcs.get('_adjust_hue_torchvision_uint8',old_hue), funcs.get('adjust_saturation_torchvision',old_sat))}
    hue_extract, sat_reuse=routes['final']

checks=0
cv_diff=0
broadcast_diff=0
for h,w in ((1,1),(1,23),(17,1),(31,73),(224,224),(256,256),(512,512),(1024,1024)):
    raw=rng.integers(0,256,(h,w+4,3),dtype=np.uint8)
    for image in (raw[:,2:-2].copy(),raw[:,2:-2],raw[:,2:-2][:,::-1]):
        image.setflags(write=False)
        before=image.copy()
        for factor in (-.5,-.03125,-.001,.001,.03125,.5):
            assert bits(old_hue(image,factor),hue_extract(image,factor))
            assert bits(old_hue(image,factor),hue_cv(image,factor))
            checks+=1
        for factor in (0,.01,.125,.5,1,1.25,1.46,2.5,6):
            result=old_sat(image,factor)
            assert bits(result,sat_reuse(image,factor)), (image.shape, image.strides, factor)
            cv_diff += not bits(result,sat_cv(image,factor))
            broadcast_diff += not bits(result,sat_broadcast(image,factor))
            checks+=1
        assert bits(image,before)
print(f'uint8 cases {checks}; cv2.addWeighted mismatches {cv_diff}; broadcast mismatches {broadcast_diff}',flush=True)

with output.open('w',newline='',buffering=1) as out:
    w=csv.writer(out)
    w.writerow(['recipe','scope',*routes])
    if len(sys.argv)==4:
        for size in (32,224,512,1024):
            raw=rng.integers(0,256,(size,size+4,3),dtype=np.uint8)
            for image,layout in ((raw[:,2:-2].copy(),'dense'),(raw[:,2:-2],'crop')):
                mh=measure([lambda f=f:f(image,.13) for f in (old_hue,hue_extract,hue_cv)],170 if size<512 else 50,1)
                ms=measure([lambda f=f:f(image,.63) for f in (old_sat,sat_reuse,sat_broadcast,sat_cv)],170 if size<512 else 50,1)
                print('kernel',size,layout,'hue',mh,'sat',ms,flush=True)
    for recipe in load_recipe_catalog(catalog/'catalog/recipes/rgb.yaml').recipes:
        if recipe.augmentation_id not in ('ColorJitter','ColorJiggle'):
            continue
        pipeline=build_recipe(recipe,'albumentationsx_cpu').cpu_transform
        assert isinstance(pipeline.transforms[-1],A.ToTensorV2)
        def run(arm,seed=137,jpeg=False):
            with patch.object(tv,'_adjust_hue_torchvision_uint8',arm[0]),patch.object(tv,'adjust_saturation_torchvision',arm[1]):
                return _collate_images([pipeline(image=image,invocation_seed=seed+i)['image'] for i,image in enumerate(decode() if jpeg else decoded)])
        for seed in (0,137,731,1761,16476):
            old=run(routes['base'],seed)
            for name,arm in routes.items():
                assert torch.equal(old,run(arm,seed)),(recipe.augmentation_id,name,seed)
        print(recipe.augmentation_id, 'matched bitwise at five seeds',flush=True)
        for jpeg in (False,True):
            if len(sys.argv)==4 and jpeg:
                continue
            m=measure([lambda arm=arm:run(arm,jpeg=jpeg) for arm in routes.values()],18 if not jpeg else 16,16,11)
            w.writerow([recipe.augmentation_id,'JPEG' if jpeg else 'decoded',*m])
            print(recipe.augmentation_id,'JPEG' if jpeg else 'decoded',dict(zip(routes,m)),flush=True)
