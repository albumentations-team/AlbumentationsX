import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'

import csv
import platform
import sys
from pathlib import Path
from statistics import median
from timeit import Timer
from unittest.mock import patch

import albucore
import albumentations as A
import cv2 as cv
import numpy as np
import torch
from albumentations.augmentations.pixel import _functional_illumination as f, functional as pixel
from augbench.adapters.sources import SourceRef, load_albumentationsx_rgb
from augbench.execution.pipeline import _collate_images
from augbench.implementations.albumentationsx_pipeline_impl import build_recipe
from augbench.recipes.load import load_recipe_catalog
from candidates import candidates
from numkong import get_capabilities

cv.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if sys.platform=='darwin': assert get_capabilities()['neon']
print(platform.platform(),sys.version.split()[0],A.__file__,albucore.__version__,np.__version__,cv.__version__,flush=True)
catalog, fixtures, output = map(Path,sys.argv[1:4])
fixtures.mkdir(exist_ok=True,parents=True)
rng=np.random.default_rng(137)
paths=[fixtures / f'{i:02d}.jpg' for i in range(16)]
for i,path in enumerate(paths):
    if not path.exists():
        assert cv.imwrite(str(path),rng.integers(0,256,(480+i*3,640-i*2,3),dtype=np.uint8),[cv.IMWRITE_JPEG_QUALITY,95])
def decode():
    return [load_albumentationsx_rgb(SourceRef(p)) for p in paths]
decoded=decode()
base=f.generate_plasma_pattern
routes=candidates(f)
if len(sys.argv)==5:
    ns=vars(f).copy()
    p=Path(sys.argv[4])
    exec(compile(p.read_text(),str(p),'exec',flags=0x1000000),ns)
    routes={'base':base,'final':ns[base.__name__]}

def paired(functions,count,mul=1):
    timers=[Timer(fn) for fn in functions]
    for t in timers: t.timeit(3)
    batches=[[] for t in timers]
    for rep in range(9):
        for i in range(len(timers))[::1 if rep%2==0 else -1]:
            batches[i].append(timers[i].timeit(count)*1e6/count/mul)
    return list(map(median,batches))

checks={}
for name,fn in list(routes.items())[1:]:
    mismatches=[]
    n=0
    for shape in ((1,7),(2,2),(3,3),(16,16),(63,92),(128,128),(224,224),(225,331),(512,512)):
        for seed in (0,137,731,1761,16476):
            for rough in (.1,.25,.5,1.0,3.0,9.3):
                g1,g2=(np.random.default_rng(seed) for _ in range(2))
                a,b=base(shape,rough,g1),fn(shape,rough,g2)
                if not np.array_equal(a.view('u4'),b.view('u4')) or g1.bit_generator.state!=g2.bit_generator.state:
                    mismatches.append((shape,seed,rough,np.count_nonzero(a.view('u4')!=b.view('u4'))))
                n+=1
    print(name,n,'parity checks; mismatches',len(mismatches),mismatches[:4],flush=True)
    checks[name]=not mismatches
if len(sys.argv)==5:
    assert all(checks.values())
else:
    routes={name:fn for name,fn in routes.items() if checks.get(name,True)}
with output.open('w',newline='',buffering=1) as stream:
    csvw=csv.writer(stream)
    csvw.writerow(('recipe','scope',*routes))
    for size in (32,128,224,256,512):
        times=paired([lambda fn=fn:fn((size,size),.5,np.random.default_rng(137)) for fn in routes.values()],60 if size<=256 else 22)
        csvw.writerow((size,'map',*times))
    for recipe in load_recipe_catalog(catalog/'catalog/recipes/rgb.yaml').recipes:
        if recipe.augmentation_id not in ('PlasmaContrast','PlasmaShadow'): continue
        pipeline=build_recipe(recipe,'albumentationsx_cpu').cpu_transform
        assert isinstance(pipeline.transforms[-1],A.ToTensorV2)
        def run(fn,seed=137,jpeg=False):
            with patch.object(pixel,'generate_plasma_pattern',fn):
                return _collate_images([pipeline(image=image,invocation_seed=seed+i)['image'] for i,image in enumerate(decode() if jpeg else decoded)])
        for seed in (0,137,731,1761,16476):
            for jpeg in ((False,True) if len(sys.argv)==5 else (False,)):
                a=run(base,seed,jpeg)
                for fn in routes.values(): assert torch.equal(a,run(fn,seed,jpeg)), (recipe.augmentation_id,seed,jpeg)
        print(recipe.augmentation_id,'bitwise at 5 seeds',flush=True)
        for jpeg in ((False,True) if len(sys.argv)==5 else (False,)):
            vals=paired([lambda fn=fn:run(fn,jpeg=jpeg) for fn in routes.values()],12 if not jpeg else 10,16)
            csvw.writerow((recipe.augmentation_id,'JPEG' if jpeg else 'decoded',*vals))
