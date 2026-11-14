import csv
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import median
from timeit import Timer
from unittest.mock import patch

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import albucore as core
import albumentations as A
import cv2 as cv
import numpy as np
import torch
from albucore import arithmetic
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
print(platform.platform(),sys.version.split()[0],A.__file__,core.__file__,{p: version(p) for p in ('numpy','opencv-python-headless','torch','numkong','simplejpeg')},flush=True)
print('threads', cv.getNumThreads(),torch.get_num_threads(),torch.get_num_interop_threads(),flush=True)

baseline = arithmetic._add_channels
def prospective(img, value, *, out=None):
    if (
        img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] == 3
        and img.strides[1:] == (3, 1) and img.strides[0] >= img.shape[1] * 3
        and (out is None or out.flags.c_contiguous)
    ):
        red, green, blue = map(float, value)
        if -255 <= red <= 255 and -255 <= green <= 255 and -255 <= blue <= 255:
            dst = None if out is None else out.reshape(-1).reshape(img.shape)
            result = cv.add(img, (int(red), int(green), int(blue), 0), dst=dst)
            return result.reshape(img.shape) if out is None else out
    return baseline(img,value,out=out)

if len(sys.argv) == 6:
    source = Path(sys.argv[5]).read_text()
    ns = vars(arithmetic).copy()
    exec(compile(source,sys.argv[5],'exec'),ns)
    prospective = ns['_add_channels']

catalog, fixtures, output, checks_file = map(Path,sys.argv[1:5])
fixtures.mkdir(parents=True,exist_ok=True)
rng = np.random.default_rng(137)
paths = [fixtures/f'{i:02d}.jpg' for i in range(16)]
for i,path in enumerate(paths):
    arr = rng.integers(0,256,(480+i*3,640-i*2,3),dtype=np.uint8)
    assert cv.imwrite(str(path),arr,[cv.IMWRITE_JPEG_QUALITY,95])
def decode():
    return [load_albumentationsx_rgb(SourceRef(p)) for p in paths]

images = decode()
recipe=next(r for r in load_recipe_catalog(catalog/'catalog/recipes/rgb.yaml').recipes if r.augmentation_id=='RGBShift')
pipeline=build_recipe(recipe, 'albumentationsx_cpu').cpu_transform
assert isinstance(pipeline.transforms[-1], A.ToTensorV2)
def run(seed, jpeg):
    data=decode() if jpeg else images
    return _collate_images([pipeline(image=image,invocation_seed=seed+i)['image'] for i,image in enumerate(data)])

def run_core(image, field):
    return core.add(image,field)

def with_route(route, call):
    with patch.object(arithmetic,'_add_channels',route):
        return call()

cases = 0
vectors = [np.array(v,dtype=np.float32) for v in (
    (0,0,0), (0.99,-0.99,-1.0), (-254.999,254.999,1.75), (-255,255,127),
    (500,-600,-0.25), (float('inf'),-float('inf'),float('nan')),
)]
almost_one = np.nextafter(np.longdouble(1), np.longdouble(0))
vectors.append(np.array([almost_one, -almost_one, 1], dtype=np.longdouble))
vectors.extend(rng.uniform(-255,255,(50,3)).astype('f4'))
with np.errstate(invalid='ignore',over='ignore'):
    for shape in ((1,1,3),(1,2,3),(1,4,3),(1,257,3),(2,1,3),(256,256,3),(3,9,14,3)):
        src = rng.integers(0,256,shape,dtype=np.uint8)
        if shape == (1,257,3):
            src[:] = (np.arange(257,dtype=np.int16) % 256).astype('u1')[None,:,None]
        for img in (src, src[...,::2,:], src[...,::-1,:], src[...,::-1]):
            for vector in vectors:
                for storage in ('new','alias','separate','strided'):
                    a=img.copy() if storage=='alias' else img
                    b=img.copy() if storage=='alias' else img
                    da=a if storage=='alias' else np.empty((*img.shape[:-1],6),dtype='u1')[...,::2] if storage=='strided' else np.empty_like(img) if storage=='separate' else None
                    db=b if storage=='alias' else np.empty((*img.shape[:-1],6),dtype='u1')[...,::2] if storage=='strided' else np.empty_like(img) if storage=='separate' else None
                    expect=with_route(baseline,lambda:core.add(a,vector,out=da))
                    actual=with_route(prospective,lambda:core.add(b,vector,out=db))
                    if db is not None: assert actual is db
                    np.testing.assert_array_equal(expect,actual,strict=True,err_msg=str((img.shape,img.strides,vector,storage)))
                    cases+=1

for dtype in (np.uint8,np.float16,np.float32):
    for target,shape in (('image',(19,23,3)),('images',(4,19,23,3)),('volume',(4,19,23,3)),('volumes',(2,4,19,23,3))):
        x=rng.integers(0,256,shape,dtype=np.uint8)
        if dtype != np.uint8: x=(x.astype('f4')/255).astype(dtype)
        x.setflags(write=False)
        aug=A.Compose([A.RGBShift(p=1)],telemetry=False)
        for seed in (0,137,731,1761,16476):
            fn=lambda:aug(**{target:x},invocation_seed=seed)[target]
            a,b=(with_route(arm,fn) for arm in (baseline,prospective))
            np.testing.assert_array_equal(a.view(f'u{a.itemsize}'),b.view(f'u{b.itemsize}'),strict=True)
            cases+=1
for seed in (0,137,731,1761,16476):
    for jpeg in (False,True):
        a,b=(with_route(arm,lambda:run(seed,jpeg)) for arm in (baseline,prospective))
        assert torch.equal(a,b)
checks_file.write_text(f'{cases} kernel/transform cases and 10 catalog batch comparisons matched bitwise.\n')
print(checks_file.read_text().strip(),flush=True)

def pair(fn,count,batch_size,rounds=13):
    values=([],[])
    timer=Timer(fn)
    for i in range(rounds):
        for j in (range(2) if i%2==0 else (1,0)):
            with patch.object(arithmetic,'_add_channels',(baseline,prospective)[j]):
                timer.timeit(2)
                values[j].append(timer.timeit(count)*1e6/count/batch_size)
    return list(map(median, values)), median(b/a for a,b in zip(*values))

with output.open('w',newline='',buffering=1) as result:
    w=csv.writer(result)
    w.writerow(['route','before_us','after_us','median_paired_ratio'])
    for jpeg in (False, True):
        m,ratio=pair(lambda:run(137,jpeg),24,len(images),17)
        w.writerow(['R14 JPEG' if jpeg else 'R14 decoded',*m,ratio])
    for size in (16,224,256,512,1024):
        raw = rng.integers(0,256,(size,size+10,3),dtype=np.uint8)
        color = np.array([99.9,0.9,-99.9],dtype=np.float32)
        for label,image in (('dense',raw[:,2:-8].copy()),('roi',raw[:,2:-8])):
            m,ratio=pair(lambda:run_core(image,color),max(100,160000//size),1,9)
            w.writerow([f'Core {size} {label}',*m,ratio])
