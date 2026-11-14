import inspect
import os
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import cv2
import numpy as np
from statistics import median
from timeit import Timer
from unittest.mock import patch
import albumentations as A
from albumentations.augmentations.pixel import dithering_functional as f
cv2.setNumThreads(0)
old=f.ordered_dither_uint8
s=inspect.getsource(old)
marker='return ((img > tiled[:, :, np.newaxis]) * 255).astype(np.uint8)'
assert s.count(marker)==1
n=vars(f).copy()
exec(s.replace(marker,'return np.multiply(img > tiled[:, :, np.newaxis], 255, dtype=np.uint8)'),n)
new=n[old.__name__]
rng=np.random.default_rng(137)
for shape in ((32,32,1),(32,32,3),(224,224,1),(224,224,3),(512,512,3),(1024,1024,3)):
 img=rng.integers(0,256,(shape[0],shape[1]+2,shape[2]),dtype=np.uint8)[:,1:-1]
 a,b=[fn(img,2,4) for fn in (old,new)]
 np.testing.assert_array_equal(a,b,strict=True)
 vals=[[],[]]
 timers=[Timer(lambda fn=fn:fn(img,2,4)) for fn in (old,new)]
 count=100 if shape[0]<512 else 20
 for i in range(7):
  for j in ((0,1) if not i%2 else (1,0)):
   vals[j].append(timers[j].timeit(count)*1e6/count)
 print('kernel',shape,*[round(median(v),2) for v in vals],flush=True)
 pipeline=A.Compose([A.Dithering(method='ordered',color_mode='per_channel',n_colors=2,p=1)],telemetry=False)
 def run(fn):
  with patch.object(f,'ordered_dither_uint8',fn):
   return pipeline(image=img,invocation_seed=137)['image']
 np.testing.assert_array_equal(run(old),run(new),strict=True)
 timers=[Timer(lambda fn=fn:run(fn)) for fn in (old,new)]
 vals=[[],[]]
 for i in range(9):
  for j in ((0,1) if not i%2 else (1,0)):
   vals[j].append(timers[j].timeit(count)*1e6/count)
 print('Compose',shape,*[round(median(v),2) for v in vals],flush=True)
