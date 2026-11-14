import ctypes as ct
import csv
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import median
from timeit import Timer

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import cv2 as cv
import numpy as np

cv.setNumThreads(0)
print(platform.platform(), sys.version, version('opencv-python-headless'), np.__version__, 'threads', cv.getNumThreads(), 'IPP', cv.ipp.useIPP(), cv.ipp.useIPP_NotExact(), file=sys.stderr)
print('OpenCV build features', [s.strip() for s in cv.getBuildInformation().splitlines() if any(f in s for f in ('Custom HAL','Intel IPP','Baseline:','Dispatched code generation:'))],file=sys.stderr)
lib = ct.CDLL(str(Path(__file__).resolve().with_name('area.so')))
area = lib.area
area.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_ssize_t, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_int]
area.restype = ct.c_int

def resized(image, size, mode):
    ih, iw, c = image.shape
    oh, ow = size
    if image.strides[-2:] != (c, 1):
        image = np.ascontiguousarray(image)
    dst = np.empty((oh, ow, c), np.uint8)
    area(image.ctypes.data, ih, iw, image.strides[0], oh, ow, c, dst.ctypes.data, mode)
    return dst

def opencv(image, size, interpolation=cv.INTER_AREA):
    return cv.resize(image, size[::-1], interpolation=interpolation).reshape((*size, image.shape[-1]))

rng = np.random.default_rng(137)
cases = [(480,640,224,224,3), (480,640,240,320,3), (1080,1920,512,910,3), (1024,1024,512,512,3), (480,640,224,224,1), (480,640,224,224,4), (512,768,224,224,3), (483,641,224,224,3)]
output = csv.writer(sys.stdout)
selected = (3, 6, 8) if sys.platform == 'darwin' else (2, 7)
output.writerow(('case','OpenCV area','OpenCV linear','OpenCV area reused dst','tables only','variants ' + str(selected)))
frac = []
for ih,iw,oh,ow,c in cases:
    raw = rng.integers(0,256,(ih,iw,c),dtype=np.uint8)
    size = (oh,ow)
    dst = np.empty((*size,c),np.uint8)
    tag = f'{ih}x{iw}->{oh}x{ow} C{c}'
    if ih%oh or iw%ow:
        frac.append((ih,iw,oh,ow,c))
        counts = [np.count_nonzero(opencv(raw,size)!=resized(raw,size,mode)) for mode in selected]
        print('mismatches', tag, counts, file=sys.stderr, flush=True)
    methods = [lambda:opencv(raw,size), lambda:opencv(raw,size,cv.INTER_LINEAR),
               lambda:cv.resize(raw,size[::-1],dst=dst,interpolation=cv.INTER_AREA),
               lambda:area(None,ih,iw,0,oh,ow,c,None,-1)]
    if ih%oh or iw%ow:
        methods += [lambda mode=mode:resized(raw,size,mode) for mode in selected]
    timers = [Timer(fn) for fn in methods]
    counts = [max(6,min(1500,int(.07/(t.timeit(5)/5)))) for t in timers]
    series = [[] for t in timers]
    for rep in range(7):
        for i in range(len(timers))[::1 if rep%2==0 else -1]:
            series[i].append(timers[i].timeit(counts[i])*1e6/counts[i])
    output.writerow((tag, *(round(median(s),1) for s in series)))
    sys.stdout.flush()
errors = [0] * len(selected)
checks = 0
for ih,iw,oh,ow,c in frac + [(*[int(j) for j in dims], 3) for dims in rng.integers(224, 1000, (280,4)) if dims[0]>dims[2] and dims[1]>dims[3]]:
    for layout in ('dense','roi','reversed'):
        for seed in (0,137,731,16476):
            image = np.random.default_rng(seed).integers(0,256,(ih,iw+4,c),dtype=np.uint8)[:,2:-2]
            if layout == 'dense':
                image = image.copy()
            if layout == 'reversed':
                image = image[:,::-1]
            image.setflags(write=False)
            expected = opencv(image,(oh,ow))
            for i, mode in enumerate(selected):
                errors[i] += np.count_nonzero(expected != resized(image,(oh,ow),mode))
            checks += 1
print('parity:',checks,'cases; mismatched channel values by mode',dict(zip(selected,errors)),file=sys.stderr,flush=True)
target = rng.integers(0,256,(480,640,3),dtype=np.uint8)
if cv.ipp.useIPP():
    def sample_cv():
        return median(Timer(lambda:opencv(target,(224,224))).repeat(5,180))*1e6/180
    before = sample_cv()
    cv.ipp.setUseIPP(False)
    after = sample_cv()
    print('IPP control area on/off in us', before, after, file=sys.stderr,flush=True)
