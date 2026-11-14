import ast
import csv
import importlib
import os
from pathlib import Path
from statistics import median
import sys
from timeit import Timer
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albucore
import albumentations as A
import cv2 as cv
import numpy as np
import torch

cv.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(albucore.__file__, A.__file__, cv.__version__, np.__version__, torch.__version__, cv.getNumThreads(), flush=True)

def candidate(module, name, replacements):
    fn = getattr(module, name)
    file = Path(fn.__code__.co_filename)
    tree = ast.parse(file.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    source = ast.get_source_segment(file.read_text(), node)
    for old, new in replacements:
        assert old in source, (name, old)
        source = source.replace(old, new)
    ns = vars(module).copy()
    ns.update(cv2=cv,np=np)
    exec(compile("from __future__ import annotations\n" + source, str(file), "exec"), ns)
    return fn, ns[name]

P="albumentations.augmentations.pixel."
G="albumentations.augmentations."
cases = [
("draw",G+"other.annotation_artifacts_functional","_draw_channelwise",
 [("np.ascontiguousarray(image[..., channel_index])","cv2.extractChannel(image, channel_index)"),
  ("image[..., channel_index] = channel","cv2.insertChannel(channel, image, channel_index)")]),
("hue",P+"_functional_torchvision","_adjust_hue_torchvision_uint8",
 [("img[..., 0] = apply_uint8_lut(img[..., 0], lut)","hue = cv2.extractChannel(img, 0)\n    apply_uint8_lut(hue, lut, out=hue)\n    cv2.insertChannel(hue, img, 0)")]),
("equalize_luma",P+"_functional_color","equalize",
 [("ycrcb[..., :1] = albucore.equalize_uint8(ycrcb[..., :1], mode=mode, mask=selection)",
   "plane = cv2.extractChannel(ycrcb, 0)\n        plane = albucore.equalize_uint8(plane[..., None], mode=mode, mask=selection)\n        cv2.insertChannel(plane, ycrcb, 0)")]),
("iso",P+"_functional_color","iso_noise",
 [("hls[..., 1] = add(\n        hls[..., 1],","lightness = cv2.extractChannel(hls, 1)\n    lightness = add(\n        lightness,"),
  ("(1.0 - hls[..., 1])","(1.0 - lightness)"),("    result = cast","    cv2.insertChannel(lightness, hls, 1)\n    result = cast")]),
("snow",P+"_functional_weather","add_snow_bleach",
 [("image_hls[:, :, 1].astype(np.float32)","cv2.extractChannel(image_hls, 1).astype(np.float32, copy=False)"),
  ("image_hls[:, :, 1] = (","lightness_channel = ("),
  ("    result = cast(","    cv2.insertChannel(lightness_channel, image_hls, 1)\n    result = cast(")]),
("ordered",P+"dithering_functional","ordered_dither_uint8",
 [("result[:, :, channel_idx] = apply_uint8_lut(dithered.astype(np.uint8), lut)",
   "cv2.insertChannel(apply_uint8_lut(dithered.astype(np.uint8), lut), result, channel_idx)")]),
("diffusion",P+"dithering_functional","error_diffusion_dither",
 [("result[:, :, channel_idx].copy()","cv2.extractChannel(result, channel_idx)"),
  ("result[:, :, channel_idx] = channel","cv2.insertChannel(channel, result, channel_idx)")]),
("autocontrast",P+"_functional_illumination","_auto_contrast_multichannel_hist",
 [("result[..., channel_idx] = apply_uint8_lut(cast(\"ImageUInt8\", channel), lut, out=cast(\"ImageUInt8\", channel))",
   "apply_uint8_lut(cast(\"ImageUInt8\", channel), lut, out=cast(\"ImageUInt8\", channel))\n        cv2.insertChannel(channel, result, channel_idx)")]),
("chromatic",P+"_functional_sharpness","chromatic_aberration",
 [("img[..., 0],","cv2.extractChannel(img, 0),"),("img[..., 2],","cv2.extractChannel(img, 2),"),
  ("np.dstack([red_distorted, img[..., 1], blue_distorted])","cv2.merge((red_distorted, cv2.extractChannel(img, 1), blue_distorted))")]),
("histmatch",G+"mixing.domain_adaptation_functional","match_histograms",
 [("image[..., channel], reference[..., channel]","cv2.extractChannel(image, channel), cv2.extractChannel(reference, channel)"),
  ("matched[..., channel] = matched_channel","cv2.insertChannel(matched_channel.reshape(image.shape[:2]), matched, channel)")]),
("fourier",G+"mixing.domain_adaptation_functional","fourier_domain_adaptation",
 [("src_in_trg[:, :, channel_id] = np.real(src_in_trg_channel)",
   "cv2.insertChannel(np.real(src_in_trg_channel).astype(np.float32), src_in_trg, channel_id)")]),
("filter","albucore.filtering","convolve",
 [("        for channel, channel_kernel in enumerate(kernel):","        plane = np.empty(working.shape[:2], dtype=working.dtype)\n        for channel, channel_kernel in enumerate(kernel):\n            cv2.extractChannel(working, channel, dst=plane)"),
  ("result[..., channel] = cv2.filter2D(\n                working[..., channel],","cv2.filter2D(\n                plane,\n                dst=plane,"),
  ("                borderType=border_mode,\n            )","                borderType=border_mode,\n            )\n            cv2.insertChannel(plane, result, channel)")]),
("equalize_many","albucore.histogram","_equalize_cv_unmasked",
 [("    for channel in range(channels):","    plane = np.empty(img.shape[:2], dtype=img.dtype)\n    for channel in range(channels):"),
  ("result[..., channel] = cv2.equalizeHist(img[..., channel])","cv2.extractChannel(img, channel, dst=plane)\n        cv2.equalizeHist(plane, dst=plane)\n        cv2.insertChannel(plane, result, channel)")]),
("equalize_masked","albucore.histogram","equalize_uint8",
 [("    for channel in range(channels):","    scratch = np.empty(img.shape[:2], dtype=img.dtype)\n    for channel in range(channels):"),
  ("plane = img[..., channel]","plane = cv2.extractChannel(img, channel, dst=scratch)"),
  ("result[..., channel] = plane if lut is None else apply_uint8_lut(plane, lut)","if lut is not None:\n            apply_uint8_lut(plane, lut, out=plane)\n        cv2.insertChannel(plane, result, channel)")]),
("lut_f32","albucore.lut","_apply_float_lut",
 [("result[..., i] = cv2.LUT(img[..., i], lut[:, 0, i])","plane = cv2.extractChannel(img, i)\n        cv2.insertChannel(cv2.LUT(plane, lut[:, 0, i]), result, i)")]),
("lut_u8","albucore.lut","_apply_per_channel_uint8_luts",
 [("    for i in range(num_channels):","    plane = np.empty(img.shape[:2], dtype=img.dtype)\n    for i in range(num_channels):"),
  ("apply_uint8_lut(img[..., i], np.ascontiguousarray(luts[:, 0, i]), out=result[..., i])",
   "cv2.extractChannel(img, i, dst=plane)\n        apply_uint8_lut(plane, np.ascontiguousarray(luts[:, 0, i]), out=plane)\n        cv2.insertChannel(plane, result, i)")]),
]

if len(sys.argv)>2:
    cases = [c for c in cases if c[0] in sys.argv[2:]]

def pair(a,b):
    timers=[Timer(c) for c in (a,b)]
    counts=[max(3,min(2500,int(0.009/(t.timeit(3)/3)))) for t in timers]
    draws=[[],[]]
    for i in range(7):
        for k in ((0,1) if i%2==0 else (1,0)):
            draws[k].append(timers[k].timeit(counts[k])*1e6/counts[k])
    return [median(x) for x in draws]

rng = np.random.default_rng(137)
def compare_bits(a,b):
    np.testing.assert_array_equal(a.view(f"u{a.itemsize}"),b.view(f"u{b.itemsize}"),strict=True)

def input_shape(n,c,layout):
    if layout=="planar":
        return rng.integers(0,256,(c,n,n),dtype=np.uint8).transpose(1,2,0)
    raw=rng.integers(0,256,(n,n+4,c),dtype=np.uint8)
    pixels=raw[:,2:-2]
    return pixels.copy() if layout=="dense" else pixels

with open(sys.argv[1],"w",newline="",buffering=1) as out:
    writer=csv.writer(out)
    writer.writerow(("operation","size","channels","dtype","layout","base_us","new_us"))
    for label, path, name, replacements in cases:
        print(label,flush=True)
        module = importlib.import_module(path)
        if label=="equalize_luma":
            replacements=[(x.replace("mode=mode","mode=mode"),y) for x,y in replacements]
        try:
            baseline, changed = candidate(module,name,replacements)
        except (AssertionError, TypeError) as exc:
            print("source mismatch",exc,flush=True)
            continue
        sizes=(64,256,512,1024) if label!="diffusion" else (32,64)
        for n in sizes:
            channels=(1,3,5) if label in ("draw","ordered","autocontrast","histmatch","fourier","filter","equalize_many","equalize_masked") else (3,)
            for c in channels:
                if label=="equalize_many" and c<5: continue
                for layout in (("planar","crop","dense") if label.startswith("lut") else ("dense","crop")):
                    source=input_shape(n,c,layout)
                    for dtype in ((np.uint8,np.float32) if label in ("draw","equalize_luma","iso","snow","chromatic","histmatch","fourier","filter") else (np.uint8,)):
                        img=source if dtype==np.uint8 else (source.astype(dtype)/255)
                        ref=img[::-1].copy()
                        lut=np.ascontiguousarray(rng.integers(0,256,(256,1,c),dtype=np.uint8))
                        mask=(source[...,0]>40).astype(np.uint8)
                        kernel=rng.random((c,3,3),dtype=np.float32)/9
                        def call(fn):
                            if label=="draw":
                                # Measure the drawing transform's complete functional entry point.
                                artifacts=[{"type":"rectangle","top_left":(2,2),"bottom_right":(n-2,n-2),"color":(130,240,80),"thickness":2}]
                                with patch.object(module,name,fn):
                                    return module.draw_annotation_artifacts(img, artifacts)
                            if label=="hue": return fn(img,.2)
                            if label=="equalize_luma": return fn(img,mode="cv",by_channels=False)
                            if label=="iso": return fn(img,.05,.5,np.random.default_rng(137))
                            if label=="snow": return fn(img,.2,2.5)
                            if label=="ordered": return fn(img,4)
                            if label=="diffusion": return fn(img,4,"floyd_steinberg",True)
                            if label=="autocontrast":
                                with patch.object(module,name,fn):
                                    return module.auto_contrast(img,1,33,"pil")
                            if label=="chromatic": return fn(img,.1,.06,-.1,-.06,cv.INTER_LINEAR)
                            if label=="histmatch": return fn(img,ref)
                            if label=="fourier": return fn(img,ref,.05)
                            if label=="filter": return fn(img,kernel)
                            if label=="equalize_many":
                                with patch.object(module,name,fn):
                                    return module.equalize_uint8(img)
                            if label=="equalize_masked": return fn(img,mode="pil",mask=mask)
                            if label=="lut_f32":
                                with patch.object(module,name,fn):
                                    return module._apply_float_lut(img,lut.astype(np.float32),out=None) if fn is baseline else fn(img,lut.astype(np.float32),out=None)
                            if label=="lut_u8":
                                with patch.object(module,name,fn):
                                    return module.apply_uint8_lut(img,lut)
                            raise AssertionError(label)
                        try:
                            before,after=call(baseline),call(changed)
                            compare_bits(before,after)
                        except (AssertionError,ValueError,TypeError,cv.error) as exc:
                            print("failed",label,n,c,dtype.__name__,layout,str(exc).splitlines()[:3],flush=True)
                            continue
                        if label=="diffusion" and n==256:
                            continue
                        writer.writerow((label,n,c,dtype.__name__,layout,*pair(lambda:call(baseline),lambda:call(changed))))
