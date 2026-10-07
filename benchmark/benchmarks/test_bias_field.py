"""Bias-field generation and application benchmarks for images and true volumes."""

from __future__ import annotations

import albumentations
from albumentations.augmentations.medical import functional as fmedical
from benchmarks.common import DTYPES, MRI_VOLUME_SIZES, SIZES, make_image, make_volume

BIAS_FIELD_CASES = tuple(
    f"{target}|{size}|{channels}|{dtype}|{per_channel}"
    for target, sizes in (("image", SIZES), ("volume", MRI_VOLUME_SIZES))
    for size in sizes
    for channels in (1, 3, 5)
    for dtype in DTYPES
    for per_channel in (False, True)
)


class TimeBiasField:
    """Measure the positive field generator, full kernel, and public route across sharing modes."""

    params = (BIAS_FIELD_CASES,)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        target, size, channels, dtype, per_channel = case_id.split("|")
        self.target = target
        self.data = (
            make_image(size, int(channels), DTYPES[dtype])
            if target == "image"
            else make_volume(size, int(channels), DTYPES[dtype], sizes=MRI_VOLUME_SIZES)
        )
        kwargs = {"std_range": (0.25, 0.25), "per_channel": per_channel == "True", "p": 1}
        self.pipeline = albumentations.Compose([albumentations.BiasField(**kwargs)], seed=137, strict=True)
        capture = albumentations.ReplayCompose([albumentations.BiasField(**kwargs)])
        capture.set_random_seed(137)
        groups = capture(**{target: self.data})["replay"]["transforms"][0]["params"]["target_params"]
        self.coarse_field = groups[0]["params"]["coarse_field"]

    def time_generate_field(self, case_id: str) -> None:
        fmedical.generate_bias_field(self.coarse_field, self.data.shape[:-1])

    def time_kernel(self, case_id: str) -> None:
        fmedical.bias_field(self.data, self.coarse_field)

    def time_compose(self, case_id: str) -> None:
        self.pipeline(**{self.target: self.data})

    def peakmem_compose(self, case_id: str) -> None:
        self.pipeline(**{self.target: self.data})
