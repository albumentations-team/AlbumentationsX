"""Bias-field generation and application benchmarks for images and true volumes."""

from __future__ import annotations

import albumentations
from albumentations.augmentations.pixel import _functional_noise as fnoise
from benchmarks.common import DTYPES, SIZES, VOLUME_SIZES, make_image, make_volume

BIAS_FIELD_CASES = tuple(
    f"{target}|{size}|{channels}|{dtype}|{per_channel}"
    for target, sizes in (("image", SIZES), ("volume", VOLUME_SIZES))
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
        factory = make_image if target == "image" else make_volume
        self.data = factory(size, int(channels), DTYPES[dtype])
        kwargs = {"std_range": (0.25, 0.25), "per_channel": per_channel == "True", "p": 1}
        self.pipeline = albumentations.Compose([albumentations.BiasField(**kwargs)], seed=137, strict=True)
        capture = albumentations.ReplayCompose([albumentations.BiasField(**kwargs)])
        capture.set_random_seed(137)
        groups = capture(**{target: self.data})["replay"]["transforms"][0]["params"]["target_params"]
        self.coarse_field = groups[0]["params"]["coarse_field"]

    def time_generate_field(self, case_id: str) -> None:
        fnoise.generate_bias_field(self.coarse_field, self.data.shape[:-1])

    def time_kernel(self, case_id: str) -> None:
        fnoise.bias_field(self.data, self.coarse_field)

    def time_compose(self, case_id: str) -> None:
        self.pipeline(**{self.target: self.data})

    def peakmem_compose(self, case_id: str) -> None:
        self.pipeline(**{self.target: self.data})
