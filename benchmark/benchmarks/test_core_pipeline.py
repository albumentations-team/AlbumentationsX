"""Core Compose and ReplayCompose benchmarks."""

from __future__ import annotations

import albumentations
from benchmarks.common import IMAGE_PARAMS, make_batch, make_image


class TimeCorePipeline:
    """Benchmark core Compose and ReplayCompose execution paths."""

    params = IMAGE_PARAMS
    param_names = ("size_name", "channels")

    def setup(self, size_name: str, channels: int) -> None:
        self.image = make_image(size_name, channels)
        self.images = make_batch(size_name, channels, batch_size=4)
        self.single = albumentations.Compose([albumentations.HorizontalFlip(p=1.0)], strict=True)
        self.multi = albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=1.0),
                albumentations.VerticalFlip(p=1.0),
                albumentations.RandomRotate90(p=1.0),
            ],
            seed=137,
            strict=True,
        )
        self.replay = albumentations.ReplayCompose(
            [albumentations.HorizontalFlip(p=1.0), albumentations.RandomRotate90(p=1.0)],
            seed=137,
        )

    def time_single_transform_compose(self, size_name: str, channels: int) -> None:
        self.single(image=self.image)

    def time_multi_transform_compose(self, size_name: str, channels: int) -> None:
        self.multi(image=self.image)

    def time_batch_image_compose(self, size_name: str, channels: int) -> None:
        self.single(images=self.images)

    def time_replay_compose(self, size_name: str, channels: int) -> None:
        self.replay(image=self.image)

    def peakmem_batch_image_compose(self, size_name: str, channels: int) -> None:
        self.single(images=self.images)
