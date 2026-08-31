"""Reference-data transform benchmark smoke path."""

from __future__ import annotations

import albumentations
from benchmarks.common import make_image


class TimeMixingTransforms:
    """Benchmark reference-data mixing transforms."""

    def setup(self) -> None:
        self.image = make_image("small", 3)
        self.metadata = [{"image": self.image.copy()} for _ in range(4)]
        self.transform = albumentations.Compose(
            [
                albumentations.Mosaic(
                    grid_yx=(2, 2),
                    target_size=(128, 128),
                    cell_shape=(128, 128),
                    p=1.0,
                ),
            ],
            seed=137,
            strict=True,
        )

    def time_mosaic(self) -> None:
        self.transform(image=self.image, mosaic_metadata=self.metadata)

    def peakmem_mosaic(self) -> None:
        self.transform(image=self.image, mosaic_metadata=self.metadata)
