"""Core Compose and ReplayCompose benchmarks."""

from __future__ import annotations

import albumentations
from benchmarks.common import (
    ANNOTATION_COUNTS,
    IMAGE_PARAMS,
    make_batch,
    make_hbb_bboxes,
    make_image,
    make_keypoints,
    make_labels,
    make_mask,
)


class TimeCorePipeline:
    """Benchmark core Compose and ReplayCompose execution paths."""

    params = IMAGE_PARAMS
    param_names = ("size_name", "channels")

    def setup(self, size_name: str, channels: int) -> None:
        self.image = make_image(size_name, channels)
        self.image2 = self.image.copy()
        self.mask = make_mask(size_name)
        self.mask2 = self.mask.copy()
        self.images = make_batch(size_name, channels, batch_size=4)
        self.single = albumentations.Compose([albumentations.HorizontalFlip(p=1.0)], strict=True)
        self.skip = albumentations.Compose([albumentations.HorizontalFlip(p=0.0)], strict=True)
        self.additional_targets = albumentations.Compose(
            [albumentations.HorizontalFlip(p=1.0)],
            additional_targets={"image2": "image", "mask2": "mask"},
            strict=True,
        )
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

    def time_skip_transform_compose(self, size_name: str, channels: int) -> None:
        self.skip(image=self.image)

    def time_multi_transform_compose(self, size_name: str, channels: int) -> None:
        self.multi(image=self.image)

    def time_additional_targets_compose(self, size_name: str, channels: int) -> None:
        self.additional_targets(image=self.image, image2=self.image2, mask=self.mask, mask2=self.mask2)

    def time_batch_image_compose(self, size_name: str, channels: int) -> None:
        self.single(images=self.images)

    def time_replay_compose(self, size_name: str, channels: int) -> None:
        self.replay(image=self.image)

    def peakmem_batch_image_compose(self, size_name: str, channels: int) -> None:
        self.single(images=self.images)


class TimeCorePipelineSetup:
    """Benchmark Compose construction and processor setup costs."""

    def time_single_transform_compose_setup(self) -> None:
        albumentations.Compose([albumentations.HorizontalFlip(p=1.0)], strict=True)

    def time_multi_transform_compose_setup(self) -> None:
        albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=1.0),
                albumentations.VerticalFlip(p=1.0),
                albumentations.RandomRotate90(p=1.0),
            ],
            seed=137,
            strict=True,
        )

    def time_bbox_keypoint_processor_setup(self) -> None:
        albumentations.Compose(
            [albumentations.NoOp(p=1.0)],
            bbox_params=albumentations.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
            keypoint_params=albumentations.KeypointParams(
                coord_format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={},
                remove_invisible=False,
            ),
            strict=True,
        )


class TimeCorePipelineTargetProcessors:
    """Benchmark bbox/keypoint processor overhead with a no-op transform."""

    params = (ANNOTATION_COUNTS,)
    param_names = ("count",)

    def setup(self, count: int) -> None:
        size_name = "large" if count >= 1000 else "medium" if count >= 100 else "small"
        self.transform = albumentations.Compose(
            [albumentations.NoOp(p=1.0)],
            bbox_params=albumentations.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
            keypoint_params=albumentations.KeypointParams(
                coord_format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={},
                remove_invisible=False,
            ),
            strict=True,
        )
        self.data = {
            "bbox_labels": make_labels(count),
            "bboxes": make_hbb_bboxes(size_name, count),
            "image": make_image(size_name, 3),
            "keypoint_labels": make_labels(count),
            "keypoints": make_keypoints(size_name, count),
        }

    def time_bbox_keypoint_processor_roundtrip(self, count: int) -> None:
        self.transform(**self.data)
