import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.other import annotation_artifacts_functional as fannotation
from tests.helpers import TestDataFactory


@pytest.mark.parametrize("element_type", ["text", "rectangle", "arrow", "line", "callout"])
def test_annotation_artifacts_each_element_type_changes_image(element_type: str) -> None:
    image = np.full((128, 128, 3), 137, dtype=np.uint8)
    transform = A.Compose(
        [
            A.AnnotationArtifacts(
                element_types=(element_type,),
                element_probabilities=(1.0,),
                count_range=(3, 3),
                p=1,
            ),
        ],
        seed=137,
        strict=True,
    )

    result = transform(image=image)["image"]

    assert result.dtype == image.dtype
    assert result.shape == image.shape
    assert not np.array_equal(result, image)


def test_annotation_artifacts_deterministic_with_compose_seed() -> None:
    image = TestDataFactory.create_image((160, 160, 3), dtype=np.uint8, seed=137)
    transform = A.Compose([A.AnnotationArtifacts(count_range=(5, 5), p=1)], seed=137, strict=True)
    same_seed_transform = A.Compose([A.AnnotationArtifacts(count_range=(5, 5), p=1)], seed=137, strict=True)

    result = transform(image=image)["image"]
    same_seed_result = same_seed_transform(image=image)["image"]

    np.testing.assert_array_equal(result, same_seed_result)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
def test_annotation_artifacts_dtype_and_channels(dtype: type[np.generic], num_channels: int) -> None:
    if dtype == np.uint8:
        image = np.full((96, 96, num_channels), 137, dtype=np.uint8)
    else:
        image = np.full((96, 96, num_channels), 0.5, dtype=np.float32)

    transform = A.Compose([A.AnnotationArtifacts(count_range=(4, 4), p=1)], seed=137, strict=True)
    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not np.array_equal(result, image)


@pytest.mark.parametrize("shape", [(2, 2, 1), (2, 2, 3), (4, 3, 5)])
def test_annotation_artifacts_tiny_images_do_not_fail(shape: tuple[int, int, int]) -> None:
    image = np.full(shape, 137, dtype=np.uint8)
    transform = A.Compose([A.AnnotationArtifacts(count_range=(5, 5), p=1)], seed=137, strict=True)

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_annotation_artifacts_white_color_affects_extra_channels() -> None:
    image = np.zeros((16, 16, 5), dtype=np.uint8)
    artifacts = [
        {
            "type": "line",
            "start": (1, 8),
            "end": (14, 8),
            "color": (255, 255, 255),
            "thickness": 1,
            "style": "solid",
        },
    ]

    result = fannotation.draw_annotation_artifacts(image, artifacts)

    assert np.any(result[..., 3] > 0)
    assert np.any(result[..., 4] > 0)


def test_annotation_artifacts_line_length_range_controls_lines() -> None:
    image = np.full((100, 100, 3), 137, dtype=np.uint8)
    transform = A.AnnotationArtifacts(
        element_types=("line",),
        element_probabilities=(1.0,),
        count_range=(10, 10),
        line_length_ratio_range=(0.25, 0.25),
        p=1,
    )
    transform.set_random_seed(137)

    artifacts = transform.get_params_dependent_on_data({"shape": image.shape}, {"image": image})["artifacts"]
    lengths = np.array(
        [
            int(np.hypot(artifact["end"][0] - artifact["start"][0], artifact["end"][1] - artifact["start"][1]))
            for artifact in artifacts
        ],
    )

    np.testing.assert_array_equal(lengths, np.full(10, 25))


@pytest.mark.parametrize(
    "params",
    [
        {"element_types": ("text", "line"), "element_probabilities": (1.0,)},
        {"element_types": ("text",), "element_probabilities": (-1.0,)},
        {"element_types": ("text",), "element_probabilities": (0.0,)},
        {"element_types": ("unknown",), "element_probabilities": (1.0,)},
        {"count_range": (3, 1)},
        {"font_scale_range": (0.0, 1.0)},
        {"corner_prob": 1.5},
        {"black_white_prob": -0.1},
    ],
)
def test_annotation_artifacts_validation_errors(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        A.AnnotationArtifacts(**params)
