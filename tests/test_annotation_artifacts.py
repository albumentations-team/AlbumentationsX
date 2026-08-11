import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.other import annotation_artifacts_functional as fannotation
from albumentations.core.invocation import SamplingContext
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


def test_annotation_artifacts_default_sampling_sequence_is_unchanged() -> None:
    transform = A.AnnotationArtifacts(
        element_types=("line", "arrow", "callout"),
        element_probabilities=(1.0, 1.0, 1.0),
        count_range=(6, 6),
        p=1,
    )
    transform.set_random_seed(137)

    artifacts = transform.sample_parameters(
        {"shape": (160, 160, 3)},
        {"image": np.empty((160, 160, 3), dtype=np.uint8)},
        SamplingContext.from_owner(transform, {}),
    )["artifacts"]

    assert artifacts == [
        {
            "type": "line",
            "start": (25, 130),
            "end": (149, 130),
            "color": (255, 255, 255),
            "thickness": 3,
            "style": "solid",
        },
        {
            "type": "line",
            "start": (123, 19),
            "end": (123, 110),
            "color": (255, 0, 0),
            "thickness": 1,
            "style": "solid",
        },
        {
            "type": "line",
            "start": (29, 87),
            "end": (153, 87),
            "color": (0, 0, 0),
            "thickness": 3,
            "style": "solid",
        },
        {
            "type": "arrow",
            "start": (132, 32),
            "end": (103, 109),
            "color": (255, 0, 0),
            "thickness": 3,
            "tip_length": 0.31288073889965917,
            "style": "dashed",
        },
        {
            "type": "line",
            "start": (31, 78),
            "end": (31, 121),
            "color": (0, 0, 0),
            "thickness": 3,
            "style": "dashed",
        },
        {
            "type": "arrow",
            "start": (129, 153),
            "end": (122, 107),
            "color": (255, 255, 255),
            "thickness": 1,
            "tip_length": 0.33652830990705285,
            "style": "dashed",
        },
    ]


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
    if dtype == np.float32:
        assert 0 <= result.min() <= result.max() <= 1
    assert not np.array_equal(result, image)


@pytest.mark.parametrize(
    ("target", "shape"),
    [("image", (32, 32)), ("images", (2, 32, 32))],
)
def test_annotation_artifacts_direct_grayscale_inputs(target: str, shape: tuple[int, ...]) -> None:
    data = np.full(shape, 137, dtype=np.uint8)
    transform = A.AnnotationArtifacts(
        element_types=("line",),
        element_probabilities=(1.0,),
        count_range=(4, 4),
        random_color_prob=1.0,
        p=1,
    )
    transform.set_random_seed(137)

    result = transform(**{target: data})[target]

    assert result.shape == data.shape
    assert result.dtype == data.dtype
    assert all(len(artifact["color"]) == 1 for artifact in transform.params["artifacts"])
    assert not np.array_equal(result, data)


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


@pytest.mark.parametrize("style", ["solid", "dashed", "dotted"])
def test_annotation_artifacts_line_geometry_is_preserved_at_image_border(style: str) -> None:
    artifact = {
        "type": "line",
        "start": (50, 20),
        "end": (123, 93),
        "color": (255,),
        "thickness": 1,
        "style": style,
    }

    small_result = fannotation.draw_annotation_artifacts(np.zeros((100, 100, 1), dtype=np.uint8), [artifact])
    large_result = fannotation.draw_annotation_artifacts(np.zeros((150, 150, 1), dtype=np.uint8), [artifact])

    np.testing.assert_array_equal(
        small_result[1:-1, 1:-1] > 0,
        large_result[1:99, 1:99] > 0,
    )


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

    artifacts = transform.sample_parameters(
        {"shape": image.shape},
        {"image": image},
        SamplingContext.from_owner(transform, {}),
    )["artifacts"]
    lengths = np.array(
        [
            int(np.hypot(artifact["end"][0] - artifact["start"][0], artifact["end"][1] - artifact["start"][1]))
            for artifact in artifacts
        ],
    )

    np.testing.assert_array_equal(lengths, np.full(10, 25))


def test_annotation_artifacts_random_endpoints_are_replayable() -> None:
    image = np.full((96, 128, 3), 137, dtype=np.uint8)
    transform = A.AnnotationArtifacts(
        element_types=("line",),
        element_probabilities=(1.0,),
        count_range=(20, 20),
        line_geometry="random_endpoints",
        p=1,
    )
    pipeline = A.ReplayCompose([transform], seed=137)

    result = pipeline(image=image)
    artifacts = transform.params["artifacts"]
    replayed = A.ReplayCompose.replay(result["replay"], image=image)

    assert any(
        artifact["start"][0] != artifact["end"][0] and artifact["start"][1] != artifact["end"][1]
        for artifact in artifacts
    )
    assert all(
        0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]
        for artifact in artifacts
        for point in (artifact["start"], artifact["end"])
    )
    np.testing.assert_array_equal(replayed["image"], result["image"])


@pytest.mark.parametrize("element_type", ["line", "arrow", "callout"])
def test_annotation_artifacts_line_style_policy_applies_to_all_styled_elements(element_type: str) -> None:
    image = np.full((96, 96, 3), 137, dtype=np.uint8)
    transform = A.AnnotationArtifacts(
        element_types=(element_type,),
        element_probabilities=(1.0,),
        count_range=(4, 4),
        line_styles=("dotted",),
        line_style_probabilities=(1.0,),
        p=1,
    )

    A.Compose([transform], save_applied_params=True, seed=137, strict=True)(image=image)

    assert {artifact["style"] for artifact in transform.params["artifacts"]} == {"dotted"}


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
def test_annotation_artifacts_random_colors_match_image_channels(
    dtype: type[np.generic],
    num_channels: int,
) -> None:
    fill = 137 if dtype == np.uint8 else 0.5
    image = np.full((96, 96, num_channels), fill, dtype=dtype)
    transform = A.AnnotationArtifacts(
        element_types=("line",),
        element_probabilities=(1.0,),
        count_range=(4, 4),
        random_color_prob=1.0,
        p=1,
    )

    result = A.Compose([transform], save_applied_params=True, seed=137, strict=True)(image=image)["image"]
    colors = [artifact["color"] for artifact in transform.params["artifacts"]]

    assert all(len(color) == num_channels for color in colors)
    assert all(0 <= value <= 255 for color in colors for value in color)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not np.array_equal(result, image)


def test_annotation_artifacts_weighted_color_palette() -> None:
    image = np.full((96, 96, 5), 137, dtype=np.uint8)
    transform = A.AnnotationArtifacts(
        element_types=("line",),
        element_probabilities=(1.0,),
        count_range=(4, 4),
        color_palette=((1, 2, 3, 4, 5), (10, 20, 30, 40, 50)),
        color_palette_probabilities=(0.0, 1.0),
        p=1,
    )

    A.Compose([transform], save_applied_params=True, seed=137, strict=True)(image=image)

    assert {artifact["color"] for artifact in transform.params["artifacts"]} == {(10, 20, 30, 40, 50)}


def test_annotation_artifacts_fixed_palette_extends_to_extra_channels() -> None:
    image = np.zeros((32, 32, 5), dtype=np.uint8)
    transform = A.AnnotationArtifacts(
        element_types=("line",),
        element_probabilities=(1.0,),
        count_range=(1, 1),
        color_palette=((255, 255, 255),),
        line_styles=("solid",),
        line_style_probabilities=(1.0,),
        p=1,
    )

    result = A.Compose([transform], seed=137, strict=True)(image=image)["image"]

    assert np.any(result[..., 3] > 0)
    assert np.any(result[..., 4] > 0)


def test_annotation_artifacts_random_angle_supports_pixel_lengths() -> None:
    image = np.full((128, 128, 3), 137, dtype=np.uint8)
    transform = A.AnnotationArtifacts(
        element_types=("line",),
        element_probabilities=(1.0,),
        count_range=(10, 10),
        line_geometry="random_angle",
        line_length_range=(20, 20),
        p=1,
    )

    A.Compose([transform], save_applied_params=True, seed=137, strict=True)(image=image)
    lengths = [
        np.hypot(
            artifact["end"][0] - artifact["start"][0],
            artifact["end"][1] - artifact["start"][1],
        )
        for artifact in transform.params["artifacts"]
    ]

    assert all(18.5 <= length <= 21.5 for length in lengths)


def test_annotation_artifacts_random_angle_has_nonzero_relative_length() -> None:
    image = np.full((4, 4, 1), 137, dtype=np.uint8)
    transform = A.AnnotationArtifacts(
        element_types=("line",),
        element_probabilities=(1.0,),
        count_range=(10, 10),
        line_geometry="random_angle",
        line_length_ratio_range=(0.1, 0.1),
        p=1,
    )

    A.Compose([transform], save_applied_params=True, seed=137, strict=True)(image=image)

    assert all(artifact["start"] != artifact["end"] for artifact in transform.params["artifacts"])


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
        {"line_geometry": "diagonal"},
        {"line_styles": ()},
        {"line_styles": ("solid", "dashed"), "line_style_probabilities": (1.0,)},
        {"line_style_probabilities": (-0.1, 1.0, 1.0)},
        {"line_style_probabilities": (0.0, 0.0, 0.0)},
        {"random_color_prob": 1.1},
        {"color_palette": ()},
        {"color_palette": ((),)},
        {"color_palette": ((256,),)},
        {"color_palette_probabilities": (1.0,)},
        {"color_palette": ((0,), (255,)), "color_palette_probabilities": (1.0,)},
        {"color_palette": ((0,), (255,)), "color_palette_probabilities": (-1.0, 1.0)},
        {"color_palette": ((0,), (255,)), "color_palette_probabilities": (0.0, 0.0)},
        {"line_length_range": (5, 10)},
        {"line_length_range": (10, 5)},
    ],
)
def test_annotation_artifacts_validation_errors(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        A.AnnotationArtifacts(**params)
