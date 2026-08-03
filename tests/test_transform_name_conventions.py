import pytest

from tools.check_transform_names import LEGACY_RANDOM_TRANSFORM_NAMES, check_source, find_package_transform_class_names


def test_new_transform_name_cannot_start_with_random() -> None:
    source = """
class RandomNewTransform(ImageOnlyTransform):
    pass
"""

    violations = check_source(source)

    assert violations == [
        (
            2,
            "New transform class 'RandomNewTransform' must not use the 'Random' prefix.",
        ),
    ]


@pytest.mark.parametrize(
    ("class_name", "base_class_name"),
    [
        ("RandomBrightnessContrast", "ImageOnlyTransform"),
        ("RandomOrder", "BaseCompose"),
        ("RandomRotate90_3D", "Transform3D"),
    ],
)
def test_legacy_random_transform_name_remains_allowed(class_name: str, base_class_name: str) -> None:
    source = f"""
class {class_name}({base_class_name}):
    pass
"""

    assert check_source(source) == []


def test_random_prefix_is_allowed_for_non_transform_class() -> None:
    source = """
class RandomParameterGenerator:
    pass
"""

    assert check_source(source) == []


def test_new_volume_only_transform_name_cannot_start_with_random() -> None:
    source = """
class RandomNewVolumeTransform(VolumeOnlyTransform):
    pass
"""

    assert check_source(source) == [
        (
            2,
            "New transform class 'RandomNewVolumeTransform' must not use the 'Random' prefix.",
        ),
    ]


def test_indirect_transform_subclass_cannot_start_with_random() -> None:
    source = """
class CustomTransform(ImageOnlyTransform):
    pass

class RandomNewTransform(CustomTransform):
    pass
"""

    assert check_source(source) == [
        (
            5,
            "New transform class 'RandomNewTransform' must not use the 'Random' prefix.",
        ),
    ]


def test_legacy_random_transform_allowlist_matches_package() -> None:
    package_random_transform_names = {
        name for name in find_package_transform_class_names() if name.startswith("Random")
    }

    assert package_random_transform_names == LEGACY_RANDOM_TRANSFORM_NAMES
