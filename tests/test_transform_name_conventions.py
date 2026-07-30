from tools.check_transform_names import check_source


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


def test_legacy_random_transform_name_remains_allowed() -> None:
    source = """
class RandomBrightnessContrast(ImageOnlyTransform):
    pass
"""

    assert check_source(source) == []
