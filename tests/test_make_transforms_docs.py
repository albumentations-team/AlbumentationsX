from pathlib import Path

import pytest

from albumentations.augmentations.medical import transforms as medical
from tools import make_transforms_docs


def test_medical_catalog_follows_module_ownership() -> None:
    names = set(make_transforms_docs.get_medical_transforms_info())
    assert names == set(medical.__all__)
    assert names.isdisjoint(make_transforms_docs.get_image_only_transforms_info())
    assert names.isdisjoint(make_transforms_docs.get_dual_transforms_info())
    assert names.isdisjoint(make_transforms_docs.get_3d_transforms_info())


def test_missing_medical_row_fails_readme_validation(tmp_path: Path) -> None:
    docs = make_transforms_docs.generated_transform_docs()
    text = "\n\n".join(
        f"### {category} transforms\n\n{content}"
        for category, content in zip(("Pixel-level", "Spatial-level", "3D", "Medical"), docs, strict=True)
    )
    readme = tmp_path / "README.md"
    readme.write_text(text)
    make_transforms_docs.check_transform_docs(readme)

    readme.write_text("\n".join(line for line in text.splitlines() if "/GibbsRinging/" not in line))
    with pytest.raises(ValueError, match="Medical"):
        make_transforms_docs.check_transform_docs(readme)
