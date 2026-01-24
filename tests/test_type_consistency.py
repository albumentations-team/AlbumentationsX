"""Test type consistency between InitSchema and __init__ constructors.

This test verifies that type hints in InitSchema match those in __init__ constructors
for all AlbumentationsX transforms.

Run tests:
    # Test all transforms
    pytest tests/test_type_consistency.py -v

    # Test specific transforms
    pytest tests/test_type_consistency.py -k "Blur or Affine" -v

    # See which transforms have InitSchema
    pytest tests/test_type_consistency.py::test_transforms_with_init_schema_exist -v -s
"""

import pytest
from albu_spec import (
    compare_types,
    get_common_param_names,
    get_init_param_type,
    get_init_schema_param_type,
    get_type_mismatch,
)

from tests.utils import get_transforms


def get_transforms_with_init_schema():
    """Get all transforms that have InitSchema."""
    # Deduplicate transform classes while preserving order in case
    # get_transforms() returns multiple parameter sets per transform.
    return list(dict.fromkeys(transform_cls for transform_cls, _ in get_transforms()))


@pytest.mark.parametrize("transform_class", get_transforms_with_init_schema())
def test_init_vs_initschema_types(transform_class):
    """Test that __init__ types match InitSchema types for all parameters.

    This test ensures type consistency between:
    - Constructor signature (__init__)
    - Pydantic validation schema (InitSchema)

    The comparison uses semantic equivalence:
    - Union order doesn't matter: int | float == float | int
    - Optional handling: Optional[int] == int | None
    - Annotated unwrapping: Annotated[int, Field(ge=0)] == int
    - Literal value sets are order-independent

    Args:
        transform_class: Transform class to test

    """
    transform_name = transform_class.__name__


    param_names = get_common_param_names(transform_class)

    mismatches = []

    for param_name in param_names:
        # Skip special parameters
        if param_name in {"self", "strict"}:
            continue


        # Extract types from both sources
        init_type = get_init_param_type(transform_class, param_name)
        schema_type = get_init_schema_param_type(transform_class, param_name)

        # Compare types using semantic comparison
        if not compare_types(init_type, schema_type):
            # Get detailed mismatch information
            mismatch = get_type_mismatch(init_type, schema_type)
            if mismatch:
                mismatches.append(
                    f"  Parameter '{param_name}':\n"
                    f"    __init__:    {init_type}\n"
                    f"    InitSchema:  {schema_type}\n"
                    f"    Mismatch: {mismatch}"
                )

    # Assert no mismatches found
    if mismatches:
        error_msg = f"\n\nType mismatches in {transform_name}:\n" + "\n".join(mismatches)
        pytest.fail(error_msg)


def test_albu_spec_imports():
    """Test that all required albu-spec functions are available."""
    # This test will fail with ImportError if albu-spec is not installed
    assert compare_types is not None
    assert get_common_param_names is not None
    assert get_init_param_type is not None
    assert get_init_schema_param_type is not None
    assert get_type_mismatch is not None


def test_transforms_with_init_schema_exist():
    """Test that we can find transforms with InitSchema."""
    transforms = get_transforms_with_init_schema()
    assert len(transforms) > 0, "No transforms with InitSchema found"
    print(f"\nFound {len(transforms)} transforms with InitSchema")
