"""Test that FrozenParams prevents parameter mutation."""

import copy

import pytest

from tests.utils import FrozenParams


def test_frozen_params_read_access():
    """Test that read access works normally."""
    params = FrozenParams({"a": 1, "b": 2})
    assert params["a"] == 1
    assert params["b"] == 2
    assert len(params) == 2


def test_frozen_params_prevents_setitem():
    """Test that direct assignment raises RuntimeError."""
    params = FrozenParams({"a": 1})

    with pytest.raises(RuntimeError, match="Cannot modify FrozenParams"):
        params["b"] = 2


def test_frozen_params_prevents_delitem():
    """Test that deletion raises RuntimeError."""
    params = FrozenParams({"a": 1, "b": 2})

    with pytest.raises(RuntimeError, match="Cannot modify FrozenParams"):
        del params["a"]


def test_frozen_params_deepcopy_returns_mutable():
    """Test that deepcopy returns a regular mutable dict."""
    params = FrozenParams({"a": 1, "b": 2})

    # Deepcopy should return a mutable dict
    mutable = copy.deepcopy(params)
    assert not isinstance(mutable, FrozenParams)
    assert isinstance(mutable, dict)

    # Can modify the copy without error
    mutable["c"] = 3
    assert mutable == {"a": 1, "b": 2, "c": 3}

    # Original is unchanged
    assert params == {"a": 1, "b": 2}


def test_frozen_params_unpacking_creates_copy():
    """Test that unpacking with ** creates a deep copy."""
    params = FrozenParams({"x": 1, "y": 2})

    def modify_kwargs(**kwargs):
        # Modification happens on the unpacked copy
        kwargs["z"] = 3
        return kwargs

    result = modify_kwargs(**params)

    # Function received a mutable copy
    assert result == {"x": 1, "y": 2, "z": 3}

    # Original params unchanged
    assert params == {"x": 1, "y": 2}
    assert "z" not in params


def test_frozen_params_nested_values():
    """Test that nested mutable values are also copied."""
    params = FrozenParams({"nested": {"a": 1}, "list": [1, 2, 3]})

    # Deepcopy should deep copy nested structures
    mutable = copy.deepcopy(params)
    mutable["nested"]["b"] = 2
    mutable["list"].append(4)

    # Original nested values unchanged
    assert params["nested"] == {"a": 1}
    assert params["list"] == [1, 2, 3]
