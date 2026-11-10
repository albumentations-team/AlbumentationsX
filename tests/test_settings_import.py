"""Fresh-process contracts for settings loaded during package import."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from albumentations.core.analytics.settings import SettingsManager

_WORKTREE_ROOT = Path(__file__).resolve().parents[1]
_OVERRIDE_VARIABLES = ("ALBUMENTATIONS_NO_TELEMETRY", "ALBUMENTATIONS_OFFLINE")
_IMPORT_SETTINGS = """
import json
from pathlib import Path
from unittest.mock import patch

with (
    patch("socket.socket.connect", side_effect=OSError("Network disabled in settings test")),
    patch("socket.socket.connect_ex", side_effect=OSError("Network disabled in settings test")),
    patch("socket.create_connection", side_effect=OSError("Network disabled in settings test")),
):
    import albumentations
    from albumentations.core.analytics import settings as settings_module

    settings = settings_module.settings
    print(json.dumps({
        "package_file": str(Path(albumentations.__file__).resolve()),
        "module_file": str(Path(settings_module.__file__).resolve()),
        "settings_file": str(settings.settings_file.resolve()),
        "settings": settings._settings,
        "telemetry_enabled": settings.telemetry_enabled,
    }))
"""


def _import_settings(
    tmp_path: Path,
    content: str | None,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    settings_file = cache_dir / "settings.json"
    original_bytes = content.encode("utf-8") if content is not None else None
    if original_bytes is not None:
        settings_file.write_bytes(original_bytes)

    environment = os.environ.copy()
    for variable in (*_OVERRIDE_VARIABLES, "PYTHONPATH", "PYTHONHOME"):
        environment.pop(variable, None)
    environment.update(
        ALBUMENTATIONS_CACHE_DIR=str(cache_dir),
        ALBUMENTATIONS_CONFIG_DIR=str(tmp_path / "config"),
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    environment.update(overrides or {})
    # The interpreter and embedded script are fixed; settings contents are passed only through the file.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _IMPORT_SETTINGS],
        cwd=_WORKTREE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    if original_bytes is None:
        assert not settings_file.exists()
    else:
        assert settings_file.read_bytes() == original_bytes
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert Path(state["package_file"]) == _WORKTREE_ROOT / "albumentations/__init__.py"
    assert Path(state["module_file"]) == _WORKTREE_ROOT / "albumentations/core/analytics/settings.py"
    assert Path(state["settings_file"]) == settings_file
    return state


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("null", id="null"),
        pytest.param("42", id="number"),
        pytest.param("[1]", id="array"),
        pytest.param("false", id="boolean"),
        pytest.param('"settings"', id="string"),
        pytest.param("[]", id="empty-array"),
        pytest.param('[["telemetry", false], ["sentinel", 137]]', id="pair-array"),
        pytest.param('[["telemetry", false], ["broken"]]', id="mixed-pair-array"),
    ],
)
def test_import_ignores_non_object_settings(tmp_path: Path, content: str) -> None:
    state = _import_settings(tmp_path, content)
    assert state["settings"] == {"telemetry": True}
    assert state["telemetry_enabled"] is True


@pytest.mark.parametrize(
    ("content", "expected_settings"),
    [
        pytest.param(None, {"telemetry": True}, id="missing-file"),
        pytest.param("{", {"telemetry": True}, id="malformed-json"),
        pytest.param("{}", {"telemetry": True}, id="empty-object"),
        pytest.param(
            '{"telemetry": false, "sentinel": 137}',
            {"telemetry": False, "sentinel": 137},
            id="object",
        ),
        pytest.param(
            '{"telemetry": "custom", "sentinel": 137}',
            {"telemetry": "custom", "sentinel": 137},
            id="unvalidated-object-value",
        ),
    ],
)
def test_import_preserves_settings_controls(
    tmp_path: Path,
    content: str | None,
    expected_settings: dict[str, Any],
) -> None:
    state = _import_settings(tmp_path, content)
    assert state["settings"] == expected_settings
    assert state["telemetry_enabled"] == expected_settings["telemetry"]


@pytest.mark.parametrize("variable", _OVERRIDE_VARIABLES)
@pytest.mark.parametrize(
    ("value", "expected_telemetry"),
    [("1", False), ("true", False), ("TRUE", False), ("0", True), ("false", True)],
)
@pytest.mark.parametrize(
    "content",
    [pytest.param("null", id="non-object"), pytest.param('{"telemetry": true}', id="object")],
)
def test_import_applies_each_environment_override(
    tmp_path: Path,
    content: str,
    variable: str,
    value: str,
    expected_telemetry: bool,
) -> None:
    state = _import_settings(tmp_path, content, {variable: value})
    assert state["settings"] == {"telemetry": expected_telemetry}
    assert state["telemetry_enabled"] is expected_telemetry


def test_settings_read_error_preserves_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in _OVERRIDE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    settings_file = tmp_path / "settings.json"
    original_bytes = b'{"telemetry": false, "sentinel": 137}'
    settings_file.write_bytes(original_bytes)

    with patch.object(Path, "open", side_effect=OSError("Settings read failed")) as open_settings:
        manager = SettingsManager(settings_file=settings_file)
    open_settings.assert_called_once_with()
    assert manager.get("telemetry") is True
    assert manager.telemetry_enabled is True
    assert manager.get("sentinel") is None
    assert settings_file.read_bytes() == original_bytes
