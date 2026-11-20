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
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

with (
    patch("socket.socket.connect", side_effect=OSError("Network disabled in settings test")),
    patch("socket.socket.connect_ex", side_effect=OSError("Network disabled in settings test")),
    patch("socket.create_connection", side_effect=OSError("Network disabled in settings test")),
):
    import albumentations
    from albumentations.core.analytics import settings as settings_module

    settings = settings_module.settings
    telemetry_calls = {}
    if sys.argv[1] == "check-telemetry":
        from albumentations.core.analytics import telemetry as telemetry_module

        with (
            patch.object(telemetry_module, "is_ci_environment", return_value=False),
            patch.object(telemetry_module, "is_pytest_running", return_value=False),
            patch.object(
                telemetry_module,
                "Thread",
                side_effect=lambda *, target, args, **kwargs: SimpleNamespace(start=lambda: target(*args)),
            ),
        ):
            client = telemetry_module.get_telemetry_client()
            assert client.enabled
            with (
                patch.object(client.user_id_manager, "get_or_create_user_id", return_value="settings-test-user"),
                patch.object(client.backend, "send_event") as send_event,
            ):
                albumentations.Compose([], telemetry=True)
                telemetry_calls["compose"] = send_event.call_count
                send_event.reset_mock()
                client.reset()
                client.track_compose_init({"pipeline_hash": "settings-test"}, use_thread=False)
                telemetry_calls["client"] = send_event.call_count

    print(json.dumps({
        "package_file": str(Path(albumentations.__file__).resolve()),
        "module_file": str(Path(settings_module.__file__).resolve()),
        "settings_file": str(settings.settings_file.resolve()),
        "settings": settings._settings,
        "telemetry_enabled": settings.telemetry_enabled,
        "telemetry_calls": telemetry_calls,
    }))
"""


def _import_settings(
    tmp_path: Path,
    content: str | None,
    overrides: dict[str, str] | None = None,
    *,
    check_telemetry: bool = False,
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
        [sys.executable, "-c", _IMPORT_SETTINGS, "check-telemetry" if check_telemetry else "settings-only"],
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
        pytest.param('[["telemetry", false], ["broken"]]', id="mixed-pair-array"),
        pytest.param('[["telemetry", false], [[], 137]]', id="unhashable-key"),
        pytest.param('[["telemetry", false], 137]', id="non-iterable-element"),
    ],
)
def test_import_ignores_non_object_settings(tmp_path: Path, content: str) -> None:
    state = _import_settings(tmp_path, content)
    assert state["settings"] == {"telemetry": True}
    assert state["telemetry_enabled"] is True


@pytest.mark.parametrize(
    ("content", "expected_telemetry"),
    [
        pytest.param('[["telemetry", false], ["sentinel", 137]]', False, id="false"),
        pytest.param('[["telemetry", 0]]', False, id="zero"),
        pytest.param('[["telemetry", null]]', False, id="null"),
        pytest.param('[["telemetry", ""]]', False, id="empty-string"),
        pytest.param('[["telemetry", []]]', False, id="empty-array"),
        pytest.param('[["telemetry", {}]]', False, id="empty-object"),
        pytest.param('[["telemetry", false], ["telemetry", true]]', True, id="last-value-true"),
        pytest.param('[["telemetry", true], ["telemetry", false]]', False, id="last-value-false"),
        pytest.param('[["telemetry", "false"], ["sentinel", 137]]', True, id="truthy-string"),
        pytest.param('[["sentinel", 137]]', True, id="missing-telemetry"),
        pytest.param('[["telemetry", false], "ab"]', False, id="legacy-string-element"),
    ],
)
def test_import_preserves_only_legacy_telemetry_opt_out(
    tmp_path: Path,
    content: str,
    expected_telemetry: bool,
) -> None:
    state = _import_settings(tmp_path, content)
    assert state["settings"] == {"telemetry": expected_telemetry}
    assert state["telemetry_enabled"] is expected_telemetry


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


@pytest.mark.parametrize("value", [0, None, "", [], {}])
def test_import_preserves_falsey_object_values(tmp_path: Path, value: Any) -> None:
    state = _import_settings(tmp_path, json.dumps({"telemetry": value, "sentinel": 137}))
    assert state["settings"] == {"telemetry": value, "sentinel": 137}
    assert state["telemetry_enabled"] == value
    assert type(state["telemetry_enabled"]) is type(value)


@pytest.mark.parametrize("variable", _OVERRIDE_VARIABLES)
@pytest.mark.parametrize(
    ("value", "expected_telemetry"),
    [("1", False), ("true", False), ("TRUE", False), ("0", True), ("false", True)],
)
@pytest.mark.parametrize(
    "content",
    [
        pytest.param("null", id="non-object"),
        pytest.param('{"telemetry": true}', id="object"),
        pytest.param('[["telemetry", true]]', id="legacy-enabled"),
    ],
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


@pytest.mark.parametrize("variable", _OVERRIDE_VARIABLES)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "0", "false"])
def test_environment_overrides_preserve_legacy_opt_out(tmp_path: Path, variable: str, value: str) -> None:
    state = _import_settings(tmp_path, '[["telemetry", false]]', {variable: value})
    assert state["settings"] == {"telemetry": False}
    assert state["telemetry_enabled"] is False


@pytest.mark.parametrize("content", ['{"telemetry": true}', '[["telemetry", false]]'])
def test_import_applies_both_environment_overrides(tmp_path: Path, content: str) -> None:
    state = _import_settings(tmp_path, content, dict.fromkeys(_OVERRIDE_VARIABLES, "1"))
    assert state["settings"] == {"telemetry": False}
    assert state["telemetry_enabled"] is False


@pytest.mark.parametrize(
    ("content", "expected_calls"),
    [
        pytest.param('[["telemetry", false]]', 0, id="legacy-opt-out"),
        pytest.param('[["telemetry", true]]', 1, id="enabled-positive-control"),
        pytest.param('{"telemetry": false}', 0, id="object-opt-out-control"),
    ],
)
def test_loaded_settings_control_telemetry_delivery(tmp_path: Path, content: str, expected_calls: int) -> None:
    state = _import_settings(tmp_path, content, check_telemetry=True)
    assert state["telemetry_calls"] == {"compose": expected_calls, "client": expected_calls}


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
