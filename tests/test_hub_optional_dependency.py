"""Offline coverage of Hub loading with and without its optional dependency."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import albumentations as A
from albumentations.core import hub_mixin


@pytest.mark.parametrize("route", ["remote", "local", "missing-local-file", "push"])
def test_hub_without_optional_dependency(tmp_path: Path, route: str) -> None:
    """Verify remote, local, missing-file, and push routes with Hub imports blocked."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from collections.abc import Sequence
        from pathlib import Path
        from types import ModuleType

        class BlockHubImports(importlib.abc.MetaPathFinder):
            '''Simulate an absent Hub package in a fresh interpreter.'''

            def find_spec(
                self,
                fullname: str,
                path: Sequence[str] | None = None,
                target: ModuleType | None = None,
            ) -> None:
                '''Reject Hub imports and let other finders resolve unrelated modules.'''
                if fullname == "huggingface_hub" or fullname.startswith("huggingface_hub."):
                    raise ModuleNotFoundError("Hub import blocked by test", name=fullname)
                return None

        sys.meta_path.insert(0, BlockHubImports())

        import pytest
        import albumentations as A
        from albumentations.core import hub_mixin

        assert not hub_mixin.is_huggingface_hub_available
        assert not hasattr(hub_mixin, "hf_hub_download")
        assert not hasattr(hub_mixin, "HfHubHTTPError")
        assert Path(A.__file__).resolve().is_relative_to(Path(sys.argv[3]).resolve())
        transform = A.Compose([A.NoOp()], telemetry=False)
        route, root = sys.argv[1], Path(sys.argv[2])

        if route == "remote":
            with pytest.raises(ImportError, match="huggingface_hub") as error:
                A.Compose.from_pretrained("example/nonexistent", local_files_only=True)
            assert (
                "pip install 'albumentationsx[hub]'" in str(error.value)
                or "pip install huggingface_hub" in str(error.value)
            )
        elif route == "local":
            transform.save_pretrained(root)
            loaded = A.Compose.from_pretrained(root, local_files_only=True)
            assert A.to_dict(loaded) == A.to_dict(transform)
        elif route == "missing-local-file":
            with pytest.raises(FileNotFoundError, match="huggingface_hub"):
                A.Compose.from_pretrained(root, local_files_only=True)
        elif route == "push":
            with pytest.raises(ImportError, match="huggingface_hub"):
                transform.push_to_hub("example/nonexistent")
        """,
    )
    repository_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = os.pathsep.join(filter(None, (str(repository_root), existing_pythonpath)))
    # The interpreter/script are fixed; arguments come from pytest and no shell is used.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, route, str(tmp_path), str(repository_root)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": pythonpath,
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "ALBUMENTATIONS_NO_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not hub_mixin.is_huggingface_hub_available, reason="huggingface_hub is not available")
def test_hub_download_forwards_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserve download arguments and warn that proxies are unsupported."""
    monkeypatch.chdir(tmp_path)
    transform = A.Compose([A.NoOp()], telemetry=False)
    save_directory = tmp_path / "saved"
    transform.save_pretrained(save_directory, key="train")
    filename = "albumentations_config_train.json"
    cache_dir = tmp_path / "cache"
    token = "test-token"

    with patch.object(hub_mixin, "hf_hub_download", return_value=str(save_directory / filename)) as download:
        loaded = A.Compose.from_pretrained(
            "example/nonexistent",
            key="train",
            force_download=True,
            proxies={"https": "http://proxy.example"},
            token=token,
            cache_dir=cache_dir,
            local_files_only=True,
            revision="test-revision",
        )

    download.assert_called_once_with(
        repo_id="example/nonexistent",
        filename=filename,
        revision="test-revision",
        cache_dir=cache_dir,
        force_download=True,
        token=token,
        local_files_only=True,
    )
    assert A.to_dict(loaded) == A.to_dict(transform)
    assert "`proxies` is not supported" in caplog.text


@pytest.mark.skipif(not hub_mixin.is_huggingface_hub_available, reason="huggingface_hub is not available")
def test_local_load_with_hub_installed(tmp_path: Path) -> None:
    """Load valid local configurations without downloading when Hub is installed."""
    transform = A.Compose([A.NoOp()], telemetry=False)
    transform.save_pretrained(tmp_path)

    with patch.object(hub_mixin, "hf_hub_download") as download:
        loaded = A.Compose.from_pretrained(tmp_path, local_files_only=True)

    assert A.to_dict(loaded) == A.to_dict(transform)
    download.assert_not_called()


def test_hub_uncached_offline_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the real Hub cache-miss error without opening a network connection."""
    hub_errors = pytest.importorskip("huggingface_hub.errors")
    hub_constants = pytest.importorskip("huggingface_hub.constants")
    # Hub user-agent discovery can fetch metadata independently of the download route.
    monkeypatch.setattr(hub_constants, "HF_HUB_DISABLE_TELEMETRY", True)
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch(
        "socket.socket.connect",
        side_effect=pytest.fail.Exception("Offline Hub lookup attempted a network connection"),
    ) as connect:
        with pytest.raises(hub_errors.LocalEntryNotFoundError):
            A.Compose.from_pretrained(
                "example/nonexistent",
                cache_dir=cache_dir,
                local_files_only=True,
                token=False,
            )

    connect.assert_not_called()


@pytest.mark.skipif(not hub_mixin.is_huggingface_hub_available, reason="huggingface_hub is not available")
def test_hub_http_error_preserves_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the Hub HTTP error type, response, and cause when adding context."""
    monkeypatch.chdir(tmp_path)
    response = Mock(headers={}, request=Mock())
    original_error = hub_mixin.HfHubHTTPError("download failed", response=response)

    with patch.object(hub_mixin, "hf_hub_download", side_effect=original_error):
        with pytest.raises(
            hub_mixin.HfHubHTTPError,
            match=r"albumentations_config_eval\.json not found on the HuggingFace Hub",
        ) as error:
            A.Compose.from_pretrained("example/nonexistent", local_files_only=True)

    assert type(error.value) is hub_mixin.HfHubHTTPError
    assert error.value.response is response
    assert error.value.__cause__ is original_error
