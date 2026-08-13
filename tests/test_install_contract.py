"""Tests for the package-specific half of the clean wheel install contract."""

from pathlib import Path

import pytest

from tools import install_contract


def test_resolve_wheel_requires_exactly_one_match(tmp_path: Path) -> None:
    wheel = tmp_path / "albumentationsx-2.4.0-py3-none-any.whl"
    wheel.touch()

    assert install_contract._resolve_wheel(tmp_path / "*.whl") == wheel


def test_resolve_wheel_rejects_multiple_matches(tmp_path: Path) -> None:
    (tmp_path / "one.whl").touch()
    (tmp_path / "two.whl").touch()

    with pytest.raises(RuntimeError, match="Expected exactly one wheel"):
        install_contract._resolve_wheel(tmp_path / "*.whl")


def test_main_records_prepared_interpreter_for_the_foundation_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "github-output"
    interpreter = tmp_path / "environment" / "bin" / "python"
    monkeypatch.setattr(install_contract, "prepare_install_contract", lambda *_: interpreter)
    monkeypatch.setattr(
        "sys.argv",
        [
            "install_contract.py",
            "prepare",
            "--wheel",
            "dist/*.whl",
            "--python",
            "3.12",
            "--github-output",
            str(output),
        ],
    )

    assert install_contract.main() == 0
    assert output.read_text(encoding="utf-8") == f"python={interpreter}\n"
