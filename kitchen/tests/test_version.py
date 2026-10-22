"""`kitchen --version` / `kitchen.__version__` (CBB-044)."""

import json
from importlib import metadata

from typer.testing import CliRunner

import kitchen
from kitchen import _version
from kitchen.cli import app


class _FakeDist:
    def __init__(self, version, direct_url=None):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name):
        if name == "direct_url.json" and self._direct_url is not None:
            return json.dumps(self._direct_url)
        return None


def _pyproject(root, version, name="rkoren-kitchen"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n')


def _patch(monkeypatch, dist):
    monkeypatch.setattr(_version.metadata, "distribution", lambda _name: dist)


def test_regular_install_uses_metadata(monkeypatch):
    _patch(monkeypatch, _FakeDist("1.2.0"))
    info = _version.version_info()
    assert (info.version, info.editable_path, info.stale_metadata) == ("1.2.0", None, False)
    assert _version.describe(info) == "kitchen 1.2.0"


def test_editable_install_reads_source_and_flags_stale_metadata(monkeypatch, tmp_path):
    src = tmp_path / "kitchen"
    _pyproject(src, "1.2.0")
    _patch(monkeypatch, _FakeDist("0.1.0", {"url": src.as_uri(), "dir_info": {"editable": True}}))
    info = _version.version_info()
    assert info.version == "1.2.0"
    assert info.editable_path == src
    assert info.stale_metadata
    out = _version.describe(info)
    assert out.startswith(f"kitchen 1.2.0 (editable: {src})")
    assert "metadata says 0.1.0" in out and "pip install -e" in out


def test_editable_install_current_metadata_no_warning(monkeypatch, tmp_path):
    src = tmp_path / "kitchen"
    _pyproject(src, "1.2.0")
    _patch(monkeypatch, _FakeDist("1.2.0", {"url": src.as_uri(), "dir_info": {"editable": True}}))
    assert "warning" not in _version.describe()


def test_editable_install_foreign_pyproject_falls_back(monkeypatch, tmp_path):
    src = tmp_path / "other"
    _pyproject(src, "9.9.9", name="something-else")
    _patch(monkeypatch, _FakeDist("1.1.0", {"url": src.as_uri(), "dir_info": {"editable": True}}))
    info = _version.version_info()
    assert info.version == "1.1.0" and not info.stale_metadata


def test_not_installed_reads_package_source(monkeypatch):
    def _missing(_name):
        raise metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(_version.metadata, "distribution", _missing)
    assert _version.version_info().version != "unknown"


def test_dunder_version_is_a_string():
    assert isinstance(kitchen.__version__, str) and kitchen.__version__


def test_cli_version_flag_and_command(monkeypatch):
    _patch(monkeypatch, _FakeDist("1.2.0"))
    runner = CliRunner()
    for args in (["--version"], ["version"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "kitchen 1.2.0"
