"""Tests for kitchen.ingest.run — the DVC-stage entrypoint.

``kitchen.ingest.run.main()`` is invoked by ``dvc repro ingest`` (or
``python -m kitchen.ingest.run``).  It reads ``params.yaml`` from cwd,
builds an IngestSource via ``source_from_params``, and downloads files
into ``DataStore().raw_dir``.

These tests use ``monkeypatch.chdir`` so that the module-level
``PARAMS_PATH = Path("params.yaml")`` resolves to a temp directory,
and patch ``kitchen.ingest.run.source_from_params`` to avoid real
network/filesystem I/O.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Import main at module level — PARAMS_PATH = Path("params.yaml") is evaluated
# at import time but the actual .read_text() call inside main() resolves against
# the current working directory at call time, so chdir works correctly.
from kitchen.ingest.run import main

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

KAGGLE_PARAMS = """\
experiment: test
data:
  source: kaggle
  competition: test-comp
"""

S3_PARAMS = """\
experiment: test
data:
  source: s3
  bucket: my-bucket
  prefix: raw/
"""

LOCAL_PARAMS = """\
experiment: test
data:
  source: local
  path: /tmp/data
"""


def _mock_source(files: list[str]) -> MagicMock:
    """Return a mock IngestSource whose .download() returns *files*."""
    src = MagicMock()
    src.download.return_value = files
    return src


# ---------------------------------------------------------------------------
# Happy paths — source dispatch
# ---------------------------------------------------------------------------


def test_main_kaggle_prints_file_count(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    with patch("kitchen.ingest.run.source_from_params", return_value=_mock_source(["train.csv", "test.csv"])):
        main()
    assert "2 file(s)" in capsys.readouterr().out


def test_main_s3_prints_file_count(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)
    with patch("kitchen.ingest.run.source_from_params", return_value=_mock_source(["data.parquet"])):
        main()
    assert "1 file(s)" in capsys.readouterr().out


def test_main_local_prints_file_count(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(LOCAL_PARAMS)
    with patch("kitchen.ingest.run.source_from_params", return_value=_mock_source(["raw.csv", "extra.csv"])):
        main()
    assert "2 file(s)" in capsys.readouterr().out


def test_main_zero_files(tmp_path, monkeypatch, capsys):
    """Empty download is valid — print 0 file(s)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)
    with patch("kitchen.ingest.run.source_from_params", return_value=_mock_source([])):
        main()
    assert "0 file(s)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Output contains destination path
# ---------------------------------------------------------------------------


def test_main_prints_raw_dir(tmp_path, monkeypatch, capsys):
    """Output should include the destination path (DataStore.raw_dir)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)
    with patch("kitchen.ingest.run.source_from_params", return_value=_mock_source(["x.csv"])):
        main()
    expected = str(tmp_path / "data" / "raw")
    assert expected in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Wiring — correct arguments passed to source_from_params and download
# ---------------------------------------------------------------------------


def test_main_passes_data_dict_to_source_from_params(tmp_path, monkeypatch):
    """source_from_params must receive the `data` sub-dict, not the full params."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)
    src = _mock_source(["x.csv"])
    with patch("kitchen.ingest.run.source_from_params", return_value=src) as mock_factory:
        main()
    mock_factory.assert_called_once_with(
        {"source": "s3", "bucket": "my-bucket", "prefix": "raw/"}
    )


def test_main_calls_download_with_raw_dir(tmp_path, monkeypatch):
    """source.download() must be called with DataStore().raw_dir."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)
    src = _mock_source(["x.csv"])
    with patch("kitchen.ingest.run.source_from_params", return_value=src):
        main()
    src.download.assert_called_once_with(tmp_path / "data" / "raw")


def test_main_kaggle_passes_correct_data_dict(tmp_path, monkeypatch):
    """Kaggle params dict forwarded correctly — competition key present."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    src = _mock_source(["train.csv"])
    with patch("kitchen.ingest.run.source_from_params", return_value=src) as mock_factory:
        main()
    mock_factory.assert_called_once_with({"source": "kaggle", "competition": "test-comp"})


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_main_missing_params_yaml_raises(tmp_path, monkeypatch):
    """No params.yaml → FileNotFoundError propagates to caller (DVC handles it)."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        main()


def test_main_missing_data_section_raises(tmp_path, monkeypatch):
    """params.yaml without a `data` key → KeyError propagates."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    with pytest.raises(KeyError):
        main()


def test_main_download_failure_propagates(tmp_path, monkeypatch):
    """Exceptions from source.download() are not swallowed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)
    src = MagicMock()
    src.download.side_effect = RuntimeError("S3 access denied")
    with patch("kitchen.ingest.run.source_from_params", return_value=src):
        with pytest.raises(RuntimeError, match="S3 access denied"):
            main()


def test_main_source_from_params_failure_propagates(tmp_path, monkeypatch):
    """Unknown source type raises ValueError before download is attempted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\ndata:\n  source: ftp\n")
    # Don't patch source_from_params — let the real one raise
    with pytest.raises(ValueError, match="Unknown source"):
        main()
