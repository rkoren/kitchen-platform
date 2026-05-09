"""Tests for `kitchen ingest`."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()

KAGGLE_PARAMS = """\
experiment: test
data:
  source: kaggle
  competition: march-machine-learning-mania-2026
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


def _fake_download(files):
    def _download(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        return files
    return _download


# ---------------------------------------------------------------------------
# Missing / invalid params
# ---------------------------------------------------------------------------

def test_ingest_missing_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["ingest", "--params", "missing.yaml"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_ingest_no_data_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code != 0
    assert "data" in result.output


# ---------------------------------------------------------------------------
# Kaggle credential checks
# ---------------------------------------------------------------------------

def test_ingest_kaggle_missing_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    with patch("pathlib.Path.home", return_value=tmp_path):
        result = runner.invoke(app, ["ingest"], env={})
    assert result.exit_code != 0
    assert "Kaggle credentials" in result.output


def test_ingest_kaggle_credentials_via_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("kitchen.ingest.KaggleSource.download", _fake_download(["train.csv", "test.csv"])):
        result = runner.invoke(app, ["ingest"], env={"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"})
    assert result.exit_code == 0
    assert "2 file(s)" in result.output


def test_ingest_kaggle_credentials_via_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text('{"username":"u","key":"k"}')
    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("kitchen.ingest.KaggleSource.download", _fake_download(["train.csv"])):
        result = runner.invoke(app, ["ingest"], env={})
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Source dispatch
# ---------------------------------------------------------------------------

def test_ingest_kaggle_lists_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("kitchen.ingest.KaggleSource.download",
               _fake_download(["train.csv", "test.csv", "sample_submission.csv"])):
        result = runner.invoke(app, ["ingest"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code == 0
    assert "3 file(s)" in result.output
    assert "train.csv" in result.output
    assert "sample_submission.csv" in result.output


def test_ingest_s3(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)
    with patch("kitchen.ingest.S3Source.download", _fake_download(["data.parquet"])):
        result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    assert "1 file(s)" in result.output
    assert "data.parquet" in result.output


def test_ingest_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(LOCAL_PARAMS)
    with patch("kitchen.ingest.LocalSource.download", _fake_download(["raw.csv"])):
        result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    assert "raw.csv" in result.output


# ---------------------------------------------------------------------------
# Custom output directory
# ---------------------------------------------------------------------------

def test_ingest_custom_out_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)
    custom = tmp_path / "my-data"
    with patch("kitchen.ingest.S3Source.download", _fake_download(["file.csv"])):
        result = runner.invoke(app, ["ingest", "--out", str(custom)])
    assert result.exit_code == 0
    assert str(custom) in result.output


# ---------------------------------------------------------------------------
# Download failure
# ---------------------------------------------------------------------------

def test_ingest_download_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(S3_PARAMS)

    def boom(self, out_dir):
        raise RuntimeError("S3 access denied")

    with patch("kitchen.ingest.S3Source.download", boom):
        result = runner.invoke(app, ["ingest"])
    assert result.exit_code != 0
    assert "S3 access denied" in result.output
