import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from kitchen.ingest import KaggleSource, LocalSource, S3Source, source_from_params


def _make_zip(path: Path, filenames: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name in filenames:
            zf.writestr(name, "col1,col2\n1,2\n")


# --- KaggleSource ---

def _mock_kaggle(tmp_path, monkeypatch):
    mock_module = MagicMock()
    mock_module.api.competition_download_files.side_effect = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "kaggle", mock_module)
    return mock_module


def test_kaggle_extracts_and_removes_zip(tmp_path, monkeypatch):
    csv_names = ["MTeams.csv", "MTourneyResults.csv"]
    _make_zip(tmp_path / "competition.zip", csv_names)
    _mock_kaggle(tmp_path, monkeypatch)

    extracted = KaggleSource("test-comp").download(tmp_path)

    assert set(extracted) == set(csv_names)
    assert not list(tmp_path.glob("*.zip"))
    for name in csv_names:
        assert (tmp_path / name).exists()


def test_kaggle_creates_out_dir(tmp_path, monkeypatch):
    out_dir = tmp_path / "data" / "raw"
    _mock_kaggle(tmp_path, monkeypatch)
    KaggleSource("test-comp").download(out_dir)
    assert out_dir.exists()


# --- LocalSource ---

def test_local_copies_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.csv").write_text("1,2\n")
    (src / "b.csv").write_text("3,4\n")
    out = tmp_path / "out"

    copied = LocalSource(src).download(out)

    assert set(copied) == {"a.csv", "b.csv"}
    assert (out / "a.csv").exists()
    assert (out / "b.csv").exists()


def test_local_creates_out_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "nested" / "out"
    LocalSource(src).download(out)
    assert out.exists()


# --- S3Source ---

def test_s3_downloads_files(tmp_path):
    mock_s3 = MagicMock()
    mock_s3.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "raw/teams.csv"}, {"Key": "raw/results.csv"}]}
    ]
    with patch("kitchen.ingest.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_s3
        downloaded = S3Source("my-bucket", "raw/").download(tmp_path)

    assert set(downloaded) == {"teams.csv", "results.csv"}


# --- source_from_params ---

def test_source_from_params_kaggle():
    src = source_from_params({"source": "kaggle", "competition": "test-comp"})
    assert isinstance(src, KaggleSource)
    assert src.competition == "test-comp"


def test_source_from_params_s3():
    src = source_from_params({"source": "s3", "bucket": "my-bucket", "prefix": "raw/"})
    assert isinstance(src, S3Source)
    assert src.bucket == "my-bucket"


def test_source_from_params_local(tmp_path):
    src = source_from_params({"source": "local", "path": str(tmp_path)})
    assert isinstance(src, LocalSource)


def test_source_from_params_unknown():
    import pytest
    with pytest.raises(ValueError, match="Unknown source"):
        source_from_params({"source": "ftp"})
