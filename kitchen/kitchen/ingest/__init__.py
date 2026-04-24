import shutil
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import boto3


class IngestSource(ABC):
    @abstractmethod
    def download(self, out_dir: Path) -> list[str]:
        """Download data into out_dir. Returns list of filenames written."""


class KaggleSource(IngestSource):
    """Downloads a Kaggle competition dataset.

    Credentials are read from ~/.kaggle/kaggle.json or the KAGGLE_USERNAME /
    KAGGLE_KEY environment variables.
    """

    def __init__(self, competition: str) -> None:
        self.competition = competition

    def download(self, out_dir: Path) -> list[str]:
        import kaggle

        out_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(self.competition, path=str(out_dir), quiet=False)

        extracted: list[str] = []
        for zip_path in sorted(out_dir.glob("*.zip")):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
                extracted.extend(zf.namelist())
            zip_path.unlink()
        return extracted


class S3Source(IngestSource):
    """Downloads all objects under a given S3 prefix."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix

    def download(self, out_dir: Path) -> list[str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        downloaded: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                filename = Path(obj["Key"]).name
                s3.download_file(self.bucket, obj["Key"], str(out_dir / filename))
                downloaded.append(filename)
        return downloaded


class LocalSource(IngestSource):
    """Copies files from a local directory."""

    def __init__(self, src_dir: Path | str) -> None:
        self.src_dir = Path(src_dir)

    def download(self, out_dir: Path) -> list[str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for f in sorted(self.src_dir.iterdir()):
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)
                copied.append(f.name)
        return copied


def source_from_params(params: dict) -> IngestSource:
    """Build an IngestSource from the `data` section of params.yaml."""
    source = params["source"]
    if source == "kaggle":
        return KaggleSource(params["competition"])
    if source == "s3":
        return S3Source(params["bucket"], params.get("prefix", ""))
    if source == "local":
        return LocalSource(params["path"])
    raise ValueError(f"Unknown source: {source!r}. Choose from: kaggle, s3, local")
