"""Standard data paths and pandas I/O helpers.

Usage::

    from kitchen.store import DataStore

    store = DataStore()                    # root = cwd (where dvc.yaml lives)
    df = store.load_csv("teams.csv")       # reads from data/raw/
    store.save_parquet(df, "teams.parquet") # writes to data/processed/
    df = store.load_parquet("teams.parquet")
"""
from pathlib import Path

import pandas as pd


class DataStore:
    def __init__(self, root: Path | str | None = None) -> None:
        """Root defaults to cwd — the directory where dvc.yaml lives."""
        self.root = Path(root) if root else Path.cwd()

    @property
    def raw_dir(self) -> Path:
        """data/raw/ — written by the ingest stage."""
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        """data/processed/ — written by the features stage."""
        return self.root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        """models/ — written by the train stage."""
        return self.root / "models"

    def load_csv(self, filename: str, **kwargs: object) -> pd.DataFrame:
        """Read a CSV from data/raw/."""
        return pd.read_csv(self.raw_dir / filename, **kwargs)

    def save_parquet(self, df: pd.DataFrame, filename: str, stage: str = "processed") -> Path:
        """Write df as Parquet to the given stage directory, creating it if needed."""
        dest_dir: Path = getattr(self, f"{stage}_dir")
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / filename
        df.to_parquet(path, index=False)
        return path

    def load_parquet(self, filename: str, stage: str = "processed") -> pd.DataFrame:
        """Read a Parquet file from the given stage directory."""
        dest_dir: Path = getattr(self, f"{stage}_dir")
        return pd.read_parquet(dest_dir / filename)
