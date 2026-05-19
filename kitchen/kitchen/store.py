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

_KNOWN_STAGES = frozenset({"raw", "processed", "models"})

_STAGE_COMMAND = {
    "raw": "kitchen ingest",
    "processed": "kitchen run features",
    "models": "kitchen run train",
}


class DataStore:
    def __init__(self, root: Path | str | None = None) -> None:
        """Root defaults to cwd — the directory where dvc.yaml lives."""
        if root is not None:
            resolved = Path(root)
            if not resolved.is_dir():
                raise FileNotFoundError(f"DataStore root does not exist: {resolved}")
            self.root = resolved
        else:
            self.root = Path.cwd()

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

    def _stage_dir(self, stage: str) -> Path:
        if stage not in _KNOWN_STAGES:
            raise ValueError(
                f"Unknown stage {stage!r}. Valid stages: {sorted(_KNOWN_STAGES)}"
            )
        return getattr(self, f"{stage}_dir")

    def load_csv(self, filename: str, **kwargs: object) -> pd.DataFrame:
        """Read a CSV from data/raw/."""
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `{_STAGE_COMMAND['raw']}` first"
            )
        return pd.read_csv(path, **kwargs)

    def save_parquet(self, df: pd.DataFrame, filename: str, stage: str = "processed") -> Path:
        """Write df as Parquet to the given stage directory, creating it if needed."""
        dest_dir = self._stage_dir(stage)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / filename
        df.to_parquet(path, index=False)
        return path

    def load_parquet(self, filename: str, stage: str = "processed") -> pd.DataFrame:
        """Read a Parquet file from the given stage directory."""
        dest_dir = self._stage_dir(stage)
        path = dest_dir / filename
        if not path.exists():
            cmd = _STAGE_COMMAND.get(stage, f"kitchen run {stage}")
            raise FileNotFoundError(f"{path} not found — run `{cmd}` first")
        return pd.read_parquet(path)
