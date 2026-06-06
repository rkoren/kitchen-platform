"""Standard data paths and pandas I/O helpers.

Usage::

    from kitchen.store import DataStore

    store = DataStore()                    # root = cwd (where dvc.yaml lives, if using DVC)
    df = store.load_csv("teams.csv")       # reads from data/raw/
    store.save_parquet(df, "teams.parquet") # writes to data/processed/
    df = store.load_parquet("teams.parquet")
"""

from __future__ import annotations

import warnings
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
        """Root defaults to cwd — the directory where dvc.yaml lives (if using DVC)."""
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
            raise ValueError(f"Unknown stage {stage!r}. Valid stages: {sorted(_KNOWN_STAGES)}")
        return getattr(self, f"{stage}_dir")

    def load_csv(self, filename: str, **kwargs: object) -> pd.DataFrame:
        """Read a CSV from data/raw/."""
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `{_STAGE_COMMAND['raw']}` first")
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

    def list(self, stage: str = "raw") -> list[str]:
        """Return a sorted list of filenames in the given stage directory.

        ``stage`` accepts ``"raw"``, ``"processed"``, ``"models"``, or a custom
        relative path from the store root.  Returns an empty list when the
        directory does not exist or is empty.  Only files are returned —
        subdirectories are excluded.
        """
        if stage in _KNOWN_STAGES:
            directory = getattr(self, f"{stage}_dir")
        else:
            directory = self.root / stage
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_file())

    def preview(self, filename: str, n: int = 5) -> pd.DataFrame:
        """Return the first n rows of a file, searching processed/ then raw/.

        If the file exists in both stages, processed/ is returned and a warning
        is emitted. Raises FileNotFoundError with a listing of available files
        in both stages if the file is not found in either.

        Supported formats: .csv, .parquet.
        """
        processed_path = self.processed_dir / filename
        raw_path = self.raw_dir / filename

        if processed_path.exists():
            if raw_path.exists():
                warnings.warn(
                    f"{filename!r} found in both processed/ and raw/; returning processed/ copy",
                    stacklevel=2,
                )
            return self._preview_read(processed_path, n)

        if raw_path.exists():
            return self._preview_read(raw_path, n)

        available: list[str] = []
        for stage_dir, label in [(self.processed_dir, "processed"), (self.raw_dir, "raw")]:
            if stage_dir.is_dir():
                names = sorted(p.name for p in stage_dir.iterdir() if p.is_file())
                if names:
                    available.append(f"  {label}/: {', '.join(names)}")
        detail = "\n".join(available) or "  (no data files found)"
        raise FileNotFoundError(
            f"{filename!r} not found in data/processed/ or data/raw/.\nAvailable files:\n{detail}"
        )

    def is_stale(self, output_file: str | Path, deps: list[str | Path]) -> bool:
        """Return True if any dep is newer than output_file, or output_file is missing.

        Useful for skipping expensive feature regeneration when inputs haven't changed.
        Paths are resolved relative to the store root when not absolute.

        Args:
            output_file: The generated artifact to check (e.g. "data/processed/features.parquet").
            deps: Source files that output_file depends on (e.g. raw CSVs, feature scripts).

        Returns:
            True  — output_file is missing, or at least one dep has a newer mtime.
            False — output_file exists and is newer than all deps.
        """
        out = Path(output_file) if Path(output_file).is_absolute() else self.root / output_file
        if not out.exists():
            return True
        out_mtime = out.stat().st_mtime
        for dep in deps:
            dep_path = Path(dep) if Path(dep).is_absolute() else self.root / dep
            if dep_path.exists() and dep_path.stat().st_mtime > out_mtime:
                return True
        return False

    @staticmethod
    def _preview_read(path: Path, n: int) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path).head(n)
        if suffix == ".csv":
            return pd.read_csv(path).head(n)
        raise ValueError(
            f"Unsupported file extension for preview: {path.suffix!r}. Supported: .csv, .parquet"
        )
