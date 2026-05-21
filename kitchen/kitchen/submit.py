"""Submission validation and upload for Kaggle competitions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_submission(
    sub: pd.DataFrame,
    sample: pd.DataFrame,
    id_col: str,
    target_col: str,
) -> list[str]:
    """Run all submission checks. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []

    # KG-005: required columns present
    for col in (id_col, target_col):
        if col not in sub.columns:
            errors.append(f"missing column: {col!r}")

    # KG-006: row count matches sample submission
    if len(sub) != len(sample):
        errors.append(
            f"row count mismatch: submission has {len(sub)} rows, "
            f"sample_submission has {len(sample)}"
        )

    # KG-007: no null predictions
    if target_col in sub.columns and sub[target_col].isna().any():
        n = int(sub[target_col].isna().sum())
        errors.append(f"{n} null value(s) in target column {target_col!r}")

    # KG-008: no duplicate IDs
    if id_col in sub.columns and sub[id_col].duplicated().any():
        n = int(sub[id_col].duplicated().sum())
        errors.append(f"{n} duplicate ID(s) in column {id_col!r}")

    return errors


def upload(file_path: Path, message: str, competition: str) -> None:
    """Authenticate and submit a CSV to a Kaggle competition."""
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.competition_submit(str(file_path), message, competition, quiet=False)


def fetch_score(competition: str, timeout: int = 120, interval: int = 10) -> float | None:
    """Poll Kaggle for the most recent submission's public leaderboard score.

    Returns the score as a float, or None if the timeout expires, the submission
    errors, or the score cannot be parsed.
    """
    import time

    import kaggle

    try:
        kaggle.api.authenticate()
    except Exception:
        return None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            submissions = kaggle.api.competition_submissions(competition)
        except Exception:
            return None
        if submissions:
            latest = submissions[0]
            status = getattr(latest, "status", None)
            if status == "complete":
                raw = getattr(latest, "publicScore", None)
                if raw is None:
                    return None
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return None
            if status == "error":
                return None
        time.sleep(interval)
    return None
