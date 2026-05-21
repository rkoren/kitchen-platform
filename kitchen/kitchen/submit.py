"""Submission validation and upload for Kaggle competitions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def log_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    file_path: Path,
    id_col: str = "ID",
    target_col: str = "Pred",
    competition: str | None = None,
    message: str = "",
    fetch_lb_score: bool = False,
    fetch_timeout: int = 300,
) -> dict[str, float]:
    """Validate, log, and optionally upload a competition submission.

    Attaches the submission CSV as an MLflow artifact on the *active* run and
    (when ``competition`` is provided) uploads it to Kaggle. When
    ``fetch_lb_score=True``, polls Kaggle for the public leaderboard score and
    logs it as ``lb_score`` on the same run.

    Args:
        submission: Submission DataFrame (already written to ``file_path``).
        sample: Sample submission DataFrame used for row-count validation.
        file_path: Path to the on-disk CSV (written by the caller).
        id_col: ID column name (default "ID").
        target_col: Prediction column name (default "Pred").
        competition: Kaggle competition slug. When None, skips upload/score fetch.
        message: Submission message shown on the leaderboard.
        fetch_lb_score: Poll for the public LB score after uploading.
        fetch_timeout: Seconds to wait for Kaggle to score the submission.

    Returns:
        Dict with ``lb_score`` key if a score was retrieved, otherwise empty.

    Raises:
        ValueError: If validation fails — call is a no-op for MLflow/Kaggle.
    """
    import mlflow

    errors = validate_submission(submission, sample, id_col, target_col)
    if errors:
        raise ValueError("Submission validation failed:\n" + "\n".join(f"  {e}" for e in errors))

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(file_path), artifact_path="submission")

    if competition is None:
        return {}

    upload(file_path, message, competition)

    if not fetch_lb_score:
        return {}

    score = fetch_score(competition, timeout=fetch_timeout)
    if score is not None and mlflow.active_run() is not None:
        mlflow.log_metric("lb_score", score)
    return {"lb_score": score} if score is not None else {}


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
