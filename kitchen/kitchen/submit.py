"""Submission validation and upload for Kaggle competitions."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Outcomes of polling Kaggle for a submission's score. `None` alone can't tell "still
# scoring" (retry later) from "errored" (don't) from "couldn't reach Kaggle" — callers
# need the distinction to message the user correctly (S6E7-003).
SCORED = "scored"  # Kaggle finished scoring; `.score` is set
PENDING = "pending"  # timed out while still scoring — safe to poll again later
ERRORED = "errored"  # Kaggle reported the submission itself errored
UNAVAILABLE = "unavailable"  # auth/API failure, or a scored submission with no parseable score


class ScoreResult(NamedTuple):
    """Result of polling Kaggle for a submission score."""

    status: str  # one of SCORED / PENDING / ERRORED / UNAVAILABLE
    score: float | None
    detail: str = ""  # human-readable context for non-SCORED outcomes


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


def check_feature_parity(
    expected: list[str],
    df: pd.DataFrame,
) -> list[str]:
    """Check that all model training features are present in an inference DataFrame.

    Returns a list of error strings, one per missing feature (empty = all clear).
    Missing features cause silent wrong predictions; call this before predict_batch.

    Args:
        expected: Feature names the model was trained on (e.g. loto.features).
        df: DataFrame that will be passed to the model at inference time.

    Returns:
        List of error strings. Empty list means all features are present.
    """
    # KG-013: train/test feature parity
    return [f"missing feature: {f!r}" for f in expected if f not in df.columns]


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


def _normalize_status(status: object) -> str:
    """Lowercase leaf token of a Kaggle submission status.

    Kaggle may hand back a plain string (``"complete"``) or an enum whose ``str()`` is
    ``"SubmissionStatus.COMPLETE"``; both normalize to ``"complete"``.
    """
    return str(status).lower().rsplit(".", 1)[-1]


def poll_submission_score(
    competition: str,
    timeout: int = 300,
    interval: int = 5,
    max_interval: int = 30,
    backoff: float = 1.5,
) -> ScoreResult:
    """Poll Kaggle for the latest submission's public leaderboard score.

    Kaggle typically lags tens of seconds behind an upload, so this waits up to ``timeout``
    seconds, polling on an exponential backoff (``interval`` → ``max_interval``) to avoid
    hammering the API. Returns a :class:`ScoreResult` distinguishing "scored" from "still
    scoring" (``PENDING`` — safe to poll again), "errored", and "unavailable".

    Args:
        competition: Kaggle competition slug.
        timeout: Max seconds to wait for a terminal status.
        interval: Initial poll interval in seconds.
        max_interval: Cap the backing-off interval at this many seconds.
        backoff: Multiplier applied to the interval after each poll.
    """
    import time

    import kaggle

    try:
        kaggle.api.authenticate()
    except Exception as exc:
        return ScoreResult(UNAVAILABLE, None, f"could not authenticate with Kaggle: {exc}")

    deadline = time.monotonic() + timeout
    wait = interval
    while True:
        try:
            submissions = kaggle.api.competition_submissions(competition)
        except Exception as exc:
            return ScoreResult(UNAVAILABLE, None, f"could not fetch submissions: {exc}")

        if submissions:
            latest = submissions[0]
            status = _normalize_status(getattr(latest, "status", None))
            if status == "complete":
                raw = getattr(latest, "publicScore", None)
                if raw is None:
                    return ScoreResult(UNAVAILABLE, None, "submission scored but no publicScore")
                try:
                    return ScoreResult(SCORED, float(raw))
                except (TypeError, ValueError):
                    return ScoreResult(UNAVAILABLE, None, f"could not parse publicScore {raw!r}")
            if status == "error":
                return ScoreResult(ERRORED, None, "Kaggle reported the submission errored")

        if time.monotonic() >= deadline:
            return ScoreResult(PENDING, None, f"still scoring after {timeout}s")
        time.sleep(wait)
        wait = min(max(int(wait * backoff), interval), max_interval)


def fetch_score(competition: str, timeout: int = 120, interval: int = 10) -> float | None:
    """Poll Kaggle for the most recent submission's public leaderboard score.

    Thin backward-compatible wrapper over :func:`poll_submission_score`; returns the score
    as a float, or None if it timed out, errored, or couldn't be parsed. New callers that
    need to tell those cases apart should use :func:`poll_submission_score` directly.
    """
    return poll_submission_score(competition, timeout=timeout, interval=interval).score
