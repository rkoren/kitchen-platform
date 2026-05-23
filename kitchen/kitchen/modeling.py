"""Modeling helpers: splits and metrics.

Quick usage::

    from kitchen.modeling import train_val_split, classification_metrics, regression_metrics

    train_df, val_df = train_val_split(df, target_col="target")
    metrics = classification_metrics(y_true, y_pred, y_proba=y_proba)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def train_val_split(
    df: pd.DataFrame,
    target_col: str,
    val_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train and validation DataFrames.

    Args:
        df: Full dataset including the target column.
        target_col: Column name used as the stratification key.
        val_size: Fraction of rows to reserve for validation (default 0.2).
        seed: Random seed for reproducibility (default 42).
        stratify: When True, preserves class balance across splits. Set False
            for regression targets or when a class has fewer than 2 samples.

    Returns:
        ``(train_df, val_df)`` — both retain the original column order and
        index values.
    """
    from sklearn.model_selection import train_test_split

    strat = df[target_col] if stratify else None
    train, val = train_test_split(df, test_size=val_size, random_state=seed, stratify=strat)
    return train, val


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike | None = None,
    average: str = "binary",
) -> dict[str, float]:
    """Compute standard classification metrics.

    Always returns ``accuracy`` and ``f1``.  When *y_proba* is supplied,
    also returns ``log_loss`` and ``roc_auc``.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted class labels (hard predictions).
        y_proba: Predicted probabilities for the positive class (binary) or
            probability matrix (multiclass). Optional.
        average: Averaging strategy passed to ``f1_score`` — ``"binary"``
            (default), ``"macro"``, ``"micro"``, or ``"weighted"``.

    Returns:
        Flat ``dict[str, float]`` suitable for ``mlflow.log_metrics()``.
    """
    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average)),
    }
    if y_proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_proba))
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """Compute standard regression metrics.

    Returns ``rmse``, ``mae``, and ``r2``.

    Args:
        y_true: Ground-truth continuous values.
        y_pred: Predicted continuous values.

    Returns:
        Flat ``dict[str, float]`` suitable for ``mlflow.log_metrics()``.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
