"""Modeling helpers: splits, metrics, cross-validation, clipping, seeding, and ensembling.

Quick usage::

    from kitchen.modeling import (
        train_val_split,
        cross_validate,
        classification_metrics,
        regression_metrics,
        clip_proba,
        clip_predictions,
        set_seed,
        blend_predictions,
        rank_average,
        voting_predict,
        make_stack_features,
    )

    set_seed(42)                            # reproducible run — call once at entry point
    train_df, val_df = train_val_split(df, target_col="target")
    metrics = classification_metrics(y_true, y_pred, y_proba=y_proba)

    safe_proba = clip_proba(raw_proba)      # avoid log(0) in log_loss
    safe_pred  = clip_predictions(y_pred, low=0.0, high=1.0)  # regression range guard

    cv = cross_validate(
        df=train_df,
        target_col="target",
        estimator_fn=lambda: LogisticRegression(max_iter=200),
        metric_fn=classification_metrics,
        return_proba=True,
    )
    # {'accuracy_mean': 0.82, 'accuracy_std': 0.03, 'f1_mean': ..., ...}

    # Ensemble: blend three model outputs
    blended = blend_predictions([lgbm_proba, xgb_proba, cat_proba], weights=[0.5, 0.3, 0.2])

    # Stacking: generate OOF features then train a meta-learner
    oof, test_feats = make_stack_features(
        [lambda: LogisticRegression(), lambda: RandomForestClassifier()],
        X_train, y_train, X_test,
    )
    meta = LogisticRegression().fit(oof, y_train)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Any, Callable

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


def cross_validate(
    df: pd.DataFrame,
    target_col: str,
    estimator_fn: Callable[[], Any],
    metric_fn: Callable[..., dict[str, float]],
    n_splits: int = 5,
    seed: int = 42,
    stratify: bool = True,
    return_proba: bool = False,
) -> dict[str, float]:
    """Run K-fold cross-validation and return aggregated metrics.

    Trains a fresh estimator on each fold and accumulates the metrics
    returned by *metric_fn*.  Returns mean and standard deviation for every
    metric key, as a flat ``dict[str, float]`` suitable for
    ``mlflow.log_metrics()``.

    Args:
        df: Full dataset including the target column.
        target_col: Name of the target column.
        estimator_fn: Zero-argument callable that returns a new, unfitted
            sklearn-compatible estimator.  Called once per fold so each fold
            starts from scratch.
            Example: ``lambda: XGBClassifier(**params["model"])``.
        metric_fn: Metric function with signature
            ``(y_true, y_pred, **kwargs) -> dict[str, float]``.
            Use :func:`classification_metrics` or :func:`regression_metrics`.
        n_splits: Number of CV folds (default 5).
        seed: Random seed passed to the fold splitter (default 42).
        stratify: When ``True`` (default), uses ``StratifiedKFold`` to
            preserve class balance across folds.  Set ``False`` for regression
            targets or when any class has fewer than *n_splits* samples.
        return_proba: When ``True``, calls ``estimator.predict_proba()`` after
            each fold and passes the result to *metric_fn* as ``y_proba``.
            For binary classifiers the positive-class column is extracted
            automatically.  Set ``True`` when using :func:`classification_metrics`
            to obtain ``roc_auc`` and ``log_loss``.

    Returns:
        ``dict[str, float]`` with ``{metric}_mean`` and ``{metric}_std`` keys
        for every metric returned by *metric_fn*.

    Example::

        from sklearn.linear_model import LogisticRegression
        from kitchen.modeling import cross_validate, classification_metrics

        cv = cross_validate(
            df=train_df,
            target_col="target",
            estimator_fn=lambda: LogisticRegression(max_iter=200),
            metric_fn=classification_metrics,
            return_proba=True,
        )
        # {'accuracy_mean': 0.82, 'accuracy_std': 0.03, 'f1_mean': ..., ...}
        tracker.log_metrics(cv)
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    splitter: KFold | StratifiedKFold = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )

    fold_metrics: list[dict[str, float]] = []
    for train_idx, val_idx in splitter.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        estimator = estimator_fn()
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)

        kwargs: dict = {}
        if return_proba and hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_val)
            # Binary: pass positive-class column; multiclass: pass full matrix
            kwargs["y_proba"] = proba[:, 1] if proba.shape[1] == 2 else proba

        fold_metrics.append(metric_fn(y_val, y_pred, **kwargs))

    # Aggregate across folds — mean and std for each metric key
    all_keys = fold_metrics[0].keys()
    result: dict[str, float] = {}
    for key in all_keys:
        vals = np.array([fm[key] for fm in fold_metrics])
        result[f"{key}_mean"] = float(vals.mean())
        result[f"{key}_std"] = float(vals.std())

    return result


# ---------------------------------------------------------------------------
# M-008: Prediction clipping
# ---------------------------------------------------------------------------


def clip_proba(
    arr: ArrayLike,
    eps: float = 1e-6,
) -> np.ndarray:
    """Clip probability predictions to the open interval ``(eps, 1 - eps)``.

    Probabilities of exactly 0 or 1 produce ``-inf`` / ``inf`` in log-loss
    and similar metrics.  Clipping by a small epsilon is standard practice
    before scoring or uploading competition submissions.

    Works on both 1-D arrays (binary positive-class probabilities) and 2-D
    matrices (multiclass probability rows).  Each element is clipped
    independently; row sums are **not** re-normalised — if you need that,
    call ``arr / arr.sum(axis=1, keepdims=True)`` afterwards.

    Args:
        arr: Probability array of shape ``(n,)`` or ``(n, k)``.
        eps: Small positive value defining the clipping bounds
            ``[eps, 1 - eps]``.  Default ``1e-6``.

    Returns:
        NumPy array of the same shape as *arr* with values in
        ``[eps, 1 - eps]``.

    Example::

        from kitchen.modeling import clip_proba

        safe = clip_proba(model.predict_proba(X)[:, 1])
        metrics = classification_metrics(y_true, y_pred, y_proba=safe)
    """
    return np.clip(np.asarray(arr, dtype=float), eps, 1.0 - eps)


def clip_predictions(
    arr: ArrayLike,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    """Clip continuous predictions to ``[low, high]``.

    Useful for regression targets with a known valid range (e.g. ratings
    between 1–5, prices that must be positive).  Either bound may be
    ``None`` to leave that side unconstrained.

    Args:
        arr: Prediction array of any shape.
        low: Lower bound (inclusive).  ``None`` means no lower clipping.
        high: Upper bound (inclusive).  ``None`` means no upper clipping.

    Returns:
        NumPy array of the same shape as *arr*.

    Raises:
        ValueError: If both *low* and *high* are ``None`` (no-op is almost
            certainly a mistake).

    Example::

        from kitchen.modeling import clip_predictions

        y_pred = clip_predictions(raw_pred, low=0.0, high=5.0)  # 1–5 rating
        y_pred = clip_predictions(raw_pred, low=0.0)            # prices ≥ 0
    """
    if low is None and high is None:
        raise ValueError("At least one of `low` or `high` must be specified.")
    return np.clip(np.asarray(arr, dtype=float), low, high)


# ---------------------------------------------------------------------------
# M-009: Seed management
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and optional deep-learning frameworks.

    Call once at the entry point of a training script (before any random
    operations) to make runs reproducible across restarts.

    Frameworks seeded:

    * ``random`` (Python stdlib)
    * ``numpy``
    * ``torch`` + ``torch.cuda`` — only if PyTorch is installed
    * ``tensorflow`` — only if TensorFlow is installed

    Args:
        seed: Integer seed to apply everywhere (default 42).

    Example::

        from kitchen.modeling import set_seed

        set_seed(42)
        train_df, val_df = train_val_split(df, target_col="target")
        model = XGBClassifier(random_state=seed).fit(X_train, y_train)
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-untyped]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf  # type: ignore[import-untyped]

        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# M-011: Ensemble helpers
# ---------------------------------------------------------------------------


def _rank_normalize(a: np.ndarray) -> np.ndarray:
    """Convert a 1-D array to average-rank percentiles in ``(0, 1)``.

    Ties receive the mean of their ordinal ranks.  The result is divided
    by ``n + 1`` so no value reaches exactly 0 or 1.
    """
    n = len(a)
    sorter = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[sorter] = np.arange(1, n + 1, dtype=float)
    # Average ranks within each group of identical values
    sorted_a = a[sorter]
    change = np.concatenate(([True], sorted_a[1:] != sorted_a[:-1], [True]))
    boundaries = np.where(change)[0]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        ranks[sorter[start:end]] = ranks[sorter[start:end]].mean()
    return ranks / (n + 1)


def blend_predictions(
    predictions: list[ArrayLike],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Weighted average of a list of prediction arrays.

    The workhorse of competition ensembling: average probabilities or
    regression outputs from several models.  Works on both 1-D arrays
    (binary probabilities or regression values) and 2-D matrices
    (multiclass probability rows).

    Args:
        predictions: List of arrays, each of the same shape ``(n,)`` or
            ``(n, k)``.  At least one array must be provided.
        weights: Optional importance weights for each model.  Need not
            sum to 1 — they are normalised automatically.  When ``None``
            all models are weighted equally (simple average).

    Returns:
        NumPy array of the same shape as each element of *predictions*.

    Raises:
        ValueError: If *predictions* is empty, arrays differ in shape,
            or *weights* has a different length than *predictions*.

    Example::

        from kitchen.modeling import blend_predictions

        blended = blend_predictions(
            [lgbm_proba, xgb_proba, cat_proba],
            weights=[0.5, 0.3, 0.2],
        )
    """
    if not predictions:
        raise ValueError("predictions list is empty")
    arrs = [np.asarray(p, dtype=float) for p in predictions]
    ref_shape = arrs[0].shape
    for i, a in enumerate(arrs[1:], 1):
        if a.shape != ref_shape:
            raise ValueError(
                f"predictions[{i}] has shape {a.shape}; expected {ref_shape}"
            )
    if weights is None:
        w = np.ones(len(arrs)) / len(arrs)
    else:
        if len(weights) != len(arrs):
            raise ValueError(
                f"len(weights)={len(weights)} != len(predictions)={len(arrs)}"
            )
        w = np.asarray(weights, dtype=float)
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / total
    return sum(wi * a for wi, a in zip(w, arrs))  # type: ignore[return-value]


def rank_average(
    predictions: list[ArrayLike],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Rank-normalise each prediction array, then compute a weighted average.

    Converting predictions to percentile ranks before blending is a classic
    Kaggle technique: it eliminates scale differences between models (e.g.
    tree outputs vs. calibrated logistic probabilities) and reduces the
    influence of outlier predictions.

    Each array is ranked with tie-averaging, then divided by ``n + 1`` so
    the result lives strictly inside ``(0, 1)``.

    Args:
        predictions: List of 1-D arrays of the same length.
        weights: Optional importance weights (normalised automatically).
            ``None`` means equal weights.

    Returns:
        1-D NumPy array of averaged normalised ranks in ``(0, 1)``.

    Raises:
        ValueError: If *predictions* is empty, any array is not 1-D, or
            arrays differ in length.

    Example::

        from kitchen.modeling import rank_average

        blended = rank_average([lgbm_pred, xgb_pred, cat_pred])
    """
    if not predictions:
        raise ValueError("predictions list is empty")
    arrs = [np.asarray(p, dtype=float) for p in predictions]
    for i, a in enumerate(arrs):
        if a.ndim != 1:
            raise ValueError(
                f"predictions[{i}] must be 1-D; got shape {a.shape}"
            )
    n = len(arrs[0])
    for i, a in enumerate(arrs[1:], 1):
        if len(a) != n:
            raise ValueError(
                f"predictions[{i}] has length {len(a)}; expected {n}"
            )
    ranked = [_rank_normalize(a) for a in arrs]
    return blend_predictions(ranked, weights=weights)


def voting_predict(
    predictions: list[ArrayLike],
    threshold: float = 0.5,
) -> np.ndarray:
    """Majority vote from a list of binary (0/1) prediction arrays.

    Averages the hard predictions and applies *threshold*.  When the
    fraction of models voting 1 meets or exceeds *threshold*, the ensemble
    predicts 1.

    Args:
        predictions: List of 1-D integer or float arrays with values in
            ``{0, 1}``.  At least one array must be provided.
        threshold: Fraction of votes required to predict 1 (default 0.5).
            With an odd number of models and ``threshold=0.5`` this is a
            strict majority vote with no ties.

    Returns:
        1-D integer array with values in ``{0, 1}``.

    Raises:
        ValueError: If *predictions* is empty or arrays differ in length.

    Example::

        from kitchen.modeling import voting_predict

        final = voting_predict([lgbm_pred, xgb_pred, cat_pred])
    """
    if not predictions:
        raise ValueError("predictions list is empty")
    arrs = [np.asarray(p, dtype=float) for p in predictions]
    n = len(arrs[0])
    for i, a in enumerate(arrs[1:], 1):
        if len(a) != n:
            raise ValueError(
                f"predictions[{i}] has length {len(a)}; expected {n}"
            )
    avg = sum(a for a in arrs) / len(arrs)
    return (avg >= threshold).astype(int)


def make_stack_features(
    estimator_fns: list[Callable[[], Any]],
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    n_splits: int = 5,
    seed: int = 42,
    stratify: bool = True,
    return_proba: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold (OOF) predictions for stacking.

    For each estimator and each fold:

    1. Train on the ``K-1`` in-fold rows.
    2. Predict on the held-out fold → fills the OOF array without leakage.
    3. Predict on *X_test* and accumulate; the final test column is the
       mean across all *K* fold models.

    Train a meta-learner on the returned ``oof_features`` (paired with
    *y_train*) and use ``test_features`` as its inputs at inference time.

    Args:
        estimator_fns: List of zero-argument callables, each returning a
            new unfitted sklearn-compatible estimator.
        X_train: Training features, shape ``(n_train, p)``.
        y_train: Training labels, shape ``(n_train,)``.
        X_test: Test features, shape ``(n_test, p)``.
        n_splits: Number of CV folds (default 5).
        seed: Random seed (default 42).
        stratify: Use ``StratifiedKFold`` (default ``True``); set
            ``False`` for regression targets.
        return_proba: When ``True``, call ``predict_proba()`` and extract
            the positive-class column for binary classifiers.  Set
            ``True`` when base models are classifiers and you want
            calibrated probabilities as meta-features.

    Returns:
        ``(oof_features, test_features)`` where:

        - ``oof_features`` — shape ``(n_train, n_estimators)``, one column
          per estimator, covering every training row exactly once.
        - ``test_features`` — shape ``(n_test, n_estimators)``, each
          column is the mean of that estimator's *K* fold predictions on
          *X_test*.

    Example::

        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from kitchen.modeling import make_stack_features

        oof, test_feats = make_stack_features(
            [lambda: LogisticRegression(), lambda: RandomForestClassifier()],
            X_train, y_train, X_test,
            return_proba=True,
        )
        meta = LogisticRegression().fit(oof, y_train)
        preds = meta.predict(test_feats)
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train)
    X_te = np.asarray(X_test, dtype=float)
    n_train, n_test, n_est = X_tr.shape[0], X_te.shape[0], len(estimator_fns)

    oof = np.zeros((n_train, n_est))
    test_preds = np.zeros((n_test, n_est))

    splitter: KFold | StratifiedKFold = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )

    for j, est_fn in enumerate(estimator_fns):
        fold_test = np.zeros((n_test, n_splits))
        for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X_tr, y_tr)):
            est = est_fn()
            est.fit(X_tr[train_idx], y_tr[train_idx])
            if return_proba and hasattr(est, "predict_proba"):
                val_proba = est.predict_proba(X_tr[val_idx])
                oof[val_idx, j] = val_proba[:, 1] if val_proba.shape[1] == 2 else val_proba.max(axis=1)
                te_proba = est.predict_proba(X_te)
                fold_test[:, fold_i] = te_proba[:, 1] if te_proba.shape[1] == 2 else te_proba.max(axis=1)
            else:
                oof[val_idx, j] = est.predict(X_tr[val_idx])
                fold_test[:, fold_i] = est.predict(X_te)
        test_preds[:, j] = fold_test.mean(axis=1)

    return oof, test_preds
