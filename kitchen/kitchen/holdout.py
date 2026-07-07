"""Frozen-holdout scoring — a trusted generalization metric distinct from CV (CBB-017).

The platform scores *every* training run's model against a project-produced, parity-matched
holdout parquet and logs ``holdout_<metric>`` onto the run — a number the model was never
trained on, so it separates CV-overfit from real generalization. The boundary: the *project*
produces the parquet (leak-free by construction, living outside ``data/raw``); the *platform*
owns scoring it, the parity guard, and the distinct metric name. See ``HoldoutSpec``.

Discipline: iterate on the CV metric, check the holdout sparingly — best-of-N by the holdout
overfits it by selection even when no row leaks (treat it like a Kaggle private leaderboard).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from kitchen.config import HOLDOUT_CLASSIFICATION_METRICS, HoldoutSpec

log = logging.getLogger(__name__)

_FLAVOR_LOADERS = {
    "xgboost": "mlflow.xgboost",
    "lightgbm": "mlflow.lightgbm",
    "sklearn": "mlflow.sklearn",
}


def _load_logged_model(run_id: str, artifact_path: str) -> Any:
    """Load the model an in-progress/just-finished run logged, picking the right flavor.

    Mirrors ``kitchen run evaluate`` (CBB-010): prefer a framework flavor
    (xgboost/lightgbm/sklearn) so a composite sklearn-flavor model loads as its real object
    with its custom predict surface intact; fall back to ``pyfunc`` only when that's the
    flavor actually present. The model is the one logged under ``mlflow.model_artifact_path``.
    """
    import mlflow

    uri = f"runs:/{run_id}/{artifact_path}"
    loader = "mlflow.sklearn"
    try:
        flavors = mlflow.models.get_model_info(uri).flavors
        framework = next((f for f in ("xgboost", "lightgbm", "sklearn") if f in flavors), None)
        if framework is not None:
            loader = _FLAVOR_LOADERS[framework]
        elif "python_function" in flavors:
            loader = "mlflow.pyfunc"
    except Exception:
        pass
    return importlib.import_module(loader).load_model(uri)


def _predict(model: Any, features_df: Any, *, metric: str, predict_method: str | None) -> np.ndarray:
    """Get model outputs to score: probabilities for proba/classification metrics, else points.

    With ``predict_method`` set, that exact method is called (e.g. cbb's ``predict_batch``); a
    2-D probability return is reduced to the positive-class column. Otherwise the sklearn
    convention is used: ``predict_proba`` for a classification metric, ``predict`` for a
    regression one.
    """
    needs_proba = metric in HOLDOUT_CLASSIFICATION_METRICS
    if predict_method is not None:
        out = np.asarray(getattr(model, predict_method)(features_df), dtype=float)
    elif needs_proba:
        out = np.asarray(model.predict_proba(features_df), dtype=float)
    else:
        out = np.asarray(model.predict(features_df), dtype=float)
    if out.ndim == 2 and out.shape[1] == 2 and needs_proba:
        out = out[:, 1]  # positive-class probability
    return out


def _compute_metric(metric: str, y_true: np.ndarray, preds: np.ndarray) -> float:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    if metric == "brier":
        return float(brier_score_loss(y_true, preds))
    if metric == "log_loss":
        return float(log_loss(y_true, preds))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, preds))
    if metric == "accuracy":
        return float(accuracy_score(y_true, (preds >= 0.5).astype(int)))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, preds)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, preds))
    if metric == "r2":
        return float(r2_score(y_true, preds))
    raise ValueError(f"unsupported holdout metric {metric!r}")  # unreachable: HoldoutSpec gates it


def score_run_holdout(
    run_id: str,
    params: dict[str, Any],
    *,
    model_artifact_path: str = "model",
    client: Any | None = None,
) -> dict[str, float]:
    """Score the run's logged model on the frozen holdout; log ``holdout_<metric>`` + return it.

    A no-op (returns ``{}``) when no ``holdout:`` is configured, when the parquet is absent and
    ``optional`` is true, or when feature parity is broken (a loud warning, not a silently-wrong
    metric). Raises on a misconfigured holdout (absent-but-required path, missing label column,
    no resolvable feature list) so the mistake surfaces rather than passing quietly.

    The metric is logged onto ``run_id`` after the run has closed, via an ``MlflowClient`` — so
    this runs in ``_train`` once the model artifact exists. The metric name is deliberately
    ``holdout_<metric>`` (distinct from the CV metric) so it never collides with it (CBB-019).
    """
    raw = params.get("holdout")
    if not raw:
        return {}
    spec = raw if isinstance(raw, HoldoutSpec) else HoldoutSpec(**raw)

    path = Path(spec.path)
    if not path.exists():
        if spec.optional:
            log.info("holdout: %s absent — skipping (holdout.optional is true)", path)
            return {}
        raise FileNotFoundError(
            f"holdout.path {path} not found (set holdout.optional: true to skip until it exists)"
        )

    import pandas as pd

    df = pd.read_parquet(path)
    if spec.label not in df.columns:
        raise ValueError(
            f"holdout.label {spec.label!r} not in {path} (columns: {sorted(df.columns)})"
        )

    features = spec.features or params.get("feature_candidates")
    if not features:
        raise ValueError(
            "holdout: no features to score — set holdout.features or feature_candidates "
            "(the model's training feature list)"
        )

    from kitchen.submit import check_feature_parity

    missing = check_feature_parity(features, df)
    if missing:
        log.warning(
            "holdout: %d model feature(s) missing from %s — SKIPPING scoring (a trusted metric "
            "must not be computed on zero-filled features): %s",
            len(missing),
            path,
            ", ".join(sorted(missing)),
        )
        return {}

    model = _load_logged_model(run_id, model_artifact_path)
    preds = _predict(
        model, df[features], metric=spec.metric, predict_method=spec.predict_method
    )
    y_true = df[spec.label].to_numpy()
    value = _compute_metric(spec.metric, y_true, preds)
    results = {f"holdout_{spec.metric}": value, "holdout_n": float(len(df))}

    # CBB-025: score named subpopulations too. `preds` is row-aligned to `df`, so each segment is
    # a mask over the *same* predictions — never re-predicted — keeping the full-set and segment
    # numbers exactly consistent.
    for name, seg in (spec.segments or {}).items():
        if seg.col not in df.columns:
            raise ValueError(
                f"holdout.segments.{name}: column {seg.col!r} not in {path} "
                f"(columns: {sorted(df.columns)})"
            )
        mask = (df[seg.col] == seg.eq).to_numpy()
        n = int(mask.sum())
        if n == 0:
            log.warning(
                "holdout: segment %r (%s == %r) matched 0 rows — skipping", name, seg.col, seg.eq
            )
            continue
        try:
            seg_value = _compute_metric(spec.metric, y_true[mask], preds[mask])
            if not np.isfinite(seg_value):  # e.g. roc_auc on a single-class subset → nan
                raise ValueError(f"{spec.metric} is undefined on this subset (got {seg_value})")
        except Exception as exc:  # noqa: BLE001 — skip this segment loudly, keep the rest
            log.warning(
                "holdout: segment %r (%s == %r, %d rows) — %s not computable: %s — skipping",
                name, seg.col, seg.eq, n, spec.metric, exc,
            )
            continue
        results[f"holdout_{spec.metric}_{name}"] = seg_value
        results[f"holdout_n_{name}"] = float(n)

    import mlflow.tracking

    mlflow_client = client or mlflow.tracking.MlflowClient()
    for key, val in results.items():
        mlflow_client.log_metric(run_id, key, val)
    log.info("holdout: holdout_%s=%.6f on %d rows", spec.metric, value, len(df))
    return results
