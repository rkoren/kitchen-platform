"""Evaluation for the offline quickstart — binary classification.

Loads the champion model from the MLflow registry and scores it on the same
held-out validation split used in training (same seed → consistent partition).
Reports accuracy, f1, log_loss, and roc_auc, plus a reliability curve for the
dashboard (DASH-006).
"""
from __future__ import annotations

import pandas as pd
from kitchen.modeling import (
    classification_metrics,
    compute_calibration_curve,
    train_val_split,
)
from kitchen.steps import Evaluator
from kitchen.store import DataStore


class OfflineEvaluator(Evaluator):
    def run(self, model: object, store: DataStore, params: dict) -> dict[str, float]:
        self._params = params  # stash so evaluate() can read target + seed
        return super().run(model, store, params)

    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        params = self._params
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        _, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in val_df.columns if c != target]
        X_val, y_val = val_df[features], val_df[target]

        y_pred = model.predict(X_val)
        y_proba = (
            model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        )
        metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        if y_proba is not None:
            metrics["calibration"] = compute_calibration_curve(y_val, y_proba)
        return metrics


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return OfflineEvaluator().run(model, store, params)
