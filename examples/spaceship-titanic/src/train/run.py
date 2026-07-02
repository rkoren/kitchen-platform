"""Model training for Spaceship Titanic — an XGBoost baseline.

Loads ``data/processed/features.parquet``, splits into train/val, fits an
``XGBClassifier``, and logs validation metrics (``val_accuracy``, ``val_f1``,
``val_log_loss``, ``val_roc_auc``) to the active MLflow run. ``Trainer.run()``
opens the run before ``fit()`` is called, so ``mlflow.log_metrics()`` is always
inside a live run.

Tune the model under ``model:`` in ``menu.yaml`` (e.g. ``max_depth``, ``eta``) or
override at the CLI: ``kitchen run train --override model.max_depth=6``.
"""
from __future__ import annotations

import mlflow
import pandas as pd
import xgboost as xgb

from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class SpaceshipTrainer(Trainer):
    model_flavour = "xgboost"

    def fit(self, df: pd.DataFrame, params: dict) -> xgb.XGBClassifier:
        model_cfg = params.get("model", {})
        target = model_cfg["target"]
        seed = model_cfg.get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        model = xgb.XGBClassifier(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 4),
            learning_rate=model_cfg.get("eta", 0.1),
            subsample=model_cfg.get("subsample", 0.9),
            eval_metric="logloss",
            random_state=seed,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]  # P(Transported=1)
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return SpaceshipTrainer().run(store, tracker, params)
