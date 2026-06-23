"""Model training for the offline quickstart — logistic regression baseline.

Loads ``data/processed/features.parquet``, splits into train/val, fits a
scikit-learn ``LogisticRegression``, and logs validation metrics
(``val_accuracy``, ``val_f1``, ``val_log_loss``, ``val_roc_auc``) to the active
MLflow run. ``Trainer.run()`` opens the run before ``fit()`` is called, so
``mlflow.log_metrics()`` here is always inside a live run.

Swap ``LogisticRegression`` for any sklearn-compatible estimator to extend it.
"""
from __future__ import annotations

import mlflow
import pandas as pd
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class OfflineTrainer(Trainer):
    model_flavour = "sklearn"

    def fit(self, df: pd.DataFrame, params: dict) -> object:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]  # col 1 = P(passed=1)
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return OfflineTrainer().run(store, tracker, params)
