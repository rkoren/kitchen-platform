"""Abstract base classes defining the contract for project-defined pipeline steps.

Project repos implement these and wire them into dvc.yaml stages:

    # src/features/run.py
    from kitchen.steps import FeatureBuilder
    class MyFeatures(FeatureBuilder):
        def build(self, raw: pd.DataFrame, params: dict) -> pd.DataFrame: ...

    # src/train/run.py
    from kitchen.steps import Trainer
    class MyTrainer(Trainer):
        def fit(self, df: pd.DataFrame, params: dict) -> object: ...

    # src/evaluate/run.py
    from kitchen.steps import Evaluator
    class MyEvaluator(Evaluator):
        def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]: ...
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from kitchen.store import DataStore
    from kitchen.tracking import Tracker

# Standard top-level sections that may contain file keys in nested param dicts.
_SECTIONS = ("features", "model", "evaluate")


def _resolve(params: dict, key: str, default: str) -> str:
    """Look up a file-path key in params, checking nested sections before top-level.

    Projects may store file keys either flat (``{"processed_file": "f.parquet"}``)
    or nested under a section (``{"features": {"processed_file": "f.parquet"}}``).
    Both conventions work without any subclass changes.
    """
    for section in _SECTIONS:
        val = params.get(section, {}).get(key)
        if val is not None:
            return val
    return params.get(key, default)


class FeatureBuilder(ABC):
    """Transforms raw data into model-ready features."""

    @abstractmethod
    def build(self, raw: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Return a processed DataFrame from raw input."""

    def run(self, store: DataStore, params: dict) -> None:
        """Load raw data, build features, persist to processed stage."""
        raw = store.load_csv(_resolve(params, "raw_file", "data.csv"))
        processed = self.build(raw, params)
        store.save_parquet(processed, _resolve(params, "processed_file", "features.parquet"))


def _log_feature_importances(model: object) -> None:
    """Best-effort: log feature importances to the active MLflow run.

    Supports XGBoost Booster (get_score) and sklearn estimators that expose
    feature_importances_ alongside feature_names_in_.
    """
    try:
        import mlflow as _mlflow
        if hasattr(model, "get_score"):
            # XGBoost Booster — feature names are embedded in the model
            importances = model.get_score(importance_type="gain")
        elif hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
            # sklearn estimators trained on a DataFrame (feature names auto-captured)
            importances = dict(zip(model.feature_names_in_, model.feature_importances_.tolist()))
        else:
            return
        _mlflow.log_dict(importances, "feature_importances.json")
    except Exception:
        pass  # importance logging is always best-effort


class Trainer(ABC):
    """Fits a model and persists it.

    Contract for subclasses
    -----------------------
    ``fit()`` **must** log at least one validation metric to the active MLflow
    run so that ``flows/promote.py`` can rank and compare runs.  The metric
    name should match ``MLFLOW_PROMOTE_METRIC`` in ``.env`` (default:
    ``val_accuracy``).  Use ``Tracker.log_metrics({"val_accuracy": ...})``
    or ``mlflow.log_metric(...)`` directly — either works because
    ``Trainer.run()`` ensures an active run exists before calling ``fit()``.
    """

    model_flavour: str = "sklearn"

    @abstractmethod
    def fit(self, df: pd.DataFrame, params: dict) -> object:
        """Train and return a model object.

        Log at least one validation metric (e.g. ``val_accuracy``) to the
        active MLflow run before returning.
        """

    def run(self, store: DataStore, tracker: Tracker, params: dict) -> object:
        """Load features, fit model, log to MLflow, save artifact.

        If an MLflow run is already active (opened by an experiment script),
        fits and logs inside it instead of starting a new nested run.
        """
        import mlflow as _mlflow  # noqa: PLC0415 — lazy to keep steps.py lightweight
        df = store.load_parquet(_resolve(params, "processed_file", "features.parquet"))
        store.models_dir.mkdir(parents=True, exist_ok=True)
        if _mlflow.active_run() is not None:
            model = self.fit(df, params)
            tracker.log_model(model, artifact_path="model", flavour=self.model_flavour)
            _log_feature_importances(model)
            return model
        with tracker.run(run_name=params.get("run_name"), params=params):
            model = self.fit(df, params)
            tracker.log_model(model, artifact_path="model", flavour=self.model_flavour)
            _log_feature_importances(model)
            return model


class Evaluator(ABC):
    """Scores a trained model and emits metrics."""

    @abstractmethod
    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        """Return a flat dict of metric_name -> value."""

    def run(self, model: object, store: DataStore, params: dict) -> dict[str, float]:
        """Load eval data, compute metrics, write metrics.json."""
        df = store.load_parquet(_resolve(params, "processed_file", "features.parquet"))
        metrics = self.evaluate(model, df)
        metrics_path = Path(_resolve(params, "metrics_file", "metrics.json"))
        metrics_path.write_text(json.dumps(metrics, indent=2))
        return metrics
