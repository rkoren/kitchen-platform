"""Abstract base classes defining the contract for project-defined pipeline steps.

Project repos implement these and wire them into dvc.yaml stages:

    # src/features/run.py
    from kitchen.steps import FeatureBuilder
    class MyFeatures(FeatureBuilder):
        def build(self, raw: pd.DataFrame) -> pd.DataFrame: ...

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

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class FeatureBuilder(ABC):
    """Transforms raw data into model-ready features."""

    @abstractmethod
    def build(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return a processed DataFrame from raw input."""

    def run(self, store: "DataStore", params: dict) -> None:
        """Load raw data, build features, persist to processed stage."""
        from kitchen.store import DataStore  # avoid circular at module level

        assert isinstance(store, DataStore)
        raw = store.load_csv(params.get("raw_file", "data.csv"))
        processed = self.build(raw)
        store.save_parquet(processed, params.get("processed_file", "features.parquet"))


class Trainer(ABC):
    """Fits a model and persists it."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, params: dict) -> object:
        """Train and return a model object."""

    def run(self, store: "DataStore", tracker: "Tracker", params: dict) -> None:
        """Load features, fit model, log to MLflow, save artifact."""
        from kitchen.store import DataStore
        from kitchen.tracking import Tracker

        assert isinstance(store, DataStore)
        assert isinstance(tracker, Tracker)

        df = store.load_parquet(params.get("processed_file", "features.parquet"))
        with tracker.run(run_name=params.get("run_name"), params=params) as run:
            model = self.fit(df, params)
            tracker.log_model(model, artifact_path="model")
            store.models_dir.mkdir(parents=True, exist_ok=True)
            return model


class Evaluator(ABC):
    """Scores a trained model and emits metrics."""

    @abstractmethod
    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        """Return a flat dict of metric_name -> value."""

    def run(self, model: object, store: "DataStore", params: dict) -> dict[str, float]:
        """Load eval data, compute metrics, write metrics.json."""
        import json

        from kitchen.store import DataStore

        assert isinstance(store, DataStore)

        df = store.load_parquet(params.get("processed_file", "features.parquet"))
        metrics = self.evaluate(model, df)

        metrics_path = Path(params.get("metrics_file", "metrics.json"))
        metrics_path.write_text(json.dumps(metrics, indent=2))
        return metrics
