"""Abstract base classes defining the contract for project-defined pipeline steps.

Project repos implement these and optionally wire them into dvc.yaml stages (requires ``pip install kitchen[dvc]``):

    # src/features/run.py
    from kitchen.steps import FeatureBuilder
    class MyFeatures(FeatureBuilder):
        def build(self, raw: pd.DataFrame | dict[str, pd.DataFrame], params: dict) -> pd.DataFrame: ...

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
from typing import TYPE_CHECKING, Any

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

    def sources(self, params: dict) -> list[str]:
        """Return the list of raw source filenames to load.

        Override to declare multiple input files; the default returns the single
        ``raw_file`` from params (backward-compatible with existing subclasses).
        """
        return [_resolve(params, "raw_file", "data.csv")]

    @abstractmethod
    def build(self, raw: pd.DataFrame | dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
        """Return a processed DataFrame from raw input.

        ``raw`` is a plain DataFrame when ``sources()`` returns a single file
        (the default); it is a ``dict[filename, DataFrame]`` when ``sources()``
        returns multiple files.
        """

    def run(self, store: DataStore, params: dict) -> None:
        """Load raw data, build features, persist to processed stage."""
        src = self.sources(params)
        if len(src) == 1:
            raw = store.load_csv(src[0])
        else:
            raw = {f: store.load_csv(f) for f in src}
        processed = self.build(raw, params)
        store.save_parquet(processed, _resolve(params, "processed_file", "features.parquet"))


def _log_feature_importances(model: object) -> None:
    """Best-effort: log feature importances to the active MLflow run.

    Logs raw values as feature_importances.json (Artifacts tab) and normalized
    values as fi.<name> metrics so they appear in the run comparison view.

    Supports XGBoost Booster (get_score) and sklearn estimators that expose
    feature_importances_ alongside feature_names_in_.
    """
    try:
        import mlflow as _mlflow

        if hasattr(model, "get_score"):
            importances = model.get_score(importance_type="gain")
        elif hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
            importances = dict(zip(model.feature_names_in_, model.feature_importances_.tolist()))
        else:
            return
        if not importances:
            return
        _mlflow.log_dict(importances, "feature_importances.json")
        total = sum(importances.values())
        normalized = {k: v / total for k, v in importances.items()} if total > 0 else importances
        _mlflow.log_metrics({f"fi.{k}": v for k, v in normalized.items()})
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

        from kitchen.tracking import log_run_context  # noqa: PLC0415

        processed_file = _resolve(params, "processed_file", "features.parquet")
        df = store.load_parquet(processed_file)
        store.models_dir.mkdir(parents=True, exist_ok=True)
        data_path = store.processed_dir / processed_file
        # NB-008: getattr default is True so the model is logged unless a caller
        # (kitchen.init_run(log_model=False)) explicitly opts out — keep the default
        # here; mock-based tests rely on a truthy attribute meaning "log the model".
        log_model = getattr(tracker, "log_model_enabled", True)
        if _mlflow.active_run() is not None:
            log_run_context(params=params, data_path=data_path)
            model = self.fit(df, params)
            if log_model:
                tracker.log_model(model, artifact_path="model", flavour=self.model_flavour)
            _log_feature_importances(model)
            return model
        with tracker.run(run_name=params.get("run_name"), params=params):
            log_run_context(params=params, data_path=data_path)
            model = self.fit(df, params)
            if log_model:
                tracker.log_model(model, artifact_path="model", flavour=self.model_flavour)
            _log_feature_importances(model)
            return model


class Evaluator(ABC):
    """Scores a trained model and emits metrics."""

    @abstractmethod
    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, Any]:
        """Return a flat dict of metric_name -> value.

        Values are normally floats. Two keys are reserved for non-scalar
        payloads that ``run()`` extracts before writing ``metrics.json`` so the
        metrics file stays numeric:

        - ``"calibration"`` — a reliability-curve list from
          :func:`kitchen.modeling.compute_calibration_curve`; ``run()`` writes
          it to a sibling ``calibration.json`` and logs it to the active MLflow
          run, where ``kitchen push`` (DASH-006) picks it up for the dashboard.
        """

    def run(self, model: object, store: DataStore, params: dict) -> dict[str, Any]:
        """Load eval data, compute metrics, write metrics.json.

        Reserved non-scalar keys (currently ``"calibration"``) are split out of
        the returned dict, persisted to sibling artifacts, and removed from
        ``metrics.json`` so the leaderboard sees only scalar metrics.
        """
        df = store.load_parquet(_resolve(params, "processed_file", "features.parquet"))
        metrics = self.evaluate(model, df)
        metrics_path = Path(_resolve(params, "metrics_file", "metrics.json"))

        # DASH-006: split the calibration curve out of the scalar metrics. Write
        # it next to metrics.json (so `kitchen push` can read it without a live
        # MLflow server) and log it to the active run when one is open.
        calibration = metrics.pop("calibration", None)
        if isinstance(calibration, list) and calibration:
            cal_path = metrics_path.parent / "calibration.json"
            cal_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            try:
                import mlflow as _mlflow  # noqa: PLC0415 — lazy, optional at eval time

                if _mlflow.active_run() is not None:
                    _mlflow.log_dict(calibration, "calibration.json")
            except Exception:
                pass  # MLflow logging is best-effort; the disk copy is canonical

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
