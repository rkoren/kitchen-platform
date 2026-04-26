"""MLflow tracking wrapper.

Usage::

    from kitchen.tracking import Tracker

    tracker = Tracker("my-experiment")
    with tracker.run(params=params) as run:
        model = train(...)
        tracker.log_metrics({"accuracy": 0.92})
        tracker.log_model(model, "model", flavour="xgboost")
"""
import contextlib
from collections.abc import Generator
from typing import Any

import mlflow


_FLAVOURS: dict[str, Any] = {
    "sklearn": mlflow.sklearn,
    "xgboost": mlflow.xgboost,
    "pyfunc": mlflow.pyfunc,
}


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dot-separated keys for MLflow log_params."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, experiment: str, tracking_uri: str | None = None) -> None:
        """Set the MLflow experiment, optionally pointing at a remote tracking server."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    @contextlib.contextmanager
    def run(
        self,
        run_name: str | None = None,
        params: dict | None = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Context manager that starts an MLflow run and logs flattened params on entry."""
        with mlflow.start_run(run_name=run_name) as active_run:
            if params:
                mlflow.log_params(_flatten(params))
            yield active_run

    @staticmethod
    def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of scalar metrics to the active MLflow run."""
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_model(model: Any, artifact_path: str, flavour: str = "sklearn") -> None:
        """Persist a model artifact using the named MLflow flavour (sklearn, xgboost, pyfunc)."""
        mod = _FLAVOURS.get(flavour)
        if mod is None:
            raise ValueError(f"Unknown flavour: {flavour!r}. Choose from: {list(_FLAVOURS)}")
        mod.log_model(model, artifact_path)
