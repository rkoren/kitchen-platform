"""MLflow tracking setup and run wrapper.

Usage::

    from kitchen.tracking import configure_from_env, init_experiment, Tracker

    configure_from_env()
    init_experiment("my-project")

    tracker = Tracker("my-experiment")
    with tracker.run(params=params) as run:
        model = train(...)
        tracker.log_metrics({"accuracy": 0.92})
        tracker.log_model(model, "model", flavour="xgboost")

Environment variables:
    MLFLOW_TRACKING_URI    — tracking store (default: ./mlruns for local file store)
    MLFLOW_ARTIFACT_BUCKET — S3 bucket name; when set, new experiments store artifacts
                             at s3://<bucket>/mlflow-artifacts/<experiment-name>
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any

import mlflow

_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
_ARTIFACT_BUCKET_ENV = "MLFLOW_ARTIFACT_BUCKET"
_ARTIFACT_PREFIX = "mlflow-artifacts"


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


# ── Functional API for env-driven setup ───────────────────────────────────────

def configure(tracking_uri: str, artifact_bucket: str | None = None) -> None:
    """Set MLflow tracking URI and optional S3 artifact bucket."""
    mlflow.set_tracking_uri(tracking_uri)
    if artifact_bucket:
        os.environ[_ARTIFACT_BUCKET_ENV] = artifact_bucket


def configure_from_env() -> None:
    """Configure MLflow from standard environment variables.

    Falls back to a local file store (./mlruns) when MLFLOW_TRACKING_URI is
    not set, so local dev works without a running server.
    """
    tracking_uri = os.environ.get(_TRACKING_URI_ENV, "./mlruns")
    artifact_bucket = os.environ.get(_ARTIFACT_BUCKET_ENV)
    configure(tracking_uri=tracking_uri, artifact_bucket=artifact_bucket)


def init_experiment(name: str) -> str:
    """Get or create an MLflow experiment by name.

    When MLFLOW_ARTIFACT_BUCKET is set, new experiments are created with
    artifacts stored at s3://<bucket>/mlflow-artifacts/<name>. Existing
    experiments keep their original artifact location.

    Returns the experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    artifact_bucket = os.environ.get(_ARTIFACT_BUCKET_ENV)
    artifact_location = (
        f"s3://{artifact_bucket}/{_ARTIFACT_PREFIX}/{name}" if artifact_bucket else None
    )
    return mlflow.create_experiment(name, artifact_location=artifact_location)
