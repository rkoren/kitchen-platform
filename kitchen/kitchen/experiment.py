"""Notebook-friendly MLflow experiment context manager.

Usage::

    import kitchen
    with kitchen.experiment("my-project") as run:
        run.log(val_accuracy=0.81, accuracy=0.79)
        run.log_params(max_depth=6, eta=0.05)
        run.set_tag("note", "first baseline")
"""

from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

import mlflow

from kitchen.tracking import _flatten, configure_from_env, init_experiment, log_run_context


class ExperimentRun:
    """Handle to an active MLflow run, returned by kitchen.experiment()."""

    def __init__(self, active_run: Any) -> None:
        self._run = active_run

    @property
    def run_id(self) -> str:
        return self._run.info.run_id  # type: ignore[no-any-return]

    def log(self, **metrics: float) -> None:
        """Log scalar metrics to the active run."""
        mlflow.log_metrics(metrics)

    def log_params(self, **params: Any) -> None:
        """Log hyperparameters to the active run."""
        mlflow.log_params(params)

    def set_tag(self, key: str, value: str) -> None:
        """Set a string tag on the active run."""
        mlflow.set_tag(key, value)


def _find_params_yaml(start: Path) -> Path | None:
    """Search start and its parents for params.yaml, return the first found."""
    for directory in [start, *start.parents]:
        candidate = directory / "params.yaml"
        if candidate.exists():
            return candidate
    return None


def _seed_env_from_params_yaml(params_yaml: Path) -> None:
    """Set MLFLOW_TRACKING_URI / MLFLOW_ARTIFACT_BUCKET from params.yaml if not already set.

    Only reads the top-level ``mlflow:`` section; uses raw yaml.safe_load so a
    partial or project-specific params.yaml never triggers KitchenConfig validation.
    """
    try:
        import yaml

        with open(params_yaml, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        mlflow_cfg = raw.get("mlflow", {}) or {}
        tracking_uri = mlflow_cfg.get("tracking_uri")
        artifact_bucket = mlflow_cfg.get("artifact_bucket")
        if tracking_uri and not os.environ.get("MLFLOW_TRACKING_URI"):
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        if artifact_bucket and not os.environ.get("MLFLOW_ARTIFACT_BUCKET"):
            os.environ["MLFLOW_ARTIFACT_BUCKET"] = artifact_bucket
    except Exception:
        pass


@contextlib.contextmanager
def experiment(
    name: str,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
) -> Generator[ExperimentRun, None, None]:
    """Zero-ceremony MLflow experiment context manager.

    Precedence for tracking URI: env var > params.yaml > sqlite:///mlruns.db.
    Auto-discovers params.yaml by searching the current directory and its parents.
    Falls back to the local SQLite store if the configured MLflow server is
    unreachable — never raises so notebook cells continue to run.

    Args:
        name: MLflow experiment name.
        run_name: Optional display name for this run.
        params: Optional dict of hyperparameters to log at run start.

    Yields:
        ExperimentRun with .log(), .log_params(), and .set_tag() helpers.
    """
    params_yaml = _find_params_yaml(Path.cwd())
    if params_yaml is not None:
        _seed_env_from_params_yaml(params_yaml)

    try:
        configure_from_env()
        init_experiment(name)
    except Exception as exc:
        warnings.warn(
            f"MLflow setup failed ({exc}); falling back to sqlite:///mlruns.db",
            stacklevel=2,
        )
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment(name)

    with mlflow.start_run(run_name=run_name) as active_run:
        if params:
            mlflow.log_params(_flatten(params))
        log_run_context(params=params)
        yield ExperimentRun(active_run)
