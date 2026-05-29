"""Notebook-friendly MLflow helpers: experiment() and init_run().

Usage::

    # Ad-hoc metrics without a Trainer class:
    import kitchen
    with kitchen.experiment("my-project") as run:
        run.log(val_accuracy=0.81)
        run.log_params(max_depth=6)

    # With a project-defined Trainer subclass:
    with kitchen.init_run(params) as tracker:
        MyTrainer().run(store, tracker, params)
"""

from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

import mlflow

from kitchen.tracking import (
    Tracker,
    _flatten,
    configure_from_env,
    init_experiment,
    log_run_context,
)


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


@contextlib.contextmanager
def init_run(
    params: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
) -> Generator[Tracker, None, None]:
    """Context manager that opens a tracked MLflow run and yields a Tracker.

    Designed for notebook use with project-defined Trainer subclasses:

        params = yaml.safe_load(open("params.yaml"))
        store = DataStore()
        with kitchen.init_run(params) as tracker:
            MyTrainer().run(store, tracker, params)

    The yielded Tracker holds an active MLflow run, so Trainer.run() detects
    it and logs into the existing run rather than opening a nested one.

    When params is None, auto-discovers and loads params.yaml by searching the
    current directory and its parents. Falls back to sqlite:///mlruns.db if
    MLflow is unreachable — never raises in a notebook context.

    Precedence for tracking URI: env var > params mlflow section > sqlite:///mlruns.db.

    Args:
        params: project params dict; auto-discovers and loads params.yaml when None.
        run_name: optional display name for this MLflow run.

    Yields:
        Tracker configured for the project's MLflow experiment.
    """
    if params is None:
        params_yaml = _find_params_yaml(Path.cwd())
        if params_yaml is not None:
            _seed_env_from_params_yaml(params_yaml)
            try:
                import yaml

                with open(params_yaml, encoding="utf-8") as f:
                    params = yaml.safe_load(f) or {}
            except Exception:
                params = {}
        else:
            params = {}
    else:
        mlflow_cfg = params.get("mlflow", {}) or {}
        uri = mlflow_cfg.get("tracking_uri")
        bucket = mlflow_cfg.get("artifact_bucket")
        if uri and not os.environ.get("MLFLOW_TRACKING_URI"):
            os.environ["MLFLOW_TRACKING_URI"] = uri
        if bucket and not os.environ.get("MLFLOW_ARTIFACT_BUCKET"):
            os.environ["MLFLOW_ARTIFACT_BUCKET"] = bucket

    experiment_name = params.get("experiment", "default")

    try:
        configure_from_env()
        init_experiment(experiment_name)
    except Exception as exc:
        warnings.warn(
            f"MLflow setup failed ({exc}); falling back to sqlite:///mlruns.db",
            stacklevel=2,
        )
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment(experiment_name)

    tracker = Tracker(experiment_name)
    with tracker.run(run_name=run_name, params=params):
        yield tracker
