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


def _find_project_config(start: Path) -> Path | None:
    """Search ``start`` and its parents for the project manifest — ``menu.yaml`` (canonical)
    or a legacy ``params.yaml`` — returning the first found. Menu-aware so a notebook in a
    menu-only project still discovers its MLflow config (INT-009)."""
    for directory in [start, *start.parents]:
        for name in ("menu.yaml", "params.yaml"):
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _seed_env_from_config(config_path: Path) -> None:
    """Set MLFLOW_TRACKING_URI / MLFLOW_ARTIFACT_BUCKET from the manifest's ``mlflow:`` section
    if not already set.

    Reads only **literal string** values: a menu's ``{from_role}`` reference (a dict) is
    resolved to the environment by INT-003 (``kitchen menu materialize``), so it's skipped
    here. Uses raw yaml.safe_load so a partial/menu manifest never triggers KitchenConfig
    validation.
    """
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        mlflow_cfg = raw.get("mlflow", {}) or {}
        tracking_uri = mlflow_cfg.get("tracking_uri")
        artifact_bucket = mlflow_cfg.get("artifact_bucket")
        if isinstance(tracking_uri, str) and tracking_uri and not os.environ.get("MLFLOW_TRACKING_URI"):
            if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
                db_name = tracking_uri[len("sqlite:///"):]
                abs_path = (config_path.parent / db_name).resolve()
                tracking_uri = f"sqlite:///{abs_path}"
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        if isinstance(artifact_bucket, str) and artifact_bucket and not os.environ.get(
            "MLFLOW_ARTIFACT_BUCKET"
        ):
            os.environ["MLFLOW_ARTIFACT_BUCKET"] = artifact_bucket
    except Exception:
        pass


@contextlib.contextmanager
def experiment(
    name: str,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    *,
    exploratory: bool = False,
) -> Generator[ExperimentRun, None, None]:
    """Zero-ceremony MLflow experiment context manager.

    Precedence for tracking URI: env var > menu.yaml/params.yaml > sqlite:///mlruns.db.
    Auto-discovers the project manifest (menu.yaml or params.yaml) by searching cwd + parents.
    Falls back to the local SQLite store if the configured MLflow server is
    unreachable — never raises so notebook cells continue to run.

    Metric naming matters: ``kitchen leaderboard`` ranks by the metric in your
    ``thresholds:`` section (e.g. ``loto_brier``). Log under that exact name —
    a run that logs ``val_brier`` when the threshold is ``loto_brier`` will not
    appear in the default leaderboard (``kitchen leaderboard --metric val_brier``
    still finds it, and the command hints at the mismatch).

    Args:
        name: MLflow experiment name.
        run_name: Optional display name for this run.
        params: Optional dict of hyperparameters to log at run start.
        exploratory: When True, tags the run ``run_type=exploratory`` so notebook
            sketches can be isolated from or suppressed in ``kitchen leaderboard``
            via ``--only-exploratory`` / ``--exclude-exploratory``. Use it for
            throwaway experiments that should not be mistaken for pipeline runs.

    Yields:
        ExperimentRun with .log(), .log_params(), and .set_tag() helpers.
    """
    config_path = _find_project_config(Path.cwd())
    if config_path is not None:
        _seed_env_from_config(config_path)

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
        if exploratory:
            mlflow.set_tag("run_type", "exploratory")
        if params:
            mlflow.log_params(_flatten(params))
        log_run_context(params=params)
        yield ExperimentRun(active_run)


@contextlib.contextmanager
def init_run(
    params: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    exploratory: bool = False,
    log_model: bool = True,
) -> Generator[Tracker, None, None]:
    """Context manager that opens a tracked MLflow run and yields a Tracker.

    Designed for notebook use with project-defined Trainer subclasses:

        params = kitchen.load_params("menu.yaml")   # or params.yaml
        store = DataStore()
        with kitchen.init_run(params) as tracker:
            MyTrainer().run(store, tracker, params)

    The yielded Tracker holds an active MLflow run, so Trainer.run() detects
    it and logs into the existing run rather than opening a nested one.

    When params is None, auto-discovers and loads the manifest (menu.yaml or params.yaml) by searching the
    current directory and its parents. Falls back to sqlite:///mlruns.db if
    MLflow is unreachable — never raises in a notebook context.

    Precedence for tracking URI: env var > params mlflow section > sqlite:///mlruns.db.

    Args:
        params: project params dict; auto-discovers + loads menu.yaml/params.yaml when None.
        run_name: optional display name for this MLflow run.
        exploratory: When True, tags the run ``run_type=exploratory`` (see
            :func:`experiment`) so notebook runs can be filtered in
            ``kitchen leaderboard``.
        log_model: When False, ``Trainer.run()`` skips ``tracker.log_model()`` for
            this session, so a throwaway notebook experiment does not persist a
            model artifact alongside production candidates (NB-008). Validation
            metrics and feature importances are still logged.

    Yields:
        Tracker configured for the project's MLflow experiment.
    """
    if params is None:
        config_path = _find_project_config(Path.cwd())
        if config_path is not None:
            _seed_env_from_config(config_path)
            try:
                from kitchen.menu import load_params

                # load_params normalizes a menu (injects experiment from project) so the
                # experiment name below resolves for menu-only notebook projects (INT-009).
                params = load_params(str(config_path))
            except Exception:
                params = {}
        else:
            params = {}
    else:
        mlflow_cfg = params.get("mlflow", {}) or {}
        uri = mlflow_cfg.get("tracking_uri")
        bucket = mlflow_cfg.get("artifact_bucket")
        if isinstance(uri, str) and uri and not os.environ.get("MLFLOW_TRACKING_URI"):
            os.environ["MLFLOW_TRACKING_URI"] = uri
        if isinstance(bucket, str) and bucket and not os.environ.get("MLFLOW_ARTIFACT_BUCKET"):
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
    tracker.log_model_enabled = log_model
    with tracker.run(run_name=run_name, params=params):
        if exploratory:
            mlflow.set_tag("run_type", "exploratory")
        yield tracker
