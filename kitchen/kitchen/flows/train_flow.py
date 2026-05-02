"""Generic single-run training flow for kitchen competition projects.

Scaffolded projects import this via ``flows/train_flow.py``. It is a thin
wrapper that reads ``params.yaml``, runs feature engineering, and trains
the model — delegating all project-specific logic to the project's own
``src.features.run`` and ``src.train.run`` modules.

Run from the project root:
    python flows/train_flow.py
"""
from __future__ import annotations

import os

import yaml
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

from kitchen.store import DataStore
from kitchen.tracking import Tracker, configure_from_env, init_experiment

load_dotenv()

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "default")


@task
def _build(params: dict) -> None:
    from src.features.run import build  # project-provided
    build(params, DataStore())


@task
def _train(params: dict) -> None:
    from src.train.run import train  # project-provided
    log = get_run_logger()
    configure_from_env()
    experiment = params.get("experiment", EXPERIMENT)
    init_experiment(experiment)
    tracker = Tracker(experiment)
    train(params, DataStore(), tracker)
    log.info("Training complete — see MLflow for metrics.")


@flow(name="kitchen-train")
def train_pipeline(params_file: str = "params.yaml") -> None:
    """Run a single training pass: features → train → log to MLflow."""
    with open(params_file) as f:
        params = yaml.safe_load(f)
    _build(params)
    _train(params)


if __name__ == "__main__":
    train_pipeline()
