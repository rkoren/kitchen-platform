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

import mlflow
import yaml
from dotenv import load_dotenv

from kitchen.store import DataStore
from kitchen.tracking import Tracker, configure_from_env, init_experiment

load_dotenv()

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "default")


def _apply_overrides(params: dict, overrides: dict) -> None:
    """Apply dot-notation key=value overrides to params in-place.

    Creates missing intermediate dicts so --override new_section.key=v works
    even when new_section is absent from params.yaml.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        target = params
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


def _build(params: dict) -> None:
    from src.features.run import build  # project-provided

    build(params, DataStore())


def _train(params: dict, overrides: dict | None = None) -> None:
    from src.train.run import train  # project-provided

    configure_from_env()
    experiment = params.get("experiment", EXPERIMENT)
    init_experiment(experiment)
    tracker = Tracker(experiment)
    with tracker.run(run_name=params.get("run_name"), params=params):
        if overrides:
            mlflow.set_tags({f"override.{k}": str(v) for k, v in overrides.items()})
        train(params, DataStore(), tracker)
    print("Training complete — see MLflow for metrics.")


def train_pipeline(params_file: str = "params.yaml", overrides: dict | None = None) -> None:
    """Run a single training pass: features → train → log to MLflow."""
    with open(params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if overrides:
        _apply_overrides(params, overrides)
    _build(params)
    _train(params, overrides=overrides)


if __name__ == "__main__":
    train_pipeline()
