"""Structured experiment configuration for reproducible ML runs.

Projects define named ExperimentConfigs and log them uniformly so every
MLflow run has consistent param coverage regardless of project domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlflow


@dataclass
class ExperimentConfig:
    """A named, self-describing set of hyperparameters for one experiment run."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


def log_config(config: ExperimentConfig) -> None:
    """Log an ExperimentConfig to the active MLflow run.

    Must be called inside an active mlflow.start_run() context.
    """
    if config.params:
        mlflow.log_params(config.params)
    if config.description:
        mlflow.set_tag("description", config.description)
    mlflow.set_tag("experiment_name", config.name)
