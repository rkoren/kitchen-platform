"""MLflow Model Registry helpers: register, promote, and look up production models."""

from __future__ import annotations

import mlflow
import mlflow.tracking


def register_model(run_id: str, artifact_path: str, name: str) -> str:
    """Register a logged artifact as a versioned entry in the MLflow Model Registry.

    Args:
        run_id: MLflow run ID that contains the artifact.
        artifact_path: Path within the run's artifact store (e.g. "calibrator").
        name: Registered model name to create or append a version to.

    Returns:
        The new model version string.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, name)
    return mv.version


def get_best_run(
    experiment_name: str,
    metric: str,
    lower_is_better: bool = True,
    tag_filter: dict[str, str] | None = None,
) -> mlflow.entities.Run:
    """Find the run with the best metric value in an experiment.

    Args:
        experiment_name: MLflow experiment name.
        metric: Metric name to rank by (e.g. "brier_2026").
        lower_is_better: True for loss metrics (Brier), False for reward metrics.
        tag_filter: Optional tag key→value pairs to narrow the search
                    (e.g. {"model_variant": "challenger"}).

    Returns:
        The Run object with the best metric value.

    Raises:
        ValueError: If the experiment doesn't exist or no matching runs are found.
    """
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name!r} not found")

    filter_str = " and ".join(
        f"tags.{k} = '{v}'" for k, v in (tag_filter or {}).items()
    )
    order = "ASC" if lower_is_better else "DESC"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str or "",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if not runs:
        desc = f" with tags {tag_filter}" if tag_filter else ""
        raise ValueError(f"No runs found in experiment {experiment_name!r}{desc}")
    return runs[0]


def promote_model(name: str, version: str, stage: str = "Production") -> None:
    """Transition a registered model version to the given stage.

    Args:
        name: Registered model name.
        version: Model version string (returned by register_model).
        stage: Target stage — "Production", "Staging", or "Archived".
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name, version, stage)


def get_production_uri(name: str) -> str | None:
    """Return the model URI for the current Production version, or None if none exists.

    The URI is suitable for mlflow.sklearn.load_model() or similar loaders.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(name, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        return None
    if not versions:
        return None
    v = versions[0]
    return f"models:/{name}/{v.version}"
