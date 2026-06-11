"""MLflow Model Registry helpers: register, promote, and look up production models."""

from __future__ import annotations

import mlflow
import mlflow.exceptions
import mlflow.tracking


def _run_logged_model_names(run_id: str) -> list[str] | None:
    """Best-effort: names of the models a run logged (MLflow 3.x logged models).

    Returns ``None`` if the lookup fails for any reason — this only feeds an error
    message, so it must never raise.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        models = client.search_logged_models(experiment_ids=[run.info.experiment_id])
        # Keep models sourced from this run; if the attribute is absent, include it.
        return [m.name for m in models if getattr(m, "source_run_id", run_id) == run_id]
    except Exception:
        return None


def register_model(run_id: str, artifact_path: str, name: str) -> str:
    """Register a logged model as a versioned entry in the MLflow Model Registry.

    Args:
        run_id: MLflow run ID that logged the model.
        artifact_path: The name the project logged its model under
            (``mlflow.<flavor>.log_model(model, <artifact_path>)``); ``"model"`` by default.
        name: Registered model name to create or append a version to.

    Returns:
        The new model version string.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        mv = mlflow.register_model(model_uri, name)
    except mlflow.exceptions.MlflowException as exc:
        # MLflow 3.x: log_model creates a *logged model* keyed by name. When the
        # configured name doesn't match, replace the opaque "Unable to find a
        # logged_model" trace with the actual names + how to fix it.
        available = _run_logged_model_names(run_id)
        if available is not None and artifact_path not in available:
            listed = ", ".join(repr(n) for n in available) if available else "(none logged)"
            raise mlflow.exceptions.MlflowException(
                f"No model named {artifact_path!r} was logged by run {run_id[:8]}. "
                f"Models logged by this run: {listed}. Set `mlflow.model_artifact_path` "
                f"in params.yaml to one of those names (or pass `kitchen promote "
                f"--model-artifact-path <name>`)."
            ) from exc
        raise
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

    filter_str = " and ".join(f"tags.{k} = '{v}'" for k, v in (tag_filter or {}).items())
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


def promote_model(name: str, version: str, alias: str = "champion") -> None:
    """Set a named alias on a registered model version.

    Aliases replace the deprecated stage system. The default alias "champion"
    identifies the current production model; load it with
    mlflow.sklearn.load_model('models:/<name>@champion').

    Args:
        name: Registered model name.
        version: Model version string (returned by register_model).
        alias: Alias to assign (default: "champion").
    """
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(name, alias, version)


def get_production_uri(name: str, alias: str = "champion") -> str | None:
    """Return the model URI for the current champion version, or None if none exists.

    The URI is suitable for mlflow.sklearn.load_model() or similar loaders.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        client.get_model_version_by_alias(name, alias)
    except mlflow.exceptions.MlflowException:
        return None
    return f"models:/{name}@{alias}"


def get_champion_metrics(name: str, alias: str = "champion") -> dict[str, float] | None:
    """Return the champion run's logged metrics, or None if no champion exists.

    Resolves the model version behind ``alias`` and reads the metrics logged to
    its source run — the source of truth for a comparison baseline (a
    ``metrics.json`` artifact is not always logged, but ``run.data.metrics``
    always is). Feature-importance metrics (``fi.*``) and the Kaggle
    ``lb_score`` are excluded so the result matches the keys a project's
    ``metrics.json`` carries.

    Returns None when the alias is unset or the run can't be read, so callers can
    fall back gracefully (e.g. the first PR of a project, before any promote).
    """
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, alias)
        run = client.get_run(mv.run_id)
    except mlflow.exceptions.MlflowException:
        return None
    return {
        k: float(v)
        for k, v in run.data.metrics.items()
        if not k.startswith("fi.") and k != "lb_score"
    }
