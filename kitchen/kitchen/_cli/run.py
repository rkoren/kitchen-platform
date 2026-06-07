from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def _promote_metric_from_thresholds(params_file: str) -> tuple[str, bool] | None:
    """Return (metric, lower_is_better) from first thresholds entry, or None if absent."""
    p = Path(params_file)
    if not p.exists():
        return None
    try:
        from kitchen.config import KitchenConfig, ThresholdSpec

        cfg = KitchenConfig.from_yaml(str(p))
        if cfg.thresholds:
            name, spec = next(iter(cfg.thresholds.items()))
            if isinstance(spec, ThresholdSpec):
                lower = spec.max is not None and spec.min is None
            else:
                lower = False  # plain float = lower bound = higher-is-better
            return name, lower
    except Exception:
        pass
    return None


def _try_auto_promote(
    params_file: str,
    metric: str,
    lower_is_better: bool,
    model_name: str | None,
) -> None:
    """Compare the latest run against the current champion; promote if it wins."""
    import os

    import mlflow.tracking

    from kitchen.config import KitchenConfig
    from kitchen.registry import promote_model, register_model
    from kitchen.tracking import configure_from_env

    configure_from_env()
    cfg = KitchenConfig.from_yaml(params_file)
    exp_name = cfg.experiment
    resolved_model = model_name or os.environ.get("MLFLOW_MODEL_NAME", f"{exp_name}-model")

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        typer.echo(f"auto-promote: experiment {exp_name!r} not found.", err=True)
        return

    new_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"metrics.{metric} > -99999",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not new_runs:
        typer.echo(f"auto-promote: no runs with metric {metric!r} in {exp_name!r}.")
        return

    new_run = new_runs[0]
    new_score = new_run.data.metrics.get(metric)
    if new_score is None:
        typer.echo(f"auto-promote: metric {metric!r} missing from latest run.")
        return

    # Look up current champion score (None if no champion yet).
    champ_score: float | None = None
    try:
        mv = client.get_model_version_by_alias(resolved_model, "champion")
        champ_run = client.get_run(mv.run_id)
        champ_score = champ_run.data.metrics.get(metric)
    except Exception:
        pass

    direction_label = "lower=better" if lower_is_better else "higher=better"
    if champ_score is None:
        wins, reason = True, "no current champion"
    elif lower_is_better:
        wins = new_score < champ_score
        reason = f"{new_score:.6f} < {champ_score:.6f} ({direction_label})"
    else:
        wins = new_score > champ_score
        reason = f"{new_score:.6f} > {champ_score:.6f} ({direction_label})"

    typer.echo()
    typer.echo(f"auto-promote: metric={metric} ({direction_label})")
    if wins:
        reg_version = register_model(new_run.info.run_id, "model", resolved_model)
        promote_model(resolved_model, reg_version, alias="champion")
        typer.echo(f"auto-promote: {new_run.info.run_id[:8]} → champion  ({reason})")
        typer.echo(f"             {resolved_model} v{reg_version} @ champion")
        # Write run_id to metrics.json so `kitchen push` can resolve the champion.
        import json as _json_ap

        try:
            _ap_metrics_path = Path(cfg.metrics_file or "metrics.json")
            if _ap_metrics_path.exists():
                _ap_existing = _json_ap.loads(_ap_metrics_path.read_text(encoding="utf-8"))
                _ap_existing["run_id"] = new_run.info.run_id
                _ap_metrics_path.write_text(
                    _json_ap.dumps(_ap_existing, indent=2) + "\n", encoding="utf-8"
                )
        except Exception:
            pass
    else:
        typer.echo(f"auto-promote: skipped — new run did not beat champion  ({reason})")



# ---------------------------------------------------------------------------
# Run sub-commands
# ---------------------------------------------------------------------------


def _coerce_override_value(s: str) -> int | float | bool | str:
    """Coerce a string override value to the most specific type that fits."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s



run_app = typer.Typer(help="Run pipeline stages.", no_args_is_help=True)




@run_app.command("features")
def run_features(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
) -> None:
    """Run the feature engineering step: raw → processed features.

    Loads src/features/run.py from the project root, calls build(params, store),
    and writes the processed DataFrame to data/processed/.

    Note: `kitchen run train` already runs features internally before training.
    Use this command to run the features step standalone (e.g. as a DVC stage
    or to inspect the processed output before committing to a full train run).
    """
    import sys

    import yaml

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    from kitchen.store import DataStore  # noqa: PLC0415

    try:
        from src.features.run import build  # project-provided  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        typer.echo(
            f"error: {exc}\nRun from the project root and make sure src/features/run.py is implemented.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        build(params, DataStore())
    except NotImplementedError:
        typer.echo(
            "error: src/features/run.py is scaffolded but not yet implemented — fill in build().",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    processed = params.get("features", {}).get("processed_file", "features.parquet")
    typer.echo(f"Features built → data/processed/{processed}")


@run_app.command("train")
def run_train(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    auto_promote: Annotated[
        bool,
        typer.Option("--auto-promote", help="Promote after training if new run beats the champion"),
    ] = False,
    promote_metric: Annotated[
        str | None,
        typer.Option(
            "--promote-metric",
            help="Metric to compare for auto-promote. Auto-detected from params.yaml thresholds when omitted.",
        ),
    ] = None,
    lower_is_better: Annotated[
        bool,
        typer.Option("--lower-is-better/--higher-is-better", help="Metric direction for promotion comparison"),
    ] = False,
    promote_model_name: Annotated[
        str | None,
        typer.Option("--model-name", help="Registered model name for auto-promote (defaults to <experiment>-model)"),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            help="Shadow a params.yaml value for this run only: key=value (repeatable). "
            "Example: --override model.max_depth=6 --override model.eta=0.05",
        ),
    ] = None,
) -> None:
    """Run the full train pipeline: features → train → log to MLflow.

    With --auto-promote, compares the new run against the current champion on
    --promote-metric and promotes automatically if it wins. When --promote-metric
    is omitted, the metric is auto-detected from the first key in params.yaml
    thresholds (direction inferred from spec type).

    With --override, one or more params.yaml values are shadowed for this run
    only without modifying the file. Overrides are logged as MLflow run tags
    (override.<key>) so they appear in kitchen leaderboard and kitchen diff.
    """
    import sys

    if auto_promote and not promote_metric:
        detected = _promote_metric_from_thresholds(params_file)
        if detected is None:
            typer.echo(
                "error: --promote-metric is required when using --auto-promote "
                "(or add a 'thresholds:' section to params.yaml for auto-detection)",
                err=True,
            )
            raise typer.Exit(1)
        promote_metric, lower_is_better = detected

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    parsed_overrides: dict | None = None
    if override:
        parsed_overrides = {}
        for item in override:
            if "=" not in item:
                typer.echo(
                    f"error: --override {item!r} must be in key=value format", err=True
                )
                raise typer.Exit(1)
            key, _, raw_val = item.partition("=")
            parsed_overrides[key.strip()] = _coerce_override_value(raw_val)

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        from kitchen.flows.train_flow import train_pipeline
    except ImportError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    try:
        train_pipeline(params_file=params_file, overrides=parsed_overrides)
    except ModuleNotFoundError as exc:
        typer.echo(
            f"error: {exc}\nRun from the project root and make sure src/ is implemented.",
            err=True,
        )
        raise typer.Exit(1)

    if auto_promote:
        _try_auto_promote(params_file, promote_metric, lower_is_better, promote_model_name)  # type: ignore[arg-type]


@run_app.command("evaluate")
def run_evaluate(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    model_uri: Annotated[
        str | None,
        typer.Option("--model-uri", help="MLflow model URI (runs:/… or models:/name@alias)"),
    ] = None,
    alias: Annotated[
        str, typer.Option("--alias", help="Registry alias when model-uri is not set")
    ] = "champion",
    flavor: Annotated[
        str, typer.Option("--flavor", help="MLflow loader flavor: sklearn, xgboost, pyfunc")
    ] = "sklearn",
) -> None:
    """Load a model from MLflow and run the project's evaluator."""
    import os
    import sys

    import yaml

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if model_uri is None:
        from kitchen.config import KitchenConfig

        cfg = KitchenConfig.from_yaml(str(path))
        model_name = os.environ.get("MLFLOW_MODEL_NAME", f"{cfg.experiment}-model")
        model_uri = f"models:/{model_name}@{alias}"

    from kitchen.tracking import configure_from_env

    configure_from_env()

    _loaders = {"sklearn": "mlflow.sklearn", "xgboost": "mlflow.xgboost", "lightgbm": "mlflow.lightgbm", "pyfunc": "mlflow.pyfunc"}
    if flavor not in _loaders:
        typer.echo(
            f"error: unknown flavor {flavor!r} — choose from: {', '.join(_loaders)}", err=True
        )
        raise typer.Exit(1)

    if flavor == "sklearn":
        try:
            import mlflow as _mlflow_fl
            _info = _mlflow_fl.models.get_model_info(model_uri)
            for _f in ("xgboost", "lightgbm", "sklearn"):
                if _f in _info.flavors and _f in _loaders:
                    flavor = _f
                    break
        except Exception:
            pass

    import importlib

    loader = importlib.import_module(_loaders[flavor])
    try:
        model = loader.load_model(model_uri)
    except Exception as exc:
        import mlflow.exceptions

        exc_lower = str(exc).lower()
        is_missing_alias = isinstance(exc, mlflow.exceptions.MlflowException) and (
            "alias" in exc_lower or alias.lower() in exc_lower
        )
        if is_missing_alias:
            typer.echo(
                f"error: No {alias!r} model registered yet for {model_uri!r}.\n"
                f"  Run `kitchen run train --auto-promote --promote-metric <metric>` first,\n"
                f"  or `kitchen promote <metric>` to promote an existing run.",
                err=True,
            )
        else:
            typer.echo(f"error loading model from {model_uri!r}: {exc}", err=True)
        raise typer.Exit(1)

    try:
        from src.evaluate.run import evaluate  # project-provided  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        typer.echo(
            f"error: {exc}\nRun from the project root and make sure src/ is implemented.",
            err=True,
        )
        raise typer.Exit(1)

    from kitchen.store import DataStore

    try:
        metrics = evaluate(model, params, DataStore())
    except Exception as exc:
        typer.echo(f"error during evaluation: {exc}", err=True)
        raise typer.Exit(1)

    # Patch run_id into metrics.json so `kitchen push` can identify the champion run.
    import json as _json_eval

    try:
        import mlflow.tracking as _mlt_eval

        _eval_client = _mlt_eval.MlflowClient()
        _mv_eval = _eval_client.get_model_version_by_alias(model_name, alias)
        _metrics_path = Path(params.get("metrics_file", "metrics.json"))
        if _metrics_path.exists():
            _existing = _json_eval.loads(_metrics_path.read_text(encoding="utf-8"))
            _existing["run_id"] = _mv_eval.run_id
            _metrics_path.write_text(_json_eval.dumps(_existing, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    typer.echo(f"\nEvaluation results ({model_uri}):")
    for k, v in metrics.items():
        typer.echo(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    typer.echo()


@run_app.command("monitor")
def run_monitor(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    local: Annotated[
        str | None,
        typer.Option("--local", help="Write report to this local path (overrides params.yaml monitor config)"),
    ] = None,
) -> None:
    """Run drift monitoring and generate an Evidently report."""
    import sys

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    from kitchen.flows.monitor_flow import monitor_pipeline

    try:
        result = monitor_pipeline(params_file=params_file, local_path_override=local)
        if result:
            typer.echo(f"Report saved to: {result}")
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

