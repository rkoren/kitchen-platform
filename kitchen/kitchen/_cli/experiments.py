from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def _resolve_experiment(experiment: str | None, params_file: str) -> str:
    if experiment:
        return experiment
    from kitchen.config import KitchenConfig

    p = Path(params_file)
    if p.exists():
        cfg = KitchenConfig.from_yaml(str(p))
        return cfg.experiment
    raise typer.BadParameter(
        f"No experiment name given and {params_file!r} not found. "
        "Pass --experiment or run from a project directory."
    )


def _time_ago(ms: int) -> str:
    import time

    diff = int(time.time()) - (ms // 1000)
    if diff < 60:
        return f"{diff}s ago"
    if diff < 3600:
        return f"{diff // 60}m ago"
    if diff < 86400:
        return f"{diff // 3600}h ago"
    return f"{diff // 86400}d ago"


def _fmt_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


# ---------------------------------------------------------------------------
# Experiments sub-commands
# ---------------------------------------------------------------------------

experiments_app = typer.Typer(help="List and compare MLflow experiment runs.", no_args_is_help=True)



@experiments_app.command("list")
def experiments_list(
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help="Experiment name")
    ] = None,
    params_file: Annotated[
        str, typer.Option("--params", help="params.yaml to read experiment from")
    ] = "params.yaml",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max runs to show")] = 10,
) -> None:
    """List recent runs in an MLflow experiment."""
    import mlflow.tracking

    exp_name = _resolve_experiment(experiment, params_file)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        typer.echo(f"Experiment {exp_name!r} not found.", err=True)
        raise typer.Exit(1)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=limit,
    )
    if not runs:
        typer.echo(f"No runs found in experiment {exp_name!r}.")
        return

    # Collect metric keys for display (priority columns, then any others, skip fi.*)
    priority = ["val_accuracy", "val_brier", "val_log_loss"]
    seen: set[str] = set()
    metric_keys: list[str] = []
    for key in priority:
        if any(key in r.data.metrics for r in runs):
            metric_keys.append(key)
            seen.add(key)
    for run in runs:
        for key in run.data.metrics:
            if not key.startswith("fi.") and key not in seen:
                metric_keys.append(key)
                seen.add(key)
    metric_keys = metric_keys[:4]

    col_w = max(12, *(len(k) for k in metric_keys), 0) if metric_keys else 12
    header = f"{'RUN ID':<10}  {'NAME':<20}  {'STATUS':<10}  {'STARTED':<12}"
    for k in metric_keys:
        header += f"  {k:>{col_w}}"
    typer.echo(f"\nExperiment: {exp_name}\n")
    typer.echo(header)
    typer.echo("-" * len(header))

    for run in runs:
        run_id = run.info.run_id[:8]
        name = (run.info.run_name or "")[:20]
        run_status = (run.info.status or "")[:10]
        started = _time_ago(run.info.start_time) if run.info.start_time else "-"
        row = f"{run_id:<10}  {name:<20}  {run_status:<10}  {started:<12}"
        for k in metric_keys:
            row += f"  {_fmt_metric(run.data.metrics.get(k)):>{col_w}}"
        typer.echo(row)

    typer.echo()


@experiments_app.command("compare")
def experiments_compare(
    metric: str = typer.Argument(..., help="Metric to rank by"),
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help="Experiment name")
    ] = None,
    params_file: Annotated[
        str, typer.Option("--params", help="params.yaml to read experiment from")
    ] = "params.yaml",
    lower_is_better: Annotated[bool, typer.Option("--lower-is-better/--higher-is-better")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max runs to show")] = 20,
) -> None:
    """Rank runs by a metric."""
    import mlflow.tracking

    exp_name = _resolve_experiment(experiment, params_file)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        typer.echo(f"Experiment {exp_name!r} not found.", err=True)
        raise typer.Exit(1)

    order = "ASC" if lower_is_better else "DESC"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"metrics.{metric} > -99999",
        order_by=[f"metrics.{metric} {order}"],
        max_results=limit,
    )
    if not runs:
        typer.echo(f"No runs with metric {metric!r} found in {exp_name!r}.")
        return

    direction = "lower=better" if lower_is_better else "higher=better"
    typer.echo(f"\nExperiment: {exp_name}  |  {metric} ({direction})\n")
    typer.echo(f"{'#':<4}  {'RUN ID':<10}  {'NAME':<20}  {'VARIANT':<12}  {metric}")
    typer.echo("-" * 65)

    for i, run in enumerate(runs):
        rank = "★" if i == 0 else str(i + 1)
        run_id = run.info.run_id[:8]
        name = (run.info.run_name or "")[:20]
        variant = run.data.tags.get("model_variant", "")[:12]
        val = _fmt_metric(run.data.metrics.get(metric))
        typer.echo(f"{rank:<4}  {run_id:<10}  {name:<20}  {variant:<12}  {val}")

    typer.echo()


# ---------------------------------------------------------------------------
# Leaderboard command
# ---------------------------------------------------------------------------


def _autodetect_metric(
    params_file: str,
    client: object,
    experiment_id: str,
) -> tuple[str, bool]:
    """Return (metric_name, higher_is_better).

    Priority:
    1. First key in params.yaml thresholds — direction inferred from spec type.
    2. First val_* key logged in the most recent run — assumed higher-is-better.
    3. Hard fallback: "val_accuracy", higher-is-better.
    """
    p = Path(params_file)
    if p.exists():
        try:
            from kitchen.config import KitchenConfig, ThresholdSpec

            cfg = KitchenConfig.from_yaml(str(p))
            if cfg.thresholds:
                name, spec = next(iter(cfg.thresholds.items()))
                if isinstance(spec, ThresholdSpec):
                    higher = not (spec.max is not None and spec.min is None)
                else:
                    higher = True  # plain float = lower bound = higher-is-better
                return name, higher
        except Exception:
            pass

    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=5,
            order_by=["start_time DESC"],
        )
        for run in runs:
            for key in sorted(run.data.metrics):
                if key.startswith("val_"):
                    return key, True
    except Exception:
        pass

    return "val_accuracy", True


