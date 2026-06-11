from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

_DEBUG_OPTION = typer.Option(
    "--debug", help="Show the full traceback on failure (or set KITCHEN_DEBUG=1)"
)


def _debug_enabled(flag: bool) -> bool:
    """Whether to surface full tracebacks — via the --debug flag or KITCHEN_DEBUG env."""
    return flag or os.environ.get("KITCHEN_DEBUG", "").strip().lower() in ("1", "true", "yes")


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
        reg_version = register_model(
            new_run.info.run_id, cfg.mlflow.model_artifact_path, resolved_model
        )
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


def _set_nested(d: dict, dotted_key: str, value: object) -> None:
    """Set ``d[a][b][c] = value`` from a dotted key, creating missing dicts."""
    parts = dotted_key.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _reproduced_params_file(params_file: str, run_id: str) -> str:
    """Write a temp params.yaml that reproduces *run_id*'s logged params (K-020).

    The run's MLflow params (flattened, dotted keys) are layered onto the current
    params.yaml so non-logged structure (lists, etc.) is preserved while every
    logged value is restored. Returns the temp file path; the caller deletes it.
    """
    import tempfile

    import yaml

    from kitchen.tracking import configure_from_env

    configure_from_env()
    import mlflow

    try:
        run = mlflow.get_run(run_id)
    except Exception as exc:  # noqa: BLE001 — surface any MLflow error as a clean CLI message
        typer.echo(f"error: could not load run {run_id!r} from MLflow: {exc}", err=True)
        raise typer.Exit(1)

    logged = dict(run.data.params)
    if not logged:
        typer.echo(f"error: run {run_id!r} has no logged params to reproduce from", err=True)
        raise typer.Exit(1)

    with open(params_file, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    for dotted, raw in logged.items():
        _set_nested(base, dotted, _coerce_override_value(raw))

    tmp = tempfile.NamedTemporaryFile(
        "w", prefix="kitchen-from-run-", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(base, tmp, sort_keys=False)
    tmp.close()
    return tmp.name


run_app = typer.Typer(help="Run pipeline stages.", no_args_is_help=True)




@run_app.command("features")
def run_features(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    debug: Annotated[bool, _DEBUG_OPTION] = False,
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
        if _debug_enabled(debug):
            raise
        typer.echo(f"error: {exc}\n  (re-run with --debug or KITCHEN_DEBUG=1 for the traceback)", err=True)
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
    from_run: Annotated[
        str | None,
        typer.Option(
            "--from-run",
            help="Reproduce a past run: load its logged params from MLflow as the base "
            "(any --override still applies on top), then re-train.",
        ),
    ] = None,
    debug: Annotated[bool, _DEBUG_OPTION] = False,
) -> None:
    """Run the full train pipeline: features → train → log to MLflow.

    With --auto-promote, compares the new run against the current champion on
    --promote-metric and promotes automatically if it wins. When --promote-metric
    is omitted, the metric is auto-detected from the first key in params.yaml
    thresholds (direction inferred from spec type).

    With --override, one or more params.yaml values are shadowed for this run
    only without modifying the file. Overrides are logged as MLflow run tags
    (override.<key>) so they appear in kitchen leaderboard and kitchen diff.

    With --from-run <run_id>, the target run's logged params are loaded from
    MLflow and used as the base params (layered onto the current params.yaml so
    non-logged structure is preserved); any --override still applies on top. The
    new run is tagged kitchen.from_run=<run_id> for provenance.
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

    # Reproduce a past run by training from its logged params (original
    # params_file is kept for auto-promote threshold lookup).
    train_params_file = params_file
    reproduced_file: str | None = None
    if from_run:
        reproduced_file = _reproduced_params_file(params_file, from_run)
        train_params_file = reproduced_file
        typer.echo(f"reproducing run {from_run} — params loaded from MLflow")

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        from kitchen.flows.train_flow import train_pipeline
    except ImportError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    try:
        run_id = train_pipeline(params_file=train_params_file, overrides=parsed_overrides)
    except ModuleNotFoundError as exc:
        typer.echo(
            f"error: {exc}\nRun from the project root and make sure src/ is implemented.",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as exc:
        if _debug_enabled(debug):
            raise
        typer.echo(
            f"error: training failed: {exc}\n  (re-run with --debug or KITCHEN_DEBUG=1 for the traceback)",
            err=True,
        )
        raise typer.Exit(1)
    finally:
        if reproduced_file:
            Path(reproduced_file).unlink(missing_ok=True)

    if from_run and run_id:
        from mlflow.tracking import MlflowClient

        MlflowClient().set_tag(run_id, "kitchen.from_run", from_run)

    if auto_promote:
        _try_auto_promote(params_file, promote_metric, lower_is_better, promote_model_name)  # type: ignore[arg-type]


def run_sweep(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    override: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            help="Sweep a params.yaml value over multiple values: key=v1,v2,... (repeatable). "
            "Multiple keys form a Cartesian product. Example: "
            "--override model.max_depth=4,6,8 --override model.eta=0.05,0.1",
        ),
    ] = None,
    metric: Annotated[
        str | None,
        typer.Option(
            "--metric",
            help="Metric to rank runs by. Auto-detected from params.yaml thresholds when omitted.",
        ),
    ] = None,
    lower_is_better: Annotated[
        bool,
        typer.Option("--lower-is-better/--higher-is-better", help="Metric direction for ranking"),
    ] = False,
    max_combos: Annotated[
        int,
        typer.Option("--max-combos", help="Refuse to launch a sweep larger than this"),
    ] = 50,
) -> None:
    """Run a hyperparameter sweep: train one run per param combination, rank, report.

    Expands each ``--override key=v1,v2,...`` into a value list and runs the full
    Cartesian product through the normal `kitchen run train` pipeline — one MLflow
    run per combination, each tagged with ``override.<key>`` (like a single
    ``--override`` run) plus a shared ``sweep.group`` id. After all runs complete,
    the runs are ranked by *metric* and the best run is printed with a
    copy-pasteable `kitchen promote --run-id` command.

    Error handling: a failure on the *first* combination is treated as a setup
    problem (e.g. ``src/`` not implemented) and aborts the sweep; a failure on a
    later combination is reported and the sweep continues with the rest.
    """
    import itertools
    import sys
    import uuid

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    if not override:
        typer.echo(
            "error: --override is required — e.g. --override model.max_depth=4,6,8", err=True
        )
        raise typer.Exit(1)

    # Parse each --override key=v1,v2,... into {key: [coerced values]}.
    grid: dict[str, list] = {}
    for item in override:
        if "=" not in item:
            typer.echo(f"error: --override {item!r} must be in key=value format", err=True)
            raise typer.Exit(1)
        key, _, raw_val = item.partition("=")
        values = [_coerce_override_value(v.strip()) for v in raw_val.split(",") if v.strip()]
        if not values:
            typer.echo(f"error: --override {item!r} has no values", err=True)
            raise typer.Exit(1)
        grid[key.strip()] = values

    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    if len(combos) < 2:
        typer.echo(
            "error: a sweep needs more than one combination — pass multiple comma-separated "
            "values (or use `kitchen run train --override` for a single run).",
            err=True,
        )
        raise typer.Exit(1)
    if len(combos) > max_combos:
        typer.echo(
            f"error: sweep would launch {len(combos)} runs (limit {max_combos}); "
            "narrow the grid or raise --max-combos.",
            err=True,
        )
        raise typer.Exit(1)

    if metric is None:
        detected = _promote_metric_from_thresholds(params_file)
        if detected is None:
            typer.echo(
                "error: --metric is required when params.yaml has no 'thresholds:' section "
                "to auto-detect from.",
                err=True,
            )
            raise typer.Exit(1)
        metric, lower_is_better = detected

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        from kitchen.flows.train_flow import train_pipeline
    except ImportError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    import mlflow.tracking

    from kitchen.tracking import configure_from_env

    configure_from_env()
    client = mlflow.tracking.MlflowClient()
    group = uuid.uuid4().hex[:8]

    direction = "lower=better" if lower_is_better else "higher=better"
    typer.echo(
        f"\nSweep {group}: {len(combos)} runs over {', '.join(keys)}  |  "
        f"metric={metric} ({direction})\n"
    )

    results: list[tuple[dict, str | None, float | None]] = []
    for i, combo in enumerate(combos):
        combo_str = ", ".join(f"{k}={v}" for k, v in combo.items())
        typer.echo(f"[{i + 1}/{len(combos)}] {combo_str}")
        try:
            run_id = train_pipeline(params_file=params_file, overrides=combo)
        except Exception as exc:
            if i == 0:
                # First combo failing is a setup problem affecting all combos.
                typer.echo(f"error: first sweep run failed — aborting: {exc}", err=True)
                raise typer.Exit(1)
            typer.echo(f"  ! run failed, skipping: {exc}", err=True)
            results.append((combo, None, None))
            continue

        score: float | None = None
        if run_id:
            try:
                client.set_tag(run_id, "sweep.group", group)
                score = client.get_run(run_id).data.metrics.get(metric)
            except Exception:
                pass
        results.append((combo, run_id, score))

    scored = [(c, r, s) for c, r, s in results if s is not None]
    if not scored:
        typer.echo(f"\nNo runs logged metric {metric!r} — nothing to rank.", err=True)
        raise typer.Exit(1)

    scored.sort(key=lambda row: row[2], reverse=not lower_is_better)
    best_combo, best_run_id, best_score = scored[0]

    typer.echo(f"\nSweep results ({metric}, {direction}):")
    for combo, run_id, score in scored:
        marker = "★" if run_id == best_run_id else " "
        combo_str = ", ".join(f"{k}={v}" for k, v in combo.items())
        typer.echo(f"  {marker} {score:>12.6f}  {run_id[:8]}  {combo_str}")
    # Surface any runs that produced no metric rather than dropping them silently.
    for combo, run_id, score in results:
        if score is None:
            combo_str = ", ".join(f"{k}={v}" for k, v in combo.items())
            rid = run_id[:8] if run_id else "(no run)"
            typer.echo(f"    {'—':>12}  {rid}  {combo_str}")

    best_str = ", ".join(f"{k}={v}" for k, v in best_combo.items())
    typer.echo(f"\nBest: {best_run_id[:8]}  {metric}={best_score:.6f}  ({best_str})")
    typer.echo(f"Promote it with: kitchen promote --run-id {best_run_id}")


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
    debug: Annotated[bool, _DEBUG_OPTION] = False,
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
        if not is_missing_alias and _debug_enabled(debug):
            raise
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
        if _debug_enabled(debug):
            raise
        typer.echo(
            f"error during evaluation: {exc}\n  (re-run with --debug or KITCHEN_DEBUG=1 for the traceback)",
            err=True,
        )
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
    """Run drift monitoring and generate an HTML drift report."""
    import sys

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    from kitchen.flows.monitor_flow import DriftThresholdExceeded, monitor_pipeline

    try:
        result = monitor_pipeline(params_file=params_file, local_path_override=local)
        if result:
            typer.echo(f"Report saved to: {result}")
    except DriftThresholdExceeded as exc:
        # The report was still written; surface where it is, then fail the run.
        if exc.report_path:
            typer.echo(f"Report saved to: {exc.report_path}")
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

