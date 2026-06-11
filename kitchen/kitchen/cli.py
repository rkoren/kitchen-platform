"""kitchen CLI — scaffold, validate, and manage competition projects.

Usage:
    kitchen init <name>                          # scaffold a new project
    kitchen check                                # pre-flight env/credential check
    kitchen run features                         # raw data → processed features (standalone)
    kitchen run train                            # features → train → log to MLflow
    kitchen run train --override model.max_depth=6  # one-off param override (repeatable)
    kitchen run train --auto-promote \
        --promote-metric <m> [--lower-is-better] # train + auto-promote if new run wins
    kitchen run evaluate                         # evaluate champion model
    kitchen run monitor [--local report.html]    # generate drift report
    kitchen status                               # project summary: champion + recent runs
    kitchen leaderboard                          # rank runs; [C]=champion ★=metric leader
    kitchen leaderboard --show-params model.eta,model.max_depth  # add param columns
    kitchen leaderboard --expand-metrics         # show per-fold metrics as sub-columns
    kitchen diff <run_id_a> <run_id_b>           # show param, metric, and feature importance deltas
    kitchen promote METRIC                       # manually promote best run
    kitchen promote --run-id <run_id>            # promote a specific run (e.g. from dashboard)
    kitchen ui                                   # open MLflow UI in browser
    kitchen experiments list                     # list recent runs
    kitchen experiments compare METRIC           # rank runs by a metric
    kitchen submit                               # submit to Kaggle
    kitchen report                               # markdown metrics summary
    kitchen serve local                          # start FastAPI serving app locally
    kitchen dashboard generate                   # re-render dashboard/index.html from results branch
    kitchen dashboard generate --serve           # generate + start local HTTP server + open browser
    kitchen dashboard generate --show-params model.eta,model.max_depth  # add param columns
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,redefined-outer-name
# (structural limits and fixture-name shadowing are suppressed via .pylintrc; these three
# remain at function granularity because they're the most targeted suppressions here)

from __future__ import annotations

import re
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import typer

from kitchen._cli._templates import (
    _BASELINE_PY,
    _CHALLENGER_PY,
    _CI_WORKFLOW,
    _CI_WORKFLOW_KAGGLE,
    _CLAUDE_MD,
    _DASHBOARD_HTML,
    _DVC_YAML,
    _DVC_YAML_KAGGLE,
    _DVCIGNORE,
    _ENV_EXAMPLE,
    _EVALUATE_RUN,
    _EVALUATE_RUN_BINARY_CLS,
    _EVALUATE_RUN_MULTICLASS_CLS,
    _EVALUATE_RUN_REGRESSION,
    _EVALUATE_RUN_TABULAR_TS,
    _FEATURES_RUN,
    _GENERATE_SUBMISSION_PY,
    _GITIGNORE,
    _INFRA_YAML,
    _MODEL_SECTION_GENERIC,
    _MODEL_SECTION_LGBM,
    _MODEL_SECTION_LR,
    _MODEL_SECTION_RF,
    _MODEL_SECTION_TS,
    _MODEL_SECTION_XGB,
    _PARAMS_YAML,
    _PARAMS_YAML_KAGGLE,
    _PREDICTOR_PY,
    _PROMOTE_PY,
    _PYPROJECT_TOML,
    _TEST_FEATURES,
    _TRAIN_FLOW_PY,
    _TRAIN_RUN,
    _TRAIN_RUN_BINARY_CLS,
    _TRAIN_RUN_LGBM,
    _TRAIN_RUN_LR,
    _TRAIN_RUN_MULTICLASS_CLS,
    _TRAIN_RUN_REGRESSION,
    _TRAIN_RUN_RF,
    _TRAIN_RUN_TABULAR_TS,
    _TRAIN_RUN_XGB,
    _build_exploration_notebook,
)
from kitchen._cli.dvc import (
    _render,
    _run_dvc_init,
    _to_class_name,
    _write,
    dvc_app,
)
from kitchen._cli.experiments import _autodetect_metric, experiments_app
from kitchen._cli.run import _coerce_override_value, run_app, run_sweep  # noqa: F401
from kitchen._cli.serve import _serve_local_dashboard, dashboard_app, serve_app

# Load .env from the project root (CWD) so MLFLOW_TRACKING_URI and other
# credentials are available to all commands without the user needing to
# `source .env` first.  Variables already set in the environment take
# precedence (override=False is the default).
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set externally

app = typer.Typer(
    help="kitchen ML platform CLI",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command()
def version() -> None:
    """Print the kitchen version."""
    typer.echo(f"kitchen {_pkg_version('kitchen')}")


@app.command()
def ui(
    port: Annotated[int, typer.Option("--port", "-p", help="Port for the local MLflow UI")] = 5000,
) -> None:
    """Open the MLflow tracking UI in your browser.

    For a remote tracking URI (http/https), opens the URL directly.
    For a local SQLite URI, starts `mlflow ui` and opens localhost.
    """
    import os
    import subprocess
    import threading
    import webbrowser

    from kitchen.tracking import configure_from_env

    configure_from_env()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")

    if tracking_uri.startswith(("http://", "https://")):
        typer.echo(f"Opening {tracking_uri}")
        webbrowser.open(tracking_uri)
        return

    url = f"http://localhost:{port}"
    typer.echo(f"MLflow UI → {url}")
    typer.echo(f"Tracking  → {tracking_uri}")
    typer.echo("Press Ctrl+C to stop.\n")

    def _open_after_delay() -> None:
        import time

        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=_open_after_delay, daemon=True).start()

    try:
        subprocess.run(
            ["mlflow", "ui", "--backend-store-uri", tracking_uri, "--port", str(port)],
            check=False,
        )
    except KeyboardInterrupt:
        typer.echo("\nStopped.")


@app.command(name="open")
def open_dashboard(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
) -> None:
    """Open the GitHub Pages dashboard in your browser.

    Reads dashboard_url from params.yaml, then falls back to the DASHBOARD_URL
    environment variable. If neither is set, opens the MLflow UI instead.
    """
    import os
    import webbrowser

    import yaml

    url: str | None = None
    params_path = Path(params_file)
    if params_path.exists():
        raw = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
        url = raw.get("dashboard_url")

    if not url:
        url = os.environ.get("DASHBOARD_URL")

    if url:
        typer.echo(f"Opening dashboard → {url}")
        webbrowser.open(url)
    else:
        local_dash = Path("dashboard/index.html")
        if local_dash.exists():
            _serve_local_dashboard(local_dash)
        else:
            typer.echo(
                "No dashboard_url found in params.yaml or DASHBOARD_URL env var. "
                "Falling back to MLflow UI."
            )
            ui()


@app.command()
def status(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help="Experiment name")
    ] = None,
    model_name: Annotated[
        str | None, typer.Option("--model-name", help="Registered model name")
    ] = None,
    n_runs: Annotated[
        int, typer.Option("--runs", "-n", help="Number of recent runs to show")
    ] = 5,
) -> None:
    """One-screen project summary: champion, recent runs with thresholds, and submission file.

    Always exits 0 — informational only, even when thresholds are violated.
    """
    import os

    import mlflow.tracking

    from kitchen.tracking import configure_from_env

    configure_from_env()

    cfg = None
    thresholds: dict = {}
    exp_name: str | None = experiment
    params_path = Path(params_file)
    if params_path.exists():
        try:
            from kitchen.config import KitchenConfig

            cfg = KitchenConfig.from_yaml(str(params_path))
            if exp_name is None:
                exp_name = cfg.experiment
            thresholds = cfg.thresholds or {}
        except Exception:
            pass

    if exp_name is None:
        typer.echo(
            "error: no experiment found — pass --experiment or run from a project directory.",
            err=True,
        )
        raise typer.Exit(1)

    resolved_model = model_name or os.environ.get("MLFLOW_MODEL_NAME", f"{exp_name}-model")
    typer.echo(f"\nProject: {exp_name}  ({params_file})\n")

    # Data section (LML-009): flag when processed features lag the raw inputs.
    data_state = _data_status(params_path.parent if params_path.exists() else Path.cwd())
    if data_state is not None:
        state, hint = data_state
        marker = {"missing": "○", "stale": "!", "fresh": "✓"}[state]
        typer.echo("Data")
        typer.echo(f"  processed : {marker} {state.upper()}  ({hint})")
        typer.echo()

    client = mlflow.tracking.MlflowClient()

    # Champion section
    champion_run_id: str | None = None
    typer.echo("Champion")
    try:
        mv = client.get_model_version_by_alias(resolved_model, "champion")
        champion_run_id = mv.run_id
        champ_run = client.get_run(champion_run_id)
        typer.echo(f"  model   : {resolved_model} @ champion  (v{mv.version})")
        typer.echo(
            f"  run     : {champion_run_id[:8]}  ({_time_ago(champ_run.info.start_time)})"
        )
        variant = champ_run.data.tags.get("model_variant", "")
        if variant:
            typer.echo(f"  variant : {variant}")
        for line in _summarize_champion_metrics(champ_run.data.metrics):
            typer.echo(line)
    except Exception:
        typer.echo(f"  (no champion registered for {resolved_model!r})")
        typer.echo("  Run `kitchen promote METRIC` to register the best run.")

    typer.echo()

    # Recent Runs section
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        typer.echo(f"No experiment {exp_name!r} found — no runs to show.\n")
        return

    recent = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=n_runs,
    )

    # Pick the primary display metric: first threshold key, then priority list, then first non-fi
    display_metric: str | None = None
    if thresholds:
        display_metric = sorted(thresholds.keys())[0]
    if display_metric is None:
        for candidate in ("loto_brier", "val_accuracy", "val_brier", "val_log_loss", "val_auc"):
            if any(candidate in r.data.metrics for r in recent):
                display_metric = candidate
                break
    if display_metric is None:
        for run in recent:
            for k in run.data.metrics:
                if not k.startswith("fi."):
                    display_metric = k
                    break
            if display_metric:
                break

    has_thresholds = bool(thresholds)
    metric_label = display_metric or "—"
    metric_w = max(12, len(metric_label))
    typer.echo(f"Recent Runs (last {n_runs})  —  {metric_label}")
    header = f"  {'#':<4}  {'RUN ID':<10}  {'VARIANT':<12}  {metric_label:>{metric_w}}"
    if has_thresholds:
        header += "  STATUS"
    typer.echo(header)
    typer.echo("  " + "-" * (len(header) - 2))

    if not recent:
        typer.echo("  No runs found.")
    else:
        for i, run in enumerate(recent):
            run_id_short = run.info.run_id[:8]
            is_champ = run.info.run_id == champion_run_id
            rank = "[C]" if is_champ else str(i + 1)
            variant = (run.data.tags.get("model_variant") or run.info.run_name or "")[:12]
            val = _fmt_metric(
                run.data.metrics.get(display_metric) if display_metric else None
            )
            row = f"  {rank:<4}  {run_id_short:<10}  {variant:<12}  {val:>{metric_w}}"
            if has_thresholds:
                fails: list[str] = []
                for tname, spec in thresholds.items():
                    actual = run.data.metrics.get(tname)
                    if actual is None:
                        continue
                    if isinstance(spec, (int, float)):
                        if actual < spec:
                            fails.append(f"{tname}<{spec:.4f}")
                    else:
                        if spec.min is not None and actual < spec.min:
                            fails.append(f"{tname}<{spec.min:.4f}")
                        if spec.max is not None and actual > spec.max:
                            fails.append(f"{tname}>{spec.max:.4f}")
                row += f"  {'FAIL' if fails else 'PASS'}"
                if fails:
                    row += f"  ({', '.join(fails)})"
            typer.echo(row)

    typer.echo()

    if has_thresholds:
        typer.echo("Thresholds:")
        for tname, spec in sorted(thresholds.items()):
            if isinstance(spec, (int, float)):
                typer.echo(f"  {tname}: >= {spec:.6f}")
            else:
                parts = []
                if spec.min is not None:
                    parts.append(f">= {spec.min:.6f}")
                if spec.max is not None:
                    parts.append(f"<= {spec.max:.6f}")
                typer.echo(f"  {tname}: {' and '.join(parts)}")
        typer.echo()

    # Local Submission File section
    sub_path = Path("submissions/submission.csv")
    if sub_path.exists():
        age_str = _time_ago(int(sub_path.stat().st_mtime * 1000))
        size_kb = sub_path.stat().st_size / 1024
        typer.echo(
            f"Local Submission File: {sub_path}  ({size_kb:.0f} KB, modified {age_str})"
        )
        typer.echo()


@app.command()
def validate(
    params_file: Annotated[str, typer.Argument(help="Path to params.yaml")] = "params.yaml",
) -> None:
    """Validate a params.yaml file against the KitchenConfig schema."""
    from pydantic import ValidationError

    from kitchen.config import KitchenConfig

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    try:
        cfg = KitchenConfig.from_yaml(str(path))
    except ValidationError as exc:
        typer.echo(f"validation failed: {params_file}", err=True)
        for error in exc.errors():
            loc = ".".join(str(p) for p in error["loc"])
            typer.echo(f"  {loc}: {error['msg']}", err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"error reading {params_file}: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"✓ {params_file}")
    typer.echo(f"  experiment : {cfg.experiment}")
    typer.echo(f"  mlflow     : {cfg.mlflow.tracking_uri}")
    if cfg.data:
        typer.echo(f"  data       : source={cfg.data.source}")
    if cfg.monitor:
        output = cfg.monitor.report_bucket or cfg.monitor.local_path
        typer.echo(f"  monitor    : output={output}")
    if cfg.ci:
        bits = [f"auto_submit={cfg.ci.auto_submit}", f"fail_on_threshold={cfg.ci.fail_on_threshold}"]
        if cfg.ci.notifications:
            bits.append(f"notify_when={cfg.ci.notifications.when}")
        typer.echo(f"  ci         : {', '.join(bits)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_to_git_branch(content: str, file_path: str, branch: str, commit_msg: str) -> str:
    """Write content to file_path on branch using git plumbing. Returns commit SHA.

    Never touches the working tree or index — safe to call from any checkout state.
    Uses a temporary index file isolated via GIT_INDEX_FILE so it doesn't disturb
    the caller's staged changes.
    """
    import os
    import subprocess
    import tempfile

    git_empty_tree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

    blob_sha = subprocess.check_output(
        ["git", "hash-object", "-w", "--stdin"], input=content.encode()
    ).decode().strip()

    idx_fd, idx_path = tempfile.mkstemp(prefix="kitchen-push-")
    os.close(idx_fd)
    try:
        env = {**os.environ, "GIT_INDEX_FILE": idx_path}
        branch_ref = f"refs/heads/{branch}"
        branch_exists = (
            subprocess.run(
                ["git", "rev-parse", "--verify", branch_ref], capture_output=True, check=False
            ).returncode == 0
        )
        if branch_exists:
            subprocess.run(["git", "read-tree", branch], env=env, check=True, capture_output=True)
        else:
            subprocess.run(
                ["git", "read-tree", git_empty_tree], env=env, check=True, capture_output=True
            )
        subprocess.run(
            ["git", "update-index", "--add", "--cacheinfo", f"100644,{blob_sha},{file_path}"],
            env=env,
            check=True,
        )
        tree_sha = subprocess.check_output(["git", "write-tree"], env=env).decode().strip()
        commit_cmd = ["git", "commit-tree", tree_sha, "-m", commit_msg]
        if branch_exists:
            parent_sha = subprocess.check_output(["git", "rev-parse", branch]).decode().strip()
            commit_cmd += ["-p", parent_sha]
        commit_sha = subprocess.check_output(commit_cmd).decode().strip()
        subprocess.run(["git", "update-ref", branch_ref, commit_sha], check=True)
        return commit_sha
    finally:
        os.unlink(idx_path)


_SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def _validate_name(name: str) -> str | None:
    """Return an error message if name is not a valid Kaggle-style slug, else None."""
    if not _SLUG_RE.match(name):
        return (
            "name must be a lowercase slug: letters, digits, and hyphens only, "
            "starting with a letter (e.g. spaceship-titanic)"
        )
    return None


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


def _resolve_model_artifact_path(params_file: str) -> str:
    """The name the project logs its model under (mlflow.model_artifact_path), default 'model'."""
    from kitchen.config import KitchenConfig

    p = Path(params_file)
    if p.exists():
        try:
            return KitchenConfig.from_yaml(str(p)).mlflow.model_artifact_path
        except Exception:
            pass
    return "model"


def _load_hint(model_uri: str) -> str:
    """Return the ``load_model`` expression matching the model's logged flavor.

    ``pyfunc`` is preferred when a ``python_function`` flavor exists (it is
    flavor-agnostic), but composite / sklearn-only models that log no pyfunc
    flavor must be loaded with their framework loader. Inspect the registered
    version's flavors (same mechanism as ``kitchen run evaluate``) so the printed
    hint is copy-paste-correct instead of a hardcoded ``mlflow.pyfunc`` that fails
    with ``Model does not have the "python_function" flavor``. Falls back to the
    pyfunc hint if the flavors can't be read.
    """
    loaders = {
        "python_function": "mlflow.pyfunc",
        "xgboost": "mlflow.xgboost",
        "lightgbm": "mlflow.lightgbm",
        "sklearn": "mlflow.sklearn",
    }
    try:
        import mlflow

        flavors = mlflow.models.get_model_info(model_uri).flavors
        for key, module in loaders.items():
            if key in flavors:
                return f"{module}.load_model('{model_uri}')"
    except Exception:
        pass
    return f"mlflow.pyfunc.load_model('{model_uri}')"


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


def _data_status(root: Path) -> tuple[str, str] | None:
    """LML-009: is ``data/processed/`` current with ``data/raw/`` + the feature script?

    Returns ``(state, hint)`` where state is ``"missing"`` / ``"stale"`` / ``"fresh"``,
    or ``None`` when there is no ``data/`` directory (nothing meaningful to report).
    Compares the newest mtime under ``data/processed/`` against every raw input and
    ``src/features/run.py`` — the inputs the features stage reads — rather than a single
    hardcoded output filename, so multi-output feature steps are handled correctly.
    The hint is DVC-aware: it points at ``dvc repro`` when ``dvc.yaml`` is present and
    ``kitchen run features`` otherwise.
    """
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    if not (root / "data").is_dir():
        return None

    uses_dvc = (root / "dvc.yaml").exists()
    refresh = "dvc repro" if uses_dvc else "kitchen run features"

    def _files(d: Path) -> list[Path]:
        return [p for p in d.iterdir() if p.is_file()] if d.is_dir() else []

    processed = _files(processed_dir)
    if not processed:
        return ("missing", f"not built — run `{refresh}`")

    deps = _files(raw_dir)
    feature_script = root / "src" / "features" / "run.py"
    if feature_script.exists():
        deps.append(feature_script)

    newest_processed = max(p.stat().st_mtime for p in processed)
    stale = [d for d in deps if d.stat().st_mtime > newest_processed]
    if stale:
        changed = ", ".join(sorted(p.name for p in stale)[:3])
        more = "" if len(stale) <= 3 else f" (+{len(stale) - 3} more)"
        return ("stale", f"inputs changed ({changed}{more}) — re-run `{refresh}`")
    return ("fresh", "up to date with data/raw/")


def _summarize_champion_metrics(metrics: dict[str, float], max_scalars: int = 12) -> list[str]:
    """LML-009/CBB-005: compact metric lines for the `kitchen status` champion block.

    Collapses per-fold/period families ({base}_{numeric} keys logged by time_series_cv /
    loto_cv, e.g. brier_2003..brier_2025) into one summary line so a model with many
    per-season metrics doesn't flood the one-screen summary; drops fi.* importances; and
    caps the remaining scalar lines with a "+N more" pointer to the leaderboard.
    """
    scored = {k: v for k, v in metrics.items() if not k.startswith("fi.")}

    # Group {base}_{numeric-suffix} keys; a base with ≥3 numeric siblings is a family.
    families: dict[str, list[float]] = {}
    scalars: dict[str, float] = {}
    for k, v in scored.items():
        base, _, suffix = k.rpartition("_")
        if base and suffix.isdigit():
            families.setdefault(base, []).append(v)
        else:
            scalars[k] = v
    families = {b: vals for b, vals in families.items() if len(vals) >= 3}
    # Any base that didn't reach the family threshold stays an individual scalar.
    for k, v in scored.items():
        base, _, suffix = k.rpartition("_")
        if base and suffix.isdigit() and base not in families:
            scalars[k] = v

    lines: list[str] = []
    for name in sorted(scalars)[:max_scalars]:
        lines.append(f"  {name:<14}: {scalars[name]:.6f}")
    hidden = len(scalars) - min(len(scalars), max_scalars)
    for base in sorted(families):
        vals = families[base]
        lines.append(
            f"  {base + '_*':<14}: {len(vals)} values  "
            f"(min {min(vals):.4f}, mean {sum(vals) / len(vals):.4f}, max {max(vals):.4f})"
        )
    if hidden:
        lines.append(f"  (+{hidden} more metrics — see `kitchen leaderboard` / `kitchen ui`)")
    return lines


def _fmt_delta_row(
    pr_val: object, base_val: object
) -> tuple[str, str, str]:
    """Return (pr_str, base_str, delta_str) for a comparison metric row."""
    pr_str = f"{pr_val:.6f}" if isinstance(pr_val, float) else str(pr_val) if pr_val is not None else "(new)"
    base_str = f"{base_val:.6f}" if isinstance(base_val, float) else str(base_val) if base_val is not None else "(new)"
    if isinstance(pr_val, (int, float)) and isinstance(base_val, (int, float)):
        delta = pr_val - base_val  # type: ignore[operator]
        delta_str = (
            f"{float(delta):+.6f}"
            if isinstance(pr_val, float) or isinstance(base_val, float)
            else f"{delta:+d}"
        )
    else:
        delta_str = "—"
    return pr_str, base_str, delta_str


app.add_typer(experiments_app, name="experiments")
@app.command()
def leaderboard(
    metric: Annotated[
        str | None, typer.Option("--metric", "-m", help="Primary metric to rank by")
    ] = None,
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help="Experiment name")
    ] = None,
    params_file: Annotated[
        str, typer.Option("--params", help="params.yaml to read experiment from")
    ] = "params.yaml",
    higher_is_better: Annotated[
        bool, typer.Option("--higher-is-better", help="Rank highest first (default: lowest first)")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max runs to show")] = 20,
    model_name: Annotated[
        str | None,
        typer.Option("--model-name", help="Registered model name to resolve champion alias"),
    ] = None,
    show_params: Annotated[
        str | None,
        typer.Option(
            "--show-params",
            help="Comma-separated param paths to show as extra columns (e.g. model.eta,model.max_depth)",
        ),
    ] = None,
    expand_metrics: Annotated[
        bool,
        typer.Option(
            "--expand-metrics/--no-expand-metrics",
            help="Show per-fold metric sub-columns for any {metric}_{fold} keys logged by time_series_cv or loto_cv",
        ),
    ] = False,
    exclude_exploratory: Annotated[
        bool,
        typer.Option(
            "--exclude-exploratory",
            help="Hide runs tagged run_type=exploratory (notebook sketches from kitchen.experiment)",
        ),
    ] = False,
    only_exploratory: Annotated[
        bool,
        typer.Option(
            "--only-exploratory",
            help="Show only runs tagged run_type=exploratory",
        ),
    ] = False,
) -> None:
    """Rank runs by a metric; shows full run_id and lb_score for easy replay.

    When --metric is omitted, the primary metric is auto-detected: first from
    the thresholds section in params.yaml (direction inferred from spec type),
    then from the first val_* key logged in recent runs.

    [C] marks the promoted champion from the model registry. ★ marks the
    top-ranked run by metric (they may differ if a newer run hasn't been promoted yet).

    --exclude-exploratory / --only-exploratory filter on the run_type tag set by
    kitchen.experiment(exploratory=True), so notebook sketches can be isolated
    from or suppressed in the ranked view.
    """
    if exclude_exploratory and only_exploratory:
        typer.echo(
            "error: --exclude-exploratory and --only-exploratory are mutually exclusive.",
            err=True,
        )
        raise typer.Exit(1)
    import os

    import mlflow.tracking

    from kitchen.tracking import configure_from_env

    configure_from_env()
    exp_name = _resolve_experiment(experiment, params_file)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        typer.echo(f"Experiment {exp_name!r} not found.", err=True)
        raise typer.Exit(1)

    if metric is None:
        metric, higher_is_better = _autodetect_metric(params_file, client, exp.experiment_id)

    order = "DESC" if higher_is_better else "ASC"
    # Over-fetch when an exploratory filter is active so post-filtering still
    # yields up to `limit` rows (MLflow's filter_string can't reliably express
    # "tag absent or != value", so run_type filtering is done in Python below).
    filtering = exclude_exploratory or only_exploratory
    fetch_n = max(limit * 5, 100) if filtering else limit
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"metrics.{metric} > -99999",
        order_by=[f"metrics.{metric} {order}"],
        max_results=fetch_n,
    )
    if not runs:
        # NB-006: the configured/auto-detected metric may not match what runs
        # actually logged — point the user at the val_* metrics that do exist.
        sample = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=25,
        )
        val_metrics = sorted({k for r in sample for k in r.data.metrics if k.startswith("val_")})
        msg = f"No runs with metric {metric!r} found in {exp_name!r}."
        if val_metrics:
            msg += (
                f"\n  Runs logged these val_* metrics instead: {', '.join(val_metrics)}."
                f"\n  Try: kitchen leaderboard --metric {val_metrics[0]}"
                f"\n  (or log under {metric!r} so runs show up in the default leaderboard)."
            )
        typer.echo(msg)
        return

    # NB-007: filter notebook sketches in/out by the run_type tag.
    if only_exploratory:
        runs = [r for r in runs if r.data.tags.get("run_type") == "exploratory"]
    elif exclude_exploratory:
        runs = [r for r in runs if r.data.tags.get("run_type") != "exploratory"]
    runs = runs[:limit]
    if not runs:
        which = "exploratory" if only_exploratory else "non-exploratory"
        typer.echo(f"No {which} runs with metric {metric!r} found in {exp_name!r}.")
        return

    # Resolve the champion run_id from the model registry (best-effort — no crash if absent).
    resolved_model_name = model_name or os.environ.get(
        "MLFLOW_MODEL_NAME", f"{exp_name}-model"
    )
    champion_run_id: str | None = None
    try:
        mv = client.get_model_version_by_alias(resolved_model_name, "champion")
        champion_run_id = mv.run_id
    except Exception:
        pass

    param_keys: list[str] = (
        [p.strip() for p in show_params.split(",") if p.strip()] if show_params else []
    )
    param_widths: list[int] = [
        max(len(key), max((len(r.data.params.get(key, "-")) for r in runs), default=0), 6)
        for key in param_keys
    ]

    # Discover per-fold keys: any {metric}_{suffix} key logged by time_series_cv / loto_cv,
    # excluding the aggregate _mean and _std keys.
    fold_suffixes: list[str] = []
    if expand_metrics:
        prefix = f"{metric}_"
        all_fold_suffixes: set[str] = set()
        for run in runs:
            for key in run.data.metrics:
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    if suffix not in ("mean", "std"):
                        all_fold_suffixes.add(suffix)
        fold_suffixes = sorted(all_fold_suffixes)
    fold_widths: list[int] = [max(len(s), 6) for s in fold_suffixes]

    direction = "higher=better" if higher_is_better else "lower=better"
    typer.echo(f"\nExperiment: {exp_name}  |  {metric} ({direction})\n")

    id_w = 32
    param_col_header = "".join(f"  {key:>{w}}" for key, w in zip(param_keys, param_widths))
    fold_col_header = "".join(f"  {s:>{w}}" for s, w in zip(fold_suffixes, fold_widths))
    header = f"{'#':<4}  {'RUN ID':<{id_w}}  {'VARIANT':<12}  {metric:>12}{fold_col_header}  {'lb_score':>10}{param_col_header}  STARTED"
    typer.echo(header)
    typer.echo("-" * len(header))

    for i, run in enumerate(runs):
        run_id = run.info.run_id
        is_champion = run_id == champion_run_id
        is_top = i == 0
        if is_champion and is_top:
            rank = "★[C]"
        elif is_champion:
            rank = "[C]"
        elif is_top:
            rank = "★"
        else:
            rank = str(i + 1)
        variant = (run.data.tags.get("model_variant") or run.info.run_name or "")[:12]
        primary = _fmt_metric(run.data.metrics.get(metric))
        lb = _fmt_metric(run.data.metrics.get("lb_score"))
        fold_col_vals = "".join(
            f"  {_fmt_metric(run.data.metrics.get(f'{metric}_{s}')):>{w}}"
            for s, w in zip(fold_suffixes, fold_widths)
        )
        param_col_vals = "".join(
            f"  {run.data.params.get(key, '-'):>{w}}" for key, w in zip(param_keys, param_widths)
        )
        started = _time_ago(run.info.start_time) if run.info.start_time else "-"
        typer.echo(f"{rank:<4}  {run_id:<{id_w}}  {variant:<12}  {primary:>12}{fold_col_vals}  {lb:>10}{param_col_vals}  {started}")

    typer.echo()
    if champion_run_id:
        typer.echo(f"[C] = current champion  (models:/{resolved_model_name}@champion)")


# ---------------------------------------------------------------------------
# Diff command (CMP-001, CMP-004)
# ---------------------------------------------------------------------------


def _diff_load_fi(run_id: str) -> dict[str, float] | None:
    """Download feature_importances.json for a run; returns None if absent.

    Only JSON is supported (M-007 only logs JSON; add CSV handling here if that changes).
    """
    import json
    import tempfile

    try:
        import mlflow.artifacts

        with tempfile.TemporaryDirectory() as tmp:
            fi_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="feature_importances.json",
                dst_path=tmp,
            )
            return json.loads(Path(fi_path).read_text(encoding="utf-8"))
    except Exception:
        return None


@app.command()
def diff(
    run_id_a: str = typer.Argument(..., help="First run ID (a)"),
    run_id_b: str = typer.Argument(..., help="Second run ID (b)"),
) -> None:
    """Show what changed between two MLflow runs.

    Prints a two-column table of params and metrics that differ; identical
    values are suppressed. Params are listed before metrics.
    """
    import mlflow.tracking

    from kitchen.tracking import configure_from_env

    configure_from_env()
    client = mlflow.tracking.MlflowClient()

    try:
        run_a = client.get_run(run_id_a)
    except Exception as exc:
        typer.echo(f"error: could not fetch run {run_id_a!r}: {exc}", err=True)
        raise typer.Exit(1)

    try:
        run_b = client.get_run(run_id_b)
    except Exception as exc:
        typer.echo(f"error: could not fetch run {run_id_b!r}: {exc}", err=True)
        raise typer.Exit(1)

    a_short = run_a.info.run_id[:8]
    b_short = run_b.info.run_id[:8]
    a_name = run_a.data.tags.get("mlflow.runName", "")
    b_name = run_b.data.tags.get("mlflow.runName", "")

    typer.echo("\nComparing:")
    typer.echo(f"  a  {a_short}  {a_name}".rstrip())
    typer.echo(f"  b  {b_short}  {b_name}".rstrip())

    # --- Param diffs ---
    params_a = run_a.data.params
    params_b = run_b.data.params
    param_rows: list[tuple[str, str, str]] = []
    for key in sorted(set(params_a) | set(params_b)):
        va = params_a.get(key, "(missing)")
        vb = params_b.get(key, "(missing)")
        if va != vb:
            param_rows.append((key, va, vb))

    # --- Metric diffs ---
    metrics_a = {k: v for k, v in run_a.data.metrics.items() if not k.startswith("fi.")}
    metrics_b = {k: v for k, v in run_b.data.metrics.items() if not k.startswith("fi.")}
    metric_rows: list[tuple[str, str, str]] = []
    for key in sorted(set(metrics_a) | set(metrics_b)):
        va_raw = metrics_a.get(key)
        vb_raw = metrics_b.get(key)
        if va_raw != vb_raw:
            metric_rows.append((key, _fmt_metric(va_raw), _fmt_metric(vb_raw)))

    # --- Feature importance rank diffs (CMP-004) ---
    _fi_a = _diff_load_fi(run_id_a)
    _fi_b = _diff_load_fi(run_id_b)
    fi_rows: list[tuple[int, int, int, str]] = []
    if _fi_a is not None and _fi_b is not None:
        _rank_a = {n: i + 1 for i, (n, _) in enumerate(sorted(_fi_a.items(), key=lambda x: (-x[1], x[0])))}
        _rank_b = {n: i + 1 for i, (n, _) in enumerate(sorted(_fi_b.items(), key=lambda x: (-x[1], x[0])))}
        fi_rows = sorted(
            [
                (abs(_rank_b[f] - _rank_a[f]), _rank_a[f], _rank_b[f], f)
                for f in set(_rank_a) & set(_rank_b)
                if _rank_a[f] != _rank_b[f]
            ],
            reverse=True,
        )[:5]

    if not param_rows and not metric_rows and not fi_rows:
        typer.echo("\nNo differences found.\n")
        return

    if param_rows or metric_rows:
        all_rows = param_rows + metric_rows
        key_w = max(len(r[0]) for r in all_rows)
        a_w = max(len(r[1]) for r in all_rows)
        header = f"\n  {'FIELD':<{key_w}}  {a_short:>{a_w}}  {b_short}"
        sep = "  " + "-" * (key_w + a_w + 4 + len(b_short))

        if param_rows:
            typer.echo(f"\nParams{header}")
            typer.echo(sep)
            for key, va, vb in param_rows:
                typer.echo(f"  {key:<{key_w}}  {va:>{a_w}}  {vb}")

        if metric_rows:
            typer.echo(f"\nMetrics{header}")
            typer.echo(sep)
            for key, va, vb in metric_rows:
                typer.echo(f"  {key:<{key_w}}  {va:>{a_w}}  {vb}")

    if fi_rows:
        fw = max(len(f) for _, _, _, f in fi_rows)
        fi_header = f"\n  {'FEATURE':<{fw}}  {'rank(a)':>7}  {'rank(b)':>7}  {'Δ':>5}"
        fi_sep = "  " + "-" * (fw + 26)
        typer.echo(f"\nFeature importance{fi_header}")
        typer.echo(fi_sep)
        for _, ra, rb, feat in fi_rows:
            delta = rb - ra
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            typer.echo(f"  {feat:<{fw}}  {ra:>7}  {rb:>7}  {delta_str:>5}")

    typer.echo()


# ---------------------------------------------------------------------------
# Ingest command
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    out_dir: Annotated[str | None, typer.Option("--out", help="Override output directory")] = None,
) -> None:
    """Download raw competition data as configured in params.yaml."""
    import os

    from kitchen.config import KitchenConfig
    from kitchen.ingest import source_from_params
    from kitchen.store import DataStore

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    try:
        cfg = KitchenConfig.from_yaml(str(path))
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    if cfg.data is None:
        typer.echo(
            "error: no 'data' section in params.yaml — add source, competition/bucket/path",
            err=True,
        )
        raise typer.Exit(1)

    if cfg.data.source == "kaggle":
        has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
        if not has_env and not has_json:
            typer.echo(
                "error: Kaggle credentials not found.\n"
                "  Create ~/.kaggle/kaggle.json  or  set KAGGLE_USERNAME + KAGGLE_KEY.",
                err=True,
            )
            raise typer.Exit(1)

    dest = Path(out_dir) if out_dir else DataStore().raw_dir

    try:
        source = source_from_params(cfg.data.model_dump())
        files = source.download(dest)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nIngested {len(files)} file(s) → {dest}")
    for f in files:
        typer.echo(f"  {f}")
    typer.echo()


# ---------------------------------------------------------------------------
# Submit command
# ---------------------------------------------------------------------------


def _write_kaggle_score(score: float, metrics_file: str = "metrics.json") -> None:
    import json

    path = Path(metrics_file)
    try:
        metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        metrics["kaggle_public_score"] = score
        path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


@app.command()
def submit(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    file: Annotated[
        str, typer.Option("--file", help="Submission CSV to upload")
    ] = "submissions/submission.csv",
    message: Annotated[str | None, typer.Option("--message", help="Submission message")] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait", help="Poll for leaderboard score after upload and write to metrics.json"
        ),
    ] = False,
) -> None:
    """Validate and upload a submission CSV to Kaggle."""
    import os

    import pandas as pd

    from kitchen.config import KitchenConfig
    from kitchen.store import DataStore
    from kitchen.submit import upload, validate_submission

    path = Path(params_file)
    if not path.exists():
        typer.echo(f"error: file not found: {params_file}", err=True)
        raise typer.Exit(1)

    try:
        cfg = KitchenConfig.from_yaml(str(path))
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    sub_cfg = cfg.submission
    id_col = sub_cfg.id_col if sub_cfg else "Id"
    target_col = sub_cfg.target_col if sub_cfg else "target"
    submit_msg = message or (sub_cfg.message if sub_cfg else "kitchen submit")
    sample_filename = sub_cfg.sample_submission if sub_cfg else "sample_submission.csv"

    # Resolve competition: submission.competition → data.competition → error
    competition = (sub_cfg.competition if sub_cfg else None) or (
        cfg.data.competition if cfg.data else None
    )
    if not competition:
        typer.echo(
            "error: no competition specified — add 'submission.competition' or 'data.competition' to params.yaml",
            err=True,
        )
        raise typer.Exit(1)

    # Kaggle credential check
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not has_env and not has_json:
        typer.echo(
            "error: Kaggle credentials not found.\n"
            "  Create ~/.kaggle/kaggle.json  or  set KAGGLE_USERNAME + KAGGLE_KEY.",
            err=True,
        )
        raise typer.Exit(1)

    sub_path = Path(file)
    if not sub_path.exists():
        typer.echo(f"error: submission file not found: {file}", err=True)
        raise typer.Exit(1)

    sample_path = DataStore().raw_dir / sample_filename
    if not sample_path.exists():
        typer.echo(f"error: sample submission not found: {sample_path}", err=True)
        raise typer.Exit(1)

    try:
        sub_df = pd.read_csv(sub_path)
        sample_df = pd.read_csv(sample_path)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    errors = validate_submission(sub_df, sample_df, id_col, target_col)
    if errors:
        typer.echo("Submission validation failed:", err=True)
        for e in errors:
            typer.echo(f"  • {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Validated {len(sub_df)} rows — uploading to '{competition}' …")
    try:
        upload(sub_path, submit_msg, competition)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Submitted {sub_path} → {competition}")

    if wait:
        from kitchen.submit import fetch_score

        typer.echo("Waiting for Kaggle to score submission…")
        score = fetch_score(competition)
        if score is not None:
            typer.echo(f"Leaderboard score: {score:.6f}")
            _write_kaggle_score(score)
            typer.echo("Score written to metrics.json")
        else:
            typer.echo("Score not yet available — check the Kaggle leaderboard.")



app.add_typer(run_app, name="run")
app.command(name="sweep")(run_sweep)

# ---------------------------------------------------------------------------
# Check command
# ---------------------------------------------------------------------------


@app.command()
def check(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
) -> None:
    """Check that all tools, credentials, and project files are ready."""
    import os
    import shutil
    import subprocess
    import sys

    issues = 0

    def _ok(label: str, detail: str = "") -> None:
        suffix = f"  {detail}" if detail else ""
        typer.echo(f"  ✓ {label:<26}{suffix}")

    def _fail(label: str, hint: str = "") -> None:
        nonlocal issues
        issues += 1
        suffix = f"  → {hint}" if hint else ""
        typer.echo(f"  ✗ {label:<26}{suffix}")

    def _warn(label: str, hint: str = "") -> None:
        suffix = f"  → {hint}" if hint else ""
        typer.echo(f"  ~ {label:<26}{suffix}")

    def _bin_version(name: str) -> str:
        try:
            out = subprocess.check_output([name, "--version"], stderr=subprocess.STDOUT, text=True)
            return out.strip().splitlines()[0]
        except Exception:
            return ""

    typer.echo()

    v = sys.version_info
    if v >= (3, 11):
        _ok("python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        _fail("python", f"found {v.major}.{v.minor} — requires >=3.11")

    for name, hint in [
        ("terraform", "needed for `recipes generate`"),
        ("docker", "needed for `kitchen serve`"),
    ]:
        if shutil.which(name):
            _ok(name, _bin_version(name))
        else:
            _fail(name, hint)

    # DVC: hard-fail only if this project uses it (dvc.yaml present); otherwise soft-warn.
    if shutil.which("dvc"):
        _ok("dvc", _bin_version("dvc"))
    elif Path("dvc.yaml").exists():
        _fail("dvc", "project uses DVC but binary not found — run `pip install kitchen[dvc]`")
    else:
        _warn("dvc", "not installed — run `pip install kitchen[dvc]` to enable data versioning")

    # DVC-012: warn when .dvc/config still has the scaffolded YOUR-BUCKET placeholder.
    _dvc_config = Path(".dvc/config")
    if _dvc_config.exists():
        try:
            if "YOUR-BUCKET" in _dvc_config.read_text(encoding="utf-8"):
                _warn(
                    "DVC remote",
                    "s3 remote not configured — run: dvc remote modify s3remote url s3://<bucket>/dvc",
                )
        except OSError:
            pass  # unreadable config is not a check failure

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        _ok("MLFLOW_TRACKING_URI", tracking_uri)
    else:
        _warn(
            "MLFLOW_TRACKING_URI",
            "not set — defaulting to sqlite:///mlruns.db (add to .env to silence)",
        )

    # Determine whether this project actually needs AWS credentials before checking.
    # Hard-fail only when data.source=s3 or mlflow.artifact_bucket is set; skip for
    # pure Kaggle+SQLite projects that have no AWS dependency.
    _needs_aws: bool | None = None  # None = unknown (no params.yaml)
    _params_path_early = Path(params_file)
    if _params_path_early.exists():
        try:
            import yaml as _yaml_aws

            _raw = _yaml_aws.safe_load(_params_path_early.read_text(encoding="utf-8")) or {}
            _data_cfg = _raw.get("data", {}) or {}
            _mlflow_cfg = _raw.get("mlflow", {}) or {}
            _needs_aws = _data_cfg.get("source") == "s3" or bool(_mlflow_cfg.get("artifact_bucket"))
        except Exception:
            pass  # parse failure → treat as unknown

    if _needs_aws is not False:
        # Check creds when project needs AWS (True) or project type is unknown (None).
        try:
            import boto3

            creds = boto3.Session().get_credentials()
            if creds is not None:
                creds.get_frozen_credentials()
                _ok("AWS credentials", "present")
            else:
                raise RuntimeError("no credentials found")
        except Exception:
            if _needs_aws:
                _fail(
                    "AWS credentials",
                    "run `aws configure` or set AWS_ACCESS_KEY_ID / AWS_PROFILE",
                )
            else:
                # Unknown project type — soft-warn, don't block.
                _warn(
                    "AWS credentials",
                    "not found — needed if data.source=s3 or mlflow.artifact_bucket is set",
                )

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if os.environ.get("KAGGLE_USERNAME") or kaggle_json.exists():
        _ok("Kaggle credentials", "present")
    else:
        _fail(
            "Kaggle credentials", "create ~/.kaggle/kaggle.json or set KAGGLE_USERNAME + KAGGLE_KEY"
        )

    params_path = Path(params_file)
    if params_path.exists():
        try:
            from pydantic import ValidationError

            from kitchen.config import KitchenConfig

            cfg = KitchenConfig.from_yaml(str(params_path))
            _ok(params_file, f"experiment={cfg.experiment!r}")
            if cfg.monitor:
                output = cfg.monitor.report_bucket or cfg.monitor.local_path
                _ok("monitor config", f"output={output}")
            # CBB-012: validate project-declared required secrets (e.g. KENPOM_API_KEY).
            # Satisfied by the process env or a local .env so it matches how the
            # project actually runs (dotenv-loaded), not just the current shell.
            if cfg.check and cfg.check.required_env:
                _env_file: dict[str, str | None] = {}
                try:
                    from dotenv import dotenv_values

                    if Path(".env").exists():
                        _env_file = dotenv_values(".env")
                except Exception:
                    pass  # python-dotenv absent or .env unreadable → env-only check
                for _var in cfg.check.required_env:
                    if os.environ.get(_var) or _env_file.get(_var):
                        _ok(f"env: {_var}", "present")
                    else:
                        _fail(f"env: {_var}", "required by check.required_env — set it or add to .env")
        except ValidationError:
            _fail(params_file, f"invalid — run `kitchen validate {params_file}`")
        except Exception as exc:
            _fail(params_file, str(exc))
    else:
        typer.echo(f"  - {params_file:<26}  not found (run from a project directory)")

    # --- Prep: project src modules ---
    _step_probes = [
        (Path("src/features/run.py"), "src.features.run", "FeatureBuilder", "build"),
        (Path("src/train/run.py"),    "src.train.run",    "Trainer",        "fit"),
        (Path("src/evaluate/run.py"), "src.evaluate.run", "Evaluator",      "evaluate"),
    ]
    src_candidates = [p for p, *_ in _step_probes]
    if any(p.exists() for p in src_candidates):
        import inspect as _inspect

        _check_cwd = str(Path.cwd())
        if _check_cwd not in sys.path:
            sys.path.insert(0, _check_cwd)

        try:
            from kitchen.steps import Evaluator as _Ev
            from kitchen.steps import FeatureBuilder as _FB
            from kitchen.steps import Trainer as _Tr
            _base_map = {"FeatureBuilder": _FB, "Trainer": _Tr, "Evaluator": _Ev}
        except Exception:
            _base_map = {}

        for _p, _mod_name, _base_name, _method_name in _step_probes:
            if not _p.exists():
                _fail(str(_p), "implement to run the pipeline")
                continue
            if not _base_map:
                _ok(str(_p))
                continue
            try:
                import importlib.util as _ilu_check

                _file_path = Path(_check_cwd) / _p
                _spec = _ilu_check.spec_from_file_location(_mod_name, str(_file_path))
                if _spec is None or _spec.loader is None:
                    raise ImportError(f"cannot load {_file_path}")
                _mod = _ilu_check.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
                _base_cls = _base_map[_base_name]
                _is_stub = False
                for _, _cls in _inspect.getmembers(_mod, _inspect.isclass):
                    if issubclass(_cls, _base_cls) and _cls is not _base_cls:
                        try:
                            _method_src = _inspect.getsource(getattr(_cls, _method_name))
                            if "NotImplementedError" in _method_src:
                                _is_stub = True
                        except (OSError, TypeError):
                            pass
                        break
                if _is_stub:
                    _warn(str(_p), f"{_method_name}() is a stub — fill in your implementation")
                else:
                    _ok(str(_p))
            except Exception:
                _ok(str(_p))  # can't probe (import failed) — treat as file-exists ✓

    # --- Project package importability ---
    # Read the package name from pyproject.toml so this works for any project,
    # not just cbb-model.  Covers the common case where `kitchen` runs in a
    # different Python env than the one where `pip install -e .` was last run
    # (e.g. pipx kitchen on Python 3.13 vs Homebrew kitchen on Python 3.14).
    _pyproject = Path("pyproject.toml")
    if _pyproject.exists() and any(p.exists() for p in src_candidates):
        _pkg_names: list[str] = []
        try:
            import importlib.util as _ilu

            if _ilu.find_spec("tomllib"):
                import tomllib as _toml
            else:
                import tomli as _toml  # type: ignore[no-reuse-declared]

            with open(_pyproject, "rb") as _fh:
                _pdata = _toml.load(_fh)
            # hatch: packages = ["src/cbb"] → importable name is "cbb"
            _wheel_pkgs = (
                _pdata.get("tool", {})
                .get("hatch", {})
                .get("build", {})
                .get("targets", {})
                .get("wheel", {})
                .get("packages", [])
            )
            for _wp in _wheel_pkgs:
                _pkg_names.append(Path(_wp).name)
            # setuptools / flit: [tool.setuptools.packages.find] root = "src"
            if not _pkg_names:
                _pkg_names = (
                    _pdata.get("tool", {})
                    .get("setuptools", {})
                    .get("packages", {})
                    .get("find", {})
                    .get("include", [])
                )
        except Exception:
            pass  # toml parse failure — skip importability check

        if _pkg_names:
            import importlib as _il

            _missing: list[str] = []
            for _pn in _pkg_names:
                try:
                    _il.import_module(_pn)
                except ModuleNotFoundError:
                    _missing.append(_pn)

            if _missing:
                # Detect whether kitchen itself was installed via pipx so we
                # can give the right fix command.
                _kitchen_bin = shutil.which("kitchen") or ""
                _via_pipx = ".local/pipx" in _kitchen_bin or "pipx" in _kitchen_bin
                if _via_pipx:
                    _fix = "pipx inject rkoren-kitchen . --force"
                else:
                    _fix = "pip install -e ."
                _fail(
                    "project package",
                    f"'{', '.join(_missing)}' not importable — run: {_fix}",
                )
            else:
                _ok("project package", f"'{', '.join(_pkg_names)}' importable ✓")

    # --- Summary ---
    typer.echo()
    if issues == 0:
        typer.echo("All checks passed — your kitchen is ready.")
    else:
        noun = "issue" if issues == 1 else "issues"
        typer.echo(f"{issues} {noun} found — see above.")
    typer.echo()

    if issues > 0:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Report command
# ---------------------------------------------------------------------------


@app.command()
def report(
    metrics_file: Annotated[
        str, typer.Option("--metrics", help="Path to metrics.json")
    ] = "metrics.json",
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    output_format: Annotated[
        str, typer.Option("--format", help="Output format: github, plain")
    ] = "github",
    compare: Annotated[
        str | None,
        typer.Option(
            "--compare",
            help=(
                "Baseline for delta comparison: a path to a base metrics.json, or the "
                "literal 'champion' to auto-fetch the registry champion's metrics (GH-011)."
            ),
        ),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option("--model-name", help="Registered model name for --compare champion lookup"),
    ] = None,
) -> None:
    """Write a metrics summary to stdout (pipe to $GITHUB_STEP_SUMMARY in CI)."""
    import json
    import os

    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        typer.echo(f"error: {metrics_file} not found — run `kitchen run evaluate` first", err=True)
        raise typer.Exit(1)

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"error: could not parse {metrics_file}: {exc}", err=True)
        raise typer.Exit(1)

    experiment = "unknown"
    cfg = None
    params_path = Path(params_file)
    if params_path.exists():
        try:
            from kitchen.config import KitchenConfig

            cfg = KitchenConfig.from_yaml(str(params_path))
            experiment = cfg.experiment
        except Exception:
            pass

    base_metrics: dict | None = None
    if compare == "champion":
        # GH-011: auto-fetch the registry champion's metrics as the baseline.
        # No champion (e.g. the first PR before any promote) is not an error —
        # warn and fall back to a plain single-column report so CI stays green.
        from kitchen.registry import get_champion_metrics
        from kitchen.tracking import configure_from_env

        configure_from_env()
        resolved_name = model_name or os.environ.get(
            "MLFLOW_MODEL_NAME", f"{experiment}-model"
        )
        base_metrics = get_champion_metrics(resolved_name)
        if base_metrics is None:
            typer.echo(
                f"warning: no champion registered for {resolved_name!r} — "
                "skipping comparison (run `kitchen promote` first).",
                err=True,
            )
    elif compare is not None:
        compare_path = Path(compare)
        if not compare_path.exists():
            typer.echo(f"error: compare file {compare} not found", err=True)
            raise typer.Exit(1)
        try:
            base_metrics = json.loads(compare_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            typer.echo(f"error: could not parse {compare}: {exc}", err=True)
            raise typer.Exit(1)
        base_metrics.pop("_run", None)

    run_meta = metrics.pop("_run", {}) if isinstance(metrics.get("_run"), dict) else {}
    run_name = run_meta.get("run_name") or run_meta.get("run_id", "")

    # Extract leaderboard score before the table loop so it renders in its own section.
    def _to_float(v: object) -> float | None:
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    kaggle_score: float | None = _to_float(metrics.pop("kaggle_public_score", None))
    base_kaggle_score: float | None = (
        _to_float(base_metrics.pop("kaggle_public_score", None))
        if base_metrics is not None
        else None
    )

    if output_format == "github":
        typer.echo(f"## Kitchen Report — `{experiment}`")
        if run_name:
            typer.echo(f"\n**Run:** `{run_name}`\n")
        else:
            typer.echo()
        if base_metrics is not None:
            typer.echo("| Metric | Base | PR | Delta |")
            typer.echo("| --- | --- | --- | --- |")
            for key in sorted(set(metrics) | set(base_metrics)):
                pr_str, base_str, delta_str = _fmt_delta_row(metrics.get(key), base_metrics.get(key))
                typer.echo(f"| `{key}` | {base_str} | {pr_str} | {delta_str} |")
        else:
            typer.echo("| Metric | Value |")
            typer.echo("| --- | --- |")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    typer.echo(f"| `{key}` | {value:.6f} |")
                else:
                    typer.echo(f"| `{key}` | {value} |")
    else:
        typer.echo(f"Experiment: {experiment}")
        if run_name:
            typer.echo(f"Run:        {run_name}")
        typer.echo()
        if base_metrics is not None:
            for key in sorted(set(metrics) | set(base_metrics)):
                pr_str, base_str, delta_str = _fmt_delta_row(metrics.get(key), base_metrics.get(key))
                typer.echo(f"  {key}: {pr_str} (base: {base_str}, delta: {delta_str})")
        else:
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    typer.echo(f"  {key}: {value:.6f}")
                else:
                    typer.echo(f"  {key}: {value}")

    if kaggle_score is not None:
        if output_format == "github":
            if base_kaggle_score is not None:
                delta = kaggle_score - base_kaggle_score
                typer.echo(
                    f"\n**Kaggle Public Leaderboard:** {kaggle_score:.6f}"
                    f" (base: {base_kaggle_score:.6f}, delta: {delta:+.6f})"
                )
            else:
                typer.echo(f"\n**Kaggle Public Leaderboard:** {kaggle_score:.6f}")
        else:
            if base_kaggle_score is not None:
                delta = kaggle_score - base_kaggle_score
                typer.echo(
                    f"Kaggle Public Leaderboard: {kaggle_score:.6f}"
                    f" (base: {base_kaggle_score:.6f}, delta: {delta:+.6f})"
                )
            else:
                typer.echo(f"Kaggle Public Leaderboard: {kaggle_score:.6f}")

    thresholds = cfg.thresholds if cfg is not None else {}
    if thresholds:
        failures: list[tuple[str, float | int, str]] = []
        for name in sorted(thresholds):
            if name not in metrics:
                continue
            actual = metrics[name]
            if not isinstance(actual, (int, float)):
                continue
            spec = thresholds[name]
            if isinstance(spec, (int, float)):
                if actual < spec:
                    bound = f"{spec:.6f}" if isinstance(spec, float) else str(spec)
                    failures.append((name, actual, f">= {bound}"))
            else:
                if spec.min is not None and actual < spec.min:
                    bound = f"{spec.min:.6f}"
                    failures.append((name, actual, f">= {bound}"))
                if spec.max is not None and actual > spec.max:
                    bound = f"{spec.max:.6f}"
                    failures.append((name, actual, f"<= {bound}"))
        if failures:
            if output_format == "github":
                typer.echo("\n### Threshold Violations\n")
                typer.echo("| Metric | Constraint | Actual |")
                typer.echo("| --- | --- | --- |")
                for name, actual, constraint in failures:
                    actual_str = f"{actual:.6f}" if isinstance(actual, float) else str(actual)
                    typer.echo(f"| `{name}` | {constraint} | {actual_str} |")
            else:
                typer.echo("\nThreshold violations:")
                for name, actual, constraint in failures:
                    actual_str = f"{actual:.6f}" if isinstance(actual, float) else str(actual)
                    typer.echo(f"  FAIL  {name}: {actual_str} {constraint}")
            # ci.fail_on_threshold (default true) gates whether a breach fails the job.
            if cfg is not None and cfg.ci is not None and not cfg.ci.fail_on_threshold:
                typer.echo(
                    "\nnote: ci.fail_on_threshold is false — reporting violations without failing.",
                    err=True,
                )
            else:
                raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Promote command
# ---------------------------------------------------------------------------


@app.command()
def promote(
    metric: Annotated[
        str | None, typer.Argument(help="Metric to rank runs by (omit when using --run-id)")
    ] = None,
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help="Experiment name")
    ] = None,
    params_file: Annotated[
        str, typer.Option("--params", help="params.yaml to read experiment from")
    ] = "params.yaml",
    model_name: Annotated[
        str | None, typer.Option("--model-name", help="Registered model name")
    ] = None,
    alias: Annotated[str, typer.Option("--alias", help="Model alias to set")] = "champion",
    lower_is_better: Annotated[bool, typer.Option("--lower-is-better/--higher-is-better")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show winner without registering")
    ] = False,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Promote a specific run by ID instead of ranking by metric"),
    ] = None,
    model_artifact_path: Annotated[
        str | None,
        typer.Option(
            "--model-artifact-path",
            help="Logged-model name to register (default: mlflow.model_artifact_path, or 'model')",
        ),
    ] = None,
) -> None:
    """Promote a run to the model registry.

    Pass METRIC to promote whichever run leads on that metric.
    Pass --run-id to promote a specific run directly (e.g. copied from the dashboard).
    Both METRIC and --run-id may be combined: --run-id targets the run, METRIC is shown for context.
    """
    import os

    import mlflow.tracking

    from kitchen.registry import get_best_run, get_production_uri, promote_model, register_model
    from kitchen.tracking import configure_from_env

    configure_from_env()

    if run_id is None and metric is None:
        typer.echo(
            "error: provide a METRIC to rank by, or --run-id to target a specific run.",
            err=True,
        )
        raise typer.Exit(1)

    exp_name = _resolve_experiment(experiment, params_file)

    if model_name is None:
        model_name = os.environ.get("MLFLOW_MODEL_NAME", f"{exp_name}-model")

    if run_id is not None:
        client = mlflow.tracking.MlflowClient()
        try:
            run = client.get_run(run_id)
        except Exception as exc:
            typer.echo(f"error: could not fetch run {run_id!r}: {exc}", err=True)
            raise typer.Exit(1)
    else:
        try:
            run = get_best_run(exp_name, metric, lower_is_better=lower_is_better)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1)

    actual_run_id = run.info.run_id
    variant = run.data.tags.get("model_variant", "")
    variant_str = f" ({variant})" if variant else ""

    typer.echo(f"\nExperiment : {exp_name}")
    if metric:
        score = run.data.metrics.get(metric, float("nan"))
        direction = "lower=better" if lower_is_better else "higher=better"
        label = "Run" if run_id else "Best run"
        typer.echo(f"{label:<11}: {actual_run_id[:8]}  {metric}={score:.6f}{variant_str}  ({direction})")
    else:
        run_name = run.data.tags.get("mlflow.runName", "")
        name_str = f"  {run_name}" if run_name else ""
        typer.echo(f"Run        : {actual_run_id[:8]}{name_str}{variant_str}")

    current = get_production_uri(model_name, alias)
    if current:
        typer.echo(f"Current    : {current}")

    if dry_run:
        typer.echo("\nDry run — skipping registration and promotion.")
        return

    resolved_artifact_path = model_artifact_path or _resolve_model_artifact_path(params_file)
    reg_version = register_model(actual_run_id, resolved_artifact_path, model_name)
    typer.echo(f"\nRegistered : {model_name} v{reg_version}")
    promote_model(model_name, reg_version, alias=alias)
    typer.echo(f"Promoted   : {model_name} v{reg_version} → {alias}")
    typer.echo(f"Load with  : {_load_hint(f'models:/{model_name}@{alias}')}")
    typer.echo()


# ---------------------------------------------------------------------------
# Push command
# ---------------------------------------------------------------------------


@app.command()
def push(
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml")
    ] = "params.yaml",
    metrics_file: Annotated[
        str, typer.Option("--metrics", help="Path to metrics.json")
    ] = "metrics.json",
    run_id_override: Annotated[
        str | None, typer.Option("--run-id", help="Override the MLflow run ID stored in metrics.json")
    ] = None,
    model_name: Annotated[
        str | None, typer.Option("--model-name", help="Registered model name for champion lookup")
    ] = None,
    branch: Annotated[
        str, typer.Option("--branch", help="Branch to write results to")
    ] = "results",
    push_to_remote: Annotated[
        bool, typer.Option("--push/--no-push", help="Push branch to remote after writing")
    ] = False,
    remote: Annotated[
        str, typer.Option("--remote", help="Git remote name")
    ] = "origin",
    message: Annotated[
        str | None, typer.Option("--message", "-m", help="Custom commit message")
    ] = None,
    top_features_n: Annotated[
        int, typer.Option("--top-features", help="Max feature importances to include (0 = disable).")
    ] = 20,
) -> None:
    """Publish current run metrics to the results branch as results/<sha>.json.

    Reads metrics.json and writes a snapshot to results/<git-sha>.json on the
    results branch using git plumbing — never touches the working tree or index.
    Optionally pushes to remote.
    """
    import json
    import os
    import subprocess
    import tempfile
    from datetime import datetime, timezone

    from kitchen.tracking import configure_from_env

    configure_from_env()

    # --- Resolve git SHA ---
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as exc:
        typer.echo(f"error: could not determine git HEAD SHA: {exc}", err=True)
        raise typer.Exit(1)

    # --- Load metrics ---
    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        typer.echo(f"error: {metrics_file!r} not found — run training first.", err=True)
        raise typer.Exit(1)

    try:
        metrics: dict = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.echo(f"error: could not read {metrics_file}: {exc}", err=True)
        raise typer.Exit(1)

    # --- Resolve experiment and model ---
    exp_name: str | None = None
    params_path = Path(params_file)
    if params_path.exists():
        try:
            from kitchen.config import KitchenConfig

            cfg = KitchenConfig.from_yaml(str(params_path))
            exp_name = cfg.experiment
            if model_name is None:
                model_name = os.environ.get("MLFLOW_MODEL_NAME", f"{exp_name}-model")
        except Exception:
            pass

    if model_name is None:
        model_name = os.environ.get("MLFLOW_MODEL_NAME", "")

    # --- Resolve run_id ---
    run_id: str | None = run_id_override or metrics.get("run_id")

    # --- Fetch MLflow metadata (champion flag, params, top features, calibration) ---
    # All fetches are best-effort: any failure silently leaves the field as None.
    is_champion = False
    params_from_run: dict | None = None
    top_features: list | None = None
    calibration_data: list | None = None
    run_metrics: dict = {}

    if run_id:
        try:
            import mlflow as _mlflow
            import mlflow.tracking as _mlflow_tracking

            _client = _mlflow_tracking.MlflowClient()

            # Champion flag
            if model_name:
                try:
                    _mv = _client.get_model_version_by_alias(model_name, "champion")
                    is_champion = _mv.run_id == run_id
                except Exception:
                    pass

            # Params + metrics from the MLflow run.
            try:
                _run = _client.get_run(run_id)
                _p = dict(_run.data.params)
                params_from_run = _p if _p else None
                # LML-010 follow-up: surface per-fold/aggregate metrics logged to the
                # run (e.g. by loto_cv / time_series_cv) into the results JSON so the
                # dashboard can render them (DASH-005). Exclude fi.* (feature-importance
                # metrics) and lb_score (carried as a top-level field).
                run_metrics = {
                    k: float(v)
                    for k, v in _run.data.metrics.items()
                    if not k.startswith("fi.") and k != "lb_score"
                }
            except Exception:
                pass

            # Top features from feature_importances.json artifact (logged by M-007)
            if top_features_n > 0:
                try:
                    with tempfile.TemporaryDirectory() as _tmp:
                        _fi_path = _mlflow.artifacts.download_artifacts(
                            run_id=run_id,
                            artifact_path="feature_importances.json",
                            dst_path=_tmp,
                        )
                        _fi_raw: dict = json.loads(
                            Path(_fi_path).read_text(encoding="utf-8")
                        )
                        _sorted_fi = sorted(
                            _fi_raw.items(), key=lambda x: x[1], reverse=True
                        )
                        top_features = [
                            {"name": k, "importance": v}
                            for k, v in _sorted_fi[:top_features_n]
                        ]
                except Exception:
                    pass

            # Calibration curves from calibration.json artifact (logged by DASH-006)
            try:
                with tempfile.TemporaryDirectory() as _tmp:
                    _cal_path = _mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path="calibration.json",
                        dst_path=_tmp,
                    )
                    calibration_data = json.loads(
                        Path(_cal_path).read_text(encoding="utf-8")
                    )
            except Exception:
                pass

        except ImportError:
            pass  # mlflow not installed — all metadata fields remain None

    # DASH-006: calibration.json disk fallback. In the CLI evaluate flow no MLflow
    # run is active, so Evaluator.run() writes the curve next to metrics.json rather
    # than as a run artifact — read it here when the MLflow lookup above came up empty.
    if calibration_data is None:
        _cal_sibling = metrics_path.parent / "calibration.json"
        if _cal_sibling.exists():
            try:
                calibration_data = json.loads(_cal_sibling.read_text(encoding="utf-8"))
            except Exception:
                pass

    # Fill in run metrics not already present in metrics.json (metrics.json wins).
    for _k, _v in run_metrics.items():
        metrics.setdefault(_k, _v)

    # --- lb_score ---
    lb_score: float | None = metrics.pop("kaggle_public_score", None)
    if isinstance(lb_score, str):
        try:
            lb_score = float(lb_score)
        except ValueError:
            lb_score = None

    # --- Build payload ---
    payload: dict = {
        "sha": git_sha,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id or "",
        "metrics": {k: v for k, v in metrics.items() if k != "run_id"},
        "params": params_from_run,
        "top_features": top_features,
        "calibration": calibration_data,
        "lb_score": lb_score,
        "champion": is_champion,
    }

    content = json.dumps(payload, indent=2) + "\n"
    dest_path = f"results/{git_sha[:8]}.json"
    commit_message = message or f"push: {git_sha[:8]} ({exp_name or 'unknown'})"

    try:
        commit_sha = _write_to_git_branch(content, dest_path, branch, commit_message)
    except Exception as exc:
        typer.echo(f"error: git write failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nPushed results to branch '{branch}'")
    typer.echo(f"  file   : {dest_path}")
    typer.echo(f"  commit : {commit_sha[:8]}")
    if is_champion:
        typer.echo("  status : champion")

    if push_to_remote:
        try:
            subprocess.run(
                ["git", "push", remote, f"refs/heads/{branch}:refs/heads/{branch}"],
                check=True,
            )
            typer.echo(f"  remote : pushed to {remote}/{branch}")
        except subprocess.CalledProcessError as exc:
            typer.echo(f"error: push to remote failed: {exc}", err=True)
            raise typer.Exit(1)

    typer.echo()


app.add_typer(dvc_app, name="dvc")
@app.command()
def init(
    name: str = typer.Argument(..., help="Project / competition name (e.g. spaceship-titanic)"),
    here: bool = typer.Option(False, "--here", help="Scaffold into cwd, not a new subdirectory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
    source: str = typer.Option("local", "--source", help="Data source: local, kaggle, s3"),
    competition: str | None = typer.Option(
        None, "--competition", help="Kaggle competition slug (required when --source kaggle)"
    ),
    template: str = typer.Option(
        "none",
        "--template",
        help="Starter template: none, baseline-xgb, baseline-lgbm, baseline-lr, baseline-rf, binary-cls, multiclass-cls, regression, tabular-ts",
    ),
    ci: bool = typer.Option(
        False, "--ci", help="Scaffold a .github/workflows/train-evaluate.yml CI workflow"
    ),
    with_dvc: bool = typer.Option(
        False, "--with-dvc", help="Scaffold dvc.yaml, .dvcignore, .dvc/config and run dvc init"
    ),
) -> None:
    """Scaffold a new kitchen competition project."""
    err = _validate_name(name)
    if err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(1)

    valid_sources = {"local", "kaggle", "s3"}
    if source not in valid_sources:
        typer.echo(
            f"error: invalid source {source!r} — choose from: {', '.join(sorted(valid_sources))}",
            err=True,
        )
        raise typer.Exit(1)

    if source == "kaggle" and not competition:
        typer.echo("error: --competition is required when --source kaggle", err=True)
        raise typer.Exit(1)

    valid_templates = {"none", "baseline-xgb", "baseline-lgbm", "baseline-lr", "baseline-rf", "binary-cls", "multiclass-cls", "regression", "tabular-ts"}
    if template not in valid_templates:
        typer.echo(
            f"error: invalid template {template!r} — choose from: {', '.join(sorted(valid_templates))}",
            err=True,
        )
        raise typer.Exit(1)

    if with_dvc:
        import shutil as _shutil

        if not _shutil.which("dvc"):
            typer.echo(
                "error: --with-dvc requires the dvc binary — run `pip install kitchen[dvc]` first",
                err=True,
            )
            raise typer.Exit(1)

    class_name = _to_class_name(name)
    root = Path.cwd() if here else Path.cwd() / name

    typer.echo(f"\nScaffolding '{name}' → {root}\n")

    r = _render  # shorthand

    params_tmpl = _PARAMS_YAML_KAGGLE if source == "kaggle" else _PARAMS_YAML
    params_extra = {"competition": competition} if source == "kaggle" else {}

    params_extra["model_section"] = {
        "baseline-xgb": _MODEL_SECTION_XGB,
        "binary-cls": _MODEL_SECTION_XGB,
        "multiclass-cls": _MODEL_SECTION_XGB,
        "regression": _MODEL_SECTION_XGB,
        "baseline-lgbm": _MODEL_SECTION_LGBM,
        "tabular-ts": _MODEL_SECTION_TS,
        "baseline-lr": _MODEL_SECTION_LR,
        "baseline-rf": _MODEL_SECTION_RF,
    }.get(template, _MODEL_SECTION_GENERIC)

    model_deps = {
        "baseline-xgb": '    "xgboost>=1.7",\n',
        "binary-cls": '    "xgboost>=1.7",\n',
        "multiclass-cls": '    "xgboost>=1.7",\n',
        "regression": '    "xgboost>=1.7",\n',
        "baseline-lgbm": '    "lightgbm>=4.0",\n',
        "tabular-ts": '    "lightgbm>=4.0",\n',
    }.get(template, "")

    train_tmpl = {
        "baseline-xgb": _TRAIN_RUN_XGB,
        "baseline-lgbm": _TRAIN_RUN_LGBM,
        "baseline-lr": _TRAIN_RUN_LR,
        "baseline-rf": _TRAIN_RUN_RF,
        "binary-cls": _TRAIN_RUN_BINARY_CLS,
        "multiclass-cls": _TRAIN_RUN_MULTICLASS_CLS,
        "regression": _TRAIN_RUN_REGRESSION,
        "tabular-ts": _TRAIN_RUN_TABULAR_TS,
    }.get(template, _TRAIN_RUN)

    eval_tmpl = {
        "baseline-xgb": _EVALUATE_RUN_BINARY_CLS,
        "baseline-lgbm": _EVALUATE_RUN_BINARY_CLS,
        "baseline-lr": _EVALUATE_RUN_BINARY_CLS,
        "baseline-rf": _EVALUATE_RUN_BINARY_CLS,
        "binary-cls": _EVALUATE_RUN_BINARY_CLS,
        "multiclass-cls": _EVALUATE_RUN_MULTICLASS_CLS,
        "regression": _EVALUATE_RUN_REGRESSION,
        "tabular-ts": _EVALUATE_RUN_TABULAR_TS,
    }.get(template, _EVALUATE_RUN)

    files: list[tuple[Path, str]] = [
        (root / "CLAUDE.md", r(_CLAUDE_MD, name, class_name)),
        (root / ".env.example", r(_ENV_EXAMPLE, name, class_name)),
        (root / ".gitignore", r(_GITIGNORE, name, class_name)),
        (root / "params.yaml", r(params_tmpl, name, class_name, **params_extra)),
        (root / "pyproject.toml", r(_PYPROJECT_TOML, name, class_name, model_deps=model_deps)),
        (root / "infra" / f"{name}.yaml", r(_INFRA_YAML, name, class_name)),
        (root / "src" / "__init__.py", ""),
        (root / "src" / "features" / "__init__.py", ""),
        (root / "src" / "features" / "run.py", r(_FEATURES_RUN, name, class_name)),
        (root / "src" / "train" / "__init__.py", ""),
        (root / "src" / "train" / "run.py", r(train_tmpl, name, class_name)),
        (root / "src" / "evaluate" / "__init__.py", ""),
        (root / "src" / "evaluate" / "run.py", r(eval_tmpl, name, class_name)),
        (root / "src" / "serve" / "__init__.py", ""),
        (root / "src" / "serve" / "predictor.py", r(_PREDICTOR_PY, name, class_name)),
        (root / "src" / "tests" / "__init__.py", ""),
        (root / "src" / "tests" / "test_features.py", r(_TEST_FEATURES, name, class_name)),
        (root / "experiments" / "__init__.py", ""),
        (root / "experiments" / "baseline.py", r(_BASELINE_PY, name, class_name)),
        (root / "experiments" / "challenger.py", r(_CHALLENGER_PY, name, class_name)),
        (root / "notebooks" / "exploration.ipynb", _build_exploration_notebook(name, class_name)),
        (root / "flows" / "train_flow.py", r(_TRAIN_FLOW_PY, name, class_name)),
        (root / "flows" / "promote.py", r(_PROMOTE_PY, name, class_name)),
        (root / "flows" / "generate_submission.py", r(_GENERATE_SUBMISSION_PY, name, class_name)),
        (root / "data" / "raw" / ".gitkeep", ""),
        (root / "data" / "processed" / ".gitkeep", ""),
        (root / "submissions" / ".gitkeep", ""),
        (root / "docs" / "index.html", r(_DASHBOARD_HTML, name, class_name)),
    ]

    if ci:
        ci_tmpl = _CI_WORKFLOW_KAGGLE if source == "kaggle" else _CI_WORKFLOW
        files.append(
            (root / ".github" / "workflows" / "train-evaluate.yml", r(ci_tmpl, name, class_name))
        )

    if with_dvc:
        dvc_tmpl = _DVC_YAML_KAGGLE if source == "kaggle" else _DVC_YAML
        files.append((root / "dvc.yaml", r(dvc_tmpl, name, class_name)))
        files.append((root / ".dvcignore", _DVCIGNORE))

    for path, content in files:
        _write(path, content, overwrite)

    if with_dvc:
        _run_dvc_init(root)

    cd_line = f"  cd {root.name}\n" if not here else ""
    if source == "kaggle":
        data_step = "  kitchen ingest                      # download competition data → data/raw/"
        submit_step = "  kitchen submit                      # validate and upload to Kaggle"
    else:
        data_step = "  # Download data to data/raw/"
        submit_step = "  python flows/generate_submission.py # generate submission CSV"

    ci_note = ""
    if ci:
        if source == "kaggle":
            ci_note = "\n  # CI: add KAGGLE_USERNAME and KAGGLE_KEY as GitHub Actions secrets"
        ci_note += "\n  # CI workflow scaffolded → .github/workflows/train-evaluate.yml"
        ci_note += "\n  # Dashboard: in repo Settings → Pages, set source to 'GitHub Actions'"

    dvc_note = ""
    if with_dvc:
        dvc_note = (
            "\n  dvc remote modify s3remote url s3://YOUR-BUCKET/dvc"
            "  # set your S3 remote"
            "\n  dvc push                            # upload data/processed/ + models/ to S3"
            "\n  # dvc pull                          # restore on a new machine"
            "\n  # dvc repro                         # run full pipeline (skips unchanged stages)"
        )

    typer.echo(f"""
Done. Next steps:

{cd_line}  pip install rkoren-kitchen -e .
  # Monorepo contributors: pip install -e ../kitchen-platform/kitchen -e .
  # If kitchen was installed via pipx: pipx inject rkoren-kitchen .
  cp .env.example .env
  kitchen check                       # verify tools, credentials, and config
                                      # (includes a check that your package is importable)
{data_step}
  # Implement src/features/run.py, src/train/run.py, src/evaluate/run.py
  kitchen run train                   # features → train → log to MLflow
  kitchen run evaluate                # load champion model, compute metrics
  kitchen leaderboard                 # rank runs by primary metric
  kitchen promote METRIC              # promote best run to the registry
{submit_step}{ci_note}{dvc_note}
""")


app.add_typer(serve_app, name="serve")
app.add_typer(dashboard_app, name="dashboard")


def main() -> None:
    """Console-script entry point: render known fatal errors cleanly (no traceback)."""
    from kitchen.tracking import MlflowSchemaError

    try:
        app()
    except MlflowSchemaError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
