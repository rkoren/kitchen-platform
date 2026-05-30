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
    kitchen status                               # one-screen project summary: champion + recent runs
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
import string
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import typer

# Load .env from the project root (CWD) so MLFLOW_TRACKING_URI and other
# credentials are available to all commands without the user needing to
# `source .env` first.  Variables already set in the environment take
# precedence (override=False is the default).
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set externally

app = typer.Typer(help="kitchen ML platform CLI", add_completion=False, no_args_is_help=True)


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
        for k, v in sorted(champ_run.data.metrics.items()):
            if not k.startswith("fi."):
                typer.echo(f"  {k:<14}: {v:.6f}")
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
            variant = run.data.tags.get("model_variant", "")[:12]
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


# ---------------------------------------------------------------------------
# Templates
# Each template uses $name (slug) and $class_name (PascalCase) as substitution vars.
# Literal $$ → $ in output.
# ---------------------------------------------------------------------------

_CLAUDE_MD = """\
# $name

Kaggle competition project on the [kitchen platform](../kitchen-platform/kitchen).

## Setup

```bash
pip install -e ../kitchen-platform/kitchen -e .
cp .env.example .env
# Download competition data to data/raw/
```

## The contract — 3 files to implement

| File | Class | Method |
|---|---|---|
| `src/features/run.py` | `${class_name}Features(FeatureBuilder)` | `build(raw_or_sources, params) -> df` |
| `src/train/run.py` | `${class_name}Trainer(Trainer)` | `fit(df, params) -> model` |
| `src/evaluate/run.py` | `${class_name}Evaluator(Evaluator)` | `evaluate(model, df) -> dict` |

All config lives in `params.yaml`. File paths resolve from `params["features"].*`;
model hyperparams from `params["model"].*`.

## Running experiments

```bash
# Train baseline (first approach)
python experiments/baseline.py

# Train challenger (improved approach — edit experiments/challenger.py first)
python experiments/challenger.py

# Compare runs and promote best model
python flows/promote.py --dry-run
python flows/promote.py

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db   # → http://localhost:5000

# Generate Kaggle submission
python flows/generate_submission.py
```

## Kitchen modules

- `kitchen.steps` — `FeatureBuilder`, `Trainer` (set `model_flavour`), `Evaluator` ABCs
- `kitchen.tracking` — `Tracker`, `configure_from_env()`, `init_experiment()`
- `kitchen.registry` — `get_best_run()`, `register_model()`, `promote_model()`
- `kitchen.evaluate` — `brier_score(y_true, y_prob)`, `log_loss(y_true, y_prob)`
- `kitchen.store` — `DataStore` (wraps `data/raw/`, `data/processed/`, `models/`)

## Experiment tagging

Both experiment scripts tag runs with `model_variant=baseline` or `model_variant=challenger`.
`flows/promote.py` compares across variants and promotes the winner to the `champion` alias.
Load the champion with `mlflow.sklearn.load_model('models:/$name-model@champion')`.
"""

_ENV_EXAMPLE = """\
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
export MLFLOW_EXPERIMENT=$name
export MLFLOW_MODEL_NAME=$name-model
export MLFLOW_PROMOTE_METRIC=val_accuracy
export MLFLOW_PROMOTE_LOWER_IS_BETTER=false
export AWS_PROFILE=default
"""

_GITIGNORE = """\
# secrets
.env

# data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# ml artifacts
mlruns/
mlruns.db
*.pkl
*.joblib
*.ubj

# python
__pycache__/
*.py[cod]
*.egg-info/
dist/
.venv/

# notebooks
.ipynb_checkpoints/

# outputs
metrics.json
submissions/

# infra (generated)
infra/tf/
"""

_PARAMS_YAML = """\
experiment: $name

data:
  source: local          # switch to "kaggle" once data is downloaded
  competition: $name
  raw_file: train.csv

features:
  raw_file: train.csv
  processed_file: features.parquet
  test_file: test.csv

model:
  target: label          # TODO: change to your actual target column name
  test_size: 0.2
  random_state: 42
  # Add model-specific hyperparams here, e.g.:
  # xgb:                       # --template baseline-xgb / binary-cls
  #   n_estimators: 300
  #   max_depth: 6
  #   learning_rate: 0.05
  # lgbm:                      # --template baseline-lgbm
  #   n_estimators: 300
  #   num_leaves: 31
  #   max_depth: -1
  #   learning_rate: 0.05
  # lr:                        # --template baseline-lr
  #   C: 1.0
  #   max_iter: 1000
  # rf:                        # --template baseline-rf
  #   n_estimators: 300
  #   max_depth: null
  #   min_samples_leaf: 1
  # tabular-ts:               # --template tabular-ts (chronological split, LightGBM regressor)
  #   date_col: date          # column to sort by; omit if data is already time-ordered
  #   val_frac: 0.2           # last val_frac of rows reserved for validation
  #   lgbm:
  #     n_estimators: 300
  #     num_leaves: 31
  #     max_depth: -1
  #     learning_rate: 0.05

mlflow:
  tracking_uri: sqlite:///mlruns.db

run_name: baseline
metrics_file: metrics.json

# dashboard_url: https://<owner>.github.io/<repo>/  # set to open via `kitchen open`

# thresholds:               # optional: fail CI if a metric violates its constraint
#   val_accuracy: 0.80      # plain float = lower bound (>= 0.80)
#   val_logloss:
#     max: 0.45             # upper bound for lower-is-better metrics (<= 0.45)
"""

_PYPROJECT_TOML = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$name"
version = "0.1.0"
description = "Kaggle $name — built on kitchen"
requires-python = ">=3.11"
dependencies = [
    "kitchen",           # pip install -e ../kitchen-platform/kitchen
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ipykernel>=6.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.pytest.ini_options]
testpaths = ["src/tests"]
"""

_PARAMS_YAML_KAGGLE = """\
experiment: $name

data:
  source: kaggle
  competition: $competition

submission:
  id_col: Id            # TODO: change to this competition's ID column
  target_col: target    # TODO: change to this competition's target column
  message: $name v1
  sample_submission: sample_submission.csv

features:
  raw_file: train.csv
  processed_file: features.parquet
  test_file: test.csv

model:
  target: target        # TODO: match submission.target_col
  test_size: 0.2
  random_state: 42
  # XGBoost — uncomment if using --template baseline-xgb or binary-cls:
  # xgb:
  #   n_estimators: 300
  #   max_depth: 6
  #   learning_rate: 0.05
  #   subsample: 0.8
  #   colsample_bytree: 0.8
  # LightGBM — uncomment if using --template baseline-lgbm:
  # lgbm:
  #   n_estimators: 300
  #   num_leaves: 31
  #   max_depth: -1
  #   learning_rate: 0.05
  #   subsample: 0.8
  #   colsample_bytree: 0.8
  # Logistic Regression — uncomment if using --template baseline-lr:
  # lr:
  #   C: 1.0
  #   max_iter: 1000
  # Random Forest — uncomment if using --template baseline-rf:
  # rf:
  #   n_estimators: 300
  #   max_depth: null
  #   min_samples_leaf: 1
  # Tabular time series — uncomment if using --template tabular-ts:
  # date_col: date          # column to sort by; omit if data is already time-ordered
  # val_frac: 0.2           # last val_frac of rows reserved for validation
  # lgbm:
  #   n_estimators: 300
  #   num_leaves: 31
  #   max_depth: -1
  #   learning_rate: 0.05
  #   subsample: 0.8
  #   colsample_bytree: 0.8

mlflow:
  tracking_uri: sqlite:///mlruns.db

run_name: baseline
metrics_file: metrics.json

# thresholds:               # optional: fail CI if a metric violates its constraint
#   val_accuracy: 0.80      # plain float = lower bound (>= 0.80)
#   val_logloss:
#     max: 0.45             # upper bound for lower-is-better metrics (<= 0.45)
"""

_INFRA_YAML = """\
name: $name
region: us-east-1
resources:
  - type: s3
    name: $name-data
    versioning: true

  - type: ecr
    name: $name-serve
    lambda_access: true

  - type: iam_role
    name: $name-lambda-role
    service: lambda.amazonaws.com
    policies:
      - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

  - type: lambda
    name: $name-serve
    role: $name-lambda-role
    ecr_repo: $name-serve
    memory: 1024
    timeout: 30
"""

_FEATURES_RUN = """\
\"\"\"Feature engineering for $name.

TODO:
  1. Implement ${class_name}Features.build() to transform raw CSV into model-ready features.
  2. Update FEATURES to list every column passed to the model (exclude the target).
  3. Keep the target column in the returned DataFrame — train.py separates it.
  4. If your project has multiple raw input files, override sources() to declare them:
       def sources(self, params: dict) -> list[str]:
           return ["train.csv", "other.csv"]
     build() will then receive a dict[filename, DataFrame] instead of a plain DataFrame.
\"\"\"
from __future__ import annotations

import pandas as pd
from kitchen.steps import FeatureBuilder
from kitchen.store import DataStore

# Columns passed to the model (exclude the target column).
FEATURES: list[str] = []  # TODO: fill in after feature engineering


class ${class_name}Features(FeatureBuilder):
    def build(self, raw: pd.DataFrame | dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
        \"\"\"Transform raw data into model-ready features + target column.

        ``raw`` is a plain DataFrame for single-source projects (the default).
        Override sources() and ``raw`` becomes a dict[filename, DataFrame].
        \"\"\"
        raise NotImplementedError


def build(params: dict, store: DataStore) -> None:
    ${class_name}Features().run(store, params)
"""

_TRAIN_RUN = """\
\"\"\"Model training for $name.\"\"\"
from __future__ import annotations

import pandas as pd
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class ${class_name}Trainer(Trainer):
    model_flavour = "sklearn"  # change to "xgboost" or "pyfunc" as needed

    def fit(self, df: pd.DataFrame, params: dict) -> object:
        \"\"\"Train and return a model. Log metrics to the active MLflow run.\"\"\"
        raise NotImplementedError


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_TRAIN_RUN_XGB = """\
\"\"\"Model training for $name — XGBoost baseline (binary classification).

Trains on a stratified train split, logs val_* metrics to the active MLflow
run, and returns the fitted model.  Swap XGBClassifier → XGBRegressor and
replace classification_metrics with regression_metrics for regression targets.
\"\"\"
from __future__ import annotations

import mlflow
import pandas as pd
import xgboost as xgb
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class ${class_name}Trainer(Trainer):
    model_flavour = "xgboost"

    def fit(self, df: pd.DataFrame, params: dict) -> xgb.XGBClassifier:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = params["model"].get("xgb", {})
        model = xgb.XGBClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=seed,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_TRAIN_RUN_LGBM = """\
\"\"\"Model training for $name — LightGBM baseline (binary classification).

Trains on a stratified train split, logs val_* metrics to the active MLflow
run, and returns the fitted model.  Swap LGBMClassifier → LGBMRegressor and
replace classification_metrics with regression_metrics for regression targets.
\"\"\"
from __future__ import annotations

import lightgbm as lgb
import mlflow
import pandas as pd
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class ${class_name}Trainer(Trainer):
    model_flavour = "lightgbm"

    def fit(self, df: pd.DataFrame, params: dict) -> lgb.LGBMClassifier:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = params["model"].get("lgbm", {})
        model = lgb.LGBMClassifier(
            n_estimators=p.get("n_estimators", 300),
            num_leaves=p.get("num_leaves", 31),
            max_depth=p.get("max_depth", -1),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_TRAIN_RUN_LR = """\
\"\"\"Model training for $name — Logistic Regression baseline (binary classification).

Trains on a stratified train split and logs val_* metrics to the active MLflow run.
\"\"\"
from __future__ import annotations

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class ${class_name}Trainer(Trainer):
    model_flavour = "sklearn"

    def fit(self, df: pd.DataFrame, params: dict) -> LogisticRegression:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = params["model"].get("lr", {})
        model = LogisticRegression(
            C=p.get("C", 1.0),
            max_iter=p.get("max_iter", 1000),
            random_state=seed,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_TRAIN_RUN_RF = """\
\"\"\"Model training for $name — Random Forest baseline (binary classification).

Trains on a stratified train split and logs val_* metrics to the active MLflow run.
Swap RandomForestClassifier → RandomForestRegressor and replace
classification_metrics with regression_metrics for regression targets.
\"\"\"
from __future__ import annotations

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class ${class_name}Trainer(Trainer):
    model_flavour = "sklearn"

    def fit(self, df: pd.DataFrame, params: dict) -> RandomForestClassifier:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = params["model"].get("rf", {})
        model = RandomForestClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_TRAIN_RUN_BINARY_CLS = """\
\"\"\"Model training for $name — binary classification (XGBoost baseline).

Splits features into train/val, fits XGBClassifier, then logs validation
metrics (val_accuracy, val_f1, val_log_loss, val_roc_auc) to the active
MLflow run.  The run is opened by Trainer.run() before fit() is called, so
mlflow.log_metrics() here is always inside a live run.

Swap XGBClassifier for any sklearn-compatible estimator.
\"\"\"
from __future__ import annotations

import mlflow
import pandas as pd
import xgboost as xgb
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class ${class_name}Trainer(Trainer):
    model_flavour = "xgboost"

    def fit(self, df: pd.DataFrame, params: dict) -> xgb.XGBClassifier:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = params["model"].get("xgb", {})
        model = xgb.XGBClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=seed,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]  # col 1 = P(class=1) for 0/1 labels
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_EVALUATE_RUN = """\
\"\"\"Evaluation for $name.\"\"\"
from __future__ import annotations

import pandas as pd
from kitchen.steps import Evaluator
from kitchen.store import DataStore


class ${class_name}Evaluator(Evaluator):
    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        \"\"\"Return metric_name -> value. Logged to MLflow and written to metrics.json.\"\"\"
        raise NotImplementedError


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return ${class_name}Evaluator().run(model, store, params)
"""

_EVALUATE_RUN_BINARY_CLS = """\
\"\"\"Evaluation for $name — binary classification.

Scores the model on a held-out validation split using the same seed as
training so the val partition is consistent across runs.
Reports accuracy, f1, log_loss, and roc_auc.

Note: predict_proba()[:, 1] assumes class labels are 0 and 1 (sklearn default
ordering). If your target uses other labels, inspect model.classes_ first.
\"\"\"
from __future__ import annotations

import pandas as pd
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Evaluator
from kitchen.store import DataStore


class ${class_name}Evaluator(Evaluator):
    \"\"\"Binary classification evaluator.

    Overrides run() to stash params as an instance attribute so that
    evaluate() can access the target column and random seed — the base class
    does not forward params to evaluate().
    \"\"\"

    def run(self, model: object, store: DataStore, params: dict) -> dict[str, float]:
        self._params = params  # stash so evaluate() can read target + seed
        return super().run(model, store, params)

    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        params = self._params
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        _, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in val_df.columns if c != target]
        X_val, y_val = val_df[features], val_df[target]

        y_pred = model.predict(X_val)
        y_proba = (
            model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        )
        return classification_metrics(y_val, y_pred, y_proba=y_proba)


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return ${class_name}Evaluator().run(model, store, params)
"""

_TRAIN_RUN_MULTICLASS_CLS = """\
\"\"\"Model training for $name — multiclass classification (XGBoost baseline).

Splits features into train/val, fits XGBClassifier with multi:softprob
objective, logs validation metrics (val_accuracy, val_f1, val_roc_auc) to the
active MLflow run.  The run is opened by Trainer.run() before fit() is called.

Set params.model.num_classes to the number of classes (XGBoost ≥1.6 can infer
it, but setting it explicitly is safer).
\"\"\"
from __future__ import annotations

import mlflow
import pandas as pd
import xgboost as xgb
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class ${class_name}Trainer(Trainer):
    model_flavour = "xgboost"

    def fit(self, df: pd.DataFrame, params: dict) -> xgb.XGBClassifier:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = params["model"].get("xgb", {})
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=seed,
            eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)  # full probability matrix for roc_auc (ovr)
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba, average="macro")
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_EVALUATE_RUN_MULTICLASS_CLS = """\
\"\"\"Evaluation for $name — multiclass classification.

Scores the model on a held-out validation split using the same seed as
training so the val partition is consistent across runs.
Reports accuracy, macro-f1, and macro roc_auc (one-vs-rest).
\"\"\"
from __future__ import annotations

import pandas as pd
from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Evaluator
from kitchen.store import DataStore


class ${class_name}Evaluator(Evaluator):
    \"\"\"Multiclass classification evaluator.

    Overrides run() to stash params as an instance attribute so that
    evaluate() can access the target column and random seed — the base class
    does not forward params to evaluate().
    \"\"\"

    def run(self, model: object, store: DataStore, params: dict) -> dict[str, float]:
        self._params = params  # stash so evaluate() can read target + seed
        return super().run(model, store, params)

    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        params = self._params
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        _, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in val_df.columns if c != target]
        X_val, y_val = val_df[features], val_df[target]

        y_pred = model.predict(X_val)
        y_proba = (
            model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        )
        return classification_metrics(y_val, y_pred, y_proba=y_proba, average="macro")


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return ${class_name}Evaluator().run(model, store, params)
"""

_TRAIN_RUN_REGRESSION = """\
\"\"\"Model training for $name — regression (XGBoost baseline).

Splits features into train/val, fits XGBRegressor, logs validation metrics
(val_rmse, val_mae, val_r2) to the active MLflow run.
The run is opened by Trainer.run() before fit() is called.

For lower-is-better metrics use `kitchen run train --lower-is-better`.
\"\"\"
from __future__ import annotations

import mlflow
import pandas as pd
import xgboost as xgb
from kitchen.modeling import regression_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class ${class_name}Trainer(Trainer):
    model_flavour = "xgboost"

    def fit(self, df: pd.DataFrame, params: dict) -> xgb.XGBRegressor:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        # stratify=False: regression targets are continuous, not class labels
        train_df, val_df = train_val_split(df, target_col=target, seed=seed, stratify=False)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = params["model"].get("xgb", {})
        model = xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=seed,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        val_metrics = regression_metrics(y_val, y_pred)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_EVALUATE_RUN_REGRESSION = """\
\"\"\"Evaluation for $name — regression.

Scores the model on a held-out validation split using the same seed as
training so the val partition is consistent across runs.
Reports rmse, mae, and r2.
\"\"\"
from __future__ import annotations

import pandas as pd
from kitchen.modeling import regression_metrics, train_val_split
from kitchen.steps import Evaluator
from kitchen.store import DataStore


class ${class_name}Evaluator(Evaluator):
    \"\"\"Regression evaluator.

    Overrides run() to stash params as an instance attribute so that
    evaluate() can access the target column and random seed — the base class
    does not forward params to evaluate().
    \"\"\"

    def run(self, model: object, store: DataStore, params: dict) -> dict[str, float]:
        self._params = params  # stash so evaluate() can read target + seed
        return super().run(model, store, params)

    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        params = self._params
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        # stratify=False: regression targets are continuous, not class labels
        _, val_df = train_val_split(df, target_col=target, seed=seed, stratify=False)
        features = [c for c in val_df.columns if c != target]
        X_val, y_val = val_df[features], val_df[target]

        y_pred = model.predict(X_val)
        return regression_metrics(y_val, y_pred)


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return ${class_name}Evaluator().run(model, store, params)
"""

_TRAIN_RUN_TABULAR_TS = """\
\"\"\"Model training for $name — tabular time series (LightGBM baseline).

Uses a **chronological** train/val split: the last *val_frac* fraction of rows
(sorted by *date_col* if provided) is held out as validation.  This avoids
the data leakage that occurs with a random split on time-ordered data.

Defaults to LGBMRegressor (forecasting / demand / energy competitions).
For classification: swap LGBMRegressor → LGBMClassifier and replace
regression_metrics with classification_metrics.

Params read from params.yaml under model:
  target       — target column name (required)
  date_col     — column to sort by before splitting (omit if data is pre-sorted)
  val_frac     — fraction of rows reserved for validation, default 0.2
  random_state — random seed, default 42
  lgbm:        — LightGBM hyperparams (see commented section in params.yaml)
\"\"\"
from __future__ import annotations

import mlflow
import lightgbm as lgb
import pandas as pd
from kitchen.modeling import regression_metrics
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


def _time_split(
    df: pd.DataFrame,
    date_col: str | None = None,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    \"\"\"Chronological train/val split — no shuffling.

    Sorts by *date_col* when provided, then holds out the last *val_frac*
    fraction of rows for validation.  Call with the same arguments in both
    train.py and evaluate.py so the partitions are consistent across runs.
    \"\"\"
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_frac))
    return df.iloc[:-n_val].copy(), df.iloc[-n_val:].copy()


class ${class_name}Trainer(Trainer):
    model_flavour = "lightgbm"

    def fit(self, df: pd.DataFrame, params: dict) -> lgb.LGBMRegressor:
        mp = params["model"]
        target = mp["target"]
        date_col = mp.get("date_col")
        val_frac = mp.get("val_frac", 0.2)

        train_df, val_df = _time_split(df, date_col=date_col, val_frac=val_frac)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = mp.get("lgbm", {})
        model = lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 300),
            num_leaves=p.get("num_leaves", 31),
            max_depth=p.get("max_depth", -1),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=mp.get("random_state", 42),
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        val_metrics = regression_metrics(y_val, y_pred)
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return ${class_name}Trainer().run(store, tracker, params)
"""

_EVALUATE_RUN_TABULAR_TS = """\
\"\"\"Evaluation for $name — tabular time series.

Scores the model on the chronological validation split using the same
date_col and val_frac as training, so the partition is consistent.
Reports rmse, mae, and r2.
\"\"\"
from __future__ import annotations

import pandas as pd
from kitchen.modeling import regression_metrics
from kitchen.steps import Evaluator
from kitchen.store import DataStore


def _time_split(
    df: pd.DataFrame,
    date_col: str | None = None,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    \"\"\"Chronological train/val split — no shuffling.\"\"\"
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_frac))
    return df.iloc[:-n_val].copy(), df.iloc[-n_val:].copy()


class ${class_name}Evaluator(Evaluator):
    \"\"\"Tabular time series evaluator.

    Overrides run() to stash params as an instance attribute so that
    evaluate() can access target, date_col, and val_frac — the base class
    does not forward params to evaluate().
    \"\"\"

    def run(self, model: object, store: DataStore, params: dict) -> dict[str, float]:
        self._params = params
        return super().run(model, store, params)

    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        mp = self._params["model"]
        target = mp["target"]
        date_col = mp.get("date_col")
        val_frac = mp.get("val_frac", 0.2)

        _, val_df = _time_split(df, date_col=date_col, val_frac=val_frac)
        features = [c for c in val_df.columns if c != target]
        X_val, y_val = val_df[features], val_df[target]

        y_pred = model.predict(X_val)
        return regression_metrics(y_val, y_pred)


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return ${class_name}Evaluator().run(model, store, params)
"""

_PREDICTOR_PY = """\
\"\"\"Predictor for $name — plug your trained model in here.

This module is loaded by ``kitchen serve local`` (and the Lambda handler) via
``kitchen.serve.loader``.  It must expose::

    def predict(payload: dict) -> dict: ...

Optionally export ``RequestModel`` and ``ResponseModel`` (Pydantic
``BaseModel`` subclasses) to enable typed OpenAPI docs on ``/predict``.
If either is absent the endpoint accepts and returns raw dicts.
\"\"\"
from __future__ import annotations

# ---------------------------------------------------------------------------
# Uncomment once your champion model is promoted to the registry:
# ---------------------------------------------------------------------------
# import mlflow
# model = mlflow.sklearn.load_model(\"models:/$name@champion\")

# ---------------------------------------------------------------------------
# Optional: typed OpenAPI schema (requires pydantic, already a FastAPI dep)
# ---------------------------------------------------------------------------
# from pydantic import BaseModel
#
# class RequestModel(BaseModel):
#     feature_a: float
#     feature_b: str
#
# class ResponseModel(BaseModel):
#     label: int
#     score: float

# ---------------------------------------------------------------------------
# Optional: feature list — surfaced on GET /metadata so callers know which
# input keys the model expects.
# ---------------------------------------------------------------------------
# FEATURES: list[str] = [\"feature_a\", \"feature_b\"]


def predict(payload: dict) -> dict:
    \"\"\"Return a prediction for *payload*.

    Args:
        payload: Arbitrary JSON dict from the caller.  When ``RequestModel``
                 is configured this will be ``RequestModel.model_dump()``.

    Returns:
        Prediction result (must be JSON-serialisable).  When ``ResponseModel``
        is configured FastAPI validates the return value against the schema.
    \"\"\"
    # TODO: replace with real model inference, e.g.:
    #   features = [payload[\"feature_a\"], payload[\"feature_b\"]]
    #   return {\"label\": int(model.predict([features])[0]), \"score\": 0.0}
    raise NotImplementedError(
        \"Implement predict() in src/serve/predictor.py — \"
        \"see the commented examples above.\"
    )
"""

_TEST_FEATURES = """\
\"\"\"Tests for $name feature engineering.\"\"\"
import pandas as pd
import pytest

from src.features.run import ${class_name}Features, FEATURES


@pytest.fixture
def raw_row() -> pd.DataFrame:
    # TODO: replace with a representative row from your raw training data
    return pd.DataFrame([{}])


def test_feature_builder_raises_not_implemented(raw_row):
    # build() raises NotImplementedError until you implement it.
    # Remove this test and add real assertions once build() is done.
    with pytest.raises(NotImplementedError):
        ${class_name}Features().build(raw_row, params={})


def test_features_list_is_defined():
    # Populate FEATURES once build() is implemented.
    assert isinstance(FEATURES, list)
"""

_BASELINE_PY = """\
\"\"\"Baseline experiment for $name.

First approach — simpler features, default hyperparams.
Tag: model_variant=baseline.

Usage:
    python experiments/baseline.py
\"\"\"
from __future__ import annotations

import os
import yaml
from dotenv import load_dotenv

load_dotenv()

import mlflow
from prefect import flow, task, get_run_logger

from kitchen.tracking import Tracker, configure_from_env, init_experiment
from kitchen.store import DataStore

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "$name")
VARIANT = "baseline"


@task
def run_variant(params: dict, variant: str) -> None:
    from src.features.run import build
    from src.train.run import train

    log = get_run_logger()
    configure_from_env()
    init_experiment(EXPERIMENT)

    store = DataStore()
    tracker = Tracker(EXPERIMENT)

    with tracker.run(run_name=variant, params=params) as _run:
        mlflow.set_tag("model_variant", variant)
        build(params, store)
        train(params, store, tracker)   # logs val_* metrics to the active run
        log.info("%s run complete — see MLflow for val metrics", variant)


@flow(name="$name-baseline")
def baseline(params_file: str = "params.yaml") -> None:
    with open(params_file) as f:
        params = yaml.safe_load(f)
    run_variant(params, VARIANT)


if __name__ == "__main__":
    baseline()
"""

_CHALLENGER_PY = """\
\"\"\"Challenger experiment for $name.

Extend the baseline: add features, tune hyperparams, or swap the model.
Tag: model_variant=challenger.

Usage:
    python experiments/challenger.py
\"\"\"
from __future__ import annotations

import yaml
from dotenv import load_dotenv

load_dotenv()

from prefect import flow

from experiments.baseline import run_variant

VARIANT = "challenger"


@flow(name="$name-challenger")
def challenger(params_file: str = "params.yaml") -> None:
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # TODO: Override params for the challenger approach, e.g.:
    # params["model"]["max_depth"] = 8
    # params["model"]["learning_rate"] = 0.01

    run_variant(params, VARIANT)


if __name__ == "__main__":
    challenger()
"""

_TRAIN_FLOW_PY = """\
\"\"\"Single-run training pipeline — delegates to kitchen's generic flow.

Use this for quick one-off training runs. For the full baseline/challenger
experiment loop, use experiments/baseline.py and experiments/challenger.py.
\"\"\"
from dotenv import load_dotenv

load_dotenv()

from kitchen.flows.train_flow import train_pipeline

if __name__ == "__main__":
    train_pipeline()
"""

_PROMOTE_PY = """\
\"\"\"Promote the best model to champion in the MLflow Model Registry.

Compares baseline vs challenger runs by metric, registers the winner,
and sets the 'champion' alias for serving.

Usage:
    python flows/promote.py              # compare both variants, promote best
    python flows/promote.py --variant challenger
    python flows/promote.py --dry-run    # print winner without promoting
    mlflow ui --backend-store-uri sqlite:///mlruns.db
\"\"\"
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from kitchen import tracking
from kitchen.registry import get_best_run, get_production_uri, promote_model, register_model

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "$name")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "$name-model")
DEFAULT_METRIC = os.environ.get("MLFLOW_PROMOTE_METRIC", "val_accuracy")
LOWER_IS_BETTER = os.environ.get("MLFLOW_PROMOTE_LOWER_IS_BETTER", "false").lower() == "true"


def show_comparison(experiment: str, metric: str) -> None:
    import mlflow.tracking
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        print(f"Experiment {experiment!r} not found.")
        return

    direction = "lower=better" if LOWER_IS_BETTER else "higher=better"
    print(f"\\nExperiment: {experiment}  |  {metric} ({direction})\\n")
    print(f"{'Variant':<15} {'Run ID':<12} {metric}")
    print("-" * 50)

    for variant in ("baseline", "challenger"):
        try:
            run = get_best_run(experiment, metric, lower_is_better=LOWER_IS_BETTER,
                               tag_filter={"model_variant": variant})
            val = run.data.metrics.get(metric, float("nan"))
            val_str = f"{val:.6f}" if val == val else "n/a"
            print(f"{variant:<15} {run.info.run_id[:8]:<12} {val_str}")
        except ValueError:
            print(f"{variant:<15} {'(no runs)'}")
    print()


def promote(
    metric: str = DEFAULT_METRIC,
    variant: str | None = None,
    model_name: str = MODEL_NAME,
    dry_run: bool = False,
) -> None:
    tracking.configure_from_env()
    show_comparison(EXPERIMENT, metric)

    if variant:
        run = get_best_run(EXPERIMENT, metric, lower_is_better=LOWER_IS_BETTER,
                           tag_filter={"model_variant": variant})
    else:
        candidates = []
        for v in ("baseline", "challenger"):
            try:
                candidates.append(
                    get_best_run(EXPERIMENT, metric, lower_is_better=LOWER_IS_BETTER,
                                 tag_filter={"model_variant": v})
                )
            except ValueError:
                pass
        if not candidates:
            raise ValueError("No baseline or challenger runs found in experiment")
        pick = min if LOWER_IS_BETTER else max
        run = pick(candidates, key=lambda r: r.data.metrics.get(metric, float("inf")))

    run_id = run.info.run_id
    score = run.data.metrics.get(metric, float("nan"))
    variant_tag = run.data.tags.get("model_variant", "unknown")
    print(f"Winner: {run_id} ({variant_tag})  {metric}={score:.6f}")

    current = get_production_uri(model_name)
    if current:
        print(f"Current champion: {current}")

    if dry_run:
        print("Dry run — skipping registration and promotion.")
        return

    version = register_model(run_id, "model", model_name)
    print(f"Registered {model_name} v{version}")
    promote_model(model_name, version)
    print(f"Promoted {model_name} v{version} → champion")
    print(f"Load with: mlflow.sklearn.load_model('models:/{model_name}@champion')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote the best model to champion.")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help=f"Metric to rank by. Default: {DEFAULT_METRIC}")
    parser.add_argument("--variant", default=None,
                        help="Restrict to baseline or challenger.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    promote(args.metric, args.variant, args.model_name, args.dry_run)
"""


_GENERATE_SUBMISSION_PY = """\
\"\"\"Generate a Kaggle submission CSV from the champion model.

TODO: set ID_COL and TARGET_COL for this competition, then uncomment
the prediction block that matches your task type.
\"\"\"
from __future__ import annotations

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import mlflow
import pandas as pd

from kitchen.registry import get_production_uri
from kitchen.store import DataStore
from kitchen.tracking import configure_from_env
from src.features.run import FEATURES

ID_COL = "Id"          # TODO: change to this competition's ID column
TARGET_COL = "target"  # TODO: change to the submission target column name

MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "$name-model")


def generate(params_file: str = "params.yaml") -> None:
    with open(params_file) as f:
        params = yaml.safe_load(f)

    configure_from_env()
    store = DataStore()

    test_raw = store.load_csv(params["features"]["test_file"])

    # TODO: apply your feature engineering to the test set, e.g.:
    #   from src.features.run import _engineer
    #   test_df = _engineer(test_raw)[FEATURES]
    raise NotImplementedError(
        "Apply feature engineering to test_raw, then remove this line."
    )

    uri = get_production_uri(MODEL_NAME)
    if uri is None:
        raise RuntimeError(
            f"No champion model found for {MODEL_NAME!r}. "
            "Run flows/promote.py first."
        )
    # TODO: choose the loader that matches your model flavour, then delete the others:
    #
    # XGBoost (model_flavour = "xgboost" in src/train/run.py):
    # import xgboost as xgb
    # model = mlflow.xgboost.load_model(uri)
    # pred = model.predict(xgb.DMatrix(test_df))
    #
    # scikit-learn (model_flavour = "sklearn"):
    # model = mlflow.sklearn.load_model(uri)
    # pred = model.predict(test_df)
    #
    # Generic / pyfunc fallback:
    # model = mlflow.pyfunc.load_model(uri)
    # pred = model.predict(test_df)

    sub = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET_COL: pred})
    out = Path("submissions/submission.csv")
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"Saved {len(sub)} rows → {out}")


if __name__ == "__main__":
    generate()
"""


# ---------------------------------------------------------------------------
# CI workflow templates
# $${{ }} and $$VAR escape the $ so string.Template passes them through as
# ${{ }} and $VAR in the rendered YAML (GitHub Actions / shell syntax).
# ---------------------------------------------------------------------------

_CI_WORKFLOW = """\
name: Train and Evaluate — $name

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  train-evaluate:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write

    env:
      MLFLOW_TRACKING_URI: sqlite:///mlruns.db

    steps:
      - uses: actions/checkout@v4

      - name: Check for raw data in git
        run: |
          FILES=$$(git ls-files data/raw/ | grep -v '\\.gitkeep' || true)
          if [ -n "$$FILES" ]; then
            echo "Raw data files found in git — remove with git rm --cached and track with DVC instead:"
            echo "$$FILES"
            exit 1
          fi

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install kitchen
        run: pip install "kitchen @ git+https://github.com/rkoren/kitchen-platform#subdirectory=kitchen"

      - name: Install project
        run: pip install -e ".[dev]"

      - name: Train
        run: kitchen run train

      - name: Evaluate
        run: kitchen run evaluate

      - name: Push results
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        continue-on-error: true
        run: |
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git config user.name "github-actions[bot]"
          git fetch origin results:results 2>/dev/null || true
          kitchen push --push

      - name: Download base metrics
        if: github.event_name == 'pull_request'
        continue-on-error: true
        env:
          GH_TOKEN: $${{ github.token }}
        run: |
          RUN_ID=$$(gh run list --branch main --workflow train-evaluate.yml --status success --limit 1 --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null || true)
          if [ -n "$$RUN_ID" ]; then
            gh run download "$$RUN_ID" --name metrics --dir base-metrics || true
          fi

      - name: Report
        id: report
        run: |
          BASE=base-metrics/metrics.json
          COMPARE_ARG=""
          if [ -f "$$BASE" ]; then COMPARE_ARG="--compare $$BASE"; fi
          set +e
          REPORT=$$(kitchen report $$COMPARE_ARG)
          REPORT_EXIT=$$?
          set -e
          echo "$$REPORT" >> $$GITHUB_STEP_SUMMARY
          echo "KITCHEN_REPORT<<EOF" >> $$GITHUB_ENV
          echo "$$REPORT" >> $$GITHUB_ENV
          echo "EOF" >> $$GITHUB_ENV
          exit $$REPORT_EXIT

      - name: Upload metrics
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: metrics
          path: metrics.json

      - name: Find PR comment
        uses: peter-evans/find-comment@v3
        id: find-comment
        if: github.event_name == 'pull_request'
        with:
          issue-number: $${{ github.event.pull_request.number }}
          comment-author: github-actions[bot]
          body-includes: Kitchen Report

      - name: Post PR comment
        uses: peter-evans/create-or-update-comment@v4
        if: github.event_name == 'pull_request'
        with:
          comment-id: $${{ steps.find-comment.outputs.comment-id }}
          issue-number: $${{ github.event.pull_request.number }}
          body: $${{ env.KITCHEN_REPORT }}
          edit-mode: replace

  deploy-pages:
    needs: train-evaluate
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: $${{ steps.deploy.outputs.page_url }}
    concurrency:
      group: pages
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install kitchen
        run: pip install "kitchen @ git+https://github.com/rkoren/kitchen-platform#subdirectory=kitchen"

      - name: Fetch results branch
        run: git fetch origin results:results 2>/dev/null || true

      - name: Generate dashboard
        continue-on-error: true
        run: kitchen dashboard generate --output docs/index.html

      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs
      - id: deploy
        uses: actions/deploy-pages@v4
      - name: Link dashboard in summary
        run: echo "**Dashboard:** $${{ steps.deploy.outputs.page_url }}" >> $$GITHUB_STEP_SUMMARY
"""

_CI_WORKFLOW_KAGGLE = """\
name: Train and Evaluate — $name

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
    inputs:
      submit:
        description: 'Submit to Kaggle leaderboard after evaluate'
        type: boolean
        default: false

jobs:
  train-evaluate:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write

    env:
      MLFLOW_TRACKING_URI: sqlite:///mlruns.db

    steps:
      - uses: actions/checkout@v4

      - name: Check for raw data in git
        run: |
          FILES=$$(git ls-files data/raw/ | grep -v '\\.gitkeep' || true)
          if [ -n "$$FILES" ]; then
            echo "Raw data files found in git — remove with git rm --cached and track with DVC instead:"
            echo "$$FILES"
            exit 1
          fi

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install kitchen
        run: pip install "kitchen @ git+https://github.com/rkoren/kitchen-platform#subdirectory=kitchen"

      - name: Install project
        run: pip install -e ".[dev]"

      - name: Ingest data
        env:
          KAGGLE_USERNAME: $${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: $${{ secrets.KAGGLE_KEY }}
        run: kitchen ingest

      - name: Train
        run: kitchen run train

      - name: Evaluate
        run: kitchen run evaluate

      - name: Submit to Kaggle
        if: inputs.submit
        env:
          KAGGLE_USERNAME: $${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: $${{ secrets.KAGGLE_KEY }}
        run: kitchen submit --wait

      - name: Push results
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        continue-on-error: true
        run: |
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git config user.name "github-actions[bot]"
          git fetch origin results:results 2>/dev/null || true
          kitchen push --push

      - name: Download base metrics
        if: github.event_name == 'pull_request'
        continue-on-error: true
        env:
          GH_TOKEN: $${{ github.token }}
        run: |
          RUN_ID=$$(gh run list --branch main --workflow train-evaluate.yml --status success --limit 1 --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null || true)
          if [ -n "$$RUN_ID" ]; then
            gh run download "$$RUN_ID" --name metrics --dir base-metrics || true
          fi

      - name: Report
        id: report
        run: |
          BASE=base-metrics/metrics.json
          COMPARE_ARG=""
          if [ -f "$$BASE" ]; then COMPARE_ARG="--compare $$BASE"; fi
          set +e
          REPORT=$$(kitchen report $$COMPARE_ARG)
          REPORT_EXIT=$$?
          set -e
          echo "$$REPORT" >> $$GITHUB_STEP_SUMMARY
          echo "KITCHEN_REPORT<<EOF" >> $$GITHUB_ENV
          echo "$$REPORT" >> $$GITHUB_ENV
          echo "EOF" >> $$GITHUB_ENV
          exit $$REPORT_EXIT

      - name: Upload metrics
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: metrics
          path: metrics.json

      - name: Find PR comment
        uses: peter-evans/find-comment@v3
        id: find-comment
        if: github.event_name == 'pull_request'
        with:
          issue-number: $${{ github.event.pull_request.number }}
          comment-author: github-actions[bot]
          body-includes: Kitchen Report

      - name: Post PR comment
        uses: peter-evans/create-or-update-comment@v4
        if: github.event_name == 'pull_request'
        with:
          comment-id: $${{ steps.find-comment.outputs.comment-id }}
          issue-number: $${{ github.event.pull_request.number }}
          body: $${{ env.KITCHEN_REPORT }}
          edit-mode: replace

  deploy-pages:
    needs: train-evaluate
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: $${{ steps.deploy.outputs.page_url }}
    concurrency:
      group: pages
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install kitchen
        run: pip install "kitchen @ git+https://github.com/rkoren/kitchen-platform#subdirectory=kitchen"

      - name: Fetch results branch
        run: git fetch origin results:results 2>/dev/null || true

      - name: Generate dashboard
        continue-on-error: true
        run: kitchen dashboard generate --output docs/index.html

      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs
      - id: deploy
        uses: actions/deploy-pages@v4
      - name: Link dashboard in summary
        run: echo "**Dashboard:** $${{ steps.deploy.outputs.page_url }}" >> $$GITHUB_STEP_SUMMARY
"""

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>$name &mdash; Results Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; background: #f9fafb; color: #111; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    #chart-wrap { max-width: 860px; margin-bottom: 2rem; }
    table { border-collapse: collapse; width: 100%; max-width: 860px; font-size: 0.875rem; }
    th, td { padding: 0.35rem 0.75rem; border: 1px solid #e2e8f0; text-align: left; }
    th { background: #f1f5f9; }
    tr.champion { background: #fef9c3; font-weight: 600; }
    #status { color: #6b7280; margin-bottom: 1rem; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>$name &mdash; Results Dashboard</h1>
  <p id="status">Loading&hellip;</p>
  <div id="chart-wrap"><canvas id="chart"></canvas></div>
  <table>
    <thead>
      <tr><th>SHA</th><th>Timestamp</th><th>Run ID</th><th>LB Score</th><th>&#916; LB</th><th>Metrics</th></tr>
    </thead>
    <tbody id="runs-body"></tbody>
  </table>
  <script>
    (function () {
      var loc = window.location;
      var owner = loc.hostname.split('.')[0];
      var repo = loc.pathname.replace(/^\\//, '').split('/')[0] || '$name';
      var apiUrl = 'https://api.github.com/repos/' + owner + '/' + repo + '/contents/results?ref=results';

      function fmtTime(ts) {
        return ts ? new Date(ts).toLocaleString() : '';
      }

      function fmtMetrics(m) {
        return Object.keys(m).sort().map(function (k) { return k + ': ' + m[k]; }).join(', ');
      }

      fetch(apiUrl)
        .then(function (r) {
          if (!r.ok) { throw new Error('HTTP ' + r.status + ' — is the results branch pushed?'); }
          return r.json();
        })
        .then(function (files) {
          if (!Array.isArray(files)) { throw new Error('Unexpected API response'); }
          var jsonFiles = files.filter(function (f) { return f.name.endsWith('.json'); });
          document.getElementById('status').textContent = 'Fetching ' + jsonFiles.length + ' result(s)\\u2026';
          return Promise.all(jsonFiles.map(function (f) {
            return fetch(f.download_url).then(function (r) { return r.json(); });
          }));
        })
        .then(function (runs) {
          runs.sort(function (a, b) { return a.timestamp < b.timestamp ? -1 : 1; });
          document.getElementById('status').textContent = runs.length + ' run(s) loaded.';

          // Show LB Score chart and columns only when at least one run has an lb_score.
          // For non-Kaggle projects (or before any submission), fall back to the first
          // numeric local metric so the chart remains useful.
          var hasLbScore = runs.some(function (r) {
            return r.lb_score !== null && r.lb_score !== undefined;
          });

          var chartLabel, chartData, showLegend;
          if (hasLbScore) {
            chartLabel = 'LB Score';
            chartData = runs.map(function (r) { return r.lb_score; });
            showLegend = false;
          } else {
            var metricKey = null;
            runs.forEach(function (r) {
              if (metricKey) { return; }
              var m = r.metrics || {};
              Object.keys(m).sort().forEach(function (k) {
                if (!metricKey && typeof m[k] === 'number') { metricKey = k; }
              });
            });
            if (metricKey) {
              chartLabel = metricKey;
              chartData = runs.map(function (r) {
                var v = (r.metrics || {})[metricKey];
                return typeof v === 'number' ? v : null;
              });
              showLegend = true;
            } else {
              document.getElementById('chart-wrap').style.display = 'none';
              chartLabel = null;
              chartData = [];
              showLegend = false;
            }
          }

          if (chartLabel) {
            var ctx = document.getElementById('chart').getContext('2d');
            new Chart(ctx, {
              type: 'line',
              data: {
                labels: runs.map(function (r) { return r.sha.slice(0, 8); }),
                datasets: [{
                  label: chartLabel,
                  data: chartData,
                  borderColor: '#3b82f6',
                  backgroundColor: 'rgba(59,130,246,0.08)',
                  tension: 0.2,
                  spanGaps: true
                }]
              },
              options: { responsive: true, plugins: { legend: { display: showLegend } } }
            });
          }

          // Hide LB Score and Δ LB header columns when no LB data is present.
          if (!hasLbScore) {
            var headerCells = document.querySelectorAll('thead tr th');
            [3, 4].forEach(function (i) {
              if (headerCells[i]) { headerCells[i].style.display = 'none'; }
            });
          }

          var champ = runs.find(function (r) { return r.champion; });

          var tbody = document.getElementById('runs-body');
          runs.forEach(function (run) {
            var tr = document.createElement('tr');
            if (run.champion) { tr.className = 'champion'; }

            [
              run.sha.slice(0, 8),
              fmtTime(run.timestamp),
              run.run_id || ''
            ].forEach(function (val) {
              var td = document.createElement('td');
              td.textContent = val;
              tr.appendChild(td);
            });

            if (hasLbScore) {
              var lbTd = document.createElement('td');
              lbTd.textContent = run.lb_score !== null && run.lb_score !== undefined
                ? run.lb_score : '\\u2014';
              tr.appendChild(lbTd);

              var deltaTd = document.createElement('td');
              if (run.champion) {
                deltaTd.textContent = '\\u2605';
              } else if (
                champ &&
                champ.lb_score !== null && champ.lb_score !== undefined &&
                run.lb_score !== null && run.lb_score !== undefined
              ) {
                var delta = run.lb_score - champ.lb_score;
                deltaTd.textContent = (delta >= 0 ? '+' : '') + delta.toFixed(4);
                deltaTd.style.color = delta >= 0 ? '#16a34a' : '#dc2626';
              } else {
                deltaTd.textContent = '\\u2014';
              }
              tr.appendChild(deltaTd);
            }

            var metricsTd = document.createElement('td');
            metricsTd.textContent = fmtMetrics(run.metrics || {});
            tr.appendChild(metricsTd);

            tbody.appendChild(tr);
          });
        })
        .catch(function (err) {
          document.getElementById('status').textContent = 'Error: ' + err.message;
        });
    })();
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Dashboard generated template (kitchen dashboard generate)
# Uses __PLACEHOLDER__ substitution — not string.Template — to avoid conflicts
# with $ in project names and JSON data.
# ---------------------------------------------------------------------------

_DASHBOARD_GENERATED_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__PROJECT_ESCAPED__ — Results Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; background: #f9fafb; color: #111; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    #chart-wrap { max-width: 860px; margin-bottom: 2rem; }
    #runs-table { border-collapse: collapse; width: 100%; font-size: 0.875rem; }
    th, td { padding: 0.35rem 0.75rem; border: 1px solid #e2e8f0; text-align: left; white-space: nowrap; }
    th { background: #f1f5f9; }
    tr.champion { background: #fef9c3; font-weight: 600; }
    .dpos { color: #16a34a; }
    .dneg { color: #dc2626; }
    #status { color: #6b7280; margin-bottom: 1rem; font-size: 0.9rem; }
    .scroll-wrap { overflow-x: auto; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>__PROJECT_ESCAPED__ — Results Dashboard</h1>
  <p id="status">__STATUS_ESCAPED__</p>
  <div id="chart-wrap"><canvas id="chart"></canvas></div>
  <div class="scroll-wrap">
    <table id="runs-table"><thead id="thead"></thead><tbody id="tbody"></tbody></table>
  </div>
  <script>
    var RESULTS = __RESULTS_JSON__;
    var METRIC = __METRIC_JS__;
    var PARAM_KEYS = __PARAM_KEYS_JSON__;
    var HAS_LB = __HAS_LB__;
    (function () {
      var champ = RESULTS.find(function (r) { return r.champion; });
      var champVal = (champ && METRIC) ? ((champ.metrics || {})[METRIC]) : null;
      if (champVal === undefined) { champVal = null; }
      if (METRIC) {
        var cvals = RESULTS.map(function (r) { return (r.metrics || {})[METRIC]; });
        var hasVals = cvals.some(function (v) { return v !== undefined && v !== null; });
        if (hasVals) {
          new Chart(document.getElementById('chart').getContext('2d'), {
            type: 'line',
            data: {
              labels: RESULTS.map(function (r) { return (r.sha || '').slice(0, 8); }),
              datasets: [{
                label: METRIC,
                data: cvals,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59,130,246,0.08)',
                tension: 0.2,
                spanGaps: true,
                pointBackgroundColor: RESULTS.map(function (r) {
                  return r.champion ? '#eab308' : '#3b82f6';
                }),
                pointRadius: 5
              }]
            },
            options: {
              responsive: true,
              plugins: { legend: { display: true } },
              scales: { y: { title: { display: true, text: METRIC } } }
            }
          });
        } else {
          document.getElementById('chart-wrap').style.display = 'none';
        }
      } else {
        document.getElementById('chart-wrap').style.display = 'none';
      }
      var hdrs = ['#', 'SHA', 'Run ID', METRIC || 'Metric', 'Δ vs Champion'];
      if (HAS_LB) { hdrs.push('LB Score'); }
      PARAM_KEYS.forEach(function (k) { hdrs.push(k); });
      hdrs.push('Started');
      var htr = document.createElement('tr');
      hdrs.forEach(function (h) {
        var th = document.createElement('th');
        th.textContent = h;
        htr.appendChild(th);
      });
      document.getElementById('thead').appendChild(htr);
      var tbody = document.getElementById('tbody');
      RESULTS.forEach(function (run, i) {
        var tr = document.createElement('tr');
        if (run.champion) { tr.className = 'champion'; }
        var mVal = METRIC ? (run.metrics || {})[METRIC] : null;
        if (mVal === undefined) { mVal = null; }
        var mStr = (mVal !== null) ? mVal.toFixed(4) : '—';
        var deltaStr, deltaCls = '';
        if (run.champion) {
          deltaStr = '★';
        } else if (champVal !== null && mVal !== null) {
          var d = mVal - champVal;
          deltaStr = (d >= 0 ? '+' : '') + d.toFixed(4);
          deltaCls = d >= 0 ? 'dpos' : 'dneg';
        } else {
          deltaStr = '—';
        }
        [run.champion ? '[C]' : String(i + 1),
         (run.sha || '').slice(0, 8),
         (run.run_id || '').slice(0, 8),
         mStr
        ].forEach(function (v) {
          var td = document.createElement('td');
          td.textContent = v;
          tr.appendChild(td);
        });
        var dtd = document.createElement('td');
        dtd.textContent = deltaStr;
        if (deltaCls) { dtd.className = deltaCls; }
        tr.appendChild(dtd);
        if (HAS_LB) {
          var ltd = document.createElement('td');
          var lb = run.lb_score;
          ltd.textContent = (lb !== null && lb !== undefined) ? String(lb) : '—';
          tr.appendChild(ltd);
        }
        PARAM_KEYS.forEach(function (k) {
          var ptd = document.createElement('td');
          var p = run.params;
          ptd.textContent = (p && p[k] !== undefined) ? p[k] : '-';
          tr.appendChild(ptd);
        });
        var std = document.createElement('td');
        std.textContent = run.timestamp ? new Date(run.timestamp).toLocaleString() : '—';
        tr.appendChild(std);
        tbody.appendChild(tr);
      });
    }());
  </script>
</body>
</html>
"""


def _render_dashboard_html(
    project: str,
    status: str,
    results: "list[dict]",
    metric: str,
    param_keys: "list[str]",
    has_lb: bool,
) -> str:
    import html as _html
    import json as _json

    html = _DASHBOARD_GENERATED_HTML
    html = html.replace("__PROJECT_ESCAPED__", _html.escape(project))
    html = html.replace("__STATUS_ESCAPED__", _html.escape(status))
    html = html.replace("__RESULTS_JSON__", _json.dumps(results))
    html = html.replace("__METRIC_JS__", _json.dumps(metric))
    html = html.replace("__PARAM_KEYS_JSON__", _json.dumps(param_keys))
    html = html.replace("__HAS_LB__", "true" if has_lb else "false")
    return html


def _find_free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _serve_local_dashboard(html_path: Path) -> None:
    """Start a local HTTP server in CWD and open html_path in the browser.

    Blocks until the user presses Ctrl+C.
    """
    import http.server
    import threading
    import time
    import webbrowser

    port = _find_free_port()

    try:
        rel = html_path.resolve().relative_to(Path.cwd())
    except ValueError:
        rel = html_path

    url = f"http://localhost:{port}/{rel}"

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # suppress request log lines
            pass

    httpd = http.server.HTTPServer(("", port), _QuietHandler)

    typer.echo(f"Dashboard → {url}")
    typer.echo("Press Ctrl+C to stop.\n")

    def _open() -> None:
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nStopped.")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# DVC scaffold templates (--with-dvc)
# ---------------------------------------------------------------------------

_DVC_YAML = """\
# DVC pipeline for $name.
# Run `dvc repro` to execute stages in dependency order, skipping unchanged ones.
#
# First-time setup:
#   pip install kitchen[dvc]
#   dvc remote modify s3remote url s3://YOUR-BUCKET/dvc
#   dvc push         # upload processed data and models to your S3 remote
#   dvc pull         # restore on a new machine or CI runner
stages:

  # Uncomment and customise for script-driven ingest (custom API, S3 bucket, etc.).
  # For manual downloads, place files in data/raw/ then run `dvc add data/raw/`.
  # ingest:
  #   cmd: python src/ingest/run.py
  #   deps:
  #     - src/ingest/run.py
  #   outs:
  #     - data/raw/

  features:
    cmd: kitchen run features
    deps:
      - src/features/run.py
      - data/raw/
    params:
      - features
    outs:
      - data/processed/

  train:
    cmd: kitchen run train
    deps:
      - src/train/run.py
      - data/processed/
    params:
      - model
    outs:
      - models/

  evaluate:
    cmd: kitchen run evaluate
    deps:
      - src/evaluate/run.py
      - models/
      - data/processed/
    params:
      - model
    metrics:
      - metrics.json:
          cache: false
"""

_DVC_YAML_KAGGLE = """\
# DVC pipeline for $name.
# Run `dvc repro` to execute stages in dependency order, skipping unchanged ones.
#
# First-time setup:
#   pip install kitchen[dvc]
#   dvc remote modify s3remote url s3://YOUR-BUCKET/dvc
#   dvc push         # upload processed data and models to your S3 remote
#   dvc pull         # restore on a new machine or CI runner
stages:

  # Kaggle raw data is pinned by competition slug and re-downloaded on demand
  # via `kitchen ingest` — no DVC tracking needed for data/raw/.

  features:
    cmd: kitchen run features
    deps:
      - src/features/run.py
      - data/raw/
    params:
      - features
    outs:
      - data/processed/

  train:
    cmd: kitchen run train
    deps:
      - src/train/run.py
      - data/processed/
    params:
      - model
    outs:
      - models/

  evaluate:
    cmd: kitchen run evaluate
    deps:
      - src/evaluate/run.py
      - models/
      - data/processed/
    params:
      - model
    metrics:
      - metrics.json:
          cache: false

  submit:
    cmd: kitchen submit
    deps:
      - models/
      - data/raw/
    outs:
      - submissions/
"""

_DVCIGNORE = """\
# DVC will not track files matching these patterns (same syntax as .gitignore).
__pycache__/
*.py[cod]
.venv/
mlruns/
mlruns.db
"""

_DVC_CONFIG = """\
[core]
    remote = s3remote
[remote "s3remote"]
    url = s3://YOUR-BUCKET/dvc
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    if champ_score is None:
        wins, reason = True, "no current champion"
    elif lower_is_better:
        wins = new_score < champ_score
        reason = f"{new_score:.6f} < {champ_score:.6f} (lower=better)"
    else:
        wins = new_score > champ_score
        reason = f"{new_score:.6f} > {champ_score:.6f} (higher=better)"

    typer.echo()
    if wins:
        reg_version = register_model(new_run.info.run_id, "model", resolved_model)
        promote_model(resolved_model, reg_version, alias="champion")
        typer.echo(f"auto-promote: {new_run.info.run_id[:8]} → champion  ({reason})")
        typer.echo(f"             {resolved_model} v{reg_version} @ champion")
    else:
        typer.echo(f"auto-promote: skipped — new run did not beat champion  ({reason})")


def _to_class_name(name: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[-_\s]+", name))


def _render(tmpl: str, name: str, class_name: str, **extra) -> str:
    return string.Template(tmpl).substitute(name=name, class_name=class_name, **extra)


def _write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        typer.echo(f"  skip   {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    typer.echo(f"  create {path}")


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


# ---------------------------------------------------------------------------
# Experiments sub-commands
# ---------------------------------------------------------------------------

experiments_app = typer.Typer(help="List and compare MLflow experiment runs.", no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")


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


@app.command()
def leaderboard(
    metric: Annotated[
        str, typer.Option("--metric", "-m", help="Primary metric to rank by")
    ] = "loto_brier",
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
) -> None:
    """Rank runs by a metric; shows full run_id and lb_score for easy replay.

    Defaults to loto_brier (lower=better) for competition use. The full run_id
    is shown so it can be copied directly into flows/submit.py --run-id.

    [C] marks the promoted champion from the model registry. ★ marks the
    top-ranked run by metric (they may differ if a newer run hasn't been promoted yet).
    """
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

    order = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"metrics.{metric} > -99999",
        order_by=[f"metrics.{metric} {order}"],
        max_results=limit,
    )
    if not runs:
        typer.echo(f"No runs with metric {metric!r} found in {exp_name!r}.")
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
        variant = run.data.tags.get("model_variant", "")[:12]
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
app.add_typer(run_app, name="run")


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
        typer.Option("--promote-metric", help="Metric to compare for auto-promote (required with --auto-promote)"),
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
    --promote-metric and promotes automatically if it wins.

    With --override, one or more params.yaml values are shadowed for this run
    only without modifying the file. Overrides are logged as MLflow run tags
    (override.<key>) so they appear in kitchen leaderboard and kitchen diff.
    """
    import sys

    if auto_promote and not promote_metric:
        typer.echo("error: --promote-metric is required when using --auto-promote", err=True)
        raise typer.Exit(1)

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
        except ValidationError:
            _fail(params_file, f"invalid — run `kitchen validate {params_file}`")
        except Exception as exc:
            _fail(params_file, str(exc))
    else:
        typer.echo(f"  - {params_file:<26}  not found (run from a project directory)")

    # --- Prep: project src modules ---
    src_candidates = [
        Path("src/features/run.py"),
        Path("src/train/run.py"),
        Path("src/evaluate/run.py"),
    ]
    if any(p.exists() for p in src_candidates):
        for p in src_candidates:
            if p.exists():
                _ok(str(p))
            else:
                _fail(str(p), "implement to run the pipeline")

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
        str | None, typer.Option("--compare", help="Path to base metrics.json for delta comparison")
    ] = None,
) -> None:
    """Write a metrics summary to stdout (pipe to $GITHUB_STEP_SUMMARY in CI)."""
    import json

    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        typer.echo(f"error: {metrics_file} not found — run `kitchen run evaluate` first", err=True)
        raise typer.Exit(1)

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"error: could not parse {metrics_file}: {exc}", err=True)
        raise typer.Exit(1)

    base_metrics: dict | None = None
    if compare is not None:
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
                pr_val = metrics.get(key)
                base_val = base_metrics.get(key)
                pr_str = (
                    f"{pr_val:.6f}"
                    if isinstance(pr_val, float)
                    else str(pr_val)
                    if pr_val is not None
                    else "(new)"
                )
                base_str = (
                    f"{base_val:.6f}"
                    if isinstance(base_val, float)
                    else str(base_val)
                    if base_val is not None
                    else "(new)"
                )
                if isinstance(pr_val, (int, float)) and isinstance(base_val, (int, float)):
                    delta = pr_val - base_val
                    delta_str = (
                        f"{float(delta):+.6f}"
                        if isinstance(pr_val, float) or isinstance(base_val, float)
                        else f"{delta:+d}"
                    )
                else:
                    delta_str = "—"
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
                pr_val = metrics.get(key)
                base_val = base_metrics.get(key)
                pr_str = (
                    f"{pr_val:.6f}"
                    if isinstance(pr_val, float)
                    else str(pr_val)
                    if pr_val is not None
                    else "(new)"
                )
                base_str = (
                    f"{base_val:.6f}"
                    if isinstance(base_val, float)
                    else str(base_val)
                    if base_val is not None
                    else "(new)"
                )
                if isinstance(pr_val, (int, float)) and isinstance(base_val, (int, float)):
                    delta = pr_val - base_val
                    delta_str = (
                        f"{float(delta):+.6f}"
                        if isinstance(pr_val, float) or isinstance(base_val, float)
                        else f"{delta:+d}"
                    )
                else:
                    delta_str = "—"
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

    reg_version = register_model(actual_run_id, "model", model_name)
    typer.echo(f"\nRegistered : {model_name} v{reg_version}")
    promote_model(model_name, reg_version, alias=alias)
    typer.echo(f"Promoted   : {model_name} v{reg_version} → {alias}")
    typer.echo(f"Load with  : mlflow.sklearn.load_model('models:/{model_name}@{alias}')")
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

            # Params from the MLflow run (stored as strings by MLflow)
            try:
                _p = dict(_client.get_run(run_id).data.params)
                params_from_run = _p if _p else None
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


# ---------------------------------------------------------------------------
# kitchen dvc — add DVC scaffolding to an existing project
# ---------------------------------------------------------------------------

dvc_app = typer.Typer(help="DVC scaffolding helpers.", no_args_is_help=True)
app.add_typer(dvc_app, name="dvc")


@dvc_app.command("init")
def dvc_init(
    params_file: Annotated[
        str,
        typer.Option("--params", help="Path to params.yaml (used to detect project name and source)"),
    ] = "params.yaml",
    kaggle: Annotated[
        bool,
        typer.Option(
            "--kaggle/--no-kaggle",
            help="Use the Kaggle DVC template (submit stage, no ingest placeholder)",
        ),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing dvc.yaml and .dvcignore")
    ] = False,
) -> None:
    """Add DVC scaffolding (dvc.yaml, .dvcignore, .dvc/config) to an existing project.

    Reads params.yaml for the project name and data source. If params.yaml is
    not found, falls back to the current directory name and the non-Kaggle template
    (override with --kaggle).

    Requires the dvc binary: pip install kitchen[dvc]
    """
    import shutil as _shutil
    import subprocess as _subprocess

    if not _shutil.which("dvc"):
        typer.echo(
            "error: dvc binary not found — run `pip install kitchen[dvc]` first",
            err=True,
        )
        raise typer.Exit(1)

    # Resolve project name and source from params.yaml, or fall back to cwd name.
    project_name = Path.cwd().name
    is_kaggle = kaggle
    p = Path(params_file)
    if p.exists():
        import yaml as _yaml

        try:
            raw = _yaml.safe_load(p.read_text(encoding="utf-8"))
            project_name = raw.get("experiment", project_name)
            if not kaggle:
                is_kaggle = raw.get("data", {}).get("source") == "kaggle"
        except Exception:
            pass  # unparseable params.yaml — use defaults
    else:
        typer.echo(
            f"note: {params_file!r} not found — using directory name {project_name!r} and "
            f"{'kaggle' if is_kaggle else 'non-kaggle'} template. "
            "Pass --params or --kaggle to override.",
        )

    root = Path.cwd()
    class_name = _to_class_name(project_name)

    typer.echo(f"\nAdding DVC scaffolding to {root}\n")

    dvc_tmpl = _DVC_YAML_KAGGLE if is_kaggle else _DVC_YAML
    files = [
        (root / "dvc.yaml", _render(dvc_tmpl, project_name, class_name)),
        (root / ".dvcignore", _DVCIGNORE),
    ]
    for path, content in files:
        _write(path, content, overwrite)

    dvc_dir = root / ".dvc"
    if not dvc_dir.exists():
        try:
            _subprocess.run(
                ["dvc", "init"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            typer.echo(f"  dvc    initialized DVC repository in {root}")
        except _subprocess.CalledProcessError as exc:
            typer.echo(
                f"warning: dvc init failed: {exc.stderr.strip() or exc.stdout.strip()}",
                err=True,
            )
    else:
        typer.echo("  dvc    DVC already initialized — skipping dvc init")

    # Write .dvc/config with S3 remote placeholder (always overwrite dvc init default)
    _write(dvc_dir / "config", _DVC_CONFIG, overwrite=True)

    typer.echo("""
Done. Next steps:

  dvc remote modify s3remote url s3://YOUR-BUCKET/dvc  # set your S3 remote
  dvc push                  # upload data/processed/ + models/ to S3
  # dvc pull               # restore on a new machine
  # dvc repro              # run the full pipeline (skips unchanged stages)
""")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


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
        (root / "pyproject.toml", r(_PYPROJECT_TOML, name, class_name)),
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
        import subprocess as _subprocess

        dvc_dir = root / ".dvc"
        if not dvc_dir.exists():
            try:
                _subprocess.run(
                    ["dvc", "init"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                typer.echo(f"  dvc    initialized DVC repository in {root}")
            except _subprocess.CalledProcessError as exc:
                typer.echo(
                    f"warning: dvc init failed: {exc.stderr.strip() or exc.stdout.strip()}",
                    err=True,
                )
        else:
            typer.echo("  dvc    DVC already initialized — skipping dvc init")
        # Write .dvc/config with S3 remote placeholder (always overwrite default from dvc init)
        _write(dvc_dir / "config", _DVC_CONFIG, overwrite=True)

    cd_target = root.name if not here else "."
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

  cd {cd_target}
  pip install -e ../kitchen-platform/kitchen -e .
  # If kitchen was installed via pipx, inject the project package instead:
  # pipx inject rkoren-kitchen .
  cp .env.example .env
  kitchen check                       # verify tools, credentials, and config
                                      # (includes a check that your package is importable)
{data_step}
  # Implement src/features/run.py, src/train/run.py, src/evaluate/run.py
  kitchen run train                   # features → train → log to MLflow
  kitchen run evaluate                # load champion model, compute metrics
  kitchen experiments compare METRIC  # rank runs by metric
  kitchen promote METRIC              # promote best run to the registry
{submit_step}{ci_note}{dvc_note}
""")


# ---------------------------------------------------------------------------
# kitchen serve — start the FastAPI serving app locally
# ---------------------------------------------------------------------------

serve_app = typer.Typer(help="Serving helpers.", no_args_is_help=True)
app.add_typer(serve_app, name="serve")


@serve_app.command("local")
def serve_local(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind (default 8080)")] = 8080,
    predictor_dir: Annotated[
        str | None,
        typer.Option(
            "--predictor-dir",
            help=(
                "Directory containing predictor.py. "
                "Defaults to src/serve/ if it contains predictor.py, else ./"
            ),
        ),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload/--no-reload",
            help="Enable uvicorn auto-reload on code changes (requires watchfiles)",
        ),
    ] = True,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open /docs in the browser after startup"),
    ] = True,
) -> None:
    """Start the kitchen FastAPI serving app locally with uvicorn.

    Resolves predictor.py in this order:
      1. --predictor-dir <dir>/predictor.py
      2. src/serve/predictor.py       (project default location)
      3. ./predictor.py               (current directory)

    The resolved directory is prepended to PYTHONPATH so the app can
    import the predictor module at startup. If none of the above
    locations contain predictor.py the app still starts and returns
    HTTP 501 until predictor.py is created.

    Press Ctrl+C to stop.
    """
    import os
    import subprocess
    import sys
    import threading
    import webbrowser

    # ── Resolve the directory that contains predictor.py ─────────────────────
    cwd = Path.cwd()
    if predictor_dir is not None:
        pred_dir = Path(predictor_dir).resolve()
    elif (cwd / "src" / "serve" / "predictor.py").exists():
        pred_dir = (cwd / "src" / "serve").resolve()
    elif (cwd / "predictor.py").exists():
        pred_dir = cwd.resolve()
    else:
        # Default: src/serve/ — app returns 501 if predictor.py is absent.
        pred_dir = (cwd / "src" / "serve").resolve()

    url = f"http://localhost:{port}"
    typer.echo(f"Serving   → {url}")
    typer.echo(f"Predictor → {pred_dir}")
    if reload:
        typer.echo("Reload    → enabled (watchfiles)")
    typer.echo("Press Ctrl+C to stop.\n")

    # ── Open /docs after a short delay ───────────────────────────────────────
    if open_browser:
        def _open_after_delay() -> None:
            import time

            time.sleep(1.5)
            webbrowser.open(f"{url}/docs")

        threading.Thread(target=_open_after_delay, daemon=True).start()

    # ── Prepend pred_dir to PYTHONPATH and set KITCHEN_PREDICTOR_DIR ─────────
    # PYTHONPATH: backwards-compatible for any code that scans sys.path.
    # KITCHEN_PREDICTOR_DIR: used by kitchen.serve.loader for deterministic
    # resolution without a full sys.path scan.
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = (
        f"{pred_dir}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(pred_dir)
    )
    env = {
        **os.environ,
        "PYTHONPATH": new_pythonpath,
        "KITCHEN_PREDICTOR_DIR": str(pred_dir),
    }

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "kitchen.serve.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, env=env, check=False)
    except KeyboardInterrupt:
        typer.echo("\nStopped.")


# ---------------------------------------------------------------------------
# kitchen dashboard — generate and view the static results dashboard
# ---------------------------------------------------------------------------

dashboard_app = typer.Typer(help="Dashboard helpers.", no_args_is_help=True)
app.add_typer(dashboard_app, name="dashboard")


@dashboard_app.command("generate")
def dashboard_generate(
    output: Annotated[
        str, typer.Option("--output", help="Path to write the HTML file")
    ] = "dashboard/index.html",
    branch: Annotated[
        str, typer.Option("--branch", help="Git branch holding results/*.json files")
    ] = "results",
    metric: Annotated[
        str | None, typer.Option("--metric", help="Primary metric to chart (auto-detected if omitted)")
    ] = None,
    show_params: Annotated[
        str | None,
        typer.Option(
            "--show-params",
            help="Comma-separated param keys to show as extra columns (e.g. model.eta,model.max_depth)",
        ),
    ] = None,
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml (used for project name)")
    ] = "params.yaml",
    serve: Annotated[
        bool,
        typer.Option(
            "--serve/--no-serve",
            help="After generating, start a local HTTP server and open the dashboard in the browser",
        ),
    ] = False,
) -> None:
    """Re-render dashboard/index.html from results/*.json on the results branch.

    Reads result snapshots written by `kitchen push` from the local results branch
    and produces a self-contained HTML file with all data embedded — no server or
    GitHub Pages needed to view it locally.

    Pass --serve to start a local HTTP server and open the dashboard immediately.
    The generated file is NOT committed to git; it is a local build artifact.
    """
    import json
    import subprocess
    from datetime import datetime, timezone

    branch_ref = f"refs/heads/{branch}"
    check = subprocess.run(
        ["git", "rev-parse", "--verify", branch_ref],
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        typer.echo(
            f"error: branch '{branch}' not found locally — run `kitchen push` first.",
            err=True,
        )
        raise typer.Exit(1)

    ls_raw = subprocess.check_output(
        ["git", "ls-tree", "--name-only", branch_ref, "results/"]
    ).decode().strip()
    json_files = [f for f in ls_raw.splitlines() if f.endswith(".json")]
    if not json_files:
        typer.echo(
            f"error: no .json files found under results/ on branch '{branch}'.",
            err=True,
        )
        raise typer.Exit(1)

    results: list[dict] = []
    for file_path in json_files:
        raw = subprocess.check_output(
            ["git", "cat-file", "-p", f"{branch_ref}:{file_path}"]
        ).decode()
        try:
            results.append(json.loads(raw))
        except json.JSONDecodeError:
            typer.echo(f"warning: skipping malformed result file {file_path}", err=True)

    if not results:
        typer.echo("error: no valid result files found.", err=True)
        raise typer.Exit(1)

    results.sort(key=lambda r: r.get("timestamp", ""))

    project = "project"
    if Path(params_file).exists():
        try:
            from kitchen.config import KitchenConfig

            cfg = KitchenConfig.from_yaml(params_file)
            project = cfg.experiment
        except Exception:
            pass

    resolved_metric = metric
    if resolved_metric is None:
        priority = [
            "val_accuracy", "val_brier", "val_log_loss", "val_roc_auc",
            "val_rmse", "val_mae", "loto_brier",
        ]
        for candidate in priority:
            if any(candidate in (r.get("metrics") or {}) for r in results):
                resolved_metric = candidate
                break
        if resolved_metric is None:
            for r in results:
                for k in (r.get("metrics") or {}):
                    if not k.startswith("fi."):
                        resolved_metric = k
                        break
                if resolved_metric:
                    break

    param_keys: list[str] = []
    if show_params:
        requested = [p.strip() for p in show_params.split(",") if p.strip()]
        if any(r.get("params") for r in results):
            param_keys = requested

    has_lb = any(r.get("lb_score") is not None for r in results)

    status = (
        f"{len(results)} run(s) loaded. "
        f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    html = _render_dashboard_html(
        project=project,
        status=status,
        results=results,
        metric=resolved_metric or "",
        param_keys=param_keys,
        has_lb=has_lb,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    typer.echo(f"Dashboard written → {out_path}  ({len(results)} run(s))")
    typer.echo(f"  metric  : {resolved_metric or '—'}")
    if param_keys:
        typer.echo(f"  params  : {', '.join(param_keys)}")

    if serve:
        _serve_local_dashboard(out_path)
    else:
        typer.echo("\nOpen with: kitchen open  (or kitchen dashboard generate --serve)")


if __name__ == "__main__":
    app()
