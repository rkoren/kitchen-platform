"""kitchen CLI — scaffold, validate, and manage competition projects.

Usage:
    kitchen init <name>                 # create ./<name>/ with full project scaffold
    kitchen init <name> --here          # scaffold into current directory
    kitchen validate [params.yaml]      # validate a params.yaml against KitchenConfig
    kitchen experiments list            # list recent runs in an experiment
    kitchen experiments compare METRIC  # rank runs by a metric
    kitchen promote METRIC              # promote best run to the model registry
"""
from __future__ import annotations

import re
import string
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="kitchen ML platform CLI", add_completion=False, no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the kitchen version."""
    typer.echo(f"kitchen {_pkg_version('kitchen')}")


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

Kaggle competition project on the [kitchen platform](../light-ml-platform/kitchen).

## Setup

```bash
pip install -e ../light-ml-platform/kitchen -e .
cp .env.example .env
# Download competition data to data/raw/
```

## The contract — 3 files to implement

| File | Class | Method |
|---|---|---|
| `src/features/run.py` | `${class_name}Features(FeatureBuilder)` | `build(raw_df) -> df` |
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
MLFLOW_TRACKING_URI=sqlite:///mlruns.db
MLFLOW_EXPERIMENT=$name
MLFLOW_MODEL_NAME=$name-model
MLFLOW_PROMOTE_METRIC=val_accuracy
MLFLOW_PROMOTE_LOWER_IS_BETTER=false
AWS_PROFILE=default
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
  # xgb:
  #   n_estimators: 500
  #   max_depth: 6
  #   learning_rate: 0.05

mlflow:
  tracking_uri: sqlite:///mlruns.db

run_name: baseline
metrics_file: metrics.json
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
    "kitchen",           # pip install -e ../light-ml-platform/kitchen
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
\"\"\"
from __future__ import annotations

import pandas as pd
from kitchen.steps import FeatureBuilder
from kitchen.store import DataStore

# Columns passed to the model (exclude the target column).
FEATURES: list[str] = []  # TODO: fill in after feature engineering


class ${class_name}Features(FeatureBuilder):
    def build(self, raw: pd.DataFrame, params: dict) -> pd.DataFrame:
        \"\"\"Transform raw CSV data into model-ready features + target column.\"\"\"
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

_TEST_FEATURES = """\
\"\"\"Tests for $name feature engineering.\"\"\"
import pandas as pd
import pytest

from src.features.run import ${class_name}Features, FEATURES


@pytest.fixture
def raw_row() -> pd.DataFrame:
    # TODO: replace with a representative row from your raw training data
    return pd.DataFrame([{}])


def test_feature_builder_returns_expected_columns(raw_row):
    out = ${class_name}Features().build(raw_row)
    assert set(FEATURES).issubset(out.columns)
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
    model = mlflow.pyfunc.load_model(uri)

    # TODO: uncomment and adapt one of these prediction styles:
    #
    # Binary classification — hard label (e.g. True/False, 0/1):
    # pred = model.predict(test_df)
    #
    # Binary classification — probability (e.g. for log-loss competitions):
    # pred = model.predict(test_df)  # returns probabilities for pyfunc models
    #
    # Regression:
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
# Helpers
# ---------------------------------------------------------------------------

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


def _to_class_name(name: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[-_\s]+", name))


def _render(tmpl: str, name: str, class_name: str) -> str:
    return string.Template(tmpl).substitute(name=name, class_name=class_name)


def _write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        typer.echo(f"  skip   {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    typer.echo(f"  create {path}")


# ---------------------------------------------------------------------------
# Experiments sub-commands
# ---------------------------------------------------------------------------

experiments_app = typer.Typer(help="List and compare MLflow experiment runs.", no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")


@experiments_app.command("list")
def experiments_list(
    experiment: Annotated[str | None, typer.Option("--experiment", "-e", help="Experiment name")] = None,
    params_file: Annotated[str, typer.Option("--params", help="params.yaml to read experiment from")] = "params.yaml",
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
        status = (run.info.status or "")[:10]
        started = _time_ago(run.info.start_time) if run.info.start_time else "-"
        row = f"{run_id:<10}  {name:<20}  {status:<10}  {started:<12}"
        for k in metric_keys:
            row += f"  {_fmt_metric(run.data.metrics.get(k)):>{col_w}}"
        typer.echo(row)

    typer.echo()


@experiments_app.command("compare")
def experiments_compare(
    metric: str = typer.Argument(..., help="Metric to rank by"),
    experiment: Annotated[str | None, typer.Option("--experiment", "-e", help="Experiment name")] = None,
    params_file: Annotated[str, typer.Option("--params", help="params.yaml to read experiment from")] = "params.yaml",
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
# Promote command
# ---------------------------------------------------------------------------

@app.command()
def promote(
    metric: str = typer.Argument(..., help="Metric to rank runs by"),
    experiment: Annotated[str | None, typer.Option("--experiment", "-e", help="Experiment name")] = None,
    params_file: Annotated[str, typer.Option("--params", help="params.yaml to read experiment from")] = "params.yaml",
    model_name: Annotated[str | None, typer.Option("--model-name", help="Registered model name")] = None,
    alias: Annotated[str, typer.Option("--alias", help="Model alias to set")] = "champion",
    lower_is_better: Annotated[bool, typer.Option("--lower-is-better/--higher-is-better")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show winner without registering")] = False,
) -> None:
    """Promote the best-performing run to the model registry."""
    import os

    from kitchen.registry import get_best_run, get_production_uri, promote_model, register_model
    from kitchen.tracking import configure_from_env

    configure_from_env()
    exp_name = _resolve_experiment(experiment, params_file)

    if model_name is None:
        model_name = os.environ.get("MLFLOW_MODEL_NAME", f"{exp_name}-model")

    try:
        run = get_best_run(exp_name, metric, lower_is_better=lower_is_better)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    run_id = run.info.run_id
    score = run.data.metrics.get(metric, float("nan"))
    variant = run.data.tags.get("model_variant", "")
    variant_str = f" ({variant})" if variant else ""
    direction = "lower=better" if lower_is_better else "higher=better"

    typer.echo(f"\nExperiment : {exp_name}")
    typer.echo(f"Best run   : {run_id[:8]}  {metric}={score:.6f}{variant_str}  ({direction})")

    current = get_production_uri(model_name, alias)
    if current:
        typer.echo(f"Current    : {current}")

    if dry_run:
        typer.echo("\nDry run — skipping registration and promotion.")
        return

    version = register_model(run_id, "model", model_name)
    typer.echo(f"\nRegistered : {model_name} v{version}")
    promote_model(model_name, version, alias=alias)
    typer.echo(f"Promoted   : {model_name} v{version} → {alias}")
    typer.echo(f"Load with  : mlflow.sklearn.load_model('models:/{model_name}@{alias}')")
    typer.echo()


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

@app.command()
def init(
    name: str = typer.Argument(..., help="Project / competition name (e.g. spaceship-titanic)"),
    here: bool = typer.Option(False, "--here", help="Scaffold into cwd, not a new subdirectory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
) -> None:
    """Scaffold a new kitchen competition project."""
    class_name = _to_class_name(name)
    root = Path.cwd() if here else Path.cwd() / name

    typer.echo(f"\nScaffolding '{name}' → {root}\n")

    r = _render  # shorthand

    files: list[tuple[Path, str]] = [
        (root / "CLAUDE.md",                        r(_CLAUDE_MD, name, class_name)),
        (root / ".env.example",                     r(_ENV_EXAMPLE, name, class_name)),
        (root / ".gitignore",                       r(_GITIGNORE, name, class_name)),
        (root / "params.yaml",                      r(_PARAMS_YAML, name, class_name)),
        (root / "pyproject.toml",                   r(_PYPROJECT_TOML, name, class_name)),
        (root / "infra" / f"{name}.yaml",           r(_INFRA_YAML, name, class_name)),
        (root / "src" / "__init__.py",              ""),
        (root / "src" / "features" / "__init__.py", ""),
        (root / "src" / "features" / "run.py",     r(_FEATURES_RUN, name, class_name)),
        (root / "src" / "train" / "__init__.py",   ""),
        (root / "src" / "train" / "run.py",        r(_TRAIN_RUN, name, class_name)),
        (root / "src" / "evaluate" / "__init__.py", ""),
        (root / "src" / "evaluate" / "run.py",     r(_EVALUATE_RUN, name, class_name)),
        (root / "src" / "tests" / "__init__.py",   ""),
        (root / "src" / "tests" / "test_features.py", r(_TEST_FEATURES, name, class_name)),
        (root / "experiments" / "__init__.py",      ""),
        (root / "experiments" / "baseline.py",     r(_BASELINE_PY, name, class_name)),
        (root / "experiments" / "challenger.py",   r(_CHALLENGER_PY, name, class_name)),
        (root / "flows" / "train_flow.py",              r(_TRAIN_FLOW_PY, name, class_name)),
        (root / "flows" / "promote.py",               r(_PROMOTE_PY, name, class_name)),
        (root / "flows" / "generate_submission.py",   r(_GENERATE_SUBMISSION_PY, name, class_name)),
        (root / "data" / "raw" / ".gitkeep",       ""),
        (root / "data" / "processed" / ".gitkeep", ""),
        (root / "submissions" / ".gitkeep",         ""),
    ]

    for path, content in files:
        _write(path, content, overwrite)

    typer.echo(f"""
Done. Next steps:

  cd {root.name if not here else '.'}
  pip install -e ../light-ml-platform/kitchen -e .
  cp .env.example .env
  # Download competition data to data/raw/
  # Implement src/features/run.py, src/train/run.py, src/evaluate/run.py
  python experiments/baseline.py
  python experiments/challenger.py
  python flows/promote.py --dry-run
  python flows/promote.py
  python flows/generate_submission.py
""")


if __name__ == "__main__":
    app()
