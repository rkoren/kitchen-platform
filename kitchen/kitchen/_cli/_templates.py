"""Scaffold file templates rendered by `kitchen init`.

All strings use string.Template substitution: $name (slug) and
$class_name (PascalCase). Literal $$ renders as $ in output.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Templates
# Each template uses $name (slug) and $class_name (PascalCase) as substitution vars.
# Literal $$ → $ in output.
# ---------------------------------------------------------------------------

_CLAUDE_MD = """\
# $name

Kaggle competition project built on the [kitchen platform](https://github.com/rkoren/kitchen-platform).

## Setup

```bash
pip install rkoren-kitchen -e .
# Contributors working from the monorepo: pip install -e ../kitchen-platform/kitchen -e .
cp .env.example .env
# Download competition data to data/raw/
```

## The contract — 3 files to implement

| File | Class | Method |
|---|---|---|
| `src/features/run.py` | `${class_name}Features(FeatureBuilder)` | `build(raw_or_sources, params) -> df` |
| `src/train/run.py` | `${class_name}Trainer(Trainer)` | `fit(df, params) -> model` |
| `src/evaluate/run.py` | `${class_name}Evaluator(Evaluator)` | `evaluate(model, df) -> dict` |

All config lives in `menu.yaml`. File paths resolve from `params["features"].*`;
model hyperparams from `params["model"].*`.

## Running experiments

```bash
kitchen run train                   # features → train → log to MLflow
kitchen run evaluate                # load champion model, compute metrics
kitchen leaderboard                 # rank all runs by primary metric
kitchen promote METRIC              # promote best run to the registry
kitchen ui                          # open MLflow UI in browser

# Experiment variants (edit first, then run)
python experiments/baseline.py
python experiments/challenger.py

# Generate Kaggle submission
kitchen submit
```

## Kitchen modules

- `kitchen.steps` — `FeatureBuilder`, `Trainer` (set `model_flavour`), `Evaluator` ABCs
- `kitchen.tracking` — `Tracker`, `configure_from_env()`, `init_experiment()`
- `kitchen.store` — `DataStore` (wraps `data/raw/`, `data/processed/`, `models/`)
- `kitchen.modeling` — `train_val_split`, `classification_metrics`, `regression_metrics`

## Experiment tagging

Both experiment scripts tag runs with `model_variant=baseline` or `model_variant=challenger`.
`kitchen promote METRIC` promotes the best run to the `champion` alias.
Load the champion with `mlflow.pyfunc.load_model('models:/$name-model@champion')`.
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
runs.jsonl
sweep.jsonl
sweep-runs/

# infra (generated)
infra/tf/
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
    "rkoren-kitchen>=1.0",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
$model_deps]

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

_MODEL_SECTION_XGB = """\
  xgb:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8"""

_MODEL_SECTION_LGBM = """\
  lgbm:
    n_estimators: 300
    num_leaves: 31
    max_depth: -1
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8"""

_MODEL_SECTION_LR = """\
  lr:
    C: 1.0
    max_iter: 1000"""

_MODEL_SECTION_RF = """\
  rf:
    n_estimators: 300
    max_depth: null
    min_samples_leaf: 1"""

_MODEL_SECTION_TS = """\
  date_col: date
  val_frac: 0.2
  lgbm:
    n_estimators: 300
    num_leaves: 31
    max_depth: -1
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8"""

_MODEL_SECTION_GENERIC = """\
  # Uncomment the section for your chosen --template:
  # xgb:                # baseline-xgb / binary-cls / multiclass-cls / regression
  #   n_estimators: 300
  #   max_depth: 6
  #   learning_rate: 0.05
  # lgbm:               # baseline-lgbm
  #   n_estimators: 300
  #   num_leaves: 31
  #   max_depth: -1
  #   learning_rate: 0.05
  # lr:                 # baseline-lr
  #   C: 1.0
  #   max_iter: 1000
  # rf:                 # baseline-rf
  #   n_estimators: 300
  #   max_depth: null
  #   min_samples_leaf: 1"""


_MENU_YAML = """\
# Unified project manifest (menu.yaml) — the single source of truth the platform reads.
# It carries the run `pipeline`, each `recipe` (a stage's code or an infra resource), and the
# ML config. Drive the whole pipeline with `kitchen menu run`; every `kitchen` command also
# reads this file directly (no --params flag needed).

project: $name
region: us-east-1
experiment: $name

# The ordered run sequence. `train` runs features internally and promotes a champion;
# `evaluate` scores that champion. Add `provision` first (and switch mlflow below to a
# {from_role} reference) to stand up + wire AWS infra on the same run.
pipeline: [train, evaluate]

recipes:
  # --- stages (project code) ---
  features:
    kind: stage
    source: src/features/run.py
  train:
    kind: stage
    source: src/train/run.py
    args: ["--auto-promote"]   # register/compare a champion each run
  evaluate:
    kind: stage
    source: src/evaluate/run.py

  # --- infra (deploy on demand with `recipes apply menu.yaml`; not in the default pipeline) ---
  $name-data:
    kind: s3
    versioning: true
  $name-serve:
    kind: ecr
    lambda_access: true
  $name-lambda-role:
    kind: iam_role
    service: lambda.amazonaws.com
    policies:
      - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
  # Serving lambda — uncomment after pushing your first image to the ECR repo above:
  # $name-fn:
  #   kind: lambda
  #   iam_role: $name-lambda-role   # menu `role` is a discovery marker; the IAM role is `iam_role`
  #   ecr_repo: $name-serve
  #   source: src/serve/            # injected as the function's KITCHEN_PREDICTOR_DIR
  #   memory: 1024
  #   timeout: 30

data:
  source: local          # switch to "kaggle" once data is downloaded
  path: data/raw         # where raw CSVs live (relative to project root)
  raw_file: train.csv

features:
  raw_file: train.csv
  processed_file: features.parquet
  test_file: test.csv

model:
  target: label          # TODO: set to your actual target column name
  test_size: 0.2
  random_state: 42
$model_section

mlflow:
  tracking_uri: sqlite:///mlruns.db
  # For a persistent backend, define an `rds` recipe above and reference it by role:
  #   tracking_uri: {from_role: mlflow-backend}

run_name: baseline
metrics_file: metrics.json

thresholds:
  val_accuracy: 0.0       # TODO: set your primary metric + bound (drives leaderboard + --auto-promote)

# dashboard_url: https://<owner>.github.io/<repo>/  # set to open via `kitchen open`

# ci:                       # optional: CI behavior knobs (read by the scaffolded workflow)
#   fail_on_threshold: true # whether a threshold breach fails the CI job
#   notifications:
#     slack_webhook_secret: SLACK_WEBHOOK_URL  # GH secret holding the incoming-webhook URL
#     when: failure         # failure | success | always
"""


_MENU_YAML_KAGGLE = """\
# Unified project manifest (menu.yaml) — the single source of truth the platform reads.
# Drive the whole pipeline with `kitchen menu run`; every `kitchen` command also reads it.

project: $name
region: us-east-1
experiment: $name

pipeline: [train, evaluate]

recipes:
  # --- stages (project code) ---
  features:
    kind: stage
    source: src/features/run.py
  train:
    kind: stage
    source: src/train/run.py
    args: ["--auto-promote"]   # register/compare a champion each run
  evaluate:
    kind: stage
    source: src/evaluate/run.py

  # --- infra (deploy on demand with `recipes apply menu.yaml`; not in the default pipeline) ---
  $name-data:
    kind: s3
    versioning: true
  $name-serve:
    kind: ecr
    lambda_access: true
  $name-lambda-role:
    kind: iam_role
    service: lambda.amazonaws.com
    policies:
      - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
  # Serving lambda — uncomment after pushing your first image to the ECR repo above:
  # $name-fn:
  #   kind: lambda
  #   iam_role: $name-lambda-role
  #   ecr_repo: $name-serve
  #   source: src/serve/
  #   memory: 1024
  #   timeout: 30

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
  target: target        # TODO: set to this competition's target column (match submission.target_col)
  test_size: 0.2
  random_state: 42
$model_section

mlflow:
  tracking_uri: sqlite:///mlruns.db

run_name: baseline
metrics_file: metrics.json

thresholds:
  val_accuracy: 0.0       # TODO: set your primary metric + bound (drives leaderboard + --auto-promote)

# ci:                       # optional: CI behavior knobs (read by the scaffolded workflow)
#   auto_submit: false      # submit to Kaggle after evaluate on a main-branch push
#   fail_on_threshold: true # whether a threshold breach fails the CI job
#   notifications:
#     slack_webhook_secret: SLACK_WEBHOOK_URL  # GH secret holding the incoming-webhook URL
#     when: failure         # failure | success | always
"""


# --- `--kind pipeline` scaffold (GEN-007): a lean, non-tabular project ---------------------
# A command stage instead of the FeatureBuilder/Trainer/Evaluator ABCs — the fit for
# inference-only, non-tabular, or separate-interpreter pipelines.

_MENU_YAML_PIPELINE = """\
# Unified project manifest (menu.yaml). This is a `--kind pipeline` project: the work runs as a
# command stage (a subprocess), not the tabular train/evaluate ABCs. Drive it with
# `kitchen menu run`; run the single stage with `kitchen stage run`.

project: $name

pipeline: [run]

recipes:
  run:
    kind: stage
    cmd: python -m src.pipeline.run    # the subprocess to run (a list is the argv, verbatim)
    # python: .venv-pipeline/bin/python  # optional per-stage interpreter; `cmd:` is then its args
    # inputs: [data/raw]                 # checked to exist before the stage runs (fail fast)
    # outputs: [predictions.parquet]     # a missing declared output warns after

data:
  source: local
  path: data/raw

# Rank runs by this metric — your stage writes it to $$KITCHEN_METRICS_FILE (see src/pipeline/run.py).
# Sweep it with `kitchen sweep --run "python -m src.pipeline.run" --param ... --metric score`.
thresholds:
  score: 0.0
"""


_MENU_YAML_PIPELINE_KAGGLE = """\
# Unified project manifest (menu.yaml). This is a `--kind pipeline` project: the work runs as a
# command stage (a subprocess), not the tabular train/evaluate ABCs. `kitchen ingest` fetches the
# competition data; drive the pipeline with `kitchen menu run`.

project: $name

pipeline: [run]

recipes:
  run:
    kind: stage
    cmd: python -m src.pipeline.run    # the subprocess to run (a list is the argv, verbatim)
    # python: .venv-pipeline/bin/python  # optional per-stage interpreter; `cmd:` is then its args
    # inputs: [data/raw]                 # checked to exist before the stage runs (fail fast)
    # outputs: [submission.csv]

data:
  source: kaggle
  competition: $competition

# Rank runs by this metric — your stage writes it to $$KITCHEN_METRICS_FILE (see src/pipeline/run.py).
thresholds:
  score: 0.0
"""


_PIPELINE_RUN = """\
\"\"\"$name pipeline — a command stage (GEN-002/003).

Run the whole pipeline with ``kitchen menu run`` (or just this stage with ``kitchen stage run``).
Sweep it with, e.g.::

    kitchen sweep --run "python -m src.pipeline.run --thresh {t}" --param t=0.4,0.5,0.6 --metric score

Report your metric where kitchen looks for it: write a JSON object to the path in
``$$KITCHEN_METRICS_FILE`` (it falls back to ``metrics.json``). ``kitchen leaderboard --store`` and
``kitchen sweep`` then rank runs by it.

Alternative for pure-Python inference: declare a ``scorer:`` block in menu.yaml
(``source: src/pipeline/run.py``, ``function: score`` returning ``{name: value}``) and run
``kitchen score`` — see docs/kitchen/experiment-tracking.md.
\"\"\"
import json
import os


def main() -> None:
    # TODO: implement your pipeline (ingest → process → infer → score, in whatever shape fits).
    score = 0.0

    metrics_file = os.environ.get("KITCHEN_METRICS_FILE", "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"score": score}, f)
    print(f"score={score}")


if __name__ == "__main__":
    main()
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
from kitchen.modeling import (
    classification_metrics,
    compute_calibration_curve,
    train_val_split,
)
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
        metrics = classification_metrics(y_val, y_pred, y_proba=y_proba)
        # DASH-006: reliability-curve data — split out of metrics.json by
        # Evaluator.run() into calibration.json and surfaced on the dashboard.
        if y_proba is not None:
            metrics["calibration"] = compute_calibration_curve(y_val, y_proba)
        return metrics


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return ${class_name}Evaluator().run(model, store, params)
"""

_TRAIN_RUN_MULTICLASS_CLS = """\
\"\"\"Model training for $name — multiclass classification (XGBoost baseline).

Fits an XGBClassifier (multi:softprob) and logs validation metrics — including
val_balanced_accuracy, the metric to rank imbalanced multiclass on — to the active MLflow
run, which Trainer.run() opens before fit().

This baseline handles two things a plain XGBClassifier does not:

* Class imbalance: sample_weight uses sklearn's "balanced" weights so a majority class does
  not swamp training (without it the model collapses onto one class and val_balanced_accuracy
  sits at chance, 1/n_classes).
* Categorical features: enable_categorical=True consumes pandas `category` columns directly.
  In src/features/run.py, give each categorical a fixed category list so train and test encode
  identically, e.g. df[col] = pd.Categorical(df[col], categories=["a", "b", "c"]).

The target must be integer-encoded 0..n-1 in the features stage — XGBoost rejects string
labels. Encode with a fixed class order (reproducible) and decode predictions back in
flows/generate_submission.py:
    CLASSES = ["class_a", "class_b", "class_c"]                 # in src/features/run.py
    df[target] = df[target].map({c: i for i, c in enumerate(CLASSES)})
    labels = [CLASSES[i] for i in model.predict(X_test)]        # in generate_submission.py
\"\"\"
from __future__ import annotations

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

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
            enable_categorical=True,  # consume pandas `category` feature columns directly
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=seed,
            eval_metric="mlogloss",
        )
        # Balance the classes so val_balanced_accuracy reflects real per-class skill.
        sample_weight = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)

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
Reports accuracy, balanced_accuracy (mean per-class recall — the imbalanced-multiclass
metric), macro-f1, and macro roc_auc (one-vs-rest).
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

Params read from menu.yaml under model:
  target       — target column name (required)
  date_col     — column to sort by before splitting (omit if data is pre-sorted)
  val_frac     — fraction of rows reserved for validation, default 0.2
  random_state — random seed, default 42
  lgbm:        — LightGBM hyperparams (see commented section in menu.yaml)
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

Reserved environment variables — set by ``kitchen serve local`` / the loader; do
not reuse them for your own settings (use a project-specific name instead):
``KITCHEN_PREDICTOR_DIR`` (directory of this file), ``KITCHEN_MODEL_NAME``,
``KITCHEN_MODEL_VERSION``.

Optionally export ``MODEL_NAME`` / ``MODEL_VERSION`` (strings) to surface the
model identity on ``GET /metadata``.
\"\"\"
from __future__ import annotations

# ---------------------------------------------------------------------------
# Uncomment once your champion model is promoted to the registry.
# lazy_model defers the (slow) load to the first prediction instead of import
# time, so Lambda cold starts are faster; it loads once and caches thereafter.
# load_champion translates an unreachable-artifact failure (e.g. after migrating
# the tracking store from local SQLite to a remote server) into a clear error.
# ---------------------------------------------------------------------------
# from kitchen.serve import lazy_model, load_champion
# model = lazy_model(lambda: load_champion(\"models:/$name-model@champion\"))
# # model.predict(...) works transparently and triggers the load on first use.

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
# Optional: feature list + model identity — surfaced on GET /metadata so callers
# know which input keys the model expects and which model is serving.
# ---------------------------------------------------------------------------
# FEATURES: list[str] = [\"feature_a\", \"feature_b\"]
# MODEL_NAME = \"$name-model\"
# MODEL_VERSION = \"champion\"


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

from kitchen.store import DataStore
from kitchen.tracking import Tracker, configure_from_env, init_experiment

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "$name")
VARIANT = "baseline"


def run_variant(params: dict, variant: str) -> None:
    from src.features.run import build
    from src.train.run import train

    configure_from_env()
    init_experiment(EXPERIMENT)

    store = DataStore()
    tracker = Tracker(EXPERIMENT)

    with tracker.run(run_name=variant, params=params) as _run:
        mlflow.set_tag("model_variant", variant)
        build(params, store)
        train(params, store, tracker)
        print(f"{variant} run complete — see MLflow for val metrics")


if __name__ == "__main__":
    with open("menu.yaml") as f:
        params = yaml.safe_load(f)
    run_variant(params, VARIANT)
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

from experiments.baseline import run_variant

VARIANT = "challenger"


def challenger(params_file: str = "menu.yaml") -> None:
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


def generate(params_file: str = "menu.yaml") -> None:
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

# GitHub Environments: main-branch runs use `production`, all other runs (PRs) use
# `staging`. Configure required reviewers / env secrets in Settings → Environments.
# Two caveats:
#   - Adding required reviewers to `staging` makes PR jobs HANG awaiting approval.
#   - Fork PRs do not receive environment secrets — keep any secret you need on PRs
#     as a repo secret (secrets.*), not an environment secret.
jobs:
  train-evaluate:
    runs-on: ubuntu-latest
    environment:
      name: $${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
    permissions:
      contents: write
      pull-requests: write

    env:
      # Default: ephemeral per-run SQLite — the registry is empty each run, so champions
      # do NOT persist across runs (`--auto-promote` always sees no champion and promotes
      # unconditionally). For champions that carry across runs, deploy a persistent backend
      # (`recipes apply mlflow-tracking-backend.yaml` — RDS + S3), store the Postgres URL in
      # Secrets Manager, declare it in menu.yaml `secrets:` as MLFLOW_TRACKING_URI, then:
      # delete the line below, set MLFLOW_ARTIFACT_BUCKET, and uncomment the "Resolve
      # persistent MLflow backend" step. See docs/kitchen/configuration.md.
      MLFLOW_TRACKING_URI: sqlite:///mlruns.db
      # MLFLOW_ARTIFACT_BUCKET: my-project-mlflow-artifacts

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
        run: pip install rkoren-kitchen

      - name: Install project
        run: pip install -e ".[dev]"

      # Persistent backend only: resolve MLFLOW_TRACKING_URI from the secrets manifest into
      # the GitHub env file so the steps below use the RDS store (needs AWS creds — add an
      # aws-actions/configure-aws-credentials step with your OIDC role first). The `secrets`
      # command ships with kitchen; see docs/kitchen/configuration.md.
      # - name: Resolve persistent MLflow backend
      #   run: kitchen secrets export --name MLFLOW_TRACKING_URI

      - name: Train
        run: kitchen run train --auto-promote

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

      # Notifications (ci.notifications): uncomment to alert on job failure (e.g. a
      # threshold breach). Add the webhook URL as a repo secret and keep the name below
      # in sync with ci.notifications.slack_webhook_secret in menu.yaml.
      # - name: Notify on failure
      #   if: failure()
      #   env:
      #     SLACK_WEBHOOK_URL: $${{ secrets.SLACK_WEBHOOK_URL }}
      #   run: |
      #     [ -z "$$SLACK_WEBHOOK_URL" ] && exit 0
      #     curl -sf -X POST -H 'Content-type: application/json' \\
      #       --data "{\\"text\\":\\"$name CI failed on $${{ github.ref_name }} ($${{ github.sha }})\\"}" \\
      #       "$$SLACK_WEBHOOK_URL"

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
        run: pip install rkoren-kitchen

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

# GitHub Environments: main-branch runs use `production`, all other runs (PRs) use
# `staging`. Configure required reviewers / env secrets in Settings → Environments.
# Two caveats:
#   - Adding required reviewers to `staging` makes PR jobs HANG awaiting approval.
#   - Fork PRs do not receive environment secrets — keep KAGGLE_USERNAME / KAGGLE_KEY
#     as repo secrets (secrets.*), not environment secrets, if fork PRs must ingest.
jobs:
  train-evaluate:
    runs-on: ubuntu-latest
    environment:
      name: $${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
    permissions:
      contents: write
      pull-requests: write

    env:
      # Default: ephemeral per-run SQLite — the registry is empty each run, so champions
      # do NOT persist across runs (`--auto-promote` always sees no champion and promotes
      # unconditionally). For champions that carry across runs, deploy a persistent backend
      # (`recipes apply mlflow-tracking-backend.yaml` — RDS + S3), store the Postgres URL in
      # Secrets Manager, declare it in menu.yaml `secrets:` as MLFLOW_TRACKING_URI, then:
      # delete the line below, set MLFLOW_ARTIFACT_BUCKET, and uncomment the "Resolve
      # persistent MLflow backend" step. See docs/kitchen/configuration.md.
      MLFLOW_TRACKING_URI: sqlite:///mlruns.db
      # MLFLOW_ARTIFACT_BUCKET: my-project-mlflow-artifacts

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
        run: pip install rkoren-kitchen

      - name: Install project
        run: pip install -e ".[dev]"

      - name: Ingest data
        # Write ~/.kaggle/kaggle.json from the secrets rather than relying on
        # KAGGLE_USERNAME/KAGGLE_KEY env vars — current kaggle clients 401 on the
        # env-var auth path against api.kaggle.com, but authenticate fine via the file.
        run: |
          mkdir -p ~/.kaggle
          printf '{"username":"%s","key":"%s"}' '$${{ secrets.KAGGLE_USERNAME }}' '$${{ secrets.KAGGLE_KEY }}' > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json
          kitchen ingest

      # Persistent backend only: resolve MLFLOW_TRACKING_URI from the secrets manifest into
      # the GitHub env file so the steps below use the RDS store (needs AWS creds — add an
      # aws-actions/configure-aws-credentials step with your OIDC role first). The `secrets`
      # command ships with kitchen; see docs/kitchen/configuration.md.
      # - name: Resolve persistent MLflow backend
      #   run: kitchen secrets export --name MLFLOW_TRACKING_URI

      - name: Train
        run: kitchen run train --auto-promote

      - name: Evaluate
        run: kitchen run evaluate

      # Reads ci.auto_submit from menu.yaml so a main-branch push can submit
      # automatically, in addition to the manual workflow_dispatch toggle.
      - name: Read CI config
        id: ci
        run: |
          AUTO=$$(python -c "import yaml; print(str(yaml.safe_load(open('menu.yaml')).get('ci', {}).get('auto_submit', False)).lower())")
          echo "auto_submit=$$AUTO" >> $$GITHUB_OUTPUT

      - name: Submit to Kaggle
        if: $${{ inputs.submit || (steps.ci.outputs.auto_submit == 'true' && github.event_name == 'push' && github.ref == 'refs/heads/main') }}
        run: |
          mkdir -p ~/.kaggle
          printf '{"username":"%s","key":"%s"}' '$${{ secrets.KAGGLE_USERNAME }}' '$${{ secrets.KAGGLE_KEY }}' > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json
          kitchen submit --wait

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

      # Notifications (ci.notifications): uncomment to alert on job failure (e.g. a
      # threshold breach). Add the webhook URL as a repo secret and keep the name below
      # in sync with ci.notifications.slack_webhook_secret in menu.yaml.
      # - name: Notify on failure
      #   if: failure()
      #   env:
      #     SLACK_WEBHOOK_URL: $${{ secrets.SLACK_WEBHOOK_URL }}
      #   run: |
      #     [ -z "$$SLACK_WEBHOOK_URL" ] && exit 0
      #     curl -sf -X POST -H 'Content-type: application/json' \\
      #       --data "{\\"text\\":\\"$name CI failed on $${{ github.ref_name }} ($${{ github.sha }})\\"}" \\
      #       "$$SLACK_WEBHOOK_URL"

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
        run: pip install rkoren-kitchen

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
    h2 { font-size: 1.1rem; margin: 2rem 0 0.5rem; }
    #lb-wrap, #fold-wrap, #pcoord-wrap, #cal-wrap { max-width: 860px; margin-bottom: 2rem; }
    #pcoord-note, #cal-note { color: #6b7280; font-size: 0.8rem; margin: 0 0 0.5rem; }
    #fi-table td { text-align: center; font-variant-numeric: tabular-nums; }
    #fi-table td.fi-empty { background: #f8fafc; color: #cbd5e1; }
    #fi-table th:first-child, #fi-table td:first-child { text-align: left; font-weight: 500; }
  </style>
</head>
<body>
  <h1>__PROJECT_ESCAPED__ — Results Dashboard</h1>
  <p id="status">__STATUS_ESCAPED__</p>
  <div id="chart-wrap"><canvas id="chart"></canvas></div>
  <div id="lb-wrap" style="display:none"><h2>Submission history — LB score vs local metric</h2><canvas id="lb-chart"></canvas></div>
  <div id="fold-wrap" style="display:none"><h2>Per-fold metric breakdown</h2><canvas id="fold-chart"></canvas></div>
  <div id="pcoord-wrap" style="display:none"><h2>Parameter parallel coordinates</h2><p id="pcoord-note"></p><canvas id="pcoord-chart"></canvas></div>
  <div id="cal-wrap" style="display:none"><h2>Calibration (reliability) curve</h2><p id="cal-note">A perfectly calibrated model sits on the dashed diagonal.</p><canvas id="cal-chart"></canvas></div>
  <div class="scroll-wrap">
    <table id="runs-table"><thead id="thead"></thead><tbody id="tbody"></tbody></table>
  </div>
  <div id="fi-wrap" style="display:none"><h2>Feature importance across runs</h2><div id="fi-heatmap" class="scroll-wrap"></div></div>
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

    // DASH-007: submission history — lb_score (left) vs local primary metric (right).
    // Activates when at least 2 results carry an lb_score. Diverging lines (local
    // metric improving while LB flattens) signal overfitting the public test set.
    (function () {
      var withLb = RESULTS.filter(function (r) {
        return r.lb_score !== null && r.lb_score !== undefined;
      });
      if (withLb.length < 2) { return; }
      document.getElementById('lb-wrap').style.display = 'block';
      var labels = withLb.map(function (r) {
        return r.timestamp ? new Date(r.timestamp).toLocaleDateString() : (r.sha || '').slice(0, 8);
      });
      var datasets = [{
        label: 'LB score',
        data: withLb.map(function (r) { return r.lb_score; }),
        borderColor: '#16a34a',
        backgroundColor: 'rgba(22,163,74,0.08)',
        yAxisID: 'yLb', tension: 0.2, spanGaps: true, pointRadius: 4
      }];
      if (METRIC) {
        datasets.push({
          label: METRIC + ' (local)',
          data: withLb.map(function (r) {
            var v = (r.metrics || {})[METRIC];
            return (v === undefined) ? null : v;
          }),
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.08)',
          yAxisID: 'yMetric', tension: 0.2, spanGaps: true, pointRadius: 4
        });
      }
      new Chart(document.getElementById('lb-chart').getContext('2d'), {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
          responsive: true,
          plugins: { legend: { display: true } },
          scales: {
            yLb: { type: 'linear', position: 'left', title: { display: true, text: 'LB score' } },
            yMetric: {
              type: 'linear', position: 'right',
              title: { display: true, text: METRIC || 'local metric' },
              grid: { drawOnChartArea: false }
            }
          }
        }
      });
    }());

    // DASH-005: per-fold metric breakdown — grouped bar chart, one bar group per
    // fold (the {METRIC}_{fold} keys emitted by loto_cv / time_series_cv), one bar
    // per run. Activates when any per-fold key for the primary metric is present.
    (function () {
      if (!METRIC) { return; }
      var prefix = METRIC + '_';
      var foldSet = {};
      RESULTS.forEach(function (r) {
        Object.keys(r.metrics || {}).forEach(function (k) {
          if (k.indexOf(prefix) === 0) {
            var suffix = k.slice(prefix.length);
            if (suffix !== 'mean' && suffix !== 'std') { foldSet[suffix] = true; }
          }
        });
      });
      var folds = Object.keys(foldSet).sort();
      if (!folds.length) { return; }
      document.getElementById('fold-wrap').style.display = 'block';

      var runsWithFolds = RESULTS.filter(function (r) {
        return folds.some(function (f) { return (r.metrics || {})[prefix + f] !== undefined; });
      });
      var palette = ['#3b82f6', '#16a34a', '#eab308', '#dc2626', '#8b5cf6', '#0891b2', '#db2777', '#65a30d'];
      var datasets = runsWithFolds.map(function (r, i) {
        return {
          label: ((r.sha || '').slice(0, 7) || ('run ' + (i + 1))) + (r.champion ? ' [C]' : ''),
          data: folds.map(function (f) {
            var v = (r.metrics || {})[prefix + f];
            return (v === undefined) ? null : v;
          }),
          backgroundColor: palette[i % palette.length]
        };
      });
      new Chart(document.getElementById('fold-chart').getContext('2d'), {
        type: 'bar',
        data: { labels: folds, datasets: datasets },
        options: {
          responsive: true,
          plugins: { legend: { display: true } },
          scales: {
            y: { title: { display: true, text: METRIC } },
            x: { title: { display: true, text: 'fold / period' } }
          }
        }
      });
    }());

    // DASH-004: feature importance heatmap. Rows = union of the top features across
    // runs (max 20), columns = runs by date, cell shade = importance normalised within
    // each run. Activates when any result includes a top_features list (LML-010).
    (function () {
      var N = 20;
      var withFi = RESULTS.filter(function (r) {
        return Array.isArray(r.top_features) && r.top_features.length;
      });
      if (!withFi.length) { return; }
      document.getElementById('fi-wrap').style.display = 'block';

      var best = {};
      withFi.forEach(function (r) {
        r.top_features.slice(0, N).forEach(function (f) {
          if (best[f.name] === undefined || f.importance > best[f.name]) {
            best[f.name] = f.importance;
          }
        });
      });
      var feats = Object.keys(best).sort(function (a, b) { return best[b] - best[a]; }).slice(0, N);

      var head = '<tr><th>feature</th>';
      withFi.forEach(function (r) {
        head += '<th>' + ((r.sha || '').slice(0, 7) || '—') + '</th>';
      });
      head += '</tr>';

      var body = '';
      feats.forEach(function (name) {
        body += '<tr><td>' + name + '</td>';
        withFi.forEach(function (r) {
          var imp = null;
          var maxImp = 0;
          r.top_features.forEach(function (f) {
            if (f.importance > maxImp) { maxImp = f.importance; }
            if (f.name === name) { imp = f.importance; }
          });
          if (imp === null) {
            body += '<td class="fi-empty">·</td>';
          } else {
            var intensity = maxImp > 0 ? imp / maxImp : 0;
            var alpha = (0.10 + 0.90 * intensity).toFixed(2);
            body += '<td style="background: rgba(59,130,246,' + alpha + ')" title="' +
              imp.toFixed(4) + '">' + imp.toFixed(3) + '</td>';
          }
        });
        body += '</tr>';
      });
      document.getElementById('fi-heatmap').innerHTML =
        '<table id="fi-table"><thead>' + head + '</thead><tbody>' + body + '</tbody></table>';
    }());

    // DASH-008: parameter parallel coordinates. One axis per numeric param, one
    // polyline per run, coloured by the primary metric (greener = higher). Each
    // axis is independently min-max normalised to [0, 1]; reveals which param
    // combinations cluster around good outcomes. Activates with >=2 runs carrying
    // params and >=2 numeric axes. MLflow params arrive as strings, so every
    // candidate is parsed with parseFloat and non-numeric keys are dropped.
    (function () {
      var withParams = RESULTS.filter(function (r) {
        return r.params && typeof r.params === 'object';
      });
      if (withParams.length < 2) { return; }

      var numericByKey = {};
      withParams.forEach(function (r) {
        Object.keys(r.params).forEach(function (k) {
          var v = parseFloat(r.params[k]);
          if (!isNaN(v)) {
            (numericByKey[k] = numericByKey[k] || []).push(v);
          }
        });
      });
      var axes = Object.keys(numericByKey).filter(function (k) {
        return numericByKey[k].length >= 2;
      }).sort();
      if (axes.length < 2) { return; }

      var bounds = {};
      axes.forEach(function (k) {
        var vals = numericByKey[k];
        bounds[k] = { min: Math.min.apply(null, vals), max: Math.max.apply(null, vals) };
      });
      function norm(k, raw) {
        var b = bounds[k];
        if (b.max === b.min) { return 0.5; }  // constant axis → centre line
        return (raw - b.min) / (b.max - b.min);
      }

      var mvals = withParams.map(function (r) {
        return METRIC ? (r.metrics || {})[METRIC] : null;
      }).filter(function (v) { return v !== undefined && v !== null; });
      var mMin = mvals.length ? Math.min.apply(null, mvals) : 0;
      var mMax = mvals.length ? Math.max.apply(null, mvals) : 1;
      function metricColour(r) {
        var v = METRIC ? (r.metrics || {})[METRIC] : null;
        if (v === undefined || v === null || mMax === mMin) { return '#94a3b8'; }
        var t = (v - mMin) / (mMax - mMin);  // 0 = lowest, 1 = highest
        return 'hsl(' + Math.round(t * 120) + ', 70%, 45%)';  // red → green
      }

      document.getElementById('pcoord-wrap').style.display = 'block';
      document.getElementById('pcoord-note').textContent =
        'Axes independently normalised 0–1' + (METRIC ? '; greener = higher ' + METRIC : '') + '.';

      var datasets = withParams.map(function (r) {
        return {
          label: (r.sha || '').slice(0, 7) + (r.champion ? ' [C]' : ''),
          data: axes.map(function (k) {
            var raw = parseFloat(r.params[k]);
            return isNaN(raw) ? null : norm(k, raw);
          }),
          borderColor: metricColour(r),
          backgroundColor: metricColour(r),
          spanGaps: true,
          tension: 0,
          pointRadius: 3
        };
      });
      new Chart(document.getElementById('pcoord-chart').getContext('2d'), {
        type: 'line',
        data: { labels: axes, datasets: datasets },
        options: {
          responsive: true,
          plugins: { legend: { display: true } },
          scales: {
            y: { min: 0, max: 1, title: { display: true, text: 'normalised value' } },
            x: { title: { display: true, text: 'parameter' } }
          }
        }
      });
    }());

    // DASH-006: calibration (reliability) curve. Plots observed positive rate
    // against mean predicted probability for the champion and most-recent runs,
    // with a dashed diagonal as the perfectly-calibrated reference. Activates when
    // any result carries a `calibration` list (DASH-006 / LML-010).
    (function () {
      function curveOf(r) {
        return Array.isArray(r.calibration) && r.calibration.length ? r.calibration : null;
      }
      var withCal = RESULTS.filter(curveOf);
      if (!withCal.length) { return; }
      document.getElementById('cal-wrap').style.display = 'block';

      var champ = withCal.find(function (r) { return r.champion; });
      var recent = withCal[withCal.length - 1];
      var picks = [];
      if (champ) { picks.push({ run: champ, label: 'champion', colour: '#eab308' }); }
      if (recent && recent !== champ) {
        picks.push({ run: recent, label: 'most recent', colour: '#3b82f6' });
      }
      if (!picks.length) {
        picks.push({ run: recent, label: 'latest', colour: '#3b82f6' });
      }

      var datasets = picks.map(function (p) {
        return {
          label: p.label + ' (' + (p.run.sha || '').slice(0, 7) + ')',
          data: curveOf(p.run).map(function (b) {
            return { x: b.bin_center, y: b.fraction_positive };
          }),
          borderColor: p.colour,
          backgroundColor: p.colour,
          tension: 0.2,
          pointRadius: 4
        };
      });
      datasets.push({
        label: 'perfectly calibrated',
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: '#9ca3af',
        borderDash: [6, 4],
        pointRadius: 0,
        fill: false
      });
      new Chart(document.getElementById('cal-chart').getContext('2d'), {
        type: 'line',
        data: { datasets: datasets },
        options: {
          responsive: true,
          plugins: { legend: { display: true } },
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'mean predicted probability' } },
            y: { min: 0, max: 1, title: { display: true, text: 'observed positive rate' } }
          }
        }
      });
    }());
  </script>
</body>
</html>
"""



# ---------------------------------------------------------------------------
# DVC scaffold templates (--with-dvc)
# ---------------------------------------------------------------------------

_DVC_YAML = """\
# DVC pipeline for $name.
# Run `dvc repro` to execute stages in dependency order, skipping unchanged ones.
#
# First-time setup:
#   pip install rkoren-kitchen[dvc]
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
#   pip install rkoren-kitchen[dvc]
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
# Exploration notebook (NB-009) — scaffolded into kitchen init as
# notebooks/exploration.ipynb. Built programmatically (not via string.Template)
# so notebook code containing `$` or `{}` never collides with substitution.
# Placeholders __NAME__ / __CLASS__ are replaced with the project slug and class.
# ---------------------------------------------------------------------------

_NB_INTRO = """\
# Exploration — __NAME__

Notebook-first iteration on this project, using the same MLflow tracking the CLI uses.

| Step | What | API |
|------|------|-----|
| 1 | Peek at processed features | `DataStore.preview()` |
| 2 | Try a quick idea inline | `kitchen.experiment(exploratory=True)` |
| 3 | Run a Trainer with tracking | `kitchen.init_run(exploratory=True, log_model=False)` |
| 4 | Compare runs | `kitchen leaderboard` |

**Prerequisite:** run `kitchen run features` first so `data/processed/` exists.

Exploratory runs are tagged `run_type=exploratory`, so they stay separable from
pipeline runs: `kitchen leaderboard --exclude-exploratory` hides them and
`--only-exploratory` shows just these notebook sketches."""

_NB_SETUP = """\
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import kitchen
from kitchen.store import DataStore

with open("menu.yaml") as f:
    params = yaml.safe_load(f)

EXPERIMENT = params.get("experiment", "__NAME__")
TARGET = params.get("model", {}).get("target", "target")
PROCESSED_FILE = params.get("features", {}).get("processed_file", "features.parquet")

store = DataStore()
print(f"experiment={EXPERIMENT}  target={TARGET}  features={PROCESSED_FILE}")"""

_NB_STEP1 = """\
## Step 1 — Peek at the processed features

`preview()` searches `processed/` first, then `raw/`, and returns the first rows —
just the filename, no path juggling."""

_NB_PREVIEW = """store.preview(PROCESSED_FILE)"""

_NB_STEP2 = """\
## Step 2 — Try a quick idea with `kitchen.experiment()`

Write model code directly in the cell — no `Trainer` subclass needed. `exploratory=True`
tags the run so it stays out of the default leaderboard ranking.

**Metric naming matters:** `kitchen leaderboard` ranks by the metric in your
`thresholds:` section. Log under that exact name, or the run won't appear in the
default view (`kitchen leaderboard --metric <name>` still finds it)."""

_NB_EXPERIMENT = """\
df = store.load_parquet(PROCESSED_FILE)
if TARGET not in df.columns:
    raise ValueError(
        f"target column {TARGET!r} is not in the features ({list(df.columns)}). "
        "Set model.target in menu.yaml (or change TARGET in the setup cell)."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

with kitchen.experiment(EXPERIMENT, run_name="nb-logreg", exploratory=True) as run:
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))
    run.log(val_accuracy=acc)

print(f"val_accuracy: {acc:.4f}  (run {run.run_id})")"""

_NB_STEP3 = """\
## Step 3 — Run a Trainer with `kitchen.init_run()`

`init_run()` injects the same MLflow context that `kitchen run train` uses, so you can
iterate on a `Trainer` in the notebook. `log_model=False` keeps these throwaway runs out
of the model registry; `exploratory=True` tags them.

The `SimpleTrainer` below is a stand-in so this cell runs on a fresh project. In your
project, swap it for your own: `from src.train.run import __CLASS__Trainer`."""

_NB_TRAINER = """\
import mlflow

from kitchen.steps import Trainer


class SimpleTrainer(Trainer):
    \"\"\"Stand-in Trainer — replace with __CLASS__Trainer from src/train/run.py.\"\"\"

    def fit(self, df, params):
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=300)
        model.fit(X_tr, y_tr)
        mlflow.log_metric("val_accuracy", accuracy_score(y_vl, model.predict(X_vl)))
        return model


with kitchen.init_run(params, run_name="nb-trainer", exploratory=True, log_model=False) as tracker:
    SimpleTrainer().run(store, tracker, params)

print("Logged — see it with: kitchen leaderboard --only-exploratory")"""

_NB_STEP4 = """\
## Step 4 — Compare runs

From a terminal in this project directory:

```bash
kitchen leaderboard                       # ranked runs
kitchen leaderboard --only-exploratory    # just these notebook sketches
kitchen leaderboard --exclude-exploratory # just pipeline runs
kitchen diff <run_a> <run_b>              # what changed between two runs
kitchen ui                                # open the MLflow UI
```

Found a keeper? Promote it: `kitchen promote --run-id <run_id>`."""


def _build_exploration_notebook(name: str, class_name: str) -> str:
    """Return a project-specific exploration notebook (NB-009) as ipynb JSON text."""
    import json

    def _sub(text: str) -> str:
        return text.replace("__NAME__", name).replace("__CLASS__", class_name)

    def _cell(cell_id: str, cell_type: str, text: str) -> dict:
        cell = {
            "id": cell_id,
            "cell_type": cell_type,
            "metadata": {},
            "source": _sub(text).splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return cell

    cells = [
        _cell("nb-intro", "markdown", _NB_INTRO),
        _cell("nb-setup", "code", _NB_SETUP),
        _cell("nb-step1", "markdown", _NB_STEP1),
        _cell("nb-preview", "code", _NB_PREVIEW),
        _cell("nb-step2", "markdown", _NB_STEP2),
        _cell("nb-experiment", "code", _NB_EXPERIMENT),
        _cell("nb-step3", "markdown", _NB_STEP3),
        _cell("nb-trainer", "code", _NB_TRAINER),
        _cell("nb-step4", "markdown", _NB_STEP4),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"
