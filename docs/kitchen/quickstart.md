# Quickstart

> **Live example:** [rkoren/spaceship-titanic](https://github.com/rkoren/spaceship-titanic) — a complete kitchen-platform competition project with CI, GitHub Pages dashboard, and MLflow tracking.

## Prerequisites

- Python 3.11+
- [Kaggle API credentials](https://www.kaggle.com/settings) — download `kaggle.json` from **Account → API**
- AWS CLI configured (`aws configure`) — only needed if using S3 artifacts or remote MLflow

## Install

```bash
pip install rkoren-kitchen
```

For a local development install from the repo:

```bash
pip install -e path/to/kitchen-platform/kitchen
```

## Scaffold a new competition project

```bash
kitchen init spaceship-titanic \
  --source kaggle \
  --competition spaceship-titanic \
  --template baseline-xgb \
  --ci
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `local` | Data source: `local`, `kaggle`, or `s3` |
| `--competition` | — | Kaggle competition slug (required when `--source kaggle`) |
| `--template` | `none` | Starter model: `none`, `baseline-xgb`, `baseline-lr` |
| `--ci` | off | Scaffold `.github/workflows/train-evaluate.yml` |
| `--here` | off | Scaffold into the current directory instead of a new subdirectory |

## Set up credentials

```bash
cd spaceship-titanic
cp .env.example .env
# Edit .env and add:
#   KAGGLE_USERNAME=your-username
#   KAGGLE_KEY=your-api-key
```

Run the pre-flight check to confirm tools, credentials, and config are all wired up:

```bash
kitchen check
```

## Download competition data

```bash
kitchen ingest
```

Data is downloaded to `data/raw/` as configured in `params.yaml`.

## Implement the three required files

| File | What to implement |
|---|---|
| `src/features/run.py` | `build(raw_df) -> df` — feature engineering |
| `src/train/run.py` | `fit(df, params) -> model` — model training |
| `src/evaluate/run.py` | `evaluate(model, df) -> dict` — metrics |

The scaffold includes a working stub for each file. If you passed `--template baseline-xgb` or `--template baseline-lr`, `src/train/run.py` already has a runnable baseline.

## Run experiments

```bash
# Train — builds features, fits the model, logs everything to MLflow
kitchen run train

# Evaluate — loads the champion model, computes metrics, writes metrics.json
kitchen run evaluate

# Inspect runs
kitchen experiments compare val_accuracy

# Promote the best run to the model registry
kitchen promote val_accuracy

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
# Open http://localhost:5000
```

## Generate a Kaggle submission

```bash
kitchen submit
```

This validates the submission file (columns, row count, nulls, duplicate IDs) before uploading. Pass `--wait` to poll for the public leaderboard score after upload.

## View the report

```bash
# Print a metrics summary to stdout
kitchen report

# GitHub-flavored markdown (piped to Actions job summary in CI)
kitchen report --format github

# Compare current metrics against a previous run's metrics.json
kitchen report --compare path/to/base/metrics.json
```
