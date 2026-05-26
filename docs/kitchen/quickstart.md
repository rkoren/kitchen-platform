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
| `--template` | `none` | Starter model: `none`, `baseline-xgb`, `baseline-lr`, `baseline-rf`, `binary-cls`, `multiclass-cls`, `regression` |
| `--ci` | off | Scaffold `.github/workflows/train-evaluate.yml` |
| `--with-dvc` | off | Scaffold `dvc.yaml`, `.dvcignore`, `.dvc/config` and run `dvc init` (requires `pip install kitchen[dvc]`) |
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

| File | Function signature |
|---|---|
| `src/features/run.py` | `build(params: dict, store: DataStore) -> None` |
| `src/train/run.py` | `train(params: dict, store: DataStore, tracker: Tracker) -> model` |
| `src/evaluate/run.py` | `evaluate(model, params: dict, store: DataStore) -> dict[str, float]` |

The scaffold includes a working stub for each file. If you passed `--template baseline-xgb`, `--template baseline-lr`, or another template, `src/train/run.py` already has a runnable baseline.

## Run experiments

```bash
# Build features — raw data → data/processed/
kitchen run features

# Train — builds features, fits the model, logs everything to MLflow
kitchen run train

# Train and auto-promote if this run beats the current champion
kitchen run train --auto-promote --promote-metric val_accuracy

# Evaluate — loads the champion model, computes metrics, writes metrics.json
kitchen run evaluate
```

## Inspect runs and manage the champion

```bash
# One-screen project summary: champion metrics, last 5 runs, threshold pass/fail
kitchen status

# Rank all runs by metric; [C] = registered champion, ★ = metric leader
kitchen leaderboard

# Manually promote the best run to the model registry
kitchen promote val_accuracy

# Open the MLflow UI in your browser (starts a local server for SQLite tracking)
kitchen ui
```

## Generate a Kaggle submission

```bash
kitchen submit
```

This validates the submission file (columns, row count, nulls, duplicate IDs) before uploading. Pass `--wait` to poll for the public leaderboard score after upload.

## Publish results and view the dashboard

```bash
# Publish current run metrics to the results branch as results/<sha>.json
kitchen push

# Open the GitHub Pages dashboard in your browser
kitchen open
```

`kitchen push` can be run locally after any training run — no CI trigger required. The dashboard updates as results accumulate.

## View the report

```bash
# Print a metrics summary to stdout
kitchen report

# GitHub-flavored markdown (piped to Actions job summary in CI)
kitchen report --format github

# Compare current metrics against a previous run's metrics.json
kitchen report --compare path/to/base/metrics.json
```
