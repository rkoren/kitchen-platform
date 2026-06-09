# Offline quickstart

A complete, runnable `kitchen` project that trains, evaluates, and promotes a champion
model **with zero credentials and zero network** — no Kaggle, no AWS, no DVC. Data is a
tiny synthetic CSV bundled in `data/raw/`, and MLflow tracking is a local SQLite file
created on first run.

It's the smallest honest version of the full loop: the same `FeatureBuilder` /
`Trainer` / `Evaluator` contract and the same CLI commands a real competition project
uses, just with toy data so it runs in seconds.

## The data

`data/raw/train.csv` — 240 synthetic rows of a "did the student pass?" classification
toy: four numeric features (`study_hours`, `prior_score`, `attendance`, `sleep_hours`)
and a binary `passed` target. It's generated data committed as a fixture, not raw
competition data, so it lives in the repo by design.

## Run it

From this directory, with `kitchen` installed (`pip install -e kitchen/` from the repo
root):

```bash
cd examples/offline-quickstart

kitchen run train --auto-promote   # features → train → log to MLflow → promote champion
kitchen run evaluate               # load champion, score on the held-out split
kitchen leaderboard                # rank runs; [C] marks the promoted champion
```

`kitchen run train` runs the feature step first, so there's no separate
`kitchen run features` call needed.

## What you should see

Training registers a champion (no prior champion exists, so the first run wins):

```
auto-promote: metric=val_accuracy (higher=better)
auto-promote: <run_id> → champion  (no current champion)
             offline-quickstart-model v1 @ champion
```

Evaluation scores the champion on the validation split (~0.79 accuracy, ~0.91 ROC AUC):

```
Evaluation results (models:/offline-quickstart-model@champion):
  accuracy: 0.79
  f1: 0.78
  log_loss: 0.37
  roc_auc: 0.91
```

And the leaderboard flags the champion row:

```
#     RUN ID        VARIANT    val_accuracy   lb_score  STARTED
★[C]  <run_id>      baseline         0.7917          -  …
[C] = current champion
```

## What's committed vs. generated

Committed: `params.yaml`, `src/`, and `data/raw/train.csv`. Everything the pipeline
produces — `mlruns.db`, the `mlruns/` artifact store, `data/processed/features.parquet`,
`metrics.json`, `calibration.json` — is regenerated on each run and gitignored. Delete
those any time to start clean.
