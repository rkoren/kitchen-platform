# Spaceship Titanic — end-to-end showcase

The whole `kitchen` loop on a real Kaggle problem, in one small project: **features → train →
evaluate → promote a champion**, driven by a single `menu.yaml` and run with one command. It
runs **offline with zero credentials** — the bundled `data/raw/train.csv` is a small *synthetic*
sample that mirrors the real [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
schema, so you can see the platform work without a Kaggle download.

## Run it

```bash
cd examples/spaceship-titanic
kitchen menu run
```

That executes the `pipeline` in `menu.yaml`:

1. **train** (`src/train/run.py`) — runs feature engineering internally, fits an `XGBClassifier`,
   logs `val_accuracy`/`val_f1`/`val_roc_auc` to MLflow, and (via `--auto-promote`) registers the
   model and promotes it to `champion` if it beats the current one.
2. **evaluate** (`src/evaluate/run.py`) — loads `models:/spaceship-titanic-model@champion` and
   scores it on the held-out split.

Expected: `val_accuracy ≈ 0.83` (f1 ≈ 0.83, roc_auc ≈ 0.88) — a genuine PASS against the `0.78`
threshold in `menu.yaml`, right where real Spaceship Titanic baselines land (~0.80). The
synthetic data is generated with a realistic signal strength (see `data/make_sample.py`), so
the feature engineering and tuning actually matter — the engineered **`has_spent`** feature
(awake-and-spending, which cryosleep rules out) turns out to be the single biggest driver.

Run individual stages too:

```bash
kitchen run features                       # build data/processed/features.parquet
kitchen run train --override model.max_depth=6   # one-off hyperparameter
kitchen leaderboard                        # compare runs; [C] marks the champion
kitchen ui                                 # open the MLflow UI
```

Everything under `mlruns*`, `metrics.json`, and `data/processed/` is regenerated on each run and
gitignored.

## What's in here

| File | Role |
|------|------|
| `menu.yaml` | The whole project as one manifest — pipeline, stage sources, model knobs, MLflow, thresholds. |
| `src/features/run.py` | Real SST feature engineering — split `Cabin` into deck/side, `total_spend` + `has_spent` + `luxury`, `group_size` from the `PassengerId`, fill missing values, integer-encode categoricals. |
| `src/train/run.py` | An XGBoost baseline (`model_flavour = "xgboost"`); tune it in `menu.yaml` or with `--override`. |
| `src/evaluate/run.py` | Scores the champion on the same held-out split. |
| `data/make_sample.py` | The deterministic generator for the synthetic `train.csv` (seeded → reproducible). |

## Using the real competition data

Swap the synthetic sample for the real thing:

```bash
kitchen init my-spaceship --source kaggle --competition spaceship-titanic --template baseline-xgb
```

That scaffolds the same structure and pulls the real data via the Kaggle API (accept the
competition rules first). The stage code here works unchanged on the real columns — the synthetic
sample was built to match them.
