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

Or from anywhere (the repo root, say), point at the project with `-C`:

```bash
kitchen menu run -C examples/spaceship-titanic
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

Everything under `mlruns*`, `metrics.json`, `submissions/`, and `data/processed/` is regenerated
on each run and gitignored.

## Generating a Kaggle submission

The pipeline promotes a champion; `flows/generate_submission.py` turns its predictions on the
held-out `data/raw/test.csv` into a submission CSV and validates it — the same
`validate_submission` check `kitchen submit` runs before uploading:

```bash
kitchen menu run -C examples/spaceship-titanic          # train + promote a champion first
python examples/spaceship-titanic/flows/generate_submission.py
```

It writes `submissions/submission.csv` (`PassengerId,Transported`), validates it against
`sample_submission.csv` (row count, columns, no nulls, no duplicate IDs), and **stops before
uploading**.

> **The synthetic bundle can't be submitted for real.** `test.csv` uses fabricated
> `PassengerId`s, so the CSV is well-formed but Kaggle would reject it. To exercise the actual
> upload, fetch the real competition data first (below), then `kitchen submit --file
> submissions/submission.csv` from the project directory. `submission:` in `menu.yaml` supplies
> the competition slug, ID/target columns, and message.

`generate_submission.py` shares one `_engineer()` with `src/features/run.py`, so training and
inference apply identical feature engineering (the surest guard against train/serve skew). One
honest simplification: the categorical encoders are fit per-frame rather than persisted from
training — fine for this showcase's aligned synthetic data, but a real project should save the
fitted encoders and re-apply them at inference.

## What's in here

| File | Role |
|------|------|
| `menu.yaml` | The whole project as one manifest — pipeline, stage sources, model knobs, MLflow, thresholds. |
| `src/features/run.py` | Real SST feature engineering — split `Cabin` into deck/side, `total_spend` + `has_spent` + `luxury`, `group_size` from the `PassengerId`, fill missing values, integer-encode categoricals. |
| `src/train/run.py` | An XGBoost baseline (`model_flavour = "xgboost"`); tune it in `menu.yaml` or with `--override`. |
| `src/evaluate/run.py` | Scores the champion on the same held-out split. |
| `flows/generate_submission.py` | Predicts the champion on `test.csv` → `submissions/submission.csv`, validates it, stops before upload. |
| `data/make_sample.py` | The deterministic generator for the synthetic `train.csv`, `test.csv`, and `sample_submission.csv` (seeded → reproducible). |

## Using the real competition data

Swap the synthetic sample for the real thing:

```bash
kitchen init my-spaceship --source kaggle --competition spaceship-titanic --template baseline-xgb
```

That scaffolds the same structure and pulls the real data via the Kaggle API (accept the
competition rules first). The stage code here works unchanged on the real columns — the synthetic
sample was built to match them.
