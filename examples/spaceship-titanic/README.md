# Spaceship Titanic — end-to-end showcase

The whole `kitchen` loop on a real Kaggle competition, in one small project: **ingest → features →
train → evaluate → promote a champion → submit**, driven by a single `menu.yaml`.

The target is [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) — Kaggle's
permanent *Getting Started* sandbox (no deadline, **unlimited submissions, no stakes**), which makes
it the ideal end-to-end proof of concept: you can run the real thing, submit for real, and see a
public leaderboard score, all for free.

## Prerequisites

The competition data isn't committed here (it's Kaggle's to distribute, and downloading it requires
accepting the rules). You need a Kaggle account and API credentials:

1. Accept the rules on the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/rules).
2. Create an API token (Kaggle → Account → *Create New API Token*) and either save it to
   `~/.kaggle/kaggle.json` or export `KAGGLE_USERNAME` + `KAGGLE_KEY`.

Everything runs under **your own** Kaggle account — nothing here is tied to anyone else's.

## Run it

```bash
cd examples/spaceship-titanic
kitchen ingest        # pull train.csv / test.csv / sample_submission.csv → data/raw/
kitchen menu run      # features → train → evaluate → promote a champion
```

Or from anywhere (the repo root, say) — every command takes `-C` to run from the project
directory, like `git -C`:

```bash
kitchen ingest   -C examples/spaceship-titanic
kitchen menu run -C examples/spaceship-titanic
```

`kitchen menu run` executes the `pipeline` in `menu.yaml`:

1. **train** (`src/train/run.py`) — runs feature engineering internally, fits an `XGBClassifier`,
   logs `val_accuracy`/`val_f1`/`val_roc_auc` to MLflow, and (via `--auto-promote`) registers the
   model and promotes it to `champion` if it beats the current one.
2. **evaluate** (`src/evaluate/run.py`) — loads `models:/spaceship-titanic-model@champion` and
   scores it on the held-out split.

Expected: **`val_accuracy ≈ 0.81`** (f1 ≈ 0.82, roc_auc ≈ 0.90) on the local holdout — a genuine
PASS against the `0.78` threshold in `menu.yaml`, right where solid Spaceship Titanic baselines
land. The engineered **`has_spent`** feature (awake-and-spending, which cryosleep rules out) turns
out to be one of the biggest drivers. (The public leaderboard score is a touch lower than the local
holdout, as usual — submit to see it.)

Run individual stages too:

```bash
kitchen run features                       # build data/processed/features.parquet
kitchen run train --override model.max_depth=6   # one-off hyperparameter
kitchen leaderboard                        # compare runs; [C] marks the champion
kitchen ui                                 # open the MLflow UI
```

Everything under `data/raw`, `data/processed`, `mlruns*`, `metrics.json`, and `submissions/` is
regenerated or fetched on demand and is **gitignored** — no data or submissions are committed.

## Generating and submitting a Kaggle submission

The pipeline promotes a champion; `flows/generate_submission.py` turns its predictions on the
competition's `test.csv` into a submission CSV and validates it — the same `validate_submission`
check `kitchen submit` runs before uploading:

```bash
python examples/spaceship-titanic/flows/generate_submission.py
```

It writes `submissions/submission.csv` (`PassengerId,Transported`) and validates it against
`sample_submission.csv` (row count, columns, no nulls, no duplicate IDs). Preview it first with
`--dry-run` (validates and reports, no credentials, no upload), then upload it to the sandbox
(add `-C examples/spaceship-titanic` to run from elsewhere):

```bash
kitchen submit --file submissions/submission.csv --dry-run  # validate + preview, no upload
kitchen submit --file submissions/submission.csv            # uploads under your Kaggle account
kitchen submit --file submissions/submission.csv --wait     # also poll for the public LB score
```

`submission:` in `menu.yaml` supplies the competition slug, ID/target columns, and message.

`generate_submission.py` shares one `_engineer()` with `src/features/run.py`, so training and
inference apply identical feature engineering (the surest guard against train/serve skew). One
honest simplification worth knowing: the categorical encoders are fit per-frame rather than
persisted from training — fine here because train and test share the same categories, but a
production project should save the fitted encoders and re-apply them at inference.

## What's in here

| File | Role |
|------|------|
| `menu.yaml` | The whole project as one manifest — data source, pipeline, stage sources, model knobs, MLflow, thresholds, submission config. |
| `src/features/run.py` | SST feature engineering — split `Cabin` into deck/side, `total_spend` + `has_spent` + `luxury`, `group_size` from the `PassengerId`, fill missing values, integer-encode categoricals. |
| `src/train/run.py` | An XGBoost baseline (`model_flavour = "xgboost"`); tune it in `menu.yaml` or with `--override`. |
| `src/evaluate/run.py` | Scores the champion on the same held-out split. |
| `flows/generate_submission.py` | Predicts the champion on `test.csv` → `submissions/submission.csv`, then validates it. |

## Starting your own from scratch

This example is already wired up; to scaffold a *new* Kaggle project with the same structure:

```bash
kitchen init my-spaceship --source kaggle --competition spaceship-titanic --template baseline-xgb
```

That generates the same layout (with feature/submission TODOs to fill in) and the same
`kitchen ingest → menu run → generate_submission → submit` flow.
