# Kaggle Competition Quickstart

A complete walkthrough of the Kaggle workflow: scaffold → ingest → train → evaluate → submit.

## Prerequisites

- Python 3.11+
- Kaggle API credentials — download `kaggle.json` from [kaggle.com](https://www.kaggle.com/settings) under **Account → API**
- `rkoren-kitchen` installed: `pip install rkoren-kitchen`

## 1. Scaffold the project

```bash
kitchen init spaceship-titanic \
  --source kaggle \
  --competition spaceship-titanic \
  --template baseline-xgb \
  --ci
```

`--template baseline-xgb` gives you a runnable XGBoost model from the start. `--ci` scaffolds `.github/workflows/train-evaluate.yml` so every push trains and evaluates automatically.

```
cd spaceship-titanic
pip install -e .
```

## 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```bash
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

Confirm everything is wired up:

```bash
kitchen check
# ✓ tools: python, kaggle
# ✓ credentials: KAGGLE_USERNAME, KAGGLE_KEY
# ✓ params.yaml found
```

## 3. Download competition data

```bash
kitchen ingest
# Downloading spaceship-titanic.zip → data/raw/
# ✓ train.csv
# ✓ test.csv
# ✓ sample_submission.csv
```

Data lands in `data/raw/` and is gitignored by default.

## 4. Implement the three required files

The scaffold generates stubs for each file. Fill them in for your competition:

**`src/features/run.py`** — feature engineering

```python
def build(raw_df):
    df = raw_df.copy()
    # e.g. df["CabinDeck"] = df["Cabin"].str.split("/").str[0]
    return df
```

**`src/train/run.py`** — already populated if you used `--template baseline-xgb`

```python
import xgboost as xgb

def fit(df, params):
    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(int)
    model = xgb.XGBClassifier(**params.get("model", {}))
    model.fit(X, y)
    return model
```

**`src/evaluate/run.py`** — metrics

```python
from sklearn.metrics import accuracy_score, log_loss

def evaluate(model, df):
    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(int)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "val_accuracy": accuracy_score(y, preds),
        "val_logloss": log_loss(y, probs),
    }
```

## 5. Run experiments

```bash
# Train — build features, fit model, log everything to MLflow
kitchen run train

# Evaluate — load champion model, compute metrics, write metrics.json
kitchen run evaluate
```

After evaluate, `metrics.json` contains your current scores:

```json
{
  "val_accuracy": 0.812,
  "val_logloss": 0.421
}
```

Inspect experiment history and promote the best run:

```bash
kitchen experiments compare val_accuracy
kitchen promote val_accuracy
```

View the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
# Open http://localhost:5000
```

## 6. Generate a submission

```bash
kitchen submit
```

This validates the CSV (column names, row count, nulls, duplicate IDs) before uploading. To wait for the public leaderboard score:

```bash
kitchen submit --wait
```

The score is written back to `metrics.json` under `kaggle_public_score`.

## 7. Set up CI (optional but recommended)

If you passed `--ci` during init, `.github/workflows/train-evaluate.yml` is already in place. Add your Kaggle credentials as GitHub Actions secrets:

1. Go to **Settings → Secrets and variables → Actions**.
2. Add `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Every push to `main` and every PR will now train, evaluate, and post a metrics comment automatically. See [CI/CD Integration](ci-cd.md) for branch protection setup and metric thresholds.

## Metric thresholds

Gate CI on minimum acceptable performance by adding a `thresholds` block to `params.yaml`:

```yaml
thresholds:
  val_accuracy: 0.80          # fail if below 0.80
  val_logloss:
    max: 0.50                 # fail if above 0.50
```

The CI evaluate step exits non-zero when any threshold is violated, which blocks the PR merge.

## Params reference

Key sections in `params.yaml` and what they control:

| Section | Key | Description |
|---|---|---|
| `data` | `source` | `kaggle`, `local`, or `s3` |
| `data` | `competition` | Kaggle competition slug |
| `data` | `id_col` | ID column in the test set |
| `data` | `target_col` | Target column in the training set |
| `model` | any | Passed directly to `fit()` as `params["model"]` |
| `thresholds` | metric name | Minimum (or `max:` maximum) acceptable value for CI |
