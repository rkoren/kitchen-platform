# Quickstart

## Prerequisites

- Python 3.11+
- AWS CLI configured (`aws configure`)
- Docker (for model serving)

## Start a new competition project

```bash
pip install -e path/to/light-ml-platform/kitchen
kitchen init my-competition
cd my-competition
pip install -e ../light-ml-platform/kitchen -e .
cp .env.example .env
```

## Download competition data

```bash
# Kaggle competitions
kaggle competitions download -c <competition-slug> -p data/raw/
unzip data/raw/<competition-slug>.zip -d data/raw/
```

## Implement the three required files

| File | What to implement |
|---|---|
| `src/features/run.py` | `build(raw_df) -> df` — feature engineering |
| `src/train/run.py` | `fit(df, params) -> model` — model training |
| `src/evaluate/run.py` | `evaluate(model, df) -> dict` — metrics |

## Run experiments

```bash
# Baseline approach
python experiments/baseline.py

# Challenger (after editing experiments/challenger.py)
python experiments/challenger.py

# Compare and promote best model
python flows/promote.py --dry-run
python flows/promote.py

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
# Open http://localhost:5000
```

## Generate a Kaggle submission

```bash
python flows/generate_submission.py
```
