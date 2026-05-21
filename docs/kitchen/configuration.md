# Configuration Reference

`kitchen` pulls configuration from four places. The rule of thumb: if it controls training behavior, put it in `params.yaml`. If it's a credential or a local override, put it in `.env`. If it's needed only in CI, put it in GitHub secrets or variables.

## Decision guide

| Question | Answer | Where it goes |
|---|---|---|
| Does it affect model training or evaluation? | Yes | `params.yaml` |
| Is it a credential or API key? | Yes | `.env` locally; GitHub secret in CI |
| Is it non-sensitive CI-only config (e.g. a bucket name)? | Yes | GitHub Actions variable |
| Is it a one-off override for a single manual run? | Yes | `workflow_dispatch` input |

---

## `params.yaml`

Version-controlled. Lives in the project root and is committed to git. Controls all training and evaluation behavior.

```yaml
experiment: spaceship-titanic          # MLflow experiment name

data:
  source: kaggle                       # "local", "kaggle", or "s3"
  competition: spaceship-titanic       # Kaggle competition slug

submission:
  id_col: PassengerId                  # ID column in test set
  target_col: Transported              # target column
  message: spaceship-titanic v1        # submission message shown on Kaggle
  sample_submission: sample_submission.csv

features:
  raw_file: train.csv
  processed_file: features.parquet
  test_file: test.csv

model:
  target: Transported                  # must match submission.target_col
  test_size: 0.2
  random_state: 42
  xgb:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05

mlflow:
  tracking_uri: sqlite:///mlruns.db    # override with MLFLOW_TRACKING_URI for S3-backed server

run_name: baseline
metrics_file: metrics.json

thresholds:                            # optional: fail CI if a metric violates its constraint
  val_accuracy: 0.80                   # lower bound — fail if below 0.80
  val_logloss:
    max: 0.50                          # upper bound — fail if above 0.50
```

### What belongs in `params.yaml`

- Data source and file names
- Model hyperparameters
- Feature engineering config
- MLflow experiment and tracking URI (non-secret)
- Metric thresholds for CI gating
- Any value that should be reproducible and reviewable in a PR

### What does NOT belong in `params.yaml`

- Credentials (`KAGGLE_KEY`, `AWS_SECRET_ACCESS_KEY`) — use `.env` or GitHub secrets
- Account IDs, ARNs, or bucket names tied to a specific AWS account — use environment variables or GitHub variables

---

## `.env` (local development)

Never committed — `.gitignore` excludes it. Sourced automatically by `kitchen` at startup via `python-dotenv`. Copy `.env.example` and fill in your values.

```bash
# Kaggle credentials — required for kitchen ingest and kitchen submit
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key

# MLflow — these override params.yaml.mlflow.tracking_uri
MLFLOW_TRACKING_URI=sqlite:///mlruns.db
MLFLOW_EXPERIMENT=spaceship-titanic
MLFLOW_MODEL_NAME=spaceship-titanic-model

# AWS — only needed for S3-backed MLflow or model serving
AWS_PROFILE=default
```

Environment variables take precedence over `params.yaml` for any key they share (e.g. `MLFLOW_TRACKING_URI` overrides `params.yaml → mlflow → tracking_uri`).

---

## GitHub Actions secrets

Repository-level or Environment-level secrets for values that are sensitive and needed in CI. Set at **Settings → Secrets and variables → Actions**.

| Secret | Required by | Description |
|---|---|---|
| `KAGGLE_USERNAME` | `kitchen ingest`, `kitchen submit` | Your Kaggle account username |
| `KAGGLE_KEY` | `kitchen ingest`, `kitchen submit` | Kaggle API token |
| `AWS_ACCESS_KEY_ID` | S3 artifacts, OIDC fallback | IAM access key (prefer OIDC) |
| `AWS_SECRET_ACCESS_KEY` | S3 artifacts, OIDC fallback | IAM secret key (prefer OIDC) |

!!! tip "Use GitHub Environments"
    Prefer **Environment secrets** over repository secrets — they are scoped by branch, support approval gates, and keep `production` credentials out of reach of untrusted branches. See [Secrets management](ci-cd.md#secrets-management) in the CI/CD guide for step-by-step setup.

---

## GitHub Actions variables

Repository-level or Environment-level variables for non-sensitive CI config. Set at **Settings → Secrets and variables → Actions → Variables**.

| Variable | Used by | Description |
|---|---|---|
| `RECIPES_STATE_BUCKET` | `recipes apply` | S3 bucket for Terraform state |
| `AWS_ACCOUNT_ID` | OIDC role ARN | AWS account ID for `configure-aws-credentials` |

Variables are not masked in logs, so never store credentials here.

---

## `workflow_dispatch` inputs

One-off overrides for manually triggered runs. Defined in the scaffolded `.github/workflows/train-evaluate.yml` and visible in **Actions → Run workflow**.

| Input | Type | Default | Description |
|---|---|---|---|
| `submit` | boolean | `false` | Submit to Kaggle leaderboard after evaluate |

Workflow inputs are intentionally minimal — if you find yourself adding many inputs, the value probably belongs in `params.yaml` (where it's version-controlled and reviewable) rather than as a runtime override.

---

## Precedence order

When the same setting can come from multiple places, later entries win:

```
params.yaml  →  .env / environment variable  →  workflow_dispatch input
(committed)       (local or CI secret)            (one-off manual override)
```

The `mlflow.tracking_uri` in `params.yaml` is the most common example: it defaults to `sqlite:///mlruns.db` for local runs, and a CI job that sets `MLFLOW_TRACKING_URI` in its `env:` block overrides it without touching `params.yaml`.
