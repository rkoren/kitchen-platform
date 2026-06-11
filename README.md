# kitchen-platform

Platform to work on ML competitions or iteratively improve and deploy models. **recipes** lets you spin up cloud resources from YAML specs, and **kitchen** is the framework that repos can install to build and deploy with.

The workflow has two modes:

- **Exploratory mode** — `kitchen.experiment()` in a notebook, `DataStore.preview()`, one-off overrides (`kitchen run train --override model.max_depth=6`), and `kitchen diff` to understand why one run beat another
- **Production mode** — `kitchen run features/train/evaluate`, `kitchen run train --auto-promote`, CI-triggered pipelines, Lambda serving via `predictor.py`

---

## Modules

### `recipes/` — IaC CLI

Generates Terraform from a declarative YAML spec. Supports S3, ECR, IAM roles, and Lambda. Used in CI to provision infrastructure before deploying.

```bash
pip install -e recipes/
recipes generate infra.yaml --out ./tf
```

**`infra.yaml` example:**

```yaml
name: my-project
region: us-east-1

resources:
  - type: s3
    name: my-project-data
    versioning: true

  - type: ecr
    name: my-project-serve
    scan_on_push: true
    lambda_access: true       # adds resource policy for Lambda pulls

  - type: iam_role
    name: my-project-exec
    service: lambda.amazonaws.com
    policies:
      - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

  - type: lambda
    name: my-project-serve
    role: my-project-exec
    ecr_repo: my-project-serve   # resolves to the ECR URL at apply time
    memory: 1024
    timeout: 30
```

| Resource type | Terraform resources generated |
|---|---|
| `s3` | `aws_s3_bucket`, optional `aws_s3_bucket_versioning` |
| `ecr` | `aws_ecr_repository`, optional `aws_ecr_repository_policy` |
| `iam_role` | `aws_iam_role`, `aws_iam_role_policy_attachment` × N |
| `lambda` | `aws_lambda_function` with `depends_on` for IAM propagation |

---

### `kitchen/` — MLOps Framework

A reusable Python library that handles the platform concerns (data I/O, experiment tracking, serving, orchestration)

```bash
pip install "kitchen @ git+https://github.com/rkoren/kitchen-platform#subdirectory=kitchen"
```

#### Components

| Module | What it does |
|---|---|
| `kitchen.ingest` | Download raw data from Kaggle, S3, or local paths |
| `kitchen.store` | `DataStore` — typed paths and parquet/CSV I/O |
| `kitchen.tracking` | `Tracker` — MLflow wrapper with nested param flattening |
| `kitchen.steps` | `FeatureBuilder`, `Trainer`, `Evaluator` ABCs |
| `kitchen.submit` | Validate, log, upload, and score Kaggle submissions |
| `kitchen.serve` | FastAPI + Mangum app; plug in a `predictor.py` |
| `flows/` | Prefect `train` and `monitor` flows |

#### How a project repo uses kitchen

A project implements three functions and drops them in `src/`:

```
my-kaggle-project/
├── src/
│   ├── features/run.py    # def build(params, store) -> None
│   ├── train/run.py       # def train(params, store, tracker) -> model
│   ├── evaluate/run.py    # def evaluate(model, params, store) -> dict
│   └── submit/run.py      # def generate(model, params, store) -> (df, path)
├── predictor.py           # def predict(payload: dict) -> dict
├── params.yaml
├── infra.yaml
└── dvc.yaml
```

**`src/train/run.py` example:**

```python
from kitchen.steps import Trainer
import xgboost as xgb

class XGBTrainer(Trainer):
    def fit(self, df, params):
        X, y = df.drop("target", axis=1), df["target"]
        model = xgb.XGBClassifier(**params["train"])
        model.fit(X, y)
        return model

if __name__ == "__main__":
    from kitchen.store import DataStore
    from kitchen.tracking import Tracker
    import yaml

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    XGBTrainer().run(DataStore(), Tracker(params["experiment"]), params)
```

**`predictor.py` example** (deployed on Lambda):

```python
import joblib

_model = joblib.load("model.pkl")

def predict(payload: dict) -> dict:
    features = [payload[k] for k in sorted(payload)]
    return {"prediction": int(_model.predict([features])[0])}
```

#### `params.yaml` schema

```yaml
experiment: my-project          # MLflow experiment name (required)
run_name: baseline-xgb          # optional label for each MLflow run

data:
  source: kaggle                # kaggle | s3 | local
  competition: my-competition   # Kaggle slug (source: kaggle)
  bucket: my-bucket             # S3 bucket (source: s3)
  prefix: raw/                  # S3 prefix (source: s3)
  path: /data                   # local path (source: local)

mlflow:
  tracking_uri: http://localhost:5000
  artifact_bucket: my-mlflow-artifacts
  model_artifact_path: model     # name your model is logged under; what `kitchen promote` registers

train:
  # passed directly to Trainer.fit() — project-defined shape

monitor:
  reference_file: reference.parquet
  current_file: current.parquet
  report_bucket: my-mlflow-artifacts
  report_key: monitoring/drift_report.html

submission:
  sample_submission: SampleSubmission.csv  # relative to data/raw/
  id_col: Id
  target_col: target
  competition: my-competition              # Kaggle slug; omit to skip upload
  message: "baseline run"

metrics_file: metrics.json      # output path for `kitchen run evaluate` and `kitchen report`

# CI gate thresholds — kitchen report exits non-zero if any threshold is breached.
# Use a plain float as a minimum (higher-is-better) or a {min:, max:} object.
thresholds:
  val_accuracy: 0.80            # shorthand: fail if val_accuracy < 0.80
  log_loss:                     # explicit: fail if log_loss > 0.5
    max: 0.5

ci:                             # optional: CI behavior knobs (read by the scaffolded workflow)
  auto_submit: false            # submit to Kaggle after evaluate on a main-branch push
  fail_on_threshold: true       # whether a threshold breach fails `kitchen report` (and the CI job)
  notifications:
    slack_webhook_secret: SLACK_WEBHOOK_URL  # name of the GitHub secret holding the webhook URL
    when: failure               # failure | success | always  (not `on:` — YAML reads it as true)
```

**Framework-owned fields** (`experiment`, `data`, `mlflow`, `monitor`, `submission`, `run_name`, `metrics_file`, `thresholds`, `ci`) are validated by `KitchenConfig` when `kitchen validate` or `kitchen run *` commands load `params.yaml`. All other top-level keys (`train`, `features`, `model`, `evaluate`, etc.) are project-defined and passed through without validation.

The scaffolded `train-evaluate.yml` runs main-branch pushes under the `production` GitHub Environment and all other runs (PRs) under `staging`, so you can attach branch rules, required reviewers, and scoped secrets per environment (see [`docs/kitchen/ci-cd.md`](docs/kitchen/ci-cd.md)). `ci.auto_submit` lets a main-branch push submit to Kaggle without the manual `workflow_dispatch` toggle.

The submission step (`kitchen.submit.log_submission`) validates the CSV against the sample submission, attaches it as an artifact to the active MLflow run, and (when `competition` is set) uploads to Kaggle — logging the public leaderboard score as `lb_score` on the same run. This closes the loop: every MLflow run records your local metric, the submission file, and the actual LB score together.

---

## CI/CD

A reusable GitHub Actions workflow handles CI/CD for any project repo that calls it.

```yaml
# .github/workflows/ci.yml (in your project repo)
jobs:
  kitchen-deploy:
    uses: rkoren/kitchen-platform/.github/workflows/ml-pipeline.yml@main
    with:
      ecr-repository: my-project-serve
      tf-state-key: my-project/terraform.tfstate
    secrets:
      AWS_ROLE_ARN: ${{ secrets.AWS_ROLE_ARN }}
      TF_STATE_BUCKET: ${{ secrets.TF_STATE_BUCKET }}
```

**On push to main:**

```
test → infra-apply (S3, ECR, IAM) → docker-build → lambda-deploy
```

**On pull request:**

```
test → infra-plan (shows Terraform diff as a check)
```

AWS authentication uses OIDC — no long-lived credentials stored in GitHub. Run the one-time account setup with your repo, then add the two printed values as GitHub Actions secrets:

```bash
GITHUB_REPO=<owner>/<repo> bash scripts/bootstrap-aws.sh
```

The script is parameterised (`AWS_REGION`, `CI_ROLE_NAME`, `TF_STATE_BUCKET`, … all override via the environment) and idempotent — safe to re-run; it reuses an existing OIDC provider, state bucket, or CI role rather than failing. See [`scripts/bootstrap-aws.sh`](scripts/bootstrap-aws.sh) for the full list of variables.

---

## Stack

| Concern | Tool |
|---|---|
| IaC generation | recipes CLI (this repo) + Terraform |
| Data versioning | DVC + S3 |
| Experiment tracking | MLflow |
| Serving | FastAPI + Mangum → AWS Lambda (ECR image) |
| Drift monitoring | Built-in (KS / χ² / PSI) |
| Orchestration | Prefect |
| Auth | AWS OIDC (keyless GitHub Actions) |

## Examples

Runnable demos live in [`examples/`](examples/) — see
[`examples/README.md`](examples/README.md) for the index.
[`examples/offline-quickstart/`](examples/offline-quickstart/)
