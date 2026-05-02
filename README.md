# light-ml-platform

A reusable MLOps platform to work on Kaggle projects. Two modules work together: **recipes** let you spin up resources from YAML specs and **kitchen** is a framework repos install to build ML pipelines.

See [`docs/backlog.md`](docs/backlog.md) for the current roadmap, priorities, and milestone plan.

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
pip install "kitchen @ git+https://github.com/rkoren/light-ml-platform#subdirectory=kitchen"
```

#### Components

| Module | What it does |
|---|---|
| `kitchen.ingest` | Download raw data from Kaggle, S3, or local paths |
| `kitchen.store` | `DataStore` — typed paths and parquet/CSV I/O |
| `kitchen.tracking` | `Tracker` — MLflow wrapper with nested param flattening |
| `kitchen.steps` | `FeatureBuilder`, `Trainer`, `Evaluator` ABCs |
| `kitchen.serve` | FastAPI + Mangum app; plug in a `predictor.py` |
| `flows/` | Prefect `train` and `monitor` flows |

#### How a project repo uses kitchen

A project implements three functions and drops them in `src/`:

```
my-kaggle-project/
├── src/
│   ├── features/run.py    # def build(params, store) -> None
│   ├── train/run.py       # def train(params, store, tracker) -> model
│   └── evaluate/run.py    # def evaluate(model, params, store) -> dict
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
experiment: my-project          # MLflow experiment name

data:
  source: kaggle                # kaggle | s3 | local
  competition: my-competition   # Kaggle slug (source: kaggle)
  bucket: my-bucket             # S3 bucket (source: s3)
  prefix: raw/                  # S3 prefix (source: s3)
  path: /data                   # local path (source: local)

mlflow:
  tracking_uri: http://localhost:5000
  artifact_bucket: my-mlflow-artifacts

train:
  # passed directly to Trainer.fit() — project-defined shape

monitor:
  reference_file: reference.parquet
  current_file: current.parquet
  report_bucket: my-mlflow-artifacts
  report_key: monitoring/drift_report.html
```

---

## CI/CD

A reusable GitHub Actions workflow handles CI/CD for any project repo that calls it.

```yaml
# .github/workflows/ci.yml (in your project repo)
jobs:
  kitchen-deploy:
    uses: rkoren/light-ml-platform/.github/workflows/ml-pipeline.yml@main
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

AWS authentication uses OIDC — no long-lived credentials stored in GitHub. See [`scripts/bootstrap-aws.sh`](scripts/bootstrap-aws.sh) for one-time account setup.

---

## Stack

| Concern | Tool |
|---|---|
| IaC generation | recipes CLI (this repo) + Terraform |
| Data versioning | DVC + S3 |
| Experiment tracking | MLflow |
| Serving | FastAPI + Mangum → AWS Lambda (ECR image) |
| Drift monitoring | Evidently AI |
| Orchestration | Prefect |
| Auth | AWS OIDC (keyless GitHub Actions) |
