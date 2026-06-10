# Project Lifecycle

A kitchen project moves through seven phases. Each phase has a single entry-point command and produces a well-defined output that the next phase consumes.

---

## Lifecycle diagram

```
kitchen init
     │  Creates project scaffold: src/, params.yaml, .env.example,
     │  CI workflow, DVC config (optional)
     ▼
kitchen check
     │  Verifies: Python version, CLI tools (dvc, git), credentials
     │  (.env present, Kaggle key reachable), params.yaml valid
     ▼
kitchen ingest
     │  Downloads raw data to data/raw/ from Kaggle, S3, or local path
     │  DVC tracks data/raw/ with S3 as remote (if --with-dvc)
     ▼
kitchen run features
     │  Runs src/features/run.py → data/processed/features.parquet
     │  DVC tracks data/processed/ (if --with-dvc)
     ▼
kitchen run train
     │  Runs src/train/run.py → model artifact logged to MLflow
     │  Optionally auto-promotes if --auto-promote flag set
     ▼
kitchen leaderboard / kitchen promote
     │  leaderboard: ranks all runs by metric, marks champion [C] and leader ★
     │  promote: registers best run in MLflow Model Registry as champion alias
     ▼
kitchen run evaluate  (optional standalone step)
     │  Runs src/evaluate/run.py → metrics.json; CI gates on thresholds
     ▼
kitchen serve local
     │  Starts FastAPI server backed by src/serve/predictor.py
     │  predictor.py loads champion from MLflow registry
     ▼
[CI: docker build + ECR push + Lambda update]
     │  kitchen does not own the deploy step; recipes provisions the infra,
     │  and the CI workflow handles the container push + Lambda update
     ▼
GET /health   GET /metadata   POST /predict   POST /predict/batch
     │  Live serving via Lambda + Function URL
     ▼
kitchen run monitor
     │  Generates a drift report comparing reference vs current data
     │  Uploads HTML report to S3 (or saves locally)
     ▼
[repeat: new data → kitchen ingest → … → kitchen promote]
```

---

## Phase details

### `kitchen init`

Creates a project directory with all scaffold files. Key outputs:

| File | Purpose |
|---|---|
| `params.yaml` | Single source of truth for all training config |
| `src/features/run.py` | Feature engineering stub |
| `src/train/run.py` | Model training stub (or template if `--template` set) |
| `src/evaluate/run.py` | Evaluation stub |
| `src/serve/predictor.py` | Inference stub with MLflow load example |
| `.env.example` | Credential template |
| `.github/workflows/train-evaluate.yml` | CI scaffold (with `--ci`) |
| `dvc.yaml` | Stage definitions (with `--with-dvc`) |

### `kitchen run train`

The core loop. Every run is logged to MLflow with:

- All `params.yaml` values as parameters
- Metrics returned by `src/evaluate/run.py`
- The trained model as a registered artifact

Pass `--auto-promote --promote-metric val_accuracy` to automatically update the champion alias when the new run outperforms the current best.

### `kitchen promote`

Scans all runs in the current experiment, picks the best by the given metric, and registers it in the MLflow Model Registry under the `champion` alias. The champion is what `predictor.py` loads at serving time:

```python
model = mlflow.sklearn.load_model("models:/my-project@champion")
```

### Monitoring loop

After a model is live, `kitchen run monitor` compares a reference dataset (the training distribution) against current production inputs. The output is an HTML drift report (KS / chi-square / PSI per column) that flags drift in any input feature or the label distribution. When drift is detected, the remediation path is: re-ingest → re-train → promote → redeploy.

---

## What kitchen does NOT own

| Step | Who owns it |
|---|---|
| AWS infrastructure provisioning | `recipes` + Terraform |
| Docker image build and ECR push | CI workflow (scaffolded by `kitchen init --ci`) |
| Lambda function update | `aws lambda update-function-code` in CI |
| Kaggle submission | `kitchen run submit` (Kaggle-specific projects) |
