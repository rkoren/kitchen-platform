# Troubleshooting

Common issues and how to fix them.

---

## Pipeline steps

### A `kitchen run` step fails with a one-line error

`kitchen run features`, `kitchen run train`, and `kitchen run evaluate` execute your
project's own `build` / `train` / `evaluate` code. When that code raises, the command
prints a clean one-line summary so the failure isn't buried in a stack trace:

```
error during evaluation: 'is_tourn'
  (re-run with --debug or KITCHEN_DEBUG=1 for the traceback)
```

To debug your own step, re-run with the full traceback (file and line):

```bash
kitchen run evaluate --debug      # or: KITCHEN_DEBUG=1 kitchen run evaluate
```

`KITCHEN_DEBUG=1` applies to every `kitchen run` step, which is handy in CI logs.

---

## Kaggle credentials

### `401 Unauthorized` from `kitchen ingest`

```
kaggle.rest.ApiException: (401) Unauthorized
```

**Cause:** Kaggle API credentials are missing or wrong.

**Fix:**

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **Account** → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Copy the values into your project `.env`:

    ```bash
    KAGGLE_USERNAME=your-username
    KAGGLE_KEY=your-api-key
    ```

3. Run `kitchen check` to confirm the credentials are picked up.

---

### `kitchen check` says Kaggle credentials missing

`kitchen check` looks for `KAGGLE_USERNAME` and `KAGGLE_KEY` in the environment (loaded from `.env` via python-dotenv).

- Confirm `.env` exists (not `.env.example`) in the project root.
- Confirm the values are not blank.
- If you moved the project directory, re-run `cp .env.example .env` and re-fill it.

---

### `403 Forbidden` — competition access

```
404 - Not Found (competition not found or you have not accepted the rules)
```

**Fix:** Go to the competition page on Kaggle and click **Join Competition** / **Accept Rules**. You must accept the rules before the API can download data.

---

## MLflow — local SQLite

### `error: MLflow tracking store schema is out of date`

After upgrading kitchen (which may bump MLflow), an `mlruns.db` created by an older MLflow
can no longer be read. Kitchen detects this and prints remediation instead of a raw MLflow
traceback:

**Fix:** Upgrade the schema in place (preserves run history):

```bash
mlflow db upgrade sqlite:///mlruns.db
```

If that fails with `Can't locate revision …`, the database predates the current MLflow's
migration history — archive it and start fresh (local run history is lost):

```bash
mv mlruns.db mlruns.db.bak
```

To keep run history across MLflow upgrades, use a remote/S3-backed tracking server instead
of local SQLite.

---

### `MlflowException: Could not find experiment with ID`

Usually happens when `mlruns.db` exists from a different project or was moved.

**Fix:** Either delete `mlruns.db` and start fresh, or point `MLFLOW_TRACKING_URI` to the correct path:

```bash
export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlruns.db
```

---

### MLflow UI shows no runs

`kitchen ui` opens the UI for the tracking URI in `params.yaml`. If runs were logged with a different URI (e.g. a previous absolute path), they won't appear.

**Fix:** Check the URI used during training:

```bash
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

Then run `kitchen ui` with that URI or update `params.yaml → mlflow → tracking_uri`.

---

### `sqlite3.OperationalError: database is locked`

Two processes are writing to the same `mlruns.db` simultaneously.

**Fix:** Stop all running `kitchen run train` processes before starting another, or switch to a PostgreSQL backend for parallel runs.

---

### Runs logged but model not in registry

`kitchen promote` registers the model. If you only ran `kitchen run train` without `--auto-promote`, or if promote was run against a different tracking URI, the registry will be empty.

**Fix:**

```bash
kitchen leaderboard           # confirm runs are visible
kitchen promote val_accuracy  # register the best run
```

---

## MLflow — S3 artifacts

### `botocore.exceptions.NoCredentialsError`

MLflow is trying to upload artifacts to S3 but AWS credentials are not configured.

**Fix (local):** Set credentials in `.env`:

```bash
AWS_PROFILE=default
# or
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

**Fix (CI):** Add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as GitHub Actions secrets, or configure OIDC federation (recommended).

---

### `ClientError: Access Denied` on S3 artifact upload

The IAM identity doesn't have `s3:PutObject` on the artifact bucket.

**Fix:** Add `AmazonS3FullAccess` or a scoped policy to the IAM role/user. If using `recipes`, add `arn:aws:iam::aws:policy/AmazonS3FullAccess` to the `iam_role.policies` list (or scope it to the specific bucket ARN).

---

### Artifact URI shows `./mlartifacts` instead of `s3://...`

`mlflow.artifact_bucket` in `params.yaml` is not set, and `MLFLOW_ARTIFACT_ROOT` is not in the environment.

**Fix:** Set `mlflow.artifact_bucket` in `params.yaml`:

```yaml
mlflow:
  tracking_uri: sqlite:///mlruns.db
  artifact_bucket: my-project-data
```

---

### `error: the champion's stored artifact location is not reachable from this environment`

`kitchen run evaluate` (or a `predictor.py` loading the champion via
`kitchen.serve.load_champion`) detected that the registered model version's stored
artifact `source` points at a location this machine can't reach. The usual cause is
**migrating the tracking store** — e.g. from a local `sqlite:///mlruns.db` to a remote
MLflow server — or moving the project: the model version still records its original,
local artifact path, which no longer exists here. A model version's `source` is
immutable, so it can't be edited in place.

**Fix:** pick one:

1. Re-train and re-promote against the current store:

   ```bash
   kitchen run train --auto-promote --promote-metric <metric>
   ```

2. Re-register a run whose artifacts already live in the current store, then re-point
   the alias: `kitchen promote <metric>` (or `kitchen promote --run-id <id>`).

3. If you still have the original artifacts, copy them to the location named in the
   error so the recorded `source` resolves again.

Re-run with `--debug` (or `KITCHEN_DEBUG=1`) to see the underlying MLflow traceback.

---

## Terraform state

The generated S3 backend encrypts state at rest (`encrypt = true`) and locks it using
the S3-native lockfile (`use_lockfile = true`) — no DynamoDB table is required. The
`bootstrap-aws.sh` state bucket also enables default SSE (AES256) and versioning.
S3-native locking needs **Terraform >= 1.10**; the backend declares
`required_version = ">= 1.10"`, so older Terraform fails `init` with a clear version
error rather than running unlocked.

### `Error: No valid credential sources found`

Terraform cannot authenticate to AWS when running `terraform apply`.

**Fix:**

```bash
aws configure          # set up default profile
# or
export AWS_PROFILE=my-profile
```

In CI, set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as secrets, or configure the `aws-actions/configure-aws-credentials` action with OIDC.

---

### `Error acquiring the state lock`

Another `terraform apply` is running (or a previous run crashed without releasing the lock).

**Fix:** Check for other running applies. If none, force-unlock:

```bash
# Get the lock ID from the error message
terraform force-unlock <LOCK-ID>
```

---

### `Error: S3 bucket does not exist` (remote state)

The S3 bucket configured as the Terraform state backend doesn't exist yet.

**Fix:** Create the state bucket manually before running `terraform init`. The state bucket itself cannot be managed by the same Terraform config that uses it as a backend.

```bash
aws s3 mb s3://my-project-tf-state --region us-east-1
aws s3api put-bucket-versioning \
  --bucket my-project-tf-state \
  --versioning-configuration Status=Enabled
```

---

### `Error: Resource already exists` after importing

If a resource was created outside Terraform (e.g. via the console), import it:

```bash
terraform import aws_s3_bucket.my_bucket my-bucket-name
```

---

## ECR and Lambda deployment

### `denied: Your authorization token has expired`

The ECR login token is only valid for 12 hours.

**Fix:**

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS \
    --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
```

---

### `Error: ImageNotFoundException` on Lambda invoke

The Lambda is pointing to an image tag that doesn't exist in ECR (e.g. `latest` was overwritten without updating the function, or the wrong repo was pushed to).

**Fix:** Confirm the image exists, then update the function:

```bash
aws ecr describe-images \
  --repository-name my-project-serve \
  --image-ids imageTag=latest

aws lambda update-function-code \
  --function-name my-project-serve \
  --image-uri 123456789.dkr.ecr.us-east-1.amazonaws.com/my-project-serve:latest
```

---

### `Task timed out` on first Lambda invoke

The Lambda is cold-starting and loading a large model (e.g. from MLflow S3 artifacts) within the configured timeout.

**Fix:** Increase `timeout` in your `recipes` spec (max 900 seconds for Lambda):

```yaml
- type: lambda
  name: my-project-serve
  timeout: 60    # was 3
  memory: 1024   # more memory → more CPU → faster model load
```

Re-generate Terraform and apply.

---

### `GET /predict` returns 501 Not Implemented

`predictor.py` was not found or `KITCHEN_PREDICTOR_DIR` is not set.

**Fix:** Confirm the Lambda environment has:

```
KITCHEN_PREDICTOR_DIR=/var/task/src/serve
```

And that `src/serve/predictor.py` is present in the Docker image at that path.

---

### Lambda logs show `PredictorLoadError`

`predictor.py` was found but raised on import or is missing `predict()`.

**Fix:** Run `kitchen serve local` first — it surfaces the same error locally with a full traceback. Fix the predictor, rebuild the image, and redeploy.
