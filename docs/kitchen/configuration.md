# Configuration Reference

`kitchen` pulls configuration from four places. The rule of thumb: if it controls training behavior, put it in `menu.yaml`. If it's a credential or a local override, put it in `.env`. If it's needed only in CI, put it in GitHub secrets or variables.

## Decision guide

| Question | Answer | Where it goes |
|---|---|---|
| Does it affect model training or evaluation? | Yes | `menu.yaml` |
| Is it a credential or API key? | Yes | `.env` locally; GitHub secret in CI |
| Is it non-sensitive CI-only config (e.g. a bucket name)? | Yes | GitHub Actions variable |
| Is it a one-off override for a single manual run? | Yes | `workflow_dispatch` input |

---

## `menu.yaml`

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
  artifact_bucket: my-project-data    # S3 bucket for model artifacts (optional)

run_name: baseline
metrics_file: metrics.json

thresholds:                            # optional: fail CI if a metric violates its constraint
  val_accuracy: 0.80                   # lower bound — fail if below 0.80
  val_logloss:
    max: 0.50                          # upper bound — fail if above 0.50

monitor:                               # optional: drift monitoring configuration
  reference_file: reference.parquet
  current_file: current.parquet
  report_bucket: my-project-data      # S3 bucket for HTML reports (or omit for local only)
  report_key: monitoring/drift_report.html
  local_path: monitoring/drift_report.html
```

### `data:` section

The `source` field determines which other fields are required:

| `source` | Required fields | Optional fields |
|---|---|---|
| `kaggle` | `competition` | — |
| `s3` | `bucket` | `prefix` |
| `local` | `path` | — |

```yaml
# Kaggle
data:
  source: kaggle
  competition: spaceship-titanic

# S3
data:
  source: s3
  bucket: my-project-data
  prefix: raw/                   # defaults to "" (bucket root)

# Local
data:
  source: local
  path: /data/my-project/raw
```

### `thresholds:` section

Used to gate CI on metric quality. `kitchen run evaluate` exits with a non-zero code if any threshold is violated.

```yaml
thresholds:
  # Shorthand: float is treated as a minimum (lower bound)
  val_accuracy: 0.80          # fail if val_accuracy < 0.80

  # Explicit: use min/max for full control
  val_logloss:
    max: 0.50                 # fail if val_logloss > 0.50 (lower-is-better metric)

  val_brier:
    min: 0.0
    max: 0.25                 # valid range constraint
```

### `secrets:` section

Declares the secrets a project needs and where each one resolves from. It is the single
source of truth for credentials — `kitchen check` validates it, and the resolver reads it.
Each entry names one secret:

```yaml
secrets:
  KAGGLE_KEY:                     # SM JSON bundle: field `key` within bundle `aws_secret`
    aws_secret: my-project/prod
    key: KAGGLE_KEY
    required: true                # default true; gates `kitchen check`
  SLACK_WEBHOOK_URL:              # optional — absence won't fail check
    aws_secret: my-project/prod
    key: SLACK_WEBHOOK_URL
    required: false
  DB_PASSWORD:
    ssm: /my-project/db-password  # SSM Parameter Store path (SecureString)
  LOCAL_TOKEN: {}                 # no source → env-only (must come from env / .env)
```

A secret declares **either** `aws_secret` (+ optional `key` to select a field in the JSON
bundle) **or** `ssm`, or neither (env-only). `kitchen check` **resolves** every required secret
through the chain below and hard-fails (with the exact remediation) when one can't be resolved —
so a missing credential is caught at pre-flight, not mid-run. Optional (`required: false`)
secrets only warn. An env var of the same name always overrides the declared cloud source.

> **Deprecated:** the earlier `check.required_env: [NAME, ...]` list still works — it folds
> into the manifest as env-only required secrets and `kitchen check`/`validate` warn — but
> migrate each entry to `secrets:` (it is removed in a future release).

**Resolving at runtime.** Read a secret in code through one call:

```python
from kitchen import secrets
kaggle_key = secrets.get("KAGGLE_KEY")   # raises SecretNotFound if unresolved
```

`get()` resolves through an ordered chain — (1) process env / `.env` → (2) the declared cloud
source (Secrets Manager or SSM, attempted only when an AWS identity resolves) → (3) raise
`SecretNotFound` naming the secret and how to provide it. An env var always overrides the cloud
source. Resolved values are cached in-process for `KITCHEN_SECRETS_TTL` seconds (default 300) so
a rotated secret is picked up without a restart. The resolver never logs secret values. Use
`secrets.try_get(name)` for a `None`-instead-of-raise variant.

To pass secrets into a subprocess that needs them in its environment (e.g. DVC needs `AWS_*` to
reach S3), use `secrets.resolve_into_env([...])` — it resolves each secret, **masks** it under
GitHub Actions (emits `::add-mask::` so the value is scrubbed from CI logs), and returns an
environment mapping to hand to `subprocess.run(..., env=...)`:

```python
from kitchen import secrets
env = secrets.resolve_into_env(["AWS_SECRET_ACCESS_KEY"])
subprocess.run(["dvc", "pull"], env=env, check=True)
```

**Generating `.env.example`.** `kitchen secrets template` renders an annotated `.env.example`
from the manifest — one blank `NAME=` line per secret, tagged required/optional with its source
— so a fresh clone self-documents what to set (use `--stdout` to preview, `--force` to overwrite):

```bash
kitchen secrets template            # writes .env.example
kitchen secrets template --stdout   # print instead
```

**Least-privilege IAM for CI.** When secrets live in Secrets Manager / SSM, the CI or deploy
role needs read access to exactly those — and no more. `kitchen secrets iam-policy` emits that
policy from the manifest (one statement per source type, scoped to the declared ARNs):

```bash
kitchen secrets iam-policy --account 123456789012 --region us-east-1 \
  | aws iam put-role-policy --role-name <ci-role> \
      --policy-name kitchen-secrets --policy-document file:///dev/stdin
```

With no `--account`/`--region` the ARNs use `*` wildcards (still name-scoped; no account ID
embedded). Pair this with a GitHub-OIDC role (`bootstrap-aws.sh` / SEC-007–008) so CI assumes a
short-lived role and resolves secrets from the cloud — no long-lived keys.

### `monitor:` section

Required by `kitchen run monitor`. At least one of `report_bucket` or `local_path` must be set.

| Field | Default | Description |
|---|---|---|
| `reference_file` | `reference.parquet` | Baseline dataset (training distribution) |
| `current_file` | `current.parquet` | Recent production inputs |
| `report_bucket` | `""` | S3 bucket to upload the HTML drift report to |
| `report_key` | `monitoring/drift_report.html` | S3 key for the report |
| `local_path` | `""` | Local file path to save the report (can be set alongside `report_bucket`) |

### `mlflow:` section

| Field | Default | Description |
|---|---|---|
| `tracking_uri` | `sqlite:///mlruns.db` | MLflow tracking backend URI |
| `artifact_bucket` | `null` | S3 bucket for model artifacts (overrides MLflow default artifact root) |

#### Persistent tracking backend (champions across runs)

The default `sqlite:///mlruns.db` is **per-run** in CI: the registry starts empty every run, so `kitchen run train --auto-promote` never sees a champion and promotes unconditionally — cross-run comparison is a no-op. To make champions persist (and `--auto-promote` actually compare against the last good model), point MLflow at a **persistent backend store** (the registry/champion lives there) plus a shared **artifact store** (the model files). Rationale and the full decision: [`mlflow-tracking-backend`](../decisions/mlflow-tracking-backend.md).

1. **Deploy** the backend with `recipes` — `recipes/examples/mlflow-tracking-backend.yaml` provisions a security group, an RDS Postgres instance (master password managed in Secrets Manager — never in Terraform/state), and a versioned S3 artifact bucket:

   ```bash
   recipes apply mlflow-tracking-backend.yaml --state-bucket <your-tf-state-bucket>
   ```

   Terraform outputs the connection `…_endpoint`, the `…_master_user_secret_arn`, and the bucket.

2. **Install the PostgreSQL driver** alongside kitchen — MLflow needs it to talk to a `postgresql://` store (it is not in the base install): `pip install 'rkoren-kitchen[postgres]'`. `kitchen check` fails fast with this hint if a postgresql tracking URI is set without it.

3. **Set the artifact bucket** in `menu.yaml` (the connection URL itself comes from the RDS-managed secret at run time — step 4 — so it is never written to menu.yaml or a second secret):

   ```yaml
   mlflow:
     artifact_bucket: my-project-mlflow-artifacts
   ```

4. **CI** — in the scaffolded `train-evaluate.yml`, set `MLFLOW_ARTIFACT_BUCKET`, remove the SQLite `MLFLOW_TRACKING_URI` line, and (after an `aws-actions/configure-aws-credentials` step with your OIDC role — see [aws-deployment](aws-deployment.md)) assemble the tracking URI from the RDS-managed secret + endpoint into the job environment:

   ```yaml
   - name: Resolve persistent MLflow backend
     run: kitchen secrets db-url --secret-id <…_master_user_secret_arn> --endpoint <…_endpoint>
   ```

   `kitchen secrets db-url` fetches the RDS-managed username/password, URL-encodes them, and writes `MLFLOW_TRACKING_URI=postgresql://…` to `$GITHUB_ENV` (masked, never to stdout) — no hand-built URL and no second secret. If the recipes Terraform workspace is present in the job (e.g. you `recipes apply` in the same job), drop the flags entirely and let it read the outputs: `kitchen secrets db-url --from-terraform ~/.recipes/tf/<spec>`. (For a backend you manage yourself, the general path still works: store the full URL as a secret, declare it in `menu.yaml` `secrets:` as `MLFLOW_TRACKING_URI`, and use `kitchen secrets export --name MLFLOW_TRACKING_URI`.)

5. **Local / notebooks** — assemble the URL straight from the recipes workspace into `.env` (loaded automatically at startup); needs AWS credentials locally:

   ```bash
   kitchen secrets db-url --from-terraform ~/.recipes/tf/<spec> --output .env
   ```

   `--from-terraform` reads the `<name>_endpoint` + `<name>_master_user_secret_arn` outputs directly — no copying ARNs/endpoints by hand. (Pass `--secret-id`/`--endpoint` instead if you're not using recipes.)

**Migrating from local SQLite:** champions registered against the old `mlruns.db` do not carry over — re-train against the new store (the first `--auto-promote` registers a fresh champion), or `mlflow db upgrade <uri>` if you are moving an existing DB (see [troubleshooting](troubleshooting.md) for the schema-mismatch guidance). A champion whose **artifacts** were written to a local path before the move won't load against the new store — `kitchen run evaluate` / `predictor.py` detect this and explain it (`ArtifactLocationError`), so keep `MLFLOW_ARTIFACT_BUCKET` set so artifacts land in S3 from the start.

**Reachability:** the example RDS is `publicly_accessible` with an open security group so GitHub-hosted runners (dynamic IPs) and your laptop can connect — gated by TLS and the RDS-managed password. For a private backend, drop public access and run an MLflow tracking server in front (the decision doc's upgrade path).

**Validate it:** deploy the throwaway `recipes/examples/mlflow-backend-validation.yaml` (tiny instance, `deletion_protection: false` for clean teardown), point `MLFLOW_TRACKING_URI` at it, and run `examples/validate_persistent_backend.py` **twice** — run 2 finding run 1's champion confirms the backend persists champions across runs. Then `recipes destroy`.

### `recipes:` — stages, including command stages

A `pipeline:` step runs a `recipes:` entry. A `kind: stage` recipe is either an **in-process**
Python callable (`source:`, the tabular default — `kitchen run <stage>` imports and calls it) or a
**command stage** (`cmd:`) that runs as a subprocess. Command stages fit pipelines that aren't
single-table supervised — inference-only, non-Python, or needing a different interpreter.

```yaml
pipeline: [detect, track]
recipes:
  detect:
    kind: stage
    cmd: ["./scripts/detect.sh", "--fast"]   # a list is the argv, used verbatim (no shell)

  track:
    kind: stage
    python: .venv-track/bin/python           # per-stage interpreter (GEN-003)
    cmd: -m pipeline.run --thresh 0.5        # when `python:` is set, `cmd:` is the ARGS to it
    inputs: [data/frames]                    # must exist before the stage runs (fail fast)
    outputs: [tracks.parquet]                # a missing declared output warns after
```

- **`cmd`** — a list (the argv, used verbatim) or a string (`shlex`-split, **no shell**; wrap
  pipes/globs in a `.sh` and call that). A stage declares **either** `source` **or** `cmd`.
- **`python`** — an interpreter path; when set, `cmd` is the *args* passed to it (so
  `python: .venv/bin/python`, `cmd: -m pipeline.run`). Omit it to run any other program directly.
- **`inputs` / `outputs`** — declared paths (relative to the project dir); inputs are checked to
  exist before running, a missing output warns after.

A command stage inherits the process environment and working directory. **Metrics are the stage's
job** — call `kitchen log` (framework-agnostic tracking) or write `metrics.json` from inside it to
feed `leaderboard`/`thresholds`. Run one stage in isolation with `kitchen stage <name>` (add
`--dry-run` to preview the exact argv, or `-C <dir>` to run from elsewhere); `kitchen menu run`
runs the whole pipeline.

To scaffold a lean project built around a command stage — no `FeatureBuilder`/`Trainer`/`Evaluator`
ABCs — use `kitchen init <name> --kind pipeline` (vs the default `--kind tabular`). It generates a
`menu.yaml` with a command stage and a `src/pipeline/run.py` stub that writes its metric to
`$KITCHEN_METRICS_FILE`; `--source kaggle` wires `kitchen ingest` too.

### What belongs in `menu.yaml`

- Data source and file names
- Model hyperparameters
- Feature engineering config
- MLflow experiment and tracking URI (non-secret)
- Metric thresholds for CI gating
- Any value that should be reproducible and reviewable in a PR

### What does NOT belong in `menu.yaml`

- Credentials (`KAGGLE_KEY`, `AWS_SECRET_ACCESS_KEY`) — use `.env` or GitHub secrets
- Account IDs, ARNs, or bucket names tied to a specific AWS account — use environment variables or GitHub variables

---

## `.env` (local development)

Never committed — `.gitignore` excludes it. Sourced automatically by `kitchen` at startup via `python-dotenv`. Copy `.env.example` and fill in your values.

```bash
# Kaggle credentials — required for kitchen ingest and kitchen submit
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key

# MLflow — these override menu.yaml.mlflow.tracking_uri
MLFLOW_TRACKING_URI=sqlite:///mlruns.db
MLFLOW_EXPERIMENT=spaceship-titanic
MLFLOW_MODEL_NAME=spaceship-titanic-model

# AWS — only needed for S3-backed MLflow or model serving
AWS_PROFILE=default
```

Environment variables take precedence over `menu.yaml` for any key they share (e.g. `MLFLOW_TRACKING_URI` overrides `menu.yaml → mlflow → tracking_uri`).

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

Workflow inputs are intentionally minimal — if you find yourself adding many inputs, the value probably belongs in `menu.yaml` (where it's version-controlled and reviewable) rather than as a runtime override.

---

## Precedence order

When the same setting can come from multiple places, later entries win:

```
menu.yaml  →  .env / environment variable  →  workflow_dispatch input
(committed)       (local or CI secret)            (one-off manual override)
```

The `mlflow.tracking_uri` in `menu.yaml` is the most common example: it defaults to `sqlite:///mlruns.db` for local runs, and a CI job that sets `MLFLOW_TRACKING_URI` in its `env:` block overrides it without touching `menu.yaml`.
