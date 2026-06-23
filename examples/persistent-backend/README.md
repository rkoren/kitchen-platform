# Persistent-backend validation (the AWS gate, INT-008b)

The same tiny train → evaluate loop as [`offline-quickstart`](../offline-quickstart/),
but on a **real persistent MLflow backend** — RDS Postgres registry + S3 artifacts —
provisioned by `recipes` from this one `menu.yaml`. It's the acceptance test for the thing
the whole persistent-champion effort exists for: **a champion registered by one run is found
by the next run, in a separate process, on a remote backend** (extends VAL-008 to the
unified `kitchen menu run` flow).

Why it matters: with the default per-run `sqlite:///mlruns.db`, every run starts from an
empty registry, so `--auto-promote` never sees a prior champion and promotes
unconditionally — cross-run comparison is a no-op. A remote Postgres registry fixes that.

## One file does it all

`menu.yaml` declares both halves of the platform:

- **infra `recipes`** — a security group (Postgres 5432, **narrow `cidr_blocks` to your IP**),
  an `rds` Postgres backend store (`role: mlflow-backend`), and an `s3` artifact bucket
  (`role: mlflow-artifacts`). The recipe *key* is the bucket name — change it (bucket names
  are globally unique).
- **`pipeline: [provision, train, evaluate]`** — provision the infra, then the runner
  materializes `MLFLOW_TRACKING_URI` (assembled from the RDS Terraform outputs +
  RDS-managed Secrets Manager password) and `MLFLOW_ARTIFACT_BUCKET` into the environment
  before the stages run. `mlflow.tracking_uri` / `artifact_bucket` are declared by `role`,
  not re-typed.

## Run it

Needs the Postgres driver (`pip install -e 'kitchen/[postgres]'`), Terraform ≥ 1.10, AWS
credentials, and a Terraform state bucket.

```bash
cd examples/persistent-backend
# edit menu.yaml: set the SG cidr_blocks to your IP and pick a unique bucket name

kitchen menu run --state-bucket <your-tf-state-bucket>   # run 1: provision + register champion v1
kitchen menu run --state-bucket <your-tf-state-bucket>   # run 2: provision is a no-op; champion persists
```

Run 2 is the gate. Expect:

```
auto-promote: skipped — new run did not beat champion  (0.791667 > 0.791667 (higher=better))
```

That line proves run 2 (a separate process) found run 1's champion on the **remote**
registry. With ephemeral sqlite it would instead say `no current champion` and re-promote.

## Tear down (always)

RDS bills by the hour — destroy when done:

```bash
recipes destroy menu.yaml --state-bucket <your-tf-state-bucket>
```

Note: the artifact bucket is versioned, so if it holds logged models the bucket delete fails
with `BucketNotEmpty` (Terraform won't force-delete a non-empty versioned bucket). Empty all
object **versions** first (e.g. `aws s3api delete-objects` over `list-object-versions`), then
re-run `recipes destroy`.

## What's committed vs. generated

Committed: `menu.yaml`, `src/`, `data/raw/train.csv`. Generated and gitignored:
`mlruns*`, `mlartifacts/`, `metrics.json`, `.env`, `data/processed/*`.
