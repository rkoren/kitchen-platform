# Data Versioning

Data versioning is handled by [DVC](https://dvc.org) with an S3 backend. Raw and processed datasets are tracked outside Git but tied to specific commits via `.dvc` pointer files.

## How it works

1. Raw data is downloaded to `data/raw/` during the `ingest` stage
2. DVC tracks the data directory and stores it in S3
3. `.dvc` pointer files are committed to Git — collaborators run `dvc pull` to fetch the actual data

## Pipeline stages

```
ingest → features → train → evaluate
```

Defined in `dvc.yaml`. Parameters are in `params.yaml` and tracked by DVC so changing a hyperparameter invalidates the downstream cache automatically.

## DVC remote setup

```bash
dvc remote add -d myremote s3://your-bucket/dvc-store
dvc remote modify myremote region us-east-1
```

## Common commands

```bash
dvc repro          # Run the full pipeline (skips cached stages)
dvc repro train    # Run from a specific stage
dvc push           # Push data artifacts to S3
dvc pull           # Pull data artifacts from S3
dvc params diff    # Show changed parameters vs last run
dvc metrics show   # Show tracked metrics
```

## params.yaml reference

<!-- TODO: document all params once pipeline is finalized -->

```yaml
data:
  source: kaggle
  dataset: heart-disease-uci
  target_col: target
  test_size: 0.2
  random_seed: 42
```
