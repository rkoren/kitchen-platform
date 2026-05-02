# Orchestration

Pipeline runs are orchestrated with [Prefect](https://www.prefect.io). Each competition project gets two experiment flows plus a generic single-run flow, all scaffolded by `kitchen init`.

## Flows

### `experiments/baseline.py`

Runs the full pipeline tagged as the baseline approach:

```
build_features → train_model → evaluate_model
```

Tags the MLflow run with `model_variant=baseline`.

```bash
python experiments/baseline.py
```

### `experiments/challenger.py`

Same structure as baseline but tagged `model_variant=challenger`. Edit this file to override params or add features before calling `run_variant`.

```bash
python experiments/challenger.py
```

### `flows/train_flow.py`

Generic single-run training pipeline from `kitchen.flows.train_flow`. Useful for quick iteration without experiment tagging.

```bash
python flows/train_flow.py
```

### `flows/promote.py`

Compares baseline vs challenger by metric and promotes the winner to the `champion` alias in the MLflow Model Registry.

```bash
python flows/promote.py --dry-run   # compare without promoting
python flows/promote.py             # promote best model
```

### `kitchen/flows/monitor_flow.py`

Drift detection flow — generates Evidently reports and uploads to S3:

```
load_reference_data → load_current_data → run_drift_report → upload_report
```

```bash
python -m kitchen.flows.monitor_flow
```

## Relationship to DVC

DVC and Prefect serve different purposes and complement each other:

| Concern | Tool |
|---|---|
| Stage caching and data lineage | DVC |
| Experiment scheduling, retries, observability | Prefect |

Projects that want DVC-tracked pipelines can add a `dvc.yaml` alongside the Prefect flows — the kitchen `DataStore` paths (`data/raw/`, `data/processed/`) align with DVC stage conventions.
