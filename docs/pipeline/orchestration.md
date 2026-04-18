# Orchestration

Pipeline runs are orchestrated with [Prefect](https://www.prefect.io). Both the training pipeline and the monitoring pipeline are defined as Prefect flows.

## Flows

### `train_flow.py`

Runs the full training pipeline end-to-end:

```
ingest → build_features → train_model → evaluate_model
```

```bash
python pipeline/flows/train_flow.py
```

### `monitor_flow.py`

Runs drift detection and generates Evidently reports:

```
load_reference_data → load_current_data → run_drift_report → upload_report
```

```bash
python pipeline/flows/monitor_flow.py
```

## Scheduling

<!-- TODO: document deployment to Prefect Cloud or self-hosted Prefect server -->

To schedule flows on Prefect Cloud:

```bash
prefect cloud login
prefect deploy
```

## Relationship to DVC

DVC manages **data pipeline stages** (ingest → features → train → evaluate) with caching and artifact tracking. Prefect manages **execution scheduling and observability** — when and how the pipeline runs, retries, and notifications. They complement each other rather than overlap.

| Concern | Tool |
|---|---|
| Stage caching and data lineage | DVC |
| Scheduling, retries, observability | Prefect |
