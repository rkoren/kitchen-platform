# rkoren-kitchen

A reusable MLOps framework for competition and prediction-modeling projects: data ingestion,
experiment tracking, model serving, monitoring, and cloud provisioning — driven by one
`menu.yaml` manifest and one `kitchen` CLI.

It is intentionally generic: the same workflow helpers and CLI cover a large surface of Kaggle
competitions and prediction-modeling projects with minimal per-project configuration.

## Install

```bash
pip install rkoren-kitchen
```

Everything ships in the base install — the CLI, the training/serving/monitoring stack, and the
`recipes` provisioning sub-package (YAML → Terraform → AWS).

## What's in the box

- **Modeling helpers** — `train_val_split`, `classification_metrics`/`regression_metrics`,
  `time_series_cv`/`loto_cv`, calibration and ensembling utilities.
- **Experiment tracking** — an MLflow wrapper (`experiment()`, `Tracker`) with a champion
  registry and one-command auto-promotion.
- **Pipeline** — `FeatureBuilder` / `Trainer` / `Evaluator` stages plus a `DataStore`, run from
  the `kitchen` CLI (`kitchen run …`) or a `menu.yaml` (`kitchen menu run`).
- **Serving & monitoring** — a FastAPI serving scaffold (Lambda/ECR) and in-house drift reports.
- **Provisioning** — `kitchen recipes …` (a.k.a. the `recipes` CLI) turns a YAML spec into
  Terraform for S3/ECR/IAM/Lambda/RDS.

## Quickstart

```bash
kitchen init my-project        # scaffold a project
kitchen run train              # train, track, and (optionally) auto-promote a champion
kitchen menu run               # or run the whole pipeline from menu.yaml
```

## Stability

The public Python API is the top-level `kitchen.__all__`; the `kitchen` CLI and the `menu.yaml`
schema (`kitchen menu schema`) are the other supported surfaces. The package follows SemVer.

See the [project repository](https://github.com/rkoren/kitchen-platform) for full docs.

## License

MIT — see [LICENSE](LICENSE).
