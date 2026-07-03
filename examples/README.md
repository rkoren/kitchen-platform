# Examples

How to see `kitchen` work — from a full end-to-end project down to single-feature scripts.
Each has a docstring (or README) with what it shows and how to run it; this index is the map.

## Complete projects (start here)

Full `kitchen` projects — features → train → evaluate → promote a champion — driven by one
`menu.yaml` and run with `kitchen menu run`, with local SQLite tracking.

| Project | What it is |
|---|---|
| [`spaceship-titanic/`](spaceship-titanic/) | The whole loop on a **real Kaggle competition** (predict `Transported`), end to end through a real submission — XGBoost baseline, genuine feature engineering, `kitchen ingest` → `menu run` → submit. Needs a Kaggle account (it's the free Getting Started sandbox); the flagship "here's the whole thing" reference — see its [README](spaceship-titanic/README.md). |
| [`offline-quickstart/`](offline-quickstart/) | The **smallest** honest end-to-end project (a toy "did the student pass?" set, logistic-regression baseline), **offline with no credentials**. Start here if you just want the `FeatureBuilder` / `Trainer` / `Evaluator` contract at a glance. |

## Run anywhere (synthetic data, no project setup)

These build their own in-memory data and only need `kitchen` installed. Run them from the repo root:

| Script | Demonstrates |
|---|---|
| [`loto_cv_example.py`](loto_cv_example.py) | `loto_cv()` leave-one-group-out CV — one metric key per group plus aggregate mean/std, and how `kitchen leaderboard --expand-metrics` surfaces each group as a sub-column. |
| [`multi_source_features.py`](multi_source_features.py) | `FeatureBuilder.sources()` routing multiple raw CSVs into `build()` as a `dict[filename, DataFrame]`, producing the same processed parquet as a single-file builder. |
| [`validate_persistent_backend.py`](validate_persistent_backend.py) | That a persistent MLflow backend carries champions across runs (LML-012). Point `MLFLOW_TRACKING_URI` at your RDS Postgres backend and run it **twice**: run 2 must find run 1's champion and compare against it. |

```bash
python examples/loto_cv_example.py
python examples/multi_source_features.py
```

## Conventions for new examples

- One feature (or acceptance test) per script, with a docstring covering **what it shows**,
  **prerequisites**, and **how to run it**.
- Prefer synthetic, self-contained data so anyone can run it with nothing but `kitchen` installed.
- Examples are demonstrations, not part of the test suite — keep them readable over exhaustive.
