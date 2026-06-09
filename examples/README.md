# Examples

Runnable scripts that demonstrate individual `kitchen` features end-to-end. Each
script has a module docstring with its full prerequisites and acceptance criteria;
this index is the map.

The examples fall into two groups by what they need to run.

## Complete project (start here)

[`offline-quickstart/`](offline-quickstart/) is a full `kitchen` project — features →
train → evaluate → promote — that runs end-to-end with **no credentials and no
network**, using tiny bundled data and local SQLite tracking. It's the smallest honest
version of the whole loop and the best place to see how `params.yaml` and the
`FeatureBuilder` / `Trainer` / `Evaluator` contract fit together. See its
[README](offline-quickstart/README.md) for the three-command walkthrough.

## Run anywhere (synthetic data, no project setup)

These build their own in-memory data and only need `kitchen` installed
(`pip install -e kitchen/`). Run them straight from the repo root:

| Script | Demonstrates |
|---|---|
| [`loto_cv_example.py`](loto_cv_example.py) | `loto_cv()` leave-one-group-out CV — one metric key per group plus aggregate mean/std, and how `kitchen leaderboard --expand-metrics` surfaces each group as a sub-column. |
| [`multi_source_features.py`](multi_source_features.py) | `FeatureBuilder.sources()` routing multiple raw CSVs into `build()` as a `dict[filename, DataFrame]`, producing the same processed parquet as a single-file builder. |

```bash
python examples/loto_cv_example.py
python examples/multi_source_features.py
```

## Require a project + data (CBB harnesses)

These drive the real CLI against an existing project (the College Basketball model is
the reference) and assume processed data and prior runs exist. They are validation
harnesses for the v0.6.0 acceptance tests, not standalone demos — read each script's
docstring for the exact setup, and set `PROJECT_DIR` (or run from the project root).

| Script | Demonstrates |
|---|---|
| [`cbb_param_sweep.py`](cbb_param_sweep.py) | `kitchen run train --override model.max_depth=N` produces runs with the overridden param, an `override.*` tag, and the metric logged; then `kitchen diff` shows `max_depth` as a changed param. |
| [`cbb_dashboard_params.py`](cbb_dashboard_params.py) | `kitchen dashboard generate` renders param columns (`--show-params`), the champion highlight hook, and the Kaggle LB-score column. |
| [`cbb_diff_feature_importance.py`](cbb_diff_feature_importance.py) | `kitchen diff <run_a> <run_b>` surfaces a feature-importance section listing features whose rank changed between two runs. |

## Conventions for new examples

- One feature (or one acceptance test) per script, with a docstring covering
  **what it shows**, **prerequisites**, and **how to run it**.
- Prefer synthetic, self-contained data so a new contributor can run the example
  with nothing but `kitchen` installed; reserve project-dependent harnesses for
  cases that genuinely need real data.
- Examples are demonstrations, not part of the test suite — keep them readable over
  exhaustive.
