# Modeling Helpers

`kitchen.modeling` provides cross-validation, metric, and ensemble helpers. This page covers when to choose each CV strategy.

## Choosing a CV strategy

| Scenario | Use |
|---|---|
| IID classification / regression (random test split is valid) | `cross_validate` |
| Temporal data — test set is a future time window | `time_series_cv` |
| Grouped data — test set is a held-out group (season, region, cohort) | `loto_cv` |
| Simple train/val split without multi-fold | `train_val_split` |

---

## `cross_validate` — standard K-fold

Use when rows are exchangeable and the test distribution is drawn from the same population as training data (e.g. Titanic, tabular classification without a temporal structure).

```python
from kitchen.modeling import cross_validate, classification_metrics

cv = cross_validate(
    df=train_df,
    target_col="Survived",
    estimator_fn=lambda: XGBClassifier(**params["model"]),
    metric_fn=classification_metrics,
    n_splits=5,
    return_proba=True,
)
# {"accuracy_mean": 0.82, "accuracy_std": 0.03, "f1_mean": ..., ...}
tracker.log_metrics(cv)
```

> **Warning — do not use `cross_validate` on temporal data.**  It uses `StratifiedKFold` internally, which shuffles rows randomly before splitting.  On a time-ordered dataset this leaks future data into training folds and produces inflated, unreliable estimates.  Use `time_series_cv` or `loto_cv` instead.

---

## `time_series_cv` — walk-forward validation

Use when your test set is a future time window and the model must be trained only on data that preceded each validation period. The canonical Kaggle pattern for season-by-season or year-by-year predictions.

For each of the last `n_val_periods` distinct values of `time_col`, the helper trains on all earlier periods and evaluates on the held-out one. Returns per-period metrics alongside aggregate mean and std.

```python
from kitchen.modeling import time_series_cv, classification_metrics

cv = time_series_cv(
    df=train_df,
    time_col="Season",      # sorted distinct values define temporal order
    target_col="won",
    n_val_periods=3,        # validate on the 3 most recent seasons
    trainer_fn=lambda: XGBClassifier(**params["model"]),
    metric_fn=classification_metrics,
)
# {
#   "accuracy_2021": 0.74, "accuracy_2022": 0.76, "accuracy_2023": 0.73,
#   "accuracy_mean": 0.743, "accuracy_std": 0.012,
# }
tracker.log_metrics(cv)
```

**Requires** at least `n_val_periods + 1` distinct values in `time_col` — raises `ValueError` otherwise.

---

## `loto_cv` — leave-one-group-out

Use when each group is structurally distinct and you want to measure how well the model generalises to an unseen group. Common examples: held-out geographic region, user cohort, or tournament season where the full season's data is available but you want group-level holdout rather than a trailing-window split.

For each distinct value of `leave_out_col`, the helper trains on all other groups and evaluates on the left-out one. Returns per-group metrics and aggregate mean and std.

```python
from kitchen.modeling import loto_cv, classification_metrics

cv = loto_cv(
    df=train_df,
    leave_out_col="Season",
    target_col="won",
    trainer_fn=lambda: XGBClassifier(**params["model"]),
    metric_fn=classification_metrics,
)
# {
#   "accuracy_2019": 0.71, "accuracy_2020": 0.74, ...,
#   "accuracy_mean": 0.73, "accuracy_std": 0.02,
# }
tracker.log_metrics(cv)
```

**Difference from `time_series_cv`:** `loto_cv` trains on *all other groups* when evaluating each group, while `time_series_cv` trains only on *earlier periods*. Choose `loto_cv` when group order does not imply a temporal precedence constraint; choose `time_series_cv` when training on future data would be cheating.

---

## Per-fold metrics in MLflow

All three multi-fold helpers return a flat dict keyed by `{metric}_{fold_id}` plus `{metric}_mean` and `{metric}_std`. Log the entire dict with `tracker.log_metrics()` and every key appears in the MLflow run — useful for spotting per-fold regressions even when the aggregate metric improves.

```python
tracker.log_metrics(cv)
# MLflow shows: accuracy_2021, accuracy_2022, accuracy_2023, accuracy_mean, accuracy_std
```
