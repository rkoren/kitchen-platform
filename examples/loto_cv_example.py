"""VAL-004: loto_cv per-group metric validation.

Demonstrates that loto_cv() emits one metric key per group plus aggregate
mean/std, and that the key structure matches what kitchen leaderboard
--expand-metrics expects.

This example uses synthetic data so it can be run without any project setup:

    python examples/loto_cv_example.py

Acceptance criteria:
- One {metric}_{group} key per distinct value of leave_out_col
- aggregate {metric}_mean and {metric}_std keys present
- Per-group keys excluded from aggregate (no double-counting)
- kitchen leaderboard --expand-metrics would surface each group key as a
  sub-column (demonstrated here by printing the expected column names)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kitchen.modeling import classification_metrics, loto_cv


def build_synthetic_dataset(n_per_group: int = 120, seed: int = 42) -> pd.DataFrame:
    """Temporal dataset: rows belong to one of several seasons.

    A simple linear signal means a well-fitted model should achieve ~0.75+
    accuracy per season, giving a realistic loto_cv output to inspect.
    """
    rng = np.random.default_rng(seed)
    dfs = []
    for season in range(2019, 2024):
        X = rng.normal(size=(n_per_group, 5))
        # Add a mild season-level shift so per-season accuracy varies slightly
        X[:, 0] += (season - 2021) * 0.1
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        df_s = pd.DataFrame(X, columns=[f"f{i}" for i in range(1, 6)])
        df_s["season"] = season
        df_s["target"] = y
        dfs.append(df_s)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    df = build_synthetic_dataset()
    groups = sorted(df["season"].unique())
    print(f"Dataset: {len(df)} rows, groups (seasons): {groups}\n")

    cv = loto_cv(
        df=df,
        leave_out_col="season",
        target_col="target",
        trainer_fn=lambda: LogisticRegression(max_iter=300),
        metric_fn=classification_metrics,
        return_proba=True,
    )

    # Per-group rows
    print("Per-season results:")
    per_group_keys = sorted(k for k in cv if not k.endswith(("_mean", "_std")))
    for key in per_group_keys:
        if key.startswith("accuracy_"):
            print(f"  {key}: {cv[key]:.4f}")

    print(f"\nAggregate accuracy: {cv['accuracy_mean']:.4f}  (std {cv['accuracy_std']:.4f})")
    print(f"Aggregate f1:       {cv['f1_mean']:.4f}  (std {cv['f1_std']:.4f})")

    # Acceptance checks -------------------------------------------------------

    # One accuracy key per season
    accuracy_per_season = [k for k in cv if k.startswith("accuracy_") and not k.endswith(("_mean", "_std"))]
    assert len(accuracy_per_season) == len(groups), (
        f"Expected {len(groups)} per-season keys, got {len(accuracy_per_season)}"
    )

    # Aggregate keys present
    assert "accuracy_mean" in cv
    assert "accuracy_std" in cv

    # Keys match the {base}_{group} pattern that --expand-metrics parses
    expected_suffixes = {str(g) for g in groups}
    actual_suffixes = {k.removeprefix("accuracy_") for k in accuracy_per_season}
    assert actual_suffixes == expected_suffixes, (
        f"Suffix mismatch: {actual_suffixes} != {expected_suffixes}"
    )

    # Show what leaderboard --expand-metrics would display
    print("\nExpand-metrics columns leaderboard would show:")
    for s in sorted(actual_suffixes):
        print(f"  accuracy_{s}")

    print("\nVAL-004 passed: loto_cv emits correct per-group and aggregate metric keys.")


if __name__ == "__main__":
    main()
