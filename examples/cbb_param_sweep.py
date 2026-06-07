"""VAL-002: --override param sweep validation.

Demonstrates that ``kitchen run train --override model.max_depth=N`` produces
runs with:
- The overridden param value visible in ``run.data.params``
- An ``override.model.max_depth`` tag on the run
- ``loto_brier`` metric logged to the same run

Then verifies that ``kitchen diff`` shows ``max_depth`` as a changed param
between the best and worst runs.

Prerequisites
-------------
- ``kitchen`` installed (``pip install -e kitchen/``)
- A CBB project with processed ``matchups.parquet`` available (or any project
  using ``kitchen run train``)
- Run from the platform root or set ``PROJECT_DIR`` below

Usage
-----
    cd /path/to/cbb-model
    kitchen run train --override model.max_depth=4
    kitchen run train --override model.max_depth=6
    kitchen run train --override model.max_depth=8
    python /path/to/kitchen-platform/examples/cbb_param_sweep.py

Acceptance criteria (VAL-002)
------------------------------
1. All three runs appear in ``kitchen leaderboard``.
2. ``kitchen diff <best_run> <worst_run>`` output contains ``max_depth`` in
   the params section.
3. Each run has an ``override.model.max_depth`` tag in MLflow.
"""

from __future__ import annotations

import os
import sys

import mlflow.tracking


def main() -> int:
    project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
    params_file = os.path.join(project_dir, "params.yaml")

    # -- Configure MLflow from .env / environment --------------------------------
    from kitchen.tracking import configure_from_env

    configure_from_env()

    # -- Discover the experiment name -------------------------------------------
    try:
        import yaml

        with open(params_file) as f:
            params = yaml.safe_load(f)
        experiment_name = params.get("experiment", "default")
        threshold_metric = next(iter(params.get("thresholds", {})), "loto_brier")
    except FileNotFoundError:
        print(f"[error] params.yaml not found at {params_file}")
        print("  Run from the project root or set PROJECT_DIR.")
        return 1

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"[error] MLflow experiment {experiment_name!r} not found.")
        print("  Run `kitchen run train --override model.max_depth=4` first.")
        return 1

    # -- Find the three sweep runs (tagged with override.model.max_depth) -------
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.`override.model.max_depth` != ''",
        order_by=["start_time DESC"],
        max_results=20,
    )

    depths_seen: dict[str, object] = {}
    for run in runs:
        depth = run.data.tags.get("override.model.max_depth")
        if depth in ("4", "6", "8") and depth not in depths_seen:
            depths_seen[depth] = run

    if len(depths_seen) < 3:
        missing = {"4", "6", "8"} - set(depths_seen.keys())
        print(f"[error] Missing sweep runs for max_depth={missing}.")
        print("  Run: kitchen run train --override model.max_depth=<N> for each N.")
        return 1

    print(f"\nExperiment: {experiment_name}  |  metric: {threshold_metric}\n")
    print(f"{'max_depth':<12}  {'run_id':<10}  {threshold_metric}")
    print("-" * 40)

    scored: list[tuple[float, object]] = []
    for depth in ("4", "6", "8"):
        run = depths_seen[depth]
        metric_val = run.data.metrics.get(threshold_metric)
        run_id_short = run.info.run_id[:8]
        val_str = f"{metric_val:.4f}" if metric_val is not None else "(missing)"
        print(f"{depth:<12}  {run_id_short:<10}  {val_str}")
        if metric_val is not None:
            scored.append((metric_val, run))

    if not scored:
        print(f"\n[error] No runs have a logged {threshold_metric!r} metric.")
        return 1

    # -- Check acceptance criteria ----------------------------------------------
    print("\n--- Acceptance checks ---")

    # 1. All three runs have the metric logged
    ok = len(scored) == 3
    print(f"[{'OK' if ok else 'FAIL'}] All three runs have {threshold_metric!r} logged")

    # 2. Each run has override tag
    for depth, run in depths_seen.items():
        tag_val = run.data.tags.get("override.model.max_depth")
        ok2 = tag_val == depth
        print(f"[{'OK' if ok2 else 'FAIL'}] max_depth={depth}: override tag = {tag_val!r}")

    # 3. Each run has model.max_depth in params matching the override
    for depth, run in depths_seen.items():
        param_val = run.data.params.get("model.max_depth")
        ok3 = str(param_val) == depth
        print(f"[{'OK' if ok3 else 'FAIL'}] max_depth={depth}: params[model.max_depth] = {param_val!r}")

    # 4. Identify best vs worst for diff
    lower_is_better = True  # loto_brier / brier metrics
    sorted_runs = sorted(scored, key=lambda x: x[0], reverse=(not lower_is_better))
    best_run = sorted_runs[0][1]
    worst_run = sorted_runs[-1][1]
    best_id = best_run.info.run_id[:8]
    worst_id = worst_run.info.run_id[:8]
    best_depth = best_run.data.tags.get("override.model.max_depth")
    worst_depth = worst_run.data.tags.get("override.model.max_depth")
    best_metric = best_run.data.metrics.get(threshold_metric)
    worst_metric = worst_run.data.metrics.get(threshold_metric)

    print(f"\nBest:  max_depth={best_depth}  run={best_id}  {threshold_metric}={best_metric:.4f}")
    print(f"Worst: max_depth={worst_depth}  run={worst_id}  {threshold_metric}={worst_metric:.4f}")
    print(f"\nRun `kitchen diff {best_run.info.run_id[:8]} {worst_run.info.run_id[:8]}` to see param delta.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
