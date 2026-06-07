"""VAL-007: ``kitchen diff`` feature-importance section validation.

Verifies that ``kitchen diff <run_a> <run_b>`` (CMP-004) surfaces a feature
importance section comparing the two runs' logged ``feature_importances.json``
artifacts (auto-logged by ``Trainer.run()`` via M-007), and that at least one
feature's importance rank changed by more than 3 positions between the runs.

There is a deliberate distinction between two things this script checks:

1. **CLI surface** — does ``kitchen diff`` render a "Feature importance" section
   at all? The command shows the top 5 features by absolute rank change with no
   magnitude floor, so this passes whenever both runs logged importances and any
   rank changed.
2. **VAL-007 acceptance condition** — is there at least one feature whose rank
   moved by *more than 3* positions? This is a property of the *chosen runs*, not
   the platform: two near-identical runs (same params, same seed) can produce a
   section with only tiny shuffles. The script reports these two checks
   separately so a failure is attributable to the right cause.

Prerequisites
-------------
- ``kitchen`` installed (``pip install -e kitchen/``).
- A CBB project with **two** completed ``kitchen run train`` runs whose models
  expose feature importances (XGBoost / LightGBM / sklearn tree models all do).
  The cleanest pair is the VAL-002 sweep's extremes — e.g.
  ``kitchen run train --override model.max_depth=4`` and
  ``--override model.max_depth=8`` — which differ enough to move feature ranks.
- Run from the project root (or set ``PROJECT_DIR``).

Usage
-----
    cd /path/to/cbb-model
    kitchen run train --override model.max_depth=4
    kitchen run train --override model.max_depth=8
    python /path/to/kitchen-platform/examples/cbb_diff_feature_importance.py

Acceptance criteria (VAL-007)
-----------------------------
1. Two runs that each logged a ``feature_importances.json`` artifact are found.
2. ``kitchen diff`` output includes a "Feature importance" section.
3. At least one feature changed rank by more than 3 positions between the runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import mlflow.artifacts
import mlflow.tracking

_FI_ARTIFACT = "feature_importances.json"


def _has_fi(client: "mlflow.tracking.MlflowClient", run_id: str) -> bool:
    return any(a.path == _FI_ARTIFACT for a in client.list_artifacts(run_id))


def _load_fi(run_id: str) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=_FI_ARTIFACT, dst_path=tmp
        )
        with open(path) as f:
            return json.load(f)


def _ranks(fi: dict[str, float]) -> dict[str, int]:
    # Match kitchen diff: rank 1 = highest importance; ties broken by name.
    ordered = sorted(fi.items(), key=lambda x: (-x[1], x[0]))
    return {name: i + 1 for i, (name, _) in enumerate(ordered)}


def _pick_pair(client, runs):
    """Prefer two runs that differ on override.model.max_depth (VAL-002 pair).

    Falls back to the two most recent FI-bearing runs.
    """
    fi_runs = [r for r in runs if _has_fi(client, r.info.run_id)]
    if len(fi_runs) < 2:
        return None

    by_depth: dict[str, object] = {}
    for r in fi_runs:
        depth = r.data.tags.get("override.model.max_depth")
        if depth is not None and depth not in by_depth:
            by_depth[depth] = r
    if len(by_depth) >= 2:
        depths = sorted(by_depth, key=lambda d: int(d) if d.isdigit() else 0)
        return by_depth[depths[0]], by_depth[depths[-1]]

    return fi_runs[0], fi_runs[1]


def main() -> int:
    project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
    params_file = os.path.join(project_dir, "params.yaml")

    from kitchen.tracking import configure_from_env

    configure_from_env()

    try:
        import yaml

        with open(params_file) as f:
            params = yaml.safe_load(f)
        experiment_name = params.get("experiment", "default")
    except FileNotFoundError:
        print(f"[error] params.yaml not found at {params_file}")
        print("  Run from the project root or set PROJECT_DIR.")
        return 1

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"[error] MLflow experiment {experiment_name!r} not found.")
        print("  Train at least two runs with `kitchen run train` first.")
        return 1

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=20,
    )
    pair = _pick_pair(client, runs)
    if pair is None:
        print(f"[error] Need two runs with a {_FI_ARTIFACT!r} artifact in {experiment_name!r}.")
        print("  Run `kitchen run train` twice (e.g. --override model.max_depth=4 and =8).")
        return 1

    run_a, run_b = pair
    a_id, b_id = run_a.info.run_id, run_b.info.run_id
    a_depth = run_a.data.tags.get("override.model.max_depth", "?")
    b_depth = run_b.data.tags.get("override.model.max_depth", "?")

    print(f"\nExperiment: {experiment_name}")
    print(f"  a  {a_id[:8]}  (max_depth={a_depth})")
    print(f"  b  {b_id[:8]}  (max_depth={b_depth})\n")

    # --- Compute rank changes ourselves to evaluate the >3 acceptance condition --
    rank_a = _ranks(_load_fi(a_id))
    rank_b = _ranks(_load_fi(b_id))
    shared = set(rank_a) & set(rank_b)
    big_moves = sorted(
        (
            (abs(rank_b[f] - rank_a[f]), rank_a[f], rank_b[f], f)
            for f in shared
            if abs(rank_b[f] - rank_a[f]) > 3
        ),
        reverse=True,
    )

    # --- Invoke `kitchen diff` and capture its output ---------------------------
    proc = subprocess.run(
        [sys.executable, "-m", "kitchen.cli", "diff", a_id, b_id],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    diff_out = proc.stdout
    print(diff_out, end="")

    # --- Acceptance checks ------------------------------------------------------
    print("\n--- Acceptance checks ---")

    ok1 = _has_fi(client, a_id) and _has_fi(client, b_id)
    print(f"[{'OK' if ok1 else 'FAIL'}] Both runs logged {_FI_ARTIFACT!r}")

    ok2 = "Feature importance" in diff_out
    print(f"[{'OK' if ok2 else 'FAIL'}] `kitchen diff` rendered a 'Feature importance' section")

    ok3 = bool(big_moves)
    label3 = "OK" if ok3 else "FAIL"
    print(f"[{label3}] At least one feature changed rank by more than 3 positions")
    if ok3:
        for delta, ra, rb, feat in big_moves[:5]:
            print(f"        {feat}: rank {ra} → {rb}  (Δ{rb - ra:+d})")
    elif ok2:
        print("        Section rendered, but the chosen runs are too similar to move")
        print("        any rank by >3 — re-run against a wider max_depth spread.")

    passed = ok1 and ok2 and ok3
    print(f"\n{'PASS' if passed else 'FAIL'}: VAL-007")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
