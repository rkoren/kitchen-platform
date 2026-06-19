"""VAL-008: validate that a persistent MLflow backend carries champions across runs (LML-012).

The per-run ``sqlite:///mlruns.db`` default starts with an empty registry every CI run, so
``kitchen run train --auto-promote`` never sees a champion and promotes unconditionally —
cross-run comparison is a no-op. A persistent backend (RDS Postgres + S3 artifacts; see
``docs/decisions/mlflow-tracking-backend.md``, deploy with
``recipes/examples/mlflow-tracking-backend.yaml``) fixes that. This is the acceptance test.

Prerequisites:
  - ``pip install -e 'kitchen/[postgres]'`` (the ``[postgres]`` extra pulls in the psycopg2
    driver MLflow needs for a ``postgresql://`` backend)
  - Point MLflow at your persistent backend (the whole point). Assemble the URL straight from
    the recipes Terraform workspace into ``.env`` (loaded below), then run:
        kitchen secrets db-url --from-terraform ~/.recipes/tf/mlflow-backend-validation --output .env
    optionally add ``MLFLOW_ARTIFACT_BUCKET=<bucket>`` to ``.env`` so artifacts are durable too.

Run it TWICE — each invocation is a separate process, mirroring two CI runs:

    python examples/validate_persistent_backend.py    # run 1: registers a champion
    python examples/validate_persistent_backend.py    # run 2: must FIND it -> PASS

Acceptance criteria:
  - Run 2 finds the champion registered by run 1 (the registry persisted across
    processes) and compares the new run against it — printing a real comparison,
    not "no current champion".

Note: against the default local ``sqlite:///mlruns.db`` the script also "passes" because the
file persists on disk — but that does NOT exercise a remote backend or the ephemeral-runner
scenario CI hits. Use the RDS URL for the real validation. Clean up afterwards with
``recipes destroy`` (deploy the throwaway ``mlflow-backend-validation.yaml`` so teardown isn't
blocked by deletion protection).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.linear_model import LogisticRegression

from kitchen.registry import (
    get_champion_metrics,
    get_production_uri,
    promote_model,
    register_model,
)
from kitchen.tracking import Tracker, configure_from_env, init_experiment

# Load MLFLOW_TRACKING_URI / MLFLOW_ARTIFACT_BUCKET from a .env in the working directory
# (e.g. written by `kitchen secrets db-url --output .env`). usecwd=True so find_dotenv
# searches the CWD, not this script's own directory (the default). kitchen reads these at
# runtime, not import, so loading here (after the imports) is correct and keeps lint happy.
load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

EXPERIMENT = "persistent-backend-validation"
MODEL_NAME = f"{EXPERIMENT}-model"
METRIC = "val_accuracy"


def _masked_backend() -> str:
    """The tracking URI with any password redacted (never print credentials)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return "sqlite:///mlruns.db  (default — see the module docstring note)"
    if "://" in uri and "@" in uri:
        scheme, rest = uri.split("://", 1)
        creds, _, host = rest.partition("@")
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return uri


def _train_one(seed: int) -> tuple[str, float]:
    """Train a tiny model, log it + its accuracy to a fresh MLflow run; return (run_id, accuracy)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = LogisticRegression(max_iter=200).fit(df, y)
    acc = float(model.score(df, y))
    tracker = Tracker(EXPERIMENT)
    with tracker.run(run_name=f"seed-{seed}") as run:
        tracker.log_metrics({METRIC: acc})
        tracker.log_model(model, "model", flavour="sklearn")
        return run.info.run_id, acc


def main() -> None:
    configure_from_env()
    init_experiment(EXPERIMENT)
    print(f"Backend: {_masked_backend()}")

    # The signal: is there already a champion from a previous *process* (= previous CI run)?
    champ_uri = get_production_uri(MODEL_NAME, "champion")
    champ_acc = (get_champion_metrics(MODEL_NAME, "champion") or {}).get(METRIC)

    if champ_uri is None:
        print("\nNo champion in the registry — run 1 (or an ephemeral/empty backend).")
    else:
        print(f"\n✓ PASS: champion persisted across runs — {champ_uri}  ({METRIC}={champ_acc:.4f})")
        print("  The registry survived a separate process, so --auto-promote can compare.")

    # Train a fresh candidate and apply the same compare-then-promote logic as --auto-promote.
    seed = int.from_bytes(os.urandom(2), "big")
    run_id, acc = _train_one(seed)
    print(f"\nNew run {run_id[:8]}: {METRIC}={acc:.4f}")

    if champ_acc is None:
        wins, decision = True, "promote (no current champion)"
    else:
        wins = acc > champ_acc
        decision = (
            f"promote ({acc:.4f} > {champ_acc:.4f})"
            if wins
            else f"skip ({acc:.4f} <= {champ_acc:.4f}) — champion unchanged"
        )
    print(f"auto-promote decision: {decision}")

    if wins:
        version = register_model(run_id, "model", MODEL_NAME)
        promote_model(MODEL_NAME, version, alias="champion")
        print(f"registered + promoted {MODEL_NAME} v{version} @ champion")

    if champ_uri is None:
        print("\nNow run this script again (a second process) — run 2 must find this champion to PASS.")
    else:
        print("\nValidation complete: persistent backend confirmed.")


if __name__ == "__main__":
    main()
