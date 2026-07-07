"""Generate a Kaggle submission from the champion, then validate it.

This is the filled-in twin of the `flows/generate_submission.py` that `kitchen init`
scaffolds (with the flavour/feature TODOs resolved for this competition). It:

1. loads the champion (`models:/spaceship-titanic-model@champion`) from the local
   MLflow registry — so run `kitchen menu run` (which trains + promotes) first;
2. engineers `test.csv` with the *same* `_engineer()` the training features use;
3. writes `submissions/submission.csv` (`PassengerId,Transported`);
4. validates it against `sample_submission.csv` with the platform's own
   `validate_submission` (the exact check `kitchen submit` runs before uploading).

Spaceship Titanic is a Getting Started *sandbox* (unlimited submissions, no stakes),
so the resulting file is real and submittable — `kitchen submit` uploads it under
your own Kaggle account. Fetch the data first with `kitchen ingest`. Run from anywhere:

    python examples/spaceship-titanic/flows/generate_submission.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Run project-relative regardless of the caller's cwd: `src` importable, data/ + mlruns.db found.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

import mlflow  # noqa: E402 — after sys.path/cwd setup
from src.features.run import FEATURES, _engineer  # noqa: E402

from kitchen.registry import get_production_uri  # noqa: E402
from kitchen.store import DataStore  # noqa: E402
from kitchen.submit import check_feature_parity, validate_submission  # noqa: E402


def generate(menu_file: str = "menu.yaml") -> Path:
    params = yaml.safe_load(Path(menu_file).read_text())
    project = params["project"]
    id_col = params["submission"]["id_col"]
    target_col = params["submission"]["target_col"]
    model_name = os.environ.get("MLFLOW_MODEL_NAME", f"{project}-model")

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    store = DataStore()

    # 1. Champion (trained + promoted by `kitchen menu run`).
    uri = get_production_uri(model_name)
    if uri is None:
        raise SystemExit(
            f"No champion found for {model_name!r}. Run `kitchen menu run "
            f"-C {PROJECT}` first to train and promote one."
        )
    model = mlflow.pyfunc.load_model(uri)  # flavour-agnostic → class labels for this classifier

    # 2. Engineer the test set exactly as training does, and guard feature parity (KG-013).
    test_raw = store.load_csv(params["features"]["test_file"])
    test_df = _engineer(test_raw)[FEATURES]
    parity_errors = check_feature_parity(FEATURES, test_df)
    if parity_errors:
        raise SystemExit("Feature parity failed:\n  " + "\n  ".join(parity_errors))

    # 3. Predict → the real competition wants True/False booleans.
    preds = pd.Series(model.predict(test_df)).astype(int).astype(bool)
    submission = pd.DataFrame({id_col: test_raw[id_col].to_numpy(), target_col: preds.to_numpy()})
    out = PROJECT / "submissions" / "submission.csv"
    out.parent.mkdir(exist_ok=True)
    submission.to_csv(out, index=False)
    print(f"Wrote {len(submission)} predictions → {out}")

    # 4. Validate against the sample submission — the same gate `kitchen submit` runs.
    sample = pd.read_csv(store.raw_dir / params["submission"]["sample_submission"])
    errors = validate_submission(submission, sample, id_col, target_col)
    if errors:
        raise SystemExit("Submission validation failed:\n  " + "\n  ".join(errors))

    competition = params["submission"]["competition"]
    print(f"Validated {len(submission)} rows against sample_submission — submission is well-formed.")
    print(f"Transported rate: {submission[target_col].mean():.2f}")
    print(
        f"\nReady to submit to the '{competition}' sandbox (unlimited submissions, no stakes).\n"
        "It uploads under your own Kaggle account — from the project directory run:\n"
        "  kitchen submit --file submissions/submission.csv\n"
        "Add --wait to poll for the public leaderboard score."
    )
    return out


if __name__ == "__main__":
    generate()
