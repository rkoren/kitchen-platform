"""Generate a Kaggle submission from the champion, then validate it — stopping just
short of the upload.

This is the filled-in twin of the `flows/generate_submission.py` that `kitchen init`
scaffolds (with the flavour/feature TODOs resolved for this competition). It:

1. loads the champion (`models:/spaceship-titanic-model@champion`) from the local
   MLflow registry — so run `kitchen menu run` (which trains + promotes) first;
2. engineers `test.csv` with the *same* `_engineer()` the training features use;
3. writes `submissions/submission.csv` (`PassengerId,Transported`);
4. validates it against `sample_submission.csv` with the platform's own
   `validate_submission` (the exact check `kitchen submit` runs before uploading).

It deliberately **stops before uploading**. The synthetic bundle's PassengerIds are
fabricated, so it validates but can't be sent to the real competition — the closing
message spells out the real-data path that can. Run it from anywhere:

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
        "\nStopping before upload. This synthetic bundle uses fabricated PassengerIds, so it\n"
        f"is well-formed but NOT submittable to the real '{competition}' competition.\n"
        "To exercise the full upload path, swap in the real data first:\n"
        f"  kitchen init --source kaggle --competition {competition}   # accept the rules first\n"
        "  kitchen menu run -C examples/spaceship-titanic\n"
        "  python examples/spaceship-titanic/flows/generate_submission.py\n"
        "  kitchen submit --file submissions/submission.csv          # run from the project dir"
    )
    return out


if __name__ == "__main__":
    generate()
