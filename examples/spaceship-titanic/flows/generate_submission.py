"""Generate a Kaggle submission from the champion, then validate it.

This is the filled-in twin of the `flows/generate_submission.py` that `kitchen init` scaffolds
(with the flavour/feature TODOs resolved for this competition). Run `kitchen menu run` first to
train + promote a champion, then, **from the project directory**:

    python flows/generate_submission.py

It loads the champion, engineers `test.csv` with the *same* `_engineer()` training uses, writes
`submissions/submission.csv`, and validates it against `sample_submission.csv`.

It is **notebook-safe** (GEN-005a): the real work is in `generate()`, which imports project/heavy
modules itself and uses paths relative to the working directory — no module-level `__file__` /
`sys.path` / `os.chdir`. That bootstrap lives in the `if __name__ == "__main__":` block below (so
the script still runs from any directory), and `kitchen submit --notebook` can wrap `generate()`
into a Kaggle submission notebook.
"""
from __future__ import annotations

from pathlib import Path


def generate(menu_file: str = "menu.yaml") -> Path:
    """Load the champion, predict on the test set, write + validate the submission CSV.

    Paths are relative to the current working directory (the project root), so this runs unchanged
    both as a script (see ``__main__``) and inside a `kitchen submit --notebook` notebook.
    """
    import mlflow
    import pandas as pd
    import yaml

    from kitchen.registry import get_production_uri
    from kitchen.store import DataStore
    from kitchen.submit import check_feature_parity, validate_submission
    from src.features.run import FEATURES, _engineer

    params = yaml.safe_load(Path(menu_file).read_text())
    project = params["project"]
    id_col = params["submission"]["id_col"]
    target_col = params["submission"]["target_col"]

    import os

    model_name = os.environ.get("MLFLOW_MODEL_NAME", f"{project}-model")
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    store = DataStore()

    # 1. Champion (trained + promoted by `kitchen menu run`).
    uri = get_production_uri(model_name)
    if uri is None:
        raise SystemExit(
            f"No champion found for {model_name!r}. Run `kitchen menu run` first to train + promote one."
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
    out = Path("submissions/submission.csv")
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
    # Standalone convenience: make `src` importable and run project-relative regardless of the
    # caller's cwd. (This block is excluded when `kitchen submit --notebook` wraps `generate()`.)
    import os
    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    generate()
