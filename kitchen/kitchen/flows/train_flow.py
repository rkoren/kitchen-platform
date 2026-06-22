"""Generic single-run training flow for kitchen competition projects.

Scaffolded projects import this via ``flows/train_flow.py``. It is a thin
wrapper that reads ``params.yaml``, runs feature engineering, and trains
the model — delegating all project-specific logic to the project's own
``src.features.run`` and ``src.train.run`` modules.

Run from the project root:
    python flows/train_flow.py
"""

from __future__ import annotations

import os

import mlflow
from dotenv import load_dotenv

from kitchen.store import DataStore
from kitchen.tracking import Tracker, configure_from_env, init_experiment

load_dotenv()

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "default")


def _apply_overrides(params: dict, overrides: dict) -> None:
    """Apply dot-notation key=value overrides to params in-place.

    Creates missing intermediate dicts so --override new_section.key=v works
    even when new_section is absent from params.yaml.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        target = params
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


def _build(params: dict) -> None:
    from src.features.run import build  # project-provided

    build(params, DataStore())


def _metric_lines(run_id: str, params: dict) -> list[str]:
    """One line per threshold-gated metric with PASS/FAIL; else the headline metrics.

    Gives the train success path something to show without opening MLflow (CBB-011).
    Threshold direction follows the params convention: a bare number / ``min`` is a
    lower bound (``>=``), ``max`` an upper bound (``<=``). Per-feature-importance
    (``fi.*``) keys are excluded. Best-effort: returns ``[]`` if the run can't be read.
    """
    try:
        metrics = {
            k: v for k, v in mlflow.get_run(run_id).data.metrics.items() if not k.startswith("fi.")
        }
    except Exception:
        return []
    if not metrics:
        return []

    thresholds = params.get("thresholds") or {}
    lines: list[str] = []
    for name, spec in thresholds.items():
        actual = metrics.get(name)
        if actual is None:
            continue
        lo = spec.get("min") if isinstance(spec, dict) else (spec if isinstance(spec, (int, float)) else None)
        hi = spec.get("max") if isinstance(spec, dict) else None
        bounds, fails = [], False
        if lo is not None:
            bounds.append(f">= {lo:g}")
            fails = fails or actual < lo
        if hi is not None:
            bounds.append(f"<= {hi:g}")
            fails = fails or actual > hi
        bound_str = f" ({' and '.join(bounds)})" if bounds else ""
        lines.append(f"  {name} = {actual:.6g} — {'FAIL' if fails else 'PASS'}{bound_str}")
    if lines:
        return lines

    # No thresholds configured — show a bounded set of headline metrics so the
    # user still sees numbers (per-period families like brier_2003.. are capped).
    shown = sorted(metrics.items())[:8]
    lines = [f"  {k} = {v:.6g}" for k, v in shown]
    if len(metrics) > len(shown):
        lines.append(f"  (+{len(metrics) - len(shown)} more — see kitchen leaderboard)")
    return lines


def _train(params: dict, overrides: dict | None = None) -> str | None:
    from src.train.run import train  # project-provided

    configure_from_env()
    experiment = params.get("experiment", EXPERIMENT)
    init_experiment(experiment)
    tracker = Tracker(experiment)
    with tracker.run(run_name=params.get("run_name"), params=params) as active_run:
        if overrides:
            mlflow.set_tags({f"override.{k}": str(v) for k, v in overrides.items()})
        train(params, DataStore(), tracker)
        run_id = active_run.info.run_id

    summary = _metric_lines(run_id, params) if run_id else []
    if summary:
        print("Training complete:")
        for line in summary:
            print(line)
    else:
        print("Training complete — see MLflow for metrics.")
    return run_id


def train_pipeline(params_file: str = "params.yaml", overrides: dict | None = None) -> str | None:
    """Run a single training pass: features → train → log to MLflow.

    Returns the MLflow run_id of the training run (used by ``kitchen sweep`` to
    rank runs); ``None`` only if the tracker did not expose an active run.
    """
    from kitchen.menu import load_params

    params = load_params(params_file)
    if overrides:
        _apply_overrides(params, overrides)
    _build(params)
    return _train(params, overrides=overrides)


if __name__ == "__main__":
    train_pipeline()
