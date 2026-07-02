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
    from kitchen.menu import stage_module_name

    build = __import__(stage_module_name("features", params), fromlist=["build"]).build
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


def _score_holdout(run_id: str, params: dict) -> dict[str, float]:
    """CBB-017: score the just-trained model on the frozen holdout, logging ``holdout_<metric>``.

    Runs after the model artifact is logged (re-loads it from the run). A no-op (``{}``) when no
    ``holdout:`` is configured or the parquet isn't there yet. Best-effort: a scoring failure is
    reported loudly but never fails an otherwise-successful train run — the model is already
    trained and logged; the holdout number is a supplement.
    """
    import sys

    from kitchen.holdout import score_run_holdout

    artifact_path = (params.get("mlflow") or {}).get("model_artifact_path", "model")
    try:
        return score_run_holdout(run_id, params, model_artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001 — supplementary metric must not sink a good run
        print(f"warning: holdout scoring failed: {exc}", file=sys.stderr)
        return {}


def _holdout_lines(results: dict[str, float]) -> list[str]:
    """One ``holdout_<metric> = … (n=…, never-trained-on)`` line per scored holdout metric."""
    n = int(results.get("holdout_n", 0))
    return [
        f"  {key} = {val:.6g}  (n={n}, never-trained-on)"
        for key, val in sorted(results.items())
        if key != "holdout_n"
    ]


def _train(params: dict, overrides: dict | None = None, variant: str | None = None) -> str | None:
    from kitchen.menu import stage_module_name

    train = __import__(stage_module_name("train", params), fromlist=["train"]).train
    configure_from_env()
    experiment = params.get("experiment", EXPERIMENT)
    init_experiment(experiment)
    tracker = Tracker(experiment)
    with tracker.run(run_name=params.get("run_name"), params=params) as active_run:
        if variant:
            mlflow.set_tag("model_variant", variant)  # leaderboard/diff/status group by this
        if overrides:
            mlflow.set_tags({f"override.{k}": str(v) for k, v in overrides.items()})
        train(params, DataStore(), tracker)
        run_id = active_run.info.run_id

    holdout = _score_holdout(run_id, params) if run_id else {}

    summary = _metric_lines(run_id, params) if run_id else []
    summary += _holdout_lines(holdout)
    if summary:
        print("Training complete:")
        for line in summary:
            print(line)
    else:
        print("Training complete — see MLflow for metrics.")
    return run_id


def train_pipeline(
    params_file: str = "params.yaml",
    overrides: dict | None = None,
    variant: str | None = None,
) -> str | None:
    """Run a single training pass: features → train → log to MLflow.

    Composition order is base config → ``--variant`` overlay → ``--override`` scalars (so an
    override always wins on a conflict). The ``variants:`` definition itself is dropped from
    the params before training so it isn't logged as a run param — the choice is captured by
    the ``model_variant`` tag instead.

    Returns the MLflow run_id of the training run (used by ``kitchen sweep`` to
    rank runs); ``None`` only if the tracker did not expose an active run.
    """
    from kitchen.menu import apply_variant, load_params

    params = load_params(params_file)
    if variant:
        apply_variant(params, variant)  # raises VariantNotFound (caller renders it)
    params.pop("variants", None)  # a menu definition, not a run param
    if overrides:
        _apply_overrides(params, overrides)
    _build(params)
    return _train(params, overrides=overrides, variant=variant)


if __name__ == "__main__":
    train_pipeline()
