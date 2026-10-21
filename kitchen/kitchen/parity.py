"""Champion ↔ processed-data feature parity (CBB-027).

``kitchen run evaluate`` hands the champion to the project's ``evaluate()``, which reads some
``data/processed/`` frame and predicts. When that frame was rebuilt by a different variant than
the champion, the project dies in raw pandas (``KeyError: "[...] not in index"``) while
``check``/``status`` report green. This module checks up front — from the loaded model's own
feature lists and the processed parquets' schemas (no data is read) — and names the gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _str_list(value: Any) -> list[str] | None:
    """``value`` as a list of feature names, or ``None`` if it isn't one (mocks, scalars, …)."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif type(value).__name__ == "ndarray":
        items = value.tolist()
    else:
        return None
    if items and all(isinstance(v, str) for v in items):
        return items
    return None


def model_feature_names(model: Any) -> list[str]:
    """Every feature the model will read, from its own attributes — empty if it declares none.

    Probes ``features``, ``feature_names_in_`` (sklearn), ``feature_names`` and any public
    ``*_features`` attribute (multi-head models, e.g. ``margin_features`` + ``total_features``),
    and unions them in first-seen order. Only real lists of strings count, so a model that
    doesn't expose a feature list (or a test double) yields ``[]`` and the guard is skipped.
    """
    names = ["features", "feature_names_in_", "feature_names"]
    names += sorted(
        n for n in dir(type(model)) + list(getattr(model, "__dict__", {}))
        if n.endswith("_features") and not n.startswith("_")
    )
    seen: list[str] = []
    for name in dict.fromkeys(names):
        try:
            value = getattr(model, name, None)
        except Exception:  # noqa: BLE001 — a property that raises just isn't a feature list
            continue
        for feature in _str_list(value) or []:
            if feature not in seen:
                seen.append(feature)
    return seen


@dataclass(frozen=True)
class ParityResult:
    """Outcome of :func:`check_processed_parity`."""

    ok: bool
    file: Path | None = None  # the processed file that covers every feature (ok) or comes closest
    missing: tuple[str, ...] = ()
    tied: tuple[Path, ...] = ()


def check_processed_parity(
    features: list[str], processed_dir: Path, schema_file: str | None = None
) -> ParityResult | None:
    """Does any single processed parquet carry every feature? ``None`` = nothing to check.

    ``schema_file`` (the ``feature_schema.file`` contract, KG-014) pins the file; otherwise
    every ``*.parquet`` under ``processed_dir`` is a candidate. Only schemas are read.
    """
    if not features:
        return None
    if schema_file:
        candidates = [processed_dir / schema_file]
    else:
        candidates = sorted(processed_dir.glob("*.parquet")) if processed_dir.is_dir() else []
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None

    import pyarrow.parquet as pq

    closest: list[tuple[Path, tuple[str, ...]]] = []
    for path in candidates:
        try:
            columns = set(pq.read_schema(path).names)
        except Exception:  # noqa: BLE001 — an unreadable file can't be the feature matrix
            continue
        missing = tuple(f for f in features if f not in columns)
        if not missing:
            return ParityResult(ok=True, file=path)
        if not closest or len(missing) < len(closest[0][1]):
            closest = [(path, missing)]
        elif len(missing) == len(closest[0][1]):
            closest.append((path, missing))
    if not closest:
        return None
    (path, missing), rest = closest[0], closest[1:]
    return ParityResult(ok=False, file=path, missing=missing, tied=tuple(p for p, _ in rest))


def parity_remedy(variant: str | None, model: str | None = None) -> str:
    """The fix to suggest when processed data can't feed the champion.

    Two causes, two fixes: the processed data came from another variant (rebuild it), or the
    feature set itself has moved on since the champion trained (retrain + promote). ``model``
    (a ``models:`` key) wins over ``variant`` in the suggested command, since a multi-model
    project's ``run train`` requires ``--model``.
    """
    if model:
        rebuild = (
            f"rebuild processed data for model {model!r} "
            f"(`kitchen run train --model {model}` re-runs features first)"
        )
    elif variant:
        rebuild = (
            f"rebuild processed data with the champion's variant {variant!r} "
            f"(`kitchen run train --variant {variant}` re-runs features first)"
        )
    else:
        rebuild = "re-run `kitchen run features` with the params the champion trained on"
    return (
        f"{rebuild}, or — if those features were since dropped — retrain and promote a new "
        "champion"
    )
