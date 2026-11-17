"""Metric registry and run scoring against a frozen split: kitchen computes every score.

Tier A: the run writes oof.parquet|csv (split id column + `pred`, or one column per target) and
kitchen scores it against the split's own labels. Tier B: the run writes scores.csv (`group`,
value) computed by the project's evaluator; kitchen checks the groups match the split's.
"""

import importlib.util
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from kitchen.config import KitchenError
from kitchen.splits import key, listify, load, read_table


@dataclass
class Metric:
    name: str
    fn: Callable
    direction: str  # "min" | "max"
    pointwise: Callable | None = None  # per-row values whose mean is the metric (fast bootstrap)


def _labels(p):
    p = np.asarray(p)
    return (p >= 0.5).astype(int) if np.issubdtype(p.dtype, np.floating) else p


def _auc(y, p):
    y = np.asarray(y).astype(bool)
    pos, neg = y.sum(), (~y).sum()
    if not pos or not neg:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()  # average ranks handle ties
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _auc_macro(y, p):
    y, p = np.atleast_2d(np.asarray(y).T).T, np.atleast_2d(np.asarray(p).T).T
    return float(np.nanmean([_auc(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def _balanced_accuracy(y, p):
    y, hat = np.asarray(y), _labels(p)
    return float(np.mean([np.mean(hat[y == c] == c) for c in np.unique(y)]))


def _f1(y, p):
    y, hat = np.asarray(y).astype(int), _labels(p)
    tp, fp, fn = (
        np.sum((hat == 1) & (y == 1)),
        np.sum((hat == 1) & (y == 0)),
        np.sum((hat == 0) & (y == 1)),
    )
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan")


def _ll(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


BUILTIN = {
    m.name: m
    for m in [
        Metric(
            "brier", lambda y, p: float(np.mean((p - y) ** 2)), "min", lambda y, p: (p - y) ** 2
        ),
        Metric("logloss", lambda y, p: float(np.mean(_ll(y, p))), "min", _ll),
        Metric(
            "mae", lambda y, p: float(np.mean(np.abs(p - y))), "min", lambda y, p: np.abs(p - y)
        ),
        Metric("rmse", lambda y, p: float(np.sqrt(np.mean((p - y) ** 2))), "min"),
        Metric(
            "r2",
            lambda y, p: float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)),
            "max",
        ),
        Metric(
            "accuracy",
            lambda y, p: float(np.mean(_labels(p) == y)),
            "max",
            lambda y, p: (_labels(p) == y).astype(float),
        ),
        Metric("balanced_accuracy", _balanced_accuracy, "max"),
        Metric("f1", _f1, "max"),
        Metric("auc", _auc, "max"),
        Metric("auc_macro", _auc_macro, "max"),
    ]
}


def resolve(spec: dict, cfg) -> Metric:
    """Built-in metric by name, or a custom `fn: file.py:func` with a required `direction`."""
    name = spec.get("name")
    if spec.get("fn"):
        file, _, func = str(spec["fn"]).partition(":")
        if spec.get("direction") not in ("min", "max"):
            raise KitchenError(f"custom metric {name or func!r} needs `direction: min|max`")
        mod_spec = importlib.util.spec_from_file_location("kitchen_metric", cfg.path(file))
        if mod_spec is None or not func:
            raise KitchenError(
                f"metric fn must look like path/to/file.py:function, got {spec['fn']!r}"
            )
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        return Metric(name or func, getattr(mod, func), spec["direction"])
    if name not in BUILTIN:
        raise KitchenError(
            f"unknown metric {name!r} (built in: {', '.join(BUILTIN)}; or `fn` + `direction`)"
        )
    m = BUILTIN[name]
    return Metric(m.name, m.fn, spec.get("direction", m.direction), m.pointwise)


def frame(
    run_dir, split_df: pd.DataFrame, meta: dict, mspec: dict
) -> tuple[pd.DataFrame | None, str | None]:
    """The split's validation rows joined to a run's predictions.

    Tier A -> (index=id, columns y:<label>..., p:<pred>..., fold, clusters, segments), "A".
    Tier B -> (index=group as str, column v), "B".  No outputs -> (None, None).
    """
    val = split_df[split_df["fold"] >= 0]
    idc = meta["id_col"]
    oof = next((run_dir / f for f in ("oof.parquet", "oof.csv") if (run_dir / f).exists()), None)
    if oof is not None:
        pred = read_table(oof)
        parts = meta.get("id_parts") or [idc]
        if len(parts) > 1 and idc not in pred.columns and set(parts) <= set(pred.columns):
            pred[idc] = key(pred, parts)  # composite id rebuilt from its parts
        labels = listify(mspec.get("target")) or meta["targets"]
        pcols = listify(mspec.get("pred")) or (
            ["pred"] if "pred" in pred.columns and len(labels) == 1 else labels
        )
        missing = [c for c in [idc, *pcols] if c not in pred.columns]
        if missing:
            raise KitchenError(
                f"{oof.name} lacks column(s) {missing} (write {idc!r} + `pred`, or one column per target)"
            )
        if pred[idc].duplicated().any():
            raise KitchenError(f"{oof.name} has duplicate {idc!r} values")
        p = pred[[idc, *pcols]].rename(columns={c: f"p:{c}" for c in pcols}).astype({idc: str})
        val = val.rename(columns={c: f"y:{c}" for c in labels}).astype(
            {idc: str}
        )  # ids compare as text
        f = val.merge(p, on=idc, how="left")
        return f.set_index(idc), "A"
    scores = run_dir / "scores.csv"
    if scores.exists():
        s = pd.read_csv(scores)
        value = next((c for c in s.columns if c != "group"), None)
        if not meta.get("group") or "group" not in s.columns or value is None:
            raise KitchenError(
                "scores.csv needs `group` + a value column, and a split with `group`"
            )
        groups = pd.Index(pd.unique(val[meta["group"]]).astype(str), name="group")
        s = s.assign(group=s["group"].astype(str)).set_index("group")
        return pd.DataFrame(index=groups).join(s[[value]].rename(columns={value: "v"})), "B"
    return None, None


def arrays(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ys = [c for c in f.columns if c.startswith("y:")]
    ps = [c for c in f.columns if c.startswith("p:")]
    y, p = f[ys].to_numpy(float), f[ps].to_numpy(float)
    return (y[:, 0], p[:, 0]) if y.shape[1] == 1 else (y, p)


def num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 6)


def score(cfg, ledger, rid: str, track_name: str) -> dict | None:
    """Score a finished run on its track's frozen split; None if it wrote no predictions."""
    _, track = cfg.track(track_name)
    split_df, meta = load(ledger.dir, track["split"])
    mspec = cfg.metric_spec(track)
    metric = resolve(mspec, cfg)
    f, tier = frame(ledger.run_dir(rid), split_df, meta, mspec)
    if f is None:
        return None
    if tier == "A":
        ok = f[[c for c in f.columns if c.startswith("p:")]].notna().all(axis=1)
        fc = f[ok]
        value = metric.fn(*arrays(fc)) if len(fc) else float("nan")
        folds = {str(k): num(metric.fn(*arrays(g))) for k, g in fc.groupby("fold")}
        segments = {
            f"{s}={k}": num(metric.fn(*arrays(g)))
            for s in meta["segments"]
            for k, g in fc.groupby(s)
        }
    else:
        ok = f["v"].notna()
        value, folds, segments = float(f.loc[ok, "v"].mean()) if ok.any() else float("nan"), {}, {}
    return {
        "split": meta["id"],
        "metric": metric.name,
        "direction": metric.direction,
        "value": num(value),
        "n": int(ok.sum()),
        "coverage": round(float(ok.mean()), 4) if len(ok) else 0.0,
        "tier": tier,
        "folds": folds,
        "segments": segments,
    }
