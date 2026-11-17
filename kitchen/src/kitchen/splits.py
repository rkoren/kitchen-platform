"""Frozen fold assignments: built once from kitchen.yaml, content-hashed, never rewritten.

fold >= 0 marks a validation row (its fold id); fold -1 is train-only (forward / holdout).
"""

import hashlib
import json
import re

import numpy as np
import pandas as pd

from kitchen.config import KitchenError


def listify(v) -> list:
    return [] if v is None else list(v) if isinstance(v, (list, tuple)) else [v]


def read_table(path, columns: list[str] | None = None) -> pd.DataFrame:
    name = str(path)
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, columns=columns)
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, usecols=columns)
    raise KitchenError(f"can't read {path} (use .parquet or .csv)")


def columns(path) -> list[str]:
    if str(path).endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    return list(pd.read_csv(path, nrows=0).columns)


def key(df: pd.DataFrame, parts: list[str]) -> pd.Series:
    """A composite id as text; integral floats render without '.0' so different writers agree."""
    out = None
    for c in parts:
        s = df[c]
        if pd.api.types.is_float_dtype(s) and (s.dropna() % 1 == 0).all():
            s = s.astype("Int64")
        out = s.astype(str) if out is None else out + "_" + s.astype(str)
    return out


def assign(df: pd.DataFrame, spec: dict) -> np.ndarray:
    scheme, k, n = spec["scheme"], int(spec.get("k", 5)), len(df)
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    folds = np.full(n, -1)
    if scheme == "kfold":
        folds[rng.permutation(n)] = np.arange(n) % k
    elif scheme == "stratified":  # deal each class round-robin over its shuffled rows
        y = df[listify(spec["target"])[0]].to_numpy()
        for cls in sorted(pd.unique(y), key=str):
            idx = rng.permutation(np.flatnonzero(y == cls))
            folds[idx] = np.arange(len(idx)) % k
    elif scheme == "holdout":  # rows matching `holdout` (a query) or a random `fraction` -> fold 0
        mask = (
            df.eval(spec["holdout"]).to_numpy()
            if spec.get("holdout")
            else rng.random(n) < float(spec.get("fraction", 0.2))
        )
        folds[mask] = 0
    else:
        g = df[spec["group"]]
        if scheme == "group":  # greedy size-balanced assignment, deterministic tie-break
            load, fmap = np.zeros(k), {}
            for grp, size in sorted(g.value_counts().items(), key=lambda kv: (-kv[1], str(kv[0]))):
                fmap[grp] = int(load.argmin())
                load[fmap[grp]] += size
        else:
            groups = sorted(pd.unique(g))
            if scheme == "forward":  # validate on each of the last `last` groups
                groups = groups[-int(spec.get("last", 5)) :]
            fmap = {grp: i for i, grp in enumerate(groups)}
        folds = g.map(fmap).fillna(-1).astype(int).to_numpy()
    return folds


def _paths(state, name):
    return state / "splits" / f"{name}.parquet", state / "splits" / f"{name}.json"


def ensure(cfg, ledger, name: str) -> tuple[pd.DataFrame, dict]:
    """Build split `name` from its definition; refuse if a different version already exists."""
    if name not in cfg.splits:
        raise KitchenError(f"unknown split {name!r} (have: {', '.join(cfg.splits) or 'none'})")
    spec, source = cfg.splits[name], cfg.path(cfg.splits[name]["data"])
    parts, targets = listify(spec["id"]), listify(spec["target"])
    idc = parts[0] if len(parts) == 1 else "id"  # a composite id becomes an `id` text column
    cluster = listify(spec.get("cluster")) or listify(spec.get("group"))
    segments = listify(spec.get("segments"))
    cols = list(dict.fromkeys([*parts, *targets, *listify(spec.get("group")), *cluster, *segments]))
    if not source.exists():
        raise KitchenError(f"split {name}: data file {source} not found")
    available = columns(source)
    missing = [c for c in cols if c not in available]
    if missing:
        raise KitchenError(f"split {name}: {spec['data']} has no column(s) {missing}")
    exprs = [e for e in (spec.get("where"), spec.get("holdout")) if e]
    named = {a or b for e in exprs for a, b in re.findall(r"`([^`]+)`|([A-Za-z_]\w*)", e)}
    df = read_table(source, [c for c in available if c in set(cols) | named])  # only what's needed
    try:
        df = df.query(spec["where"]) if spec.get("where") else df
        folds = assign(df, spec)
    except Exception as e:
        raise KitchenError(f"split {name}: {type(e).__name__}: {e}") from None
    out = df[cols].reset_index(drop=True)
    if len(parts) > 1:
        out.insert(0, "id", key(out, parts))
    if out[idc].duplicated().any():
        raise KitchenError(f"split {name}: id {spec['id']!r} has duplicates")
    out["fold"] = folds
    blob = out.sort_values(idc)[[idc, "fold", *targets]].to_csv(index=False).encode()
    sid = f"{name}@{hashlib.sha256(blob).hexdigest()[:8]}"
    path, meta_path = _paths(ledger.dir, name)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["id"] != sid:
            raise KitchenError(
                f"split {name} changed ({meta['id']} -> {sid}); splits are immutable, so give the "
                f"new definition a new name (e.g. {name}-v2) and re-run the champion on it"
            )
        return pd.read_parquet(path), meta
    val = out[out["fold"] >= 0]
    meta = {"id": sid, "name": name, "id_col": idc, "id_parts": parts, "targets": targets}
    meta |= {"cluster": cluster, "segments": segments, "rows": len(out), "val_rows": len(val)}
    meta |= {k: spec.get(k) for k in ("scheme", "group", "data")}
    meta["folds"] = int(val["fold"].nunique())
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    ledger.append({"ev": "split", **meta})
    return out, meta


def load(state, name: str) -> tuple[pd.DataFrame, dict]:
    path, meta_path = _paths(state, name)
    if not meta_path.exists():
        raise KitchenError(f"split {name!r} is not built yet (run `kitchen split {name}`)")
    return pd.read_parquet(path), json.loads(meta_path.read_text())
