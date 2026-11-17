"""Paired cluster-bootstrap verdicts: is run A better than run B on the same evaluation rows?"""

import numpy as np

from kitchen import scoring, splits
from kitchen.config import KitchenError

B_POINTWISE, B_RECOMPUTE, SEED = 2000, 400, 0


def incomparable(reason: str) -> dict:
    return {"verdict": "incomparable", "reason": reason}


def verdict(fa, fb, tier: str, metric, meta: dict, min_delta: float = 0.0) -> dict:
    """Bootstrap metric(A) - metric(B) over clusters (season/patient/... or rows), paired."""
    if set(fa.index) != set(fb.index):
        return incomparable("different evaluation rows")
    fb = fb.loc[fa.index]
    if tier == "A":
        (y, pa), (_, pb) = scoring.arrays(fa), scoring.arrays(fb)
        cl = meta.get("cluster") or []
        codes = (
            fa.groupby(cl, sort=False, dropna=False).ngroup().to_numpy()
            if cl
            else np.arange(len(fa))
        )
        fn, pointwise = metric.fn, metric.pointwise
    else:  # Tier B: one value per group; the metric is their mean, each group a cluster
        y, pa, pb = None, fa["v"].to_numpy(float), fb["v"].to_numpy(float)
        codes = np.arange(len(fa))
        fn, pointwise = (lambda _, p: float(np.mean(p))), (lambda _, p: p)
    point = fn(y, pa) - fn(y, pb)
    k = int(codes.max()) + 1 if len(codes) else 0
    if k < 2:
        return incomparable("fewer than 2 clusters to resample")
    rng = np.random.default_rng(SEED)
    if pointwise is not None:
        va, vb = pointwise(y, pa), pointwise(y, pb)
        va, vb = (v.mean(axis=1) if np.ndim(v) == 2 else v for v in (va, vb))
        cnt = np.bincount(codes, minlength=k).astype(float)
        sa, sb = np.bincount(codes, va, k), np.bincount(codes, vb, k)
        idx = rng.integers(0, k, size=(B_POINTWISE, k))
        deltas = (sa[idx].sum(1) - sb[idx].sum(1)) / cnt[idx].sum(1)
    else:
        members = [np.flatnonzero(codes == c) for c in range(k)]
        deltas = np.empty(B_RECOMPUTE)
        for i in range(B_RECOMPUTE):
            rows = np.concatenate([members[c] for c in rng.integers(0, k, k)])
            ys = None if y is None else y[rows]
            deltas[i] = fn(ys, pa[rows]) - fn(ys, pb[rows])
    sign = -1.0 if metric.direction == "min" else 1.0  # gain > 0 means A is better
    lo, hi = np.nanpercentile(sign * deltas, [2.5, 97.5])
    if lo > 0 and sign * point >= min_delta:
        label = "better"
    elif hi < 0:
        label = "worse"
    else:
        label = "tie"
    ci = sorted((sign * lo, sign * hi))
    return {
        "verdict": label,
        "delta": scoring.num(point),
        "ci": [scoring.num(c) for c in ci],
        "p_better": round(float(np.nanmean(sign * deltas > 0)), 3),
        "clusters": k,
    }


def compare(cfg, ledger, a: str, b: str) -> dict:
    """Verdict for run a vs run b; both must be scored on the same split and metric."""
    st = ledger.state()
    ra, rb = st.get(a), st.get(b)
    for r in (ra, rb):
        if not r.get("score"):
            return incomparable(f"{r['id']} has no score")
        if r.get("quarantine"):
            return incomparable(f"{r['id']} is quarantined ({r['quarantine']})")
    sa, sb = ra["score"], rb["score"]
    if (sa["split"], sa["metric"]) != (sb["split"], sb["metric"]):
        return incomparable(
            f"different split/metric ({sa['split']} {sa['metric']} vs {sb['split']} {sb['metric']})"
        )
    if min(sa["coverage"], sb["coverage"]) < 1:
        return incomparable("partial coverage of the split's validation rows")
    _, track = cfg.track(ra["track"])
    split_df, meta = splits.load(ledger.dir, track["split"])
    if meta["id"] != sa["split"]:
        return incomparable(f"scored on {sa['split']}, but the split is now {meta['id']}")
    mspec = cfg.metric_spec(track)
    metric = scoring.resolve(mspec, cfg)
    fa, tier = scoring.frame(ledger.run_dir(a), split_df, meta, mspec)
    fb, _ = scoring.frame(ledger.run_dir(b), split_df, meta, mspec)
    if fa is None or fb is None:
        raise KitchenError("a compared run's prediction files are missing from .kitchen/runs")
    min_delta = float((track.get("promote") or {}).get("min_delta", 0.0))
    return {"vs": b, **verdict(fa, fb, tier, metric, meta, min_delta)}


def seed_sd(st, run: dict) -> tuple[float | None, int]:
    """SD of scores across seed replicates of this run's job+params (None if < 3 replicates)."""

    def key(r):
        return (
            r.get("job"),
            r.get("score", {}).get("split"),
            tuple(sorted((k, str(v)) for k, v in (r.get("params") or {}).items() if k != "seed")),
        )

    if "seed" not in (run.get("params") or {}) or not run.get("score"):
        return None, 0
    vals = [
        r["score"]["value"]
        for r in st.runs.values()
        if r.get("score")
        and r["score"].get("value") is not None
        and not r.get("quarantine")
        and "seed" in (r.get("params") or {})
        and key(r) == key(run)
    ]
    return (float(np.std(vals, ddof=1)), len(vals)) if len(vals) >= 3 else (None, len(vals))


def ci_text(v: dict) -> str:
    return f"Δ {v['delta']:+.5f}, CI [{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}]"


def blocker(cfg, ledger, rid: str) -> str | None:
    """Why `rid` may not become its track's champion; None means it may."""
    st = ledger.state()
    r = st.get(rid)
    champ = st.champions.get(r.get("track") or "")
    checks = [
        (r["status"] != "ok", "it did not finish ok"),
        (not r.get("score"), "it has no score"),
        (r.get("quarantine"), f"it is quarantined ({r.get('quarantine')})"),
        (r.get("scratch"), "it is a scratch run (its code is not in git)"),
    ]
    problem = next((msg for bad, msg in checks if bad), None)
    if problem or not champ:
        return problem
    v = compare(cfg, ledger, rid, champ)
    if v["verdict"] != "better":
        return f"not better than champion {champ}: {v['verdict']} ({v.get('reason') or ci_text(v)})"
    sd, n = seed_sd(st, r)
    if sd and abs(v["delta"]) <= 2 * sd:
        return f"its gain ({ci_text(v)}) is within 2x the seed SD ({sd:.5f}, {n} runs)"
    return None
