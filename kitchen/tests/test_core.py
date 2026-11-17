import threading

import numpy as np
import pandas as pd
import pytest
import yaml
from kitchen.config import Config, KitchenError
from kitchen.ledger import Ledger, State

from kitchen import scoring, splits, stats


def test_ledger_allocates_unique_ids_and_survives_a_torn_line(tmp_path):
    led, ids = Ledger(tmp_path), []
    threads = [
        threading.Thread(target=lambda: ids.append(led.new_run({"cmd": "x"}))) for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(ids) == [f"r{i:04d}" for i in range(1, 21)]
    with open(led.path, "a") as fh:
        fh.write('{"ev": "end", "id": "r0001"')  # a crash mid-write
    led.append({"ev": "end", "id": "r0001", "status": "ok", "exit": 0, "seconds": 1})
    led.append({"ev": "mark", "id": "r0001", "note": "n1", "quarantine": "leak"})
    led.append({"ev": "promote", "id": "r0002", "track": "main"})
    st = led.state()
    assert st.runs["r0001"]["status"] == "ok" and st.runs["r0001"]["notes"] == ["n1"]
    assert st.runs["r0001"]["quarantine"] == "leak" and st.champions == {"main": "r0002"}
    assert led.set_champion("main", "r0002").resolve() == led.run_dir("r0002").resolve()


@pytest.mark.parametrize(
    "scheme,extra,folds",
    [
        ("kfold", {"k": 3}, 3),
        ("stratified", {"k": 3}, 3),
        ("group", {"k": 2}, 2),
        ("logo", {}, 6),
        ("forward", {"last": 2}, 2),
        ("holdout", {"holdout": "g == 3"}, 1),
    ],
)
def test_fold_schemes(scheme, extra, folds):
    df = pd.DataFrame({"g": np.repeat(np.arange(6), 10), "y": np.tile([0, 1], 30)})
    spec = {"scheme": scheme, "group": "g", "target": "y", **extra}
    f = splits.assign(df, spec)
    assert (f == splits.assign(df, spec)).all() and len(set(f[f >= 0])) == folds
    if scheme in ("group", "logo", "forward"):
        assert (
            df.assign(f=f).groupby("g")["f"].nunique().eq(1).all()
        )  # a group never straddles folds
    if scheme == "stratified":
        assert df.assign(f=f).groupby("f")["y"].mean().eq(0.5).all()
    if scheme == "forward":
        assert set(df.g[f >= 0]) == {4, 5}


def test_split_is_hashed_immutable_and_supports_composite_ids(tmp_path):
    data = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "season": [2003.0, 2003.0, 2004.0, 2004.0],
            "a": [1, 2, 1, 2],
            "y": [0, 1, 1, 0],
            "unused": 9,
        }
    ).to_csv(data, index=False)
    split = {
        "data": "d.csv",
        "id": ["season", "a"],
        "target": "y",
        "scheme": "logo",
        "group": "season",
    }
    (tmp_path / "kitchen.yaml").write_text(
        yaml.safe_dump(
            {
                "project": "t",
                "splits": {"s": split},
                "tracks": {"main": {"split": "s", "metric": "brier"}},
            }
        )
    )
    cfg = Config.load(tmp_path)
    led = Ledger(cfg.state)
    df, meta = splits.ensure(cfg, led, "s")
    assert list(df["id"]) == ["2003_1", "2003_2", "2004_1", "2004_2"] and "unused" not in df
    assert splits.ensure(cfg, led, "s")[1]["id"] == meta["id"]
    pd.read_csv(data).assign(y=[1, 1, 1, 0]).to_csv(data, index=False)  # one label changes
    with pytest.raises(KitchenError, match="immutable"):
        splits.ensure(cfg, led, "s")


def test_metrics_match_hand_computed_values():
    m, y, p = scoring.BUILTIN, np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    assert m["brier"].fn(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert m["logloss"].fn(y, p) == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert m["auc"].fn(y, p) == pytest.approx(0.75)  # 3 of 4 positive/negative pairs ordered
    assert m["auc"].fn(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.2, 0.9])) == pytest.approx(
        0.875
    )  # tie = 1/2
    assert m["auc_macro"].fn(np.c_[y, y], np.c_[p, p]) == pytest.approx(0.75)
    assert m["accuracy"].fn(y, p) == m["balanced_accuracy"].fn(y, p) == pytest.approx(0.75)
    assert m["f1"].fn(y, p) == pytest.approx(2 / 3)
    r, q = np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.0])
    assert m["mae"].fn(r, q) == pytest.approx(0.5) and m["rmse"].fn(r, q) == pytest.approx(
        np.sqrt(1.25 / 3)
    )
    assert m["r2"].fn(r, q) == pytest.approx(1 - 1.25 / 2)


def _frames(n=600, clusters=20):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(float)
    base = pd.DataFrame(
        {"y:y": y, "season": np.arange(n) % clusters}, index=[str(i) for i in range(n)]
    )
    sharp = np.clip(0.7 * y + 0.15 + rng.normal(0, 0.05, n), 0.01, 0.99)
    noisy = np.clip(0.7 * y + 0.15 + rng.normal(0, 0.3, n), 0.01, 0.99)
    return base.assign(**{"p:pred": sharp}), base.assign(**{"p:pred": noisy})


def test_verdicts_cover_better_worse_tie_and_incomparable():
    fa, fb = _frames()
    meta, brier = {"cluster": ["season"]}, scoring.BUILTIN["brier"]
    v = stats.verdict(fa, fb, "A", brier, meta)
    assert (
        v["verdict"] == "better" and v["clusters"] == 20 and v["ci"][0] <= v["delta"] <= v["ci"][1]
    )
    assert stats.verdict(fb, fa, "A", brier, meta)["verdict"] == "worse"
    assert stats.verdict(fa, fa, "A", brier, meta)["verdict"] == "tie"
    assert (
        stats.verdict(fa, fb, "A", scoring.BUILTIN["auc"], meta)["verdict"] == "better"
    )  # recompute path
    assert stats.verdict(fa, fb.iloc[:-1], "A", brier, meta)["verdict"] == "incomparable"


def test_tier_b_verdict_and_seed_noise_floor():
    fa = pd.DataFrame({"v": [0.9, 0.8, 0.85, 0.95, 0.9]}, index=list("abcde"))
    v = stats.verdict(fa, fa - 0.1, "B", scoring.Metric("aurc", None, "max"), {})
    assert v["verdict"] == "better" and v["delta"] == pytest.approx(0.1)
    runs = [
        {"ev": "run", "id": f"r{i}", "job": "j", "params": {"seed": i, "lr": 0.1}, "t": i}
        for i in range(3)
    ]
    scores = [
        {"ev": "score", "id": f"r{i}", "split": "s@1", "value": x, "t": 9}
        for i, x in enumerate([0.1, 0.2, 0.3])
    ]
    st = State(runs + scores)
    assert stats.seed_sd(st, st.runs["r0"])[0] == pytest.approx(0.1)
    two = State(runs[:2] + scores[:2])
    assert stats.seed_sd(two, two.runs["r0"]) == (None, 2)
