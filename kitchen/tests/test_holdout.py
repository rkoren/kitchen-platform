"""Tests for kitchen.holdout — frozen-holdout scoring (CBB-017)."""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from kitchen.holdout import _predict, score_run_holdout


@pytest.fixture()
def logged_clf(tmp_path):
    """A LogisticRegression logged under 'model' in a fresh sqlite store; yields (run_id, store)."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlruns.db")
    mlflow.set_experiment("holdout-exp")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=300), "f2": rng.normal(size=300)})
    y = (X["f1"] + 0.5 * X["f2"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
    return run_id, tmp_path


def _write_holdout(path, rows=40, *, label="Outcome", cols=("f1", "f2"), with_label=True):
    rng = np.random.default_rng(7)
    data = {c: rng.normal(size=rows) for c in cols}
    if with_label:
        data[label] = rng.integers(0, 2, rows)
    df = pd.DataFrame(data)
    df.to_parquet(path)
    return df


def _params(holdout: dict | None, features=("f1", "f2")):
    p: dict = {"feature_candidates": list(features)}
    if holdout is not None:
        p["holdout"] = holdout
    return p


# ── happy path ────────────────────────────────────────────────────────────────


def test_scores_and_logs_holdout_brier(logged_clf):
    run_id, tmp = logged_clf
    hp = tmp / "holdout.parquet"
    _write_holdout(hp, rows=40)
    res = score_run_holdout(run_id, _params({"path": str(hp), "label": "Outcome"}))
    assert set(res) == {"holdout_brier", "holdout_n"}
    assert res["holdout_n"] == 40.0
    assert 0.0 <= res["holdout_brier"] <= 1.0
    # logged onto the same run, under a name distinct from any CV metric (CBB-019)
    logged = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics
    assert logged["holdout_brier"] == pytest.approx(res["holdout_brier"])


def test_features_default_to_feature_candidates(logged_clf):
    run_id, tmp = logged_clf
    hp = tmp / "holdout.parquet"
    _write_holdout(hp)
    # No holdout.features → reuse feature_candidates.
    res = score_run_holdout(run_id, _params({"path": str(hp), "label": "Outcome"}))
    assert "holdout_brier" in res


def test_explicit_holdout_features_override(logged_clf):
    run_id, tmp = logged_clf
    hp = tmp / "holdout.parquet"
    _write_holdout(hp)
    # feature_candidates deliberately wrong; holdout.features names the real ones.
    params = _params(
        {"path": str(hp), "label": "Outcome", "features": ["f1", "f2"]}, features=["bogus"]
    )
    res = score_run_holdout(run_id, params)
    assert "holdout_brier" in res


# ── no-op / skip paths ──────────────────────────────────────────────────────────


def test_no_holdout_config_is_noop(logged_clf):
    run_id, _ = logged_clf
    assert score_run_holdout(run_id, _params(None)) == {}


def test_absent_optional_path_is_noop(logged_clf):
    run_id, tmp = logged_clf
    res = score_run_holdout(
        run_id, _params({"path": str(tmp / "nope.parquet"), "label": "Outcome"})
    )
    assert res == {}


def test_absent_required_path_raises(logged_clf):
    run_id, tmp = logged_clf
    with pytest.raises(FileNotFoundError):
        score_run_holdout(
            run_id,
            _params({"path": str(tmp / "nope.parquet"), "label": "Outcome", "optional": False}),
        )


def test_parity_break_skips_loudly(logged_clf, caplog):
    run_id, tmp = logged_clf
    hp = tmp / "holdout.parquet"
    _write_holdout(hp, cols=("f1",))  # f2 missing → model feature absent
    with caplog.at_level(logging.WARNING):
        res = score_run_holdout(run_id, _params({"path": str(hp), "label": "Outcome"}))
    assert res == {}  # never a silently-wrong (zero-filled) trusted metric
    assert "SKIPPING" in caplog.text and "f2" in caplog.text


# ── misconfiguration surfaces (raises, not silent) ──────────────────────────────


def test_missing_label_column_raises(logged_clf):
    run_id, tmp = logged_clf
    hp = tmp / "holdout.parquet"
    _write_holdout(hp, with_label=False)
    with pytest.raises(ValueError, match="holdout.label"):
        score_run_holdout(run_id, _params({"path": str(hp), "label": "Outcome"}))


def test_no_resolvable_features_raises(logged_clf):
    run_id, tmp = logged_clf
    hp = tmp / "holdout.parquet"
    _write_holdout(hp)
    # No holdout.features and no feature_candidates.
    with pytest.raises(ValueError, match="no features"):
        score_run_holdout(run_id, {"holdout": {"path": str(hp), "label": "Outcome"}})


# ── segments: named subpopulations (CBB-025) ────────────────────────────────────


def _write_segmented_holdout(path):
    """40 rows with a ``grp`` column: 20 ``hit`` (label 1) + 20 ``miss`` (label 0), all with the
    same strongly-positive features → the model predicts ~1 for every row, so the label-1 segment
    scores far better than the label-0 one. A misaligned mask (or a per-segment re-predict) would
    not reproduce that clean split."""
    hit = {"f1": [3.0] * 20, "f2": [0.0] * 20, "Outcome": [1] * 20, "grp": ["hit"] * 20}
    miss = {"f1": [3.0] * 20, "f2": [0.0] * 20, "Outcome": [0] * 20, "grp": ["miss"] * 20}
    df = pd.concat([pd.DataFrame(hit), pd.DataFrame(miss)], ignore_index=True)
    df.to_parquet(path)
    return df


def test_segments_score_and_log(logged_clf):
    run_id, tmp = logged_clf
    hp = tmp / "seg.parquet"
    _write_segmented_holdout(hp)
    res = score_run_holdout(
        run_id,
        _params(
            {
                "path": str(hp),
                "label": "Outcome",
                "segments": {
                    "hit": {"col": "grp", "eq": "hit"},
                    "miss": {"col": "grp", "eq": "miss"},
                },
            }
        ),
    )
    # full-set metric plus a metric + count per segment
    assert "holdout_brier" in res
    assert res["holdout_n_hit"] == 20.0 and res["holdout_n_miss"] == 20.0
    # predict-once, sliced-correctly: the label-1 segment beats the label-0 one cleanly.
    assert res["holdout_brier_hit"] < 0.25 < res["holdout_brier_miss"]
    # logged onto the run under the exact name --promote-metric would use
    logged = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics
    assert logged["holdout_brier_hit"] == pytest.approx(res["holdout_brier_hit"])


def test_segment_missing_column_raises(logged_clf):
    run_id, tmp = logged_clf
    hp = tmp / "h.parquet"
    _write_holdout(hp)  # has f1, f2, Outcome — no "grp" column
    with pytest.raises(ValueError, match=r"holdout\.segments\.women"):
        score_run_holdout(
            run_id,
            _params(
                {
                    "path": str(hp),
                    "label": "Outcome",
                    "segments": {"women": {"col": "grp", "eq": "W"}},
                }
            ),
        )


def test_segment_zero_rows_skipped(logged_clf, caplog):
    run_id, tmp = logged_clf
    hp = tmp / "seg.parquet"
    _write_segmented_holdout(hp)
    with caplog.at_level(logging.WARNING):
        res = score_run_holdout(
            run_id,
            _params(
                {
                    "path": str(hp),
                    "label": "Outcome",
                    "segments": {"ghost": {"col": "grp", "eq": "nope"}},
                }
            ),
        )
    assert "holdout_brier" in res  # full-set intact
    assert not any(k.startswith("holdout_brier_") for k in res)  # segment not emitted
    assert "0 rows" in caplog.text and "ghost" in caplog.text


def test_segment_uncomputable_metric_skipped(logged_clf, caplog):
    run_id, tmp = logged_clf
    hp = tmp / "seg.parquet"
    _write_segmented_holdout(hp)  # full set has both classes; the "hit" segment is single-class
    with caplog.at_level(logging.WARNING):
        res = score_run_holdout(
            run_id,
            _params(
                {
                    "path": str(hp),
                    "label": "Outcome",
                    "metric": "roc_auc",
                    "segments": {"hit": {"col": "grp", "eq": "hit"}},
                }
            ),
        )
    assert "holdout_roc_auc" in res  # full-set computes (both classes present)
    assert "holdout_roc_auc_hit" not in res  # single-class segment skipped
    assert "hit" in caplog.text and "not computable" in caplog.text


# ── regression metric + predict_method override (unit, no MLflow) ────────────────


def test_regression_metric_uses_predict(logged_clf, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/reg.db")
    mlflow.set_experiment("holdout-reg")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    y = 2 * X["f1"] - X["f2"]
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(LinearRegression().fit(X, y), "model")
        run_id = run.info.run_id
    hp = tmp_path / "h.parquet"
    pd.DataFrame(
        {"f1": rng.normal(size=25), "f2": rng.normal(size=25), "y": rng.normal(size=25)}
    ).to_parquet(hp)
    res = score_run_holdout(
        run_id, _params({"path": str(hp), "label": "y", "metric": "rmse"})
    )
    assert "holdout_rmse" in res and res["holdout_rmse"] >= 0.0


class _BatchModel:
    """A model exposing a custom predict surface (like cbb's composite predict_batch)."""

    def predict_batch(self, df):
        return np.full(len(df), 0.3)


def test_predict_method_override_calls_named_method():
    df = pd.DataFrame({"f1": [0.0, 1.0, 2.0]})
    out = _predict(_BatchModel(), df, metric="brier", predict_method="predict_batch")
    assert out.tolist() == [0.3, 0.3, 0.3]


def test_predict_proba_reduced_to_positive_class():
    class _Proba:
        def predict_proba(self, df):
            return np.column_stack([np.full(len(df), 0.7), np.full(len(df), 0.3)])

    out = _predict(_Proba(), pd.DataFrame({"f1": [0, 1]}), metric="brier", predict_method=None)
    assert out.tolist() == [0.3, 0.3]  # positive-class column
