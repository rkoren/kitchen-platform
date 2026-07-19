"""Tests for kitchen.registry — MLflow Model Registry helpers."""
# pylint: disable=invalid-name  # MockClient is a conventional name for mocked classes in tests

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlflow.exceptions
import pytest

from kitchen.registry import (
    _SCHEME_INSPECT_LIMIT,
    get_best_run,
    get_champion_metrics,
    get_production_uri,
    promote_model,
    register_model,
)

# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------


def test_register_model_constructs_uri():
    mv = MagicMock()
    mv.version = "3"
    with patch("mlflow.register_model", return_value=mv) as mock_reg:
        result = register_model("abc123", "model", "my-model")
    mock_reg.assert_called_once_with("runs:/abc123/model", "my-model")
    assert result == "3"


def test_register_model_returns_version_string():
    mv = MagicMock()
    mv.version = "7"
    with patch("mlflow.register_model", return_value=mv):
        assert register_model("run1", "artifacts/model", "name") == "7"


def test_register_model_wrong_name_lists_available(monkeypatch):
    monkeypatch.setattr(
        "kitchen.registry._run_logged_model_names",
        lambda _run_id: ["cbb_model", "calibrator"],
    )
    with patch(
        "mlflow.register_model",
        side_effect=mlflow.exceptions.MlflowException("Unable to find a logged_model"),
    ):
        with pytest.raises(mlflow.exceptions.MlflowException) as exc:
            register_model("abc12345", "model", "cbb-tournament-model")
    msg = str(exc.value)
    assert "cbb_model" in msg and "calibrator" in msg
    assert "mlflow.model_artifact_path" in msg


def test_register_model_reraises_when_lookup_fails(monkeypatch):
    # If we can't enumerate logged models, surface the original MLflow error unchanged.
    monkeypatch.setattr("kitchen.registry._run_logged_model_names", lambda _run_id: None)
    with patch(
        "mlflow.register_model",
        side_effect=mlflow.exceptions.MlflowException("original boom"),
    ):
        with pytest.raises(mlflow.exceptions.MlflowException, match="original boom"):
            register_model("abc12345", "model", "name")


# ---------------------------------------------------------------------------
# get_best_run
# ---------------------------------------------------------------------------


def _mock_client(exp=MagicMock(), runs=None):
    client = MagicMock()
    client.get_experiment_by_name.return_value = exp
    client.search_runs.return_value = runs if runs is not None else [MagicMock()]
    return client


def test_get_best_run_raises_when_experiment_missing():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        MockClient.return_value.get_experiment_by_name.return_value = None
        with pytest.raises(ValueError, match="not found"):
            get_best_run("missing-exp", "val_brier")


def test_get_best_run_raises_when_no_runs():
    exp = MagicMock()
    exp.experiment_id = "1"
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        MockClient.return_value.get_experiment_by_name.return_value = exp
        MockClient.return_value.search_runs.return_value = []
        with pytest.raises(ValueError, match="No runs found"):
            get_best_run("my-exp", "val_brier")


def test_get_best_run_lower_is_better_uses_asc():
    exp = MagicMock()
    exp.experiment_id = "42"
    run = MagicMock()
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = [run]
        result = get_best_run("my-exp", "val_brier", lower_is_better=True)
    client.search_runs.assert_called_once_with(
        experiment_ids=["42"],
        filter_string="",
        order_by=["metrics.val_brier ASC"],
        max_results=_SCHEME_INSPECT_LIMIT,
    )
    assert result is run


def test_get_best_run_higher_is_better_uses_desc():
    exp = MagicMock()
    exp.experiment_id = "5"
    run = MagicMock()
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = [run]
        get_best_run("my-exp", "val_accuracy", lower_is_better=False)
    _, kwargs = client.search_runs.call_args
    assert kwargs["order_by"] == ["metrics.val_accuracy DESC"]


def test_get_best_run_applies_tag_filter():
    exp = MagicMock()
    exp.experiment_id = "9"
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = [MagicMock()]
        get_best_run("my-exp", "val_brier", tag_filter={"model_variant": "challenger"})
    _, kwargs = client.search_runs.call_args
    assert "tags.model_variant = 'challenger'" in kwargs["filter_string"]


def test_get_best_run_error_mentions_tags_when_filter_set():
    exp = MagicMock()
    exp.experiment_id = "3"
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = []
        with pytest.raises(ValueError, match="tags"):
            get_best_run("my-exp", "val_brier", tag_filter={"model_variant": "x"})


# get_best_run — validation-scheme comparability guard (S6E7-002)


def _scheme_run(metric_value: float, scheme: str | None, metric: str = "val_accuracy"):
    """A run carrying a real metrics dict + optional validation_scheme tag (real dicts so
    ``metric in run.data.metrics`` and ``tags.get`` behave like production, not MagicMock)."""
    run = MagicMock()
    run.data.metrics = {metric: metric_value}
    run.data.tags = {"validation_scheme": scheme} if scheme else {}
    return run


def _run_get_best(runs, **kwargs):
    exp = MagicMock()
    exp.experiment_id = "1"
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = runs
        return get_best_run("my-exp", "val_accuracy", lower_is_better=False, **kwargs)


def test_get_best_run_refuses_when_competitive_runs_span_schemes():
    # A split-based 0.95001 must NOT silently beat an out-of-fold 0.94981 — different schemes.
    runs = [_scheme_run(0.95001, "holdout-0.2"), _scheme_run(0.94981, "oof-5fold")]
    with pytest.raises(ValueError, match="multiple validation schemes"):
        _run_get_best(runs)


def test_get_best_run_allows_single_scheme():
    runs = [_scheme_run(0.95, "oof-5fold"), _scheme_run(0.94, "oof-5fold")]
    assert _run_get_best(runs) is runs[0]


def test_get_best_run_untagged_runs_skip_the_guard():
    # Backward compatible: runs without a validation_scheme tag never trigger a refusal.
    runs = [_scheme_run(0.95, None), _scheme_run(0.94, None)]
    assert _run_get_best(runs) is runs[0]


def test_get_best_run_scheme_filter_bypasses_guard():
    # Narrowing to one scheme via tag_filter makes the pool comparable — no raise even if the
    # returned candidates carry differing tags (the server-side filter is what scopes them).
    runs = [_scheme_run(0.95, "holdout-0.2"), _scheme_run(0.94, "oof-5fold")]
    result = _run_get_best(runs, tag_filter={"validation_scheme": "oof-5fold"})
    assert result is runs[0]


# ---------------------------------------------------------------------------
# promote_model
# ---------------------------------------------------------------------------


def test_promote_model_sets_alias():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        promote_model("my-model", "4", alias="champion")
    client.set_registered_model_alias.assert_called_once_with("my-model", "champion", "4")


def test_promote_model_default_alias_is_champion():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        promote_model("my-model", "2")
    _, args, _ = client.set_registered_model_alias.mock_calls[0]
    assert args[1] == "champion"


def test_promote_model_custom_alias():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        promote_model("my-model", "5", alias="challenger")
    client.set_registered_model_alias.assert_called_once_with("my-model", "challenger", "5")


# ---------------------------------------------------------------------------
# get_production_uri
# ---------------------------------------------------------------------------


def test_get_production_uri_returns_uri_when_alias_exists():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        MockClient.return_value.get_model_version_by_alias.return_value = MagicMock()
        result = get_production_uri("my-model", "champion")
    assert result == "models:/my-model@champion"


def test_get_production_uri_returns_none_when_alias_missing():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        MockClient.return_value.get_model_version_by_alias.side_effect = (
            mlflow.exceptions.MlflowException("not found")
        )
        result = get_production_uri("my-model", "champion")
    assert result is None


def test_get_production_uri_default_alias_is_champion():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        MockClient.return_value.get_model_version_by_alias.return_value = MagicMock()
        get_production_uri("my-model")
    MockClient.return_value.get_model_version_by_alias.assert_called_once_with(
        "my-model", "champion"
    )


def test_get_production_uri_custom_alias():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        MockClient.return_value.get_model_version_by_alias.return_value = MagicMock()
        result = get_production_uri("my-model", alias="staging")
    assert result == "models:/my-model@staging"


# ---------------------------------------------------------------------------
# get_champion_metrics  (GH-011)
# ---------------------------------------------------------------------------


def test_get_champion_metrics_returns_run_metrics_excluding_fi_and_lb():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.get_model_version_by_alias.return_value = MagicMock(run_id="run123")
        client.get_run.return_value.data.metrics = {
            "val_accuracy": 0.91,
            "val_log_loss": 0.30,
            "fi.feature_a": 0.5,  # feature importance — excluded
            "lb_score": 0.80,     # carried separately — excluded
        }
        result = get_champion_metrics("my-model")
    assert result == {"val_accuracy": 0.91, "val_log_loss": 0.30}


def test_get_champion_metrics_returns_none_when_no_champion():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        MockClient.return_value.get_model_version_by_alias.side_effect = (
            mlflow.exceptions.MlflowException("alias not found")
        )
        result = get_champion_metrics("my-model")
    assert result is None


def test_get_champion_metrics_returns_none_when_run_missing():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.get_model_version_by_alias.return_value = MagicMock(run_id="gone")
        client.get_run.side_effect = mlflow.exceptions.MlflowException("run not found")
        result = get_champion_metrics("my-model")
    assert result is None


def test_get_champion_metrics_default_alias_is_champion():
    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.get_model_version_by_alias.return_value = MagicMock(run_id="r")
        client.get_run.return_value.data.metrics = {}
        get_champion_metrics("my-model")
    client.get_model_version_by_alias.assert_called_once_with("my-model", "champion")
