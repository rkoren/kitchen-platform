"""Tests for kitchen.registry — MLflow Model Registry helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kitchen.registry import get_best_run, get_production_uri, promote_model, register_model

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
        max_results=1,
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
    import mlflow.exceptions

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
