"""Tests for kitchen.menu_resolve — {from_role} resolution + materialize (INT-003)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kitchen import menu_resolve
from kitchen.menu import Menu

BACKEND_MENU = Menu.model_validate(
    {
        "project": "cbb",
        "recipes": {
            "mlflow-backend": {"kind": "rds", "role": "mlflow-backend", "db_name": "tracking"},
            "mlflow-artifacts": {"kind": "s3", "role": "mlflow-artifacts"},
        },
        "mlflow": {
            "tracking_uri": {"from_role": "mlflow-backend"},
            "artifact_bucket": {"from_role": "mlflow-artifacts"},
        },
    }
)


def test_recipes_workspace_path():
    assert menu_resolve.recipes_workspace("cbb").as_posix().endswith(".recipes/tf/cbb")


def test_resolve_rds_role_to_tracking_uri_and_s3_role_to_bucket():
    with patch(
        "kitchen.secrets.db_url_from_terraform", return_value="postgresql://mlflow:p@h:5432/tracking"
    ) as dburl:
        env = menu_resolve.resolve_mlflow_env(BACKEND_MENU, tf_dir="/ws")
    assert env["MLFLOW_TRACKING_URI"] == "postgresql://mlflow:p@h:5432/tracking"
    assert env["MLFLOW_ARTIFACT_BUCKET"] == "mlflow-artifacts"  # the s3 recipe's name
    # the rds recipe name + its db_name are threaded into the LML-015 helper
    dburl.assert_called_once_with("/ws", rds="mlflow-backend", db="tracking")


def test_literal_settings_pass_through():
    menu = Menu.model_validate(
        {"project": "p", "mlflow": {"tracking_uri": "sqlite:///mlruns.db", "artifact_bucket": "my-bucket"}}
    )
    env = menu_resolve.resolve_mlflow_env(menu)
    assert env == {"MLFLOW_TRACKING_URI": "sqlite:///mlruns.db", "MLFLOW_ARTIFACT_BUCKET": "my-bucket"}


def test_tracking_uri_role_must_be_rds():
    menu = Menu.model_validate(
        {
            "project": "p",
            "recipes": {"bucket": {"kind": "s3", "role": "store"}},
            "mlflow": {"tracking_uri": {"from_role": "store"}},
        }
    )
    with pytest.raises(ValueError, match="must point at an `rds` recipe"):
        menu_resolve.resolve_mlflow_env(menu, tf_dir="/ws")


def test_artifact_bucket_role_must_be_s3():
    menu = Menu.model_validate(
        {
            "project": "p",
            "recipes": {"db": {"kind": "rds", "role": "backend"}},
            "mlflow": {"artifact_bucket": {"from_role": "backend"}},
        }
    )
    with pytest.raises(ValueError, match="must point at an `s3` recipe"):
        menu_resolve.resolve_mlflow_env(menu, tf_dir="/ws")


def test_materialize_writes_masked_entries(tmp_path):
    out = tmp_path / ".env"
    with (
        patch("kitchen.secrets.db_url_from_terraform", return_value="postgresql://mlflow:p%40@h:5432/tracking"),
        patch("kitchen.secrets.mask") as mask,
    ):
        names = menu_resolve.materialize_mlflow_env(BACKEND_MENU, out, tf_dir="/ws")
    assert set(names) == {"MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACT_BUCKET"}
    content = out.read_text()
    assert "MLFLOW_TRACKING_URI=postgresql://mlflow:p%40@h:5432/tracking" in content
    assert "MLFLOW_ARTIFACT_BUCKET=mlflow-artifacts" in content
    mask.assert_any_call("postgresql://mlflow:p%40@h:5432/tracking")  # the URL is masked
