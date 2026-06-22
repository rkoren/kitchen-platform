"""Tests for the unified manifest schema, kitchen.menu (INT-002)."""

from __future__ import annotations

import copy
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError

from kitchen.menu import INFRA_KINDS, Menu, RecipeEntry, RoleRef

VALID: dict = {
    "project": "cbb-model",
    "region": "us-east-1",
    "aws_account": "123456789012",
    "network": {"vpc_id": "vpc-x", "subnets": ["subnet-a", "subnet-b"]},
    "pipeline": ["provision", "train", "serve", "monitor"],
    "recipes": {
        "mlflow-backend": {"kind": "rds", "role": "mlflow-backend", "instance_class": "db.t4g.small"},
        "mlflow-artifacts": {"kind": "s3", "role": "mlflow-artifacts"},
        "train": {"kind": "stage", "source": "src/train/run.py"},
        "serve": {"kind": "lambda", "role": "serving", "source": "src/serve/", "ecr_repo": "cbb-serve"},
    },
    "mlflow": {
        "tracking_uri": {"from_role": "mlflow-backend"},
        "artifact_bucket": {"from_role": "mlflow-artifacts"},
    },
    "thresholds": {"val_accuracy": 0.8},
}


def _menu(**overrides):
    data = copy.deepcopy(VALID)
    data.update(overrides)
    return data


# --- valid ---


def test_menu_valid_full():
    m = Menu.model_validate(VALID)
    assert m.project == "cbb-model"
    assert m.network.subnets == ["subnet-a", "subnet-b"]
    assert isinstance(m.mlflow.tracking_uri, RoleRef)
    assert m.mlflow.tracking_uri.from_role == "mlflow-backend"


def test_menu_experiment_defaults_to_project():
    m = Menu.model_validate(_menu())  # no `experiment` key
    assert m.experiment == "cbb-model"


def test_menu_experiment_explicit_kept():
    m = Menu.model_validate(_menu(experiment="cbb-tournament"))
    assert m.experiment == "cbb-tournament"


def test_recipe_entry_collects_kind_specific_fields():
    entry = RecipeEntry.model_validate(
        {"kind": "rds", "role": "mlflow-backend", "instance_class": "db.t4g.small", "subnet_ids": ["a", "b"]}
    )
    assert entry.kind == "rds"
    assert entry.fields == {"instance_class": "db.t4g.small", "subnet_ids": ["a", "b"]}
    assert "kind" not in entry.fields and "role" not in entry.fields


def test_menu_literal_string_mlflow_uri_allowed():
    m = Menu.model_validate(_menu(mlflow={"tracking_uri": "sqlite:///mlruns.db"}))
    assert m.mlflow.tracking_uri == "sqlite:///mlruns.db"


# --- structural errors ---


def test_menu_missing_project_raises():
    data = _menu()
    del data["project"]
    with pytest.raises(ValidationError):
        Menu.model_validate(data)


def test_menu_unknown_top_level_key_raises():
    with pytest.raises(ValidationError):
        Menu.model_validate(_menu(experimnt="typo"))  # extra=forbid


def test_recipe_entry_unknown_kind_raises():
    with pytest.raises(ValidationError):
        RecipeEntry.model_validate({"kind": "dynamodb"})


# --- cross-reference errors ---


def test_menu_pipeline_dangling_recipe_raises():
    with pytest.raises(ValidationError, match="pipeline step 'traain'"):
        Menu.model_validate(_menu(pipeline=["provision", "traain"]))


def test_menu_pipeline_accepts_verbs_and_recipe_keys():
    m = Menu.model_validate(_menu(pipeline=["provision", "train", "serve", "monitor"]))
    assert m.pipeline == ["provision", "train", "serve", "monitor"]


def test_menu_from_role_dangling_raises():
    with pytest.raises(ValidationError, match="from_role 'nope' matches no recipe role"):
        Menu.model_validate(_menu(mlflow={"tracking_uri": {"from_role": "nope"}}))


# --- from_yaml ---


def test_menu_from_yaml(tmp_path):
    p = tmp_path / "menu.yaml"
    p.write_text(yaml.safe_dump(VALID), encoding="utf-8")
    m = Menu.from_yaml(str(p))
    assert m.project == "cbb-model"


# --- locked to recipes' discriminator ---


def test_menu_infra_kinds_match_recipes_types():
    """INFRA_KINDS must equal recipes' ResourceSpec `type` discriminator — no drift."""
    pytest.importorskip("recipes")
    from recipes.schema import ResourceSpec

    union = get_args(ResourceSpec)[0]  # Annotated[Union[...], Field] → the Union
    recipes_types = {get_args(m.model_fields["type"].annotation)[0] for m in get_args(union)}
    assert set(INFRA_KINDS) == recipes_types
