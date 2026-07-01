"""Tests for the menu.yaml → RecipeSpec projection — `Menu.to_recipe_spec()` (S-1, INT-014).

Since the package merge (INT-013) there is a single menu reader: recipes consumes the
validated `Menu` model and calls `.to_recipe_spec()`; the old `recipes/menu.py` re-parser is
gone. These tests exercise that projection.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kitchen.menu import Menu, is_menu

MENU: dict = {
    "project": "cbb-model",
    "region": "us-west-2",
    "network": {"vpc_id": "vpc-0abc", "subnets": ["subnet-a", "subnet-b"]},
    "pipeline": ["provision", "train"],
    "recipes": {
        "mlflow-backend": {"kind": "rds", "role": "mlflow-backend", "instance_class": "db.t4g.small"},
        "mlflow-sg": {"kind": "security_group"},
        "mlflow-artifacts": {"kind": "s3", "role": "mlflow-artifacts", "versioning": True},
        "train": {"kind": "stage", "source": "src/train/run.py"},
    },
}


def _spec(raw: dict):
    return Menu.model_validate(raw).to_recipe_spec()


def test_is_menu_vs_standalone_spec():
    assert is_menu(MENU) is True
    assert is_menu({"name": "x", "resources": [{"type": "s3", "name": "b"}]}) is False


def test_projects_infra_and_skips_runtime_kinds():
    spec = _spec(MENU)
    assert spec.name == "cbb-model"  # keyed by project
    assert spec.region == "us-west-2"
    # the `stage` recipe is kitchen's, not infra → skipped
    assert {(r.type, r.name) for r in spec.resources} == {
        ("rds", "mlflow-backend"),
        ("security_group", "mlflow-sg"),
        ("s3", "mlflow-artifacts"),
    }


def test_drops_menu_only_keys_and_keeps_fields():
    spec = _spec(MENU)
    rds = next(r for r in spec.resources if r.type == "rds")
    assert rds.instance_class == "db.t4g.small"  # kind-specific field kept
    assert not hasattr(rds, "role") and not hasattr(rds, "source")  # menu-only keys dropped


def test_network_inherited_by_sg_and_rds():
    spec = _spec(MENU)
    sg = next(r for r in spec.resources if r.type == "security_group")
    rds = next(r for r in spec.resources if r.type == "rds")
    assert sg.vpc_id == "vpc-0abc"
    assert rds.subnet_ids == ["subnet-a", "subnet-b"]


def test_network_does_not_override_explicit_recipe_values():
    menu = {
        "project": "p",
        "network": {"vpc_id": "vpc-net", "subnets": ["s-net1", "s-net2"]},
        "recipes": {
            "sg": {"kind": "security_group", "vpc_id": "vpc-own"},
            "db": {"kind": "rds", "db_subnet_group_name": "own-group"},
        },
    }
    spec = _spec(menu)
    sg = next(r for r in spec.resources if r.type == "security_group")
    db = next(r for r in spec.resources if r.type == "rds")
    assert sg.vpc_id == "vpc-own"  # not overridden by network
    assert db.db_subnet_group_name == "own-group" and not db.subnet_ids


def test_per_kind_validation_catches_bad_field():
    """A typo in a recipe's fields is caught when RecipeSpec validates (the validation
    INT-002 deferred to this projection; INT-015/S-2 moves it earlier into Menu load)."""
    menu = {"project": "p", "recipes": {"db": {"kind": "rds", "instance_clas": "typo"}}}
    with pytest.raises(ValidationError):
        _spec(menu)


def test_empty_recipes_yields_no_resources():
    spec = _spec({"project": "p", "recipes": {}})
    assert spec.name == "p"
    assert spec.resources == []


def test_lambda_iam_role_mapped_and_source_injected():
    """INT-006 / S-4: a serve lambda's menu `iam_role` → recipes `role`; `source` →
    the function's KITCHEN_PREDICTOR_DIR."""
    menu = {
        "project": "p",
        "recipes": {
            "serve": {
                "kind": "lambda",
                "role": "serving",  # menu discovery role — dropped
                "iam_role": "arn:aws:iam::123:role/exec",  # → LambdaSpec.role
                "image_uri": "123.dkr.ecr.us-east-1.amazonaws.com/serve:latest",
                "source": "src/serve/",
            },
        },
    }
    fn = next(r for r in _spec(menu).resources if r.type == "lambda")
    assert fn.role == "arn:aws:iam::123:role/exec"
    assert fn.environment["KITCHEN_PREDICTOR_DIR"] == "src/serve/"
    assert not hasattr(fn, "iam_role")  # the menu-ism is consumed
