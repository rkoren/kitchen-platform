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


def test_bad_infra_field_errors_at_menu_load():
    """S-2 (INT-015): a typo in an infra recipe's fields errors at ``Menu.model_validate`` —
    not later at projection/provision. This is the seam's whole point, so assert the raise
    wraps *only* the load, and that the message names the offending recipe + field."""
    menu = {"project": "p", "recipes": {"db": {"kind": "rds", "instance_clas": "typo"}}}
    with pytest.raises(ValidationError) as exc:
        Menu.model_validate(menu)  # <- fails here, before any to_recipe_spec()
    msg = str(exc.value)
    assert "'db'" in msg and "instance_clas" in msg  # recipe key + bad field attributed


def test_bad_lambda_field_errors_at_load_after_wiring():
    """A serve lambda is validated *after* its iam_role/source menu-isms are wired, so a real
    typo in a lambda field still errors at load (and the wiring itself doesn't false-trip)."""
    menu = {
        "project": "p",
        "recipes": {"serve": {"kind": "lambda", "iam_role": "arn:role/x", "memoryy_size": 512}},
    }
    with pytest.raises(ValidationError, match="serve"):
        Menu.model_validate(menu)


def test_valid_infra_with_network_inheritance_loads():
    """A valid menu whose rds/sg rely on inherited networking must still load cleanly — the
    load-time validation projects with inheritance, so inherited fields aren't seen as bad."""
    Menu.model_validate(MENU)  # sg + rds inherit vpc_id/subnets; no error


def test_stage_recipe_args_stay_loose_at_load():
    """`stage` recipes have no infra spec — arbitrary fields (e.g. `args`) must pass at load."""
    menu = {
        "project": "p",
        "pipeline": ["train"],
        "recipes": {"train": {"kind": "stage", "source": "src/train/run.py", "args": ["--x"]}},
    }
    m = Menu.model_validate(menu)
    assert m.recipes["train"].fields["args"] == ["--x"]


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
