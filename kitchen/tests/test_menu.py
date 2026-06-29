"""Tests for the unified manifest schema, kitchen.menu (INT-002)."""

from __future__ import annotations

import copy
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError

from kitchen.config import KitchenConfig
from kitchen.menu import (
    INFRA_KINDS,
    Menu,
    RecipeEntry,
    RoleRef,
    VariantNotFound,
    apply_variant,
    is_menu,
    load_params,
)

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


def test_menu_allows_project_sections_at_top_level():
    """The menu is a superset of params.yaml: free-form project sections pass through to
    model_extra (INT-007). This trades top-level typo safety for stage-config passthrough."""
    m = Menu.model_validate(_menu(model={"target": "passed", "random_state": 42}))
    assert m.model_extra["model"]["target"] == "passed"


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


def test_menu_source_map():
    m = Menu.model_validate(VALID)
    assert m.source_map["train"] == "src/train/run.py"
    assert m.source_map["serve"] == "src/serve/"
    assert "mlflow-backend" not in m.source_map  # infra recipes have no source


def test_menu_stage_requires_source():
    data = _menu(
        pipeline=["train"],
        recipes={
            "mlflow-backend": {"kind": "rds", "role": "mlflow-backend"},
            "mlflow-artifacts": {"kind": "s3", "role": "mlflow-artifacts"},
            "train": {"kind": "stage"},  # no source
        },
    )
    with pytest.raises(ValidationError, match="must declare a `source`"):
        Menu.model_validate(data)


# --- from_yaml ---


def test_menu_from_yaml(tmp_path):
    p = tmp_path / "menu.yaml"
    p.write_text(yaml.safe_dump(VALID), encoding="utf-8")
    m = Menu.from_yaml(str(p))
    assert m.project == "cbb-model"


# --- INT-007: KitchenConfig bridge + menu detection ---


def test_is_menu_detects_menu_and_rejects_params():
    assert is_menu(VALID) is True
    assert is_menu({"pipeline": ["train"]}) is True  # pipeline-only is still a menu
    assert is_menu({"experiment": "x", "model": {"eta": 0.1}}) is False  # legacy params.yaml
    assert is_menu("not-a-dict") is False


def test_to_kitchen_config_maps_ml_half():
    cfg = Menu.model_validate(VALID).to_kitchen_config()
    assert isinstance(cfg, KitchenConfig)
    assert cfg.experiment == "cbb-model"
    assert cfg.thresholds["val_accuracy"] == 0.8


def test_to_kitchen_config_carries_holdout():
    """CBB-017: a typed `holdout:` block survives the menu→KitchenConfig bridge."""
    m = Menu.model_validate(
        _menu(holdout={"path": "data/holdout/h.parquet", "label": "Outcome", "metric": "brier"})
    )
    cfg = m.to_kitchen_config()
    assert cfg.holdout is not None
    assert cfg.holdout.path == "data/holdout/h.parquet"
    assert cfg.holdout.label == "Outcome"
    assert cfg.holdout.metric == "brier"
    assert cfg.holdout.optional is True  # default


def test_menu_rejects_unknown_holdout_key():
    import pytest

    with pytest.raises(Exception, match="typo"):
        Menu.model_validate(_menu(holdout={"path": "h.parquet", "label": "y", "typo": 1}))


def test_to_kitchen_config_drops_roleref_mlflow_to_default():
    """A `{from_role}` tracking_uri is env-resolved (INT-003), so the bridge falls back to
    the MLflowConfig default — the materialized env overrides it at run time."""
    cfg = Menu.model_validate(VALID).to_kitchen_config()  # tracking_uri is a RoleRef
    assert cfg.mlflow.tracking_uri == "sqlite:///mlruns.db"
    assert cfg.mlflow.artifact_bucket is None  # RoleRef artifact_bucket dropped too


def test_to_kitchen_config_passes_literal_mlflow_through():
    m = Menu.model_validate(
        _menu(mlflow={"tracking_uri": "postgresql://db/mlflow", "artifact_bucket": "my-bucket"})
    )
    cfg = m.to_kitchen_config()
    assert cfg.mlflow.tracking_uri == "postgresql://db/mlflow"
    assert cfg.mlflow.artifact_bucket == "my-bucket"


def test_to_kitchen_config_carries_project_sections():
    """Project stage config (model/features/…) must survive the bridge into model_extra —
    the raw-YAML stage code reads `params["model"]["target"]`."""
    m = Menu.model_validate(_menu(model={"target": "passed"}, features={"processed_file": "f.parquet"}))
    cfg = m.to_kitchen_config()
    assert cfg.model_extra["model"]["target"] == "passed"
    assert cfg.model_extra["features"]["processed_file"] == "f.parquet"


def test_to_kitchen_config_carries_submission_for_kaggle():
    m = Menu.model_validate(
        _menu(submission={"competition": "mens-march-mania-2025", "target_col": "Pred"})
    )
    cfg = m.to_kitchen_config()
    assert cfg.submission.competition == "mens-march-mania-2025"
    assert cfg.submission.target_col == "Pred"


def test_load_params_injects_experiment_from_project(tmp_path):
    """The raw stage path (params["experiment"]) must match the bridge — load_params derives
    experiment from a menu's project so train/auto-promote agree on the experiment name."""
    p = tmp_path / "menu.yaml"
    p.write_text(yaml.safe_dump(_menu(model={"target": "passed"})), encoding="utf-8")
    params = load_params(str(p))
    assert params["experiment"] == "cbb-model"  # derived from project
    assert params["model"]["target"] == "passed"  # project section stays flat for raw stage code


def test_load_params_leaves_params_yaml_untouched(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text(yaml.safe_dump({"experiment": "exp", "model": {"target": "y"}}), encoding="utf-8")
    params = load_params(str(p))
    assert params["experiment"] == "exp"
    assert params["model"]["target"] == "y"


# --- CBB-016: experiment variants (apply_variant + schema) ---


def _variant_params():
    return {
        "model": {"max_depth": 4, "eta": 0.01},
        "feature_candidates": ["a", "b", "c"],
        "variants": {
            "rich": {
                "model": {"max_depth": 5},
                "feature_candidates": {"add": ["d_kp_1", "a"], "remove": ["b"]},
            },
            "swap": {"feature_candidates": ["x", "y"]},
        },
    }


def test_apply_variant_deep_merges_scalars():
    p = _variant_params()
    apply_variant(p, "rich")
    assert p["model"] == {"max_depth": 5, "eta": 0.01}  # max_depth merged, eta kept


def test_apply_variant_feature_candidates_add_remove():
    p = _variant_params()
    apply_variant(p, "rich")
    # b removed, a not duplicated, d_kp_1 appended (order-preserving union)
    assert p["feature_candidates"] == ["a", "c", "d_kp_1"]


def test_apply_variant_feature_candidates_plain_list_replaces():
    p = _variant_params()
    apply_variant(p, "swap")
    assert p["feature_candidates"] == ["x", "y"]


def test_apply_variant_composes_with_overrides_override_wins():
    from kitchen.flows.train_flow import _apply_overrides

    p = _variant_params()
    apply_variant(p, "rich")  # sets model.max_depth=5
    _apply_overrides(p, {"model.max_depth": 7})  # override applied after
    assert p["model"]["max_depth"] == 7  # override wins on conflict


def test_apply_variant_unknown_name_lists_available():
    with pytest.raises(VariantNotFound, match="available: rich, swap"):
        apply_variant(_variant_params(), "nope")


def test_menu_validates_variants_schema():
    m = Menu.model_validate(
        _menu(variants={"rich": {"feature_candidates": {"add": ["d_kp_1"], "remove": ["x"]}}})
    )
    assert "rich" in m.variants
    # a bad feature_candidates overlay key is rejected up front (extra=forbid on the overlay)
    with pytest.raises(ValidationError):
        Menu.model_validate(_menu(variants={"rich": {"feature_candidates": {"addd": ["typo"]}}}))


# --- locked to recipes' discriminator ---


def test_menu_infra_kinds_match_recipes_types():
    """INFRA_KINDS must equal recipes' ResourceSpec `type` discriminator — no drift."""
    pytest.importorskip("recipes")
    from recipes.schema import ResourceSpec

    union = get_args(ResourceSpec)[0]  # Annotated[Union[...], Field] → the Union
    recipes_types = {get_args(m.model_fields["type"].annotation)[0] for m in get_args(union)}
    assert set(INFRA_KINDS) == recipes_types


def test_to_kitchen_config_carries_models():
    """CBB-020: the `models:` map survives the menu→KitchenConfig bridge."""
    m = Menu.model_validate(
        _menu(models={"reg": {"experiment": "cbb-reg", "metric": "loto_brier_reg", "lower_is_better": True}})
    )
    cfg = m.to_kitchen_config()
    assert cfg.models["reg"].experiment == "cbb-reg"
    assert cfg.models["reg"].metric == "loto_brier_reg"
    assert cfg.models["reg"].lower_is_better is True
