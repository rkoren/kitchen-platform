"""Essential tests for the standalone recipes package (deliberately small)."""

import json
from pathlib import Path

import pytest
import yaml
from recipes.cli import _classify_plan, app
from recipes.schema import RecipeSpec
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "examples").glob("*.yaml")) + [ROOT / "stacks" / "kitchen-platform.yaml"]
runner = CliRunner()


def _tf_name(resource) -> str:
    return f"{resource.type.replace('_', '-')}-{resource.name}.tf"


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_spec_validates_and_generates_one_file_per_resource(spec_path, tmp_path):
    spec = RecipeSpec.model_validate(yaml.safe_load(spec_path.read_text()))
    result = runner.invoke(app, ["generate", str(spec_path), "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert {p.name for p in tmp_path.glob("*.tf")} == {"provider.tf"} | {
        _tf_name(r) for r in spec.resources
    }
    for r in spec.resources:
        assert r.name in (tmp_path / _tf_name(r)).read_text()


def test_validate_rejects_unknown_resource_type(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nregion: us-east-1\nresources:\n  - {type: nope, name: y}\n")
    assert runner.invoke(app, ["validate", str(bad)]).exit_code != 0


def test_legacy_menu_yaml_gets_a_clear_error(tmp_path):
    menu = tmp_path / "menu.yaml"
    menu.write_text("project: x\npipeline: [train]\nrecipes: {train: {kind: stage}}\n")
    result = runner.invoke(app, ["validate", str(menu)])
    assert result.exit_code == 1
    assert "menu.yaml" in result.output


def test_schema_export_is_a_json_schema_document():
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert json.loads(result.output)["$schema"].endswith("2020-12/schema")


def test_destructive_collision_classification():
    plan = {
        "resource_changes": [
            {"address": "aws_s3_bucket.a", "change": {"actions": ["delete"]}},
            {"address": "aws_s3_bucket.b", "change": {"actions": ["create"]}},
        ]
    }
    # deletes with nothing kept = the state belongs to a different spec -> apply is refused
    assert _classify_plan(plan) == (["aws_s3_bucket.b"], ["aws_s3_bucket.a"], [])
    plan["resource_changes"].append(
        {"address": "aws_ecr_repository.c", "change": {"actions": ["update"]}}
    )
    assert _classify_plan(plan)[2] == ["aws_ecr_repository.c"]
