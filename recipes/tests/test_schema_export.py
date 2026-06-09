"""Tests for `recipes schema` — JSON Schema export of the recipe YAML spec."""

import json

import jsonschema
import yaml
from typer.testing import CliRunner

from recipes.cli import app

runner = CliRunner()

VALID_SPEC = """\
name: my-api
region: us-east-1
resources:
  - type: s3
    name: my-api-artifacts
    versioning: true
  - type: iam_role
    name: my-api-exec
    service: lambda.amazonaws.com
  - type: lambda
    name: my-api
    role: my-api-exec
    image_uri: "123456789.dkr.ecr.us-east-1.amazonaws.com/my-api:latest"
    memory: 512
    timeout: 30
"""


def _export() -> dict:
    result = runner.invoke(app, ["schema"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_schema_stdout_is_valid_json_document():
    doc = _export()
    assert doc["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert doc["title"] == "RecipeSpec"
    assert set(doc["properties"]) == {"name", "region", "resources"}
    # All four resource types are described.
    assert set(doc["$defs"]) == {"S3Spec", "IAMRoleSpec", "ECRSpec", "LambdaSpec"}


def test_exported_schema_is_itself_valid():
    """The emitted document is a well-formed draft 2020-12 schema."""
    jsonschema.Draft202012Validator.check_schema(_export())


def test_exported_schema_accepts_a_valid_spec():
    """Round-trip: a spec that RecipeSpec accepts also validates against the export."""
    jsonschema.validate(instance=yaml.safe_load(VALID_SPEC), schema=_export())


def test_exported_schema_rejects_unknown_field():
    """extra='forbid' on the model surfaces as additionalProperties: false."""
    bad = yaml.safe_load(VALID_SPEC)
    bad["nonsense"] = True
    try:
        jsonschema.validate(instance=bad, schema=_export())
    except jsonschema.ValidationError:
        return
    raise AssertionError("schema should reject an unknown top-level field")


def test_schema_out_writes_file(tmp_path):
    dest = tmp_path / "recipe.schema.json"
    result = runner.invoke(app, ["schema", "--out", str(dest)], catch_exceptions=False)
    assert result.exit_code == 0
    written = json.loads(dest.read_text())
    assert written["title"] == "RecipeSpec"
