"""Tests for YAML spec schema validation."""
import pytest
from pydantic import ValidationError

from recipes.schema import ECRSpec, IAMRoleSpec, LambdaSpec, RecipeSpec, S3Spec

FULL_SPEC = {
    "name": "my-api",
    "region": "us-east-1",
    "resources": [
        {"type": "s3", "name": "my-artifacts", "versioning": True},
        {
            "type": "iam_role",
            "name": "my-exec",
            "service": "lambda.amazonaws.com",
            "policies": ["arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"],
        },
        {"type": "ecr", "name": "my-repo"},
        {
            "type": "lambda",
            "name": "my-function",
            "role": "my-exec",
            "ecr_repo": "my-repo",
            "memory": 512,
            "timeout": 30,
        },
    ],
}


def test_full_spec_parses():
    spec = RecipeSpec.model_validate(FULL_SPEC)
    assert spec.name == "my-api"
    assert len(spec.resources) == 4


def test_resource_types_discriminated():
    spec = RecipeSpec.model_validate(FULL_SPEC)
    assert isinstance(spec.resources[0], S3Spec)
    assert isinstance(spec.resources[1], IAMRoleSpec)
    assert isinstance(spec.resources[2], ECRSpec)
    assert isinstance(spec.resources[3], LambdaSpec)


def test_region_default():
    spec = RecipeSpec.model_validate({"name": "x", "resources": []})
    assert spec.region == "us-east-1"


def test_empty_resources_allowed():
    spec = RecipeSpec.model_validate({"name": "x"})
    assert spec.resources == []


def test_s3_versioning_default_false():
    spec = S3Spec.model_validate({"type": "s3", "name": "my-bucket"})
    assert spec.versioning is False


def test_iam_role_policies_default_empty():
    spec = IAMRoleSpec.model_validate(
        {"type": "iam_role", "name": "r", "service": "lambda.amazonaws.com"}
    )
    assert spec.policies == []


def test_ecr_defaults():
    spec = ECRSpec.model_validate({"type": "ecr", "name": "my-repo"})
    assert spec.scan_on_push is True
    assert spec.image_tag_mutability == "MUTABLE"


def test_ecr_immutable():
    spec = ECRSpec.model_validate(
        {"type": "ecr", "name": "my-repo", "image_tag_mutability": "IMMUTABLE"}
    )
    assert spec.image_tag_mutability == "IMMUTABLE"


def test_ecr_invalid_mutability_raises():
    with pytest.raises(ValidationError):
        ECRSpec.model_validate({"type": "ecr", "name": "x", "image_tag_mutability": "INVALID"})


def test_lambda_defaults():
    spec = LambdaSpec.model_validate(
        {"type": "lambda", "name": "fn", "role": "my-role", "ecr_repo": "my-repo"}
    )
    assert spec.memory == 128
    assert spec.timeout == 3
    assert spec.environment == {}
    assert spec.image_uri is None
    assert spec.ecr_repo == "my-repo"
    assert spec.runtime is None


def test_lambda_ecr_repo_field():
    spec = LambdaSpec.model_validate(
        {"type": "lambda", "name": "fn", "role": "r", "ecr_repo": "my-repo"}
    )
    assert spec.ecr_repo == "my-repo"


def test_missing_name_raises():
    with pytest.raises(ValidationError):
        RecipeSpec.model_validate({"resources": []})


def test_unknown_resource_type_raises():
    with pytest.raises(ValidationError):
        RecipeSpec.model_validate({"name": "x", "resources": [{"type": "ec2", "name": "bad"}]})


def test_lambda_missing_role_raises():
    with pytest.raises(ValidationError):
        LambdaSpec.model_validate({"type": "lambda", "name": "fn"})


# --- P0-002: extra fields rejected ---

def test_unknown_field_on_s3_raises():
    with pytest.raises(ValidationError, match="extra_field"):
        S3Spec.model_validate({"type": "s3", "name": "x", "extra_field": "bad"})


def test_unknown_field_on_lambda_raises():
    with pytest.raises(ValidationError, match="memory_mb"):
        LambdaSpec.model_validate(
            {"type": "lambda", "name": "fn", "role": "r",
             "ecr_repo": "repo", "memory_mb": 512}
        )


def test_unknown_field_on_recipe_raises():
    with pytest.raises(ValidationError):
        RecipeSpec.model_validate({"name": "x", "unknown_top_level": "bad"})


# --- P0-003: Lambda package type validation ---

def test_lambda_image_and_zip_fields_raises():
    with pytest.raises(ValidationError, match="cannot mix"):
        LambdaSpec.model_validate(
            {"type": "lambda", "name": "fn", "role": "r",
             "image_uri": "123.dkr.ecr.amazonaws.com/x:latest",
             "runtime": "python3.11", "handler": "app.handler"}
        )


def test_lambda_neither_image_nor_zip_raises():
    with pytest.raises(ValidationError, match="must specify"):
        LambdaSpec.model_validate({"type": "lambda", "name": "fn", "role": "r"})


def test_lambda_zip_missing_handler_raises():
    with pytest.raises(ValidationError, match="both runtime and handler"):
        LambdaSpec.model_validate(
            {"type": "lambda", "name": "fn", "role": "r", "runtime": "python3.11"}
        )


def test_lambda_zip_missing_runtime_raises():
    with pytest.raises(ValidationError, match="both runtime and handler"):
        LambdaSpec.model_validate(
            {"type": "lambda", "name": "fn", "role": "r", "handler": "app.handler"}
        )


# --- P0-004: cross-resource reference validation ---

def test_lambda_role_references_unknown_iam_raises():
    with pytest.raises(ValidationError, match="does not match any iam_role"):
        RecipeSpec.model_validate({
            "name": "x",
            "resources": [
                {"type": "lambda", "name": "fn", "role": "nonexistent-role",
                 "ecr_repo": "my-repo"},
                {"type": "ecr", "name": "my-repo"},
            ],
        })


def test_lambda_role_arn_passes_reference_check():
    spec = RecipeSpec.model_validate({
        "name": "x",
        "resources": [
            {"type": "lambda", "name": "fn",
             "role": "arn:aws:iam::123456789:role/my-role",
             "ecr_repo": "my-repo"},
            {"type": "ecr", "name": "my-repo"},
        ],
    })
    assert spec.resources[0].role.startswith("arn:")


def test_lambda_ecr_repo_references_unknown_ecr_raises():
    with pytest.raises(ValidationError, match="does not match any ecr"):
        RecipeSpec.model_validate({
            "name": "x",
            "resources": [
                {"type": "iam_role", "name": "my-role",
                 "service": "lambda.amazonaws.com"},
                {"type": "lambda", "name": "fn", "role": "my-role",
                 "ecr_repo": "nonexistent-ecr"},
            ],
        })


def test_valid_cross_references_pass():
    spec = RecipeSpec.model_validate(FULL_SPEC)
    assert len(spec.resources) == 4


# --- Example file validates cleanly ---

def test_example_lambda_api_yaml_validates():
    import yaml
    from pathlib import Path
    example = Path(__file__).parent.parent / "examples" / "lambda-api.yaml"
    data = yaml.safe_load(example.read_text())
    spec = RecipeSpec.model_validate(data)
    assert spec.name == "my-api"
