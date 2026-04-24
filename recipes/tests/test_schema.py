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
            "memory_mb": 512,
            "timeout_s": 30,
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
    spec = LambdaSpec.model_validate({"type": "lambda", "name": "fn", "role": "my-role"})
    assert spec.memory_mb == 128
    assert spec.timeout_s == 3
    assert spec.environment == {}
    assert spec.image_uri is None
    assert spec.ecr_repo is None
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
