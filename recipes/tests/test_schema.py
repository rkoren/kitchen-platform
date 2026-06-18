"""Tests for YAML spec schema validation."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from recipes.schema import (
    ECRSpec,
    IAMRoleSpec,
    LambdaSpec,
    RDSSpec,
    RecipeSpec,
    S3Spec,
    SecurityGroupSpec,
)

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


def test_rds_defaults():
    spec = RDSSpec.model_validate({"type": "rds", "name": "mlflow-backend"})
    assert spec.engine_version == "16"
    assert spec.instance_class == "db.t4g.micro"
    assert spec.allocated_storage == 20
    assert spec.backup_retention_days == 7
    assert spec.db_name == "mlflow"
    assert spec.storage_encrypted is True
    assert spec.deletion_protection is True
    assert spec.publicly_accessible is False
    assert spec.db_subnet_group_name is None
    assert spec.vpc_security_group_ids == []


def test_rds_discriminated_in_recipe():
    spec = RecipeSpec.model_validate(
        {"name": "r", "resources": [{"type": "rds", "name": "db", "instance_class": "db.t4g.small"}]}
    )
    assert isinstance(spec.resources[0], RDSSpec)
    assert spec.resources[0].instance_class == "db.t4g.small"


def test_unknown_field_on_rds_raises():
    with pytest.raises(ValidationError):
        RDSSpec.model_validate({"type": "rds", "name": "db", "password": "hunter2"})


def test_security_group_defaults():
    spec = SecurityGroupSpec.model_validate({"type": "security_group", "name": "db-sg"})
    assert spec.vpc_id is None
    assert spec.egress_all is True
    # default ingress is a single Postgres rule open to anywhere
    assert len(spec.ingress) == 1
    assert spec.ingress[0].port == 5432
    assert spec.ingress[0].protocol == "tcp"
    assert spec.ingress[0].cidr_blocks == ["0.0.0.0/0"]


def test_rds_security_groups_field():
    spec = RDSSpec.model_validate({"type": "rds", "name": "db", "security_groups": ["db-sg"]})
    assert spec.security_groups == ["db-sg"]


def test_rds_security_group_reference_must_resolve():
    with pytest.raises(ValidationError, match="does not match any security_group"):
        RecipeSpec.model_validate(
            {
                "name": "r",
                "resources": [{"type": "rds", "name": "db", "security_groups": ["missing-sg"]}],
            }
        )


def test_rds_security_group_reference_resolves_in_spec():
    spec = RecipeSpec.model_validate(
        {
            "name": "r",
            "resources": [
                {"type": "security_group", "name": "db-sg"},
                {"type": "rds", "name": "db", "security_groups": ["db-sg"]},
            ],
        }
    )
    assert isinstance(spec.resources[0], SecurityGroupSpec)


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
            {"type": "lambda", "name": "fn", "role": "r", "ecr_repo": "repo", "memory_mb": 512}
        )


def test_unknown_field_on_recipe_raises():
    with pytest.raises(ValidationError):
        RecipeSpec.model_validate({"name": "x", "unknown_top_level": "bad"})


# --- P0-003: Lambda package type validation ---


def test_lambda_image_and_zip_fields_raises():
    with pytest.raises(ValidationError, match="cannot mix"):
        LambdaSpec.model_validate(
            {
                "type": "lambda",
                "name": "fn",
                "role": "r",
                "image_uri": "123.dkr.ecr.amazonaws.com/x:latest",
                "runtime": "python3.11",
                "handler": "app.handler",
            }
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
        RecipeSpec.model_validate(
            {
                "name": "x",
                "resources": [
                    {
                        "type": "lambda",
                        "name": "fn",
                        "role": "nonexistent-role",
                        "ecr_repo": "my-repo",
                    },
                    {"type": "ecr", "name": "my-repo"},
                ],
            }
        )


def test_lambda_role_arn_passes_reference_check():
    spec = RecipeSpec.model_validate(
        {
            "name": "x",
            "resources": [
                {
                    "type": "lambda",
                    "name": "fn",
                    "role": "arn:aws:iam::123456789:role/my-role",
                    "ecr_repo": "my-repo",
                },
                {"type": "ecr", "name": "my-repo"},
            ],
        }
    )
    assert spec.resources[0].role.startswith("arn:")


def test_lambda_ecr_repo_references_unknown_ecr_raises():
    with pytest.raises(ValidationError, match="does not match any ecr"):
        RecipeSpec.model_validate(
            {
                "name": "x",
                "resources": [
                    {"type": "iam_role", "name": "my-role", "service": "lambda.amazonaws.com"},
                    {
                        "type": "lambda",
                        "name": "fn",
                        "role": "my-role",
                        "ecr_repo": "nonexistent-ecr",
                    },
                ],
            }
        )


def test_valid_cross_references_pass():
    spec = RecipeSpec.model_validate(FULL_SPEC)
    assert len(spec.resources) == 4


# --- Example file validates cleanly ---


def test_example_lambda_api_yaml_validates():
    example = Path(__file__).parent.parent / "examples" / "lambda-api.yaml"
    data = yaml.safe_load(example.read_text())
    spec = RecipeSpec.model_validate(data)
    assert spec.name == "my-api"


def test_example_inference_api_yaml_validates():
    example = Path(__file__).parent.parent / "examples" / "ecr-lambda-inference-api.yaml"
    data = yaml.safe_load(example.read_text())
    spec = RecipeSpec.model_validate(data)
    assert spec.name == "inference-api"
    types = [r.type for r in spec.resources]
    assert types == ["s3", "ecr", "iam_role", "lambda"]
    lambda_spec = next(r for r in spec.resources if r.type == "lambda")
    assert lambda_spec.function_url is True
    assert lambda_spec.ecr_repo == "inference-api"


def test_example_s3_data_bucket_yaml_validates():
    example = Path(__file__).parent.parent / "examples" / "s3-data-bucket.yaml"
    data = yaml.safe_load(example.read_text())
    spec = RecipeSpec.model_validate(data)
    assert [r.type for r in spec.resources] == ["s3"]
    bucket = spec.resources[0]
    assert bucket.versioning is True
    assert bucket.lifecycle_expiration_days == 730


def test_example_mlflow_artifacts_yaml_validates():
    example = Path(__file__).parent.parent / "examples" / "mlflow-artifacts.yaml"
    data = yaml.safe_load(example.read_text())
    spec = RecipeSpec.model_validate(data)
    assert [r.type for r in spec.resources] == ["s3"]
    bucket = spec.resources[0]
    assert bucket.versioning is True
    # Model artifacts must not be auto-expired.
    assert bucket.lifecycle_expiration_days is None


def test_example_mlflow_tracking_backend_yaml_validates():
    example = Path(__file__).parent.parent / "examples" / "mlflow-tracking-backend.yaml"
    data = yaml.safe_load(example.read_text())
    spec = RecipeSpec.model_validate(data)
    # Security group (reachability) + RDS backend store + S3 artifact bucket.
    assert [r.type for r in spec.resources] == ["security_group", "rds", "s3"]
    rds = next(r for r in spec.resources if r.type == "rds")
    assert rds.db_name == "mlflow"
    assert rds.deletion_protection is True
    assert rds.security_groups == ["mlflow-backend-sg"]  # wired to the security_group
    artifacts = next(r for r in spec.resources if r.type == "s3")
    assert artifacts.versioning is True


def test_example_mlflow_backend_validation_yaml_validates():
    example = Path(__file__).parent.parent / "examples" / "mlflow-backend-validation.yaml"
    data = yaml.safe_load(example.read_text())
    spec = RecipeSpec.model_validate(data)
    assert [r.type for r in spec.resources] == ["security_group", "rds"]
    rds = next(r for r in spec.resources if r.type == "rds")
    # Throwaway: teardown must not be blocked by deletion protection.
    assert rds.deletion_protection is False
    assert rds.security_groups == ["mlflow-validation-sg"]


def test_example_serving_stack_yaml_validates():
    example = Path(__file__).parent.parent / "examples" / "kaggle-serving-stack.yaml"
    data = yaml.safe_load(example.read_text())
    spec = RecipeSpec.model_validate(data)
    assert spec.name == "titanic"
    # Two buckets, an ECR repo, a role, and the serving Lambda.
    assert [r.type for r in spec.resources] == ["s3", "s3", "ecr", "iam_role", "lambda"]
    role = next(r for r in spec.resources if r.type == "iam_role")
    assert role.inline_policies[0].name == "artifact-bucket-read"  # R-013 scoped access
    fn = next(r for r in spec.resources if r.type == "lambda")
    assert fn.log_retention_days == 30  # R-012
    assert fn.function_url is True  # R-011
