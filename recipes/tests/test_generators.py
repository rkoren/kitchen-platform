"""Tests for per-resource Terraform generators."""
import pytest

from recipes.generators import generate_resource
from recipes.generators.iam import generate as iam_generate
from recipes.generators.lambda_ import generate as lambda_generate
from recipes.generators.s3 import generate as s3_generate
from recipes.schema import IAMRoleSpec, LambdaSpec, S3Spec


# --- S3 ---

def test_s3_basic_resource_block():
    spec = S3Spec(type="s3", name="my-bucket")
    out = s3_generate(spec)
    assert 'resource "aws_s3_bucket" "my-bucket"' in out


def test_s3_no_versioning_block_by_default():
    spec = S3Spec(type="s3", name="my-bucket", versioning=False)
    out = s3_generate(spec)
    assert "aws_s3_bucket_versioning" not in out


def test_s3_versioning_enabled():
    spec = S3Spec(type="s3", name="my-bucket", versioning=True)
    out = s3_generate(spec)
    assert 'resource "aws_s3_bucket_versioning" "my-bucket"' in out
    assert 'status = "Enabled"' in out


# --- IAM ---

def test_iam_role_resource_block():
    spec = IAMRoleSpec(type="iam_role", name="my-role", service="lambda.amazonaws.com")
    out = iam_generate(spec)
    assert 'resource "aws_iam_role" "my-role"' in out


def test_iam_role_assume_policy():
    spec = IAMRoleSpec(type="iam_role", name="my-role", service="lambda.amazonaws.com")
    out = iam_generate(spec)
    assert "lambda.amazonaws.com" in out
    assert "sts:AssumeRole" in out


def test_iam_role_policy_attachments():
    spec = IAMRoleSpec(
        type="iam_role",
        name="my-role",
        service="lambda.amazonaws.com",
        policies=["arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole"],
    )
    out = iam_generate(spec)
    assert "aws_iam_role_policy_attachment" in out
    assert "AWSLambdaBasicExecutionRole" in out


def test_iam_role_no_policies_no_attachment_block():
    spec = IAMRoleSpec(type="iam_role", name="my-role", service="lambda.amazonaws.com")
    out = iam_generate(spec)
    assert "aws_iam_role_policy_attachment" not in out


# --- Lambda ---

def test_lambda_image_uri():
    spec = LambdaSpec(
        type="lambda",
        name="my-fn",
        role="my-role",
        image_uri="123.dkr.ecr.us-east-1.amazonaws.com/my-fn:latest",
    )
    out = lambda_generate(spec)
    assert 'package_type = "Image"' in out
    assert "123.dkr.ecr.us-east-1.amazonaws.com/my-fn:latest" in out


def test_lambda_image_omits_runtime():
    spec = LambdaSpec(
        type="lambda",
        name="my-fn",
        role="my-role",
        image_uri="123.dkr.ecr.us-east-1.amazonaws.com/my-fn:latest",
    )
    out = lambda_generate(spec)
    assert "runtime" not in out
    assert "handler" not in out


def test_lambda_zip_runtime_and_handler():
    spec = LambdaSpec(
        type="lambda",
        name="my-fn",
        role="my-role",
        runtime="python3.11",
        handler="src.main.handler",
    )
    out = lambda_generate(spec)
    assert 'runtime = "python3.11"' in out
    assert 'handler = "src.main.handler"' in out
    assert "image_uri" not in out


def test_lambda_memory_and_timeout():
    spec = LambdaSpec(type="lambda", name="my-fn", role="my-role", memory_mb=512, timeout_s=30)
    out = lambda_generate(spec)
    assert "memory_size = 512" in out
    assert "timeout     = 30" in out


def test_lambda_environment_variables():
    spec = LambdaSpec(
        type="lambda", name="my-fn", role="my-role", environment={"TABLE_NAME": "my-table"}
    )
    out = lambda_generate(spec)
    assert "environment" in out
    assert "TABLE_NAME" in out
    assert "my-table" in out


def test_lambda_no_environment_block_when_empty():
    spec = LambdaSpec(type="lambda", name="my-fn", role="my-role")
    out = lambda_generate(spec)
    assert "environment" not in out


# --- Dispatch ---

def test_generate_resource_dispatches_s3():
    spec = S3Spec(type="s3", name="dispatch-test")
    assert "aws_s3_bucket" in generate_resource(spec)


def test_generate_resource_dispatches_iam():
    spec = IAMRoleSpec(type="iam_role", name="r", service="lambda.amazonaws.com")
    assert "aws_iam_role" in generate_resource(spec)


def test_generate_resource_dispatches_lambda():
    spec = LambdaSpec(type="lambda", name="fn", role="r")
    assert "aws_lambda_function" in generate_resource(spec)
