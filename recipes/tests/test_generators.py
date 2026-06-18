"""Tests for per-resource Terraform generators."""

from recipes.generators import generate_resource
from recipes.generators.ecr import generate as ecr_generate
from recipes.generators.iam import generate as iam_generate
from recipes.generators.lambda_ import generate as lambda_generate
from recipes.generators.rds import generate as rds_generate
from recipes.generators.s3 import generate as s3_generate
from recipes.schema import ECRSpec, IAMRoleSpec, LambdaSpec, RDSSpec, S3Spec

# --- S3 ---


def test_s3_basic_resource_block():
    spec = S3Spec(type="s3", name="my-bucket")
    out = s3_generate(spec)
    assert 'resource "aws_s3_bucket" "my_bucket"' in out


def test_s3_aws_name_preserved():
    # Terraform label uses underscores; actual bucket name keeps hyphens
    spec = S3Spec(type="s3", name="my-bucket")
    out = s3_generate(spec)
    assert 'bucket = "my-bucket"' in out


def test_s3_no_versioning_block_by_default():
    spec = S3Spec(type="s3", name="my-bucket", versioning=False)
    out = s3_generate(spec)
    assert "aws_s3_bucket_versioning" not in out


def test_s3_versioning_enabled():
    spec = S3Spec(type="s3", name="my-bucket", versioning=True)
    out = s3_generate(spec)
    assert 'resource "aws_s3_bucket_versioning" "my_bucket"' in out
    assert 'status = "Enabled"' in out


def test_s3_versioning_references_bucket():
    spec = S3Spec(type="s3", name="my-bucket", versioning=True)
    out = s3_generate(spec)
    assert "aws_s3_bucket.my_bucket.id" in out


# --- IAM ---


def test_iam_role_resource_block():
    spec = IAMRoleSpec(type="iam_role", name="my-role", service="lambda.amazonaws.com")
    out = iam_generate(spec)
    assert 'resource "aws_iam_role" "my_role"' in out


def test_iam_role_aws_name_preserved():
    spec = IAMRoleSpec(type="iam_role", name="my-role", service="lambda.amazonaws.com")
    out = iam_generate(spec)
    assert 'name               = "my-role"' in out


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


# --- ECR ---


def test_ecr_resource_block():
    spec = ECRSpec(type="ecr", name="my-repo")
    out = ecr_generate(spec)
    assert 'resource "aws_ecr_repository" "my_repo"' in out


def test_ecr_aws_name_preserved():
    spec = ECRSpec(type="ecr", name="my-repo")
    out = ecr_generate(spec)
    assert 'name                 = "my-repo"' in out


def test_ecr_scan_on_push_default_true():
    spec = ECRSpec(type="ecr", name="my-repo")
    out = ecr_generate(spec)
    assert "scan_on_push = true" in out


def test_ecr_scan_on_push_false():
    spec = ECRSpec(type="ecr", name="my-repo", scan_on_push=False)
    out = ecr_generate(spec)
    assert "scan_on_push = false" in out


def test_ecr_default_mutable_tags():
    spec = ECRSpec(type="ecr", name="my-repo")
    out = ecr_generate(spec)
    assert 'image_tag_mutability = "MUTABLE"' in out


def test_ecr_immutable_tags():
    spec = ECRSpec(type="ecr", name="my-repo", image_tag_mutability="IMMUTABLE")
    out = ecr_generate(spec)
    assert 'image_tag_mutability = "IMMUTABLE"' in out


def test_ecr_no_lambda_policy_by_default():
    spec = ECRSpec(type="ecr", name="my-repo")
    out = ecr_generate(spec)
    assert "aws_ecr_repository_policy" not in out


def test_ecr_lambda_access_adds_repository_policy():
    spec = ECRSpec(type="ecr", name="my-repo", lambda_access=True)
    out = ecr_generate(spec)
    assert "aws_ecr_repository_policy" in out
    assert "lambda.amazonaws.com" in out
    assert "ecr:BatchGetImage" in out


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


def test_lambda_ecr_repo_generates_tf_reference():
    spec = LambdaSpec(type="lambda", name="my-fn", role="my-role", ecr_repo="my-repo")
    out = lambda_generate(spec)
    assert 'package_type = "Image"' in out
    assert "aws_ecr_repository.my_repo.repository_url" in out


def test_lambda_ecr_repo_takes_priority_over_image_uri():
    spec = LambdaSpec(
        type="lambda",
        name="my-fn",
        role="my-role",
        ecr_repo="my-repo",
        image_uri="should-be-ignored",
    )
    out = lambda_generate(spec)
    assert "aws_ecr_repository.my_repo.repository_url" in out
    assert "should-be-ignored" not in out


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
    spec = LambdaSpec(
        type="lambda",
        name="my-fn",
        role="my-role",
        image_uri="123456789.dkr.ecr.us-east-1.amazonaws.com/my-fn:latest",
        memory=512,
        timeout=30,
    )
    out = lambda_generate(spec)
    assert "memory_size = 512" in out
    assert "timeout     = 30" in out


def test_lambda_environment_variables():
    spec = LambdaSpec(
        type="lambda",
        name="my-fn",
        role="my-role",
        image_uri="123456789.dkr.ecr.us-east-1.amazonaws.com/my-fn:latest",
        environment={"TABLE_NAME": "my-table"},
    )
    out = lambda_generate(spec)
    assert "environment" in out
    assert "TABLE_NAME" in out
    assert "my-table" in out


_IMAGE_LAMBDA = {"ecr_repo": "my-repo"}  # minimal valid image Lambda for fixture reuse


def test_lambda_no_environment_block_when_empty():
    spec = LambdaSpec(type="lambda", name="my-fn", role="my-role", **_IMAGE_LAMBDA)
    out = lambda_generate(spec)
    assert "environment" not in out


def test_lambda_role_reference_normalised():
    spec = LambdaSpec(type="lambda", name="my-fn", role="my-exec-role", **_IMAGE_LAMBDA)
    out = lambda_generate(spec)
    assert "aws_iam_role.my_exec_role.arn" in out


def test_lambda_depends_on_generated_for_iam_role_policies():
    role = IAMRoleSpec(
        type="iam_role",
        name="my-exec-role",
        service="lambda.amazonaws.com",
        policies=[
            "arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole",
            "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
        ],
    )
    spec = LambdaSpec(type="lambda", name="my-fn", role="my-exec-role", **_IMAGE_LAMBDA)
    out = lambda_generate(spec, all_resources=[role])
    assert "depends_on" in out
    assert "aws_iam_role_policy_attachment.my_exec_role_1" in out
    assert "aws_iam_role_policy_attachment.my_exec_role_2" in out


def test_lambda_no_depends_on_without_all_resources():
    spec = LambdaSpec(type="lambda", name="my-fn", role="my-exec-role", **_IMAGE_LAMBDA)
    out = lambda_generate(spec)
    assert "depends_on" not in out


def test_lambda_no_depends_on_for_role_with_no_policies():
    role = IAMRoleSpec(type="iam_role", name="my-exec-role", service="lambda.amazonaws.com")
    spec = LambdaSpec(type="lambda", name="my-fn", role="my-exec-role", **_IMAGE_LAMBDA)
    out = lambda_generate(spec, all_resources=[role])
    assert "depends_on" not in out


# --- Dispatch ---


def test_generate_resource_dispatches_s3():
    spec = S3Spec(type="s3", name="x")
    assert "aws_s3_bucket" in generate_resource(spec)


def test_generate_resource_dispatches_iam():
    spec = IAMRoleSpec(type="iam_role", name="r", service="lambda.amazonaws.com")
    assert "aws_iam_role" in generate_resource(spec)


def test_generate_resource_dispatches_ecr():
    spec = ECRSpec(type="ecr", name="x")
    assert "aws_ecr_repository" in generate_resource(spec)


def test_generate_resource_dispatches_lambda():
    spec = LambdaSpec(type="lambda", name="fn", role="r", ecr_repo="my-repo")
    assert "aws_lambda_function" in generate_resource(spec)


def test_generate_resource_dispatches_rds():
    spec = RDSSpec(type="rds", name="db")
    assert "aws_db_instance" in generate_resource(spec)


# --- S3: encryption / public access block / lifecycle (R-009) ---


def test_s3_encryption_enabled_by_default():
    out = s3_generate(S3Spec(type="s3", name="my-bucket"))
    assert 'resource "aws_s3_bucket_server_side_encryption_configuration" "my_bucket"' in out
    assert 'sse_algorithm = "AES256"' in out


def test_s3_encryption_can_be_disabled():
    out = s3_generate(S3Spec(type="s3", name="my-bucket", encryption=False))
    assert "aws_s3_bucket_server_side_encryption_configuration" not in out


def test_s3_public_access_block_enabled_by_default():
    out = s3_generate(S3Spec(type="s3", name="my-bucket"))
    assert 'resource "aws_s3_bucket_public_access_block" "my_bucket"' in out
    assert "restrict_public_buckets = true" in out


def test_s3_public_access_block_can_be_disabled():
    out = s3_generate(S3Spec(type="s3", name="my-bucket", public_access_block=False))
    assert "aws_s3_bucket_public_access_block" not in out


def test_s3_no_lifecycle_by_default():
    out = s3_generate(S3Spec(type="s3", name="my-bucket"))
    assert "aws_s3_bucket_lifecycle_configuration" not in out


def test_s3_lifecycle_expiration_days():
    out = s3_generate(S3Spec(type="s3", name="my-bucket", lifecycle_expiration_days=30))
    assert 'resource "aws_s3_bucket_lifecycle_configuration" "my_bucket"' in out
    assert "days = 30" in out


# --- ECR: lifecycle policy (R-010) ---


def test_ecr_no_lifecycle_by_default():
    out = ecr_generate(ECRSpec(type="ecr", name="my-repo"))
    assert "aws_ecr_lifecycle_policy" not in out


def test_ecr_lifecycle_keep_last():
    out = ecr_generate(ECRSpec(type="ecr", name="my-repo", lifecycle_keep_last=5))
    assert 'resource "aws_ecr_lifecycle_policy" "my_repo"' in out
    assert "imageCountMoreThan" in out
    assert "countNumber = 5" in out


# --- Lambda: function URL (R-011) ---


def test_lambda_no_function_url_by_default():
    out = lambda_generate(LambdaSpec(type="lambda", name="fn", role="r", image_uri="x:latest"))
    assert "aws_lambda_function_url" not in out
    assert "aws_lambda_permission" not in out


def test_lambda_function_url_defaults_to_iam_auth():
    out = lambda_generate(
        LambdaSpec(type="lambda", name="fn", role="r", image_uri="x:latest", function_url=True)
    )
    assert 'resource "aws_lambda_function_url" "fn"' in out
    assert 'authorization_type = "AWS_IAM"' in out
    # IAM-authed URLs need no public-invoke permission.
    assert "aws_lambda_permission" not in out


def test_lambda_function_url_public_adds_invoke_permission():
    out = lambda_generate(
        LambdaSpec(
            type="lambda",
            name="fn",
            role="r",
            image_uri="x:latest",
            function_url=True,
            function_url_auth="NONE",
        )
    )
    assert 'authorization_type = "NONE"' in out
    assert 'resource "aws_lambda_permission" "fn_url"' in out
    assert "lambda:InvokeFunctionUrl" in out
    assert 'principal              = "*"' in out


def test_lambda_function_url_emits_output():
    out = lambda_generate(
        LambdaSpec(type="lambda", name="fn", role="r", image_uri="x:latest", function_url=True)
    )
    assert 'output "fn_url"' in out
    assert "aws_lambda_function_url.fn.function_url" in out


def test_lambda_environment_variables_aligned():
    """Multiple env vars: the `=` align so `terraform fmt` is a no-op."""
    spec = LambdaSpec(
        type="lambda",
        name="fn",
        role="r",
        image_uri="x:latest",
        environment={"SHORT": "a", "MUCH_LONGER_KEY": "b"},
    )
    out = lambda_generate(spec)
    var_lines = [ln for ln in out.splitlines() if ln.strip().endswith('"a"') or ln.strip().endswith('"b"')]
    assert len(var_lines) == 2
    assert len({ln.index("=") for ln in var_lines}) == 1, "env var `=` are not aligned"


# --- Lambda: CloudWatch log retention (R-012) ---


def test_lambda_no_log_group_by_default():
    out = lambda_generate(LambdaSpec(type="lambda", name="fn", role="r", image_uri="x:latest"))
    assert "aws_cloudwatch_log_group" not in out


def test_lambda_log_retention_creates_log_group():
    out = lambda_generate(
        LambdaSpec(type="lambda", name="fn", role="r", image_uri="x:latest", log_retention_days=14)
    )
    assert 'resource "aws_cloudwatch_log_group" "fn"' in out
    assert 'name              = "/aws/lambda/fn"' in out
    assert "retention_in_days = 14" in out
    # Function must depend on the managed group so Lambda doesn't race to create it.
    assert "depends_on" in out
    assert "aws_cloudwatch_log_group.fn," in out


# --- IAM: inline policies (R-013) ---


def test_iam_no_inline_policies_by_default():
    out = iam_generate(IAMRoleSpec(type="iam_role", name="my-role", service="lambda.amazonaws.com"))
    assert "aws_iam_role_policy" not in out  # also excludes the *_policy_attachment substring


def test_iam_inline_policy_emits_role_policy():
    spec = IAMRoleSpec(
        type="iam_role",
        name="my-role",
        service="lambda.amazonaws.com",
        inline_policies=[
            {
                "name": "artifacts",
                "actions": ["s3:GetObject", "s3:ListBucket"],
                "resources": ["arn:aws:s3:::b", "arn:aws:s3:::b/*"],
            }
        ],
    )
    out = iam_generate(spec)
    assert 'resource "aws_iam_role_policy" "my_role_artifacts"' in out
    assert 'name = "artifacts"' in out
    assert '"s3:GetObject", "s3:ListBucket"' in out
    assert '"arn:aws:s3:::b", "arn:aws:s3:::b/*"' in out


# --- RDS (R-015) ---


def test_rds_resource_block():
    out = rds_generate(RDSSpec(type="rds", name="mlflow-backend"))
    assert 'resource "aws_db_instance" "mlflow_backend"' in out
    assert 'identifier     = "mlflow-backend"' in out
    assert 'engine         = "postgres"' in out


def test_rds_defaults():
    out = rds_generate(RDSSpec(type="rds", name="db"))
    assert 'engine_version = "16"' in out
    assert 'instance_class = "db.t4g.micro"' in out
    assert "allocated_storage       = 20" in out
    assert "storage_encrypted       = true" in out
    assert "backup_retention_period = 7" in out
    assert "deletion_protection = true" in out
    assert "publicly_accessible = false" in out


def test_rds_db_name_and_username():
    out = rds_generate(RDSSpec(type="rds", name="db", db_name="tracking", username="admin"))
    assert 'db_name  = "tracking"' in out
    assert 'username = "admin"' in out


def test_rds_master_password_managed_by_secrets_manager():
    """No inline password argument — RDS manages it in Secrets Manager."""
    out = rds_generate(RDSSpec(type="rds", name="db"))
    assert "manage_master_user_password = true" in out
    # No literal `password = "..."` argument anywhere (the only "password" tokens are
    # `manage_master_user_password` and the explanatory comment).
    assert not any(ln.strip().startswith("password") and "=" in ln for ln in out.splitlines())


def test_rds_outputs_endpoint_and_secret_arn():
    out = rds_generate(RDSSpec(type="rds", name="mlflow-backend"))
    assert 'output "mlflow_backend_endpoint"' in out
    assert "aws_db_instance.mlflow_backend.endpoint" in out
    assert 'output "mlflow_backend_master_user_secret_arn"' in out
    assert "aws_db_instance.mlflow_backend.master_user_secret[0].secret_arn" in out


def test_rds_no_networking_by_default():
    out = rds_generate(RDSSpec(type="rds", name="db"))
    assert "db_subnet_group_name" not in out
    assert "vpc_security_group_ids" not in out


def test_rds_networking_rendered_when_set():
    out = rds_generate(
        RDSSpec(
            type="rds",
            name="db",
            db_subnet_group_name="mlflow-subnets",
            vpc_security_group_ids=["sg-0123", "sg-0456"],
        )
    )
    assert 'db_subnet_group_name   = "mlflow-subnets"' in out
    assert 'vpc_security_group_ids = ["sg-0123", "sg-0456"]' in out


def test_rds_settings_block_aligned_with_networking():
    """The settings `=` align (terraform fmt no-op) even when the wider subnet/sg keys render."""
    out = rds_generate(
        RDSSpec(
            type="rds",
            name="db",
            db_subnet_group_name="subnets",
            vpc_security_group_ids=["sg-1"],
        )
    )
    keys = ("multi_az", "publicly_accessible", "db_subnet_group_name", "vpc_security_group_ids")
    eq_cols = {ln.index("=") for ln in out.splitlines() if ln.strip().startswith(keys)}
    assert len(eq_cols) == 1, f"settings `=` not aligned: {eq_cols}"
