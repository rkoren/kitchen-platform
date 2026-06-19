# YAML Spec Reference

Every spec file has a root object with metadata and a list of resources.

## Machine-readable schema

This reference is generated from the same Pydantic models the CLI validates against.
A JSON Schema (draft 2020-12) is exported by the CLI and checked in at
[`recipe.schema.json`](recipe.schema.json):

```bash
recipes schema                       # print the schema to stdout
recipes schema --out recipe.schema.json
```

Point your editor's YAML tooling at it for inline validation and autocompletion, e.g.
with the VS Code YAML extension:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/rkoren/kitchen-platform/main/docs/recipes/recipe.schema.json
name: my-project
region: us-east-1
resources: []
```

## Root fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Logical name for this spec |
| `region` | string | no | `us-east-1` | AWS region for the provider block |
| `resources` | list | no | `[]` | List of resource definitions |

## Resource types

### `s3`

Provisions an S3 bucket. Encryption and public-access-block are on by default
(secure-by-default); set them to `false` to omit those resources.

```yaml
- type: s3
  name: my-bucket
  versioning: true
  lifecycle_expiration_days: 90   # expire objects after 90 days
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Bucket name |
| `versioning` | bool | no | `false` | Enable S3 versioning |
| `encryption` | bool | no | `true` | Default SSE-S3 (AES256) encryption at rest |
| `public_access_block` | bool | no | `true` | Block all public access (ACLs + policies) |
| `lifecycle_expiration_days` | int | no | — | Expire objects after N days (omit for no lifecycle rule) |

---

### `ecr`

Provisions an ECR repository for Docker images.

```yaml
- type: ecr
  name: my-project-serve
  scan_on_push: true
  lambda_access: true       # adds resource policy so Lambda can pull images
  lifecycle_keep_last: 10   # expire all but the 10 most recent images
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Repository name |
| `scan_on_push` | bool | no | `true` | Enable image vulnerability scanning on push |
| `image_tag_mutability` | string | no | `"MUTABLE"` | `"MUTABLE"` or `"IMMUTABLE"` |
| `lambda_access` | bool | no | `false` | Attach resource policy allowing Lambda to pull images from this repo |
| `lifecycle_keep_last` | int | no | — | Keep only the N most recent images; expire older (omit for no lifecycle policy) |

!!! tip
    Set `lambda_access: true` on any ECR repo that a Lambda function will pull from. This adds the `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, and `ecr:BatchCheckLayerAvailability` resource policies automatically.

---

### `iam_role`

Provisions an IAM role with an assume-role policy and optional managed policy attachments.

```yaml
- type: iam_role
  name: my-exec-role
  service: lambda.amazonaws.com
  policies:
    - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
  inline_policies:
    - name: artifacts-access      # scoped access to a specific bucket
      actions: [s3:GetObject, s3:ListBucket]
      resources:
        - arn:aws:s3:::my-project-artifacts
        - arn:aws:s3:::my-project-artifacts/*
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Role name |
| `service` | string | yes | — | AWS service principal (e.g. `lambda.amazonaws.com`) |
| `policies` | list[string] | no | `[]` | ARNs of managed policies to attach |
| `inline_policies` | list[object] | no | `[]` | Scoped allow policies — each has `name`, `actions` (list), and `resources` (list of ARNs); use for project-specific S3/model access |

---

### `lambda`

Provisions a Lambda function. Supports both image-based (ECR) and zip-based deployment.

```yaml
# Image-based — reference an ECR repo by logical name
- type: lambda
  name: my-project-serve
  role: my-exec-role
  ecr_repo: my-project-serve   # resolves to the ECR URL at apply time
  memory: 1024
  timeout: 30
  environment:
    KITCHEN_MODEL_NAME: my-project
    KITCHEN_MODEL_VERSION: "1"

# Image-based — provide a literal image URI
- type: lambda
  name: my-function
  role: my-exec-role
  image_uri: "123456789.dkr.ecr.us-east-1.amazonaws.com/my-fn:latest"
  memory: 512
  timeout: 30

# Zip-based
- type: lambda
  name: my-function
  role: my-exec-role
  runtime: python3.11
  handler: src.main.handler
  memory: 128
  timeout: 3
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Function name |
| `role` | string | yes | — | Name of an `iam_role` resource in this spec, or a literal ARN |
| `ecr_repo` | string | no | `null` | Logical name of an `ecr` resource in this spec — generates a Terraform reference to its URL |
| `image_uri` | string | no | `null` | Literal ECR image URI (use `ecr_repo` instead when the repo is in the same spec) |
| `runtime` | string | no | `null` | Lambda runtime identifier (zip package only, e.g. `python3.11`) |
| `handler` | string | no | `null` | Handler path (zip package only, e.g. `src.main.handler`) |
| `memory` | int | no | `128` | Memory in MB |
| `timeout` | int | no | `3` | Timeout in seconds |
| `environment` | dict | no | `{}` | Environment variables injected at function invocation |
| `function_url` | bool | no | `false` | Expose the function over HTTPS via a Lambda function URL |
| `function_url_auth` | string | no | `"AWS_IAM"` | `"AWS_IAM"` (SigV4-signed) or `"NONE"` (public) — only used when `function_url: true` |
| `log_retention_days` | int | no | — | Retention for the function's CloudWatch log group (omit to never expire) |

!!! tip "Serving over HTTP"
    Set `function_url: true` to get a direct HTTPS endpoint for the function — the simplest way to serve an inference Lambda. The URL is exposed as a Terraform output (`<name>_url`). Auth defaults to `AWS_IAM` (callers sign requests with SigV4); set `function_url_auth: NONE` for a public endpoint, which also adds the `lambda:InvokeFunctionUrl` permission for public access.

!!! note "Package type"
    Set either image fields (`ecr_repo` or `image_uri`) **or** zip fields (`runtime` + `handler`), not both. `ecr_repo` and `image_uri` are mutually exclusive with `runtime`/`handler`.

!!! tip "Prefer `ecr_repo` over `image_uri`"
    When the ECR repository is defined in the same spec, use `ecr_repo` (the logical name) instead of `image_uri`. Terraform will generate a reference to the repo URL so the Lambda automatically tracks the repository — no hard-coded account IDs or URLs in your spec.

---

### `rds`

Provisions a managed Postgres instance — typically the **MLflow backend store** that makes registry champions persist across runs (see [`mlflow-tracking-backend`](../decisions/mlflow-tracking-backend.md)).

```yaml
- type: rds
  name: mlflow-backend
  engine_version: "16"
  instance_class: db.t4g.micro
  allocated_storage: 20
  db_name: mlflow
  username: mlflow
  deletion_protection: true
  db_subnet_group_name: mlflow-subnets       # existing subnet group (recipes does not make a VPC)
  vpc_security_group_ids: [sg-0123, sg-0456]  # existing security groups
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | DB identifier (also the Terraform resource label) |
| `engine_version` | string | no | `"16"` | Postgres major version |
| `instance_class` | string | no | `"db.t4g.micro"` | RDS instance class |
| `allocated_storage` | int | no | `20` | Storage in GB |
| `backup_retention_days` | int | no | `7` | Automated-backup / point-in-time-recovery window (days) |
| `db_name` | string | no | `"mlflow"` | Database created on the instance |
| `username` | string | no | `"mlflow"` | Master username |
| `storage_encrypted` | bool | no | `true` | Encrypt storage at rest |
| `multi_az` | bool | no | `false` | Deploy a standby in a second AZ |
| `publicly_accessible` | bool | no | `false` | Assign a public endpoint (otherwise reach via VPC/SG) |
| `deletion_protection` | bool | no | `true` | Block accidental `terraform destroy` of the champion store |
| `skip_final_snapshot` | bool | no | `true` | Skip the final snapshot on destroy (set `false` for production retention) |
| `subnet_ids` | list[string] | no | `[]` | Subnets (≥2, in different AZs) → recipes **generates** an `aws_db_subnet_group` and points the instance at it. Use for a custom VPC. Mutually exclusive with `db_subnet_group_name` |
| `db_subnet_group_name` | string | no | — | Use an **existing** DB subnet group by name (recipes does not create it). Mutually exclusive with `subnet_ids` |
| `security_groups` | list[string] | no | `[]` | Logical names of `security_group` resources in this spec; rendered as `aws_security_group.<name>.id` references |
| `vpc_security_group_ids` | list[string] | no | `[]` | Existing (literal) security group IDs; combined with `security_groups` |

!!! note "Master password"
    There is **no password field**. RDS creates and rotates the master password in AWS Secrets Manager (`manage_master_user_password`), so it never appears in your spec, the generated Terraform, or state. The secret's ARN is exposed as the `<name>_master_user_secret_arn` output — wire it into a project's `secrets:` manifest. The connection host is the `<name>_endpoint` output.

!!! tip "VPC / networking"
    By default the RDS and its security group land in the account's **default VPC** (zero config). For an account without a default VPC — or to use a specific one — set `vpc_id` on the `security_group` and either `subnet_ids` (recipes makes the DB subnet group) or `db_subnet_group_name` (an existing one) on the `rds`; the subnets must be in the same VPC as the security group. No default VPC and want the quick path? `aws ec2 create-default-vpc` restores one.

---

### `security_group`

Provisions a security group — typically the inbound rule that makes an [`rds`](#rds) instance reachable (default: Postgres `5432`). Attach it from an `rds` via `security_groups: [<this name>]`.

```yaml
- type: security_group
  name: mlflow-backend-sg
  description: Postgres access to the MLflow backend store
  vpc_id: vpc-0abc123          # omit to use the account's default VPC
  ingress:
    - port: 5432
      protocol: tcp
      cidr_blocks: [0.0.0.0/0]  # default; narrow to known ranges where possible
      description: PostgreSQL
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Security group name |
| `description` | string | no | `"Managed by recipes"` | Group description |
| `vpc_id` | string | no | — | VPC to create the group in (omit for the default VPC) |
| `ingress` | list[object] | no | one Postgres rule | Inbound rules — each has `port`, `protocol` (`"tcp"`), `cidr_blocks` (`["0.0.0.0/0"]`), optional `description` |
| `egress_all` | bool | no | `true` | Emit an allow-all outbound rule |

The group's ID is exposed as the `<name>_id` output.

!!! warning "Default ingress is open to the internet"
    The default `cidr_blocks` is `0.0.0.0/0` so GitHub-hosted CI runners (dynamic IPs) and your local machine can reach an `rds` backend. Pair it with TLS and the RDS-managed password, or narrow `cidr_blocks` to known ranges. For a private backend, drop `publicly_accessible` and put a tracking server in front (see [`mlflow-tracking-backend`](../decisions/mlflow-tracking-backend.md)).

---

## Cross-resource validation

`recipes validate` enforces these rules at spec-parse time, before any Terraform is generated:

| Rule | Error |
|---|---|
| A `lambda.role` that is not an ARN must match an `iam_role.name` in the same spec | `Lambda 'x': role 'y' does not match any iam_role resource` |
| A `lambda.ecr_repo` must match an `ecr.name` in the same spec | `Lambda 'x': ecr_repo 'y' does not match any ecr resource` |
| An `rds.security_groups` entry must match a `security_group.name` in the same spec | `RDS 'x': security_group 'y' does not match any security_group resource` |

---

## Complete example — ML serving stack

```yaml
name: my-project
region: us-east-1

resources:
  - type: s3
    name: my-project-data
    versioning: true

  - type: ecr
    name: my-project-serve
    scan_on_push: true
    lambda_access: true

  - type: iam_role
    name: my-project-exec
    service: lambda.amazonaws.com
    policies:
      - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

  - type: lambda
    name: my-project-serve
    role: my-project-exec
    ecr_repo: my-project-serve
    memory: 1024
    timeout: 30
    environment:
      KITCHEN_MODEL_NAME: my-project
      KITCHEN_PREDICTOR_DIR: /var/task/src/serve
```
