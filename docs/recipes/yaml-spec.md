# YAML Spec Reference

Every spec file has a root object with metadata and a list of resources.

## Root fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Logical name for this spec |
| `region` | string | no | `us-east-1` | AWS region for the provider block |
| `resources` | list | no | `[]` | List of resource definitions |

## Resource types

### `s3`

Provisions an S3 bucket with optional versioning.

```yaml
- type: s3
  name: my-bucket
  versioning: true
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Bucket name |
| `versioning` | bool | no | `false` | Enable S3 versioning |

---

### `ecr`

Provisions an ECR repository for Docker images.

```yaml
- type: ecr
  name: my-project-serve
  scan_on_push: true
  lambda_access: true     # adds resource policy so Lambda can pull images
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Repository name |
| `scan_on_push` | bool | no | `true` | Enable image vulnerability scanning on push |
| `image_tag_mutability` | string | no | `"MUTABLE"` | `"MUTABLE"` or `"IMMUTABLE"` |
| `lambda_access` | bool | no | `false` | Attach resource policy allowing Lambda to pull images from this repo |

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
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Role name |
| `service` | string | yes | — | AWS service principal (e.g. `lambda.amazonaws.com`) |
| `policies` | list[string] | no | `[]` | ARNs of managed policies to attach |

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

!!! note "Package type"
    Set either image fields (`ecr_repo` or `image_uri`) **or** zip fields (`runtime` + `handler`), not both. `ecr_repo` and `image_uri` are mutually exclusive with `runtime`/`handler`.

!!! tip "Prefer `ecr_repo` over `image_uri`"
    When the ECR repository is defined in the same spec, use `ecr_repo` (the logical name) instead of `image_uri`. Terraform will generate a reference to the repo URL so the Lambda automatically tracks the repository — no hard-coded account IDs or URLs in your spec.

---

## Cross-resource validation

`recipes validate` enforces these rules at spec-parse time, before any Terraform is generated:

| Rule | Error |
|---|---|
| A `lambda.role` that is not an ARN must match an `iam_role.name` in the same spec | `Lambda 'x': role 'y' does not match any iam_role resource` |
| A `lambda.ecr_repo` must match an `ecr.name` in the same spec | `Lambda 'x': ecr_repo 'y' does not match any ecr resource` |

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
