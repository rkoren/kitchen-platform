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

| Field | Type | Required | Default |
|---|---|---|---|
| `name` | string | yes | — |
| `versioning` | bool | no | `false` |

---

### `iam_role`

Provisions an IAM role with an assume-role policy and optional managed policy attachments.

```yaml
- type: iam_role
  name: my-exec-role
  service: lambda.amazonaws.com
  policies:
    - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

| Field | Type | Required | Default |
|---|---|---|---|
| `name` | string | yes | — |
| `service` | string | yes | — |
| `policies` | list[string] | no | `[]` |

---

### `lambda`

Provisions a Lambda function. Supports both image-based (ECR) and zip-based deployment.

```yaml
# Image-based (ECR)
- type: lambda
  name: my-function
  role: my-exec-role
  image_uri: "123456789.dkr.ecr.us-east-1.amazonaws.com/my-fn:latest"
  memory_mb: 512
  timeout_s: 30
  environment:
    TABLE_NAME: my-table

# Zip-based
- type: lambda
  name: my-function
  role: my-exec-role
  runtime: python3.11
  handler: src.main.handler
  memory_mb: 128
  timeout_s: 3
```

| Field | Type | Required | Default |
|---|---|---|---|
| `name` | string | yes | — |
| `role` | string | yes | — |
| `image_uri` | string | no | `null` |
| `runtime` | string | no | `null` |
| `handler` | string | no | `null` |
| `memory_mb` | int | no | `128` |
| `timeout_s` | int | no | `3` |
| `environment` | dict | no | `{}` |

!!! note
    Set either `image_uri` (for container image deployment) or `runtime` + `handler` (for zip deployment), not both.
