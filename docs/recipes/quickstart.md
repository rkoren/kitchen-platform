# Quickstart

## Installation

```bash
git clone https://github.com/rkoren/light-ml-platform.git
cd light-ml-platform/recipes
pip install -e .
```

## Your first spec

Create `infra.yaml`:

```yaml
name: my-api
region: us-east-1

resources:
  - type: s3
    name: my-api-artifacts
    versioning: true

  - type: iam_role
    name: my-api-exec
    service: lambda.amazonaws.com
    policies:
      - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

  - type: lambda
    name: my-api
    role: my-api-exec
    image_uri: "123456789.dkr.ecr.us-east-1.amazonaws.com/my-api:latest"
    memory_mb: 512
    timeout_s: 30
```

## Generate

```bash
recipes generate infra.yaml --out ./tf
```

Output:

```
  ✓ provider.tf
  ✓ my-api-artifacts.tf (s3)
  ✓ my-api-exec.tf (iam_role)
  ✓ my-api.tf (lambda)

Generated 4 file(s) → tf/
```

## Validate without generating

```bash
recipes validate infra.yaml
# ✓ spec is valid
```

## Apply with Terraform

```bash
cd tf
terraform init
terraform plan
terraform apply
```

!!! tip
    The generated configs are plain HCL — you can edit them after generation. `recipes` is a starting point, not a wrapper.
