# recipes

`recipes` is a lightweight CLI that converts a simple YAML spec into Terraform configurations for AWS resources.

## Why recipes?

Writing Terraform for common AWS resources — Lambda functions, S3 buckets, IAM roles — involves a lot of repetitive boilerplate. `recipes` lets you declare *what* you want in a concise YAML spec and generates the correct, opinionated HCL for you.

```bash
recipes generate infra.yaml --out ./tf
```

## Design

- **Input:** a single YAML file describing your resources
- **Output:** one `.tf` file per resource + a `provider.tf`
- **Extensible:** adding a new resource type means adding a Pydantic model, a Jinja2 template, and a generator — nothing else changes

## Supported resources

| Type | Description |
|---|---|
| `s3` | S3 bucket — versioning, default encryption, public-access-block, lifecycle expiration |
| `ecr` | ECR repository — scan-on-push, tag mutability, lifecycle policy, Lambda pull access |
| `iam_role` | IAM role with assume-role policy and managed policy attachments |
| `lambda` | Lambda function — image (ECR) or zip deployment, optional HTTPS function URL |

## Commands

| Command | Description |
|---|---|
| `recipes generate SPEC [--check] [--validate]` | Generate Terraform configs (optionally `terraform fmt`/`validate` the output) |
| `recipes validate SPEC` | Validate a spec without generating files |
| `recipes schema [--out PATH]` | Export the recipe YAML JSON Schema (draft 2020-12) |
| `recipes doctor [--state-bucket B]` | Pre-flight checks: Terraform, AWS credentials, state-bucket access |
| `recipes plan SPEC --state-bucket B` | Preview changes (`terraform plan`) without applying |
| `recipes apply SPEC --state-bucket B` | Provision resources (`terraform apply`) |
| `recipes destroy SPEC --state-bucket B` | Tear down all resources in the spec |

## Examples

See [`recipes/examples/`](https://github.com/rkoren/kitchen-platform/tree/main/recipes/examples) — including `ecr-lambda-inference-api.yaml`, a complete containerised model-inference API (ECR repo + Lambda behind an HTTPS function URL + an artifacts bucket).
