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
| `s3` | S3 bucket with optional versioning |
| `iam_role` | IAM role with assume-role policy and managed policy attachments |
| `lambda` | Lambda function — image (ECR) or zip deployment |

## Commands

| Command | Description |
|---|---|
| `recipes generate SPEC` | Generate Terraform configs from a spec |
| `recipes validate SPEC` | Validate a spec without generating files |
