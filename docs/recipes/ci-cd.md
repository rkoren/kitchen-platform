# CI/CD Integration

`recipes` integrates with GitHub Actions to validate specs on every pull request and provision resources on demand.

## Workflow overview

The recommended pattern keeps destructive operations out of automated runs:

| Trigger | Step | Command |
|---|---|---|
| Pull request | Validate the spec | `recipes validate infra.yaml` |
| Push to `main` | Generate Terraform configs | `recipes generate infra.yaml --out tf/` |
| Manual (`workflow_dispatch`) | Provision resources | `recipes apply infra.yaml --state-bucket $BUCKET --yes` |

A minimal workflow that covers validate-on-PR and generate-on-push:

```yaml
# .github/workflows/recipes-ci.yml
name: recipes

on:
  pull_request:
    paths:
      - "infra/**"
  push:
    branches: [main]
    paths:
      - "infra/**"
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install rkoren-recipes
      - name: Validate spec
        run: recipes validate infra/my-project.yaml

  generate:
    if: github.ref == 'refs/heads/main'
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install rkoren-recipes
      - name: Generate Terraform configs
        run: recipes generate infra/my-project.yaml --out tf/
      - uses: actions/upload-artifact@v4
        with:
          name: terraform-configs
          path: tf/
```

## Validate on pull request

```bash
recipes validate infra.yaml
```

Exits non-zero and prints a Pydantic validation error when the spec is malformed. Wire it up as a required status check to block merges on invalid specs.

## Generate on merge

After validation passes, generate the Terraform configs and upload them as a workflow artifact so they can be reviewed before `apply` runs:

```bash
recipes generate infra.yaml --out tf/
```

This writes `provider.tf` and one `.tf` file per resource into `tf/`. No AWS credentials are needed — generation is a local, read-only step.

To run `terraform plan` against the generated configs, add a Terraform setup step:

```yaml
- uses: hashicorp/setup-terraform@v3
- name: Terraform plan
  run: terraform -chdir=tf init -backend=false && terraform -chdir=tf plan -input=false
```

`-backend=false` skips S3 state initialisation so the plan step works without AWS credentials.

## Secret management

### AWS credentials

Use OIDC (recommended) to avoid long-lived secrets:

```yaml
permissions:
  id-token: write
  contents: read

steps:
  - uses: aws-actions/configure-aws-credentials@v4
    with:
      role-to-assume: arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}:role/recipes-ci
      aws-region: us-east-1
```

If OIDC is not yet configured, fall back to access key secrets:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |

### Terraform state bucket

Store the state bucket name as a repository variable (not a secret — it is not sensitive):

| Variable | Example value |
|---|---|
| `RECIPES_STATE_BUCKET` | `my-project-tf-state` |

Pass it to `recipes apply`:

```bash
recipes apply infra.yaml --state-bucket ${{ vars.RECIPES_STATE_BUCKET }} --yes
```

Or set the environment variable so `recipes` picks it up automatically:

```yaml
env:
  RECIPES_STATE_BUCKET: ${{ vars.RECIPES_STATE_BUCKET }}
```
