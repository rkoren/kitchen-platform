# 5-Minute Quickstart

Get both packages installed and working locally — no AWS account, no Kaggle credentials required.

## Install

```bash
pip install rkoren-recipes rkoren-kitchen
```

Or install from source during development:

```bash
git clone https://github.com/rkoren/kitchen-platform.git
pip install -e kitchen-platform/recipes -e kitchen-platform/kitchen
```

Verify:

```bash
recipes --help
kitchen --help
```

## Try recipes

Create a minimal spec file:

```yaml
# infra.yaml
name: my-project
region: us-east-1

resources:
  - type: s3
    name: my-project-data
    versioning: true
```

Validate and generate:

```bash
recipes validate infra.yaml
# ✓ spec is valid

recipes generate infra.yaml --out tf/
# ✓ provider.tf
# ✓ s3-my-project-data.tf (s3)
# Generated 2 file(s) → tf/
```

The files in `tf/` are plain HCL — inspect, edit, or hand off to `terraform apply`. No AWS credentials needed for this step.

## Try kitchen

Scaffold a competition project (no Kaggle account needed for scaffolding):

```bash
kitchen init my-project --template baseline-xgb
cd my-project
```

Check what was created:

```bash
ls
# CLAUDE.md  data/  experiments/  flows/  infra/  params.yaml  pyproject.toml  src/
```

Install the project alongside `kitchen`:

```bash
pip install -e .
```

Run the pre-flight check:

```bash
kitchen check
# The check will report any missing tools or credentials.
# Missing Kaggle credentials is expected here — that's fine for a local-only workflow.
```

## Where to go next

| Goal | Guide |
|---|---|
| Run a full Kaggle competition | [Kaggle Competition Quickstart](kitchen/kaggle-quickstart.md) |
| Understand the kitchen CLI | [kitchen Quickstart](kitchen/quickstart.md) |
| Provision AWS resources with recipes | [recipes Quickstart](recipes/quickstart.md) |
| Wire up GitHub Actions CI | [CI/CD Integration](kitchen/ci-cd.md) |
