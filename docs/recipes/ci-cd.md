# CI/CD Integration

`recipes` integrates with GitHub Actions to validate specs on every push and generate Terraform on merge to `main`.

## Workflow overview

<!-- TODO: fill in once the GitHub Actions workflow is finalized -->

!!! note "Coming soon"
    This section will be completed once the CI/CD workflow is finalized.
    See `.github/workflows/recipes-ci.yml` for the current state.

## Validate on pull request

```yaml
# .github/workflows/recipes-ci.yml
- name: Validate spec
  run: recipes validate infra.yaml
```

## Generate on merge

<!-- TODO: document the generate + terraform plan step -->

## Secret management

Terraform state and AWS credentials should be stored as GitHub Actions secrets.

<!-- TODO: document required secrets and how to set them -->
