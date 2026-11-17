# recipes

YAML spec -> Terraform for a small set of AWS resources (S3, ECR, IAM role, Lambda, RDS,
security group), plus `plan` / `apply` / `destroy` against an S3 state backend.

Standalone and optional: it is not part of the kitchen 2.0 core and has no kitchen dependency.

```bash
uv tool install ./recipes                      # or: pip install ./recipes
recipes validate examples/s3-data-bucket.yaml
recipes generate examples/lambda-api.yaml --out ./tf [--check | --validate]
recipes doctor                                 # terraform + AWS credential pre-flight
recipes plan|apply|destroy SPEC --state-bucket <bucket>   # or set RECIPES_STATE_BUCKET
recipes schema > recipe.schema.json            # JSON Schema for editor validation
```

A spec is `name` + `region` + a `resources:` list (see `examples/`; field reference in
`docs/yaml-spec.md`; JSON Schema in `docs/recipe.schema.json`, regenerate with
`recipes schema -o docs/recipe.schema.json`). State lives at
`s3://<state-bucket>/<name>/terraform.tfstate`; the persistent workspace is
`~/.recipes/tf/<name>`. `apply` refuses a plan that would delete every resource in a state owned
by a different spec unless `--allow-destroy` is passed.

`stacks/kitchen-platform.yaml` is the stack the kitchen 1.x CI deployed (still live in AWS).

Tests: `uv run --directory recipes --extra dev python -m pytest -q`
