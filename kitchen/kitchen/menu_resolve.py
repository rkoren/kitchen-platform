"""Resolve a menu's ``{from_role}`` settings to concrete values (INT-003).

A menu declares infra-coupled MLflow settings by *role* — ``tracking_uri: {from_role:
mlflow-backend}``, ``artifact_bucket: {from_role: mlflow-artifacts}}`` — instead of
re-typing them (the drift fix, INT-001). This module turns those into real environment
values from the **deployed** infra:

* an ``rds`` role → ``MLFLOW_TRACKING_URI`` assembled from the recipe's Terraform outputs
  (reusing LML-015's ``kitchen.secrets.db_url_from_terraform``);
* an ``s3`` role → ``MLFLOW_ARTIFACT_BUCKET`` = the bucket name (the recipe's name).

Per the decision's resolution-timing lean, these are **materialized once** after
``provision`` (written to ``.env`` / ``$GITHUB_ENV``), not re-read every stage.

Note (simplification S-6): the recipes Terraform workspace is ``~/.recipes/tf/<project>``;
kitchen hard-codes that convention here. A merged package removes the assumption.
"""

from __future__ import annotations

from pathlib import Path

from kitchen.menu import Menu, RecipeEntry, RoleRef


def recipes_workspace(project: str) -> Path:
    """The recipes Terraform workspace for a project (the ``recipes apply`` default)."""
    return Path.home() / ".recipes" / "tf" / project


def _recipe_by_role(menu: Menu, role: str) -> tuple[str, RecipeEntry]:
    for name, entry in menu.recipes.items():
        if entry.role == role:
            return name, entry
    # Menu validation already guarantees from_role matches a recipe role; defensive only.
    raise ValueError(f"no recipe with role {role!r}")


def resolve_mlflow_env(menu: Menu, *, tf_dir: str | Path | None = None) -> dict[str, str]:
    """Resolve the menu's MLflow settings to a ``{ENV_VAR: value}`` mapping.

    ``{from_role}`` settings are resolved from the matching recipe + its deployed Terraform
    outputs; literal settings pass through. ``tf_dir`` defaults to the recipes workspace for
    the menu's ``project``. Raises ``ValueError`` if a role points at the wrong kind of recipe.
    """
    from kitchen import secrets  # lazy: pulls boto/terraform only when actually resolving

    workspace = str(tf_dir) if tf_dir is not None else str(recipes_workspace(menu.project))
    env: dict[str, str] = {}

    uri = menu.mlflow.tracking_uri
    if isinstance(uri, RoleRef):
        name, entry = _recipe_by_role(menu, uri.from_role)
        if entry.kind != "rds":
            raise ValueError(
                f"mlflow.tracking_uri from_role {uri.from_role!r} must point at an `rds` recipe "
                f"(role is on a {entry.kind!r} recipe)."
            )
        db = entry.fields.get("db_name", "mlflow")
        env["MLFLOW_TRACKING_URI"] = secrets.db_url_from_terraform(workspace, rds=name, db=db)
    elif uri:
        env["MLFLOW_TRACKING_URI"] = uri

    bucket = menu.mlflow.artifact_bucket
    if isinstance(bucket, RoleRef):
        name, entry = _recipe_by_role(menu, bucket.from_role)
        if entry.kind != "s3":
            raise ValueError(
                f"mlflow.artifact_bucket from_role {bucket.from_role!r} must point at an `s3` "
                f"recipe (role is on a {entry.kind!r} recipe)."
            )
        env["MLFLOW_ARTIFACT_BUCKET"] = name  # the s3 recipe's name is the bucket name
    elif bucket:
        env["MLFLOW_ARTIFACT_BUCKET"] = bucket

    return env


def materialize_mlflow_env(
    menu: Menu, target: str | Path, *, tf_dir: str | Path | None = None
) -> list[str]:
    """Resolve and append ``NAME=value`` entries to ``target`` (masked). Returns the names.

    The tracking URI embeds a password, so each value is registered with
    ``kitchen.secrets.mask`` (GitHub Actions log masking) and written only to the file —
    never returned or printed.
    """
    from kitchen import secrets

    env = resolve_mlflow_env(menu, tf_dir=tf_dir)
    with open(target, "a", encoding="utf-8") as f:
        for name, value in env.items():
            secrets.mask(value)
            f.write(f"{name}={value}\n")
    return list(env)
