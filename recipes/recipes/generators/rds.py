"""Terraform generator for RDS (Postgres) database resources."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from recipes.schema import RDSSpec

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    keep_trailing_newline=True,
)
_env.filters["tf_id"] = lambda s: s.replace("-", "_")


def generate(spec: RDSSpec) -> str:
    """Render rds.tf.j2 for the given spec and return the Terraform HCL string."""
    ctx = spec.model_dump()
    # Security groups: logical names of in-spec security_group resources become TF
    # references; literal IDs are quoted. Both feed the one vpc_security_group_ids list.
    sg_entries = [f"aws_security_group.{name.replace('-', '_')}.id" for name in spec.security_groups]
    sg_entries += [f'"{sgid}"' for sgid in spec.vpc_security_group_ids]
    ctx["sg_entries"] = sg_entries
    # Align the `=` in the settings block the way `terraform fmt` would: the optional
    # subnet/security-group keys widen the block when present, so compute the width
    # over exactly the keys that will render (same approach as the lambda generator).
    keys = ["multi_az", "publicly_accessible", "deletion_protection", "skip_final_snapshot"]
    if spec.db_subnet_group_name:
        keys.append("db_subnet_group_name")
    if sg_entries:
        keys.append("vpc_security_group_ids")
    ctx["settings_width"] = max(len(k) for k in keys)
    return _env.get_template("rds.tf.j2").render(ctx)
