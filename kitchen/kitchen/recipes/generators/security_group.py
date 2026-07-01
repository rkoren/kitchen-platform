"""Terraform generator for security group resources."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from kitchen.recipes.schema import SecurityGroupSpec

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    keep_trailing_newline=True,
)
_env.filters["tf_id"] = lambda s: s.replace("-", "_")


def generate(spec: SecurityGroupSpec) -> str:
    """Render security_group.tf.j2 for the given spec and return the Terraform HCL string."""
    return _env.get_template("security_group.tf.j2").render(spec.model_dump())
