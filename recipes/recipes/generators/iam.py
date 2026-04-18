"""Terraform generator for IAM role resources."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from recipes.schema import IAMRoleSpec

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    keep_trailing_newline=True,
)


def generate(spec: IAMRoleSpec) -> str:
    return _env.get_template("iam_role.tf.j2").render(spec.model_dump())
