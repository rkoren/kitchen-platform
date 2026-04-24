"""Terraform generator for ECR repository resources."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from recipes.schema import ECRSpec

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    keep_trailing_newline=True,
)
_env.filters["tf_id"] = lambda s: s.replace("-", "_")


def generate(spec: ECRSpec) -> str:
    return _env.get_template("ecr.tf.j2").render(spec.model_dump())
